from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

from policy_assistant.answering.model import CitationRef, ModelReply, ReplySegment
from policy_assistant.answering.prompts import error_text
from policy_assistant.api.app import create_app
from policy_assistant.auth.users import create_user
from policy_assistant.config import Settings
from policy_assistant.db.connection import connect
from policy_assistant.ingest.manifest import Manifest
from policy_assistant.ingest.pipeline import ingest
from policy_assistant.web.i18n import STRINGS
from tests.fakes import HashingEmbedder, OverlapReranker, ScriptedAnswerModel

pytestmark = pytest.mark.integration

PASSWORD = "correct horse battery staple"
BASE = "https://testserver"
ROOT = Path(__file__).resolve().parents[2]
FONT = ROOT / "src/policy_assistant/web/static/fonts/NotoNastaliqUrdu-wght.ttf"
FONT_LICENSE = FONT.with_name("OFL.txt")


def _reply(
    text: str = "Employees get 20 working days.",
    citation: bool = True,
    citation_text: str = "20 working days",
) -> ModelReply:
    citations = (CitationRef(0, 0, 1, citation_text),) if citation else ()
    return ModelReply(
        segments=(ReplySegment(text, citations),),
        stop_reason="end_turn",
        stop_details=None,
        model_served="fake",
        request_id="req_fake",
        usage={"input_tokens": 1, "output_tokens": 1},
        unsupported_citation_types=(),
    )


def _ingest_sample(
    manifest: Manifest,
    root: Path,
    settings: Settings,
    counter: Callable[[str], int],
    embedder: HashingEmbedder,
) -> None:
    ingest(
        manifest,
        repo_root=root,
        only_ids=None,
        force=False,
        prune=False,
        count_tokens=counter,
        embedder=embedder,
        settings=settings,
    )


@pytest.fixture
def web_client(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    monkeypatch.setattr(
        "policy_assistant.auth.passwords.HASHER",
        PasswordHasher(time_cost=1, memory_cost=1024, parallelism=1),
    )
    from policy_assistant.auth import passwords

    passwords._dummy_hash.cache_clear()
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    with connect(db_dsn) as conn:
        create_user(conn, "alice", "Alice", PASSWORD, "user", ["all_staff"])
        create_user(conn, "fatima", "Fatima", PASSWORD, "user", ["all_staff", "finance"])
        create_user(conn, "rehan", "Rehan", PASSWORD, "reviewer", ["all_staff"])
        create_user(conn, "farah", "Farah", PASSWORD, "reviewer", ["all_staff", "finance"])
        create_user(conn, "ayesha", "Ayesha", PASSWORD, "admin", [])
    app = create_app(
        settings_for,
        embedder=fake_embedder,
        reranker=fake_reranker,
        answer_model=ScriptedAnswerModel(_reply()),
    )
    with TestClient(app, base_url=BASE) as client:
        yield client


def _login(client: TestClient, username: str, password: str = PASSWORD) -> str:
    response = client.post(
        "/login",
        data={"username": username, "password": password},
        headers={"Origin": BASE},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    me = client.get("/api/me")
    assert me.status_code == 200
    return str(me.json()["csrf_token"])


def _web_ask(client: TestClient, csrf: str, question: str) -> Any:
    return client.post(
        "/ask",
        data={"question": question, "csrf_token": csrf},
        headers={"Origin": BASE},
        follow_redirects=False,
    )


def _audit_id_from_location(location: str) -> str:
    return location.split("#a-", 1)[1]


def _latest_audit_id(dsn: str, actor: str) -> str:
    with connect(dsn) as conn:
        row = conn.execute(
            "SELECT id FROM answer_audit WHERE actor = %s ORDER BY created_at DESC LIMIT 1",
            (actor,),
        ).fetchone()
    assert row is not None
    return str(row[0])


def test_login_page_and_form_guards(web_client: TestClient) -> None:
    page = web_client.get("/login")
    assert page.status_code == 200
    assert 'name="username"' in page.text
    assert 'name="password"' in page.text
    success = web_client.post(
        "/login",
        data={"username": "alice", "password": PASSWORD},
        headers={"Origin": BASE},
        follow_redirects=False,
    )
    assert success.status_code == 303
    assert success.headers["location"] == "/"
    assert "HttpOnly" in success.headers["set-cookie"]
    assert "Secure" in success.headers["set-cookie"]
    assert "SameSite=strict" in success.headers["set-cookie"]
    web_client.cookies.clear()
    wrong = web_client.post(
        "/login",
        data={"username": "alice", "password": "wrong password!"},
        headers={"Origin": BASE},
    )
    assert wrong.status_code == 401
    assert STRINGS["en"]["login_error"] in wrong.text
    foreign = web_client.post(
        "/login",
        data={"username": "alice", "password": PASSWORD},
        headers={"Origin": "https://evil.example"},
    )
    assert foreign.status_code == 403
    missing_origin = web_client.post("/login", data={"username": "alice", "password": PASSWORD})
    assert missing_origin.status_code == 403
    web_client.cookies.clear()
    assert web_client.get("/", follow_redirects=False).status_code == 303
    assert web_client.get("/", follow_redirects=False).headers["location"] == "/login"


def test_ask_render_and_urdu_block(web_client: TestClient) -> None:
    csrf = _login(web_client, "alice")
    question = "How many days of annual leave do employees get?"
    response = _web_ask(web_client, csrf, question)
    assert response.status_code == 303
    audit_id = _audit_id_from_location(response.headers["location"])
    home = web_client.get("/")
    assert home.status_code == 200
    assert f'id="a-{audit_id}"' in home.text
    assert "Employees get 20 working days." in home.text
    assert f'<sup class="cite"><a href="#src-{audit_id}-1">1</a></sup>' in home.text
    assert f'id="src-{audit_id}-1"' in home.text

    missing_csrf = web_client.post("/ask", data={"question": question}, headers={"Origin": BASE})
    assert missing_csrf.status_code == 403
    foreign = web_client.post(
        "/ask",
        data={"question": question, "csrf_token": csrf},
        headers={"Origin": "https://evil.example"},
    )
    assert foreign.status_code == 403

    urdu = "ملازمین کو کتنے دن کی سالانہ چھٹی ملتی ہے؟"
    urdu_response = _web_ask(web_client, csrf, urdu)
    assert urdu_response.status_code == 303
    urdu_home = web_client.get("/")
    assert 'lang="ur" dir="rtl"' in urdu_home.text
    assert urdu in urdu_home.text

    web_client.app.state.answer_model = ScriptedAnswerModel(_reply("NO_ANSWER", False))
    abstained_response = _web_ask(web_client, csrf, question)
    assert abstained_response.status_code == 303
    abstained_id = _audit_id_from_location(abstained_response.headers["location"])
    abstained_home = web_client.get("/")
    assert "find an answer to this in the policies available to you." in abstained_home.text
    assert f'id="src-{abstained_id}-' not in abstained_home.text

    web_client.app.state.settings = web_client.app.state.settings.model_copy(
        update={"ask_queue_timeout_seconds": 0.05}
    )
    slots = web_client.app.state.ask_slots
    acquired = 0
    try:
        for _ in range(web_client.app.state.settings.max_concurrent_asks):
            assert slots.acquire(blocking=False)
            acquired += 1
        busy = _web_ask(web_client, csrf, question)
        assert busy.status_code == 503
        assert STRINGS["en"]["busy"] in busy.text
        slots.release()
        acquired -= 1
    finally:
        for _ in range(acquired):
            slots.release()

    class FailingModel:
        def generate(
            self, system: str, documents: list[dict[str, Any]], question: str
        ) -> ModelReply:
            raise TypeError("boom")

    web_client.app.state.answer_model = FailingModel()
    failed = _web_ask(web_client, csrf, question)
    assert failed.status_code == 500
    assert error_text("en") in failed.text


def test_xss_is_escaped(web_client: TestClient, db_dsn: str) -> None:
    csrf = _login(web_client, "alice")
    web_client.app.state.answer_model = ScriptedAnswerModel(_reply("<img src=x onerror=alert(1)>"))
    response = _web_ask(web_client, csrf, "<script>alert(1)</script>")
    assert response.status_code == 303
    audit_id = _audit_id_from_location(response.headers["location"])
    home = web_client.get("/").text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in home
    assert "&lt;img src=x onerror=alert(1)&gt;" in home
    assert "<script>alert(1)</script>" not in home
    assert "<img src=x onerror=alert(1)>" not in home

    web_client.cookies.clear()
    _login(web_client, "rehan")
    review = web_client.get(f"/review/{audit_id}")
    assert review.status_code == 200
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in review.text
    assert "&lt;img src=x onerror=alert(1)&gt;" in review.text
    assert "<script>alert(1)</script>" not in review.text
    assert "<img src=x onerror=alert(1)>" not in review.text


def test_review_pages_and_scores(web_client: TestClient, db_dsn: str) -> None:
    finance_csrf = _login(web_client, "fatima")
    finance = _web_ask(
        web_client,
        finance_csrf,
        "What is the Finance Director's approval limit?",
    )
    assert finance.status_code == 303
    audit_id = _audit_id_from_location(finance.headers["location"])

    web_client.cookies.clear()
    _login(web_client, "alice")
    forbidden = web_client.get("/review")
    assert forbidden.status_code == 403
    assert STRINGS["en"]["forbidden"] in forbidden.text

    web_client.cookies.clear()
    rehan_csrf = _login(web_client, "rehan")
    assert web_client.get(f"/review/{audit_id}").status_code == 404
    assert web_client.get("/review").status_code == 200
    hidden_post = web_client.post(
        f"/review/{audit_id}",
        data={"verdict": "correct", "citation_ok": "true", "csrf_token": rehan_csrf},
        headers={"Origin": BASE},
        follow_redirects=False,
    )
    assert hidden_post.status_code == 404

    web_client.cookies.clear()
    farah_csrf = _login(web_client, "farah")
    item = web_client.get(f"/review/{audit_id}")
    assert item.status_code == 200
    assert "<mark>" in item.text
    assert "bm25_score" not in item.text
    assert "dense_distance" not in item.text
    assert "rrf_score" not in item.text
    assert "Rerank score" in item.text
    invalid = web_client.post(
        f"/review/{audit_id}",
        data={"verdict": "bogus", "citation_ok": "true", "csrf_token": farah_csrf},
        headers={"Origin": BASE},
        follow_redirects=False,
    )
    assert invalid.status_code == 422
    saved = web_client.post(
        f"/review/{audit_id}",
        data={
            "verdict": "correct",
            "citation_ok": "true",
            "notes": "checked",
            "csrf_token": farah_csrf,
        },
        headers={"Origin": BASE},
        follow_redirects=False,
    )
    assert saved.status_code == 303
    reviewed = web_client.get(f"/review/{audit_id}")
    assert "farah" in reviewed.text
    assert STRINGS["en"]["verdict_correct"] in reviewed.text

    web_client.cookies.clear()
    _login(web_client, "alice")
    csrf = web_client.get("/api/me").json()["csrf_token"]

    class FailingModel:
        def generate(
            self, system: str, documents: list[dict[str, Any]], question: str
        ) -> ModelReply:
            raise TypeError("boom")

    web_client.app.state.answer_model = FailingModel()
    failed = _web_ask(web_client, csrf, "annual leave")
    assert failed.status_code == 500
    error_id = _latest_audit_id(db_dsn, "user:alice")
    web_client.cookies.clear()
    _login(web_client, "farah")
    error_item = web_client.get(f"/review/{error_id}")
    assert error_item.status_code == 200
    assert STRINGS["en"]["review_no_form"] in error_item.text
    assert 'name="verdict"' not in error_item.text


def test_set_language_and_ui_direction(web_client: TestClient) -> None:
    bad = web_client.get(
        "/set-language", params={"lang": "ur", "next": "//evil.example"}, follow_redirects=False
    )
    assert bad.status_code == 303
    assert bad.headers["location"] == "/"
    absolute = web_client.get(
        "/set-language",
        params={"lang": "ur", "next": "https://evil.example"},
        follow_redirects=False,
    )
    assert absolute.status_code == 303
    assert absolute.headers["location"] == "/"
    good = web_client.get(
        "/set-language", params={"lang": "ur", "next": "/"}, follow_redirects=False
    )
    assert good.status_code == 303
    assert "upa_lang=ur" in good.headers["set-cookie"]
    _login(web_client, "alice")
    html = web_client.get("/")
    assert '<html lang="ur" dir="rtl">' in html.text


def test_csp_hygiene_and_assets(web_client: TestClient, db_dsn: str) -> None:
    login_page = web_client.get("/login").text
    csrf = _login(web_client, "alice")
    ask = _web_ask(web_client, csrf, "How many days of annual leave do employees get?")
    audit_id = _audit_id_from_location(ask.headers["location"])
    chat_page = web_client.get("/").text
    web_client.cookies.clear()
    _login(web_client, "rehan")
    review_queue_page = web_client.get("/review").text
    review_item_page = web_client.get(f"/review/{audit_id}").text
    pages = [login_page, chat_page, review_queue_page, review_item_page]
    for html in pages:
        for script in re.findall(r"<script[^>]*>", html):
            assert "src=" in script
        assert "<style" not in html
        assert not re.search(r"\sstyle\s*=", html)
        assert not re.search(r"\son[a-z]+\s*=", html)

    js = (ROOT / "src/policy_assistant/web/static/app.js").read_text(encoding="utf-8")
    for banned in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
    ):
        assert banned not in js

    assert hashlib.sha256(FONT.read_bytes()).hexdigest() == (
        "98a4787f34eb6fde57fb9a1121a8f216301196ab0da98ea110cad581be8abbcd"
    )
    assert hashlib.sha256(FONT_LICENSE.read_bytes()).hexdigest() == (
        "e90279e3fb2003c4d54f8685e25274bd7838b1b21c891520d71dbdb1d4a98907"
    )
    assert FONT_LICENSE.exists()


def test_i18n_keys_match_and_are_nonempty() -> None:
    assert STRINGS["en"].keys() == STRINGS["ur"].keys()
    assert all(value for value in STRINGS["en"].values())
    assert all(value for value in STRINGS["ur"].values())
