from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

from policy_assistant.answering.model import CitationRef, ModelReply, ReplySegment
from policy_assistant.answering.prompts import abstention_text, error_text
from policy_assistant.api.app import create_app
from policy_assistant.auth.users import create_user, set_user_active, set_user_groups
from policy_assistant.config import Settings
from policy_assistant.db.connection import connect
from policy_assistant.ingest.manifest import Manifest
from policy_assistant.ingest.pipeline import ingest
from tests.fakes import (
    HashingEmbedder,
    OverlapReranker,
    ScriptedAnswerModel,
)

pytestmark = pytest.mark.integration

PASSWORD = "correct horse battery staple"


def _ingest_sample(
    manifest: Manifest,
    root: Path,
    settings: Settings,
    counter: Callable[[str], int],
    embedder: HashingEmbedder,
    *,
    prune: bool = False,
) -> None:
    ingest(
        manifest,
        repo_root=root,
        only_ids=None,
        force=False,
        prune=prune,
        count_tokens=counter,
        embedder=embedder,
        settings=settings,
    )


def _reply(text: str = "Answer.", citation: bool = True) -> ModelReply:
    citations = (CitationRef(0, 0, 1, "cited"),) if citation else ()
    return ModelReply(
        segments=(ReplySegment(text, citations),),
        stop_reason="end_turn",
        stop_details=None,
        model_served="fake",
        request_id="req_fake",
        usage={"input_tokens": 1, "output_tokens": 1},
        unsupported_citation_types=(),
    )


@pytest.fixture
def api_client(
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
    model = ScriptedAnswerModel(_reply())
    app = create_app(
        settings_for,
        embedder=fake_embedder,
        reranker=fake_reranker,
        answer_model=model,
    )
    with TestClient(app, base_url="https://testserver") as client:
        yield client


def _login(client: TestClient, username: str, password: str = PASSWORD) -> str:
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return str(response.json()["csrf_token"])


def _headers(csrf: str) -> dict[str, str]:
    return {"X-CSRF-Token": csrf}


def _ask(client: TestClient, csrf: str, question: str) -> Any:
    return client.post("/api/ask", json={"question": question}, headers=_headers(csrf))


def _audit_ids(client: TestClient, username: str) -> list[str]:
    return [
        str(row[0])
        for row in client.app.state.pool.connection()
        .__enter__()
        .execute(
            "SELECT id FROM answer_audit WHERE actor = %s ORDER BY created_at",
            (f"user:{username}",),
        )
        .fetchall()
    ]


def test_login_cookie_and_distinct_sessions(api_client: TestClient) -> None:
    first = api_client.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
    assert first.status_code == 200
    cookie = first.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=strict" in cookie
    assert "Path=/" in cookie
    assert "Max-Age=28800" in cookie
    first_token = api_client.cookies.get("upa_session")
    second = api_client.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
    assert second.status_code == 200
    assert api_client.cookies.get("upa_session") != first_token


def test_login_failures_are_identical(api_client: TestClient, db_dsn: str) -> None:
    wrong = api_client.post(
        "/api/auth/login", json={"username": "alice", "password": "wrong password!"}
    )
    unknown = api_client.post("/api/auth/login", json={"username": "nobody", "password": PASSWORD})
    with connect(db_dsn) as conn:
        set_user_active(conn, "alice", False)
    disabled = api_client.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
    assert wrong.status_code == unknown.status_code == disabled.status_code == 401
    assert wrong.content == unknown.content == disabled.content


def test_login_uses_default_hasher(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("policy_assistant.auth.passwords.HASHER", PasswordHasher())
    response = api_client.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
    assert response.status_code == 200


def test_session_expiry_logout_and_deactivation(api_client: TestClient, db_dsn: str) -> None:
    csrf = _login(api_client, "alice")
    token = api_client.cookies.get("upa_session")
    assert token is not None
    with connect(db_dsn) as conn:
        conn.execute(
            """
            UPDATE sessions SET expires_at = now() - interval '1 second'
            WHERE user_id = (SELECT id FROM users WHERE username = 'alice')
            """
        )
    assert api_client.get("/api/me").status_code == 401

    csrf = _login(api_client, "alice")
    token = api_client.cookies.get("upa_session")
    logout = api_client.post("/api/auth/logout", headers=_headers(csrf))
    assert logout.status_code == 204
    old = api_client.get("/api/me", headers={"cookie": f"upa_session={token}"})
    assert old.status_code == 401

    csrf = _login(api_client, "alice")
    with connect(db_dsn) as conn:
        set_user_active(conn, "alice", False)
    assert api_client.get("/api/me").status_code == 401


def test_security_headers_and_docs(api_client: TestClient) -> None:
    _login(api_client, "alice")
    ok = api_client.get("/api/me")
    assert ok.status_code == 200
    expected_csp = "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    for response in (ok,):
        assert response.headers["content-security-policy"] == expected_csp
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["cache-control"] == "no-store"

    api_client.cookies.clear()
    unauthorized = api_client.get("/api/me")
    assert unauthorized.status_code == 401
    assert unauthorized.headers["content-security-policy"] == expected_csp
    assert unauthorized.headers["cache-control"] == "no-store"
    missing = api_client.get("/docs")
    assert missing.status_code == 404
    assert missing.headers["content-security-policy"] == expected_csp


def test_csrf_and_question_validation(api_client: TestClient) -> None:
    csrf = _login(api_client, "alice")
    without = api_client.post("/api/ask", json={"question": "annual leave"})
    assert without.status_code == 403
    assert without.json() == {"detail": "csrf token missing or invalid"}
    wrong = api_client.post(
        "/api/ask",
        json={"question": "annual leave"},
        headers={"X-CSRF-Token": "wrong"},
    )
    assert wrong.status_code == 403
    correct = api_client.post(
        "/api/ask",
        json={"question": "How many days of annual leave do employees get?"},
        headers=_headers(csrf),
    )
    assert correct.status_code == 200
    assert (
        api_client.post("/api/ask", json={"question": ""}, headers=_headers(csrf)).status_code
        == 422
    )
    assert (
        api_client.post(
            "/api/ask",
            json={"question": "x" * 1001},
            headers=_headers(csrf),
        ).status_code
        == 422
    )
    extra = api_client.post(
        "/api/ask",
        json={"question": "annual leave", "groups": ["finance"]},
        headers=_headers(csrf),
    )
    assert extra.status_code == 422


def test_ask_outcomes_and_internal_error(api_client: TestClient, db_dsn: str) -> None:
    csrf = _login(api_client, "alice")
    question = "How many days of annual leave do employees get?"
    answered = api_client.post("/api/ask", json={"question": question}, headers=_headers(csrf))
    body = answered.json()
    assert answered.status_code == 200
    assert body["status"] == "answered"
    assert body["sources"]

    api_client.app.state.answer_model = ScriptedAnswerModel(_reply("NO_ANSWER", False))
    abstained = api_client.post("/api/ask", json={"question": question}, headers=_headers(csrf))
    assert abstained.status_code == 200
    assert abstained.json()["status"] == "abstained"
    assert abstained.json()["sources"] == []
    assert abstained.json()["text"] == abstention_text("en")

    class FailingModel:
        def generate(
            self, system: str, documents: list[dict[str, Any]], question: str
        ) -> ModelReply:
            raise TypeError("Could not resolve authentication method")

    api_client.app.state.answer_model = FailingModel()
    failed = api_client.post("/api/ask", json={"question": question}, headers=_headers(csrf))
    assert failed.status_code == 500
    assert failed.json()["status"] == "error"
    assert failed.json()["audit_id"] is None
    assert failed.json()["text"] == error_text("en")
    with connect(db_dsn) as conn:
        row = conn.execute(
            """
            SELECT actor, outcome, flags FROM answer_audit
            ORDER BY created_at DESC LIMIT 1
            """
        ).fetchone()
    assert row is not None
    assert row[0] == "user:alice"
    assert row[1] == "error"
    assert "internal_error" in row[2]
    history = api_client.get("/api/history")
    statuses = {item["status"] for item in history.json()["items"]}
    assert {"answered", "abstained", "error"} <= statuses
    answered_items = [item for item in history.json()["items"] if item["status"] == "answered"]
    assert answered_items and answered_items[0]["sources"]


def test_ask_queue_busy(api_client: TestClient) -> None:
    csrf = _login(api_client, "alice")
    api_client.app.state.settings = api_client.app.state.settings.model_copy(
        update={"ask_queue_timeout_seconds": 0.05}
    )
    slots = api_client.app.state.ask_slots
    acquired = 0
    try:
        for _ in range(api_client.app.state.settings.max_concurrent_asks):
            assert slots.acquire(blocking=False)
            acquired += 1
        busy = api_client.post(
            "/api/ask",
            json={"question": "How many days of annual leave do employees get?"},
            headers=_headers(csrf),
        )
        assert busy.status_code == 503
        assert busy.json() == {"detail": "busy"}
        slots.release()
        acquired -= 1
        ok = api_client.post(
            "/api/ask",
            json={"question": "How many days of annual leave do employees get?"},
            headers=_headers(csrf),
        )
        assert ok.status_code == 200
    finally:
        for _ in range(acquired):
            slots.release()


def test_groups_refresh_on_live_session(api_client: TestClient, db_dsn: str) -> None:
    csrf = _login(api_client, "alice")
    api_client.app.state.settings = api_client.app.state.settings.model_copy(
        update={"abstain_min_rerank_score": 0.0}
    )
    model = ScriptedAnswerModel(_reply())
    api_client.app.state.answer_model = model
    question = "What is the Finance Director's approval limit?"
    first = api_client.post("/api/ask", json={"question": question}, headers=_headers(csrf))
    assert first.status_code == 200
    assert "Finance Director" not in json.dumps(model.calls[0][1], ensure_ascii=False)
    with connect(db_dsn) as conn:
        set_user_groups(conn, "alice", ["all_staff", "finance"])
    second = api_client.post("/api/ask", json={"question": question}, headers=_headers(csrf))
    assert second.status_code == 200
    assert "Finance Director" in json.dumps(model.calls[1][1], ensure_ascii=False)


def test_history_pages_and_actor_filter(api_client: TestClient) -> None:
    csrf = _login(api_client, "fatima")
    api_client.post(
        "/api/ask",
        json={"question": "How many days of annual leave do employees get?"},
        headers=_headers(csrf),
    )
    csrf = _login(api_client, "alice")
    for index in range(3):
        time.sleep(0.01)
        response = api_client.post(
            "/api/ask",
            json={"question": f"How many days of annual leave do employees get? {index}"},
            headers=_headers(csrf),
        )
        assert response.status_code == 200
    page = api_client.get("/api/history", params={"limit": 2})
    assert page.status_code == 200
    payload = page.json()
    assert len(payload["items"]) == 2
    assert payload["next_cursor"] is not None
    assert payload["items"][0]["created_at"] >= payload["items"][1]["created_at"]
    second = api_client.get("/api/history", params={"limit": 2, "cursor": payload["next_cursor"]})
    assert second.status_code == 200
    assert len(second.json()["items"]) == 1
    assert second.json()["next_cursor"] is None


def test_review_visibility_validation_and_queue(api_client: TestClient, db_dsn: str) -> None:
    finance_csrf = _login(api_client, "fatima")
    api_client.app.state.answer_model = ScriptedAnswerModel(_reply())
    finance = api_client.post(
        "/api/ask",
        json={"question": "What is the Finance Director's approval limit?"},
        headers=_headers(finance_csrf),
    )
    assert finance.status_code == 200
    audit_id = finance.json()["audit_id"]

    api_client.cookies.clear()
    rehan_csrf = _login(api_client, "rehan")
    queue = api_client.get("/api/review/queue")
    assert queue.status_code == 200
    assert audit_id not in {item["audit_id"] for item in queue.json()["items"]}
    assert api_client.get(f"/api/review/items/{audit_id}").status_code == 404
    assert (
        api_client.post(
            f"/api/review/items/{audit_id}/reviews",
            json={"verdict": "correct", "citation_ok": True},
            headers=_headers(rehan_csrf),
        ).status_code
        == 404
    )

    api_client.cookies.clear()
    farah_csrf = _login(api_client, "farah")
    queue = api_client.get("/api/review/queue")
    assert audit_id in {item["audit_id"] for item in queue.json()["items"]}
    assert api_client.get(f"/api/review/items/{audit_id}").status_code == 200
    assert (
        api_client.post(
            f"/api/review/items/{audit_id}/reviews",
            json={"verdict": "correct"},
            headers=_headers(farah_csrf),
        ).status_code
        == 422
    )
    review = api_client.post(
        f"/api/review/items/{audit_id}/reviews",
        json={"verdict": "correct", "citation_ok": True, "notes": "looks right"},
        headers=_headers(farah_csrf),
    )
    assert review.status_code == 201
    assert review.json()["reviewer_username"] == "farah"
    unreviewed = api_client.get("/api/review/queue", params={"status": "unreviewed"})
    assert audit_id not in {item["audit_id"] for item in unreviewed.json()["items"]}
    reviewed = api_client.get("/api/review/queue", params={"status": "reviewed"})
    matching = [item for item in reviewed.json()["items"] if item["audit_id"] == audit_id]
    assert matching and matching[0]["review_count"] == 1
    assert matching[0]["latest_verdict"] == "correct"

    api_client.cookies.clear()
    _login(api_client, "ayesha")
    assert api_client.get(f"/api/review/items/{audit_id}").status_code == 200

    abstain_csrf = _login(api_client, "fatima")
    api_client.app.state.answer_model = ScriptedAnswerModel(_reply("NO_ANSWER", False))
    abstained = api_client.post(
        "/api/ask",
        json={"question": "How many days of annual leave do employees get?"},
        headers=_headers(abstain_csrf),
    )
    abstained_id = abstained.json()["audit_id"]
    api_client.cookies.clear()
    farah_csrf = _login(api_client, "farah")
    assert (
        api_client.post(
            f"/api/review/items/{abstained_id}/reviews",
            json={"verdict": "missed_answer", "citation_ok": True},
            headers=_headers(farah_csrf),
        ).status_code
        == 422
    )
    assert (
        api_client.post(
            f"/api/review/items/{abstained_id}/reviews",
            json={"verdict": "correct"},
            headers=_headers(farah_csrf),
        ).status_code
        == 422
    )

    api_client.app.state.answer_model = type(
        "Failing",
        (),
        {
            "generate": lambda self, system, documents, question: (_ for _ in ()).throw(
                TypeError("boom")
            )
        },
    )()
    error_csrf = _login(api_client, "fatima")
    failed = api_client.post(
        "/api/ask",
        json={"question": "How many days of annual leave do employees get?"},
        headers=_headers(error_csrf),
    )
    assert failed.status_code == 500
    with connect(db_dsn) as conn:
        error_id = str(
            conn.execute(
                "SELECT id FROM answer_audit WHERE outcome = 'error' "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()[0]
        )
    api_client.cookies.clear()
    farah_csrf = _login(api_client, "farah")
    conflict = api_client.post(
        f"/api/review/items/{error_id}/reviews",
        json={"verdict": "correct_abstention"},
        headers=_headers(farah_csrf),
    )
    assert conflict.status_code == 409
    error_queue = api_client.get("/api/review/queue", params={"outcome": "error"})
    assert error_id in {item["audit_id"] for item in error_queue.json()["items"]}
    assert error_id not in {
        item["audit_id"] for item in api_client.get("/api/review/queue").json()["items"]
    }


def test_startup_rejects_pending_migrations_and_stale_index(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
) -> None:
    with connect(db_dsn) as conn:
        conn.execute("DELETE FROM schema_migrations WHERE version = '0004_accounts_reviews.sql'")
    pending_app = create_app(
        settings_for,
        embedder=fake_embedder,
        reranker=fake_reranker,
        answer_model=ScriptedAnswerModel(_reply()),
    )
    with pytest.raises(RuntimeError, match="pending migrations"), TestClient(pending_app):
        pass

    with connect(db_dsn) as conn:
        conn.execute(
            "INSERT INTO schema_migrations (version, checksum) "
            "VALUES ('0004_accounts_reviews.sql', 'x')"
        )
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    with connect(db_dsn) as conn:
        conn.execute("UPDATE documents SET pipeline_fingerprint = 'stale'")
    stale_app = create_app(
        settings_for,
        embedder=fake_embedder,
        reranker=fake_reranker,
        answer_model=ScriptedAnswerModel(_reply()),
    )
    with pytest.raises(RuntimeError, match="different pipeline"), TestClient(stale_app):
        pass


def test_prune_keeps_groups_with_user_members(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    with connect(db_dsn) as conn:
        create_user(conn, "haris", "Haris", PASSWORD, "user", ["hr"])
    reduced = sample_manifest.model_copy(
        update={"groups": [group for group in sample_manifest.groups if group.id != "hr"]}
    )
    _ingest_sample(reduced, sample_repo, settings_for, fake_counter, fake_embedder, prune=True)
    with connect(db_dsn) as conn:
        assert conn.execute("SELECT count(*) FROM groups WHERE id = 'hr'").fetchone()[0] == 1
