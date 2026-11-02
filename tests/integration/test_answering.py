from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from policy_assistant.answering.model import (
    AnswerModelError,
    CitationRef,
    ModelReply,
    ReplySegment,
)
from policy_assistant.answering.prompts import abstention_text, error_text
from policy_assistant.answering.service import answer
from policy_assistant.cli import app
from policy_assistant.config import Settings
from policy_assistant.db.connection import connect
from policy_assistant.ingest.manifest import Manifest
from policy_assistant.ingest.pipeline import ingest
from policy_assistant.retrieval.access import Principal
from tests.fakes import HashingEmbedder, OverlapReranker, ScriptedAnswerModel

pytestmark = pytest.mark.integration


def _ingest_sample(
    manifest: Manifest,
    root: Path,
    settings: Settings,
    counter: Callable[[str], int],
    embedder: HashingEmbedder,
    *,
    force: bool = False,
) -> None:
    ingest(
        manifest,
        repo_root=root,
        only_ids=None,
        force=force,
        prune=False,
        count_tokens=counter,
        embedder=embedder,
        settings=settings,
    )


def _reply(
    text: str = "Employees get 20 working days of annual leave.",
    citations: tuple[CitationRef, ...] = (),
    *,
    stop_reason: str = "end_turn",
) -> ModelReply:
    return ModelReply(
        segments=(ReplySegment(text, citations),),
        stop_reason=stop_reason,
        stop_details=None,
        model_served="claude-served",
        request_id="req_test",
        usage={"input_tokens": 10, "output_tokens": 5},
        unsupported_citation_types=(),
    )


def _ask(
    conn: Any,
    question: str,
    model: ScriptedAnswerModel,
    settings: Settings,
    embedder: HashingEmbedder,
    reranker: OverlapReranker,
) -> Any:
    return answer(
        conn,
        question,
        Principal(frozenset({"all_staff"})),
        actor="test",
        embedder=embedder,
        reranker=reranker,
        model=model,
        settings=settings,
    )


def test_end_to_end_answered_and_audit_snapshot(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    reply = _reply(
        "Employees get 20 working days of annual leave.",
        (CitationRef(0, 1, 2, "20 working days of paid annual leave"),),
    )
    with connect(db_dsn) as conn:
        result = _ask(
            conn,
            "How many days of annual leave do employees get?",
            ScriptedAnswerModel(reply),
            settings_for,
            fake_embedder,
            fake_reranker,
        )
        row = conn.execute(
            """
            SELECT retrieval, answer, llm, outcome, answer_text
            FROM answer_audit WHERE id = %s
            """,
            (result.audit_id,),
        ).fetchone()
    assert result.outcome == "answered"
    assert "[1]" in result.text
    assert result.sources[0]["chunk_id"] is not None
    assert result.sources[0]["start_sentence"] == 1
    assert result.sources[0]["end_sentence"] == 2
    assert row is not None
    assert row[3] == "answered"
    assert row[1]["sources"][0]["chunk_id"] == result.sources[0]["chunk_id"]
    assert any("20 working days" in chunk["text"] for chunk in row[0]["chunks"])
    assert row[2]["request_id"] == "req_test"
    assert row[1]["raw_text"] == "Employees get 20 working days of annual leave."


def test_indistinguishable_abstentions_and_no_finance_in_calls(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    no_answer = _reply("NO_ANSWER")
    finance_model = ScriptedAnswerModel(no_answer)
    absent_model = ScriptedAnswerModel(no_answer)
    with connect(db_dsn) as conn:
        finance = _ask(
            conn,
            "What is the Finance Director's approval limit?",
            finance_model,
            settings_for,
            fake_embedder,
            fake_reranker,
        )
        absent = _ask(
            conn,
            "What is the policy on submarine maintenance?",
            absent_model,
            settings_for,
            fake_embedder,
            fake_reranker,
        )
    assert finance.outcome.startswith("abstained_")
    assert absent.outcome.startswith("abstained_")
    assert finance.text == abstention_text("en")
    assert absent.text == abstention_text("en")
    for model in (finance_model, absent_model):
        for _, documents, _ in model.calls:
            payload = json.dumps(documents, ensure_ascii=False)
            assert "finance-approvals-en" not in payload
            assert "Finance Director" not in payload


@pytest.mark.parametrize(
    ("reply", "expected_outcome", "expected_text", "expected_flag"),
    [
        (_reply("partial", stop_reason="refusal"), "refused", "abstain", None),
        (_reply("partial", stop_reason="max_tokens"), "error", "error", "truncated"),
        (_reply("NO_ANSWER"), "abstained_model", "abstain", None),
        (_reply("Answer without citations."), "abstained_uncited", "abstain", None),
    ],
)
def test_outcome_rules(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
    reply: ModelReply,
    expected_outcome: str,
    expected_text: str,
    expected_flag: str | None,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    with connect(db_dsn) as conn:
        result = _ask(
            conn,
            "How many days of annual leave do employees get?",
            ScriptedAnswerModel(reply),
            settings_for,
            fake_embedder,
            fake_reranker,
        )
        row = conn.execute(
            "SELECT outcome, answer, llm FROM answer_audit WHERE id = %s",
            (result.audit_id,),
        ).fetchone()
    assert result.outcome == expected_outcome
    if expected_text == "abstain":
        assert result.text == abstention_text("en")
    else:
        assert result.text == error_text("en")
    if expected_flag:
        assert expected_flag in result.flags
    assert row is not None and row[0] == expected_outcome
    assert row[1] is not None
    assert row[2] is not None


def test_api_error_outcome(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    with connect(db_dsn) as conn:
        result = _ask(
            conn,
            "How many days of annual leave do employees get?",
            ScriptedAnswerModel(AnswerModelError("boom", status_code=500, request_id="req_err")),
            settings_for,
            fake_embedder,
            fake_reranker,
        )
        row = conn.execute(
            "SELECT outcome, answer, llm FROM answer_audit WHERE id = %s",
            (result.audit_id,),
        ).fetchone()
    assert result.outcome == "error"
    assert result.flags == ("api_error",)
    assert result.text == error_text("en")
    assert row is not None and row[0] == "error"
    assert row[1] == {"segments": [], "sources": [], "raw_text": ""}
    assert row[2]["request_id"] == "req_err"


def test_abstained_retrieval_makes_no_model_call(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    model = ScriptedAnswerModel(_reply("NO_ANSWER"))
    with connect(db_dsn) as conn:
        result = answer(
            conn,
            "What is the Finance Director's approval limit?",
            Principal(frozenset()),
            actor="test",
            embedder=fake_embedder,
            reranker=fake_reranker,
            model=model,
            settings=settings_for,
        )
        row = conn.execute(
            "SELECT answer, llm, outcome FROM answer_audit WHERE id = %s",
            (result.audit_id,),
        ).fetchone()
    assert result.outcome == "abstained_retrieval"
    assert model.calls == []
    assert row is not None
    assert row[0] is None
    assert row[1] is None
    assert row[2] == "abstained_retrieval"


def test_force_ingest_leaves_audit_row_unchanged(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    reply = _reply(
        "Employees get 20 working days.",
        (CitationRef(0, 0, 1, "20 working days"),),
    )
    with connect(db_dsn) as conn:
        result = _ask(
            conn,
            "How many days of annual leave do employees get?",
            ScriptedAnswerModel(reply),
            settings_for,
            fake_embedder,
            fake_reranker,
        )
        before = conn.execute(
            "SELECT * FROM answer_audit WHERE id = %s", (result.audit_id,)
        ).fetchone()
    _ingest_sample(
        sample_manifest,
        sample_repo,
        settings_for,
        fake_counter,
        fake_embedder,
        force=True,
    )
    with connect(db_dsn) as conn:
        after = conn.execute(
            "SELECT * FROM answer_audit WHERE id = %s", (result.audit_id,)
        ).fetchone()
    assert json.dumps(before, default=str, sort_keys=True) == json.dumps(
        after, default=str, sort_keys=True
    )


def test_audit_insert_failure_raises_without_answer(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)

    def fail(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("audit write failed")

    monkeypatch.setattr("policy_assistant.answering.service._insert_audit", fail)
    with connect(db_dsn) as conn, pytest.raises(RuntimeError, match="audit write failed"):
        _ask(
            conn,
            "How many days of annual leave do employees get?",
            ScriptedAnswerModel(_reply("Answer.", (CitationRef(0, 0, 1, "x"),))),
            settings_for,
            fake_embedder,
            fake_reranker,
        )
    with connect(db_dsn) as conn:
        count = conn.execute("SELECT count(*) FROM answer_audit").fetchone()[0]
    assert count == 0


def test_cli_ask_list_show_and_insert_failure(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    reply = _reply(
        "Employees get 20 working days.",
        (CitationRef(0, 0, 1, "20 working days"),),
    )
    model = ScriptedAnswerModel(reply)
    monkeypatch.setattr(
        "policy_assistant.cli.SentenceTransformerEmbedder",
        lambda settings: fake_embedder,
    )
    monkeypatch.setattr(
        "policy_assistant.cli.CrossEncoderReranker",
        lambda settings: fake_reranker,
    )
    monkeypatch.setattr(
        "policy_assistant.cli.build_answer_model",
        lambda settings: model,
    )
    env = {"UPA_DATABASE_URL": db_dsn}
    result = CliRunner().invoke(
        app,
        ["ask", "How many days of annual leave do employees get?", "--group", "all_staff"],
        env=env,
    )
    assert result.exit_code == 0, result.output
    assert "Sources:" in result.output
    audit_line = next(line for line in result.output.splitlines() if line.startswith("audit: "))
    audit_id = audit_line.removeprefix("audit: ").strip()

    listing = CliRunner().invoke(app, ["audit", "list"], env=env)
    assert listing.exit_code == 0
    assert audit_id in listing.output

    shown = CliRunner().invoke(app, ["audit", "show", audit_id], env=env)
    assert shown.exit_code == 0
    payload = json.loads(shown.output)
    assert payload["outcome"] == "answered"

    def fail(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("audit write failed")

    monkeypatch.setattr("policy_assistant.answering.service._insert_audit", fail)
    failed = CliRunner().invoke(
        app,
        ["ask", "How many days of annual leave do employees get?", "--group", "all_staff"],
        env=env,
    )
    assert failed.exit_code == 1


def test_urdu_abstention_uses_urdu_message(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    with connect(db_dsn) as conn:
        result = answer(
            conn,
            "کیا آپ کے پاس آبدوز کی پالیسی ہے؟",
            Principal(frozenset({"all_staff"})),
            actor="test",
            embedder=fake_embedder,
            reranker=fake_reranker,
            model=ScriptedAnswerModel(_reply("NO_ANSWER")),
            settings=settings_for,
        )
    assert result.language == "ur"
    assert result.outcome.startswith("abstained_")
    assert result.text == abstention_text("ur")
