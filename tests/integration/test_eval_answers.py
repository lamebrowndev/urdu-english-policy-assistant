from __future__ import annotations

import json
import re
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psycopg
import pytest
from argon2 import PasswordHasher
from typer.testing import CliRunner

from policy_assistant.answering.model import CitationRef, ModelReply, ReplySegment
from policy_assistant.auth.users import create_user
from policy_assistant.cli import app
from policy_assistant.config import Settings
from policy_assistant.db.connection import connect
from policy_assistant.ingest.manifest import Manifest
from policy_assistant.ingest.pipeline import ingest
from policy_assistant.reviews.store import add_review
from policy_assistant.reviews.store import item as review_item
from policy_assistant.reviews.store import queue as review_queue
from tests.fakes import HashingEmbedder, OverlapReranker, ScriptedAnswerModel

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "evals" / "answers.sample.jsonl"


def _reply() -> ModelReply:
    return ModelReply(
        segments=(ReplySegment("Cited answer.", (CitationRef(0, 0, 1, "cited"),)),),
        stop_reason="end_turn",
        stop_details=None,
        model_served="fake",
        request_id="req_eval",
        usage={"input_tokens": 1, "output_tokens": 1},
        unsupported_citation_types=(),
    )


@pytest.fixture
def eval_env(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Any]:
    monkeypatch.setattr(
        "policy_assistant.auth.passwords.HASHER",
        PasswordHasher(time_cost=1, memory_cost=1024, parallelism=1),
    )
    ingest(
        sample_manifest,
        repo_root=sample_repo,
        only_ids=None,
        force=False,
        prune=False,
        count_tokens=fake_counter,
        embedder=fake_embedder,
        settings=settings_for,
    )
    model = ScriptedAnswerModel(_reply())
    monkeypatch.setattr("policy_assistant.cli.SentenceTransformerEmbedder", lambda s: fake_embedder)
    monkeypatch.setattr("policy_assistant.cli.CrossEncoderReranker", lambda s: fake_reranker)
    monkeypatch.setattr("policy_assistant.cli.build_answer_model", lambda s: model)
    dataset = tmp_path / "answers.sample.jsonl"
    shutil.copyfile(SAMPLE, dataset)
    return {"dsn": db_dsn, "dataset": dataset, "env": {"UPA_DATABASE_URL": db_dsn}}


def _run(env: dict[str, Any], *args: str, stdin: str | None = None) -> Any:
    return CliRunner().invoke(app, list(args), env=env["env"], input=stdin)


def _run_id(output: str) -> str:
    match = re.search(r"^run: ([0-9a-f-]{36})$", output, re.M)
    assert match is not None, output
    return match.group(1)


def test_run_review_and_score(eval_env: dict[str, Any]) -> None:
    dataset = str(eval_env["dataset"])
    result = _run(eval_env, "eval", "answers", "--dataset", dataset, "--split", "sample", "--yes")
    assert result.exit_code == 0, result.output
    run_id = _run_id(result.output)
    assert result.output.count("[") >= 12
    assert f"actor=eval:{run_id}" in result.output

    with connect(eval_env["dsn"]) as conn:
        run = conn.execute(
            "SELECT split, item_count, dataset_sha256, settings FROM eval_runs WHERE id = %s",
            (run_id,),
        ).fetchone()
        assert run is not None
        assert run[0] == "sample"
        assert run[1] == 12
        assert len(run[2]) == 64
        assert run[3]["prompt_version"] == "answer-v1"
        results = conn.execute(
            """
            SELECT r.item_id, r.audit_id, a.outcome, a.actor
            FROM eval_results r JOIN answer_audit a ON a.id = r.audit_id
            WHERE r.run_id = %s ORDER BY r.item_id
            """,
            (run_id,),
        ).fetchall()
        assert len(results) == 12
        assert {row[3] for row in results} == {f"eval:{run_id}"}

    preliminary = _run(eval_env, "eval", "score", run_id, "--json")
    assert preliminary.exit_code == 0, preliminary.output
    report = json.loads(preliminary.output)
    assert report["complete"] is False
    assert report["overall"]["items"] == 12
    assert len(report["pending_review"]) == 12 - report["overall"]["errors"]

    with connect(eval_env["dsn"]) as conn:
        admin = create_user(conn, "ayesha", "Ayesha", "correct horse battery", "admin", [])
        visible = review_queue(
            conn,
            admin,
            status="all",
            outcome=None,
            limit=50,
            cursor=None,
            actor=f"eval:{run_id}",
        )[0]
        assert len(visible) == 12
        first = review_item(conn, admin, uuid.UUID(str(results[0][1])))
        assert first is not None
        assert first["evaluation"]["run_id"] == run_id
        assert first["evaluation"]["reference_answer"]
        for _item_id, audit_id, outcome, _actor in results:
            if outcome == "error":
                continue
            if outcome == "answered":
                add_review(conn, admin, uuid.UUID(str(audit_id)), "correct", True, "")
            else:
                add_review(conn, admin, uuid.UUID(str(audit_id)), "correct_abstention", None, "")

    final = _run(eval_env, "eval", "score", run_id, "--json")
    report = json.loads(final.output)
    errors = report["overall"]["errors"]
    assert report["complete"] is True
    assert report["overall"]["scored"] == 12
    assert report["overall"]["correct"] == 12 - errors
    assert report["by_language"]["ur"]["items"] == 6
    assert report["by_language"]["en"]["items"] == 6
    table = _run(eval_env, "eval", "score", run_id)
    assert table.exit_code == 0
    assert "status: COMPLETE" in table.output
    assert "95% CI" in table.output


def test_test_split_must_be_frozen_and_unchanged(eval_env: dict[str, Any]) -> None:
    dataset = eval_env["dataset"]
    test_set = dataset.with_name("answers.test.jsonl")
    shutil.copyfile(dataset, test_set)
    unfrozen = _run(
        eval_env, "eval", "answers", "--dataset", str(test_set), "--split", "test", "--yes"
    )
    assert unfrozen.exit_code == 1
    assert "not frozen" in unfrozen.output

    frozen = _run(eval_env, "eval", "freeze", "--dataset", str(test_set))
    assert frozen.exit_code == 0, frozen.output
    again = _run(eval_env, "eval", "freeze", "--dataset", str(test_set))
    assert again.exit_code == 1

    limited = _run(
        eval_env,
        "eval",
        "answers",
        "--dataset",
        str(test_set),
        "--split",
        "test",
        "--limit",
        "2",
        "--yes",
    )
    assert limited.exit_code == 1
    assert "not allowed for the test split" in limited.output

    test_set.write_text(test_set.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    changed = _run(
        eval_env, "eval", "answers", "--dataset", str(test_set), "--split", "test", "--yes"
    )
    assert changed.exit_code == 1
    assert "changed after it was frozen" in changed.output
    with connect(eval_env["dsn"]) as conn:
        assert conn.execute("SELECT count(*) FROM eval_runs").fetchone() == (0,)


def test_confirmation_and_evidence_checks(eval_env: dict[str, Any]) -> None:
    dataset = eval_env["dataset"]
    declined = _run(
        eval_env,
        "eval",
        "answers",
        "--dataset",
        str(dataset),
        "--split",
        "dev",
        "--limit",
        "2",
        stdin="n\n",
    )
    assert declined.exit_code != 0
    assert "paid API call" in declined.output

    lines = dataset.read_text(encoding="utf-8").splitlines()
    leaked = json.loads(lines[4])
    assert leaked["category"] == "restricted"
    leaked["groups"] = ["finance"]
    bad = dataset.with_name("answers.bad.jsonl")
    bad.write_text(json.dumps(leaked, ensure_ascii=False) + "\n", encoding="utf-8")
    rejected = _run(eval_env, "eval", "answers", "--dataset", str(bad), "--split", "dev", "--yes")
    assert rejected.exit_code == 1
    assert "restricted, but groups" in rejected.output
    with connect(eval_env["dsn"]) as conn:
        assert conn.execute("SELECT count(*) FROM eval_runs").fetchone() == (0,)


def test_eval_tables_are_append_only(eval_env: dict[str, Any]) -> None:
    result = _run(
        eval_env,
        "eval",
        "answers",
        "--dataset",
        str(eval_env["dataset"]),
        "--split",
        "sample",
        "--limit",
        "1",
        "--yes",
    )
    assert result.exit_code == 0, result.output
    with connect(eval_env["dsn"], autocommit=True) as conn:
        for statement in (
            "UPDATE eval_runs SET split = 'dev'",
            "DELETE FROM eval_runs",
            "TRUNCATE eval_runs CASCADE",
            "UPDATE eval_results SET category = 'restricted'",
            "DELETE FROM eval_results",
            "TRUNCATE eval_results",
        ):
            with pytest.raises(psycopg.Error, match="append-only"):
                conn.execute(statement)
