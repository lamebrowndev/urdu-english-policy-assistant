from pathlib import Path

import pytest

from policy_assistant.retrieval.evaluation import (
    EvalChunk,
    EvaluationError,
    GoldItem,
    load_dataset,
    rank_metrics,
)


def test_rank_metrics_hit_recall_rr_and_leaks() -> None:
    chunks = [
        EvalChunk("doc-a", "alpha"),
        EvalChunk("doc-b", "beta"),
        EvalChunk("doc-a", "gamma"),
    ]
    gold = (GoldItem("doc-a", "alpha"), GoldItem("doc-a", "gamma"))
    metrics = rank_metrics(chunks, gold, {"doc-b"}, 3)
    assert metrics.hit
    assert metrics.recall == 1.0
    assert metrics.reciprocal_rank == 1.0
    assert metrics.leaks == 1


def test_rank_metrics_first_match_rank() -> None:
    chunks = [EvalChunk("doc-a", "alpha"), EvalChunk("doc-a", "gamma")]
    metrics = rank_metrics(chunks, (GoldItem("doc-a", "gamma"),), set(), 2)
    assert metrics.hit
    assert metrics.recall == 1.0
    assert metrics.reciprocal_rank == 0.5


def test_rank_metrics_miss() -> None:
    metrics = rank_metrics([EvalChunk("doc-b", "beta")], (GoldItem("doc-a", "alpha"),), set(), 1)
    assert not metrics.hit
    assert metrics.recall == 0.0
    assert metrics.reciprocal_rank == 0.0
    assert metrics.leaks == 0


def test_gold_matching_survives_diacritics_and_arabic_code_points() -> None:
    chunks = [EvalChunk("doc-a", "مُلازِم يہ پالیسی")]
    metrics = rank_metrics(chunks, (GoldItem("doc-a", "ملازم یہ"),), set(), 1)
    assert metrics.hit


def test_load_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id":"x","question":"q","language":"en","groups":[],"gold":[]}\n'
        '{"id":"x","question":"q2","language":"en","groups":[],"gold":[]}\n',
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="duplicate id"):
        load_dataset(path)


def test_load_dataset_minimal(tmp_path: Path) -> None:
    path = tmp_path / "ok.jsonl"
    path.write_text(
        '{"id":"x","question":"q","language":"en","groups":["all_staff"],'
        '"gold":[{"document_id":"d","quote":"q"}],"forbidden_documents":[]}\n',
        encoding="utf-8",
    )
    items = load_dataset(path)
    assert items[0].id == "x"
    assert items[0].gold[0].document_id == "d"
