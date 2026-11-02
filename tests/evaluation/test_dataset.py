from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from policy_assistant.evaluation.dataset import (
    DatasetError,
    file_sha256,
    freeze,
    freeze_path,
    load_answer_dataset,
    verify_frozen,
)

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "evals" / "answers.sample.jsonl"


def _item(**overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": "en-annual",
        "question": "How many days of annual leave do employees get?",
        "language": "en",
        "groups": ["all_staff"],
        "category": "answerable",
        "reference_answer": "20 working days.",
        "evidence": [{"document_id": "leave-policy-en", "quote": "20 working days"}],
    }
    item.update(overrides)
    return item


def _write(tmp_path: Path, *items: dict[str, Any]) -> Path:
    path = tmp_path / "answers.dev.jsonl"
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
        encoding="utf-8",
    )
    return path


def test_sample_dataset_is_valid_and_balanced() -> None:
    items = load_answer_dataset(SAMPLE)
    assert len(items) == 12
    assert sum(item.language == "ur" for item in items) == 6
    assert {item.category for item in items} == {"answerable", "unanswerable", "restricted"}
    assert {tag for item in items for tag in item.tags} == {"lookup", "synthesis", "cross-lingual"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"id": "bad id"}, "id must match"),
        ({"question": "  "}, "question is required"),
        ({"language": "fr"}, "language must be"),
        ({"language": "ur"}, "does not match the question's script"),
        ({"groups": "all_staff"}, "groups must be a list"),
        ({"category": "maybe"}, "category must be one of"),
        ({"reference_answer": ""}, "reference_answer is required"),
        ({"evidence": []}, "need at least one evidence quote"),
        ({"category": "unanswerable"}, "must not have evidence"),
        ({"evidence": [{"document_id": "x"}]}, "need document_id and a quote"),
        ({"extra": 1}, "unknown fields"),
        ({"tags": ["Lookup"]}, "tags must be unique strings"),
        ({"tags": ["lookup", "lookup"]}, "tags must be unique strings"),
        ({"tags": "lookup"}, "tags must be unique strings"),
    ],
)
def test_invalid_items_are_rejected(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(DatasetError, match=message):
        load_answer_dataset(_write(tmp_path, _item(**overrides)))


def test_duplicate_ids_and_empty_files_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="duplicate id"):
        load_answer_dataset(_write(tmp_path, _item(), _item()))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="no items"):
        load_answer_dataset(empty)


def test_freeze_is_write_once_and_detects_changes(tmp_path: Path) -> None:
    dataset = _write(tmp_path, _item())
    with pytest.raises(DatasetError, match="not frozen"):
        verify_frozen(dataset)
    digest = freeze(dataset)
    assert digest == file_sha256(dataset)
    assert freeze_path(dataset).read_text(encoding="utf-8").startswith(digest)
    assert verify_frozen(dataset) == digest
    with pytest.raises(DatasetError, match="must not be re-frozen"):
        freeze(dataset)
    dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="changed after it was frozen"):
        verify_frozen(dataset)
