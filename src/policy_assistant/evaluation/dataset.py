"""Answer-evaluation datasets: loading, validation, and freezing.

A dataset is JSONL, one question per line:

    {"id": "en-annual-days", "question": "...", "language": "en",
     "groups": ["all_staff"], "category": "answerable",
     "reference_answer": "20 working days per calendar year.",
     "evidence": [{"document_id": "leave-policy-en", "quote": "20 working days"}]}

Categories:
- answerable: a document the asker may read answers the question (evidence required,
  every evidence document must be readable by `groups`);
- restricted: only documents the asker may NOT read answer it (evidence required,
  no evidence document may be readable by `groups`); the correct behaviour is to abstain;
- unanswerable: no document answers it (no evidence); the correct behaviour is to abstain.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, get_args

import psycopg

from policy_assistant.answering.language import answer_language
from policy_assistant.retrieval.access import Principal, allowed_document_ids
from policy_assistant.text.normalize import fold

Category = Literal["answerable", "unanswerable", "restricted"]
Split = Literal["sample", "dev", "test"]
CATEGORIES: tuple[str, ...] = get_args(Category)
SPLITS: tuple[str, ...] = get_args(Split)

_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_FIELDS = {
    "id",
    "question",
    "language",
    "groups",
    "category",
    "tags",
    "reference_answer",
    "evidence",
}


class DatasetError(RuntimeError):
    """An invalid, unverifiable, or tampered evaluation dataset."""


@dataclass(frozen=True, slots=True)
class Evidence:
    document_id: str
    quote: str


@dataclass(frozen=True, slots=True)
class AnswerItem:
    id: str
    question: str
    language: Literal["en", "ur"]
    groups: tuple[str, ...]
    category: Category
    reference_answer: str
    evidence: tuple[Evidence, ...]
    tags: tuple[str, ...] = ()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_item(payload: Any, where: str) -> AnswerItem:
    if not isinstance(payload, dict):
        raise DatasetError(f"{where}: expected a JSON object")
    unknown = sorted(set(payload) - _FIELDS)
    if unknown:
        raise DatasetError(f"{where}: unknown fields {unknown}")
    item_id = payload.get("id")
    if not isinstance(item_id, str) or not _ID_RE.fullmatch(item_id):
        raise DatasetError(f"{where}: id must match {_ID_RE.pattern}")
    where = f"item {item_id}"
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise DatasetError(f"{where}: question is required")
    language = payload.get("language")
    if language not in {"en", "ur"}:
        raise DatasetError(f"{where}: language must be 'en' or 'ur'")
    if answer_language(question) != language:
        raise DatasetError(
            f"{where}: declared language {language!r} does not match the question's script"
        )
    groups = payload.get("groups")
    if not isinstance(groups, list) or not all(isinstance(g, str) and g for g in groups):
        raise DatasetError(f"{where}: groups must be a list of group ids")
    category = payload.get("category")
    if category not in CATEGORIES:
        raise DatasetError(f"{where}: category must be one of {list(CATEGORIES)}")
    tags = payload.get("tags", [])
    if (
        not isinstance(tags, list)
        or not all(isinstance(tag, str) and _TAG_RE.fullmatch(tag) for tag in tags)
        or len(set(tags)) != len(tags)
    ):
        raise DatasetError(f"{where}: tags must be unique strings matching {_TAG_RE.pattern}")
    reference = payload.get("reference_answer")
    if not isinstance(reference, str) or not reference.strip():
        raise DatasetError(f"{where}: reference_answer is required")
    raw_evidence = payload.get("evidence", [])
    if not isinstance(raw_evidence, list):
        raise DatasetError(f"{where}: evidence must be a list")
    evidence: list[Evidence] = []
    for entry in raw_evidence:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"document_id", "quote"}
            or not isinstance(entry["document_id"], str)
            or not isinstance(entry["quote"], str)
            or not entry["quote"].strip()
        ):
            raise DatasetError(f"{where}: evidence entries need document_id and a quote")
        evidence.append(Evidence(document_id=entry["document_id"], quote=entry["quote"]))
    if category == "unanswerable" and evidence:
        raise DatasetError(f"{where}: unanswerable items must not have evidence")
    if category in {"answerable", "restricted"} and not evidence:
        raise DatasetError(f"{where}: {category} items need at least one evidence quote")
    return AnswerItem(
        id=item_id,
        question=question.strip(),
        language=cast(Literal["en", "ur"], language),
        groups=tuple(groups),
        category=cast(Category, category),
        reference_answer=reference.strip(),
        evidence=tuple(evidence),
        tags=tuple(tags),
    )


def load_answer_dataset(path: Path) -> list[AnswerItem]:
    """Parse and structurally validate a JSONL answer dataset."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"cannot read dataset {path}: {exc}") from exc
    items: list[AnswerItem] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"line {number}: invalid JSON: {exc}") from exc
        item = _parse_item(payload, f"line {number}")
        if item.id in seen:
            raise DatasetError(f"line {number}: duplicate id {item.id!r}")
        seen.add(item.id)
        items.append(item)
    if not items:
        raise DatasetError(f"dataset {path} contains no items")
    return items


def verify_evidence(conn: psycopg.Connection[Any], items: list[AnswerItem]) -> None:
    """Check every quote against the index and every category against the current ACLs."""
    for item in items:
        allowed = set(allowed_document_ids(conn, Principal(frozenset(item.groups))))
        for evidence in item.evidence:
            rows = conn.execute(
                "SELECT text FROM chunks WHERE document_id = %s",
                (evidence.document_id,),
            ).fetchall()
            if not rows:
                raise DatasetError(
                    f"item {item.id}: evidence document {evidence.document_id!r} is not indexed"
                )
            folded = fold(evidence.quote)
            if not any(folded in fold(str(row[0])) for row in rows):
                raise DatasetError(
                    f"item {item.id}: quote not found in {evidence.document_id}: {evidence.quote!r}"
                )
            readable = evidence.document_id in allowed
            if item.category == "answerable" and not readable:
                raise DatasetError(
                    f"item {item.id}: answerable, but groups {list(item.groups)} cannot read "
                    f"{evidence.document_id}"
                )
            if item.category == "restricted" and readable:
                raise DatasetError(
                    f"item {item.id}: restricted, but groups {list(item.groups)} can read "
                    f"{evidence.document_id}"
                )


def freeze_path(dataset: Path) -> Path:
    return dataset.with_name(dataset.name + ".sha256")


def freeze(dataset: Path) -> str:
    """Record the dataset's SHA-256 next to it; refuses to overwrite an existing record."""
    load_answer_dataset(dataset)
    digest = file_sha256(dataset)
    try:
        with freeze_path(dataset).open("x", encoding="utf-8") as handle:
            handle.write(f"{digest}  {dataset.name}\n")
    except FileExistsError as exc:
        raise DatasetError(
            f"{freeze_path(dataset)} already exists; a frozen dataset must not be re-frozen"
        ) from exc
    return digest


def verify_frozen(dataset: Path) -> str:
    """Return the dataset hash, or raise unless it matches its freeze record."""
    record = freeze_path(dataset)
    if not record.exists():
        raise DatasetError(f"{dataset} is not frozen; run `upa eval freeze --dataset {dataset}`")
    parts = record.read_text(encoding="utf-8").split()
    expected = parts[0] if parts else ""
    actual = file_sha256(dataset)
    if actual != expected:
        raise DatasetError(
            f"{dataset} changed after it was frozen (expected {expected}, found {actual})"
        )
    return actual
