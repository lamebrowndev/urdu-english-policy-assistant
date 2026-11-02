"""Score an evaluation run from human reviews, with Wilson 95% intervals.

An item counts as correct when its latest review verdict is `correct` or
`correct_abstention`. `partially_correct` counts as not correct (reported
separately). A system `error` outcome counts as not correct without a review.
Items still waiting for a review are excluded from accuracy and reported, and the
report is marked incomplete until none remain.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg

CORRECT_VERDICTS = frozenset({"correct", "correct_abstention"})
CATEGORY_ORDER = ("answerable", "unanswerable", "restricted")
Z_95 = 1.959963984540054


class ScoringError(RuntimeError):
    """An unknown run or inconsistent scoring input."""


@dataclass(frozen=True, slots=True)
class ScoredItem:
    item_id: str
    audit_id: str
    language: str
    category: str
    outcome: str
    verdict: str | None
    citation_ok: bool | None
    tags: tuple[str, ...] = ()


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion; None when total is 0."""
    if total <= 0:
        return None
    if not 0 <= successes <= total:
        raise ValueError("successes must be between 0 and total")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def judgement(item: ScoredItem) -> bool | None:
    """True or False once scored; None while the item waits for a human review."""
    if item.outcome == "error":
        return False
    if item.verdict is None:
        return None
    return item.verdict in CORRECT_VERDICTS


def _bucket(items: list[ScoredItem]) -> dict[str, Any]:
    judged = [judgement(item) for item in items]
    scored = [value for value in judged if value is not None]
    correct = sum(1 for value in scored if value)
    interval = wilson_interval(correct, len(scored))
    return {
        "items": len(items),
        "scored": len(scored),
        "unreviewed": sum(1 for value in judged if value is None),
        "correct": correct,
        "accuracy": correct / len(scored) if scored else None,
        "ci95": [interval[0], interval[1]] if interval else None,
        "partially_correct": sum(1 for item in items if item.verdict == "partially_correct"),
        "errors": sum(1 for item in items if item.outcome == "error"),
        "answered": sum(1 for item in items if item.outcome == "answered"),
        "abstained": sum(
            1
            for item in items
            if item.outcome.startswith("abstained_") or item.outcome == "refused"
        ),
    }


def summarize(items: list[ScoredItem]) -> dict[str, Any]:
    languages = sorted({item.language for item in items})
    categories = [c for c in CATEGORY_ORDER if any(item.category == c for item in items)]
    tags = sorted({tag for item in items for tag in item.tags})
    reviewed_answers = [
        item for item in items if item.outcome == "answered" and item.citation_ok is not None
    ]
    overall = _bucket(items)
    return {
        "complete": overall["unreviewed"] == 0,
        "overall": overall,
        "by_language": {
            language: _bucket([item for item in items if item.language == language])
            for language in languages
        },
        "by_category": {
            category: _bucket([item for item in items if item.category == category])
            for category in categories
        },
        "by_tag": {tag: _bucket([item for item in items if tag in item.tags]) for tag in tags},
        "citations": {
            "reviewed_answers": len(reviewed_answers),
            "citation_ok": sum(1 for item in reviewed_answers if item.citation_ok),
        },
    }


def load_run(
    conn: psycopg.Connection[Any], run_id: uuid.UUID
) -> tuple[dict[str, Any], list[ScoredItem]]:
    run = conn.execute(
        """
        SELECT id, created_at, actor, dataset_name, dataset_sha256, split, item_count, settings
        FROM eval_runs WHERE id = %s
        """,
        (run_id,),
    ).fetchone()
    if run is None:
        raise ScoringError(f"unknown evaluation run {run_id}")
    rows = conn.execute(
        """
        SELECT r.item_id, r.audit_id, r.language, r.category, a.outcome,
               latest.verdict, latest.citation_ok, r.tags
        FROM eval_results r
        JOIN answer_audit a ON a.id = r.audit_id
        LEFT JOIN LATERAL (
            SELECT v.verdict, v.citation_ok
            FROM answer_reviews v
            WHERE v.audit_id = r.audit_id
            ORDER BY v.created_at DESC, v.id DESC
            LIMIT 1
        ) latest ON true
        WHERE r.run_id = %s
        ORDER BY r.item_id
        """,
        (run_id,),
    ).fetchall()
    created_at = run[1]
    meta = {
        "id": str(run[0]),
        "created_at": created_at.isoformat() if isinstance(created_at, datetime) else created_at,
        "actor": run[2],
        "dataset_name": run[3],
        "dataset_sha256": run[4],
        "split": run[5],
        "item_count": int(run[6]),
        "settings": run[7],
    }
    items = [
        ScoredItem(
            item_id=str(row[0]),
            audit_id=str(row[1]),
            language=str(row[2]),
            category=str(row[3]),
            outcome=str(row[4]),
            verdict=str(row[5]) if row[5] is not None else None,
            citation_ok=row[6],
            tags=tuple(str(tag) for tag in (row[7] or [])),
        )
        for row in rows
    ]
    return meta, items


def score_run(conn: psycopg.Connection[Any], run_id: uuid.UUID) -> dict[str, Any]:
    meta, items = load_run(conn, run_id)
    report = summarize(items)
    missing = meta["item_count"] - len(items)
    report["run"] = meta
    report["missing_items"] = missing
    report["complete"] = bool(report["complete"]) and missing == 0
    report["pending_review"] = [
        {"item_id": item.item_id, "audit_id": item.audit_id}
        for item in items
        if judgement(item) is None
    ]
    return report
