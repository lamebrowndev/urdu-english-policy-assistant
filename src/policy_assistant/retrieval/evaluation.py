"""Retrieval evaluation metrics and dataset validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import psycopg

from policy_assistant.config import Settings
from policy_assistant.retrieval.access import Principal
from policy_assistant.retrieval.models import Embedder, Reranker
from policy_assistant.retrieval.retriever import (
    RetrievalMode,
    RetrievalResult,
    retrieve,
    with_top_k,
)
from policy_assistant.text.normalize import fold


class EvaluationError(RuntimeError):
    """Raised for an invalid dataset or unverifiable gold evidence."""


@dataclass(frozen=True, slots=True)
class GoldItem:
    document_id: str
    quote: str


@dataclass(frozen=True, slots=True)
class EvalItem:
    id: str
    question: str
    language: Literal["en", "ur"]
    groups: tuple[str, ...]
    gold: tuple[GoldItem, ...]
    forbidden_documents: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalChunk:
    document_id: str
    text: str


@dataclass(frozen=True, slots=True)
class ItemMetrics:
    hit: bool
    recall: float
    reciprocal_rank: float
    leaks: int


def load_dataset(path: Path) -> list[EvalItem]:
    """Load and structurally validate a JSONL retrieval dataset."""
    items: list[EvalItem] = []
    seen_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationError(f"cannot read dataset {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise EvaluationError(f"line {line_number}: expected an object")
        item_id = str(payload.get("id", ""))
        if not item_id:
            raise EvaluationError(f"line {line_number}: missing id")
        if item_id in seen_ids:
            raise EvaluationError(f"line {line_number}: duplicate id {item_id!r}")
        seen_ids.add(item_id)
        question = str(payload.get("question", ""))
        if not question:
            raise EvaluationError(f"item {item_id}: missing question")
        language = payload.get("language")
        if language not in {"en", "ur"}:
            raise EvaluationError(f"item {item_id}: language must be 'en' or 'ur'")
        groups = payload.get("groups")
        if not isinstance(groups, list) or not all(isinstance(g, str) for g in groups):
            raise EvaluationError(f"item {item_id}: groups must be a list of strings")
        gold_payload = payload.get("gold")
        if not isinstance(gold_payload, list):
            raise EvaluationError(f"item {item_id}: gold must be a list")
        gold: list[GoldItem] = []
        for entry in gold_payload:
            if not isinstance(entry, dict):
                raise EvaluationError(f"item {item_id}: invalid gold entry")
            document_id = entry.get("document_id")
            quote = entry.get("quote")
            if not isinstance(document_id, str) or not isinstance(quote, str) or not quote:
                raise EvaluationError(f"item {item_id}: invalid gold entry")
            gold.append(GoldItem(document_id=document_id, quote=quote))
        forbidden = payload.get("forbidden_documents", [])
        if not isinstance(forbidden, list) or not all(
            isinstance(document_id, str) for document_id in forbidden
        ):
            raise EvaluationError(f"item {item_id}: forbidden_documents must be a list of strings")
        items.append(
            EvalItem(
                id=item_id,
                question=question,
                language=language,
                groups=tuple(groups),
                gold=tuple(gold),
                forbidden_documents=tuple(forbidden),
            )
        )
    if not items:
        raise EvaluationError(f"dataset {path} contains no items")
    return items


def validate_gold(conn: psycopg.Connection[Any], items: list[EvalItem]) -> None:
    """Every gold quote must occur in a stored chunk of its document."""
    for item in items:
        for gold in item.gold:
            rows = conn.execute(
                "SELECT text FROM chunks WHERE document_id = %s",
                (gold.document_id,),
            ).fetchall()
            folded_quote = fold(gold.quote)
            if not any(folded_quote in fold(str(row[0])) for row in rows):
                raise EvaluationError(
                    f"item {item.id}: gold quote not found in {gold.document_id}: {gold.quote!r}"
                )


def rank_metrics(
    chunks: list[EvalChunk],
    gold: tuple[GoldItem, ...],
    forbidden_documents: set[str],
    k: int,
) -> ItemMetrics:
    """Compute hit@k, recall@k, RR@k, and leaks for one ranking."""
    top = chunks[:k]
    matched: set[int] = set()
    first_rank: int | None = None
    for rank, chunk in enumerate(top, start=1):
        for index, target in enumerate(gold):
            if chunk.document_id != target.document_id:
                continue
            if fold(target.quote) in fold(chunk.text):
                matched.add(index)
                if first_rank is None:
                    first_rank = rank
    leaks = sum(1 for chunk in top if chunk.document_id in forbidden_documents)
    return ItemMetrics(
        hit=bool(matched),
        recall=len(matched) / len(gold) if gold else 0.0,
        reciprocal_rank=1.0 / first_rank if first_rank is not None else 0.0,
        leaks=leaks,
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _latency(runs: list[ItemRun]) -> dict[str, dict[str, float | None]]:
    stages = sorted({stage for run in runs for stage in run.timings})
    return {
        stage: {
            "mean": _mean([run.timings[stage] for run in runs if stage in run.timings]),
            "p95": _p95([run.timings[stage] for run in runs if stage in run.timings]),
        }
        for stage in stages
    }


def _summary(runs: list[ItemRun]) -> dict[str, Any]:
    with_gold = [run for run in runs if run.item.gold]
    return {
        "n": len(runs),
        "n_with_gold": len(with_gold),
        "hit_at_k": _mean([1.0 if run.metrics.hit else 0.0 for run in with_gold]),
        "recall_at_k": _mean([run.metrics.recall for run in with_gold]),
        "mrr_at_k": _mean([run.metrics.reciprocal_rank for run in with_gold]),
        "leaks": sum(run.metrics.leaks for run in runs),
        "latency_ms": _latency(runs),
    }


@dataclass(frozen=True, slots=True)
class ItemRun:
    item: EvalItem
    result: RetrievalResult
    metrics: ItemMetrics
    timings: dict[str, float]
    cross_lingual: bool


def _item_payload(run: ItemRun, k: int) -> dict[str, Any]:
    return {
        "id": run.item.id,
        "question": run.item.question,
        "language": run.item.language,
        "groups": list(run.item.groups),
        "gold_count": len(run.item.gold),
        "hit_at_k": run.metrics.hit,
        "recall_at_k": run.metrics.recall,
        "reciprocal_rank": run.metrics.reciprocal_rank,
        "leaks": run.metrics.leaks,
        "cross_lingual": run.cross_lingual,
        "top": [
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
            }
            for rank, chunk in enumerate(run.result.chunks[:k], start=1)
        ],
        "timings_ms": run.timings,
    }


def evaluate(
    conn: psycopg.Connection[Any],
    items: list[EvalItem],
    *,
    modes: list[RetrievalMode],
    k: int,
    embedder: Embedder,
    reranker: Reranker,
    settings: Settings,
) -> dict[str, Any]:
    """Run every item in every mode and aggregate metrics and slices."""
    validate_gold(conn, items)
    languages = {
        str(row[0]): str(row[1])
        for row in conn.execute("SELECT id, language FROM documents").fetchall()
    }
    settings_for_k = with_top_k(settings, k)
    mode_reports: dict[str, Any] = {}
    for mode in modes:
        runs: list[ItemRun] = []
        for item in items:
            result = retrieve(
                conn,
                item.question,
                Principal(frozenset(item.groups)),
                mode=mode,
                embedder=embedder,
                reranker=reranker,
                settings=settings_for_k,
            )
            metrics = rank_metrics(
                [EvalChunk(chunk.document_id, chunk.text) for chunk in result.chunks],
                item.gold,
                set(item.forbidden_documents),
                k,
            )
            cross_lingual = bool(item.gold) and all(
                languages.get(gold.document_id) != item.language for gold in item.gold
            )
            runs.append(
                ItemRun(
                    item=item,
                    result=result,
                    metrics=metrics,
                    timings=result.timings_ms,
                    cross_lingual=cross_lingual,
                )
            )
        languages_present = sorted({run.item.language for run in runs})
        by_language = {
            language: _summary([run for run in runs if run.item.language == language])
            for language in languages_present
        }
        cross_true = [run for run in runs if run.item.gold and run.cross_lingual]
        cross_false = [run for run in runs if run.item.gold and not run.cross_lingual]
        mode_reports[mode] = {
            "overall": _summary(runs),
            "slices": {
                "language": by_language,
                "cross_lingual": {
                    "true": _summary(cross_true),
                    "false": _summary(cross_false),
                },
            },
            "items": [_item_payload(run, k) for run in runs],
        }
    return {"k": k, "modes": mode_reports}
