"""Run an answer dataset through the real answering pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from policy_assistant.answering.model import AnswerModel
from policy_assistant.answering.prompts import PROMPT_VERSION
from policy_assistant.answering.service import AnswerResult, answer
from policy_assistant.config import Settings
from policy_assistant.evaluation.dataset import AnswerItem, Split
from policy_assistant.ingest.pipeline import pipeline_fingerprint
from policy_assistant.retrieval.access import Principal
from policy_assistant.retrieval.models import Embedder, Reranker

OnResult = Callable[[int, AnswerItem, AnswerResult], None]


def run_settings(settings: Settings) -> dict[str, Any]:
    """Everything that can change an answer, recorded with the run."""
    return {
        "llm_model": settings.llm_model,
        "llm_effort": settings.llm_effort,
        "llm_max_tokens": settings.llm_max_tokens,
        "llm_fallbacks": settings.llm_fallbacks,
        "answer_mode": settings.answer_mode,
        "abstain_min_rerank_score": settings.abstain_min_rerank_score,
        "retrieve_candidates": settings.retrieve_candidates,
        "rerank_candidates": settings.rerank_candidates,
        "retrieve_top_k": settings.retrieve_top_k,
        "prompt_version": PROMPT_VERSION,
        "pipeline_fingerprint": pipeline_fingerprint(settings),
    }


def start_run(
    conn: psycopg.Connection[Any],
    *,
    actor: str,
    dataset_name: str,
    dataset_sha256: str,
    split: Split,
    item_count: int,
    settings: Settings,
) -> str:
    with conn.transaction():
        row = conn.execute(
            """
            INSERT INTO eval_runs (
                actor, dataset_name, dataset_sha256, split, item_count, settings
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                actor,
                dataset_name,
                dataset_sha256,
                split,
                item_count,
                Jsonb(run_settings(settings)),
            ),
        ).fetchone()
    if row is None:
        raise RuntimeError("eval run insert returned no id")
    if not conn.autocommit:
        conn.commit()
    return str(row[0])


def run_items(
    conn: psycopg.Connection[Any],
    run_id: str,
    items: list[AnswerItem],
    *,
    embedder: Embedder,
    reranker: Reranker,
    model: AnswerModel,
    settings: Settings,
    on_result: OnResult | None = None,
) -> list[AnswerResult]:
    """Ask every item as its own principal; each answer is audited as `eval:<run_id>`.

    Stops at the first exception: `answer()` has already audited it, and a broken
    setup (for example missing credentials) would fail every remaining item too.
    """
    results: list[AnswerResult] = []
    for index, item in enumerate(items, start=1):
        result = answer(
            conn,
            item.question,
            Principal(frozenset(item.groups)),
            actor=f"eval:{run_id}",
            embedder=embedder,
            reranker=reranker,
            model=model,
            settings=settings,
        )
        with conn.transaction():
            conn.execute(
                """
                INSERT INTO eval_results (
                    run_id, item_id, audit_id, language, category, tags, reference_answer
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    item.id,
                    result.audit_id,
                    item.language,
                    item.category,
                    list(item.tags),
                    item.reference_answer,
                ),
            )
        if not conn.autocommit:
            conn.commit()
        results.append(result)
        if on_result is not None:
            on_result(index, item, result)
    return results
