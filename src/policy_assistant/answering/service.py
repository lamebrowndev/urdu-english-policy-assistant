"""Answer service: retrieval gate, model call, citations, and audit row."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import psycopg
from psycopg.types.json import Jsonb

from policy_assistant.answering.citations import CitationResolution, resolve_answer
from policy_assistant.answering.documents import build_documents
from policy_assistant.answering.language import answer_language
from policy_assistant.answering.model import (
    AnswerModel,
    AnswerModelError,
    ModelReply,
)
from policy_assistant.answering.prompts import (
    NO_ANSWER,
    PROMPT_VERSION,
    abstention_text,
    error_text,
    system_prompt,
)
from policy_assistant.config import Settings
from policy_assistant.ingest.pipeline import pipeline_fingerprint
from policy_assistant.retrieval.access import (
    Principal,
    allowed_document_ids,
    ensure_authorized,
)
from policy_assistant.retrieval.models import Embedder, Reranker
from policy_assistant.retrieval.retriever import RetrievalResult, retrieve

AnswerOutcome = Literal[
    "answered",
    "abstained_retrieval",
    "abstained_model",
    "abstained_uncited",
    "refused",
    "error",
]


@dataclass(frozen=True, slots=True)
class AnswerResult:
    audit_id: str
    outcome: AnswerOutcome
    language: Literal["ur", "en"]
    text: str
    sources: tuple[dict[str, Any], ...]
    flags: tuple[str, ...]


def _llm_payload(
    settings: Settings,
    reply: ModelReply | None,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    return {
        "model_requested": settings.llm_model,
        "model_served": reply.model_served if reply is not None else None,
        "effort": settings.llm_effort,
        "max_tokens": settings.llm_max_tokens,
        "fallbacks": settings.llm_fallbacks,
        "stop_reason": reply.stop_reason if reply is not None else None,
        "stop_details": reply.stop_details if reply is not None else None,
        "request_id": reply.request_id if reply is not None else request_id,
        "usage": reply.usage if reply is not None else {},
        "unsupported_citation_types": (
            list(reply.unsupported_citation_types) if reply is not None else []
        ),
    }


def _answer_payload(
    reply: ModelReply,
    resolution: CitationResolution,
    raw_text: str,
) -> dict[str, Any]:
    return {
        "segments": list(resolution.segments),
        "sources": [source.as_dict() for source in resolution.sources],
        "raw_text": raw_text,
    }


def _insert_audit(
    conn: psycopg.Connection[Any],
    *,
    actor: str,
    principal: Principal,
    question: str,
    language: Literal["ur", "en"],
    outcome: AnswerOutcome,
    answer_text: str,
    answer_payload: dict[str, Any] | None,
    retrieval: RetrievalResult,
    llm_payload: dict[str, Any] | None,
    flags: list[str],
    settings: Settings,
    latency_ms: dict[str, float],
) -> str:
    with conn.transaction():
        row = conn.execute(
            """
            INSERT INTO answer_audit (
                actor, principal_groups, question, question_language, outcome,
                answer_text, answer, retrieval, llm, flags, prompt_version,
                pipeline_fingerprint, latency_ms
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                actor,
                sorted(principal.groups),
                question,
                language,
                outcome,
                answer_text,
                Jsonb(answer_payload) if answer_payload is not None else None,
                Jsonb(retrieval.as_dict()),
                Jsonb(llm_payload) if llm_payload is not None else None,
                flags,
                PROMPT_VERSION,
                pipeline_fingerprint(settings),
                Jsonb(latency_ms),
            ),
        ).fetchone()
    if row is None:
        raise RuntimeError("answer audit insert returned no id")
    if not conn.autocommit:
        conn.commit()
    return str(row[0])


def answer(
    conn: psycopg.Connection[Any],
    question: str,
    principal: Principal,
    *,
    actor: str,
    embedder: Embedder,
    reranker: Reranker,
    model: AnswerModel,
    settings: Settings,
) -> AnswerResult:
    """Answer a question and write exactly one audit row."""
    started = time.perf_counter()
    language = answer_language(question)
    retrieval = retrieve(
        conn,
        question,
        principal,
        mode=settings.answer_mode,
        embedder=embedder,
        reranker=reranker,
        settings=settings,
    )

    top_score = retrieval.chunks[0].rerank_score if retrieval.chunks else None
    gate = not retrieval.chunks or (
        settings.answer_mode == "hybrid_rerank"
        and (top_score is None or top_score < settings.abstain_min_rerank_score)
    )
    outcome: AnswerOutcome
    text: str
    flags: list[str] = []
    answer_payload: dict[str, Any] | None = None
    llm_payload: dict[str, Any] | None = None
    llm_ms = 0.0

    if gate:
        outcome = "abstained_retrieval"
        text = abstention_text(language)
    else:
        allowed = allowed_document_ids(conn, principal)
        ensure_authorized(
            [chunk.document_id for chunk in retrieval.chunks],
            allowed,
        )
        documents = build_documents(retrieval.chunks)
        llm_started = time.perf_counter()
        try:
            reply = model.generate(system_prompt(language), documents, question)
        except AnswerModelError as exc:
            llm_ms = (time.perf_counter() - llm_started) * 1000.0
            outcome = "error"
            flags.append("api_error")
            text = error_text(language)
            answer_payload = {"segments": [], "sources": [], "raw_text": ""}
            llm_payload = _llm_payload(settings, None, request_id=exc.request_id)
        else:
            llm_ms = (time.perf_counter() - llm_started) * 1000.0
            resolution = resolve_answer(reply.segments, retrieval.chunks)
            raw_text = "".join(segment.text for segment in reply.segments)
            flags.extend(resolution.flags)
            if reply.unsupported_citation_types:
                flags.append("unsupported_citation")
            if reply.stop_reason == "refusal":
                outcome = "refused"
                text = abstention_text(language)
            elif reply.stop_reason == "max_tokens":
                outcome = "error"
                flags.append("truncated")
                text = error_text(language)
            elif raw_text.strip().startswith(NO_ANSWER):
                outcome = "abstained_model"
                text = abstention_text(language)
            elif not resolution.sources:
                outcome = "abstained_uncited"
                text = abstention_text(language)
            else:
                outcome = "answered"
                text = resolution.text
            answer_payload = _answer_payload(reply, resolution, raw_text)
            llm_payload = _llm_payload(settings, reply)

    latency_ms = dict(retrieval.timings_ms)
    latency_ms["llm"] = llm_ms
    latency_ms["total"] = (time.perf_counter() - started) * 1000.0
    audit_id = _insert_audit(
        conn,
        actor=actor,
        principal=principal,
        question=question,
        language=language,
        outcome=outcome,
        answer_text=text,
        answer_payload=answer_payload,
        retrieval=retrieval,
        llm_payload=llm_payload,
        flags=flags,
        settings=settings,
        latency_ms=latency_ms,
    )
    sources = tuple(source for source in (answer_payload or {}).get("sources", []))
    return AnswerResult(
        audit_id=audit_id,
        outcome=outcome,
        language=language,
        text=text,
        sources=sources,
        flags=tuple(flags),
    )
