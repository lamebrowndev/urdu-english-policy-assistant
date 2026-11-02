"""End-to-end retrieval over BM25, dense vectors, and reciprocal rank fusion."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import psycopg

from policy_assistant.config import Settings
from policy_assistant.retrieval.access import (
    Principal,
    allowed_document_ids,
    ensure_authorized,
)
from policy_assistant.retrieval.bm25 import Hit as BM25Hit
from policy_assistant.retrieval.bm25 import search_bm25
from policy_assistant.retrieval.context import retrieval_text
from policy_assistant.retrieval.dense import search_dense
from policy_assistant.retrieval.fusion import rrf
from policy_assistant.retrieval.models import Embedder, Reranker
from policy_assistant.text.terms import bm25_terms

RetrievalMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
_MODES = {"bm25", "dense", "hybrid", "hybrid_rerank"}


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: int
    document_id: str
    document_title: str
    document_language: str
    page_start: int
    page_end: int
    section_heading: str | None
    text: str
    sentences: tuple[str, ...]
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_distance: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "document_language": self.document_language,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_heading": self.section_heading,
            "text": self.text,
            "sentences": list(self.sentences),
            "bm25_rank": self.bm25_rank,
            "bm25_score": self.bm25_score,
            "dense_rank": self.dense_rank,
            "dense_distance": self.dense_distance,
            "rrf_score": self.rrf_score,
            "rerank_score": self.rerank_score,
        }


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    mode: RetrievalMode
    query_terms: tuple[str, ...]
    principal_groups: frozenset[str]
    allowed_document_count: int
    chunks: tuple[RetrievedChunk, ...]
    timings_ms: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "query_terms": list(self.query_terms),
            "principal_groups": sorted(self.principal_groups),
            "allowed_document_count": self.allowed_document_count,
            "chunks": [chunk.as_dict() for chunk in self.chunks],
            "timings_ms": self.timings_ms,
        }


@dataclass(frozen=True, slots=True)
class _ChunkRow:
    chunk_id: int
    document_id: str
    document_title: str
    document_language: str
    page_start: int
    page_end: int
    section_heading: str | None
    text: str
    sentences: tuple[str, ...]


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _fetch_rows(
    conn: psycopg.Connection[Any], chunk_ids: list[int], allowed: list[str]
) -> dict[int, _ChunkRow]:
    if not chunk_ids:
        return {}
    rows = conn.execute(
        """
        SELECT c.id, c.document_id, d.title, d.language, c.page_start, c.page_end,
               c.section_heading, c.text, c.sentences
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id = ANY(%s)
        """,
        (chunk_ids,),
    ).fetchall()
    ensure_authorized([str(row[1]) for row in rows], allowed)
    return {
        int(row[0]): _ChunkRow(
            chunk_id=int(row[0]),
            document_id=str(row[1]),
            document_title=str(row[2]),
            document_language=str(row[3]),
            page_start=int(row[4]),
            page_end=int(row[5]),
            section_heading=str(row[6]) if row[6] is not None else None,
            text=str(row[7]),
            sentences=tuple(str(sentence) for sentence in row[8]),
        )
        for row in rows
    }


def _build_chunks(
    ordered_ids: list[int],
    rows: dict[int, _ChunkRow],
    *,
    bm25_ranks: dict[int, int],
    bm25_scores: dict[int, float],
    dense_ranks: dict[int, int],
    dense_distances: dict[int, float],
    rrf_scores: dict[int, float],
    rerank_scores: dict[int, float],
) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for chunk_id in ordered_ids:
        row = rows.get(chunk_id)
        if row is None:
            continue
        chunks.append(
            RetrievedChunk(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                document_title=row.document_title,
                document_language=row.document_language,
                page_start=row.page_start,
                page_end=row.page_end,
                section_heading=row.section_heading,
                text=row.text,
                sentences=row.sentences,
                bm25_rank=bm25_ranks.get(chunk_id),
                bm25_score=bm25_scores.get(chunk_id),
                dense_rank=dense_ranks.get(chunk_id),
                dense_distance=dense_distances.get(chunk_id),
                rrf_score=rrf_scores.get(chunk_id),
                rerank_score=rerank_scores.get(chunk_id),
            )
        )
    return chunks


def retrieve(
    conn: psycopg.Connection[Any],
    query: str,
    principal: Principal,
    *,
    mode: RetrievalMode,
    embedder: Embedder,
    reranker: Reranker,
    settings: Settings,
) -> RetrievalResult:
    """Retrieve chunks under the principal's ACL and rank them."""
    if mode not in _MODES:
        raise ValueError(f"unknown retrieval mode: {mode}")
    started = time.perf_counter()
    allowed = allowed_document_ids(conn, principal)
    timings: dict[str, float] = {"allow": _elapsed_ms(started)}
    query_terms = tuple(dict.fromkeys(bm25_terms(query)))
    empty_result = RetrievalResult(
        query=query,
        mode=mode,
        query_terms=query_terms,
        principal_groups=frozenset(principal.groups),
        allowed_document_count=len(allowed),
        chunks=(),
        timings_ms=timings,
    )
    if not allowed:
        return empty_result

    bm25_hits: list[BM25Hit] = []
    dense_hits: list[Any] = []
    dense_ranks: dict[int, int] = {}
    dense_distances: dict[int, float] = {}
    rrf_scores: dict[int, float] = {}
    selected_ids: list[int]
    depth = (
        settings.retrieve_candidates
        if mode in {"hybrid", "hybrid_rerank"}
        else settings.retrieve_top_k
    )

    if mode in {"bm25", "hybrid", "hybrid_rerank"}:
        stage = time.perf_counter()
        bm25_hits = search_bm25(
            conn,
            list(query_terms),
            allowed,
            depth,
            settings.bm25_k1,
            settings.bm25_b,
        )
        timings["bm25"] = _elapsed_ms(stage)
    if mode in {"dense", "hybrid", "hybrid_rerank"}:
        stage = time.perf_counter()
        query_vectors = embedder.embed([query])
        timings["embed_query"] = _elapsed_ms(stage)
        stage = time.perf_counter()
        dense_hits = search_dense(
            conn,
            query_vectors[0],
            allowed,
            depth,
            settings.hnsw_ef_search,
        )
        timings["dense"] = _elapsed_ms(stage)

    bm25_ranks = {hit.chunk_id: rank for rank, hit in enumerate(bm25_hits, start=1)}
    bm25_scores = {hit.chunk_id: hit.score for hit in bm25_hits}
    dense_ranks = {hit.chunk_id: rank for rank, hit in enumerate(dense_hits, start=1)}
    dense_distances = {hit.chunk_id: hit.distance for hit in dense_hits}

    if mode == "bm25":
        selected_ids = [hit.chunk_id for hit in bm25_hits]
    elif mode == "dense":
        selected_ids = [hit.chunk_id for hit in dense_hits]
    else:
        stage = time.perf_counter()
        fused = rrf(
            [
                [hit.chunk_id for hit in bm25_hits],
                [hit.chunk_id for hit in dense_hits],
            ],
            settings.rrf_k,
        )
        timings["fusion"] = _elapsed_ms(stage)
        rrf_scores = dict(fused)
        if mode == "hybrid":
            selected_ids = [chunk_id for chunk_id, _ in fused[: settings.retrieve_top_k]]
        else:
            selected_ids = [chunk_id for chunk_id, _ in fused[: settings.rerank_candidates]]

    stage = time.perf_counter()
    rows = _fetch_rows(conn, selected_ids, allowed)
    timings["fetch"] = _elapsed_ms(stage)

    rerank_scores: dict[int, float] = {}
    if mode == "hybrid_rerank":
        present_ids = [chunk_id for chunk_id in selected_ids if chunk_id in rows]
        passages = [
            retrieval_text(
                rows[chunk_id].document_title,
                rows[chunk_id].section_heading,
                rows[chunk_id].text,
            )
            for chunk_id in present_ids
        ]
        stage = time.perf_counter()
        scores = reranker.score(query, passages)
        timings["rerank"] = _elapsed_ms(stage)
        rerank_scores = dict(zip(present_ids, scores, strict=False))
        ordered_ids = sorted(
            present_ids,
            key=lambda chunk_id: (-rerank_scores.get(chunk_id, float("-inf")), chunk_id),
        )[: settings.retrieve_top_k]
    else:
        ordered_ids = selected_ids[: settings.retrieve_top_k]

    chunks = _build_chunks(
        ordered_ids,
        rows,
        bm25_ranks=bm25_ranks,
        bm25_scores=bm25_scores,
        dense_ranks=dense_ranks,
        dense_distances=dense_distances,
        rrf_scores=rrf_scores,
        rerank_scores=rerank_scores,
    )
    return RetrievalResult(
        query=query,
        mode=mode,
        query_terms=query_terms,
        principal_groups=frozenset(principal.groups),
        allowed_document_count=len(allowed),
        chunks=tuple(chunks),
        timings_ms=timings,
    )
