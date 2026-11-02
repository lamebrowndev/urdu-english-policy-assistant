"""Deterministic fakes for model-free tests."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from policy_assistant.answering.model import AnswerModelError, ModelReply
from policy_assistant.retrieval.retriever import RetrievedChunk
from policy_assistant.text.terms import bm25_terms


class HashingEmbedder:
    dimension = 1024
    device = "cpu"

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for term in bm25_terms(text):
                digest = hashlib.sha256(term.encode("utf-8")).hexdigest()
                index = int(digest[:8], 16) % self.dimension
                vectors[row, index] += 1.0
            norm = float(np.linalg.norm(vectors[row]))
            if norm == 0.0:
                vectors[row, 0] = 1.0
            else:
                vectors[row] /= norm
        return vectors


class OverlapReranker:
    device = "cpu"

    def score(self, query: str, passages: list[str]) -> list[float]:
        query_terms = set(bm25_terms(query))
        return [float(len(query_terms & set(bm25_terms(passage)))) for passage in passages]


class ScriptedAnswerModel:
    def __init__(self, reply_or_error: ModelReply | AnswerModelError) -> None:
        self.reply_or_error = reply_or_error
        self.calls: list[tuple[str, list[dict[str, object]], str]] = []

    def generate(
        self, system: str, documents: list[dict[str, object]], question: str
    ) -> ModelReply:
        self.calls.append((system, documents, question))
        if isinstance(self.reply_or_error, AnswerModelError):
            raise self.reply_or_error
        return self.reply_or_error


def make_chunk(
    *,
    chunk_id: int = 1,
    document_id: str = "doc",
    document_title: str = "Document",
    sentences: tuple[str, ...] = ("First sentence.", "Second sentence."),
    page_start: int = 1,
    page_end: int = 1,
    section_heading: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=document_title,
        document_language="en",
        page_start=page_start,
        page_end=page_end,
        section_heading=section_heading,
        text=" ".join(sentences),
        sentences=sentences,
    )
