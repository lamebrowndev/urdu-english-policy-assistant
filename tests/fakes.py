"""Deterministic fakes for model-free tests."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

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
