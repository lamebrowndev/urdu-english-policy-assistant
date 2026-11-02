"""Embedding and reranker protocols plus their real model implementations."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from policy_assistant.config import Settings


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> NDArray[np.float32]: ...


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


def resolve_device(settings: Settings) -> str:
    """Resolve ``auto`` to MPS when available, otherwise CPU."""
    if settings.device != "auto":
        return settings.device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SentenceTransformerEmbedder:
    """bge-m3 embeddings with unit-normalized float32 rows."""

    dimension = 1024

    def __init__(self, settings: Settings) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = resolve_device(settings)
        self._batch_size = settings.embed_batch_size
        self._model = SentenceTransformer(
            settings.embedding_repo,
            revision=settings.embedding_revision,
            device=self.device,
            model_kwargs={"use_safetensors": True},
        )
        self._model.max_seq_length = 512

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        array: NDArray[np.float32] = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        result: NDArray[np.float32] = (array / norms).astype(np.float32)
        return result


class CrossEncoderReranker:
    """bge-reranker-v2-m3 cross-encoder over retrieval passages."""

    def __init__(self, settings: Settings) -> None:
        from sentence_transformers import CrossEncoder

        self.device = resolve_device(settings)
        self._model = CrossEncoder(
            settings.reranker_repo,
            revision=settings.reranker_revision,
            max_length=512,
            device=self.device,
            model_kwargs={"use_safetensors": True},
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        scores = self._model.predict(pairs)
        return [float(score) for score in np.asarray(scores).reshape(-1)]
