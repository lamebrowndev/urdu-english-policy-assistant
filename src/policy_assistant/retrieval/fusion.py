"""Reciprocal rank fusion."""

from __future__ import annotations


def rrf(rankings: list[list[int]], k: int) -> list[tuple[int, float]]:
    """Fuse rankings with 1/(k + rank), ranks starting at one."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
