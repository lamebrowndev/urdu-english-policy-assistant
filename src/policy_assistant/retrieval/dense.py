"""Filtered dense search with pgvector HNSW iterative scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg
from numpy import float32
from numpy.typing import NDArray
from pgvector.psycopg import register_vector


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: int
    distance: float


def search_dense(
    conn: psycopg.Connection[Any],
    query_vector: NDArray[float32],
    allowed_ids: list[str],
    k: int,
    ef_search: int,
) -> list[Hit]:
    """Return top-k allowed chunks by cosine distance, filtering in SQL."""
    if not allowed_ids or k <= 0:
        return []
    register_vector(conn)
    with conn.transaction():
        conn.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
        rows = conn.execute(
            """
            SELECT id, embedding <=> %(q)s AS distance
            FROM chunks
            WHERE document_id = ANY(%(allowed)s::text[])
              AND embedding IS NOT NULL
            ORDER BY embedding <=> %(q)s, id
            LIMIT %(k)s
            """,
            {"q": query_vector, "allowed": allowed_ids, "k": k},
        ).fetchall()
    return [Hit(chunk_id=int(row[0]), distance=float(row[1])) for row in rows]
