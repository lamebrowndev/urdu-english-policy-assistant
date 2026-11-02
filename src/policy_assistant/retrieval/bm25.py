"""Okapi BM25 in SQL over the folded term index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg

_BM25_SQL = """
WITH q AS (SELECT unnest(%(terms)s::text[]) AS term),
stats AS (
    SELECT count(*)::float8 AS n, avg(bm25_length)::float8 AS avgdl
    FROM chunks
),
df AS (
    SELECT ct.term, count(*)::float8 AS df
    FROM chunk_terms ct
    JOIN q USING (term)
    GROUP BY ct.term
)
SELECT ct.chunk_id,
       sum(
           ln(1 + (s.n - df.df + 0.5) / (df.df + 0.5))
           * ct.tf / (
               ct.tf + %(k1)s * (
                   1 - %(b)s + %(b)s * c.bm25_length / s.avgdl
               )
           )
       ) AS score
FROM chunk_terms ct
JOIN df USING (term)
JOIN chunks c ON c.id = ct.chunk_id AND c.document_id = ANY(%(allowed)s::text[])
CROSS JOIN stats s
GROUP BY ct.chunk_id
ORDER BY score DESC, ct.chunk_id ASC
LIMIT %(k)s
"""


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: int
    score: float


def search_bm25(
    conn: psycopg.Connection[Any],
    terms: list[str],
    allowed_ids: list[str],
    k: int,
    k1: float,
    b: float,
) -> list[Hit]:
    """Search BM25 with global corpus statistics and an ACL predicate."""
    unique_terms = list(dict.fromkeys(terms))
    if not unique_terms or not allowed_ids or k <= 0:
        return []
    stats = conn.execute("SELECT count(*)::float8, avg(bm25_length)::float8 FROM chunks").fetchone()
    if stats is None:
        return []
    count, avgdl = stats
    if count == 0 or avgdl is None or float(avgdl) == 0:
        return []
    rows = conn.execute(
        _BM25_SQL,
        {
            "terms": unique_terms,
            "allowed": allowed_ids,
            "k": k,
            "k1": k1,
            "b": b,
        },
    ).fetchall()
    return [Hit(chunk_id=int(row[0]), score=float(row[1])) for row in rows]
