"""Shared index readiness checks for CLI and HTTP startup."""

from __future__ import annotations

from typing import Any

import psycopg

from policy_assistant.config import Settings
from policy_assistant.ingest.pipeline import pipeline_fingerprint


def index_problem(conn: psycopg.Connection[Any], settings: Settings) -> str | None:
    row = conn.execute(
        """
        SELECT count(*) FROM chunks c
        WHERE c.embedding IS NULL
           OR c.bm25_length IS NULL
           OR c.bm25_length <> COALESCE(
               (SELECT sum(tf) FROM chunk_terms ct WHERE ct.chunk_id = c.id),
               0
           )
        """
    ).fetchone()
    if row is not None and int(row[0]) > 0:
        return "index incomplete: run upa ingest"
    stale = conn.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM documents
            WHERE pipeline_fingerprint IS DISTINCT FROM %s
        )
        """,
        (pipeline_fingerprint(settings),),
    ).fetchone()
    if stale is not None and bool(stale[0]):
        return "index built with a different pipeline: run upa ingest"
    return None
