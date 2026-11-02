"""Shared audit-row read model."""

from __future__ import annotations

import uuid
from typing import Any

import psycopg


def load_audit_row(conn: psycopg.Connection[Any], audit_id: uuid.UUID) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT id, created_at, actor, principal_groups, question,
               question_language, outcome, answer_text, answer, retrieval,
               llm, flags, prompt_version, pipeline_fingerprint, latency_ms
        FROM answer_audit
        WHERE id = %s
        """,
        (audit_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": str(row[0]),
        "created_at": row[1].isoformat(),
        "actor": row[2],
        "principal_groups": row[3],
        "question": row[4],
        "question_language": row[5],
        "outcome": row[6],
        "answer_text": row[7],
        "answer": row[8],
        "retrieval": row[9],
        "llm": row[10],
        "flags": row[11],
        "prompt_version": row[12],
        "pipeline_fingerprint": row[13],
        "latency_ms": row[14],
    }
