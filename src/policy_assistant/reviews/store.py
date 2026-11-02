"""Review queue, review writes, and user history read models."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Protocol, cast

import psycopg

from policy_assistant.answering.audit_store import load_audit_row
from policy_assistant.retrieval.access import Principal, allowed_document_ids


class Viewer(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def username(self) -> str: ...

    @property
    def role(self) -> str: ...

    @property
    def groups(self) -> frozenset[str]: ...


class ReviewConflict(RuntimeError):
    """An audit outcome that cannot be reviewed."""


_ANSWER_VERDICTS = frozenset({"correct", "partially_correct", "incorrect"})
_ABSTENTION_VERDICTS = frozenset({"correct_abstention", "missed_answer"})
_VISIBILITY = (
    "NOT EXISTS ("
    "SELECT 1 FROM jsonb_array_elements(a.retrieval->'chunks') c "
    "WHERE NOT ((c->>'document_id') = ANY(%(allowed)s::text[]))"
    ")"
)


def _visibility(conn: psycopg.Connection[Any], viewer: Viewer) -> tuple[str, dict[str, Any]]:
    if viewer.role == "admin":
        return "TRUE", {}
    allowed = allowed_document_ids(conn, Principal(viewer.groups))
    return _VISIBILITY, {"allowed": allowed}


def _cursor_clause(cursor: tuple[datetime, uuid.UUID] | None, params: dict[str, Any]) -> str | None:
    if cursor is None:
        return None
    params["cursor_created_at"] = cursor[0]
    params["cursor_id"] = cursor[1]
    return "(a.created_at, a.id) < (%(cursor_created_at)s, %(cursor_id)s)"


def queue(
    conn: psycopg.Connection[Any],
    viewer: Viewer,
    *,
    status: str = "unreviewed",
    outcome: str | None,
    limit: int,
    cursor: tuple[datetime, uuid.UUID] | None,
) -> tuple[list[dict[str, Any]], tuple[datetime, uuid.UUID] | None]:
    if status not in {"unreviewed", "reviewed", "all"}:
        raise ValueError(f"unknown queue status: {status}")
    visibility, params = _visibility(conn, viewer)
    clauses = [visibility]
    if outcome == "error":
        clauses.append("a.outcome = 'error'")
    else:
        clauses.append("a.outcome <> 'error'")
        if outcome is not None:
            params["outcome"] = outcome
            clauses.append("a.outcome = %(outcome)s")
    if status == "unreviewed":
        clauses.append("NOT EXISTS (SELECT 1 FROM answer_reviews r WHERE r.audit_id = a.id)")
    elif status == "reviewed":
        clauses.append("EXISTS (SELECT 1 FROM answer_reviews r WHERE r.audit_id = a.id)")
    cursor_clause = _cursor_clause(cursor, params)
    if cursor_clause is not None:
        clauses.append(cursor_clause)
    params["limit"] = limit + 1
    rows = conn.execute(
        f"""
        SELECT a.id, a.created_at, a.actor, a.question, a.question_language,
               a.outcome, a.flags,
               (SELECT count(*) FROM answer_reviews r WHERE r.audit_id = a.id),
               (
                   SELECT r.verdict FROM answer_reviews r
                   WHERE r.audit_id = a.id
                   ORDER BY r.created_at DESC, r.id DESC
                   LIMIT 1
               )
        FROM answer_audit a
        WHERE {" AND ".join(clauses)}
        ORDER BY a.created_at DESC, a.id DESC
        LIMIT %(limit)s
        """,
        params,
    ).fetchall()
    next_cursor: tuple[datetime, uuid.UUID] | None = None
    if len(rows) > limit:
        page = rows[:limit]
        next_cursor = (cast(datetime, page[-1][1]), cast(uuid.UUID, page[-1][0]))
    else:
        page = rows
    items = [
        {
            "audit_id": str(row[0]),
            "created_at": row[1],
            "actor": row[2],
            "question": row[3],
            "question_language": row[4],
            "outcome": row[5],
            "flags": row[6],
            "review_count": int(row[7]),
            "latest_verdict": row[8],
        }
        for row in page
    ]
    return items, next_cursor


def _visible(conn: psycopg.Connection[Any], viewer: Viewer, audit_id: uuid.UUID) -> bool:
    visibility, params = _visibility(conn, viewer)
    params["audit_id"] = audit_id
    row = conn.execute(
        f"SELECT EXISTS (SELECT 1 FROM answer_audit a WHERE a.id = %(audit_id)s AND {visibility})",
        params,
    ).fetchone()
    return row is not None and bool(row[0])


def item(
    conn: psycopg.Connection[Any], viewer: Viewer, audit_id: uuid.UUID
) -> dict[str, Any] | None:
    if not _visible(conn, viewer, audit_id):
        return None
    payload = load_audit_row(conn, audit_id)
    if payload is None:
        return None
    rows = conn.execute(
        """
        SELECT id, audit_id, reviewer_id, reviewer_username, created_at,
               verdict, citation_ok, notes
        FROM answer_reviews
        WHERE audit_id = %s
        ORDER BY created_at ASC, id ASC
        """,
        (audit_id,),
    ).fetchall()
    payload["reviews"] = [
        {
            "id": str(row[0]),
            "audit_id": str(row[1]),
            "reviewer_id": int(row[2]),
            "reviewer_username": str(row[3]),
            "created_at": cast(datetime, row[4]).isoformat(),
            "verdict": str(row[5]),
            "citation_ok": row[6],
            "notes": str(row[7]),
        }
        for row in rows
    ]
    return payload


def add_review(
    conn: psycopg.Connection[Any],
    viewer: Viewer,
    audit_id: uuid.UUID,
    verdict: str,
    citation_ok: bool | None,
    notes: str,
) -> dict[str, Any] | None:
    if not _visible(conn, viewer, audit_id):
        return None
    row = conn.execute("SELECT outcome FROM answer_audit WHERE id = %s", (audit_id,)).fetchone()
    if row is None:
        return None
    outcome = str(row[0])
    if outcome == "error":
        raise ReviewConflict("error audit rows cannot be reviewed")
    if outcome == "answered":
        if verdict not in _ANSWER_VERDICTS:
            raise ValueError("answered rows require a correct/incorrect verdict")
        if citation_ok is None:
            raise ValueError("answered rows require citation_ok")
    else:
        if verdict not in _ABSTENTION_VERDICTS:
            raise ValueError("non-answers require an abstention verdict")
        if citation_ok is not None:
            raise ValueError("non-answers require citation_ok to be null")
    if len(notes) > 4000:
        raise ValueError("notes must be at most 4000 characters")
    with conn.transaction():
        inserted = conn.execute(
            """
            INSERT INTO answer_reviews (
                audit_id, reviewer_id, reviewer_username, verdict, citation_ok, notes
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, audit_id, reviewer_id, reviewer_username, created_at,
                      verdict, citation_ok, notes
            """,
            (audit_id, viewer.id, viewer.username, verdict, citation_ok, notes),
        ).fetchone()
    if inserted is None:
        raise RuntimeError("review insert returned no row")
    if not conn.autocommit:
        conn.commit()
    return {
        "id": str(inserted[0]),
        "audit_id": str(inserted[1]),
        "reviewer_id": int(inserted[2]),
        "reviewer_username": str(inserted[3]),
        "created_at": cast(datetime, inserted[4]).isoformat(),
        "verdict": str(inserted[5]),
        "citation_ok": inserted[6],
        "notes": str(inserted[7]),
    }


def history(
    conn: psycopg.Connection[Any],
    user: Viewer,
    *,
    limit: int,
    cursor: tuple[datetime, uuid.UUID] | None,
) -> tuple[list[dict[str, Any]], tuple[datetime, uuid.UUID] | None]:
    params: dict[str, Any] = {
        "actor": f"user:{user.username}",
        "limit": limit + 1,
    }
    clauses = ["a.actor = %(actor)s"]
    cursor_clause = _cursor_clause(cursor, params)
    if cursor_clause is not None:
        clauses.append(cursor_clause)
    rows = conn.execute(
        f"""
        SELECT a.id, a.created_at, a.question, a.question_language, a.outcome,
               a.answer_text,
               CASE WHEN a.outcome = 'answered'
                    THEN COALESCE(a.answer->'sources', '[]'::jsonb)
                    ELSE '[]'::jsonb
               END
        FROM answer_audit a
        WHERE {" AND ".join(clauses)}
        ORDER BY a.created_at DESC, a.id DESC
        LIMIT %(limit)s
        """,
        params,
    ).fetchall()
    next_cursor: tuple[datetime, uuid.UUID] | None = None
    if len(rows) > limit:
        page = rows[:limit]
        next_cursor = (cast(datetime, page[-1][1]), cast(uuid.UUID, page[-1][0]))
    else:
        page = rows
    items: list[dict[str, Any]] = []
    for row in page:
        outcome = str(row[4])
        if outcome == "answered":
            status = "answered"
        elif outcome.startswith("abstained_") or outcome == "refused":
            status = "abstained"
        else:
            status = "error"
        items.append(
            {
                "audit_id": str(row[0]),
                "created_at": row[1],
                "question": row[2],
                "status": status,
                "language": row[3],
                "text": row[5],
                "sources": row[6] if status == "answered" else [],
            }
        )
    return items, next_cursor
