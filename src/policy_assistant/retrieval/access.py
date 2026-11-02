"""Default-deny access checks for retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg


@dataclass(frozen=True, slots=True)
class Principal:
    groups: frozenset[str]


class AccessViolation(Exception):
    """Raised when a chunk's document is outside the caller's ACL."""


def allowed_document_ids(conn: psycopg.Connection[Any], principal: Principal) -> list[str]:
    """Return documents visible to any of the principal's groups."""
    if not principal.groups:
        return []
    rows = conn.execute(
        """
        SELECT DISTINCT document_id
        FROM document_acl
        WHERE group_id = ANY(%s)
        ORDER BY document_id
        """,
        (sorted(principal.groups),),
    ).fetchall()
    return [str(row[0]) for row in rows]


def ensure_authorized(document_ids: list[str], allowed: list[str]) -> None:
    """Raise if any document id is outside ``allowed``."""
    allowed_set = set(allowed)
    unauthorized = sorted({document_id for document_id in document_ids} - allowed_set)
    if unauthorized:
        raise AccessViolation(f"unauthorized documents: {unauthorized}")
