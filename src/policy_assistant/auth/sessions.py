"""Opaque server-side sessions."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import psycopg

from policy_assistant.db.connection import utc_now


@dataclass(frozen=True, slots=True)
class SessionUser:
    id: int
    username: str
    display_name: str
    role: str
    is_active: bool
    groups: frozenset[str]
    csrf_token: str


def _hash_token(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def create_session(conn: psycopg.Connection[Any], user_id: int, ttl: int) -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    expires_at = utc_now() + timedelta(hours=ttl)
    with conn.transaction():
        conn.execute("DELETE FROM sessions WHERE expires_at <= now()")
        conn.execute(
            """
            INSERT INTO sessions (token_hash, user_id, csrf_token, expires_at)
            VALUES (%s, %s, %s, %s)
            """,
            (_hash_token(token), user_id, csrf_token, expires_at),
        )
    return token, csrf_token


def load_session(conn: psycopg.Connection[Any], token: str) -> SessionUser | None:
    row = conn.execute(
        """
        SELECT u.id, u.username, u.display_name, u.role, u.is_active, s.csrf_token
        FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.token_hash = %s
          AND s.expires_at > now()
          AND u.is_active
        """,
        (_hash_token(token),),
    ).fetchone()
    if row is None:
        return None
    groups = conn.execute(
        "SELECT group_id FROM user_groups WHERE user_id = %s ORDER BY group_id",
        (int(row[0]),),
    ).fetchall()
    return SessionUser(
        id=int(row[0]),
        username=str(row[1]),
        display_name=str(row[2]),
        role=str(row[3]),
        is_active=bool(row[4]),
        groups=frozenset(str(group[0]) for group in groups),
        csrf_token=str(row[5]),
    )


def delete_session(conn: psycopg.Connection[Any], token: str) -> None:
    with conn.transaction():
        conn.execute("DELETE FROM sessions WHERE token_hash = %s", (_hash_token(token),))
