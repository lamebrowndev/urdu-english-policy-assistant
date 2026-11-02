"""User accounts and group membership."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import psycopg

from policy_assistant.auth.passwords import hash_password

ROLES = frozenset({"user", "reviewer", "admin"})
_USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,62}$")


@dataclass(frozen=True, slots=True)
class User:
    id: int
    username: str
    display_name: str
    role: str
    is_active: bool
    groups: frozenset[str]


def normalize_username(username: str) -> str:
    return username.strip().lower()


def _validate_role(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"unknown role: {role}")
    return role


def _check_groups(
    conn: psycopg.Connection[Any], groups: list[str] | tuple[str, ...] | set[str]
) -> list[str]:
    requested = sorted(set(groups))
    if not requested:
        return []
    rows = conn.execute("SELECT id FROM groups WHERE id = ANY(%s)", (requested,)).fetchall()
    known = {str(row[0]) for row in rows}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"unknown group ids: {unknown}")
    return requested


def _groups_for(conn: psycopg.Connection[Any], user_id: int) -> frozenset[str]:
    rows = conn.execute(
        "SELECT group_id FROM user_groups WHERE user_id = %s ORDER BY group_id",
        (user_id,),
    ).fetchall()
    return frozenset(str(row[0]) for row in rows)


def create_user(
    conn: psycopg.Connection[Any],
    username: str,
    display_name: str,
    password: str,
    role: str,
    groups: list[str] | tuple[str, ...] | set[str],
) -> User:
    normalized = normalize_username(username)
    if not normalized:
        raise ValueError("username is required")
    if _USERNAME_RE.fullmatch(normalized) is None:
        raise ValueError(f"invalid username: {normalized}")
    _validate_role(role)
    if not 1 <= len(display_name) <= 100:
        raise ValueError("display_name must be between 1 and 100 characters")
    requested_groups = _check_groups(conn, groups)
    password_hash = hash_password(password)
    try:
        with conn.transaction():
            row = conn.execute(
                """
                INSERT INTO users (username, display_name, password_hash, role)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (normalized, display_name, password_hash, role),
            ).fetchone()
            if row is None:
                raise RuntimeError("user insert returned no id")
            user_id = int(row[0])
            if requested_groups:
                with conn.cursor() as cursor:
                    cursor.executemany(
                        "INSERT INTO user_groups (user_id, group_id) VALUES (%s, %s)",
                        [(user_id, group_id) for group_id in requested_groups],
                    )
    except psycopg.errors.UniqueViolation as exc:
        raise ValueError(f"duplicate username: {normalized}") from exc
    user = get_user(conn, normalized)
    if user is None:
        raise RuntimeError("created user could not be loaded")
    return user


def set_user_groups(
    conn: psycopg.Connection[Any], username: str, groups: list[str] | tuple[str, ...] | set[str]
) -> None:
    normalized = normalize_username(username)
    requested_groups = _check_groups(conn, groups)
    with conn.transaction():
        row = conn.execute("SELECT id FROM users WHERE username = %s", (normalized,)).fetchone()
        if row is None:
            raise ValueError(f"unknown user: {normalized}")
        user_id = int(row[0])
        conn.execute("DELETE FROM user_groups WHERE user_id = %s", (user_id,))
        if requested_groups:
            with conn.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO user_groups (user_id, group_id) VALUES (%s, %s)",
                    [(user_id, group_id) for group_id in requested_groups],
                )


def set_user_role(conn: psycopg.Connection[Any], username: str, role: str) -> None:
    normalized = normalize_username(username)
    _validate_role(role)
    with conn.transaction():
        row = conn.execute(
            "UPDATE users SET role = %s WHERE username = %s RETURNING id",
            (role, normalized),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown user: {normalized}")


def set_user_active(conn: psycopg.Connection[Any], username: str, active: bool) -> None:
    normalized = normalize_username(username)
    with conn.transaction():
        row = conn.execute(
            "UPDATE users SET is_active = %s WHERE username = %s RETURNING id",
            (active, normalized),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown user: {normalized}")
        if not active:
            conn.execute("DELETE FROM sessions WHERE user_id = %s", (int(row[0]),))


def get_user(conn: psycopg.Connection[Any], username: str) -> User | None:
    normalized = normalize_username(username)
    row = conn.execute(
        """
        SELECT id, username, display_name, role, is_active
        FROM users WHERE username = %s
        """,
        (normalized,),
    ).fetchone()
    if row is None:
        return None
    return User(
        id=int(row[0]),
        username=str(row[1]),
        display_name=str(row[2]),
        role=str(row[3]),
        is_active=bool(row[4]),
        groups=_groups_for(conn, int(row[0])),
    )


def list_users(conn: psycopg.Connection[Any]) -> list[User]:
    rows = conn.execute(
        """
        SELECT u.id, u.username, u.display_name, u.role, u.is_active,
               COALESCE(
                   array_agg(ug.group_id ORDER BY ug.group_id)
                       FILTER (WHERE ug.group_id IS NOT NULL),
                   ARRAY[]::text[]
               ) AS groups
        FROM users u
        LEFT JOIN user_groups ug ON ug.user_id = u.id
        GROUP BY u.id
        ORDER BY u.username
        """
    ).fetchall()
    return [
        User(
            id=int(row[0]),
            username=str(row[1]),
            display_name=str(row[2]),
            role=str(row[3]),
            is_active=bool(row[4]),
            groups=frozenset(str(group) for group in row[5]),
        )
        for row in rows
    ]
