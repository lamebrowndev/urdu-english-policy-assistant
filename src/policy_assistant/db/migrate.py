"""Plain-SQL migrations with checksums and a session-level advisory lock."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

import psycopg

MIGRATION_FILENAME_RE = re.compile(r"^\d{4}_[a-z0-9_]+\.sql$")
MIGRATION_ADVISORY_LOCK_ID = 0x7570616D6967  # "upamig"
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version text PRIMARY KEY,
    checksum text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


class MigrationError(RuntimeError):
    """Raised when migrations cannot be applied safely."""


@dataclass(frozen=True, slots=True)
class Migration:
    version: str
    sql: str
    checksum: str


def _migrations_dir() -> Any:
    return resources.files("policy_assistant.db").joinpath("migrations")


def load_migrations() -> list[Migration]:
    """Load packaged migrations, rejecting filenames outside the convention."""
    directory = _migrations_dir()
    migrations: list[Migration] = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        if not MIGRATION_FILENAME_RE.fullmatch(entry.name):
            raise MigrationError(f"invalid migration filename: {entry.name}")
        payload = entry.read_bytes()
        migrations.append(
            Migration(
                version=entry.name,
                sql=payload.decode("utf-8"),
                checksum=hashlib.sha256(payload).hexdigest(),
            )
        )
    migrations.sort(key=lambda migration: migration.version)
    return migrations


def _table_exists(conn: psycopg.Connection[Any]) -> bool:
    row = conn.execute("SELECT to_regclass('schema_migrations')").fetchone()
    return row is not None and row[0] is not None


def applied_checksums(conn: psycopg.Connection[Any]) -> dict[str, str]:
    """Return {version: checksum} for migrations recorded by this database."""
    if not _table_exists(conn):
        return {}
    rows = conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def pending_migrations(conn: psycopg.Connection[Any]) -> list[Migration]:
    """Return packaged migrations that the database has not applied."""
    applied = applied_checksums(conn)
    return [migration for migration in load_migrations() if migration.version not in applied]


def _verify_applied_checksums(conn: psycopg.Connection[Any], migrations: list[Migration]) -> None:
    applied = applied_checksums(conn)
    current = {migration.version: migration for migration in migrations}
    for version, checksum in applied.items():
        migration = current.get(version)
        if migration is None:
            raise MigrationError(f"applied migration has no file: {version}")
        if migration.checksum != checksum:
            raise MigrationError(
                f"checksum mismatch for {version}: applied {checksum}, file {migration.checksum}"
            )


def migrate(conn: psycopg.Connection[Any]) -> list[str]:
    """Apply pending migrations. Returns the versions applied by this call."""
    migrations = load_migrations()
    applied_now: list[str] = []
    conn.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_ADVISORY_LOCK_ID,))
    conn.commit()
    try:
        conn.execute(_CREATE_TABLE)
        conn.commit()
        _verify_applied_checksums(conn, migrations)
        recorded = applied_checksums(conn)
        for migration in migrations:
            if migration.version in recorded:
                continue
            with conn.transaction():
                conn.execute(migration.sql)
                conn.execute(
                    "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
                    (migration.version, migration.checksum),
                )
            applied_now.append(migration.version)
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_ADVISORY_LOCK_ID,))
        conn.commit()
    return applied_now
