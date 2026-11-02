from __future__ import annotations

import pytest

from policy_assistant.db.connection import connect
from policy_assistant.db.migrate import (
    MigrationError,
    applied_checksums,
    migrate,
    pending_migrations,
)

pytestmark = pytest.mark.integration


def test_migrations_are_idempotent_and_pgvector_is_new_enough(db_dsn: str) -> None:
    with connect(db_dsn) as conn:
        assert pending_migrations(conn) == []
        assert migrate(conn) == []
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        assert row is not None
        assert tuple(int(part) for part in str(row[0]).split(".")) >= (0, 8, 0)


def test_tampered_checksum_is_detected(db_dsn: str) -> None:
    with connect(db_dsn) as conn:
        assert "0001_init.sql" in applied_checksums(conn)
        conn.execute(
            "UPDATE schema_migrations SET checksum = 'tampered' WHERE version = '0001_init.sql'"
        )
        conn.commit()
        with pytest.raises(MigrationError, match=r"0001_init\.sql"):
            migrate(conn)
