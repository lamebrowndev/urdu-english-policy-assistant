from __future__ import annotations

import psycopg
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
        assert "0003_answer_audit.sql" in applied_checksums(conn)
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        assert row is not None
        assert tuple(int(part) for part in str(row[0]).split(".")) >= (0, 8, 0)
        index_row = conn.execute(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename = 'chunks' AND indexname = 'chunks_embedding_hnsw_idx'"
        ).fetchone()
        assert index_row is not None


def test_tampered_checksum_is_detected(db_dsn: str) -> None:
    with connect(db_dsn) as conn:
        assert "0001_init.sql" in applied_checksums(conn)
        conn.execute(
            "UPDATE schema_migrations SET checksum = 'tampered' WHERE version = '0001_init.sql'"
        )
        conn.commit()
        with pytest.raises(MigrationError, match=r"0001_init\.sql"):
            migrate(conn)


def test_answer_audit_is_append_only(db_dsn: str) -> None:
    with connect(db_dsn, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO answer_audit (
                actor, principal_groups, question, question_language, outcome,
                answer_text, retrieval, prompt_version, pipeline_fingerprint,
                latency_ms
            ) VALUES (
                'test', ARRAY['all_staff'], 'question', 'en', 'error',
                'text', '{}'::jsonb, 'answer-v1', 'fingerprint', '{}'::jsonb
            )
            """
        )
        with pytest.raises(psycopg.Error, match="append-only"):
            conn.execute("UPDATE answer_audit SET actor = 'other'")
        with pytest.raises(psycopg.Error, match="append-only"):
            conn.execute("DELETE FROM answer_audit")
        with pytest.raises(psycopg.Error, match="append-only"):
            # Every table referencing answer_audit must be named, or Postgres's
            # foreign-key check fires before the append-only trigger.
            conn.execute("TRUNCATE answer_reviews, eval_results, answer_audit")


def test_answer_reviews_is_append_only(db_dsn: str) -> None:
    with connect(db_dsn, autocommit=True) as conn:
        user_id = conn.execute(
            """
            INSERT INTO users (username, display_name, password_hash, role)
            VALUES ('reviewer1', 'Reviewer', 'x', 'reviewer')
            RETURNING id
            """
        ).fetchone()[0]
        audit_id = conn.execute(
            """
            INSERT INTO answer_audit (
                actor, principal_groups, question, question_language, outcome,
                answer_text, retrieval, prompt_version, pipeline_fingerprint,
                latency_ms
            ) VALUES (
                'user:test', ARRAY['all_staff'], 'question', 'en', 'answered',
                'text', '{}'::jsonb, 'answer-v1', 'fingerprint', '{}'::jsonb
            )
            RETURNING id
            """
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO answer_reviews (
                audit_id, reviewer_id, reviewer_username, verdict, citation_ok
            ) VALUES (%s, %s, 'reviewer1', 'correct', true)
            """,
            (audit_id, user_id),
        )
        with pytest.raises(psycopg.Error, match="answer_reviews is append-only"):
            conn.execute("UPDATE answer_reviews SET notes = 'changed'")
        with pytest.raises(psycopg.Error, match="answer_reviews is append-only"):
            conn.execute("DELETE FROM answer_reviews")
        with pytest.raises(psycopg.Error, match="answer_reviews is append-only"):
            conn.execute("TRUNCATE answer_reviews")
