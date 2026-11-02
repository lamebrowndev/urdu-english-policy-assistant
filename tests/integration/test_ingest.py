from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from policy_assistant.config import Settings
from policy_assistant.db.connection import connect
from policy_assistant.ingest.manifest import Manifest
from policy_assistant.ingest.pipeline import (
    INGEST_ADVISORY_LOCK_ID,
    IngestError,
    IngestReport,
    ingest,
)

pytestmark = pytest.mark.integration


def run_ingest(
    manifest: Manifest,
    root: Path,
    settings: Settings,
    count_tokens: Callable[[str], int],
    *,
    force: bool = False,
    prune: bool = False,
) -> IngestReport:
    return ingest(
        manifest,
        repo_root=root,
        only_ids=None,
        force=force,
        prune=prune,
        count_tokens=count_tokens,
        settings=settings,
    )


def chunk_ids(dsn: str) -> list[int]:
    with connect(dsn) as conn:
        rows = conn.execute("SELECT id FROM chunks ORDER BY id").fetchall()
    return [int(row[0]) for row in rows]


def test_sample_ingest_and_rerun(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
) -> None:
    report = run_ingest(sample_manifest, sample_repo, settings_for, fake_counter)
    assert [result.status for result in report.results] == ["ingested"] * 3

    with connect(db_dsn) as conn:
        assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 3
        for document in sample_manifest.documents:
            acl_rows = conn.execute(
                "SELECT group_id FROM document_acl WHERE document_id = %s ORDER BY group_id",
                (document.id,),
            ).fetchall()
            assert [row[0] for row in acl_rows] == sorted(document.acl)
        without_acl = conn.execute(
            """
            SELECT count(*) FROM documents d
            WHERE NOT EXISTS (SELECT 1 FROM document_acl a WHERE a.document_id = d.id)
            """
        ).fetchone()[0]
        assert without_acl == 0

        page_counts = dict(
            conn.execute("SELECT id, page_count FROM documents ORDER BY id").fetchall()
        )
        stored_pages = dict(
            conn.execute("SELECT document_id, count(*) FROM pages GROUP BY document_id").fetchall()
        )
        assert stored_pages == page_counts

        bad_invariants = conn.execute(
            """
            SELECT count(*) FROM chunks
            WHERE text <> array_to_string(sentences, ' ')
               OR page_start > page_end
               OR token_count > %s
               OR token_count <= 0
               OR cardinality(sentences) = 0
            """,
            (settings_for.chunk_max_tokens,),
        ).fetchone()[0]
        assert bad_invariants == 0

    first_ids = chunk_ids(db_dsn)
    second = run_ingest(sample_manifest, sample_repo, settings_for, fake_counter)
    assert [result.status for result in second.results] == ["unchanged"] * 3
    assert chunk_ids(db_dsn) == first_ids


def test_acl_edit_syncs_without_rechunking(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
) -> None:
    run_ingest(sample_manifest, sample_repo, settings_for, fake_counter)
    initial_ids = chunk_ids(db_dsn)

    edits = [
        document.model_copy(update={"acl": ["all_staff", "hr"]})
        if document.id == "finance-approvals-en"
        else document
        for document in sample_manifest.documents
    ]
    edited = sample_manifest.model_copy(update={"documents": edits})
    report = run_ingest(edited, sample_repo, settings_for, fake_counter)
    assert [result.status for result in report.results] == ["unchanged"] * 3
    assert chunk_ids(db_dsn) == initial_ids

    with connect(db_dsn) as conn:
        acl_rows = conn.execute(
            "SELECT group_id FROM document_acl WHERE document_id = 'finance-approvals-en' "
            "ORDER BY group_id"
        ).fetchall()
    assert [row[0] for row in acl_rows] == ["all_staff", "hr"]

    forced = run_ingest(edited, sample_repo, settings_for, fake_counter, force=True)
    assert [result.status for result in forced.results] == ["ingested"] * 3
    assert chunk_ids(db_dsn) != initial_ids


def test_changed_chunk_setting_reingests(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
) -> None:
    run_ingest(sample_manifest, sample_repo, settings_for, fake_counter)
    initial_ids = chunk_ids(db_dsn)
    changed = settings_for.model_copy(update={"chunk_max_tokens": 300, "chunk_min_tokens": 50})
    report = run_ingest(sample_manifest, sample_repo, changed, fake_counter)
    assert [result.status for result in report.results] == ["ingested"] * 3
    assert chunk_ids(db_dsn) != initial_ids


def test_sha256_mismatch_leaves_rows_untouched(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
) -> None:
    run_ingest(sample_manifest, sample_repo, settings_for, fake_counter)
    before = chunk_ids(db_dsn)

    target = sample_repo / "corpus/samples/leave-policy-en.txt"
    target.write_text(target.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")

    report = run_ingest(sample_manifest, sample_repo, settings_for, fake_counter)
    by_id = {result.document_id: result for result in report.results}
    assert by_id["leave-policy-en"].status == "error"
    assert by_id["leave-policy-ur"].status == "unchanged"
    assert chunk_ids(db_dsn) == before

    with connect(db_dsn) as conn:
        row = conn.execute(
            "SELECT source_sha256 FROM documents WHERE id = 'leave-policy-en'"
        ).fetchone()
    assert row is not None
    assert row[0] == sample_manifest.selected(["leave-policy-en"])[0].sha256


def test_prune_removes_documents_and_groups(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
) -> None:
    run_ingest(sample_manifest, sample_repo, settings_for, fake_counter)
    reduced = sample_manifest.model_copy(
        update={
            "documents": [
                document
                for document in sample_manifest.documents
                if document.id != "finance-approvals-en"
            ],
            "groups": [group for group in sample_manifest.groups if group.id == "all_staff"],
        }
    )

    report = run_ingest(reduced, sample_repo, settings_for, fake_counter)
    assert report.stale_documents == ("finance-approvals-en",)
    assert report.stale_groups == ("finance", "hr")
    with connect(db_dsn) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM documents WHERE id = 'finance-approvals-en'"
            ).fetchone()[0]
            == 1
        )

    pruned = run_ingest(reduced, sample_repo, settings_for, fake_counter, prune=True)
    assert pruned.stale_documents == ("finance-approvals-en",)
    with connect(db_dsn) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM documents WHERE id = 'finance-approvals-en'"
            ).fetchone()[0]
            == 0
        )
        assert conn.execute("SELECT count(*) FROM groups").fetchone()[0] == 1
        assert (
            conn.execute(
                """
            SELECT count(*) FROM pages p
            WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = p.document_id)
            """
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                """
            SELECT count(*) FROM chunks c
            WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = c.document_id)
            """
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                """
            SELECT count(*) FROM document_acl a
            WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = a.document_id)
            """
            ).fetchone()[0]
            == 0
        )


def test_advisory_lock_blocks_concurrent_ingest(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
) -> None:
    with connect(db_dsn, autocommit=True) as blocker:
        blocker.execute("SELECT pg_advisory_lock(%s)", (INGEST_ADVISORY_LOCK_ID,))
        try:
            with pytest.raises(IngestError, match="another ingest is running"):
                run_ingest(sample_manifest, sample_repo, settings_for, fake_counter)
        finally:
            blocker.execute("SELECT pg_advisory_unlock(%s)", (INGEST_ADVISORY_LOCK_ID,))
