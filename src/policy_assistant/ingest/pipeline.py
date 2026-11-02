"""Idempotent corpus ingestion pipeline."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from policy_assistant.config import Settings
from policy_assistant.db.connection import connect, utc_now
from policy_assistant.db.migrate import pending_migrations
from policy_assistant.ingest.chunker import ChunkDraft, chunk_document
from policy_assistant.ingest.extract import extract_pages
from policy_assistant.ingest.fetch import sha256_file
from policy_assistant.ingest.manifest import DocumentSpec, Manifest, require_pinned
from policy_assistant.ingest.quality import (
    DocumentAssessment,
    QualityThresholds,
    assess_document,
)
from policy_assistant.ingest.tokens import TokenCounter

LOGGER = logging.getLogger(__name__)

PIPELINE_VERSION = "2"
INGEST_ADVISORY_LOCK_ID = 0x757061696E67  # "upaing"


class IngestError(RuntimeError):
    """Raised when ingestion cannot start or safely continue."""


@dataclass(frozen=True, slots=True)
class DocumentResult:
    document_id: str
    status: str
    message: str = ""
    quality: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class IngestReport:
    results: tuple[DocumentResult, ...]
    stale_documents: tuple[str, ...]
    stale_groups: tuple[str, ...]
    fingerprint: str
    report_path: Path

    @property
    def errors(self) -> int:
        return sum(1 for result in self.results if result.status == "error")

    @property
    def quarantined(self) -> int:
        return sum(1 for result in self.results if result.status.startswith("quarantined:"))


def pipeline_fingerprint(settings: Settings) -> str:
    """Hash every input that can change stored text or chunk boundaries."""
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "tokenizer": {
            "repo": settings.tokenizer_repo,
            "revision": settings.tokenizer_revision,
        },
        "chunk": {
            "max_tokens": settings.chunk_max_tokens,
            "overlap_tokens": settings.chunk_overlap_tokens,
            "min_tokens": settings.chunk_min_tokens,
        },
        "libraries": {
            "pypdfium2": importlib.metadata.version("pypdfium2"),
            "wordfreq": importlib.metadata.version("wordfreq"),
        },
        "quality": {
            "min_chars": settings.quality_min_chars,
            "max_pua_ratio": settings.quality_max_pua_ratio,
            "min_script_ratio": settings.quality_min_script_ratio,
            "min_lexicon_ur": settings.quality_min_lexicon_ur,
            "min_lexicon_en": settings.quality_min_lexicon_en,
            "reversed_margin": settings.quality_reversed_margin,
            "max_failed_page_ratio": settings.quality_max_failed_page_ratio,
            "max_empty_page_ratio": settings.quality_max_empty_page_ratio,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _quality_summary(assessment: DocumentAssessment) -> dict[str, Any]:
    return {
        "verdict": assessment.verdict,
        "reason": assessment.reason,
        "empty_ratio": assessment.empty_ratio,
        "failed_ratio": assessment.failed_ratio,
        "pages": [
            {"page_number": number, **page.as_quality()}
            for number, page in enumerate(assessment.pages, start=1)
        ],
    }


def _delete_document(conn: psycopg.Connection[Any], document_id: str) -> None:
    conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))


def _sync_acl(conn: psycopg.Connection[Any], document_id: str, group_ids: list[str]) -> None:
    conn.execute("DELETE FROM document_acl WHERE document_id = %s", (document_id,))
    with conn.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO document_acl (document_id, group_id) VALUES (%s, %s)",
            [(document_id, group_id) for group_id in group_ids],
        )


def _insert_document(
    conn: psycopg.Connection[Any],
    document: DocumentSpec,
    *,
    page_count: int,
    fingerprint: str,
    assessment: DocumentAssessment,
    drafts: list[ChunkDraft],
) -> None:
    conn.execute(
        """
        INSERT INTO documents (
            id, title, language, file_type, source_url, source_sha256,
            translation_of, license_note, page_count, pipeline_fingerprint
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            document.id,
            document.title,
            document.language,
            document.file_type,
            str(document.source_url) if document.source_url is not None else None,
            document.sha256,
            document.translation_of,
            document.license_note,
            page_count,
            fingerprint,
        ),
    )
    _sync_acl(conn, document.id, document.acl)
    with conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO pages (document_id, page_number, text, included, quality)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (document.id, number, page.text, page.included, Jsonb(page.as_quality()))
                for number, page in enumerate(assessment.pages, start=1)
            ],
        )

    with conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO chunks (
                document_id, ordinal, sentences, text, section_heading, language,
                page_start, page_end, token_count, content_sha256
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    document.id,
                    draft.ordinal,
                    list(draft.sentences),
                    draft.text,
                    draft.section_heading,
                    draft.language,
                    draft.page_start,
                    draft.page_end,
                    draft.token_count,
                    draft.content_sha256,
                )
                for draft in drafts
            ],
        )


def _write_report(
    *,
    data_dir: Path,
    results: list[DocumentResult],
    stale_documents: list[str],
    stale_groups: list[str],
    fingerprint: str,
    pruned: bool,
) -> Path:
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    generated_at = utc_now()
    stem = f"ingest-{generated_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    payload = {
        "generated_at": generated_at.isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_fingerprint": fingerprint,
        "pruned": pruned,
        "results": [
            {
                "document_id": result.document_id,
                "status": result.status,
                "message": result.message,
                "quality": result.quality,
            }
            for result in results
        ],
        "stale_documents": stale_documents,
        "stale_groups": stale_groups,
        "counts": {
            "ingested": sum(1 for result in results if result.status == "ingested"),
            "unchanged": sum(1 for result in results if result.status == "unchanged"),
            "quarantined": sum(1 for result in results if result.status.startswith("quarantined:")),
            "errors": sum(1 for result in results if result.status == "error"),
        },
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path = reports_dir / f"{stem}.json"
    suffix = 0
    while True:
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
            return path
        except FileExistsError:
            suffix += 1
            path = reports_dir / f"{stem}-{suffix}.json"


def ingest(
    manifest: Manifest,
    *,
    repo_root: Path,
    only_ids: list[str] | None,
    force: bool,
    prune: bool,
    count_tokens: TokenCounter,
    settings: Settings,
) -> IngestReport:
    """Ingest selected documents. Explicit inputs keep tests hermetic."""
    require_pinned(manifest, only_ids)
    manifest.validate_paths(repo_root)
    root = repo_root.resolve()
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else root / settings.data_dir
    fingerprint = pipeline_fingerprint(settings)
    thresholds = QualityThresholds.from_settings(settings)
    selected = manifest.selected(only_ids)
    manifest_ids = {document.id for document in manifest.documents}
    results: list[DocumentResult] = []

    with connect(settings.database_url, autocommit=True) as conn:
        pending = pending_migrations(conn)
        if pending:
            names = ", ".join(migration.version for migration in pending)
            raise IngestError(f"pending migrations must be applied first: {names}")

        locked_row = conn.execute(
            "SELECT pg_try_advisory_lock(%s)", (INGEST_ADVISORY_LOCK_ID,)
        ).fetchone()
        if locked_row is None or not locked_row[0]:
            raise IngestError("another ingest is running")
        try:
            with conn.transaction(), conn.cursor() as cursor:
                cursor.executemany(
                    """
                        INSERT INTO groups (id, description) VALUES (%s, %s)
                        ON CONFLICT (id) DO UPDATE SET description = EXCLUDED.description
                        """,
                    [(group.id, group.description) for group in manifest.groups],
                )

            for document in selected:
                path = manifest.document_path(document, root, data_dir)
                try:
                    if not path.is_file():
                        raise IngestError(f"missing file: {path}")
                    actual = sha256_file(path)
                    if document.sha256 is None:
                        raise IngestError("document has no pinned sha256")
                    if actual != document.sha256:
                        raise IngestError(
                            f"sha256 mismatch: expected {document.sha256}, found {actual}"
                        )

                    row = conn.execute(
                        """
                        SELECT source_sha256, language, file_type, pipeline_fingerprint
                        FROM documents WHERE id = %s
                        """,
                        (document.id,),
                    ).fetchone()
                    unchanged = (
                        not force
                        and row is not None
                        and row[0] == document.sha256
                        and row[1] == document.language
                        and row[2] == document.file_type
                        and row[3] == fingerprint
                    )
                    if unchanged:
                        with conn.transaction():
                            conn.execute(
                                """
                                UPDATE documents
                                SET title = %s, source_url = %s,
                                    license_note = %s, translation_of = %s
                                WHERE id = %s
                                """,
                                (
                                    document.title,
                                    str(document.source_url)
                                    if document.source_url is not None
                                    else None,
                                    document.license_note,
                                    document.translation_of,
                                    document.id,
                                ),
                            )
                            _sync_acl(conn, document.id, document.acl)
                        results.append(DocumentResult(document.id, "unchanged"))
                        continue

                    raw_pages = extract_pages(path, document.file_type)
                    assessment = assess_document(raw_pages, document.language, thresholds)
                    quality = _quality_summary(assessment)
                    if assessment.verdict == "quarantined":
                        with conn.transaction():
                            _delete_document(conn, document.id)
                        results.append(
                            DocumentResult(
                                document.id,
                                f"quarantined:{assessment.reason}",
                                assessment.reason or "",
                                quality,
                            )
                        )
                        continue

                    drafts = chunk_document(
                        assessment.included_pages(),
                        document.language,
                        count_tokens,
                        settings.chunk_max_tokens,
                        settings.chunk_overlap_tokens,
                        settings.chunk_min_tokens,
                    )
                    if not drafts:
                        with conn.transaction():
                            _delete_document(conn, document.id)
                        results.append(
                            DocumentResult(
                                document.id,
                                "quarantined:no_text",
                                "no_text",
                                quality,
                            )
                        )
                        continue

                    with conn.transaction():
                        _delete_document(conn, document.id)
                        _insert_document(
                            conn,
                            document,
                            page_count=len(assessment.pages),
                            fingerprint=fingerprint,
                            assessment=assessment,
                            drafts=drafts,
                        )
                    results.append(DocumentResult(document.id, "ingested", quality=quality))
                except Exception as exc:
                    LOGGER.exception("ingestion failed for %s", document.id)
                    results.append(DocumentResult(document.id, "error", str(exc)))

            existing_documents = {
                str(row[0]) for row in conn.execute("SELECT id FROM documents").fetchall()
            }
            existing_groups = {
                str(row[0]) for row in conn.execute("SELECT id FROM groups").fetchall()
            }
            stale_documents = sorted(existing_documents - manifest_ids)
            stale_groups = sorted(existing_groups - {group.id for group in manifest.groups})

            if prune:
                with conn.transaction():
                    if stale_documents:
                        conn.execute("DELETE FROM documents WHERE id = ANY(%s)", (stale_documents,))
                    if stale_groups:
                        conn.execute(
                            """
                            DELETE FROM groups
                            WHERE id = ANY(%s)
                              AND NOT EXISTS (
                                  SELECT 1 FROM document_acl WHERE group_id = groups.id
                              )
                            """,
                            (stale_groups,),
                        )
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (INGEST_ADVISORY_LOCK_ID,))

    report_path = _write_report(
        data_dir=data_dir,
        results=results,
        stale_documents=stale_documents,
        stale_groups=stale_groups,
        fingerprint=fingerprint,
        pruned=prune,
    )
    return IngestReport(
        results=tuple(results),
        stale_documents=tuple(stale_documents),
        stale_groups=tuple(stale_groups),
        fingerprint=fingerprint,
        report_path=report_path,
    )
