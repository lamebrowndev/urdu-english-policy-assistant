"""Command-line interface (``upa``)."""

from __future__ import annotations

import json
import logging
import re
from contextlib import closing
from pathlib import Path
from typing import Annotated, Any

import psycopg
import typer

from policy_assistant.config import Settings, find_repo_root, get_settings
from policy_assistant.db.connection import connect, utc_now
from policy_assistant.db.migrate import applied_checksums, migrate, pending_migrations
from policy_assistant.ingest.extract import ExtractionError, extract_pages
from policy_assistant.ingest.fetch import fetch_documents
from policy_assistant.ingest.manifest import DocumentSpec, Manifest, ManifestError
from policy_assistant.ingest.pipeline import (
    IngestError,
    IngestReport,
)
from policy_assistant.ingest.pipeline import (
    ingest as run_ingest,
)
from policy_assistant.ingest.quality import (
    DocumentAssessment,
    QualityThresholds,
    assess_document,
)
from policy_assistant.ingest.tokens import load_bge_m3_counter

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Urdu/English policy assistant.")
db_app = typer.Typer(no_args_is_help=True, add_completion=False, help="Database commands.")
corpus_app = typer.Typer(no_args_is_help=True, add_completion=False, help="Corpus commands.")
app.add_typer(db_app, name="db")
app.add_typer(corpus_app, name="corpus")


def _configure_logging() -> None:
    settings = get_settings()
    level_name = settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", version)
    return tuple(int(part) for part in parts)


def _resolve_manifest_path(option: Path | None, settings: Settings) -> Path:
    if option is not None:
        if option.is_absolute():
            return option
        candidate = Path.cwd() / option
        if candidate.exists():
            return candidate.resolve()
        return (find_repo_root() / option).resolve()
    return settings.manifest_path


def _load_manifest(option: Path | None) -> tuple[Manifest, Path, Settings]:
    settings = get_settings()
    manifest_path = _resolve_manifest_path(option, settings)
    manifest = Manifest.load(manifest_path, repo_root=find_repo_root())
    return manifest, manifest_path, settings


def _select(manifest: Manifest, document_ids: list[str] | None) -> list[DocumentSpec]:
    return manifest.selected(document_ids)


@db_app.command("migrate")
def db_migrate() -> None:
    """Apply pending SQL migrations."""
    _configure_logging()
    settings = get_settings()
    try:
        with closing(connect(settings.database_url)) as conn:
            applied = migrate(conn)
    except (psycopg.Error, RuntimeError) as exc:
        typer.secho(f"migration failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    if applied:
        for version in applied:
            typer.echo(f"applied {version}")
    else:
        typer.echo("No pending migrations.")


@db_app.command("status")
def db_status() -> None:
    """Print server, pgvector, and migration status. Exit 1 when unhealthy."""
    _configure_logging()
    settings = get_settings()
    try:
        with closing(connect(settings.database_url)) as conn:
            server_row = conn.execute("SELECT version()").fetchone()
            server_version = str(server_row[0]) if server_row else "unknown"
            extension_row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            vector_version = str(extension_row[0]) if extension_row else None
            applied = applied_checksums(conn)
            pending = pending_migrations(conn)
    except psycopg.Error as exc:
        typer.secho(f"database unavailable: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(f"server: {server_version}")
    typer.echo(f"pgvector: {vector_version or 'not installed'}")
    typer.echo(f"applied migrations: {len(applied)}")
    for version in sorted(applied):
        typer.echo(f"  applied {version}")
    typer.echo(f"pending migrations: {len(pending)}")
    for migration in pending:
        typer.echo(f"  pending {migration.version}")

    healthy = vector_version is not None and _version_tuple(vector_version) >= (0, 8, 0)
    if not healthy:
        typer.secho("pgvector >= 0.8.0 is required", err=True, fg=typer.colors.RED)
    if pending:
        typer.secho("pending migrations must be applied", err=True, fg=typer.colors.RED)
    if not healthy or pending:
        raise typer.Exit(code=1)


@corpus_app.command("fetch")
def corpus_fetch(
    manifest_option: Annotated[
        Path | None, typer.Option("--manifest", help="Manifest path.")
    ] = None,
    document_ids: Annotated[
        list[str] | None, typer.Option("--doc", help="Document id (repeatable).")
    ] = None,
) -> None:
    """Download missing sources and verify pinned hashes."""
    _configure_logging()
    try:
        manifest, _, settings = _load_manifest(manifest_option)
    except ManifestError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    report = fetch_documents(
        manifest,
        repo_root=find_repo_root(),
        data_dir=settings.data_dir,
        only_ids=document_ids,
    )
    for result in report.results:
        if result.status == "ok":
            typer.echo(f"OK {result.document_id}")
        elif result.status == "unpinned":
            typer.echo(f"UNPINNED {result.document_id} {result.sha256}")
        else:
            typer.secho(
                f"MISMATCH {result.document_id}: {result.message}", err=True, fg=typer.colors.RED
            )
    if report.errors:
        raise typer.Exit(code=1)


def _assess_documents(
    manifest: Manifest,
    settings: Settings,
    document_ids: list[str] | None,
) -> tuple[list[tuple[Any, list[str], DocumentAssessment]], list[tuple[str, str]]]:
    thresholds = QualityThresholds.from_settings(settings)
    root = find_repo_root()
    assessed: list[tuple[Any, list[str], DocumentAssessment]] = []
    errors: list[tuple[str, str]] = []
    for document in _select(manifest, document_ids):
        path = manifest.document_path(document, root, settings.data_dir)
        try:
            raw_pages = extract_pages(path, document.file_type)
        except (OSError, ExtractionError) as exc:
            errors.append((document.id, str(exc)))
            continue
        assessment = assess_document(raw_pages, document.language, thresholds)
        assessed.append((document, raw_pages, assessment))
    return assessed, errors


@corpus_app.command("validate")
def corpus_validate(
    manifest_option: Annotated[
        Path | None, typer.Option("--manifest", help="Manifest path.")
    ] = None,
    document_ids: Annotated[
        list[str] | None, typer.Option("--doc", help="Document id (repeatable).")
    ] = None,
) -> None:
    """Validate the manifest, extract every source, and summarize quality."""
    _configure_logging()
    try:
        manifest, _, settings = _load_manifest(manifest_option)
    except ManifestError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    assessed, errors = _assess_documents(manifest, settings, document_ids)
    quarantined = 0
    for document, raw_pages, assessment in assessed:
        typer.echo(
            f"{document.id}: {assessment.verdict}"
            + (f" ({assessment.reason})" if assessment.reason else "")
            + f", pages={len(raw_pages)}, included={len(assessment.included_pages())}"
        )
        if assessment.verdict == "quarantined":
            quarantined += 1
    for document_id, message in errors:
        typer.secho(f"{document_id}: error: {message}", err=True, fg=typer.colors.RED)
    if errors:
        raise typer.Exit(code=1)
    if quarantined:
        raise typer.Exit(code=2)


def _dump_page(path: Path, page_number: int, text: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / f"page-{page_number}.txt").write_text(text, encoding="utf-8")


@corpus_app.command("inspect")
def corpus_inspect(
    manifest_option: Annotated[
        Path | None, typer.Option("--manifest", help="Manifest path.")
    ] = None,
    document_ids: Annotated[
        list[str] | None, typer.Option("--doc", help="Document id (repeatable).")
    ] = None,
    dump_text: Annotated[
        bool, typer.Option("--dump-text", help="Write cleaned pages for review.")
    ] = False,
) -> None:
    """Extract and assess the corpus without touching the database."""
    _configure_logging()
    try:
        manifest, manifest_path, settings = _load_manifest(manifest_option)
    except ManifestError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    assessed, errors = _assess_documents(manifest, settings, document_ids)
    generated_at = utc_now()
    timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    reports_dir = settings.data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"inspect-{timestamp}.json"
    dump_dir = reports_dir / f"inspect-{timestamp}"

    report_documents: list[dict[str, Any]] = []
    quarantined = 0
    for document, raw_pages, assessment in assessed:
        typer.echo(f"{document.id}: {assessment.verdict}")
        typer.echo(f"  language: {document.language}")
        typer.echo(f"  pages: {len(raw_pages)} total, {len(assessment.included_pages())} included")
        typer.echo(
            f"  empty_ratio: {assessment.empty_ratio:.3f}, "
            f"failed_ratio: {assessment.failed_ratio:.3f}"
        )
        if assessment.reason:
            typer.secho(f"  reason: {assessment.reason}", fg=typer.colors.YELLOW)
            quarantined += 1
        for number, page in enumerate(assessment.pages, start=1):
            if page.flags:
                typer.echo(
                    f"  page {number}: {', '.join(page.flags)} "
                    f"(n_chars={page.n_chars}, pua={page.pua_ratio:.3f}, "
                    f"script={page.script_ratio}, lexicon={page.lexicon_hit_rate}, "
                    f"reversed={page.reversed_hit_rate})"
                )
        report_documents.append(
            {
                "id": document.id,
                "title": document.title,
                "language": document.language,
                "file_type": document.file_type,
                "path": str(manifest.document_path(document, find_repo_root(), settings.data_dir)),
                "verdict": assessment.verdict,
                "reason": assessment.reason,
                "page_count": len(raw_pages),
                "included_pages": len(assessment.included_pages()),
                "empty_ratio": assessment.empty_ratio,
                "failed_ratio": assessment.failed_ratio,
                "pages": [
                    {"page_number": number, **page.as_quality()}
                    for number, page in enumerate(assessment.pages, start=1)
                ],
            }
        )
        if dump_text:
            document_dir = dump_dir / document.id
            included_seen = 0
            for number, page in enumerate(assessment.pages, start=1):
                if page.flags:
                    _dump_page(document_dir, number, page.text)
                elif included_seen < 2:
                    _dump_page(document_dir, number, page.text)
                    included_seen += 1

    report = {
        "generated_at": generated_at.isoformat(),
        "manifest": str(manifest_path),
        "documents": report_documents,
        "errors": [{"id": document_id, "message": message} for document_id, message in errors],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(f"report: {report_path}")
    for document_id, message in errors:
        typer.secho(f"{document_id}: error: {message}", err=True, fg=typer.colors.RED)
    if errors or quarantined:
        raise typer.Exit(code=2)


def _print_ingest_report(report: IngestReport) -> None:
    width = max((len(result.document_id) for result in report.results), default=10)
    typer.echo(f"{'document'.ljust(width)}  status")
    for result in report.results:
        suffix = f"  {result.message}" if result.message and result.status == "error" else ""
        typer.echo(f"{result.document_id.ljust(width)}  {result.status}{suffix}")
    if report.stale_documents:
        typer.echo(f"stale documents: {', '.join(report.stale_documents)}")
    if report.stale_groups:
        typer.echo(f"stale groups: {', '.join(report.stale_groups)}")
    typer.echo(f"report: {report.report_path}")


@app.command("ingest")
def ingest_command(
    manifest_option: Annotated[
        Path | None, typer.Option("--manifest", help="Manifest path.")
    ] = None,
    document_ids: Annotated[
        list[str] | None, typer.Option("--doc", help="Document id (repeatable).")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-ingest unchanged documents.")] = False,
    prune: Annotated[
        bool, typer.Option("--prune", help="Delete documents and groups absent from the manifest.")
    ] = False,
) -> None:
    """Extract, quality-gate, chunk, and store selected documents."""
    _configure_logging()
    try:
        manifest, _, settings = _load_manifest(manifest_option)
        count_tokens = load_bge_m3_counter(settings.tokenizer_repo, settings.tokenizer_revision)
        report = run_ingest(
            manifest,
            repo_root=find_repo_root(),
            only_ids=document_ids,
            force=force,
            prune=prune,
            count_tokens=count_tokens,
            settings=settings,
        )
    except (ManifestError, IngestError, psycopg.Error) as exc:
        typer.secho(f"ingest failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    _print_ingest_report(report)
    if report.errors:
        raise typer.Exit(code=1)
    if report.quarantined:
        raise typer.Exit(code=2)


@app.command("stats")
def stats_command() -> None:
    """Print corpus counts and chunk token percentiles."""
    _configure_logging()
    settings = get_settings()
    try:
        with closing(connect(settings.database_url)) as conn:
            documents_by_language = conn.execute(
                "SELECT language, count(*) FROM documents GROUP BY language ORDER BY language"
            ).fetchall()
            pages_by_inclusion = conn.execute(
                "SELECT included, count(*) FROM pages GROUP BY included ORDER BY included"
            ).fetchall()
            chunks_by_language = conn.execute(
                "SELECT language, count(*) FROM chunks GROUP BY language ORDER BY language"
            ).fetchall()
            token_row = conn.execute(
                """
                SELECT
                    percentile_cont(0.5) WITHIN GROUP (ORDER BY token_count),
                    percentile_cont(0.95) WITHIN GROUP (ORDER BY token_count),
                    max(token_count)
                FROM chunks
                """
            ).fetchone()
            zero_chunk_documents = conn.execute(
                """
                SELECT count(*) FROM documents d
                WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)
                """
            ).fetchone()
    except psycopg.Error as exc:
        typer.secho(f"database unavailable: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo("documents per language:")
    for language, count in documents_by_language:
        typer.echo(f"  {language}: {count}")
    typer.echo("pages included/excluded:")
    for included, count in pages_by_inclusion:
        typer.echo(f"  {'included' if included else 'excluded'}: {count}")
    typer.echo("chunks per language:")
    for language, count in chunks_by_language:
        typer.echo(f"  {language}: {count}")
    if token_row is not None and token_row[0] is not None:
        p50, p95, maximum = token_row
        typer.echo(f"chunk token_count p50/p95/max: {p50:.1f}/{p95:.1f}/{maximum}")
    else:
        typer.echo("chunk token_count p50/p95/max: no chunks")
    typer.echo(
        f"documents with zero chunks: {zero_chunk_documents[0] if zero_chunk_documents else 0}"
    )


if __name__ == "__main__":
    app()
