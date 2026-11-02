"""Command-line interface (``upa``)."""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import re
import sys
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Annotated, Any, cast, get_args

import psycopg
import typer

from policy_assistant.answering.audit_store import load_audit_row
from policy_assistant.answering.model import build_answer_model
from policy_assistant.answering.service import AnswerOutcome, AnswerResult
from policy_assistant.answering.service import answer as run_answer
from policy_assistant.auth.users import (
    create_user,
    list_users,
    set_user_active,
    set_user_groups,
    set_user_role,
)
from policy_assistant.config import Settings, find_repo_root, get_settings
from policy_assistant.db.connection import connect, utc_now
from policy_assistant.db.migrate import applied_checksums, migrate, pending_migrations
from policy_assistant.ingest.extract import ExtractionError, extract_pages
from policy_assistant.ingest.fetch import fetch_documents, sha256_file
from policy_assistant.ingest.manifest import DocumentSpec, Manifest, ManifestError
from policy_assistant.ingest.pipeline import (
    IngestError,
    IngestReport,
    pipeline_fingerprint,
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
from policy_assistant.retrieval.access import AccessViolation, Principal
from policy_assistant.retrieval.evaluation import (
    EvaluationError,
    evaluate,
    load_dataset,
)
from policy_assistant.retrieval.models import (
    CrossEncoderReranker,
    SentenceTransformerEmbedder,
)
from policy_assistant.retrieval.readiness import index_problem
from policy_assistant.retrieval.retriever import (
    RetrievalMode,
    RetrievalResult,
    retrieve,
    with_top_k,
)

LOGGER = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Urdu/English policy assistant.")
db_app = typer.Typer(no_args_is_help=True, add_completion=False, help="Database commands.")
corpus_app = typer.Typer(no_args_is_help=True, add_completion=False, help="Corpus commands.")
eval_app = typer.Typer(no_args_is_help=True, add_completion=False, help="Evaluation commands.")
audit_app = typer.Typer(no_args_is_help=True, add_completion=False, help="Audit log commands.")
user_app = typer.Typer(no_args_is_help=True, add_completion=False, help="User account commands.")
app.add_typer(db_app, name="db")
app.add_typer(corpus_app, name="corpus")
app.add_typer(eval_app, name="eval")
app.add_typer(audit_app, name="audit")
app.add_typer(user_app, name="user")


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
    """Validate the manifest and source files without extraction.

    Prints OK, MISSING, MISMATCH, or UNPINNED per document. Exit 1 on any
    MISSING or MISMATCH, else 0.
    """
    _configure_logging()
    try:
        manifest, _, settings = _load_manifest(manifest_option)
        documents = manifest.selected(document_ids)
    except ManifestError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    root = find_repo_root()
    failed = False
    for document in documents:
        path = manifest.document_path(document, root, settings.data_dir)
        status: str
        try:
            if not path.is_file():
                status = "MISSING"
            elif document.sha256 is None:
                status = "UNPINNED"
            elif sha256_file(path) == document.sha256:
                status = "OK"
            else:
                status = "MISMATCH"
        except OSError:
            status = "MISSING"
        typer.echo(f"{status} {document.id}")
        if status in {"MISSING", "MISMATCH"}:
            failed = True
    if failed:
        raise typer.Exit(code=1)


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
    """Extract and assess the corpus without touching the database.

    Exit 1 on any error, 2 if a document is quarantined, else 0.
    """
    _configure_logging()
    try:
        manifest, manifest_path, settings = _load_manifest(manifest_option)
    except ManifestError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    assessed, errors = _assess_documents(manifest, settings, document_ids)
    generated_at = utc_now()
    stem = f"inspect-{generated_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    reports_dir = settings.data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    dumped_pages: list[tuple[str, int, str]] = []

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
            included_seen = 0
            for number, page in enumerate(assessment.pages, start=1):
                if page.flags:
                    dumped_pages.append((document.id, number, page.text))
                elif included_seen < 2:
                    dumped_pages.append((document.id, number, page.text))
                    included_seen += 1

    report = {
        "generated_at": generated_at.isoformat(),
        "manifest": str(manifest_path),
        "documents": report_documents,
        "errors": [{"id": document_id, "message": message} for document_id, message in errors],
    }
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    suffix = 0
    while True:
        candidate_stem = stem if suffix == 0 else f"{stem}-{suffix}"
        report_path = reports_dir / f"{candidate_stem}.json"
        dump_dir = reports_dir / candidate_stem
        if dump_text and dump_dir.exists():
            suffix += 1
            continue
        try:
            with report_path.open("x", encoding="utf-8") as handle:
                handle.write(content)
            break
        except FileExistsError:
            suffix += 1
    typer.echo(f"report: {report_path}")
    if dump_text:
        for document_id, number, text in dumped_pages:
            _dump_page(dump_dir / document_id, number, text)
    for document_id, message in errors:
        typer.secho(f"{document_id}: error: {message}", err=True, fg=typer.colors.RED)
    if errors:
        raise typer.Exit(code=1)
    if quarantined:
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
    """Extract, quality-gate, chunk, and store selected documents.

    Exit 1 on any error, 2 if a document is quarantined, else 0.
    """
    _configure_logging()
    try:
        manifest, _, settings = _load_manifest(manifest_option)
        count_tokens = load_bge_m3_counter(settings.tokenizer_repo, settings.tokenizer_revision)
        embedder = SentenceTransformerEmbedder(settings)
        LOGGER.info("embedding device: %s", embedder.device)
        started = time.perf_counter()
        report = run_ingest(
            manifest,
            repo_root=find_repo_root(),
            only_ids=document_ids,
            force=force,
            prune=prune,
            count_tokens=count_tokens,
            embedder=embedder,
            settings=settings,
        )
        LOGGER.info("ingest finished in %.2fs", time.perf_counter() - started)
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
            missing_embeddings = conn.execute(
                "SELECT count(*) FROM chunks WHERE embedding IS NULL"
            ).fetchone()
            missing_bm25 = conn.execute(
                """
                SELECT count(*) FROM chunks c
                WHERE c.bm25_length IS NULL
                   OR c.bm25_length <> COALESCE(
                       (SELECT sum(tf) FROM chunk_terms ct WHERE ct.chunk_id = c.id),
                       0
                   )
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
    typer.echo(f"chunks missing embedding: {missing_embeddings[0] if missing_embeddings else 0}")
    typer.echo(f"chunks missing bm25 data: {missing_bm25[0] if missing_bm25 else 0}")


def _print_retrieval_result(result: RetrievalResult) -> None:
    typer.echo(f"mode: {result.mode}")
    typer.echo(f"allowed documents: {result.allowed_document_count}")
    typer.echo(f"query terms: {' '.join(result.query_terms)}")
    if not result.chunks:
        typer.echo("no results")
        return
    for rank, chunk in enumerate(result.chunks, start=1):
        scores: list[str] = []
        if chunk.bm25_rank is not None:
            scores.append(f"bm25_rank={chunk.bm25_rank}")
            scores.append(f"bm25={chunk.bm25_score:.6f}")
        if chunk.dense_rank is not None:
            scores.append(f"dense_rank={chunk.dense_rank}")
            scores.append(f"distance={chunk.dense_distance:.6f}")
        if chunk.rrf_score is not None:
            scores.append(f"rrf={chunk.rrf_score:.6f}")
        if chunk.rerank_score is not None:
            scores.append(f"rerank={chunk.rerank_score:.6f}")
        heading = chunk.section_heading or "-"
        typer.echo(
            f"{rank:>3}  {chunk.document_id}  pages {chunk.page_start}-{chunk.page_end}  "
            f"{heading}  {' '.join(scores)}"
        )
        typer.echo(f"     {chunk.text[:120].replace(chr(10), ' ')}")
    typer.echo(
        "timings_ms: "
        + " ".join(f"{stage}={value:.2f}" for stage, value in result.timings_ms.items())
    )


@app.command("retrieve")
def retrieve_command(
    question: Annotated[str, typer.Argument(help="Question in Urdu or English.")],
    groups: Annotated[list[str], typer.Option("--group", help="Principal group (repeatable).")],
    mode: Annotated[
        RetrievalMode, typer.Option("--mode", help="bm25, dense, hybrid, or hybrid_rerank.")
    ] = "hybrid_rerank",
    k: Annotated[int, typer.Option("--k", help="Number of chunks to return.")] = 8,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Retrieve policy chunks for a question under a principal's groups."""
    _configure_logging()
    if k < 1:
        typer.secho("--k must be at least 1", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    settings = get_settings()
    try:
        with closing(connect(settings.database_url)) as conn:
            problem = index_problem(conn, settings)
            if problem is not None:
                typer.echo(problem)
                raise typer.Exit(code=1)
            embedder = SentenceTransformerEmbedder(settings)
            reranker = CrossEncoderReranker(settings)
            LOGGER.info(
                "embedding device: %s; reranker device: %s",
                embedder.device,
                reranker.device,
            )
            result = retrieve(
                conn,
                question,
                Principal(frozenset(groups)),
                mode=mode,
                embedder=embedder,
                reranker=reranker,
                settings=with_top_k(settings, k),
            )
    except (psycopg.Error, AccessViolation) as exc:
        typer.secho(f"retrieval failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        _print_retrieval_result(result)


def _write_exclusive_report(prefix: str, payload: dict[str, Any]) -> Path:
    settings = get_settings()
    reports_dir = settings.data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path = reports_dir / f"{prefix}-{stamp}.json"
    suffix = 0
    while True:
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
            return path
        except FileExistsError:
            suffix += 1
            path = reports_dir / f"{prefix}-{stamp}-{suffix}.json"


def _format_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _print_eval_table(report: dict[str, Any]) -> None:
    typer.echo(
        "mode            slice                 n   n_gold  hit@k    recall@k  MRR@k    leaks"
    )
    for mode, mode_report in report["modes"].items():
        slices: list[tuple[str, dict[str, Any]]] = [
            ("overall", mode_report["overall"]),
            ("language:en", mode_report["slices"]["language"].get("en", {"n": 0})),
            ("language:ur", mode_report["slices"]["language"].get("ur", {"n": 0})),
            (
                "cross_lingual:true",
                mode_report["slices"]["cross_lingual"]["true"],
            ),
            (
                "cross_lingual:false",
                mode_report["slices"]["cross_lingual"]["false"],
            ),
        ]
        for label, summary in slices:
            if summary["n"] == 0:
                continue
            typer.echo(
                f"{mode:<15} {label:<21} {summary['n']:>3}  "
                f"{summary['n_with_gold']:>6}  "
                f"{_format_metric(summary['hit_at_k']):<8} "
                f"{_format_metric(summary['recall_at_k']):<9} "
                f"{_format_metric(summary['mrr_at_k']):<8} "
                f"{summary['leaks']:>5}"
            )
    for mode, mode_report in report["modes"].items():
        for stage, values in mode_report["overall"]["latency_ms"].items():
            typer.echo(f"latency {mode} {stage}: mean={values['mean']} p95={values['p95']}")


@eval_app.command("retrieval")
def eval_retrieval(
    dataset: Annotated[Path, typer.Option("--dataset", help="JSONL dataset path.")],
    modes: Annotated[
        str,
        typer.Option("--modes", help="Comma-separated modes."),
    ] = "bm25,dense,hybrid,hybrid_rerank",
    k: Annotated[int, typer.Option("--k", help="Cutoff for metrics.")] = 10,
) -> None:
    """Evaluate retrieval modes on a JSONL dataset."""
    _configure_logging()
    requested_modes = [part.strip() for part in modes.split(",") if part.strip()]
    allowed_modes = {"bm25", "dense", "hybrid", "hybrid_rerank"}
    invalid = sorted(set(requested_modes) - allowed_modes)
    if invalid or not requested_modes or k < 1:
        typer.secho(
            f"invalid modes {invalid or modes!r} or k={k}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    typed_modes: list[RetrievalMode] = [cast(RetrievalMode, mode) for mode in requested_modes]
    settings = get_settings()
    try:
        items = load_dataset(dataset)
        with closing(connect(settings.database_url)) as conn:
            problem = index_problem(conn, settings)
            if problem is not None:
                typer.echo(problem)
                raise typer.Exit(code=1)
            embedder = SentenceTransformerEmbedder(settings)
            reranker = CrossEncoderReranker(settings)
            report = evaluate(
                conn,
                items,
                modes=typed_modes,
                k=k,
                embedder=embedder,
                reranker=reranker,
                settings=settings,
            )
    except (EvaluationError, psycopg.Error, AccessViolation) as exc:
        typer.secho(f"evaluation failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    report["generated_at"] = utc_now().isoformat()
    report["dataset"] = {
        "path": str(dataset),
        "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }
    report["pipeline_fingerprint"] = pipeline_fingerprint(settings)
    report["models"] = {
        "embedding_repo": settings.embedding_repo,
        "embedding_revision": settings.embedding_revision,
        "reranker_repo": settings.reranker_repo,
        "reranker_revision": settings.reranker_revision,
    }
    _print_eval_table(report)
    report_path = _write_exclusive_report("retrieval-eval", report)
    typer.echo(f"report: {report_path}")


def _print_answer(result: AnswerResult) -> None:
    typer.echo(result.text)
    if result.sources:
        typer.echo("Sources:")
        for source in result.sources:
            heading = source.get("section_heading") or "-"
            typer.echo(
                f"[{source['number']}] {source['document_title']}, "
                f"pages {source['page_start']}-{source['page_end']}, {heading}"
            )
            typer.echo(f"    {source['cited_text']}")
    typer.echo(f"audit: {result.audit_id}")


@app.command("ask")
def ask_command(
    question: Annotated[str, typer.Argument(help="Question in Urdu or English.")],
    groups: Annotated[list[str], typer.Option("--group", help="Principal group (repeatable).")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Answer a question with citations and write an audit row."""
    _configure_logging()
    settings = get_settings()
    try:
        with closing(connect(settings.database_url)) as conn:
            problem = index_problem(conn, settings)
            if problem is not None:
                typer.echo(problem)
                raise typer.Exit(code=1)
            embedder = SentenceTransformerEmbedder(settings)
            reranker = CrossEncoderReranker(settings)
            model = build_answer_model(settings)
            result = run_answer(
                conn,
                question,
                Principal(frozenset(groups)),
                actor=f"cli:{getpass.getuser()}",
                embedder=embedder,
                reranker=reranker,
                model=model,
                settings=settings,
            )
    except typer.Exit:
        raise
    except Exception as exc:
        typer.secho(f"answer failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "audit_id": result.audit_id,
                    "outcome": result.outcome,
                    "language": result.language,
                    "text": result.text,
                    "sources": list(result.sources),
                    "flags": list(result.flags),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        _print_answer(result)


@audit_app.command("list")
def audit_list(
    limit: Annotated[int, typer.Option("--limit", help="Maximum rows.")] = 20,
    outcome: Annotated[str | None, typer.Option("--outcome", help="Filter by outcome.")] = None,
) -> None:
    """List recent audit rows."""
    _configure_logging()
    if limit < 1:
        typer.secho("--limit must be at least 1", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if outcome is not None and outcome not in get_args(AnswerOutcome):
        typer.secho(f"unknown outcome: {outcome}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    settings = get_settings()
    try:
        with closing(connect(settings.database_url)) as conn:
            if outcome:
                rows = conn.execute(
                    """
                    SELECT id, created_at, outcome, question_language, question
                    FROM answer_audit
                    WHERE outcome = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (outcome, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, created_at, outcome, question_language, question
                    FROM answer_audit
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                ).fetchall()
    except psycopg.Error as exc:
        typer.secho(f"database unavailable: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    for row in rows:
        typer.echo(f"{row[0]}  {row[1].isoformat()}  {row[2]}  {row[3]}  {str(row[4])[:80]}")


@audit_app.command("show")
def audit_show(
    audit_id: Annotated[str, typer.Argument(help="Audit row UUID.")],
) -> None:
    """Pretty-print one audit row as JSON."""
    _configure_logging()
    try:
        audit_uuid = uuid.UUID(audit_id)
    except ValueError as exc:
        typer.secho(f"invalid audit id: {audit_id}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    settings = get_settings()
    try:
        with closing(connect(settings.database_url)) as conn:
            payload = load_audit_row(conn, audit_uuid)
    except psycopg.Error as exc:
        typer.secho(f"database unavailable: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    if payload is None:
        typer.secho(f"audit row not found: {audit_id}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _read_password(password_stdin: bool) -> str:
    if password_stdin:
        return sys.stdin.read().rstrip("\r\n")
    first = getpass.getpass("Password: ")
    second = getpass.getpass("Password again: ")
    if first != second:
        typer.secho("passwords do not match", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    return first


@user_app.command("add")
def user_add(
    username: Annotated[str, typer.Argument(help="Username.")],
    display_name: Annotated[str, typer.Option("--display-name", help="Display name.")],
    role: Annotated[str, typer.Option("--role", help="user, reviewer, or admin.")],
    groups: Annotated[
        list[str] | None, typer.Option("--group", help="Group id (repeatable).")
    ] = None,
    password_stdin: Annotated[
        bool, typer.Option("--password-stdin", help="Read the password from stdin.")
    ] = False,
) -> None:
    """Create a user account."""
    _configure_logging()
    password = _read_password(password_stdin)
    settings = get_settings()
    try:
        with closing(connect(settings.database_url)) as conn:
            user = create_user(
                conn,
                username,
                display_name,
                password,
                role,
                groups or [],
            )
    except (ValueError, psycopg.Error) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(f"created user {user.username}")


@user_app.command("set-groups")
def user_set_groups(
    username: Annotated[str, typer.Argument(help="Username.")],
    groups: Annotated[
        list[str] | None, typer.Option("--group", help="Group id (repeatable).")
    ] = None,
) -> None:
    """Replace a user's group memberships (no groups means none)."""
    _configure_logging()
    settings = get_settings()
    try:
        with closing(connect(settings.database_url)) as conn:
            set_user_groups(conn, username, groups or [])
    except (ValueError, psycopg.Error) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(f"updated groups for {username.strip().lower()}")


@user_app.command("set-role")
def user_set_role(
    username: Annotated[str, typer.Argument(help="Username.")],
    role: Annotated[str, typer.Argument(help="user, reviewer, or admin.")],
) -> None:
    """Change a user's role."""
    _configure_logging()
    settings = get_settings()
    try:
        with closing(connect(settings.database_url)) as conn:
            set_user_role(conn, username, role)
    except (ValueError, psycopg.Error) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(f"updated role for {username.strip().lower()}")


@user_app.command("disable")
def user_disable(
    username: Annotated[str, typer.Argument(help="Username.")],
) -> None:
    """Deactivate a user and delete their sessions."""
    _configure_logging()
    settings = get_settings()
    try:
        with closing(connect(settings.database_url)) as conn:
            set_user_active(conn, username, False)
    except (ValueError, psycopg.Error) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(f"disabled {username.strip().lower()}")


@user_app.command("enable")
def user_enable(
    username: Annotated[str, typer.Argument(help="Username.")],
) -> None:
    """Reactivate a user."""
    _configure_logging()
    settings = get_settings()
    try:
        with closing(connect(settings.database_url)) as conn:
            set_user_active(conn, username, True)
    except (ValueError, psycopg.Error) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(f"enabled {username.strip().lower()}")


@user_app.command("list")
def user_list() -> None:
    """List users, roles, active state, and groups."""
    _configure_logging()
    settings = get_settings()
    try:
        with closing(connect(settings.database_url)) as conn:
            users = list_users(conn)
    except psycopg.Error as exc:
        typer.secho(f"database unavailable: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    for user in users:
        state = "active" if user.is_active else "disabled"
        groups = ",".join(sorted(user.groups)) or "-"
        typer.echo(f"{user.username}  {user.role}  {state}  {groups}")


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind port.")] = 8000,
) -> None:
    """Run the HTTP API with uvicorn."""
    import uvicorn

    uvicorn.run(
        "policy_assistant.api.app:create_app_from_env",
        factory=True,
        host=host,
        port=port,
        workers=1,
    )


if __name__ == "__main__":
    app()
