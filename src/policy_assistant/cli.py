"""Command-line interface (``upa``)."""

from __future__ import annotations

import logging
import re
from contextlib import closing

import psycopg
import typer

from policy_assistant.config import get_settings
from policy_assistant.db.connection import connect
from policy_assistant.db.migrate import applied_checksums, migrate, pending_migrations

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Urdu/English policy assistant.")
db_app = typer.Typer(no_args_is_help=True, add_completion=False, help="Database commands.")
app.add_typer(db_app, name="db")


def _configure_logging() -> None:
    level_name = get_settings().log_level.upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", version)
    return tuple(int(part) for part in parts)


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


if __name__ == "__main__":
    app()
