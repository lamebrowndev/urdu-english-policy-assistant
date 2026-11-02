"""Connections and shared database helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg


def connect(dsn: str, *, autocommit: bool = False) -> psycopg.Connection[Any]:
    """Open a connection with a stable application name."""
    return psycopg.connect(dsn, autocommit=autocommit, application_name="upa")


def utc_now() -> datetime:
    """Timezone-aware current time (library code never uses naive datetimes)."""
    return datetime.now(UTC)
