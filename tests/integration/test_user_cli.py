from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from argon2 import PasswordHasher
from typer.testing import CliRunner

from policy_assistant.auth.sessions import create_session
from policy_assistant.auth.users import get_user
from policy_assistant.cli import app
from policy_assistant.config import Settings
from policy_assistant.db.connection import connect
from policy_assistant.ingest.manifest import Manifest
from policy_assistant.ingest.pipeline import ingest
from tests.fakes import HashingEmbedder

pytestmark = pytest.mark.integration

PASSWORD = "correct horse battery staple"


@pytest.fixture
def user_database(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    monkeypatch.setattr(
        "policy_assistant.auth.passwords.HASHER",
        PasswordHasher(time_cost=1, memory_cost=1024, parallelism=1),
    )
    from policy_assistant.auth import passwords

    passwords._dummy_hash.cache_clear()
    ingest(
        sample_manifest,
        repo_root=sample_repo,
        only_ids=None,
        force=False,
        prune=False,
        count_tokens=fake_counter,
        embedder=fake_embedder,
        settings=settings_for,
    )
    return db_dsn


def _run(dsn: str, args: list[str], *, input_text: str | None = None) -> object:
    return CliRunner().invoke(
        app,
        args,
        env={"UPA_DATABASE_URL": dsn},
        input=input_text,
    )


def test_user_cli_lifecycle(user_database: str) -> None:
    dsn = user_database
    added = _run(
        dsn,
        [
            "user",
            "add",
            "alice",
            "--display-name",
            "Alice",
            "--role",
            "user",
            "--group",
            "all_staff",
            "--password-stdin",
        ],
        input_text=PASSWORD,
    )
    assert added.exit_code == 0, added.output
    with connect(dsn) as conn:
        user = get_user(conn, "alice")
    assert user is not None
    assert user.groups == frozenset({"all_staff"})

    duplicate = _run(
        dsn,
        [
            "user",
            "add",
            "alice",
            "--display-name",
            "Alice",
            "--role",
            "user",
            "--password-stdin",
        ],
        input_text=PASSWORD,
    )
    assert duplicate.exit_code == 1
    unknown_group = _run(
        dsn,
        [
            "user",
            "add",
            "bob",
            "--display-name",
            "Bob",
            "--role",
            "user",
            "--group",
            "missing",
            "--password-stdin",
        ],
        input_text=PASSWORD,
    )
    assert unknown_group.exit_code == 1
    weak = _run(
        dsn,
        [
            "user",
            "add",
            "bob",
            "--display-name",
            "Bob",
            "--role",
            "user",
            "--password-stdin",
        ],
        input_text="short",
    )
    assert weak.exit_code == 1
    bad_role = _run(
        dsn,
        [
            "user",
            "add",
            "bob",
            "--display-name",
            "Bob",
            "--role",
            "owner",
            "--password-stdin",
        ],
        input_text=PASSWORD,
    )
    assert bad_role.exit_code == 1

    groups = _run(dsn, ["user", "set-groups", "alice", "--group", "finance"])
    assert groups.exit_code == 0
    with connect(dsn) as conn:
        assert get_user(conn, "alice").groups == frozenset({"finance"})

    no_groups = _run(dsn, ["user", "set-groups", "alice"])
    assert no_groups.exit_code == 0
    with connect(dsn) as conn:
        assert get_user(conn, "alice").groups == frozenset()

    role = _run(dsn, ["user", "set-role", "alice", "reviewer"])
    assert role.exit_code == 0
    with connect(dsn) as conn:
        assert get_user(conn, "alice").role == "reviewer"

    with connect(dsn) as conn:
        user = get_user(conn, "alice")
        assert user is not None
        create_session(conn, user.id, 8)
    disabled = _run(dsn, ["user", "disable", "alice"])
    assert disabled.exit_code == 0
    with connect(dsn) as conn:
        assert get_user(conn, "alice").is_active is False
        assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0

    enabled = _run(dsn, ["user", "enable", "alice"])
    assert enabled.exit_code == 0
    with connect(dsn) as conn:
        assert get_user(conn, "alice").is_active is True

    listing = _run(dsn, ["user", "list"])
    assert listing.exit_code == 0
    assert "alice" in listing.output
    assert "reviewer" in listing.output
    assert "active" in listing.output
