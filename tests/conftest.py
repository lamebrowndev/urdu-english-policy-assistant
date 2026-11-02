from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from policy_assistant.config import Settings
from policy_assistant.db.connection import connect
from policy_assistant.db.migrate import migrate
from policy_assistant.ingest.manifest import Manifest
from tests.fakes import HashingEmbedder, OverlapReranker

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_database_url() -> str:
    return os.environ.get("UPA_TEST_DATABASE_URL") or Settings().test_database_url


@pytest.fixture
def db_dsn() -> Iterator[str]:
    dsn = test_database_url()
    database_name = str(conninfo_to_dict(dsn).get("dbname", ""))
    if not database_name.endswith("_test"):
        pytest.fail(
            f"refusing to test against database {database_name!r}: its name must end in '_test'"
        )
    try:
        conn = connect(dsn, autocommit=True)
    except psycopg.OperationalError as exc:
        if os.environ.get("UPA_REQUIRE_DB") == "1":
            pytest.fail(f"database required but unreachable: {exc}")
        pytest.skip(f"database unreachable: {exc}")

    try:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
        migrate(conn)
    finally:
        conn.close()
    yield dsn


@pytest.fixture
def fake_counter() -> Callable[[str], int]:
    def count_tokens(text: str) -> int:
        return len(text.split())

    return count_tokens


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
    shutil.copytree(REPO_ROOT / "corpus", root / "corpus")
    return root


@pytest.fixture
def sample_manifest(sample_repo: Path) -> Manifest:
    return Manifest.load(sample_repo / "corpus" / "manifest.sample.yaml", repo_root=sample_repo)


@pytest.fixture
def settings_for(db_dsn: str, tmp_path: Path) -> Settings:
    return Settings(database_url=db_dsn, data_dir=tmp_path / "data")


@pytest.fixture
def fake_embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def fake_reranker() -> OverlapReranker:
    return OverlapReranker()
