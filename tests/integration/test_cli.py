from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from policy_assistant.cli import app
from policy_assistant.db.connection import connect

pytestmark = pytest.mark.integration

URDU = "ہر ملازم کو سال میں بیس کام کے دنوں کی چھٹی ملتی ہے۔ مینیجر درخواست کا جائزہ لیتا ہے۔"


def _write_mojibake_manifest(root: Path) -> Path:
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True)
    target = docs_dir / "mojibake.txt"
    mojibake = URDU.encode("utf-8").decode("latin-1")
    target.write_text(mojibake, encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    payload = {
        "version": 1,
        "groups": [{"id": "all_staff", "description": "Everyone."}],
        "documents": [
            {
                "id": "mojibake-ur",
                "title": "Broken Urdu text layer",
                "language": "ur",
                "file_type": "txt",
                "path": "docs/mojibake.txt",
                "sha256": digest,
                "acl": ["all_staff"],
                "license_note": "Fictional sample written for this project.",
            }
        ],
    }
    manifest_path = root / "corpus" / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return manifest_path


def test_quarantined_document_is_absent_and_cli_exits_two(
    db_dsn: str,
    tmp_path: Path,
    fake_counter: Callable[[str], int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
    manifest_path = _write_mojibake_manifest(root)

    monkeypatch.setattr("policy_assistant.cli.find_repo_root", lambda: root)
    monkeypatch.setattr(
        "policy_assistant.cli.load_bge_m3_counter",
        lambda repo, revision: fake_counter,
    )
    result = CliRunner().invoke(
        app,
        ["ingest", "--manifest", str(manifest_path)],
        env={"UPA_DATABASE_URL": db_dsn, "UPA_DATA_DIR": str(root / "data")},
    )
    assert result.exit_code == 2, result.output
    assert "quarantined:bad_text_layer" in result.output

    with connect(db_dsn) as conn:
        assert (
            conn.execute("SELECT count(*) FROM documents WHERE id = 'mojibake-ur'").fetchone()[0]
            == 0
        )
