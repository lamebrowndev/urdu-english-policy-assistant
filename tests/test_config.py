from pathlib import Path

from policy_assistant.config import Settings, find_repo_root


def test_defaults_match_ticket() -> None:
    settings = Settings()
    assert settings.database_url.endswith("/upa")
    assert settings.tokenizer_repo == "BAAI/bge-m3"
    assert settings.tokenizer_revision == "5617a9f61b028005a4858fdac845db406aefb181"
    assert settings.chunk_max_tokens == 350
    assert settings.quality_min_script_ratio == 0.60
    assert settings.log_level == "INFO"


def test_relative_paths_resolve_against_repo_root() -> None:
    settings = Settings()
    assert settings.data_dir.is_absolute()
    assert settings.manifest_path == find_repo_root() / "corpus" / "manifest.yaml"


def test_repo_root_is_nearest_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == tmp_path
