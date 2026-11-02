from pathlib import Path

import pytest
from pydantic import ValidationError

from policy_assistant.config import Settings, find_repo_root


def test_defaults_match_ticket() -> None:
    settings = Settings()
    assert settings.database_url.endswith("/upa")
    assert settings.tokenizer_repo == "BAAI/bge-m3"
    assert settings.tokenizer_revision == "9a0624b896d81da7492a910ffa53731274b6cf3d"
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


def test_chunk_limits_must_be_below_max() -> None:
    with pytest.raises(ValidationError):
        Settings(chunk_max_tokens=350, chunk_overlap_tokens=350)
    with pytest.raises(ValidationError):
        Settings(chunk_max_tokens=350, chunk_min_tokens=350)
    settings = Settings(chunk_max_tokens=350, chunk_overlap_tokens=0, chunk_min_tokens=0)
    assert settings.chunk_overlap_tokens == 0
    assert settings.chunk_min_tokens == 0


def test_retrieval_limits_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        Settings(retrieve_top_k=10, rerank_candidates=5)
    with pytest.raises(ValidationError):
        Settings(rerank_candidates=60, retrieve_candidates=50)
    settings = Settings(
        retrieve_top_k=8,
        rerank_candidates=8,
        retrieve_candidates=8,
    )
    assert settings.rerank_candidates == 8
