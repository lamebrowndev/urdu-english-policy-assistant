"""Application settings, loaded from the environment with the ``UPA_`` prefix."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_repo_root(start: Path | None = None) -> Path:
    """Return the nearest ancestor of ``start`` (or the CWD) with pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return current


class Settings(BaseSettings):
    """Runtime configuration.

    Relative paths are resolved against :func:`find_repo_root` so that commands
    behave the same from the repository root and from nested directories.
    """

    model_config = SettingsConfigDict(
        env_prefix="UPA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql://upa:upa_dev_password@127.0.0.1:5433/upa"
    test_database_url: str = "postgresql://upa:upa_dev_password@127.0.0.1:5433/upa_test"

    data_dir: Path = Path("data")
    manifest_path: Path = Path("corpus/manifest.yaml")

    tokenizer_repo: str = "BAAI/bge-m3"
    tokenizer_revision: str = "9a0624b896d81da7492a910ffa53731274b6cf3d"

    embedding_repo: str = "BAAI/bge-m3"
    embedding_revision: str = "9a0624b896d81da7492a910ffa53731274b6cf3d"
    reranker_repo: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    embed_batch_size: int = Field(default=16, gt=0)

    bm25_k1: float = Field(default=1.2, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    retrieve_candidates: int = Field(default=50, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    rerank_candidates: int = Field(default=30, gt=0)
    retrieve_top_k: int = Field(default=8, gt=0)
    hnsw_ef_search: int = Field(default=100, gt=0)

    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = Field(default=16000, gt=0)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_fallbacks: Literal["default", "off"] = "default"
    answer_mode: Literal["bm25", "dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    abstain_min_rerank_score: float = 0.05

    chunk_max_tokens: int = Field(default=350, gt=0)
    chunk_overlap_tokens: int = Field(default=50, ge=0)
    chunk_min_tokens: int = Field(default=60, ge=0)

    quality_min_chars: int = Field(default=30, ge=0)
    quality_max_pua_ratio: float = Field(default=0.01, ge=0, le=1)
    quality_min_script_ratio: float = Field(default=0.60, ge=0, le=1)
    quality_min_lexicon_ur: float = Field(default=0.50, ge=0, le=1)
    quality_min_lexicon_en: float = Field(default=0.60, ge=0, le=1)
    quality_reversed_margin: float = Field(default=0.15, ge=0, le=1)
    quality_max_failed_page_ratio: float = Field(default=0.10, ge=0, le=1)
    quality_max_empty_page_ratio: float = Field(default=0.30, ge=0, le=1)

    log_level: str = "INFO"

    @field_validator("data_dir", "manifest_path")
    @classmethod
    def _resolve_relative(cls, value: Path) -> Path:
        if value.is_absolute():
            return value
        return (find_repo_root() / value).resolve()

    @model_validator(mode="after")
    def _validate_chunk_limits(self) -> Settings:
        if not 0 <= self.chunk_overlap_tokens < self.chunk_max_tokens:
            raise ValueError("chunk_overlap_tokens must be >= 0 and < chunk_max_tokens")
        if not 0 <= self.chunk_min_tokens < self.chunk_max_tokens:
            raise ValueError("chunk_min_tokens must be >= 0 and < chunk_max_tokens")
        return self

    @model_validator(mode="after")
    def _validate_retrieval_limits(self) -> Settings:
        if not (1 <= self.retrieve_top_k <= self.rerank_candidates <= self.retrieve_candidates):
            raise ValueError(
                "retrieve_top_k <= rerank_candidates <= retrieve_candidates is required"
            )
        return self


def get_settings() -> Settings:
    """Build fresh settings (tests change the environment between calls)."""
    return Settings()
