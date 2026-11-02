"""Application settings, loaded from the environment with the ``UPA_`` prefix."""

from __future__ import annotations

from pathlib import Path

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
    tokenizer_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"

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


def get_settings() -> Settings:
    """Build fresh settings (tests change the environment between calls)."""
    return Settings()
