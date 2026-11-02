"""Corpus manifest models and validation."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

DOCUMENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,79}$")
GROUP_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,62}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Raised when a manifest is structurally invalid or inconsistent."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Group(_StrictModel):
    id: str = Field(pattern=GROUP_ID_RE.pattern)
    description: str = Field(min_length=1)


class DocumentSpec(_StrictModel):
    id: str = Field(pattern=DOCUMENT_ID_RE.pattern)
    title: str = Field(min_length=1)
    language: Literal["ur", "en"]
    file_type: Literal["pdf", "txt"]
    path: str | None = None
    source_url: HttpUrl | None = None
    sha256: str | None = Field(default=None, pattern=SHA256_RE.pattern)
    acl: list[str] = Field(min_length=1)
    translation_of: str | None = None
    license_note: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if "\\" in value:
            raise ValueError("path must use forward slashes")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("path must be relative and stay inside the repository")
        return value

    @field_validator("source_url")
    @classmethod
    def _require_https(cls, value: HttpUrl | None) -> HttpUrl | None:
        if value is not None and value.scheme != "https":
            raise ValueError("source_url must use https")
        return value


class Manifest(_StrictModel):
    version: Literal[1]
    groups: list[Group]
    documents: list[DocumentSpec]

    @model_validator(mode="after")
    def _validate_cross_references(self) -> Manifest:
        group_ids = [group.id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("duplicate group id")
        document_ids = [document.id for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("duplicate document id")

        known_groups = set(group_ids)
        documents_by_id = {document.id: document for document in self.documents}
        for document in self.documents:
            unknown = sorted(set(document.acl) - known_groups)
            if unknown:
                raise ValueError(
                    f"document {document.id!r} references undeclared groups: {unknown}"
                )
            if document.translation_of is not None:
                if document.translation_of == document.id:
                    raise ValueError(f"document {document.id!r} cannot translate itself")
                translated = documents_by_id.get(document.translation_of)
                if translated is None:
                    raise ValueError(
                        f"document {document.id!r} translates undeclared document "
                        f"{document.translation_of!r}"
                    )
                if translated.language == document.language:
                    raise ValueError(
                        f"document {document.id!r} and its translation must differ in language"
                    )
        return self

    @classmethod
    def load(cls, path: Path, *, repo_root: Path) -> Manifest:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ManifestError(f"manifest {path} must be a mapping")
        try:
            manifest = cls.model_validate(payload)
        except ValueError as exc:
            raise ManifestError(f"invalid manifest {path}: {exc}") from exc
        manifest.validate_paths(repo_root)
        return manifest

    def validate_paths(self, repo_root: Path) -> None:
        root = repo_root.resolve()
        for document in self.documents:
            if document.path is not None:
                target = (repo_root / document.path).resolve()
                if not target.is_relative_to(root):
                    raise ManifestError(
                        f"document {document.id!r} path escapes the repository: {document.path}"
                    )

    def document_path(self, document: DocumentSpec, repo_root: Path, data_dir: Path) -> Path:
        if document.path is not None:
            return repo_root / document.path
        base = data_dir if data_dir.is_absolute() else repo_root / data_dir
        return base / "raw" / f"{document.id}.{document.file_type}"

    def selected(self, only_ids: list[str] | None) -> list[DocumentSpec]:
        if not only_ids:
            return sorted(self.documents, key=lambda document: document.id)
        known = {document.id: document for document in self.documents}
        unknown = sorted(set(only_ids) - set(known))
        if unknown:
            raise ManifestError(f"unknown document ids: {unknown}")
        return sorted((known[document_id] for document_id in set(only_ids)), key=lambda d: d.id)


def require_pinned(manifest: Manifest, ids: list[str] | None) -> None:
    missing = [document.id for document in manifest.selected(ids) if document.sha256 is None]
    if missing:
        raise ManifestError(f"selected documents lack a pinned sha256: {missing}")
