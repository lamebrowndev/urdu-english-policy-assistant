"""Download corpus files with pinned-hash verification."""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from policy_assistant.ingest.manifest import Manifest

LOGGER = logging.getLogger(__name__)
USER_AGENT = "upa-corpus-fetch/0.1"
TIMEOUT_SECONDS = 60


@dataclass(frozen=True, slots=True)
class FetchResult:
    document_id: str
    status: str
    message: str = ""
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class FetchReport:
    results: tuple[FetchResult, ...]

    @property
    def errors(self) -> int:
        return sum(1 for result in self.results if result.status == "error")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with (
            urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response,
            part.open("wb") as handle,
        ):
            while block := response.read(1024 * 1024):
                handle.write(block)
                digest.update(block)
        return digest.hexdigest()
    except Exception:
        part.unlink(missing_ok=True)
        raise


def fetch_documents(
    manifest: Manifest,
    *,
    repo_root: Path,
    data_dir: Path,
    only_ids: list[str] | None,
) -> FetchReport:
    """Fetch missing sources and verify every present file against its pin."""
    results: list[FetchResult] = []
    for document in manifest.selected(only_ids):
        target = manifest.document_path(document, repo_root, data_dir)
        if target.exists():
            actual = sha256_file(target)
            if document.sha256 is None:
                LOGGER.info("unpinned document %s: %s", document.id, actual)
                results.append(FetchResult(document.id, "unpinned", sha256=actual))
            elif actual == document.sha256:
                results.append(FetchResult(document.id, "ok", sha256=actual))
            else:
                results.append(
                    FetchResult(
                        document.id,
                        "error",
                        f"hash mismatch: expected {document.sha256}, found {actual}",
                        actual,
                    )
                )
            continue

        if document.source_url is None:
            results.append(FetchResult(document.id, "error", f"missing file: {target}"))
            continue

        try:
            actual = _download(str(document.source_url), target)
        except Exception as exc:
            results.append(FetchResult(document.id, "error", f"download failed: {exc}"))
            continue

        if document.sha256 is not None and actual != document.sha256:
            target.with_name(target.name + ".part").unlink(missing_ok=True)
            results.append(
                FetchResult(
                    document.id,
                    "error",
                    f"hash mismatch: expected {document.sha256}, found {actual}",
                    actual,
                )
            )
            continue

        os.replace(target.with_name(target.name + ".part"), target)
        if document.sha256 is None:
            LOGGER.info("unpinned document %s: %s", document.id, actual)
            results.append(FetchResult(document.id, "unpinned", sha256=actual))
        else:
            results.append(FetchResult(document.id, "ok", sha256=actual))
    return FetchReport(tuple(results))
