import hashlib
from pathlib import Path

import pytest

from policy_assistant.ingest.fetch import fetch_documents
from policy_assistant.ingest.manifest import Manifest


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        url: str = "https://example.org/doc.txt",
        fail_after: int | None = None,
    ) -> None:
        self.payload = payload
        self.url = url
        self.fail_after = fail_after
        self.offset = 0

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def geturl(self) -> str:
        return self.url

    def read(self, size: int) -> bytes:
        if self.fail_after is not None and self.offset >= self.fail_after:
            raise OSError("connection lost")
        end = self.offset + size
        if self.fail_after is not None:
            end = min(end, self.fail_after)
        block = self.payload[self.offset : end]
        self.offset = end
        return block


def _manifest(tmp_path: Path, sha256: str | None) -> Manifest:
    return Manifest.model_validate(
        {
            "version": 1,
            "groups": [{"id": "all_staff", "description": "Everyone."}],
            "documents": [
                {
                    "id": "doc-en",
                    "title": "Document",
                    "language": "en",
                    "file_type": "txt",
                    "source_url": "https://example.org/doc.txt",
                    "sha256": sha256,
                    "acl": ["all_staff"],
                    "license_note": "Fictional.",
                }
            ],
        }
    )


def _fetch(
    manifest: Manifest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: FakeResponse
):
    monkeypatch.setattr(
        "policy_assistant.ingest.fetch.urllib.request.urlopen",
        lambda *args, **kwargs: response,
    )
    return fetch_documents(
        manifest,
        repo_root=tmp_path,
        data_dir=tmp_path / "data",
        only_ids=None,
    )


def test_success_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"hello world"
    digest = hashlib.sha256(payload).hexdigest()
    manifest = _manifest(tmp_path, digest)
    report = _fetch(manifest, tmp_path, monkeypatch, FakeResponse(payload))
    result = report.results[0]
    target = tmp_path / "data" / "raw" / "doc-en.txt"
    assert result.status == "ok"
    assert result.sha256 == digest
    assert target.read_bytes() == payload
    assert not (tmp_path / "data" / "raw" / "doc-en.txt.part").exists()


def test_pinned_mismatch_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest(tmp_path, "0" * 64)
    report = _fetch(manifest, tmp_path, monkeypatch, FakeResponse(b"hello"))
    result = report.results[0]
    target = tmp_path / "data" / "raw" / "doc-en.txt"
    assert result.status == "error"
    assert "hash mismatch" in result.message
    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()


def test_unpinned_file_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"unpinned content"
    manifest = _manifest(tmp_path, None)
    report = _fetch(manifest, tmp_path, monkeypatch, FakeResponse(payload))
    result = report.results[0]
    target = tmp_path / "data" / "raw" / "doc-en.txt"
    assert result.status == "unpinned"
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert target.read_bytes() == payload
    assert not target.with_name(target.name + ".part").exists()


def test_exception_mid_download_leaves_no_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"x" * 100
    manifest = _manifest(tmp_path, hashlib.sha256(payload).hexdigest())
    report = _fetch(manifest, tmp_path, monkeypatch, FakeResponse(payload, fail_after=10))
    result = report.results[0]
    target = tmp_path / "data" / "raw" / "doc-en.txt"
    assert result.status == "error"
    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()


def test_redirect_to_http_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest(tmp_path, hashlib.sha256(b"hello").hexdigest())
    report = _fetch(
        manifest,
        tmp_path,
        monkeypatch,
        FakeResponse(b"hello", url="http://example.org/doc.txt"),
    )
    result = report.results[0]
    target = tmp_path / "data" / "raw" / "doc-en.txt"
    assert result.status == "error"
    assert "non-https" in result.message
    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()
