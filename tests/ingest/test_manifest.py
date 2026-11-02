from pathlib import Path

import pytest
from pydantic import ValidationError

from policy_assistant.ingest.fetch import sha256_file
from policy_assistant.ingest.manifest import Manifest, ManifestError, require_pinned

REPO_ROOT = Path(__file__).resolve().parents[2]


def _payload(**document_overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "id": "doc-en",
        "title": "Document",
        "language": "en",
        "file_type": "txt",
        "path": "corpus/samples/leave-policy-en.txt",
        "sha256": "a" * 64,
        "acl": ["all_staff"],
        "license_note": "Fictional.",
    }
    document.update(document_overrides)
    return {
        "version": 1,
        "groups": [{"id": "all_staff", "description": "Everyone."}],
        "documents": [document],
    }


def test_valid_manifest() -> None:
    manifest = Manifest.model_validate(_payload())
    assert manifest.documents[0].id == "doc-en"


def test_duplicate_document_ids() -> None:
    payload = _payload()
    payload["documents"] = [payload["documents"][0], payload["documents"][0]]
    with pytest.raises(ValidationError, match="duplicate document id"):
        Manifest.model_validate(payload)


def test_duplicate_group_ids() -> None:
    payload = _payload()
    payload["groups"] = [
        {"id": "all_staff", "description": "One."},
        {"id": "all_staff", "description": "Two."},
    ]
    with pytest.raises(ValidationError, match="duplicate group id"):
        Manifest.model_validate(payload)


def test_empty_acl_rejected() -> None:
    with pytest.raises(ValidationError):
        Manifest.model_validate(_payload(acl=[]))


def test_unknown_group_rejected() -> None:
    with pytest.raises(ValidationError, match="undeclared groups"):
        Manifest.model_validate(_payload(acl=["hr"]))


def test_bad_sha256_rejected() -> None:
    with pytest.raises(ValidationError):
        Manifest.model_validate(_payload(sha256="xyz"))


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        Manifest.model_validate(_payload(unexpected="value"))


def test_translation_of_unknown_document() -> None:
    with pytest.raises(ValidationError, match="translates undeclared"):
        Manifest.model_validate(_payload(translation_of="missing"))


def test_translation_of_self() -> None:
    with pytest.raises(ValidationError, match="cannot translate itself"):
        Manifest.model_validate(_payload(translation_of="doc-en"))


def test_translation_of_same_language() -> None:
    payload = _payload()
    second = dict(payload["documents"][0])
    second["id"] = "doc-en-2"
    second["translation_of"] = "doc-en"
    payload["documents"] = [payload["documents"][0], second]
    with pytest.raises(ValidationError, match="must differ in language"):
        Manifest.model_validate(payload)


@pytest.mark.parametrize("path", ["/etc/passwd", "../outside.txt", "a/../../outside.txt"])
def test_absolute_or_escaping_paths_rejected(path: str) -> None:
    with pytest.raises(ValidationError):
        Manifest.model_validate(_payload(path=path))


def test_sample_manifest_hashes_match_files() -> None:
    manifest = Manifest.load(REPO_ROOT / "corpus" / "manifest.sample.yaml", repo_root=REPO_ROOT)
    assert len(manifest.documents) == 3
    for document in manifest.documents:
        assert document.sha256 is not None
        target = manifest.document_path(document, REPO_ROOT, REPO_ROOT / "data")
        assert sha256_file(target) == document.sha256


def test_require_pinned() -> None:
    manifest = Manifest.model_validate(_payload(sha256=None))
    with pytest.raises(ManifestError, match="lack a pinned sha256"):
        require_pinned(manifest, None)


def test_load_manifest_from_disk(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        """
version: 1
groups:
  - id: all_staff
    description: Everyone.
documents:
  - id: doc-en
    title: Document
    language: en
    file_type: txt
    path: corpus/samples/leave-policy-en.txt
    acl: [all_staff]
    license_note: Fictional.
""",
        encoding="utf-8",
    )
    manifest = Manifest.load(manifest_path, repo_root=tmp_path)
    assert manifest.documents[0].path == "corpus/samples/leave-policy-en.txt"
