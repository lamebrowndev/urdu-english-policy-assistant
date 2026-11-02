import pytest

from policy_assistant.retrieval.access import AccessViolation, Principal, ensure_authorized


def test_principal_is_frozen() -> None:
    principal = Principal(frozenset({"all_staff"}))
    with pytest.raises(AttributeError):
        principal.groups = frozenset()  # type: ignore[misc]


def test_ensure_authorized_accepts_allowed_documents() -> None:
    ensure_authorized(["a", "b"], ["a", "b", "c"])


def test_ensure_authorized_rejects_unknown_document() -> None:
    with pytest.raises(AccessViolation, match="unauthorized documents: \\['z'\\]"):
        ensure_authorized(["a", "z"], ["a", "b"])
