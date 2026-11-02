"""Argon2id password hashing and uniform verification cost."""

from __future__ import annotations

import contextlib
from functools import cache

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

HASHER = PasswordHasher()
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 1024


def validate_password(password: str) -> None:
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        raise ValueError(
            f"password must be between {MIN_PASSWORD_LENGTH} and {MAX_PASSWORD_LENGTH} characters"
        )


def hash_password(password: str) -> str:
    validate_password(password)
    return HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return bool(HASHER.verify(password_hash, password))
    except (VerifyMismatchError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    return bool(HASHER.check_needs_rehash(password_hash))


@cache
def _dummy_hash() -> str:
    return HASHER.hash("not-a-real-user-password")


def burn_verification(password: str) -> None:
    """Spend the same Argon2 work as a real verification for unknown users."""
    with contextlib.suppress(VerifyMismatchError, InvalidHashError):
        HASHER.verify(_dummy_hash(), password)
