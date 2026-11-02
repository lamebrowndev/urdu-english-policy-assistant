"""Script-based language detection for Urdu and English."""

from __future__ import annotations

import unicodedata
from typing import Literal

Language = Literal["ur", "en", "mixed", "unknown"]

ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
LATIN_RANGE = (0x0041, 0x024F)


def _in_ranges(code_point: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= code_point <= end for start, end in ranges)


def is_arabic_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L") and _in_ranges(ord(char), ARABIC_RANGES)


def is_latin_letter(char: str) -> bool:
    return (
        unicodedata.category(char).startswith("L") and LATIN_RANGE[0] <= ord(char) <= LATIN_RANGE[1]
    )


def script_letter_counts(text: str) -> tuple[int, int]:
    """Return (Arabic-script letters, Latin letters)."""
    arabic = 0
    latin = 0
    for char in text:
        if is_arabic_letter(char):
            arabic += 1
        elif is_latin_letter(char):
            latin += 1
    return arabic, latin


def detect_language(text: str) -> Language:
    """Classify text as Urdu, English, mixed, or unknown by letter share."""
    arabic, latin = script_letter_counts(text)
    total = arabic + latin
    if total < 10:
        return "unknown"
    share = arabic / total
    if share >= 0.7:
        return "ur"
    if share <= 0.3:
        return "en"
    return "mixed"
