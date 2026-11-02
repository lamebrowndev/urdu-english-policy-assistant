"""Sentence splitting for display text, Urdu and English aware."""

from __future__ import annotations

import re

_ABBREVIATIONS = frozenset(
    {
        "e.g",
        "i.e",
        "etc",
        "no",
        "nos",
        "sec",
        "secs",
        "art",
        "cl",
        "para",
        "vol",
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "govt",
        "dept",
        "rs",
        "pvt",
        "ltd",
        "co",
        "inc",
        "vs",
        "viz",
        "approx",
        "st",
    }
)
_BOUNDARY = re.compile(r"""[.!?۔؟]+["'”’)\]]*(?=\s|$)""")
_LIST_MARKER = re.compile(r"^\(?(\d{1,3}|[ivxlcdm]{1,6}|[a-z])[.)]$", re.IGNORECASE)
_TRAILING_TOKEN = re.compile(r"(\S+)$")


def _is_abbreviation(text: str, match: re.Match[str]) -> bool:
    if not match.group(0).startswith("."):
        return False
    before = text[: match.start()]
    token_match = _TRAILING_TOKEN.search(before)
    if token_match is None:
        return False
    token = token_match.group(1).rstrip(".").lower()
    return token in _ABBREVIATIONS


def _is_list_marker(segment: str) -> bool:
    return _LIST_MARKER.fullmatch(segment.strip()) is not None


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences on display text."""
    text = paragraph.strip()
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if _is_list_marker(text[start : match.end()]):
            continue
        if _is_abbreviation(text, match):
            continue
        sentence = text[start : match.end()].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences
