"""BM25 term extraction shared by indexing and querying."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from importlib import resources

import snowballstemmer

from policy_assistant.text.normalize import fold, tokenize

_STEMMER = snowballstemmer.stemmer("english")
_STOPWORD_FILES = ("en.txt", "ur.txt")


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    words: set[str] = set()
    package = resources.files("policy_assistant.text.stopwords")
    for name in _STOPWORD_FILES:
        payload = package.joinpath(name).read_text(encoding="utf-8")
        for line in payload.splitlines():
            word = fold(line.strip())
            if word:
                words.add(word)
    return frozenset(words)


@lru_cache(maxsize=1)
def stopword_checksums() -> dict[str, str]:
    """SHA-256 of each stopword file, used in the pipeline fingerprint."""
    package = resources.files("policy_assistant.text.stopwords")
    return {
        name: hashlib.sha256(package.joinpath(name).read_bytes()).hexdigest()
        for name in _STOPWORD_FILES
    }


def bm25_terms(text: str) -> list[str]:
    """Tokenize, remove stopwords, drop single letters, and stem ASCII words."""
    stopwords = _stopwords()
    terms: list[str] = []
    for token in tokenize(text):
        if token in stopwords:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        if token.isascii() and token.isalpha():
            token = _STEMMER.stemWord(token)
        terms.append(token)
    return terms
