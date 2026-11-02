"""Per-page text-quality gate. Fails closed: flagged pages are never indexed."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from wordfreq import zipf_frequency

from policy_assistant.config import Settings
from policy_assistant.text.language import (
    is_arabic_letter,
    is_latin_letter,
    script_letter_counts,
)
from policy_assistant.text.normalize import clean_text, tokenize

ScriptLanguage = Literal["ur", "en"]
Verdict = Literal["accepted", "quarantined"]


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    min_chars: int = 30
    max_pua_ratio: float = 0.01
    min_script_ratio: float = 0.60
    min_lexicon_ur: float = 0.50
    min_lexicon_en: float = 0.60
    reversed_margin: float = 0.15
    max_failed_page_ratio: float = 0.10
    max_empty_page_ratio: float = 0.30

    @classmethod
    def from_settings(cls, settings: Settings) -> QualityThresholds:
        return cls(
            min_chars=settings.quality_min_chars,
            max_pua_ratio=settings.quality_max_pua_ratio,
            min_script_ratio=settings.quality_min_script_ratio,
            min_lexicon_ur=settings.quality_min_lexicon_ur,
            min_lexicon_en=settings.quality_min_lexicon_en,
            reversed_margin=settings.quality_reversed_margin,
            max_failed_page_ratio=settings.quality_max_failed_page_ratio,
            max_empty_page_ratio=settings.quality_max_empty_page_ratio,
        )


@dataclass(frozen=True, slots=True)
class PageAssessment:
    text: str
    n_chars: int
    pua_ratio: float
    script_ratio: float | None
    lexicon_hit_rate: float | None
    reversed_hit_rate: float | None
    flags: tuple[str, ...]

    @property
    def included(self) -> bool:
        return not self.flags

    def as_quality(self) -> dict[str, Any]:
        return {
            "n_chars": self.n_chars,
            "pua_ratio": self.pua_ratio,
            "script_ratio": self.script_ratio,
            "lexicon_hit_rate": self.lexicon_hit_rate,
            "reversed_hit_rate": self.reversed_hit_rate,
            "flags": list(self.flags),
            "included": self.included,
        }


@dataclass(frozen=True, slots=True)
class DocumentAssessment:
    verdict: Verdict
    reason: str | None
    pages: tuple[PageAssessment, ...]
    empty_ratio: float
    failed_ratio: float

    def included_pages(self) -> list[tuple[int, str]]:
        return [
            (number, page.text) for number, page in enumerate(self.pages, start=1) if page.included
        ]


def _is_pua(char: str) -> bool:
    code_point = ord(char)
    return (
        code_point == 0xFFFD
        or 0xE000 <= code_point <= 0xF8FF
        or 0xF0000 <= code_point <= 0xFFFFD
        or 0x100000 <= code_point <= 0x10FFFD
    )


@lru_cache(maxsize=200_000)
def _zipf(token: str, language: str) -> float:
    return float(zipf_frequency(token, language))


def _token_in_script(token: str, language: ScriptLanguage) -> bool:
    if language == "ur":
        return all(is_arabic_letter(char) for char in token)
    return all(is_latin_letter(char) for char in token)


def _lexicon_tokens(text: str, language: ScriptLanguage) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if token.isalpha() and len(token) >= 2 and _token_in_script(token, language)
    ]


def _hit_rate(tokens: list[str], language: ScriptLanguage) -> float | None:
    if len(tokens) < 20:
        return None
    return sum(1 for token in tokens if _zipf(token, language) > 0) / len(tokens)


def assess_page(
    raw: str, language: ScriptLanguage, thresholds: QualityThresholds
) -> PageAssessment:
    """Clean one page and compute per-page quality metrics and flags."""
    text = clean_text(raw)
    n_chars = sum(1 for char in text if not char.isspace())
    pua_ratio = sum(1 for char in text if _is_pua(char)) / max(n_chars, 1)

    arabic, latin = script_letter_counts(text)
    script_total = arabic + latin
    script_ratio: float | None = None
    if script_total >= 20:
        expected = arabic if language == "ur" else latin
        script_ratio = expected / script_total

    tokens = _lexicon_tokens(text, language)
    lexicon_hit_rate = _hit_rate(tokens, language)
    reversed_hit_rate: float | None = None
    if language == "ur":
        reversed_hit_rate = _hit_rate([token[::-1] for token in tokens], language)

    flags: list[str] = []
    if n_chars < thresholds.min_chars:
        flags.append("empty")
    if pua_ratio > thresholds.max_pua_ratio:
        flags.append("pua")
    if script_ratio is not None and script_ratio < thresholds.min_script_ratio:
        flags.append("wrong_script")
    minimum_lexicon = thresholds.min_lexicon_ur if language == "ur" else thresholds.min_lexicon_en
    if lexicon_hit_rate is not None and lexicon_hit_rate < minimum_lexicon:
        flags.append("low_lexicon")
    if (
        reversed_hit_rate is not None
        and lexicon_hit_rate is not None
        and reversed_hit_rate > lexicon_hit_rate + thresholds.reversed_margin
    ):
        flags.append("reversed_text")

    return PageAssessment(
        text=text,
        n_chars=n_chars,
        pua_ratio=pua_ratio,
        script_ratio=script_ratio,
        lexicon_hit_rate=lexicon_hit_rate,
        reversed_hit_rate=reversed_hit_rate,
        flags=tuple(flags),
    )


def assess_document(
    raw_pages: list[str], language: ScriptLanguage, thresholds: QualityThresholds
) -> DocumentAssessment:
    """Apply the page gate and decide whether a whole document may be indexed."""
    pages = tuple(assess_page(raw, language, thresholds) for raw in raw_pages)
    total = len(pages)
    empty_count = sum(1 for page in pages if "empty" in page.flags)
    non_empty_count = total - empty_count
    empty_ratio = empty_count / total if total else 1.0
    failed_count = sum(1 for page in pages if page.flags and "empty" not in page.flags)
    failed_ratio = failed_count / max(non_empty_count, 1)

    reason: str | None
    if non_empty_count == 0:
        reason = "no_text"
    elif empty_ratio > thresholds.max_empty_page_ratio:
        reason = "likely_scanned"
    elif failed_ratio > thresholds.max_failed_page_ratio:
        reason = "bad_text_layer"
    else:
        reason = None

    return DocumentAssessment(
        verdict="quarantined" if reason else "accepted",
        reason=reason,
        pages=pages,
        empty_ratio=empty_ratio,
        failed_ratio=failed_ratio,
    )
