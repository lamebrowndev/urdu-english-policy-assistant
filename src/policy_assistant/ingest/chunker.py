"""Pure sentence-aware chunking over cleaned display text."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from typing import Literal

from policy_assistant.ingest.tokens import TokenCounter
from policy_assistant.text.language import detect_language, is_arabic_letter
from policy_assistant.text.normalize import fold
from policy_assistant.text.sentences import split_sentences

ChunkLanguage = Literal["ur", "en", "mixed"]

_URDU_HEADING_TERMS = (
    "باب",
    "حصہ",
    "دفعہ",
    "شق",
    "قاعدہ",
    "ضابطہ",
    "آرٹیکل",
    "جدول",
    "ضمیمہ",
)
_URDU_HEADING_RE = re.compile(
    "^(" + "|".join(re.escape(fold(term)) for term in _URDU_HEADING_TERMS) + r")\s*[0-9]+"
)
_LATIN_HEADING_RE = re.compile(
    r"^(chapter|part|section|article|rule|regulation|schedule|annex|annexure|appendix)"
    r"\s+[0-9ivxlcdm]+[a-z]?\b"
)
_SINGLE_LEVEL_HEADING_RE = re.compile(r"^[0-9]{1,2}[.)]\s+\S")
_MULTI_LEVEL_HEADING_RE = re.compile(r"^[0-9]{1,2}(\.[0-9]{1,2}){1,3}\.?\s+\S")
_END_PUNCTUATION = (".", "?", ":", ";", ",")


@dataclass(frozen=True, slots=True)
class Heading:
    page: int
    text: str


@dataclass(frozen=True, slots=True)
class Sentence:
    page: int
    text: str


Block = Heading | Sentence


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    ordinal: int
    sentences: tuple[str, ...]
    text: str
    section_heading: str | None
    language: ChunkLanguage
    page_start: int
    page_end: int
    token_count: int
    content_sha256: str
    overlap_count: int


def _is_upper_heading(line: str) -> bool:
    if any(is_arabic_letter(char) for char in line):
        return False
    latin = [char for char in line if char.isascii() and char.isalpha()]
    return len(latin) >= 4 and line == line.upper()


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    folded = fold(stripped)
    if not folded or folded.endswith(_END_PUNCTUATION):
        return False
    if len(folded.split()) > 12:
        return False
    if _LATIN_HEADING_RE.match(folded):
        return True
    if _URDU_HEADING_RE.match(folded):
        return True
    if _SINGLE_LEVEL_HEADING_RE.match(folded):
        return True
    if _MULTI_LEVEL_HEADING_RE.match(folded):
        return True
    return _is_upper_heading(stripped)


def _join_paragraph_lines(lines: list[str]) -> str:
    joined = lines[0]
    for line in lines[1:]:
        previous_char = joined[-1:]
        starts_lower_latin = line[:1].isascii() and line[:1].islower()
        if previous_char == "-" and starts_lower_latin:
            joined = joined[:-1] + line
        else:
            joined = joined + " " + line
    return joined


def page_blocks(page_number: int, cleaned_text: str) -> list[Block]:
    """Split a cleaned page into headings and sentence blocks."""
    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        if not paragraph:
            return
        joined = _join_paragraph_lines(paragraph)
        for sentence in split_sentences(joined):
            blocks.append(Sentence(page_number, sentence))
        paragraph.clear()

    for raw_line in cleaned_text.split("\n"):
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if _is_heading(line):
            flush()
            blocks.append(Heading(page_number, line))
        else:
            paragraph.append(line)
    flush()
    return blocks


def _joined(sentences: list[tuple[int, str]]) -> str:
    return " ".join(text for _, text in sentences)


def _split_word(word: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    pieces: list[str] = []
    buffer = ""
    for char in word:
        candidate = buffer + char
        if buffer and count_tokens(candidate) > max_tokens:
            pieces.append(buffer)
            buffer = char
        else:
            buffer = candidate
    if buffer:
        pieces.append(buffer)
    return pieces


def _split_overlong(
    sentence: Sentence, count_tokens: TokenCounter, max_tokens: int
) -> list[Sentence]:
    if count_tokens(sentence.text) <= max_tokens:
        return [sentence]
    pieces: list[str] = []
    words: list[str] = []
    for word in sentence.text.split():
        candidate = " ".join([*words, word])
        if count_tokens(candidate) <= max_tokens:
            words.append(word)
            continue
        if words:
            pieces.append(" ".join(words))
            words = []
        if count_tokens(word) <= max_tokens:
            words = [word]
        else:
            pieces.extend(_split_word(word, count_tokens, max_tokens))
    if words:
        pieces.append(" ".join(words))
    return [Sentence(sentence.page, piece) for piece in pieces if piece]


def _overlap_suffix(
    sentences: list[tuple[int, str]], count_tokens: TokenCounter, overlap_tokens: int
) -> list[tuple[int, str]]:
    suffix: list[tuple[int, str]] = []
    for sentence in reversed(sentences):
        candidate = [sentence, *suffix]
        if count_tokens(_joined(candidate)) <= overlap_tokens:
            suffix = candidate
        else:
            break
    return suffix


def _make_draft(
    ordinal: int,
    sentences: list[tuple[int, str]],
    heading: str | None,
    overlap_count: int,
    doc_language: ChunkLanguage,
    count_tokens: TokenCounter,
) -> ChunkDraft:
    text = _joined(sentences)
    pages = [page for page, _ in sentences]
    detected = detect_language(text)
    language: ChunkLanguage = doc_language if detected == "unknown" else detected
    return ChunkDraft(
        ordinal=ordinal,
        sentences=tuple(sentence for _, sentence in sentences),
        text=text,
        section_heading=heading,
        language=language,
        page_start=min(pages),
        page_end=max(pages),
        token_count=count_tokens(text),
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        overlap_count=overlap_count,
    )


def chunk_document(
    pages: list[tuple[int, str]],
    doc_language: ChunkLanguage,
    count_tokens: TokenCounter,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
) -> list[ChunkDraft]:
    """Chunk included pages. The result is deterministic and pure."""
    blocks: list[Block] = []
    for page_number, text in pages:
        blocks.extend(page_blocks(page_number, text))

    drafts: list[ChunkDraft] = []
    current: list[tuple[int, str]] = []
    current_overlap = 0
    heading: str | None = None
    pending_heading: str | None = None

    def emit() -> None:
        if current:
            drafts.append(
                _make_draft(
                    len(drafts),
                    current,
                    heading,
                    current_overlap,
                    doc_language,
                    count_tokens,
                )
            )

    for block in blocks:
        if isinstance(block, Heading):
            if not current:
                heading = block.text
                current_overlap = 0
            elif count_tokens(_joined(current)) >= min_tokens:
                emit()
                current = []
                current_overlap = 0
                heading = block.text
                pending_heading = None
            else:
                pending_heading = block.text
            content = _split_overlong(Sentence(block.page, block.text), count_tokens, max_tokens)
        else:
            content = _split_overlong(block, count_tokens, max_tokens)

        for sentence in content:
            if not current:
                if pending_heading is not None:
                    heading = pending_heading
                    pending_heading = None
                current_overlap = 0
                current.append((sentence.page, sentence.text))
                continue

            candidate = [*current, (sentence.page, sentence.text)]
            if count_tokens(_joined(candidate)) <= max_tokens:
                current.append((sentence.page, sentence.text))
                continue

            previous = current
            emit()
            overlap: list[tuple[int, str]] = []
            if pending_heading is None:
                overlap = _overlap_suffix(previous, count_tokens, overlap_tokens)
                if (
                    overlap
                    and count_tokens(_joined([*overlap, (sentence.page, sentence.text)]))
                    > max_tokens
                ):
                    overlap = []
            current = list(overlap)
            current_overlap = len(overlap)
            if pending_heading is not None:
                heading = pending_heading
                pending_heading = None
            current.append((sentence.page, sentence.text))

    emit()

    if len(drafts) > 1 and drafts[-1].token_count < min_tokens:
        prior_draft = drafts[-2]
        last_draft = drafts[-1]
        merged_sentences = [
            *prior_draft.sentences,
            *last_draft.sentences[last_draft.overlap_count :],
        ]
        merged_text = " ".join(merged_sentences)
        if count_tokens(merged_text) <= max_tokens:
            pages_for_merge = [(prior_draft.page_start, text) for text in merged_sentences]
            merged = _make_draft(
                prior_draft.ordinal,
                pages_for_merge,
                prior_draft.section_heading or last_draft.section_heading,
                prior_draft.overlap_count,
                doc_language,
                count_tokens,
            )
            merged = replace(
                merged,
                page_start=min(prior_draft.page_start, last_draft.page_start),
                page_end=max(prior_draft.page_end, last_draft.page_end),
            )
            drafts = [*drafts[:-2], merged]

    return drafts
