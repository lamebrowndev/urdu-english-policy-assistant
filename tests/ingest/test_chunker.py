import itertools
import os

import pytest

from policy_assistant.config import Settings
from policy_assistant.ingest.chunker import (
    ChunkDraft,
    Sentence,
    chunk_document,
    page_blocks,
)
from policy_assistant.ingest.tokens import load_bge_m3_counter


def word_count(text: str) -> int:
    return len(text.split())


def char_count(text: str) -> int:
    return len(text)


def input_sentences(pages: list[tuple[int, str]]) -> list[str]:
    sentences: list[str] = []
    for page_number, text in pages:
        sentences.extend(
            block.text for block in page_blocks(page_number, text) if isinstance(block, Sentence)
        )
    return sentences


def overlap_prefix(previous: ChunkDraft, current: ChunkDraft) -> int:
    for length in range(min(len(previous.sentences), len(current.sentences)), 0, -1):
        if previous.sentences[-length:] == current.sentences[:length]:
            return length
    return 0


PAGES = [
    (
        1,
        "s1 a. s2 b. s3 c. s4 d. s5 e. s6 f. s7 g. s8 h. s9 i. s10 j. s11 k. s12 l.",
    )
]


def test_chunks_respect_max_and_sequence() -> None:
    chunks = chunk_document(
        PAGES,
        "en",
        word_count,
        max_tokens=6,
        overlap_tokens=4,
        min_tokens=2,
    )
    assert len(chunks) > 1
    assert all(chunk.token_count <= 6 for chunk in chunks)
    for chunk in chunks:
        assert chunk.text == " ".join(chunk.sentences)
        assert chunk.token_count == word_count(chunk.text)
        assert chunk.page_start <= chunk.page_end

    reconstructed: list[str] = []
    for index, chunk in enumerate(chunks):
        prefix = 0 if index == 0 else overlap_prefix(chunks[index - 1], chunk)
        reconstructed.extend(chunk.sentences[prefix:])
    assert reconstructed == input_sentences(PAGES)


def test_overlap_is_bounded_and_reported() -> None:
    chunks = chunk_document(
        PAGES,
        "en",
        word_count,
        max_tokens=6,
        overlap_tokens=4,
        min_tokens=2,
    )
    for previous, current in itertools.pairwise(chunks):
        prefix = overlap_prefix(previous, current)
        if prefix:
            assert word_count(" ".join(current.sentences[:prefix])) <= 4


def test_heading_starts_new_chunk_after_min_tokens() -> None:
    pages = [
        (
            1,
            "1. Purpose\none two three four five.\n2. Scope\nsix seven eight nine ten.",
        )
    ]
    chunks = chunk_document(pages, "en", word_count, max_tokens=100, overlap_tokens=3, min_tokens=3)
    assert [chunk.section_heading for chunk in chunks] == ["1. Purpose", "2. Scope"]
    assert chunks[0].sentences == ("one two three four five.",)
    assert chunks[1].sentences == ("six seven eight nine ten.",)


def test_pending_heading_applies_to_next_chunk_without_overlap() -> None:
    pages = [
        (
            1,
            "1. Purpose\none two\n2. Scope\nthree four five six seven eight.",
        )
    ]
    chunks = chunk_document(pages, "en", word_count, max_tokens=6, overlap_tokens=2, min_tokens=3)
    assert len(chunks) == 2
    assert chunks[0].section_heading == "1. Purpose"
    assert chunks[0].sentences == ("one two",)
    assert chunks[1].section_heading == "2. Scope"
    assert chunks[1].sentences == ("three four five six seven eight.",)


def test_page_range_spans_form_feed_boundary() -> None:
    pages = [
        (1, "one two three four five six seven eight nine ten."),
        (2, "eleven twelve thirteen fourteen fifteen sixteen."),
    ]
    chunks = chunk_document(pages, "en", word_count, max_tokens=20, overlap_tokens=4, min_tokens=1)
    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2


def test_overlong_sentence_is_split() -> None:
    words = " ".join(f"w{index}" for index in range(30)) + "."
    chunks = chunk_document(
        [(1, words)], "en", word_count, max_tokens=5, overlap_tokens=0, min_tokens=1
    )
    assert len(chunks) == 6
    assert all(chunk.token_count <= 5 for chunk in chunks)
    assert " ".join(sentence for chunk in chunks for sentence in chunk.sentences) == words


def test_overlong_word_splits_by_characters() -> None:
    chunks = chunk_document(
        [(1, "abcdefghijklmnopqrst")],
        "en",
        char_count,
        max_tokens=5,
        overlap_tokens=0,
        min_tokens=1,
    )
    assert all(chunk.token_count <= 5 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == "abcdefghijklmnopqrst"


def test_tail_merge() -> None:
    pages = [
        (
            1,
            "1. Purpose\none two three four five\n2. Scope\nsix seven",
        )
    ]
    chunks = chunk_document(pages, "en", word_count, max_tokens=20, overlap_tokens=2, min_tokens=5)
    assert len(chunks) == 1
    assert chunks[0].token_count == 7
    assert chunks[0].text == " ".join(chunks[0].sentences)


def test_determinism_and_ordinals() -> None:
    first = chunk_document(PAGES, "en", word_count, 6, 4, 2)
    second = chunk_document(PAGES, "en", word_count, 6, 4, 2)
    assert first == second
    assert [chunk.ordinal for chunk in first] == list(range(len(first)))


@pytest.mark.skipif(
    os.environ.get("UPA_NETWORK_TESTS") != "1",
    reason="set UPA_NETWORK_TESTS=1 to exercise the real bge-m3 tokenizer",
)
def test_real_counter_on_urdu_and_english() -> None:
    settings = Settings()
    counter = load_bge_m3_counter(settings.tokenizer_repo, settings.tokenizer_revision)
    assert counter("یہ ایک اردو جملہ ہے۔") > 0
    assert counter("This is an English sentence.") > 0
