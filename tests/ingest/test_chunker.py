import os
import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from policy_assistant.config import Settings
from policy_assistant.ingest.chunker import (
    Heading,
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

    reconstructed = [
        sentence for chunk in chunks for sentence in chunk.sentences[chunk.overlap_count :]
    ]
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
    assert chunks[0].overlap_count == 0
    assert any(chunk.overlap_count > 0 for chunk in chunks)
    for chunk in chunks:
        if chunk.overlap_count:
            prefix = chunk.sentences[: chunk.overlap_count]
            assert word_count(" ".join(prefix)) <= 4


def test_heading_text_is_content() -> None:
    chunks = chunk_document(
        [(1, "3. Annual Leave\none two.")],
        "en",
        word_count,
        max_tokens=10,
        overlap_tokens=2,
        min_tokens=2,
    )
    assert chunks[0].section_heading == "3. Annual Leave"
    assert "3. Annual Leave" in chunks[0].sentences


def test_heading_starts_new_chunk_after_min_tokens() -> None:
    pages = [
        (
            1,
            "1. Purpose\none two three four five.\n2. Scope\nsix seven eight nine ten.",
        )
    ]
    chunks = chunk_document(pages, "en", word_count, max_tokens=100, overlap_tokens=3, min_tokens=3)
    assert [chunk.section_heading for chunk in chunks] == ["1. Purpose", "2. Scope"]
    assert chunks[0].sentences == ("1. Purpose", "one two three four five.")
    assert chunks[1].sentences == ("2. Scope", "six seven eight nine ten.")


def test_pending_heading_applies_to_next_chunk_without_overlap() -> None:
    pages = [
        (
            1,
            "1. Purpose\none two\n2. Scope\nthree four five six seven eight.",
        )
    ]
    chunks = chunk_document(pages, "en", word_count, max_tokens=7, overlap_tokens=2, min_tokens=5)
    assert len(chunks) == 2
    assert chunks[0].section_heading == "1. Purpose"
    assert chunks[0].sentences == ("1. Purpose", "one two", "2. Scope")
    assert chunks[1].section_heading == "2. Scope"
    assert chunks[1].sentences == ("three four five six seven eight.",)
    assert chunks[1].overlap_count == 0


def test_continuation_chunks_keep_section_heading() -> None:
    pages = [
        (
            1,
            "3. Annual Leave\n"
            "one two three four five.\n"
            "six seven eight nine ten.\n"
            "eleven twelve thirteen fourteen fifteen.",
        )
    ]
    chunks = chunk_document(pages, "en", word_count, max_tokens=6, overlap_tokens=2, min_tokens=2)
    assert len(chunks) > 1
    assert all(chunk.section_heading == "3. Annual Leave" for chunk in chunks)


def test_heading_detection_table() -> None:
    sentence_lines = [
        "۲۔ سالانہ چھٹی ۲۰ دن ہوگی۔",
        "دفعہ ۵ کے تحت ملازم کو تحریری اطلاع دی جائے گی۔",
        "20 working days of annual leave per calendar year, which",
        "CNIC اور NTN کی تفصیل درج کریں",
    ]
    heading_lines = [
        "3. Annual Leave",
        "3.1 Scope",
        "۱۔ مقصد",
        "Section 5",
        "دفعہ ۵",
        "CHAPTER II",
    ]
    for line in sentence_lines:
        blocks = page_blocks(1, line)
        assert len(blocks) == 1
        assert isinstance(blocks[0], Sentence)
        assert blocks[0].text == line
    for line in heading_lines:
        blocks = page_blocks(1, line)
        assert len(blocks) == 1
        assert isinstance(blocks[0], Heading)
        assert blocks[0].text == line


CONTENT_POOL = [
    "3. Annual Leave",
    "3.1 Scope",
    "۱۔ مقصد",
    "Section 5",
    "دفعہ ۵",
    "CHAPTER II",
    "۲۔ سالانہ چھٹی ۲۰ دن ہوگی۔",
    "1.5 days is the limit.",
    "2024 was a year.",
    "20 working days of annual leave per calendar year, which",
    "CNIC اور NTN کی تفصیل درج کریں",
    "Every employee receives paid annual leave.",
    "ملازم چھٹی کی درخواست مینیجر کو دیتا ہے۔",
    "",
]


@given(st.lists(st.sampled_from(CONTENT_POOL), min_size=1, max_size=20))
def test_no_content_loss(lines: list[str]) -> None:
    page_text = "\n".join(lines)
    pages = [(1, page_text)]
    chunks = chunk_document(
        pages,
        "en",
        word_count,
        max_tokens=6,
        overlap_tokens=2,
        min_tokens=2,
    )
    reconstructed = "".join(
        sentence for chunk in chunks for sentence in chunk.sentences[chunk.overlap_count :]
    )
    expected = "".join(page_text for _, page_text in pages)
    assert re.sub(r"\s", "", reconstructed) == re.sub(r"\s", "", expected)


def test_hard_wrapped_english_fixture() -> None:
    text = (
        "3. Annual Leave\n\n"
        "Every full-time employee is entitled to 20 working days of\n"
        "annual leave in each calendar year. Part-time employees receive\n"
        "leave in proportion to their contracted hours. Requests should\n"
        "normally be submitted at least 1.5 days before the absence.\n"
        "2024 records must be retained for seven years."
    )
    chunks = chunk_document(
        [(1, text)], "en", word_count, max_tokens=30, overlap_tokens=5, min_tokens=5
    )
    normalized_chunks = [re.sub(r"\s+", " ", chunk.text) for chunk in chunks]
    clauses = [
        "Every full-time employee is entitled to 20 working days of annual leave "
        "in each calendar year.",
        "Part-time employees receive leave in proportion to their contracted hours.",
        "Requests should normally be submitted at least 1.5 days before the absence.",
        "2024 records must be retained for seven years.",
    ]
    for clause in clauses:
        normalized_clause = re.sub(r"\s+", " ", clause)
        assert any(normalized_clause in chunk for chunk in normalized_chunks)


def test_hard_wrapped_urdu_fixture() -> None:
    text = (
        "۳۔ سالانہ چھٹی\n\n"
        "ہر مکمل وقتی ملازم کو ہر کیلنڈر سال میں 20 کام کے دنوں کی\n"
        "تنخواہ کے ساتھ سالانہ چھٹی کا حق ہے۔ جز وقتی ملازمین کو اُن کے\n"
        "طے شدہ اوقات کے تناسب سے چھٹی ملتی ہے۔ درخواست منصوبہ بند\n"
        "غیر حاضری سے کم از کم 1.5 دن پہلے جمع کرنی چاہیے۔\n"
        "۲۰۲۴ کے ریکارڈ سات سال تک محفوظ رکھے جائیں گے۔"
    )
    chunks = chunk_document(
        [(1, text)], "ur", word_count, max_tokens=30, overlap_tokens=5, min_tokens=5
    )
    normalized_chunks = [re.sub(r"\s+", " ", chunk.text) for chunk in chunks]
    clauses = [
        "ہر مکمل وقتی ملازم کو ہر کیلنڈر سال میں 20 کام کے دنوں کی "
        "تنخواہ کے ساتھ سالانہ چھٹی کا حق ہے۔",
        "جز وقتی ملازمین کو اُن کے طے شدہ اوقات کے تناسب سے چھٹی ملتی ہے۔",
        "درخواست منصوبہ بند غیر حاضری سے کم از کم 1.5 دن پہلے جمع کرنی چاہیے۔",
        "۲۰۲۴ کے ریکارڈ سات سال تک محفوظ رکھے جائیں گے۔",
    ]
    for clause in clauses:
        normalized_clause = re.sub(r"\s+", " ", clause)
        assert any(normalized_clause in chunk for chunk in normalized_chunks)


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
    assert chunks[0].token_count == 11
    assert chunks[0].text == " ".join(chunks[0].sentences)


def test_determinism_and_ordinals() -> None:
    first = chunk_document(PAGES, "en", word_count, 6, 4, 2)
    second = chunk_document(PAGES, "en", word_count, 6, 4, 2)
    assert first == second
    assert [chunk.ordinal for chunk in first] == list(range(len(first)))
    assert all(chunk.overlap_count >= 0 for chunk in first)


@pytest.mark.skipif(
    os.environ.get("UPA_NETWORK_TESTS") != "1",
    reason="set UPA_NETWORK_TESTS=1 to exercise the real bge-m3 tokenizer",
)
def test_real_counter_on_urdu_and_english() -> None:
    settings = Settings()
    counter = load_bge_m3_counter(settings.tokenizer_repo, settings.tokenizer_revision)
    assert counter("یہ ایک اردو جملہ ہے۔") > 0
    assert counter("This is an English sentence.") > 0
