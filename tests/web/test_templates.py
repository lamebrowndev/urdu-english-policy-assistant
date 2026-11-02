"""Template-level tests for bidirectional rendering (no database, no browser)."""

from __future__ import annotations

import re
from typing import Any

from policy_assistant.web.render import has_letters, latency_display, segment_lang, usage_display
from policy_assistant.web.routes import templates

AUDIT_ID = "00000000-0000-4000-8000-000000000001"
URDU_TITLE = "چھٹیوں کی پالیسی (اردو)"
URDU_HEADING = "۳۔ سالانہ چھٹی"


def _render_message(item: dict[str, Any]) -> str:
    return templates.env.get_template("message.html").render(item=item, t=lambda key: key)


def _answered(segments: list[dict[str, Any]], text: str) -> dict[str, Any]:
    return {
        "audit_id": AUDIT_ID,
        "question": "ملازمین کو کتنے دن کی سالانہ چھٹی ملتی ہے؟",
        "status": "answered",
        "text": text,
        "segments": segments,
        "sources": [
            {
                "number": 1,
                "document_title": URDU_TITLE,
                "page_start": 1,
                "page_end": 1,
                "section_heading": URDU_HEADING,
                "cited_text": "ملازمین کو سالانہ بیس دن کی چھٹی ملتی ہے۔",
            }
        ],
    }


def _answer_div(html: str) -> str:
    match = re.search(r'<div class="answer"[^>]*>.*?</div>', html, re.S)
    assert match is not None
    return match.group(0)


def test_has_letters_and_segment_lang() -> None:
    assert not has_letters("۔")
    assert not has_letters(" , 20 ")
    assert has_letters("Leave")
    assert segment_lang("۔", "ur") is None
    assert segment_lang("ملازمین", "ur") is None
    assert segment_lang("HR", "ur") == "en"
    assert segment_lang("چھٹی", "en") == "ur"


def test_urdu_answer_container_carries_direction_in_english_ui() -> None:
    segments = [
        {"text": "دستاویز کے مطابق ", "numbers": []},
        {"text": "ملازمین کو سالانہ چھٹی ملتی ہے", "numbers": [1]},
        {"text": "۔", "numbers": []},
    ]
    html = _render_message(
        _answered(segments, "دستاویز کے مطابق ملازمین کو سالانہ چھٹی ملتی ہے[1]۔")
    )
    answer = _answer_div(html)
    assert answer.startswith('<div class="answer" lang="ur" dir="rtl">')
    # Same-language and punctuation-only segments inherit the container direction.
    assert "<span>۔</span>" in answer
    assert 'dir="ltr"' not in answer
    assert answer.count('dir="rtl"') == 1
    # No stray whitespace between a segment and its citation marker.
    assert (
        f'<span>ملازمین کو سالانہ چھٹی ملتی ہے</span><sup class="cite">'
        f'<a href="#src-{AUDIT_ID}-1">1</a></sup><span>۔</span>'
    ) in answer


def test_foreign_segment_gets_its_own_isolate() -> None:
    segments = [
        {"text": "Employees may take ", "numbers": []},
        {"text": "سالانہ چھٹی", "numbers": [1]},
        {"text": " every year.", "numbers": []},
    ]
    html = _render_message(_answered(segments, "Employees may take سالانہ چھٹی[1] every year."))
    answer = _answer_div(html)
    assert answer.startswith('<div class="answer" lang="en" dir="ltr">')
    assert '<span lang="ur" dir="rtl">سالانہ چھٹی</span>' in answer
    assert answer.count("dir=") == 2


def test_source_titles_and_headings_are_isolated() -> None:
    segments = [{"text": "ملازمین کو سالانہ چھٹی ملتی ہے", "numbers": [1]}]
    html = _render_message(_answered(segments, "ملازمین کو سالانہ چھٹی ملتی ہے[1]"))
    assert f'<bdi lang="ur">{URDU_TITLE}</bdi>' in html
    assert f'<bdi lang="ur">{URDU_HEADING}</bdi>' in html


def test_usage_and_latency_display() -> None:
    assert usage_display({"input_tokens": 1200, "output_tokens": 85}) == (
        "input 1200 · output 85 tokens"
    )
    assert usage_display({"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 4}) == (
        "input 5 · output 1 tokens · cache read 4"
    )
    assert usage_display({}) == "-"
    assert latency_display(866.8035) == "867 ms"
    assert latency_display(None) == "-"
