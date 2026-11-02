from policy_assistant.web.render import (
    cited_flags,
    page_label,
    segment_views,
    status_key,
    text_dir,
    text_lang,
    verdicts_for,
)


def test_block_language_and_direction() -> None:
    assert text_lang("یہ اردو ہے") == "ur"
    assert text_dir("یہ اردو ہے") == "rtl"
    assert text_lang("This is English") == "en"
    assert text_dir("This is English") == "ltr"


def test_page_label() -> None:
    assert page_label(1, 1) == "p. 1"
    assert page_label(2, 5) == "pp. 2\u20135"


def test_segment_views_dedupe_numbers_in_order() -> None:
    answer = {
        "segments": [
            {
                "text": "First",
                "citations": [
                    {"number": 2},
                    {"number": 1},
                    {"number": 2},
                ],
            },
            {"text": "Second", "citations": []},
        ]
    }
    assert segment_views(answer) == [
        {"text": "First", "numbers": [2, 1]},
        {"text": "Second", "numbers": []},
    ]
    assert segment_views(None) == []


def test_cited_flags_boundaries() -> None:
    chunk = {"chunk_id": 7, "sentences": ["a", "b", "c", "d", "e"]}
    sources = [
        {"chunk_id": 7, "start_sentence": 1, "end_sentence": 3},
        {"chunk_id": 7, "start_sentence": 3, "end_sentence": 5},
        {"chunk_id": 8, "start_sentence": 0, "end_sentence": 5},
    ]
    assert cited_flags(chunk, sources) == [False, True, True, True, True]


def test_cited_flags_clamp_and_missing() -> None:
    chunk = {"chunk_id": 7, "sentences": ["a", "b"]}
    sources = [{"chunk_id": 7, "start_sentence": -2, "end_sentence": 99}]
    assert cited_flags(chunk, sources) == [True, True]
    assert cited_flags({"chunk_id": 7}, sources) == []


def test_status_and_verdicts() -> None:
    assert status_key("answered") == "answered"
    assert status_key("abstained_retrieval") == "abstained"
    assert status_key("refused") == "abstained"
    assert status_key("error") == "error"
    assert verdicts_for("answered") == ["correct", "partially_correct", "incorrect"]
    assert verdicts_for("abstained_model") == ["correct_abstention", "missed_answer"]
    assert verdicts_for("error") == []
