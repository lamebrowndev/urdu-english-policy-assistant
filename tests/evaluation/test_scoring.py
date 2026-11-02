from __future__ import annotations

import pytest

from policy_assistant.evaluation.scoring import ScoredItem, judgement, summarize, wilson_interval


def _item(
    item_id: str,
    *,
    language: str = "en",
    category: str = "answerable",
    outcome: str = "answered",
    verdict: str | None = "correct",
    citation_ok: bool | None = True,
) -> ScoredItem:
    return ScoredItem(
        item_id=item_id,
        audit_id=f"audit-{item_id}",
        language=language,
        category=category,
        outcome=outcome,
        verdict=verdict,
        citation_ok=citation_ok,
    )


def test_wilson_interval_known_values() -> None:
    assert wilson_interval(0, 0) is None
    low, high = wilson_interval(180, 200) or (0.0, 0.0)
    # 90% on 200 questions: about plus or minus four points.
    assert low == pytest.approx(0.8506, abs=1e-3)
    assert high == pytest.approx(0.9343, abs=1e-3)
    low, high = wilson_interval(0, 10) or (1.0, 1.0)
    assert low == 0.0
    assert high == pytest.approx(0.2775, abs=1e-3)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_judgement_rules() -> None:
    assert judgement(_item("a")) is True
    assert judgement(_item("b", outcome="abstained_model", verdict="correct_abstention")) is True
    assert judgement(_item("c", verdict="partially_correct")) is False
    assert judgement(_item("d", verdict="incorrect")) is False
    assert judgement(_item("e", outcome="abstained_retrieval", verdict="missed_answer")) is False
    assert judgement(_item("f", outcome="error", verdict=None)) is False
    assert judgement(_item("g", verdict=None)) is None


def test_summary_slices_and_completeness() -> None:
    items = [
        _item("en-1"),
        _item("en-2", verdict="partially_correct", citation_ok=False),
        _item(
            "en-3",
            category="restricted",
            outcome="abstained_retrieval",
            verdict="correct_abstention",
            citation_ok=None,
        ),
        _item("ur-1", language="ur"),
        _item("ur-2", language="ur", outcome="error", verdict=None, citation_ok=None),
        _item(
            "ur-3",
            language="ur",
            category="unanswerable",
            outcome="abstained_model",
            verdict=None,
            citation_ok=None,
        ),
    ]
    report = summarize(items)
    overall = report["overall"]
    assert report["complete"] is False
    assert overall["items"] == 6
    assert overall["scored"] == 5
    assert overall["unreviewed"] == 1
    assert overall["correct"] == 3
    assert overall["accuracy"] == pytest.approx(0.6)
    assert overall["partially_correct"] == 1
    assert overall["errors"] == 1
    assert overall["answered"] == 3
    assert overall["abstained"] == 2
    assert report["by_language"]["ur"]["scored"] == 2
    assert report["by_language"]["ur"]["correct"] == 1
    assert list(report["by_category"]) == ["answerable", "unanswerable", "restricted"]
    assert report["by_tag"] == {}
    tagged = summarize(
        [
            _item("t1"),
            ScoredItem(
                "t2", "a2", "ur", "answerable", "answered", "incorrect", False, ("cross-lingual",)
            ),
        ]
    )
    assert tagged["by_tag"]["cross-lingual"]["items"] == 1
    assert tagged["by_tag"]["cross-lingual"]["correct"] == 0
    assert report["by_category"]["restricted"]["accuracy"] == 1.0
    assert report["citations"] == {"reviewed_answers": 3, "citation_ok": 2}
    assert summarize([_item("x")])["complete"] is True
