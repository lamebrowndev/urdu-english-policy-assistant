from pathlib import Path

from policy_assistant.ingest.quality import (
    QualityThresholds,
    assess_document,
    assess_page,
)

THRESHOLDS = QualityThresholds()
REPO_ROOT = Path(__file__).resolve().parents[2]

EN_CLEAN = (
    "The employee may request leave from the manager. The manager will review the request "
    "and record the decision in the system. Every request must include the date and the "
    "reason for the leave."
)
UR_CLEAN = (
    "ہر ملازم کو سال میں بیس کام کے دنوں کی چھٹی ملتی ہے۔ ملازم چھٹی کی درخواست مینیجر کو "
    "دیتا ہے۔ مینیجر درخواست کا جائزہ لیتا ہے اور فیصلہ ریکارڈ کرتا ہے۔ چھٹی کی درخواست میں "
    "تاریخ اور وجہ لکھنا ضروری ہے۔"
)


def test_clean_english_page_accepted() -> None:
    page = assess_page(EN_CLEAN, "en", THRESHOLDS)
    assert page.flags == ()
    assert page.included
    assert page.lexicon_hit_rate is not None
    assert page.lexicon_hit_rate >= THRESHOLDS.min_lexicon_en


def test_clean_urdu_page_accepted() -> None:
    page = assess_page(UR_CLEAN, "ur", THRESHOLDS)
    assert page.flags == ()
    assert page.included
    assert page.script_ratio == 1.0
    assert page.reversed_hit_rate is not None


def test_blank_page_flagged_empty() -> None:
    page = assess_page("  \n\n ", "en", THRESHOLDS)
    assert page.flags == ("empty",)


def test_pua_heavy_page_flagged() -> None:
    page = assess_page(EN_CLEAN + " " + "\ue000" * 10, "en", THRESHOLDS)
    assert "pua" in page.flags
    assert not page.included


def test_mojibake_urdu_is_wrong_script() -> None:
    raw = (REPO_ROOT / "corpus/samples/leave-policy-ur.txt").read_bytes()
    mojibake = raw.decode("latin-1")
    page = assess_page(mojibake, "ur", THRESHOLDS)
    assert "wrong_script" in page.flags
    assert page.script_ratio is not None
    assert page.script_ratio < THRESHOLDS.min_script_ratio


def test_reversed_urdu_page_flagged() -> None:
    reversed_text = " ".join(token[::-1] for token in UR_CLEAN.split())
    page = assess_page(reversed_text, "ur", THRESHOLDS)
    assert "reversed_text" in page.flags
    assert page.reversed_hit_rate is not None
    assert page.lexicon_hit_rate is not None
    assert page.reversed_hit_rate > page.lexicon_hit_rate + THRESHOLDS.reversed_margin


def test_document_no_text() -> None:
    assessment = assess_document(["", "   "], "en", THRESHOLDS)
    assert assessment.verdict == "quarantined"
    assert assessment.reason == "no_text"
    assert assessment.included_pages() == []


def test_document_likely_scanned() -> None:
    assessment = assess_document(["", "", EN_CLEAN, EN_CLEAN], "en", THRESHOLDS)
    assert assessment.reason == "likely_scanned"
    assert assessment.empty_ratio == 0.5


def test_document_bad_text_layer() -> None:
    bad = EN_CLEAN + " " + "\ue000" * 20
    assessment = assess_document([bad] * 4, "en", THRESHOLDS)
    assert assessment.reason == "bad_text_layer"
    assert assessment.failed_ratio == 1.0


def test_document_accepted_with_flagged_pages_excluded() -> None:
    bad = EN_CLEAN + " " + "\ue000" * 20
    pages = [EN_CLEAN] * 9 + [bad]
    assessment = assess_document(pages, "en", THRESHOLDS)
    assert assessment.verdict == "accepted"
    assert assessment.reason is None
    assert len(assessment.included_pages()) == 9
