import re
import unicodedata
from pathlib import Path

from policy_assistant.ingest.manifest import Manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = REPO_ROOT / "corpus" / "samples"


def test_english_leave_sample_shape() -> None:
    text = (SAMPLES / "leave-policy-en.txt").read_text(encoding="utf-8")
    assert text.startswith("Fictional sample")
    assert text.count("\f") == 1
    assert 500 <= len(text.replace("\f", " ").split()) <= 800
    for heading in [
        "1. Purpose",
        "2. Scope",
        "3. Annual Leave",
        "4. Sick Leave",
        "5. Maternity and Paternity Leave",
        "6. Leave Encashment",
    ]:
        assert heading in text
    assert "20 working days" in text
    assert "2 consecutive sick days" in text
    assert "Rs. 50,000" in text
    assert "e.g." in text
    assert "1.5 days" in text
    assert "(a)" in text


def test_urdu_leave_sample_shape() -> None:
    text = (SAMPLES / "leave-policy-ur.txt").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "فرضی نمونہ — یہ حقیقی پالیسی نہیں ہے۔"
    assert text.count("\f") == 1
    for heading in [
        "۱۔ مقصد",
        "۲۔ دائرہ کار",
        "۳۔ سالانہ چھٹی",
        "۴۔ بیماری کی چھٹی",
        "۵۔ زچگی اور پدریت کی چھٹی",
        "۶۔ چھٹیوں کی نقدی",
    ]:
        assert heading in text
    assert re.search(r"[\u06f0-\u06f9]", text)
    assert re.search(r"[0-9]", text)

    # Arabic-code-point words retained deliberately for normalization tests:
    # "يہ" (Arabic yeh U+064A) and "كرتا" (Arabic kaf U+0643).
    arabic_yeh_words = {word for word in text.split() if "\u064a" in word}
    arabic_kaf_words = {word for word in text.split() if "\u0643" in word}
    assert "يہ" in arabic_yeh_words
    assert any("كرتا" in word for word in arabic_kaf_words)

    diacritic_words = {
        word for word in text.split() if any(unicodedata.category(char) == "Mn" for char in word)
    }
    assert "مُلازِم" in diacritic_words
    assert "زَرُوری" in diacritic_words
    assert len(diacritic_words) >= 2


def test_finance_sample_shape() -> None:
    text = (SAMPLES / "finance-approvals-en.txt").read_text(encoding="utf-8")
    assert text.startswith("Fictional sample")
    assert len(text.split()) >= 250
    assert "Rs. 250,000" in text
    assert "Rs. 2,000,000" in text
    assert "Chief Executive Officer" in text


def test_sample_manifest_acls_and_translation() -> None:
    manifest = Manifest.load(REPO_ROOT / "corpus" / "manifest.sample.yaml", repo_root=REPO_ROOT)
    documents = {document.id: document for document in manifest.documents}
    assert documents["leave-policy-en"].acl == ["all_staff"]
    assert documents["leave-policy-ur"].acl == ["all_staff"]
    assert documents["finance-approvals-en"].acl == ["finance"]
    assert documents["leave-policy-ur"].translation_of == "leave-policy-en"
    assert documents["leave-policy-ur"].language == "ur"
