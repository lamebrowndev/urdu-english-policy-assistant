from policy_assistant.text.language import detect_language, script_letter_counts


def test_urdu_text() -> None:
    assert detect_language("یہ ایک اردو جملہ ہے جو کافی لمبا ہے۔") == "ur"


def test_english_text() -> None:
    assert detect_language("This is an English sentence with enough letters.") == "en"


def test_mixed_text() -> None:
    assert detect_language("یہ ایک mixed sentence ہے جس میں دونوں ہیں۔") == "mixed"


def test_short_text_is_unknown() -> None:
    assert detect_language("اردو") == "unknown"
    assert detect_language("abc") == "unknown"


def test_arabic_presentation_forms_count_as_arabic() -> None:
    arabic, latin = script_letter_counts("\ufb8e\ufbfc abc")
    assert arabic == 2
    assert latin == 3
