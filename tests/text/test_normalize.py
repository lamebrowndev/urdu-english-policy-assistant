from hypothesis import given
from hypothesis import strategies as st

from policy_assistant.text.normalize import clean_text, fold, tokenize


def test_arabic_code_points_fold_to_urdu() -> None:
    assert fold("\u064a") == "\u06cc"
    assert fold("\u0649") == "\u06cc"
    assert fold("\u0643") == "\u06a9"
    assert fold("\u0647") == "\u06c1"
    assert fold("\u06c0") == "\u06c1"
    assert fold("\u06d5") == "\u06c1"


def test_diacritics_do_not_change_lookup_key() -> None:
    assert fold("مُلازِم") == fold("ملازم") == "ملازم"
    assert tokenize("کَتاب") == ["کتاب"]


def test_hamza_cases() -> None:
    assert fold("\u06cc\u0654") == "\u0626"
    assert fold("\u064a\u0654") == "\u0626"
    assert fold("\u06d3") == "\u0626\u06d2"
    assert fold("\u06c1\u0654") == "\u06c1"


def test_distinct_letters_are_not_folded() -> None:
    assert fold("\u06d2") != fold("\u06cc")
    assert fold("\u06be") != fold("\u06c1")
    assert fold("\u0622") != fold("\u0627")
    assert fold("\u0626") == "\u0626"
    assert fold("\u0624") == "\u0624"


def test_digits_become_ascii() -> None:
    assert fold("\u06f1\u06f2 \u0663\u0664") == "12 34"


def test_zero_width_and_bidi_controls() -> None:
    assert fold("ک\u0640تاب") == "کتاب"
    assert fold("ک\u200cتاب") == "کتاب"
    assert fold("ک\u200bتاب") == "کتاب"
    assert fold("ک\u202eتاب\u202c") == "کتاب"
    assert clean_text("abc\u200b\u202edef") == "abcdef"
    assert "\u200c" in clean_text("ک\u200cتاب")


def test_arabic_presentation_forms() -> None:
    assert fold("\ufb8e\ufbfc") == "کی"


def test_latin_casefold() -> None:
    assert fold("Policy ASSISTANT") == "policy assistant"


def test_clean_text_keeps_diacritics_and_zwnj() -> None:
    cleaned = clean_text("کَتاب\u200c\r\n\r\n\r\nاگلا\t صفحہ")
    assert "کَتاب\u200c" in cleaned
    assert cleaned == "کَتاب\u200c\n\nاگلا صفحہ"


@given(
    st.text(
        alphabet=st.one_of(
            st.characters(min_codepoint=0x0020, max_codepoint=0x007E),
            st.characters(min_codepoint=0x0600, max_codepoint=0x06FF),
            st.characters(min_codepoint=0xFB50, max_codepoint=0xFDFF),
            st.characters(min_codepoint=0xFE70, max_codepoint=0xFEFF),
            st.characters(min_codepoint=0x200B, max_codepoint=0x200F),
            st.just("\u0640"),
            st.just("\u00a0"),
            st.just("\n"),
        ),
        max_size=40,
    )
)
def test_fold_is_idempotent(text: str) -> None:
    assert fold(fold(text)) == fold(text)
