from policy_assistant.text.terms import bm25_terms, stopword_checksums


def test_english_stopwords_and_stemming() -> None:
    assert bm25_terms("Employees are entitled to 20 working days of annual leave.") == [
        "employe",
        "entitl",
        "20",
        "work",
        "day",
        "annual",
        "leav",
    ]


def test_urdu_stopwords_removed_and_tokens_unstemmed() -> None:
    assert bm25_terms("ملازمین کو 20 دن کی سالانہ چھٹی ملتی ہے۔") == [
        "ملازمین",
        "20",
        "دن",
        "سالانہ",
        "چھٹی",
        "ملتی",
    ]


def test_single_letters_dropped_but_digits_kept() -> None:
    assert bm25_terms("x y 7") == ["7"]


def test_all_stopwords() -> None:
    assert bm25_terms("the and of") == []


def test_stopword_checksums() -> None:
    checksums = stopword_checksums()
    assert set(checksums) == {"en.txt", "ur.txt"}
    assert all(len(value) == 64 for value in checksums.values())
