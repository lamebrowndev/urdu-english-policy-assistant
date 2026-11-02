from policy_assistant.text.sentences import split_sentences


def test_urdu_terminators() -> None:
    assert split_sentences("پہلا جملہ۔ دوسرا جملہ؟ تیسرا جملہ!") == [
        "پہلا جملہ۔",
        "دوسرا جملہ؟",
        "تیسرا جملہ!",
    ]


def test_abbreviation_rs() -> None:
    sentences = split_sentences("Rs. 5,000 is the limit. The next sentence follows.")
    assert sentences == ["Rs. 5,000 is the limit.", "The next sentence follows."]


def test_abbreviation_eg() -> None:
    sentences = split_sentences("Use e.g. this example. Then stop.")
    assert sentences == ["Use e.g. this example.", "Then stop."]


def test_abbreviation_sec() -> None:
    sentences = split_sentences("Sec. 4 applies here. Read it.")
    assert sentences == ["Sec. 4 applies here.", "Read it."]


def test_numeric_list_markers() -> None:
    assert split_sentences("1. First item. 2. Second item.") == [
        "1. First item.",
        "2. Second item.",
    ]


def test_letter_list_markers() -> None:
    assert split_sentences("(a) First item. (b) Second item.") == [
        "(a) First item.",
        "(b) Second item.",
    ]


def test_decimal_point_is_not_a_boundary() -> None:
    assert split_sentences("The limit is 1.5 days.") == ["The limit is 1.5 days."]


def test_closing_quotes_stay_with_sentence() -> None:
    assert split_sentences('He said "Stop." Then he left.') == [
        'He said "Stop."',
        "Then he left.",
    ]


def test_empty_input() -> None:
    assert split_sentences("   ") == []


def test_urdu_list_marker() -> None:
    assert split_sentences("۱۔ ملازمین کو تحریری درخواست دینی ہوگی۔ دوسرا جملہ۔") == [
        "۱۔ ملازمین کو تحریری درخواست دینی ہوگی۔",
        "دوسرا جملہ۔",
    ]
