from policy_assistant.answering.language import answer_language


def test_english_question() -> None:
    assert answer_language("How many days of annual leave do employees get?") == "en"


def test_urdu_question() -> None:
    assert answer_language("ملازمین کو کتنے دن کی سالانہ چھٹی ملتی ہے؟") == "ur"


def test_urdu_question_with_english_acronyms() -> None:
    assert answer_language("کیا HR کی پالیسی کے مطابق CNIC ضروری ہے؟") == "ur"


def test_latin_dominant_mixed_question() -> None:
    assert answer_language("What is the leave policy? چھٹی") == "en"
