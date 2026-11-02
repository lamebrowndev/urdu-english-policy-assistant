from policy_assistant.answering.prompts import (
    ABSTENTION,
    ERROR,
    abstention_text,
    error_text,
    system_prompt,
)


def test_system_prompt_contains_language_and_no_answer_rule() -> None:
    english = system_prompt("en")
    assert "Write the answer in English" in english
    assert "reply with exactly NO_ANSWER and nothing else" in english
    urdu = system_prompt("ur")
    assert "Write the answer in Urdu" in urdu


def test_localized_messages() -> None:
    assert abstention_text("en") == ABSTENTION["en"]
    assert abstention_text("ur") == ABSTENTION["ur"]
    assert error_text("en") == ERROR["en"]
    assert error_text("ur") == ERROR["ur"]
