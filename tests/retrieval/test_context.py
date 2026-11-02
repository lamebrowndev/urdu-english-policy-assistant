from policy_assistant.retrieval.context import retrieval_text


def test_retrieval_text_without_heading() -> None:
    assert retrieval_text("Title", None, "Body") == "Title\nBody"


def test_retrieval_text_with_heading() -> None:
    assert retrieval_text("Title", "1. Purpose", "Body") == "Title\n1. Purpose\nBody"
