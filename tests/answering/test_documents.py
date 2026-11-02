from policy_assistant.answering.documents import build_documents
from tests.fakes import make_chunk


def test_build_documents_rank_order_and_blocks() -> None:
    first = make_chunk(
        chunk_id=1,
        document_id="doc-a",
        document_title="Alpha",
        sentences=("One.", "Two."),
        page_start=1,
        page_end=1,
        section_heading="1. Purpose",
    )
    second = make_chunk(
        chunk_id=2,
        document_id="doc-b",
        document_title="Beta",
        sentences=("Three.",),
        page_start=2,
        page_end=3,
    )
    documents = build_documents((first, second))
    assert [document["title"] for document in documents] == [
        "Alpha (p. 1)",
        "Beta (pp. 2\u20133)",
    ]
    assert documents[0]["context"] == "Section: 1. Purpose"
    assert "context" not in documents[1]
    assert documents[0]["source"]["content"] == [
        {"type": "text", "text": "One."},
        {"type": "text", "text": "Two."},
    ]
    assert all(document["citations"] == {"enabled": True} for document in documents)
    assert all(document["type"] == "document" for document in documents)
