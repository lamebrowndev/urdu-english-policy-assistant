"""Build custom-content documents for Claude citations."""

from __future__ import annotations

from typing import Any

from policy_assistant.retrieval.retriever import RetrievedChunk


def _title(chunk: RetrievedChunk) -> str:
    if chunk.page_start == chunk.page_end:
        pages = f"p. {chunk.page_start}"
    else:
        pages = f"pp. {chunk.page_start}\u2013{chunk.page_end}"
    return f"{chunk.document_title} ({pages})"


def build_documents(chunks: tuple[RetrievedChunk, ...]) -> list[dict[str, Any]]:
    """One document per chunk, in rank order, with sentences as blocks."""
    documents: list[dict[str, Any]] = []
    for chunk in chunks:
        document: dict[str, Any] = {
            "type": "document",
            "source": {
                "type": "content",
                "content": [{"type": "text", "text": sentence} for sentence in chunk.sentences],
            },
            "title": _title(chunk),
            "citations": {"enabled": True},
        }
        if chunk.section_heading:
            document["context"] = f"Section: {chunk.section_heading}"
        documents.append(document)
    return documents
