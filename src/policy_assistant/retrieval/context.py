"""Text used for BM25, embeddings, and reranker passages."""

from __future__ import annotations


def retrieval_text(title: str, section_heading: str | None, text: str) -> str:
    """Join title, optional heading, and chunk text with newlines."""
    parts = [title]
    if section_heading:
        parts.append(section_heading)
    parts.append(text)
    return "\n".join(parts)
