"""Validate model citations and render numbered sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from policy_assistant.answering.model import CitationRef, ReplySegment
from policy_assistant.retrieval.retriever import RetrievedChunk


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    number: int
    chunk_id: int
    document_id: str
    document_title: str
    page_start: int
    page_end: int
    section_heading: str | None
    start_sentence: int
    end_sentence: int
    cited_text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_heading": self.section_heading,
            "start_sentence": self.start_sentence,
            "end_sentence": self.end_sentence,
            "cited_text": self.cited_text,
        }


@dataclass(frozen=True, slots=True)
class CitationResolution:
    text: str
    sources: tuple[ResolvedSource, ...]
    flags: tuple[str, ...]
    segments: tuple[dict[str, Any], ...]


def _valid(ref: CitationRef, chunks: tuple[RetrievedChunk, ...]) -> bool:
    if not 0 <= ref.document_index < len(chunks):
        return False
    sentence_count = len(chunks[ref.document_index].sentences)
    return 0 <= ref.start_block_index < ref.end_block_index <= sentence_count


def resolve_answer(
    segments: tuple[ReplySegment, ...], chunks: tuple[RetrievedChunk, ...]
) -> CitationResolution:
    """Resolve citations, drop invalid ones, and number sources by first use."""
    numbers: dict[tuple[int, int, int], int] = {}
    sources: list[ResolvedSource] = []
    segment_payloads: list[dict[str, Any]] = []
    rendered: list[str] = []
    flags: list[str] = []

    for segment in segments:
        citation_payloads: list[dict[str, Any]] = []
        marker_numbers: list[int] = []
        for ref in segment.citations:
            if not _valid(ref, chunks):
                if "invalid_citation" not in flags:
                    flags.append("invalid_citation")
                continue
            chunk = chunks[ref.document_index]
            key = (chunk.chunk_id, ref.start_block_index, ref.end_block_index)
            number = numbers.get(key)
            if number is None:
                number = len(numbers) + 1
                numbers[key] = number
                sources.append(
                    ResolvedSource(
                        number=number,
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        document_title=chunk.document_title,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        section_heading=chunk.section_heading,
                        start_sentence=ref.start_block_index,
                        end_sentence=ref.end_block_index,
                        cited_text=ref.cited_text,
                    )
                )
            marker_numbers.append(number)
            citation_payloads.append(
                {
                    "number": number,
                    "chunk_id": chunk.chunk_id,
                    "start_sentence": ref.start_block_index,
                    "end_sentence": ref.end_block_index,
                }
            )
        unique_numbers = list(dict.fromkeys(marker_numbers))
        rendered.append(segment.text + "".join(f"[{number}]" for number in unique_numbers))
        segment_payloads.append({"text": segment.text, "citations": citation_payloads})

    return CitationResolution(
        text="".join(rendered),
        sources=tuple(sources),
        flags=tuple(flags),
        segments=tuple(segment_payloads),
    )
