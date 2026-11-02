"""Pure view helpers for the server-rendered UI."""

from __future__ import annotations

from typing import Any

from policy_assistant.answering.language import answer_language


def text_lang(text: str) -> str:
    return answer_language(str(text))


def text_dir(text: str) -> str:
    return "rtl" if text_lang(text) == "ur" else "ltr"


def page_label(start: int, end: int) -> str:
    if start == end:
        return f"p. {start}"
    return f"pp. {start}\u2013{end}"


def segment_views(answer: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not answer:
        return []
    views: list[dict[str, Any]] = []
    for segment in answer.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        numbers: list[int] = []
        for citation in segment.get("citations") or []:
            number = citation.get("number") if isinstance(citation, dict) else None
            if isinstance(number, int) and number not in numbers:
                numbers.append(number)
        views.append({"text": str(segment.get("text", "")), "numbers": numbers})
    return views


def cited_flags(chunk: dict[str, Any], sources: list[dict[str, Any]]) -> list[bool]:
    sentences = chunk.get("sentences") or []
    flags = [False] * len(sentences)
    chunk_id = chunk.get("chunk_id")
    for source in sources:
        if source.get("chunk_id") != chunk_id:
            continue
        start = max(0, int(source.get("start_sentence", 0)))
        end = min(len(flags), int(source.get("end_sentence", 0)))
        for index in range(start, end):
            flags[index] = True
    return flags


def status_key(outcome: str) -> str:
    if outcome == "answered":
        return "answered"
    if outcome.startswith("abstained_") or outcome == "refused":
        return "abstained"
    return "error"


def verdicts_for(outcome: str) -> list[str]:
    if outcome == "answered":
        return ["correct", "partially_correct", "incorrect"]
    if outcome == "error":
        return []
    return ["correct_abstention", "missed_answer"]


def rerank_display(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"
