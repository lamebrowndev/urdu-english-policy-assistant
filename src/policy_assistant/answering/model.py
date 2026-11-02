"""Claude answer model and reply parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import anthropic

from policy_assistant.config import Settings


@dataclass(frozen=True, slots=True)
class CitationRef:
    document_index: int
    start_block_index: int
    end_block_index: int
    cited_text: str


@dataclass(frozen=True, slots=True)
class ReplySegment:
    text: str
    citations: tuple[CitationRef, ...]


@dataclass(frozen=True, slots=True)
class ModelReply:
    segments: tuple[ReplySegment, ...]
    stop_reason: str | None
    stop_details: dict[str, Any] | None
    model_served: str
    request_id: str | None
    usage: dict[str, Any]
    unsupported_citation_types: tuple[str, ...]


class AnswerModelError(Exception):
    """A model call failed. Carries the SDK status and request id when present."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class AnswerModel(Protocol):
    def generate(
        self, system: str, documents: list[dict[str, Any]], question: str
    ) -> ModelReply: ...


def _stop_details(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    details: dict[str, Any] = {}
    for key in ("category", "explanation"):
        item = value.get(key) if isinstance(value, dict) else getattr(value, key, None)
        if item is not None:
            details[key] = item
    return details


def _usage(value: Any) -> dict[str, Any]:
    usage = {
        "input_tokens": int(getattr(value, "input_tokens", 0)),
        "output_tokens": int(getattr(value, "output_tokens", 0)),
    }
    for key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
        item = getattr(value, key, None)
        if item is not None:
            usage[key] = int(item)
    return usage


def parse_reply(response: Any) -> ModelReply:
    """Convert an SDK message (or a test stub) into domain objects."""
    segments: list[ReplySegment] = []
    unsupported: list[str] = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) != "text":
            continue
        citations: list[CitationRef] = []
        for citation in getattr(block, "citations", None) or []:
            citation_type = getattr(citation, "type", None)
            if citation_type == "content_block_location":
                citations.append(
                    CitationRef(
                        document_index=int(citation.document_index),
                        start_block_index=int(citation.start_block_index),
                        end_block_index=int(citation.end_block_index),
                        cited_text=str(citation.cited_text),
                    )
                )
            else:
                unsupported.append(str(citation_type))
        segments.append(ReplySegment(str(block.text), tuple(citations)))
    return ModelReply(
        segments=tuple(segments),
        stop_reason=getattr(response, "stop_reason", None),
        stop_details=_stop_details(getattr(response, "stop_details", None)),
        model_served=str(getattr(response, "model", "")),
        request_id=getattr(response, "_request_id", None),
        usage=_usage(getattr(response, "usage", None)),
        unsupported_citation_types=tuple(unsupported),
    )


class ClaudeAnswerModel:
    """One Claude beta call per question, with citations enabled by documents."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client or anthropic.Anthropic(timeout=settings.llm_timeout_seconds)

    def generate(self, system: str, documents: list[dict[str, Any]], question: str) -> ModelReply:
        settings = self._settings
        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "max_tokens": settings.llm_max_tokens,
            "system": system,
            "output_config": {"effort": settings.llm_effort},
            "messages": [
                {
                    "role": "user",
                    "content": [*documents, {"type": "text", "text": question}],
                }
            ],
        }
        if settings.llm_fallbacks == "default":
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        try:
            response = self._client.beta.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            raise AnswerModelError(
                str(exc),
                status_code=getattr(exc, "status_code", None),
                request_id=getattr(exc, "request_id", None),
            ) from exc
        except anthropic.APIStatusError as exc:
            raise AnswerModelError(
                str(exc),
                status_code=getattr(exc, "status_code", None),
                request_id=getattr(exc, "request_id", None),
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise AnswerModelError(
                str(exc),
                status_code=None,
                request_id=getattr(exc, "request_id", None),
            ) from exc
        except anthropic.APIError as exc:
            raise AnswerModelError(
                str(exc),
                status_code=getattr(exc, "status_code", None),
                request_id=getattr(exc, "request_id", None),
            ) from exc
        return parse_reply(response)


def build_answer_model(settings: Settings) -> AnswerModel:
    """Factory for CLI use; tests monkeypatch this."""
    return ClaudeAnswerModel(settings)
