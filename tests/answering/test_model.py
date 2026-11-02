from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from policy_assistant.answering.model import (
    AnswerModelError,
    ClaudeAnswerModel,
    parse_reply,
)
from policy_assistant.config import Settings


class StubClient:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


def _response() -> Any:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="hidden"),
            SimpleNamespace(
                type="text",
                text="Employees get 20 working days.",
                citations=[
                    SimpleNamespace(
                        type="content_block_location",
                        document_index=0,
                        start_block_index=0,
                        end_block_index=1,
                        cited_text="Employees get 20 working days.",
                    ),
                    SimpleNamespace(type="char_location", document_index=0),
                ],
            ),
            SimpleNamespace(type="fallback", fallback="other"),
        ],
        stop_reason="end_turn",
        stop_details=SimpleNamespace(category="general_harms", explanation="why"),
        model="claude-served",
        _request_id="req_123",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=2,
            cache_read_input_tokens=3,
        ),
    )


def test_request_kwargs_and_parsing() -> None:
    settings = Settings(
        llm_model="claude-test",
        llm_effort="high",
        llm_max_tokens=123,
        llm_fallbacks="default",
    )
    client = StubClient(_response())
    model = ClaudeAnswerModel(settings, client=client)
    documents = [{"type": "document", "title": "Doc"}]
    reply = model.generate("system", documents, "question")
    kwargs = client.calls[0]
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    assert "thinking" not in kwargs
    assert "format" not in kwargs["output_config"]
    assert kwargs["output_config"] == {"effort": "high"}
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert kwargs["fallbacks"] == "default"
    content = kwargs["messages"][0]["content"]
    assert content[:-1] == documents
    assert content[-1] == {"type": "text", "text": "question"}
    assert reply.model_served == "claude-served"
    assert reply.request_id == "req_123"
    assert reply.stop_reason == "end_turn"
    assert reply.stop_details == {"category": "general_harms", "explanation": "why"}
    assert reply.usage["cache_read_input_tokens"] == 3
    assert reply.unsupported_citation_types == ("char_location",)
    assert len(reply.segments) == 1
    assert reply.segments[0].text == "Employees get 20 working days."
    assert reply.segments[0].citations[0].document_index == 0
    assert reply.segments[0].citations[0].cited_text == "Employees get 20 working days."


def test_fallbacks_off_omits_beta_kwargs() -> None:
    settings = Settings(llm_fallbacks="off")
    client = StubClient(_response())
    ClaudeAnswerModel(settings, client=client).generate("s", [], "q")
    assert "betas" not in client.calls[0]
    assert "fallbacks" not in client.calls[0]


def test_rate_limit_becomes_answer_model_error() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(429, request=request, headers={"request-id": "req_429"})
    error = anthropic.RateLimitError("rate limited", response=response, body=None)

    class RaisingClient:
        def __init__(self) -> None:
            self.beta = SimpleNamespace(
                messages=SimpleNamespace(create=self._create),
            )

        def _create(self, **kwargs: Any) -> Any:
            raise error

    with pytest.raises(AnswerModelError) as caught:
        ClaudeAnswerModel(Settings(), client=RaisingClient()).generate("s", [], "q")
    assert caught.value.status_code == 429
    assert caught.value.request_id == "req_429"


def test_parse_reply_ignores_non_text_blocks() -> None:
    reply = parse_reply(_response())
    assert [segment.text for segment in reply.segments] == ["Employees get 20 working days."]


def test_wire_request_through_real_sdk() -> None:
    captured: list[httpx2.Request] = []
    payload = {
        "id": "msg_wire",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [
            {"type": "thinking", "thinking": "hidden", "signature": "sig"},
            {
                "type": "text",
                "text": "Answer.",
                "citations": [
                    {
                        "type": "content_block_location",
                        "cited_text": "cited",
                        "document_index": 0,
                        "document_title": "Doc",
                        "start_block_index": 0,
                        "end_block_index": 1,
                    }
                ],
            },
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        return httpx2.Response(
            200,
            headers={"request-id": "req_wire"},
            json=payload,
            request=request,
        )

    client = anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    settings = Settings()
    documents = [
        {
            "type": "document",
            "source": {
                "type": "content",
                "content": [{"type": "text", "text": "body"}],
            },
            "title": "Doc",
            "citations": {"enabled": True},
        }
    ]
    reply = ClaudeAnswerModel(settings, client=client).generate("system", documents, "question")
    assert len(captured) == 1
    request = captured[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages?beta=true"
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    import json

    body = json.loads(request.content)
    assert set(body) == {
        "model",
        "max_tokens",
        "system",
        "messages",
        "output_config",
        "fallbacks",
    }
    assert body["output_config"] == {"effort": "medium"}
    assert body["messages"][0]["content"][-1] == {
        "type": "text",
        "text": "question",
    }
    assert reply.request_id == "req_wire"
    assert len(reply.segments[0].citations) == 1
