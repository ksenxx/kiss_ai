# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: OpenRouter reasoning fields must reveal thinking tokens.

OpenRouter normalizes reasoning output across providers into
``delta.reasoning`` (a plain string) and ``delta.reasoning_details``
(a list of ``reasoning.text`` / ``reasoning.summary`` objects) on Chat
Completions streaming deltas.  ``delta.reasoning_content`` is only the
DeepSeek-native field; OpenRouter does NOT send it.

Before the fix, :class:`OpenAICompatibleModel` probed ONLY
``delta.reasoning_content``, so every openrouter/* reasoning model
(kimi-k2-thinking, deepseek v3.1+, qwen :thinking, grok, glm, ...)
silently dropped its reasoning: no thinking_callback bracket, no
thinking tokens in the UI.

Contract locked in by these tests (both ``generate()`` and the agentic
``generate_and_process_with_tools`` streaming loop):

* ``delta.reasoning`` strings stream as thinking tokens with a
  True/False thinking_callback bracket.
* ``delta.reasoning_details`` text is used when the string fields are
  absent.
* When BOTH ``delta.reasoning`` and ``delta.reasoning_details`` carry
  the same text (OpenRouter sends both), the text is emitted exactly
  once — no double emission.
* The thinking bracket closes before tool calls are processed.

Uses a real ThreadingHTTPServer streaming real SSE — no mocks, patches,
fakes, or test doubles.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

_MODEL = "openrouter/moonshotai/kimi-k2-thinking"


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
    """Build one chat.completion.chunk SSE data payload."""
    return json.dumps(
        {
            "id": "chatcmpl-or",
            "object": "chat.completion.chunk",
            "model": "moonshotai/kimi-k2-thinking",
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
    )


def _usage_chunk() -> str:
    return json.dumps(
        {
            "id": "chatcmpl-or",
            "object": "chat.completion.chunk",
            "model": "moonshotai/kimi-k2-thinking",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        }
    )


def _chunks_reasoning_string() -> list[str]:
    """delta.reasoning (OpenRouter string field) then normal content."""
    return [
        _chunk({"role": "assistant", "content": "", "reasoning": "OR thinks"}),
        _chunk({"reasoning": " deeply."}),
        _chunk({"content": "The answer is 42."}),
        _chunk({}, finish_reason="stop"),
        _usage_chunk(),
    ]


def _chunks_reasoning_details_only() -> list[str]:
    """Only delta.reasoning_details entries carry the reasoning text."""
    return [
        _chunk(
            {
                "role": "assistant",
                "content": "",
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "RT1", "format": "unknown"}
                ],
            }
        ),
        _chunk(
            {
                "reasoning_details": [
                    {
                        "type": "reasoning.summary",
                        "summary": " RS2",
                        "format": "unknown",
                    }
                ]
            }
        ),
        _chunk({"content": "done"}),
        _chunk({}, finish_reason="stop"),
        _usage_chunk(),
    ]


def _chunks_reasoning_duplicated() -> list[str]:
    """OpenRouter sends BOTH delta.reasoning and reasoning_details."""
    return [
        _chunk(
            {
                "role": "assistant",
                "content": "",
                "reasoning": "DUPTEXT",
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "DUPTEXT", "format": "unknown"}
                ],
            }
        ),
        _chunk({"content": "done"}),
        _chunk({}, finish_reason="stop"),
        _usage_chunk(),
    ]


def _chunks_reasoning_then_tool_call() -> list[str]:
    """Reasoning string then a tool_calls delta — bracket must close first."""
    return [
        _chunk({"role": "assistant", "content": "", "reasoning": "Plan the call."}),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "dummy",
                            "arguments": "{}",
                        },
                    }
                ]
            }
        ),
        _chunk({}, finish_reason="tool_calls"),
        _usage_chunk(),
    ]


class _OpenRouterHandler(BaseHTTPRequestHandler):
    """Serves SSE streaming responses shaped like OpenRouter's."""

    mode: str = "reasoning_string"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        chunks = {
            "reasoning_string": _chunks_reasoning_string,
            "reasoning_details_only": _chunks_reasoning_details_only,
            "reasoning_duplicated": _chunks_reasoning_duplicated,
            "reasoning_then_tool_call": _chunks_reasoning_then_tool_call,
        }[self.mode]()

        for chunk in chunks:
            self.wfile.write(f"data: {chunk}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture(scope="module")
def openrouter_server() -> Generator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenRouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def _make_model(
    base_url: str,
    tokens: list[str],
    thinking_events: list[bool],
) -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        _MODEL,
        base_url=base_url,
        api_key="test-key",
        token_callback=tokens.append,
        thinking_callback=thinking_events.append,
    )


def _dummy() -> str:
    """Dummy tool.

    Returns:
        A fixed marker string.
    """
    return "ok"


class TestOpenRouterReasoningString:
    """delta.reasoning must stream as thinking tokens."""

    def test_generate_streams_reasoning_string(
        self, openrouter_server: str
    ) -> None:
        """generate(): delta.reasoning brackets thinking and streams text."""
        _OpenRouterHandler.mode = "reasoning_string"
        tokens: list[str] = []
        thinking_events: list[bool] = []
        m = _make_model(openrouter_server, tokens, thinking_events)
        m.initialize("Think.")
        content, _resp = m.generate()

        assert True in thinking_events, (
            "delta.reasoning (OpenRouter) was dropped — no thinking tokens"
        )
        assert False in thinking_events
        assert thinking_events.index(True) < (
            len(thinking_events) - 1 - thinking_events[::-1].index(False)
        )
        assert "OR thinks deeply." in "".join(tokens)
        assert "The answer is 42." in content

    def test_tool_loop_streams_reasoning_string(
        self, openrouter_server: str
    ) -> None:
        """Agentic tool loop: delta.reasoning brackets thinking too."""
        _OpenRouterHandler.mode = "reasoning_string"
        tokens: list[str] = []
        thinking_events: list[bool] = []
        m = _make_model(openrouter_server, tokens, thinking_events)
        m.initialize("Think.")
        _fc, content, _resp = m.generate_and_process_with_tools({"dummy": _dummy})

        assert True in thinking_events
        assert False in thinking_events
        assert "OR thinks deeply." in "".join(tokens)
        assert "The answer is 42." in content


class TestOpenRouterReasoningDetails:
    """delta.reasoning_details must be used when string fields are absent."""

    def test_generate_streams_reasoning_details(
        self, openrouter_server: str
    ) -> None:
        """reasoning.text and reasoning.summary detail entries stream."""
        _OpenRouterHandler.mode = "reasoning_details_only"
        tokens: list[str] = []
        thinking_events: list[bool] = []
        m = _make_model(openrouter_server, tokens, thinking_events)
        m.initialize("Think.")
        content, _resp = m.generate()

        assert True in thinking_events
        assert False in thinking_events
        combined = "".join(tokens)
        assert "RT1" in combined
        assert "RS2" in combined
        assert content == "done"

    def test_tool_loop_streams_reasoning_details(
        self, openrouter_server: str
    ) -> None:
        """Tool loop also reads reasoning_details entries."""
        _OpenRouterHandler.mode = "reasoning_details_only"
        tokens: list[str] = []
        thinking_events: list[bool] = []
        m = _make_model(openrouter_server, tokens, thinking_events)
        m.initialize("Think.")
        m.generate_and_process_with_tools({"dummy": _dummy})

        assert True in thinking_events
        assert "RT1" in "".join(tokens)


class TestOpenRouterNoDoubleEmission:
    """reasoning + reasoning_details with the same text emit exactly once."""

    def test_duplicate_text_emitted_once(self, openrouter_server: str) -> None:
        """String field wins; details must not re-emit the same text."""
        _OpenRouterHandler.mode = "reasoning_duplicated"
        tokens: list[str] = []
        thinking_events: list[bool] = []
        m = _make_model(openrouter_server, tokens, thinking_events)
        m.initialize("Think.")
        m.generate()

        combined = "".join(tokens)
        assert combined.count("DUPTEXT") == 1, (
            f"reasoning text double-emitted: {combined!r}"
        )


class TestOpenRouterReasoningBeforeToolCall:
    """The thinking bracket must close before tool calls are parsed."""

    def test_bracket_closes_then_tool_call_parsed(
        self, openrouter_server: str
    ) -> None:
        """Reasoning followed by a tool_calls delta closes the bracket."""
        _OpenRouterHandler.mode = "reasoning_then_tool_call"
        tokens: list[str] = []
        thinking_events: list[bool] = []
        m = _make_model(openrouter_server, tokens, thinking_events)
        m.initialize("Think.")
        fc, _content, _resp = m.generate_and_process_with_tools({"dummy": _dummy})

        assert thinking_events == [True, False]
        assert "Plan the call." in "".join(tokens)
        assert fc and fc[0]["name"] == "dummy"
