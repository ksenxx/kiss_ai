# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: Responses-API requests must opt in to reasoning summaries.

OpenAI's Responses API returns reasoning items with EMPTY ``summary``
arrays — and emits zero ``response.reasoning_summary_text.delta`` stream
events — unless the request explicitly opts in with
``reasoning: {"summary": "auto"}``.  Without the opt-in, gpt-5.x /
o-series agentic runs never reveal a single thinking token even though
the stream consumer handles the summary events correctly.

This is the exact analogue of the Anthropic claude-fable-5 bug where
``thinking: {"type": "adaptive"}`` without ``display: "summarized"``
silently produced empty thinking blocks.

Contract locked in by these tests:

* Whenever a ``reasoning.effort`` is sent (default from MODEL_INFO
  ``thinking`` level, explicit ``reasoning_effort``, or a caller-native
  ``reasoning: {"effort": ...}`` dict), the wire request must also carry
  ``reasoning.summary`` (default ``"auto"``).
* A caller-supplied ``reasoning.summary`` always wins over the default.
* Models without any reasoning effort must NOT get a ``reasoning`` dict.
* The v1 → v2 Responses delegation path inherits the same default.
* Once the request opts in, streamed summary deltas flow through
  thinking_callback/token_callback end to end.

Uses a real ThreadingHTTPServer capturing the wire JSON — no mocks,
patches, fakes, or test doubles.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2


def _text_response_json(text: str = "ok") -> str:
    """Return a minimal /v1/responses non-streaming JSON body."""
    return json.dumps(
        {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5.5",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": text, "annotations": []}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 5,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens": 4,
                "output_tokens_details": {"reasoning_tokens": 1},
                "total_tokens": 9,
            },
        }
    )


def _stream_sse_event(event: str, data: dict[str, Any]) -> bytes:
    """Format ``event`` + ``data`` as one SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _reasoning_stream_sse_body(thinking: str, text: str = "answer") -> bytes:
    """Build a SSE body containing reasoning summary deltas then text."""
    frames = [
        _stream_sse_event(
            "response.reasoning_summary_text.delta",
            {
                "type": "response.reasoning_summary_text.delta",
                "sequence_number": 1,
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": 0,
                "delta": thinking,
            },
        ),
        _stream_sse_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 2,
                "item_id": "msg_1",
                "output_index": 1,
                "content_index": 0,
                "delta": text,
                "logprobs": [],
            },
        ),
        _stream_sse_event(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": 3,
                "response": {
                    "id": "resp_r",
                    "object": "response",
                    "created_at": 0,
                    "model": "gpt-5.5",
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_1",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": text,
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 3,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 2,
                        "output_tokens_details": {"reasoning_tokens": 5},
                        "total_tokens": 10,
                    },
                },
            },
        ),
    ]
    return b"".join(frames)


class _CapturingHandler(BaseHTTPRequestHandler):
    """Captures every POST body and returns a configurable stock response."""

    captured_requests: list[dict[str, Any]] = []
    next_response_body: bytes = b""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {}
        self.__class__.captured_requests.append(
            {"path": self.path, "body": body}
        )

        streaming = bool(body.get("stream"))
        payload = self.__class__.next_response_body
        if not payload:
            payload = _text_response_json().encode()
            streaming_payload_is_json = True
        else:
            streaming_payload_is_json = False
        content_type = (
            "text/event-stream"
            if streaming and not streaming_payload_is_json
            else "application/json"
        )
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        if content_type == "application/json":
            self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@pytest.fixture
def capture_server() -> Generator[str]:
    """Spawn an in-process HTTP server that captures requests."""
    _CapturingHandler.captured_requests = []
    _CapturingHandler.next_response_body = b""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CapturingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()


def _last_body() -> dict[str, Any]:
    """Convenience: return the last captured JSON body."""
    assert _CapturingHandler.captured_requests, "no request reached the server"
    body: dict[str, Any] = _CapturingHandler.captured_requests[-1]["body"]
    return body


def _echo(text: str) -> str:
    """Echo back ``text`` (test-only tool stub).

    Args:
        text: The string to echo.

    Returns:
        The input string unchanged.
    """
    return text


_ECHO_TOOL_CHAT_SCHEMA: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo back the given text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }
]


class TestReasoningSummaryDefault:
    """Every request carrying reasoning.effort must also opt in to summaries."""

    def test_default_effort_gets_summary_auto(self, capture_server: str) -> None:
        """gpt-5.5 (thinking level in MODEL_INFO) → reasoning.summary=auto."""
        m = OpenAICompatibleModel2(
            "gpt-5.5", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        assert body.get("reasoning") == {"effort": "high", "summary": "auto"}, (
            "Without reasoning.summary the API returns EMPTY reasoning "
            "summaries and no thinking tokens are ever revealed"
        )

    def test_tools_request_gets_summary_auto(self, capture_server: str) -> None:
        """Agentic turns (tools attached) must also carry summary=auto."""
        m = OpenAICompatibleModel2(
            "gpt-5.5-xhigh", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json().encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        body = _last_body()
        assert body.get("tools"), "tools must be on the wire"
        assert body.get("reasoning") == {"effort": "xhigh", "summary": "auto"}

    def test_caller_supplied_summary_wins(self, capture_server: str) -> None:
        """An explicit reasoning.summary from model_config is never overridden."""
        m = OpenAICompatibleModel2(
            "gpt-5.5",
            base_url=capture_server,
            api_key="k",
            model_config={
                "reasoning_effort": "low",
                "reasoning": {"summary": "detailed"},
            },
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        assert body.get("reasoning") == {"effort": "low", "summary": "detailed"}

    def test_native_reasoning_effort_shape_gets_summary(
        self, capture_server: str
    ) -> None:
        """Caller-native ``reasoning: {"effort": ...}`` also gets summary=auto.

        gpt-4o has no default thinking level, so the only reasoning config
        is the caller's native dict — the summary default must still apply.
        """
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={"reasoning": {"effort": "high"}},
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        assert body.get("reasoning") == {"effort": "high", "summary": "auto"}

    def test_caller_config_not_mutated(self, capture_server: str) -> None:
        """Adding the summary default must not mutate the caller's dict."""
        native = {"effort": "high"}
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={"reasoning": native},
        )
        m.initialize("hi")
        m.generate()
        assert native == {"effort": "high"}, "caller config was mutated"

    def test_non_reasoning_model_gets_no_reasoning_dict(
        self, capture_server: str
    ) -> None:
        """gpt-4o without any reasoning config must not send reasoning at all."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        m.generate()
        assert "reasoning" not in _last_body()


class TestV1DelegationInheritsSummary:
    """v1 agentic runs delegated to /responses must carry summary=auto too."""

    def test_delegated_request_has_summary_auto(self, capture_server: str) -> None:
        """OpenAICompatibleModel with use_responses_api → summary on the wire."""
        _CapturingHandler.next_response_body = _text_response_json().encode()
        m = OpenAICompatibleModel(
            "gpt-5.5-xhigh",
            base_url=capture_server,
            api_key="k",
            model_config={"use_responses_api": True},
        )
        m.initialize("hi")

        m.generate_and_process_with_tools({"echo": _echo})
        body = _last_body()
        assert body.get("reasoning", {}).get("summary") == "auto", (
            "v1 → v2 delegation lost the reasoning summary opt-in"
        )
        assert body.get("reasoning", {}).get("effort") == "xhigh"


class TestSummaryStreamEndToEnd:
    """Full pipe: request opts in AND streamed summaries reach the callbacks."""

    def test_stream_reveals_thinking_and_request_opted_in(
        self, capture_server: str
    ) -> None:
        """Thinking tokens flow end-to-end once the request carries summary."""
        tokens: list[str] = []
        thinking_events: list[bool] = []
        m = OpenAICompatibleModel2(
            "gpt-5.5",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
            thinking_callback=thinking_events.append,
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _reasoning_stream_sse_body(
            thinking="THINK-DELTA", text="the answer"
        )
        text, _resp = m.generate()
        assert text == "the answer"
        assert thinking_events[:1] == [True]
        assert thinking_events[-1] is False
        assert "THINK-DELTA" in "".join(tokens)
        # The regression half: the request itself must have opted in.
        body = _last_body()
        assert body.get("reasoning", {}).get("summary") == "auto"
