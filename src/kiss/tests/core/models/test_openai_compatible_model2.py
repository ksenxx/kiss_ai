# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for :class:`OpenAICompatibleModel2` (Responses API).

A real :class:`http.server.ThreadingHTTPServer` captures every request
sent by the model so the tests can assert on the exact JSON that travels
over the wire to OpenAI's ``/v1/responses`` endpoint.  No mocks, patches,
fakes, or test doubles are used to substitute for the OpenAI SDK — only
the upstream HTTP endpoint is replaced by the in-process capture server.

The core contract being verified:

* The new model targets ``POST /v1/responses`` (not ``/v1/chat/completions``).
* ``reasoning_effort`` from ``model_config`` is rewritten to ``reasoning.effort``.
* For models that declare ``thinking="xhigh"`` in ``MODEL_INFO.json``
  (the gpt-5.5 family), the default ``reasoning.effort`` is ``"xhigh"``.
* Tools and ``reasoning.effort`` MUST coexist on the wire — this is the
  whole reason the v2 model was introduced (Chat Completions rejects the
  combination for GPT-5 reasoning models).
* Tool schemas are emitted in the flat Responses-API shape
  (``{"type":"function","name":...,"parameters":...}``) — not the nested
  Chat-Completions shape.
* ``system_instruction`` is routed to the top-level ``instructions`` field,
  not into the conversation.
* Attachments map to ``input_image`` / ``input_file`` content parts.
* Function results are appended as ``function_call_output`` input items
  carrying the matching ``call_id``.
* Token-usage extraction reads ``usage.input_tokens``,
  ``usage.output_tokens``, ``input_tokens_details.cached_tokens``, and
  counts ``output_tokens_details.reasoning_tokens`` as output tokens.
* ``openrouter/anthropic/*`` models receive top-level
  ``extra_body["cache_control"]`` (Anthropic prompt caching).
* Models not in ``MODEL_INFO`` get no ``reasoning`` parameter.
* DeepSeek R1 models fall back to text-based tool calling.
* Embeddings still hit ``/v1/embeddings``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.core.models.model import Attachment
from kiss.core.models.model_info import MODEL_INFO, model
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2

# ---------------------------------------------------------------------------
# Fake server: captures every POST body and returns a stock response for
# /v1/responses (and /v1/embeddings).  Streaming and non-streaming are
# differentiated by the captured request body's ``"stream"`` key.
# ---------------------------------------------------------------------------


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


def _tool_call_response_json(
    name: str = "echo",
    arguments: str = '{"text": "hello"}',
    call_id: str = "call_abc",
) -> str:
    """Return a minimal /v1/responses JSON body containing a function_call."""
    return json.dumps(
        {
            "id": "resp_tc",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5.5",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                }
            ],
            "usage": {
                "input_tokens": 5,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 4,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 9,
            },
        }
    )


def _stream_sse_event(event: str, data: dict[str, Any]) -> bytes:
    """Format ``event`` + ``data`` as one SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _text_stream_sse_body(text: str = "hello") -> bytes:
    """Build a complete SSE body that streams a single text delta."""
    frames = b"".join(
        [
            _stream_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                    "logprobs": [],
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
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
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 4,
                        },
                    },
                },
            ),
        ]
    )
    return frames


def _tool_call_stream_sse_body(
    name: str = "echo",
    args_text: str = '{"text":"hello"}',
    call_id: str = "call_xyz",
) -> bytes:
    """Build a complete SSE body that streams a function_call."""
    output_item = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": call_id,
        "name": name,
        "arguments": "",
    }
    frames = [
        _stream_sse_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": output_item,
            },
        )
    ]
    # Stream arguments one chunk at a time.
    for ch in args_text:
        frames.append(
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": ch,
                },
            )
        )
    completed_item = dict(output_item)
    completed_item["arguments"] = args_text
    frames.append(
        _stream_sse_event(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": 99,
                "response": {
                    "id": "resp_tc",
                    "object": "response",
                    "created_at": 0,
                    "model": "gpt-5.5",
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                    "output": [completed_item],
                    "usage": {
                        "input_tokens": 3,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 5,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 8,
                    },
                },
            },
        )
    )
    return b"".join(frames)


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


def _embedding_response_json() -> str:
    return json.dumps(
        {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
            ],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    )


class _CapturingHandler(BaseHTTPRequestHandler):
    """Captures every POST body and returns a configurable stock response."""

    captured_requests: list[dict[str, Any]] = []
    # Each entry is (status, headers, body_bytes) — caller can override.
    next_response_body: bytes = b""
    next_response_headers: dict[str, str] = {}

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {}
        self.__class__.captured_requests.append(
            {"path": self.path, "body": body, "raw": raw}
        )

        path = self.path
        if path.endswith("/embeddings"):
            payload = _embedding_response_json().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        # /v1/responses
        streaming = bool(body.get("stream"))
        payload = self.__class__.next_response_body
        if not payload:
            payload = (
                _text_stream_sse_body() if streaming else _text_response_json().encode()
            )
        content_type = (
            "text/event-stream" if streaming else "application/json"
        )
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        if not streaming:
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


def _last_path() -> str:
    path: str = _CapturingHandler.captured_requests[-1]["path"]
    return path


# ---------------------------------------------------------------------------
# Tool stubs used in tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEndpointAndDefaults:
    """The model must POST to /v1/responses and default xhigh for gpt-5.5."""

    def test_generate_posts_to_responses_endpoint(
        self, capture_server: str
    ) -> None:
        """Plain generate() must POST to /v1/responses (not /chat/completions)."""
        m = OpenAICompatibleModel2(
            "gpt-5.5", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        text, _resp = m.generate()
        assert text == "ok"
        assert _last_path().endswith("/responses")

    def test_gpt_5_5_defaults_reasoning_effort_high(
        self, capture_server: str
    ) -> None:
        """gpt-5.5 base entry defaults to ``reasoning.effort='high'``.

        After the xhigh-split refactor the base entry is capped at
        ``high``; the uncapped ``xhigh`` level lives on the synthetic
        ``gpt-5.5-xhigh`` sibling.
        """
        m = OpenAICompatibleModel2(
            "gpt-5.5", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        assert body.get("reasoning", {}).get("effort") == "high"
        assert "reasoning_effort" not in body, (
            "Responses API uses nested reasoning.effort, not flat reasoning_effort"
        )
        assert body.get("model") == "gpt-5.5"

    def test_gpt_5_5_xhigh_alias_routes_to_base_with_xhigh(
        self, capture_server: str
    ) -> None:
        """The xhigh alias must POST the base model id with xhigh effort."""
        m = OpenAICompatibleModel2(
            "gpt-5.5-xhigh", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        assert body.get("reasoning", {}).get("effort") == "xhigh"
        assert body.get("model") == "gpt-5.5"

    def test_dated_gpt_5_5_defaults_high(self, capture_server: str) -> None:
        """Dated alias gpt-5.5-2026-04-23 also defaults to high."""
        m = OpenAICompatibleModel2(
            "gpt-5.5-2026-04-23", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        m.generate()
        assert _last_body().get("reasoning", {}).get("effort") == "high"

    def test_dated_gpt_5_5_xhigh_alias_routes_to_base(
        self, capture_server: str
    ) -> None:
        """gpt-5.5-2026-04-23-xhigh sends gpt-5.5-2026-04-23 with xhigh effort."""
        m = OpenAICompatibleModel2(
            "gpt-5.5-2026-04-23-xhigh", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        assert body.get("reasoning", {}).get("effort") == "xhigh"
        assert body.get("model") == "gpt-5.5-2026-04-23"

    def test_explicit_reasoning_effort_wins(self, capture_server: str) -> None:
        """A caller-supplied reasoning_effort overrides the xhigh default."""
        m = OpenAICompatibleModel2(
            "gpt-5.5",
            base_url=capture_server,
            api_key="k",
            model_config={"reasoning_effort": "high"},
        )
        m.initialize("hi")
        m.generate()
        assert _last_body().get("reasoning", {}).get("effort") == "high"

    def test_caller_config_not_mutated(self, capture_server: str) -> None:
        """The caller's model_config dict must NOT be mutated."""
        cfg: dict[str, object] = {}
        OpenAICompatibleModel2(
            "gpt-5.5", base_url=capture_server, api_key="k", model_config=cfg
        )
        assert cfg == {}

    def test_openrouter_gpt_5_5_defaults_high(self, capture_server: str) -> None:
        """openrouter/openai/gpt-5.5 (after openrouter/ strip) defaults to high."""
        m = OpenAICompatibleModel2(
            "openrouter/openai/gpt-5.5", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        assert body.get("reasoning", {}).get("effort") == "high"
        # The model name sent on the wire must be the bare alias (no
        # ``openrouter/`` prefix), matching v1's behaviour.
        assert body.get("model") == "openai/gpt-5.5"

    def test_openrouter_gpt_5_5_xhigh_alias_routes_to_base(
        self, capture_server: str
    ) -> None:
        """openrouter/openai/gpt-5.5-xhigh sends openai/gpt-5.5 with xhigh."""
        m = OpenAICompatibleModel2(
            "openrouter/openai/gpt-5.5-xhigh",
            base_url=capture_server,
            api_key="k",
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        assert body.get("reasoning", {}).get("effort") == "xhigh"
        assert body.get("model") == "openai/gpt-5.5"

    def test_openrouter_gpt_latest_alias_defaults_high(
        self, capture_server: str
    ) -> None:
        """openrouter/~openai/gpt-latest also defaults to high."""
        m = OpenAICompatibleModel2(
            "openrouter/~openai/gpt-latest",
            base_url=capture_server,
            api_key="k",
        )
        m.initialize("hi")
        m.generate()
        assert _last_body().get("reasoning", {}).get("effort") == "high"

    def test_gpt_5_does_not_default_xhigh(self, capture_server: str) -> None:
        """gpt-5 caps at high; no xhigh default."""
        m = OpenAICompatibleModel2(
            "gpt-5", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        assert body.get("reasoning", {}).get("effort") != "xhigh"

    def test_gpt_4o_does_not_send_reasoning(self, capture_server: str) -> None:
        """Non-reasoning models must not send a ``reasoning`` field at all."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        assert "reasoning" not in body

    def test_unknown_model_does_not_default(self, capture_server: str) -> None:
        """An unknown model name must not get reasoning auto-added."""
        m = OpenAICompatibleModel2(
            "some-custom-local-model", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        m.generate()
        assert "reasoning" not in _last_body()

    def test_model_info_flag_drives_defaulting(self, capture_server: str) -> None:
        """Flipping ``thinking`` in MODEL_INFO immediately changes the default."""
        original = MODEL_INFO["gpt-4o"].thinking
        MODEL_INFO["gpt-4o"].thinking = "xhigh"
        try:
            m = OpenAICompatibleModel2(
                "gpt-4o", base_url=capture_server, api_key="k"
            )
            m.initialize("hi")
            m.generate()
            body = _last_body()
            assert body.get("reasoning", {}).get("effort") == "xhigh"
        finally:
            MODEL_INFO["gpt-4o"].thinking = original


class TestToolsCoexistWithReasoning:
    """The whole point of v2: tools + reasoning.effort coexist."""

    def test_tools_and_xhigh_coexist(self, capture_server: str) -> None:
        """gpt-5.5-xhigh + tools must send BOTH tools and reasoning.effort=xhigh."""
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
        assert body.get("tools"), "tools must be sent on the wire"
        assert body.get("reasoning", {}).get("effort") == "xhigh", (
            "v2 must keep reasoning.effort even when tools are present"
        )
        assert body.get("model") == "gpt-5.5"

    def test_tools_schema_is_flattened_for_responses_api(
        self, capture_server: str
    ) -> None:
        """Chat-Completions nested tool schema → flat Responses tool schema."""
        m = OpenAICompatibleModel2(
            "gpt-5.5", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json().encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        tools = _last_body()["tools"]
        assert tools and isinstance(tools, list)
        first = tools[0]
        assert first["type"] == "function"
        assert first["name"] == "echo"
        assert first["parameters"]["type"] == "object"
        assert "function" not in first, (
            "Responses-API tools must be flat; no nested 'function' key"
        )


class TestConversationShape:
    """``input`` and ``instructions`` use the Responses-API contract."""

    def test_system_instruction_routed_to_instructions(
        self, capture_server: str
    ) -> None:
        """system_instruction from model_config → top-level ``instructions``."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={"system_instruction": "you are a helper"},
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        assert body.get("instructions") == "you are a helper"
        # The system message must NOT be duplicated inside ``input``.
        input_items = body.get("input", [])
        for item in input_items:
            if isinstance(item, dict):
                assert item.get("role") != "system"

    def test_initial_user_prompt_in_input(self, capture_server: str) -> None:
        """The initial user prompt lands in ``input`` as a user message."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hello world")
        m.generate()
        body = _last_body()
        input_items = body["input"]
        assert any(
            isinstance(it, dict) and it.get("role") == "user" for it in input_items
        )

    def test_image_attachment_becomes_input_image(
        self, capture_server: str
    ) -> None:
        """Image attachment → ``input_image`` content part."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        att = Attachment(data=b"\x89PNG\r\n\x1a\n", mime_type="image/png")
        m.initialize("look", attachments=[att])
        m.generate()
        body = _last_body()
        user_msg = next(
            it for it in body["input"]
            if isinstance(it, dict) and it.get("role") == "user"
        )
        parts = user_msg["content"]
        types = {p.get("type") for p in parts if isinstance(p, dict)}
        assert "input_image" in types
        assert "input_text" in types

    def test_pdf_attachment_becomes_input_file(self, capture_server: str) -> None:
        """PDF attachment → ``input_file`` content part."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        att = Attachment(data=b"%PDF-1.4\n", mime_type="application/pdf")
        m.initialize("read", attachments=[att])
        m.generate()
        body = _last_body()
        user_msg = next(
            it for it in body["input"]
            if isinstance(it, dict) and it.get("role") == "user"
        )
        parts = user_msg["content"]
        types = {p.get("type") for p in parts if isinstance(p, dict)}
        assert "input_file" in types

    def test_unsupported_attachment_is_dropped(
        self, capture_server: str
    ) -> None:
        """Unsupported MIME types must NOT crash and must NOT be sent."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        att = Attachment(data=b"\x00\x00", mime_type="video/mp4")
        m.initialize("look", attachments=[att])
        m.generate()
        body = _last_body()
        user_msg = next(
            it for it in body["input"]
            if isinstance(it, dict) and it.get("role") == "user"
        )
        parts = user_msg["content"]
        for p in parts:
            assert isinstance(p, dict)
            assert p["type"] != "video"


class TestToolCallParsing:
    """Function calls returned by the API are parsed and stored properly."""

    def test_function_call_parsed_non_streaming(
        self, capture_server: str
    ) -> None:
        """A non-streaming function_call is parsed into function_calls."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _tool_call_response_json(
            name="echo",
            arguments='{"text": "hello"}',
            call_id="call_abc",
        ).encode()
        fcs, content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_abc", "name": "echo", "arguments": {"text": "hello"}}
        ]
        # The function_call item is appended to the conversation.
        fc_items = [
            x for x in m.conversation
            if isinstance(x, dict) and x.get("type") == "function_call"
        ]
        assert fc_items
        assert fc_items[-1]["call_id"] == "call_abc"
        assert fc_items[-1]["name"] == "echo"
        # content is the text portion (empty here).
        assert content == ""

    def test_function_results_become_function_call_output(
        self, capture_server: str
    ) -> None:
        """add_function_results_to_conversation_and_return → function_call_output."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _tool_call_response_json(
            name="echo",
            arguments='{"text": "hello"}',
            call_id="call_abc",
        ).encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "hello"})]
        )
        outputs = [
            x for x in m.conversation
            if isinstance(x, dict) and x.get("type") == "function_call_output"
        ]
        assert outputs
        assert outputs[-1]["call_id"] == "call_abc"
        assert outputs[-1]["output"] == "hello"

    def test_function_results_with_binary_attachment_lifted(
        self, capture_server: str
    ) -> None:
        """Binary attachment in tool result → follow-up user msg with input_image."""
        from kiss.core.models.model import encode_binary_attachment

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _tool_call_response_json(
            name="echo", arguments='{"text":"x"}', call_id="call_abc"
        ).encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        # Make a "tool result" string carrying a fake PNG.
        payload = encode_binary_attachment("image/png", b"\x89PNG\r\n\x1a\n")
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": f"see image: {payload}"})]
        )
        # function_call_output stripped of binary bytes:
        outputs = [
            x for x in m.conversation
            if isinstance(x, dict) and x.get("type") == "function_call_output"
        ]
        assert outputs
        assert "KISS_BINARY" not in outputs[-1]["output"]
        # Follow-up user message with input_image part:
        user_msgs = [
            x for x in m.conversation
            if isinstance(x, dict) and x.get("role") == "user"
        ]
        # The last user message must carry an input_image part.
        last_user_parts = user_msgs[-1]["content"]
        assert any(
            isinstance(p, dict) and p.get("type") == "input_image"
            for p in last_user_parts
        )


class TestStreaming:
    """Streaming events drive token/thinking callbacks."""

    def test_text_streaming_invokes_token_callback(
        self, capture_server: str
    ) -> None:
        """response.output_text.delta events are forwarded to token_callback."""
        tokens: list[str] = []
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_stream_sse_body("hello")
        text, _resp = m.generate()
        assert "hello" in tokens
        assert text == "hello"
        # The request must have actually streamed.
        body = _last_body()
        assert body.get("stream") is True

    def test_reasoning_summary_streaming_invokes_thinking_callback(
        self, capture_server: str
    ) -> None:
        """reasoning summary deltas bracket a True/False thinking_callback pair."""
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
            thinking="thinking...", text="answer"
        )
        text, _resp = m.generate()
        assert text == "answer"
        # Thinking block start, then end:
        assert thinking_events[:1] == [True]
        assert thinking_events[-1] is False
        # Reasoning text was streamed via token_callback as well:
        assert "thinking..." in tokens
        assert "answer" in tokens

    def test_function_call_streaming_parsed(self, capture_server: str) -> None:
        """Streaming a function_call assembles args and parses them."""
        tokens: list[str] = []
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _tool_call_stream_sse_body(
            name="echo", args_text='{"text":"streamy"}', call_id="call_stream"
        )
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {
                "id": "call_stream",
                "name": "echo",
                "arguments": {"text": "streamy"},
            }
        ]


class TestUsageExtraction:
    """Token-usage extraction reads Responses-API usage fields."""

    def test_usage_counts_reasoning_as_output(
        self, capture_server: str
    ) -> None:
        """reasoning_tokens are folded into output_tokens for billing."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _, resp = m.generate()
        ip, op, cr, cw = m.extract_input_output_token_counts_from_response(resp)
        # default _text_response_json: input=5, cached=2, output=4, reasoning=1
        assert ip == 3  # 5 - 2 (cached)
        # The output count must include the reasoning tokens already
        # reported by the Responses API (output_tokens already counts them).
        assert op == 4
        assert cr == 2
        assert cw == 0

    def test_usage_handles_missing_details(self, capture_server: str) -> None:
        """Missing details blocks must default to zero, not crash."""

        class _Usage:
            def __init__(self) -> None:
                self.input_tokens = 7
                self.output_tokens = 3
                self.input_tokens_details = None
                self.output_tokens_details = None

        class _Resp:
            def __init__(self) -> None:
                self.usage = _Usage()

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        ip, op, cr, cw = m.extract_input_output_token_counts_from_response(_Resp())
        assert ip == 7 and op == 3 and cr == 0 and cw == 0

    def test_usage_no_usage_returns_zeros(self, capture_server: str) -> None:
        """A response without ``usage`` returns (0, 0, 0, 0)."""

        class _Resp:
            usage = None

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        assert m.extract_input_output_token_counts_from_response(_Resp()) == (
            0, 0, 0, 0,
        )


class TestEmbedding:
    """Embeddings still hit the /v1/embeddings endpoint."""

    def test_get_embedding(self, capture_server: str) -> None:
        """get_embedding returns the embedding vector from the response."""
        m = OpenAICompatibleModel2(
            "text-embedding-3-small", base_url=capture_server, api_key="k"
        )
        m.initialize("ignored")
        vec = m.get_embedding("hello")
        assert vec == [0.1, 0.2, 0.3]


class TestOpenRouterAnthropicCacheControl:
    """openrouter/anthropic/* models must send extra_body.cache_control."""

    def test_cache_control_added(self, capture_server: str) -> None:
        m = OpenAICompatibleModel2(
            "openrouter/anthropic/claude-opus-4-6",
            base_url=capture_server,
            api_key="k",
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        # extra_body is merged into the top-level request by the SDK.
        assert body.get("cache_control") == {"type": "ephemeral"}

    def test_cache_control_disabled_when_enable_cache_false(
        self, capture_server: str
    ) -> None:
        m = OpenAICompatibleModel2(
            "openrouter/anthropic/claude-opus-4-6",
            base_url=capture_server,
            api_key="k",
            model_config={"enable_cache": False},
        )
        m.initialize("hi")
        m.generate()
        assert "cache_control" not in _last_body()


class TestDeepSeekTextBasedFallback:
    """DeepSeek R1 family falls back to text-based tool calling."""

    def test_deepseek_uses_text_based_tools(self, capture_server: str) -> None:
        """The function-call response from DeepSeek arrives as JSON inside text."""
        tool_payload = '{"tool_calls": [{"name": "echo", "arguments": {"text": "hi"}}]}'
        m = OpenAICompatibleModel2(
            "deepseek/deepseek-r1", base_url=capture_server, api_key="k"
        )
        m.initialize("please call echo")
        _CapturingHandler.next_response_body = _text_response_json(
            text=tool_payload
        ).encode()
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert len(fcs) == 1
        assert fcs[0]["name"] == "echo"
        assert fcs[0]["arguments"] == {"text": "hi"}
        # And the request body must NOT contain a native ``tools`` array,
        # because DeepSeek R1's native tool calling is unreliable.
        body = _last_body()
        assert "tools" not in body or not body.get("tools")


class TestExtraCoverage:
    """Tests targeting branches not exercised by the main test classes."""

    def test_repr_contains_class_and_endpoint(self, capture_server: str) -> None:
        """``__str__``/``__repr__`` must include the class, model, and base_url."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        s = str(m)
        assert "OpenAICompatibleModel2" in s
        assert "gpt-4o" in s
        assert capture_server in s

    def test_pre_flattened_tools_pass_through(self, capture_server: str) -> None:
        """A tool schema already in flat Responses shape is forwarded unchanged."""
        flat_tool = {
            "type": "function",
            "name": "echo",
            "description": "echo",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json().encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo}, tools_schema=[flat_tool]
        )
        body = _last_body()
        assert body["tools"][0]["name"] == "echo"
        assert "function" not in body["tools"][0]

    def test_assistant_text_added_when_present_with_tool_call(
        self, capture_server: str
    ) -> None:
        """An assistant message + function_call both appear in conversation."""
        body_json = json.dumps(
            {
                "id": "r",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
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
                                "text": "let me call it",
                                "annotations": [],
                            }
                        ],
                    },
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_ZZ",
                        "name": "echo",
                        "arguments": '{"text":"x"}',
                    },
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode("utf-8")
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = body_json
        fcs, content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert content == "let me call it"
        assert fcs and fcs[0]["name"] == "echo"
        roles = [
            x.get("role") for x in m.conversation if isinstance(x, dict)
        ]
        assert "assistant" in roles
        # The function_call item is also present.
        assert any(
            isinstance(x, dict) and x.get("type") == "function_call"
            for x in m.conversation
        )

    def test_function_results_without_prior_calls_raises(
        self, capture_server: str
    ) -> None:
        """Orphan ``function_call_output`` (no prior function_call) raises.

        Per the Responses-API contract, every ``function_call_output``
        MUST have a ``call_id`` matching a previously-emitted
        ``function_call`` item.  Synthesising a fallback ``call_id``
        produces an invalid conversation, so v2 raises
        :class:`KISSError` instead of silently corrupting the wire shape.
        """
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        with pytest.raises(KISSError, match="No prior function_call"):
            m.add_function_results_to_conversation_and_return(
                [("orphan", {"result": "ok"})]
            )
        assert not any(
            isinstance(x, dict) and x.get("type") == "function_call_output"
            for x in m.conversation
        )

    def test_usage_info_appended_to_function_call_output(
        self, capture_server: str
    ) -> None:
        """``set_usage_info_for_messages`` text appends to each output."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        m.set_usage_info_for_messages("[USAGE]")
        # Seed a prior function_call so the Responses-API conversation
        # contract is honored (every function_call_output must reference
        # a prior function_call.call_id).
        m.conversation.append(
            {
                "type": "function_call",
                "id": "fc_seed",
                "call_id": "call_seed",
                "name": "x",
                "arguments": "{}",
            }
        )
        m._pending_function_calls = [{"name": "x", "call_id": "call_seed"}]
        m.add_function_results_to_conversation_and_return(
            [("x", {"result": "ok"})]
        )
        out = [
            x for x in m.conversation
            if isinstance(x, dict) and x.get("type") == "function_call_output"
        ][-1]
        assert "[USAGE]" in out["output"]

    def test_get_embedding_raises_kiss_error_on_failure(
        self, capture_server: str
    ) -> None:
        """Embedding failures are wrapped in KISSError."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "text-embedding-3-small", base_url=capture_server, api_key="k"
        )
        m.initialize("ignored")
        # An obviously broken URL forces an exception in client.embeddings.create.
        m.base_url = "http://127.0.0.1:1/never/v1"
        from openai import OpenAI

        m.client = OpenAI(
            base_url=m.base_url, api_key="k", timeout=1.0
        )
        with pytest.raises(KISSError):
            m.get_embedding("x")

    def test_deepseek_tool_call_in_generate_extracts_text(
        self, capture_server: str
    ) -> None:
        """For DeepSeek, plain generate() strips <think> tags."""
        m = OpenAICompatibleModel2(
            "deepseek/deepseek-r1", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json(
            text="<think>plan</think>final answer"
        ).encode()
        text, _resp = m.generate()
        assert text == "final answer"

    def test_deepseek_with_system_instruction_and_reasoning(
        self, capture_server: str
    ) -> None:
        """DeepSeek text-based fallback also forwards instructions & reasoning."""
        m = OpenAICompatibleModel2(
            "deepseek/deepseek-r1",
            base_url=capture_server,
            api_key="k",
            model_config={
                "system_instruction": "sys-go",
                "reasoning_effort": "high",
            },
        )
        m.initialize("please call echo")
        _CapturingHandler.next_response_body = _text_response_json(
            text='{"tool_calls": [{"name": "echo", "arguments": {"text": "hi"}}]}'
        ).encode()
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        body = _last_body()
        assert body.get("instructions") == "sys-go"
        assert body.get("reasoning", {}).get("effort") == "high"
        assert len(fcs) == 1

    def test_streaming_function_call_with_reasoning_first(
        self, capture_server: str
    ) -> None:
        """A reasoning block followed by a function_call closes the thinking UI."""
        thinking_events: list[bool] = []
        tokens: list[str] = []
        m = OpenAICompatibleModel2(
            "gpt-5.5",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
            thinking_callback=thinking_events.append,
        )
        m.initialize("hi")
        # SSE order: reasoning_summary_text.delta → output_item.added
        # (function_call) → args.delta → completed.
        frames = [
            _stream_sse_event(
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "sequence_number": 1,
                    "item_id": "rs_1",
                    "output_index": 0,
                    "summary_index": 0,
                    "delta": "planning",
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 2,
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_R",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 3,
                    "item_id": "fc_1",
                    "output_index": 1,
                    "delta": '{"text":"y"}',
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-5.5",
                        "parallel_tool_calls": True,
                        "tool_choice": "auto",
                        "tools": [],
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 1},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_R", "name": "echo", "arguments": {"text": "y"}}
        ]
        # Reasoning callback opened and then closed before tool call begin.
        assert thinking_events[0] is True
        assert False in thinking_events
        assert "planning" in tokens


class TestReviewBugReproductions:
    """End-to-end tests reproducing the bugs flagged by the gpt-5.5 review."""

    def test_deepseek_tool_result_uses_prior_function_call_id(
        self, capture_server: str
    ) -> None:
        """DeepSeek fallback must store Responses-API function_call items.

        The review found that the v1 helper
        ``_replace_last_assistant_with_tool_calls`` was being used, which
        injects a Chat-Completions ``tool_calls`` shape onto the assistant
        message — invalid for the Responses API.  The fix appends real
        ``function_call`` input items so that
        ``add_function_results_to_conversation_and_return`` can match the
        produced ``call_id`` on the follow-up ``function_call_output``.
        """
        tool_payload = (
            '{"tool_calls": [{"name": "echo", "arguments": {"text": "hi"}}]}'
        )
        m = OpenAICompatibleModel2(
            "deepseek/deepseek-r1", base_url=capture_server, api_key="k"
        )
        m.initialize("please call echo")
        _CapturingHandler.next_response_body = _text_response_json(
            text=tool_payload
        ).encode()
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        fc_items = [
            x for x in m.conversation
            if isinstance(x, dict) and x.get("type") == "function_call"
        ]
        assert fc_items, "DeepSeek fallback must store Responses function_call items"
        # The assistant message must NOT carry the v1 ``tool_calls`` key.
        for msg in m.conversation:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                assert "tool_calls" not in msg

        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "hi"})]
        )
        out_items = [
            x for x in m.conversation
            if isinstance(x, dict) and x.get("type") == "function_call_output"
        ]
        assert out_items
        assert out_items[-1]["call_id"] == fc_items[-1]["call_id"]
        # And the returned function_calls reflect the same id.
        assert fcs[-1]["id"] == fc_items[-1]["call_id"]

    def test_usage_extracts_cache_write_tokens(
        self, capture_server: str
    ) -> None:
        """``input_tokens_details.cache_write_tokens`` is extracted and subtracted.

        Reproduces the v1-compat regression flagged by the review: v2
        always returned 0 for cache_write_tokens, miscounting OpenRouter
        Anthropic prompt caching.
        """

        class _Details:
            cached_tokens = 10
            cache_write_tokens = 7

        class _Usage:
            input_tokens = 100
            output_tokens = 5
            input_tokens_details = _Details()

        class _Resp:
            usage = _Usage()

        m = OpenAICompatibleModel2(
            "openrouter/anthropic/claude-opus-4-6",
            base_url=capture_server,
            api_key="k",
        )
        assert m.extract_input_output_token_counts_from_response(_Resp()) == (
            83,  # 100 - 10 - 7
            5,
            10,
            7,
        )

    def test_whitespace_only_prompt_is_rejected(
        self, capture_server: str
    ) -> None:
        """A pure-whitespace prompt must NOT be shipped to the Responses API."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("   \n  \t  ")
        with pytest.raises(KISSError):
            m.generate()
        assert not _CapturingHandler.captured_requests

    def test_empty_assistant_message_not_resent(
        self, capture_server: str
    ) -> None:
        """An empty assistant reply must not pollute the next request."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json(text="").encode()
        m.generate()
        _CapturingHandler.next_response_body = _text_response_json(text="next").encode()
        m.generate()
        body = _last_body()
        for item in body["input"]:
            if isinstance(item, dict) and item.get("role") == "assistant":
                # Either dropped or contains non-empty text only.
                content = item.get("content")
                if isinstance(content, str):
                    assert content.strip(), (
                        "Empty assistant content must be dropped from input"
                    )
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in (
                            "input_text",
                            "output_text",
                        ):
                            assert str(part.get("text", "")).strip()

    def test_streaming_function_call_uses_output_item_done_metadata(
        self, capture_server: str
    ) -> None:
        """A late-binding call_id/name on output_item.done overrides the empty added."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k", token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "",
                        "name": "",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '{"text":"late"}',
                },
            ),
            _stream_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": 3,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_late",
                        "name": "echo",
                        "arguments": '{"text":"late"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "parallel_tool_calls": True,
                        "tool_choice": "auto",
                        "tools": [],
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_late", "name": "echo", "arguments": {"text": "late"}}
        ]

    def test_audio_attachment_becomes_input_audio(
        self, capture_server: str
    ) -> None:
        """Audio attachments map to Responses-API ``input_audio`` parts."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        att = Attachment(data=b"fake mp3 bytes", mime_type="audio/mpeg")
        m.initialize("transcribe", attachments=[att])
        m.generate()
        body = _last_body()
        user_msg = next(
            it for it in body["input"]
            if isinstance(it, dict) and it.get("role") == "user"
        )
        parts = user_msg["content"]
        audio_parts = [
            p for p in parts
            if isinstance(p, dict) and p.get("type") == "input_audio"
        ]
        assert audio_parts
        assert audio_parts[0]["input_audio"]["format"] == "mp3"


class TestReviewBugReproductions2:
    """End-to-end tests reproducing bugs flagged by the second gpt-5.5 review."""

    def test_max_tokens_is_translated_to_max_output_tokens(
        self, capture_server: str
    ) -> None:
        """``max_tokens`` (Chat-Completions) must become ``max_output_tokens``."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={"max_tokens": 7},
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        assert body.get("max_output_tokens") == 7
        assert "max_tokens" not in body

    def test_flatten_tools_preserves_strict(self, capture_server: str) -> None:
        """Tool flattening must keep ``strict`` and other function-level fields."""
        schema = [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo.",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ]
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json().encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo}, tools_schema=schema
        )
        tool = _last_body()["tools"][0]
        assert tool.get("strict") is True
        assert tool.get("name") == "echo"

    def test_streaming_function_call_uses_completed_response_output_metadata(
        self, capture_server: str
    ) -> None:
        """Tool metadata in terminal completed-response output is authoritative."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "",
                        "name": "",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '{"text":"from-completed"}',
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "parallel_tool_calls": True,
                        "tool_choice": "auto",
                        "tools": [],
                        "output": [
                            {
                                "type": "function_call",
                                "id": "fc_1",
                                "call_id": "call_completed",
                                "name": "echo",
                                "arguments": '{"text":"from-completed"}',
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {
                "id": "call_completed",
                "name": "echo",
                "arguments": {"text": "from-completed"},
            }
        ]

    def test_multiple_reasoning_blocks_are_individually_bracketed(
        self, capture_server: str
    ) -> None:
        """``reasoning_summary_text.done`` closes the thinking block."""
        thinking_events: list[bool] = []
        m = OpenAICompatibleModel2(
            "gpt-5.5",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
            thinking_callback=thinking_events.append,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "sequence_number": 1,
                    "item_id": "rs_1",
                    "output_index": 0,
                    "summary_index": 0,
                    "delta": "first",
                },
            ),
            _stream_sse_event(
                "response.reasoning_summary_text.done",
                {
                    "type": "response.reasoning_summary_text.done",
                    "sequence_number": 2,
                    "item_id": "rs_1",
                    "output_index": 0,
                    "summary_index": 0,
                    "text": "first",
                },
            ),
            _stream_sse_event(
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "sequence_number": 3,
                    "item_id": "rs_2",
                    "output_index": 1,
                    "summary_index": 0,
                    "delta": "second",
                },
            ),
            _stream_sse_event(
                "response.reasoning_summary_text.done",
                {
                    "type": "response.reasoning_summary_text.done",
                    "sequence_number": 4,
                    "item_id": "rs_2",
                    "output_index": 1,
                    "summary_index": 0,
                    "text": "second",
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 5,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-5.5",
                        "parallel_tool_calls": True,
                        "tool_choice": "auto",
                        "tools": [],
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 2,
                            "output_tokens_details": {"reasoning_tokens": 2},
                            "total_tokens": 3,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        m.generate()
        assert thinking_events == [True, False, True, False]

    def test_unsupported_audio_attachment_is_dropped_not_relabelled(
        self, capture_server: str
    ) -> None:
        """OGG / FLAC / etc. audio MUST NOT be relabelled as MP3."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        att = Attachment(data=b"fake ogg bytes", mime_type="audio/ogg")
        m.initialize("transcribe", attachments=[att])
        m.generate()
        body = _last_body()
        user_msg = next(
            it for it in body["input"]
            if isinstance(it, dict) and it.get("role") == "user"
        )
        parts = user_msg["content"]
        assert not any(
            isinstance(p, dict) and p.get("type") == "input_audio"
            for p in parts
        )


class TestReviewBugReproductions3:
    """End-to-end tests reproducing bugs flagged by the third gpt-5.5 review."""

    def test_followup_request_preserves_function_call_output_item_id(
        self, capture_server: str
    ) -> None:
        """The output-item ``id`` (e.g. ``fc_1``) must be replayed next turn."""
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")

        _CapturingHandler.next_response_body = _tool_call_response_json(
            name="echo",
            arguments='{"text":"hello"}',
            call_id="call_abc",
        ).encode()

        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "hello"})]
        )

        _CapturingHandler.next_response_body = _text_response_json("done").encode()
        m.generate()

        body = _last_body()
        fc_items = [
            item for item in body["input"]
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]
        assert fc_items
        assert fc_items[-1]["id"] == "fc_1"
        assert fc_items[-1]["call_id"] == "call_abc"

    def test_streaming_late_metadata_with_nonzero_output_index_does_not_duplicate(
        self, capture_server: str
    ) -> None:
        """Final-response merge must key by item_id / output_index, not enum."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "",
                        "name": "",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "item_id": "fc_1",
                    "output_index": 1,
                    "delta": '{"text":"x"}',
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "reasoning",
                                "id": "rs_1",
                                "summary": [],
                            },
                            {
                                "type": "function_call",
                                "id": "fc_1",
                                "call_id": "call_real",
                                "name": "echo",
                                "arguments": '{"text":"x"}',
                            },
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_real", "name": "echo", "arguments": {"text": "x"}}
        ]

    def test_chat_completions_tool_choice_is_flattened_for_responses(
        self, capture_server: str
    ) -> None:
        """Chat-Completions ``tool_choice`` must be flattened for Responses API."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "echo"},
                }
            },
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json().encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        body = _last_body()
        assert body["tool_choice"] == {"type": "function", "name": "echo"}
        assert "function" not in body["tool_choice"]

    def test_deepseek_fallback_merges_existing_reasoning_config(
        self, capture_server: str
    ) -> None:
        """DeepSeek fallback must merge with caller-supplied ``reasoning``."""
        m = OpenAICompatibleModel2(
            "deepseek/deepseek-r1",
            base_url=capture_server,
            api_key="k",
            model_config={
                "reasoning": {"summary": "auto"},
                "reasoning_effort": "high",
            },
        )
        m.initialize("please call echo")

        _CapturingHandler.next_response_body = _text_response_json(
            text='{"tool_calls": [{"name": "echo", "arguments": {"text": "hi"}}]}'
        ).encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        body = _last_body()
        assert body["reasoning"] == {"summary": "auto", "effort": "high"}

    def test_incremental_parallel_tool_results_keep_original_call_ids(
        self, capture_server: str
    ) -> None:
        """Incremental add_function_results must preserve original call_ids."""
        response_json = json.dumps(
            {
                "id": "resp_tc",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": '{"text":"a"}',
                    },
                    {
                        "type": "function_call",
                        "id": "fc_2",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": '{"text":"b"}',
                    },
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = response_json
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert [fc["id"] for fc in fcs] == ["call_a", "call_b"]

        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "A"})]
        )
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "B"})]
        )
        outputs = [
            x for x in m.conversation
            if isinstance(x, dict) and x.get("type") == "function_call_output"
        ]
        assert outputs[0]["call_id"] == "call_a"
        assert outputs[1]["call_id"] == "call_b"

    def test_streaming_arguments_before_output_item_added_late_binds_metadata(
        self, capture_server: str
    ) -> None:
        """Arguments delta arriving before output_item.added must late-bind."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 1,
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '{"text":"early"}',
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 2,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_early",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_early", "name": "echo", "arguments": {"text": "early"}}
        ]


class TestReviewBugReproductions4:
    """End-to-end tests reproducing bugs flagged by the fourth gpt-5.5 review."""

    def test_streaming_completed_response_overrides_partial_arguments(
        self, capture_server: str
    ) -> None:
        """Final response.output arguments must override partial stream deltas."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '{"text":"par',
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "function_call",
                                "id": "fc_1",
                                "call_id": "call_1",
                                "name": "echo",
                                "arguments": '{"text":"final"}',
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_1", "name": "echo", "arguments": {"text": "final"}}
        ]

    def test_streaming_arguments_before_added_without_output_index_does_not_duplicate(
        self, capture_server: str
    ) -> None:
        """args.delta with no output_index then added(idx=1) must coalesce."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 1,
                    "item_id": "fc_1",
                    "delta": '{"text":"early"}',
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 2,
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_early",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_early", "name": "echo", "arguments": {"text": "early"}}
        ]

    def test_max_completion_tokens_is_translated_to_max_output_tokens(
        self, capture_server: str
    ) -> None:
        """``max_completion_tokens`` (newer Chat-Completions) → max_output_tokens."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={"max_completion_tokens": 11},
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        assert body["max_output_tokens"] == 11
        assert "max_completion_tokens" not in body

    def test_response_format_is_translated_to_text_format(
        self, capture_server: str
    ) -> None:
        """Chat-Completions ``response_format`` → Responses ``text.format``."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={"response_format": {"type": "json_object"}},
        )
        m.initialize("return json")
        m.generate()
        body = _last_body()
        assert body["text"]["format"] == {"type": "json_object"}
        assert "response_format" not in body

    def test_reasoning_merge_does_not_mutate_caller_nested_config(
        self, capture_server: str
    ) -> None:
        """Building kwargs must not mutate caller-supplied nested ``reasoning``."""
        cfg: dict[str, Any] = {
            "reasoning": {"summary": "auto"},
            "reasoning_effort": "high",
        }
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config=cfg,
        )
        m.initialize("hi")
        m.generate()
        assert cfg == {
            "reasoning": {"summary": "auto"},
            "reasoning_effort": "high",
        }
        body = _last_body()
        assert body["reasoning"] == {"summary": "auto", "effort": "high"}

    def test_deepseek_fallback_translates_max_completion_tokens(
        self, capture_server: str
    ) -> None:
        """DeepSeek fallback must translate ``max_completion_tokens``."""
        m = OpenAICompatibleModel2(
            "deepseek/deepseek-r1",
            base_url=capture_server,
            api_key="k",
            model_config={"max_completion_tokens": 9},
        )
        m.initialize("please call echo")
        _CapturingHandler.next_response_body = _text_response_json(
            text='{"tool_calls": [{"name": "echo", "arguments": {"text": "hi"}}]}'
        ).encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        body = _last_body()
        assert body["max_output_tokens"] == 9
        assert "max_completion_tokens" not in body


class TestReviewBugReproductions5:
    """End-to-end tests reproducing bugs flagged by the fifth gpt-5.5 review."""

    def test_reasoning_output_item_is_replayed_on_followup(
        self, capture_server: str
    ) -> None:
        """Reasoning items from response.output must be replayed next turn."""
        response_json = json.dumps(
            {
                "id": "resp_reasoning_tool",
                "object": "response",
                "created_at": 0,
                "model": "gpt-5.5",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [
                            {"type": "summary_text", "text": "I should call the tool."}
                        ],
                    },
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": '{"text":"hi"}',
                    },
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 2,
                    "output_tokens_details": {"reasoning_tokens": 1},
                    "total_tokens": 3,
                },
            }
        ).encode()
        m = OpenAICompatibleModel2(
            "gpt-5.5", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = response_json
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs[0]["id"] == "call_1"
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "hi"})]
        )
        _CapturingHandler.next_response_body = _text_response_json("done").encode()
        m.generate()
        body = _last_body()
        input_items = body["input"]
        assert any(
            isinstance(item, dict)
            and item.get("type") == "reasoning"
            and item.get("id") == "rs_1"
            for item in input_items
        ), "Responses reasoning output items must be replayed in the next input"
        assert any(
            isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("id") == "fc_1"
            and item.get("call_id") == "call_1"
            for item in input_items
        )

    def test_refusal_content_is_returned_non_streaming(
        self, capture_server: str
    ) -> None:
        """Assistant refusal content parts must surface as response text."""
        refusal_body = json.dumps(
            {
                "id": "resp_refusal",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "refusal",
                                "refusal": "I can't help with that.",
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("bad request")
        _CapturingHandler.next_response_body = refusal_body
        text, _resp = m.generate()
        assert text == "I can't help with that."

    def test_reasoning_effort_overrides_existing_reasoning_effort_field(
        self, capture_server: str
    ) -> None:
        """v1-compat top-level ``reasoning_effort`` must win over nested effort."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={
                "reasoning": {"summary": "auto", "effort": "low"},
                "reasoning_effort": "high",
            },
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        assert body["reasoning"] == {"summary": "auto", "effort": "high"}


class TestReviewBugReproductions6:
    """End-to-end tests reproducing bugs flagged by the sixth gpt-5.5 review."""

    def test_deepseek_generate_does_not_replay_think_block(
        self, capture_server: str
    ) -> None:
        """DeepSeek ``<think>`` content must not leak into the next request."""
        m = OpenAICompatibleModel2(
            "deepseek/deepseek-r1", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json(
            text="<think>secret plan</think>final answer"
        ).encode()
        text, _resp = m.generate()
        assert text == "final answer"

        m.conversation.append(
            {"role": "user", "content": [{"type": "input_text", "text": "next"}]}
        )
        _CapturingHandler.next_response_body = _text_response_json("ok").encode()
        m.generate()
        body = _last_body()
        serialized_input = json.dumps(body["input"])
        assert "<think>" not in serialized_input
        assert "secret plan" not in serialized_input
        assert "final answer" in serialized_input

    def test_deepseek_tool_fallback_does_not_replay_think_block(
        self, capture_server: str
    ) -> None:
        """DeepSeek fallback must strip ``<think>`` before storing in conversation."""
        payload = (
            "<think>secret tool reasoning</think>"
            '{"tool_calls": [{"name": "echo", "arguments": {"text": "hi"}}]}'
        )
        m = OpenAICompatibleModel2(
            "deepseek/deepseek-r1", base_url=capture_server, api_key="k"
        )
        m.initialize("please call echo")
        _CapturingHandler.next_response_body = _text_response_json(text=payload).encode()
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert len(fcs) == 1
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "hi"})]
        )
        _CapturingHandler.next_response_body = _text_response_json("done").encode()
        m.generate()
        body = _last_body()
        serialized_input = json.dumps(body["input"])
        assert "<think>" not in serialized_input
        assert "secret tool reasoning" not in serialized_input

    def test_openrouter_cache_control_does_not_mutate_extra_body(
        self, capture_server: str
    ) -> None:
        """Cache-control injection must not mutate caller-provided extra_body."""
        cfg: dict[str, Any] = {
            "extra_body": {"provider": {"order": ["Anthropic"]}}
        }
        m = OpenAICompatibleModel2(
            "openrouter/anthropic/claude-opus-4-6",
            base_url=capture_server,
            api_key="k",
            model_config=cfg,
        )
        m.initialize("hi")
        m.generate()
        body = _last_body()
        assert body["provider"] == {"order": ["Anthropic"]}
        assert body["cache_control"] == {"type": "ephemeral"}
        assert cfg == {"extra_body": {"provider": {"order": ["Anthropic"]}}}


class TestReviewBugReproductions7:
    """End-to-end tests reproducing bugs flagged by the seventh gpt-5.5 review."""

    def test_response_format_json_schema_is_flattened_for_responses(
        self, capture_server: str
    ) -> None:
        """``response_format`` ``json_schema`` must be flattened, not nested."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer_schema",
                        "schema": schema,
                        "strict": True,
                    },
                }
            },
        )
        m.initialize("return json")
        m.generate()
        fmt = _last_body()["text"]["format"]
        assert fmt == {
            "type": "json_schema",
            "name": "answer_schema",
            "schema": schema,
            "strict": True,
        }
        assert "json_schema" not in fmt

    def test_parallel_streaming_tool_calls_without_output_index_do_not_merge(
        self, capture_server: str
    ) -> None:
        """Parallel tool calls without ``output_index`` must not collapse."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "item_id": "fc_1",
                    "delta": '{"text":"a"}',
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 3,
                    "item": {
                        "type": "function_call",
                        "id": "fc_2",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 4,
                    "item_id": "fc_2",
                    "delta": '{"text":"b"}',
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 5,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_a", "name": "echo", "arguments": {"text": "a"}},
            {"id": "call_b", "name": "echo", "arguments": {"text": "b"}},
        ]

    def test_streaming_response_failed_raises_kiss_error(
        self, capture_server: str
    ) -> None:
        """``response.failed`` SSE event must raise KISSError."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _stream_sse_event(
            "response.failed",
            {
                "type": "response.failed",
                "sequence_number": 1,
                "response": {
                    "id": "resp_failed",
                    "object": "response",
                    "created_at": 0,
                    "model": "gpt-4o",
                    "status": "failed",
                    "error": {"message": "boom"},
                    "output": [],
                },
            },
        )
        with pytest.raises(KISSError, match="boom|failed"):
            m.generate()

    def test_streaming_completed_response_text_overrides_partial_delta(
        self, capture_server: str
    ) -> None:
        """Terminal completed-response text must override partial deltas."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "hel",
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_1",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "hello",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        text, _resp = m.generate()
        assert text == "hello"


class TestFactoryRouting:
    """The model() factory must route to v2 when explicitly requested."""

    def test_direct_instantiation_via_v2(self, capture_server: str) -> None:
        """OpenAICompatibleModel2 is directly instantiable like v1."""
        m = OpenAICompatibleModel2(
            "gpt-5.5", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        m.generate()
        assert _last_path().endswith("/responses")

    def test_factory_imports_v2_class(self) -> None:
        """The ``model()`` module exports ``OpenAICompatibleModel2``."""
        import kiss.core.models as models_pkg

        assert models_pkg.OpenAICompatibleModel2 is OpenAICompatibleModel2
        # ``model()`` is still importable and is the factory entry point;
        # whether it routes to v2 by default is intentionally unspecified
        # by this feature — direct instantiation is the supported path.
        assert callable(model)


class TestReviewBugReproductions8:
    """End-to-end tests reproducing bugs flagged by the eighth gpt-5.5 review."""

    def test_non_streaming_failed_response_raises(
        self, capture_server: str
    ) -> None:
        """A non-streaming Responses-API ``status=failed`` must raise."""
        from kiss.core.kiss_error import KISSError

        failed = json.dumps(
            {
                "id": "resp_failed",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "status": "failed",
                "error": {"message": "boom"},
                "output": [],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 1,
                },
            }
        ).encode()

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = failed
        with pytest.raises(KISSError, match="boom|failed"):
            m.generate()

    def test_non_streaming_failed_response_raises_in_tools_path(
        self, capture_server: str
    ) -> None:
        """``generate_and_process_with_tools`` must also raise on failure."""
        from kiss.core.kiss_error import KISSError

        failed = json.dumps(
            {
                "id": "resp_failed",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "status": "failed",
                "error": {"message": "boom"},
                "output": [],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 1,
                },
            }
        ).encode()

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = failed
        with pytest.raises(KISSError, match="boom|failed"):
            m.generate_and_process_with_tools(
                function_map={"echo": _echo},
                tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
            )

    def test_streaming_output_item_added_with_arguments_is_parsed(
        self, capture_server: str
    ) -> None:
        """Initial ``arguments`` on ``output_item.added`` must be captured."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_added",
                        "name": "echo",
                        "arguments": '{"text":"from-added"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {
                "id": "call_added",
                "name": "echo",
                "arguments": {"text": "from-added"},
            }
        ]

    def test_initialize_clears_pending_function_calls(
        self, capture_server: str
    ) -> None:
        """A fresh ``initialize()`` must drop stale pending tool-call ids.

        After review-14, orphan function_call_output items are rejected
        outright (KISSError).  This test now confirms that
        ``initialize()`` resets ``_pending_function_calls`` by verifying
        the orphan call raises — if it leaked the old pending entry, the
        raise would *not* mention "No prior function_call" but would
        instead mention "No pending function_call named 'orphan'" (the
        pending list would be non-empty containing the stale ``echo``
        entry).  Both error paths prove the state was cleared.
        """
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("first")

        _CapturingHandler.next_response_body = _tool_call_response_json(
            name="echo",
            arguments='{"text":"old"}',
            call_id="call_old",
        ).encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert m._pending_function_calls, (
            "sanity: pending call should be seeded after generate_and_process"
        )

        # Start a brand-new conversation without consuming the prior call.
        m.initialize("second")
        # Pending state must be empty — otherwise the orphan would raise
        # "No pending function_call named 'orphan'" rather than the
        # "No prior function_call" message we expect below.
        assert m._pending_function_calls == []
        with pytest.raises(KISSError, match="No prior function_call"):
            m.add_function_results_to_conversation_and_return(
                [("orphan", {"result": "ok"})]
            )

    def test_deepseek_cleaned_assistant_message_uses_valid_input_shape(
        self, capture_server: str
    ) -> None:
        """DeepSeek follow-up must replay cleaned text as a valid input item."""
        m = OpenAICompatibleModel2(
            "deepseek/deepseek-r1",
            base_url=capture_server,
            api_key="k",
        )
        m.initialize("hi")

        _CapturingHandler.next_response_body = _text_response_json(
            text="<think>secret</think>final"
        ).encode()
        m.generate()

        m.conversation.append(
            {"role": "user", "content": [{"type": "input_text", "text": "next"}]}
        )
        _CapturingHandler.next_response_body = _text_response_json("ok").encode()
        m.generate()

        body = _last_body()
        assistant_items = [
            item
            for item in body["input"]
            if isinstance(item, dict) and item.get("role") == "assistant"
        ]
        assert assistant_items
        for item in assistant_items:
            if item.get("type") == "message":
                continue
            # Easy-input assistant messages must use plain string content
            # (not the hybrid ``output_text`` content-part shape).
            assert isinstance(item.get("content"), str), item

    def test_streaming_completed_with_failed_status_raises(
        self, capture_server: str
    ) -> None:
        """A terminal ``response.completed`` with ``status=failed`` must raise."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _stream_sse_event(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": 1,
                "response": {
                    "id": "resp_failed",
                    "object": "response",
                    "created_at": 0,
                    "model": "gpt-4o",
                    "status": "failed",
                    "error": {"message": "boom"},
                    "output": [],
                    "usage": {
                        "input_tokens": 1,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 0,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 1,
                    },
                },
            },
        )
        with pytest.raises(KISSError, match="boom|failed"):
            m.generate()


class TestReviewBugReproductions9:
    """Reproduce + verify fix for review #9 bugs.

    Bug 1: Streaming path collects a function_call from
    ``output_item.added`` + ``function_call_arguments.delta`` events, but
    the terminal ``response.completed.response.output`` contains some
    items (e.g. a reasoning block) that omit the function_call.  The
    current ``generate_and_process_with_tools`` only manually replays
    streamed tool calls when ``raw_items`` is empty, so the conversation
    loses the prior ``function_call``.  The follow-up
    ``function_call_output`` then has no matching ``call_id`` in input,
    violating the Responses API contract.
    """

    def test_streamed_tool_call_preserved_when_completed_output_omits_it(
        self, capture_server: str
    ) -> None:
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_streamed",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '{"text":"x"}',
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "parallel_tool_calls": True,
                        "tool_choice": "auto",
                        "tools": [],
                        # Non-empty output, but no function_call item.
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_1",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)

        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_streamed", "name": "echo", "arguments": {"text": "x"}}
        ]

        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "ok"})]
        )

        m.token_callback = None  # second generate() uses non-streaming JSON body
        _CapturingHandler.next_response_body = _text_response_json("done").encode()
        m.generate()

        body = _last_body()
        input_items = body["input"]
        assert any(
            isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("call_id") == "call_streamed"
            for item in input_items
        ), "stream-collected function_call must be replayed before its output"
        assert any(
            isinstance(item, dict)
            and item.get("type") == "function_call_output"
            and item.get("call_id") == "call_streamed"
            for item in input_items
        )


class TestReviewBugReproductions10:
    """Reproduce + verify fix for review #10 bugs."""

    def test_stream_options_is_stripped_for_responses_api(
        self, capture_server: str
    ) -> None:
        """Chat-Completions-only ``stream_options`` must not be forwarded."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={"stream_options": {"include_usage": True}},
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json("ok").encode()
        text, _resp = m.generate()
        assert text == "ok"
        body = _last_body()
        assert "stream_options" not in body

    def test_streaming_completed_empty_text_overrides_partial_delta(
        self, capture_server: str
    ) -> None:
        """Terminal empty-text message must override partial streamed delta."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "partial",
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
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
                                        "text": "",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 0,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 1,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        text, _resp = m.generate()
        assert text == ""

    def test_function_call_arguments_delta_closes_reasoning_block(
        self, capture_server: str
    ) -> None:
        """Function-call argument events must close an open reasoning block.

        Out-of-order streams may emit reasoning, then function-call
        arguments before ``output_item.added``, then a second reasoning
        block.  Each reasoning block must get its own True/False pair.
        """
        thinking_events: list[bool] = []

        m = OpenAICompatibleModel2(
            "gpt-5.5",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
            thinking_callback=thinking_events.append,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "sequence_number": 1,
                    "item_id": "rs_1",
                    "output_index": 0,
                    "summary_index": 0,
                    "delta": "first",
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "item_id": "fc_1",
                    "output_index": 1,
                    "delta": '{"text":"x"}',
                },
            ),
            _stream_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": 3,
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": '{"text":"x"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "sequence_number": 4,
                    "item_id": "rs_2",
                    "output_index": 2,
                    "summary_index": 0,
                    "delta": "second",
                },
            ),
            _stream_sse_event(
                "response.reasoning_summary_text.done",
                {
                    "type": "response.reasoning_summary_text.done",
                    "sequence_number": 5,
                    "item_id": "rs_2",
                    "output_index": 2,
                    "summary_index": 0,
                    "text": "second",
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 6,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-5.5",
                        "parallel_tool_calls": True,
                        "tool_choice": "auto",
                        "tools": [],
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 2,
                            "output_tokens_details": {"reasoning_tokens": 2},
                            "total_tokens": 3,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)

        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        assert fcs == [
            {"id": "call_1", "name": "echo", "arguments": {"text": "x"}}
        ]
        assert thinking_events == [True, False, True, False]


class TestReviewBugReproductions11:
    """Reproduce + verify fix for review #11 bugs."""

    def test_generate_without_tools_strips_tool_choice(
        self, capture_server: str
    ) -> None:
        """``tool_choice`` must not be sent when no ``tools`` are sent."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "echo"},
                },
            },
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json("ok").encode()
        m.generate()
        body = _last_body()
        assert "tools" not in body
        assert "tool_choice" not in body

    def test_deepseek_text_fallback_does_not_forward_tool_choice_without_tools(
        self, capture_server: str
    ) -> None:
        """DeepSeek fallback intentionally sends no native tools; drop tool_choice."""
        m = OpenAICompatibleModel2(
            "deepseek/deepseek-r1",
            base_url=capture_server,
            api_key="k",
            model_config={
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "echo"},
                },
            },
        )
        m.initialize("please call echo")
        _CapturingHandler.next_response_body = _text_response_json(
            text=(
                '{"tool_calls": [{"name": "echo", '
                '"arguments": {"text": "hi"}}]}'
            )
        ).encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        body = _last_body()
        assert not body.get("tools")
        assert "tool_choice" not in body

    def test_function_result_non_string_is_serialized_for_function_call_output(
        self, capture_server: str
    ) -> None:
        """``function_call_output.output`` MUST be a string per Responses API."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _tool_call_response_json(
            name="echo",
            arguments='{"text":"hello"}',
            call_id="call_abc",
        ).encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": {"ok": True, "value": 3}})]
        )
        outputs = [
            x
            for x in m.conversation
            if isinstance(x, dict) and x.get("type") == "function_call_output"
        ]
        assert outputs
        assert outputs[-1]["call_id"] == "call_abc"
        assert isinstance(outputs[-1]["output"], str)
        assert json.loads(outputs[-1]["output"]) == {"ok": True, "value": 3}

        _CapturingHandler.next_response_body = _text_response_json("done").encode()
        m.generate()
        body = _last_body()
        sent_outputs = [
            x
            for x in body["input"]
            if isinstance(x, dict) and x.get("type") == "function_call_output"
        ]
        assert isinstance(sent_outputs[-1]["output"], str)


class TestReviewBugReproductions12:
    """Reproduce + verify fix for review #12 bugs."""

    def test_parallel_tool_outputs_with_attachment_do_not_interleave_user_message(
        self, capture_server: str
    ) -> None:
        """Lifted attachments must come AFTER all pending tool outputs."""
        from kiss.core.models.model import encode_binary_attachment

        response_json = json.dumps(
            {
                "id": "resp_tc",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": '{"text":"a"}',
                    },
                    {
                        "type": "function_call",
                        "id": "fc_2",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": '{"text":"b"}',
                    },
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()

        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m.initialize("hi")
        _CapturingHandler.next_response_body = response_json
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        img = encode_binary_attachment("image/png", b"\x89PNG\r\n\x1a\n")

        # Incremental result submission with attachment on first call.
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": f"first result {img}"})]
        )
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "second result"})]
        )

        _CapturingHandler.next_response_body = _text_response_json("done").encode()
        m.generate()

        input_items = _last_body()["input"]
        idx_user_attachment = next(
            i
            for i, item in enumerate(input_items)
            if isinstance(item, dict)
            and item.get("role") == "user"
            and any(
                isinstance(p, dict) and p.get("type") == "input_image"
                for p in item.get("content", [])
            )
        )
        idx_call_b_output = next(
            i
            for i, item in enumerate(input_items)
            if isinstance(item, dict)
            and item.get("type") == "function_call_output"
            and item.get("call_id") == "call_b"
        )
        assert idx_call_b_output < idx_user_attachment

    def test_incomplete_tool_call_response_raises(
        self, capture_server: str
    ) -> None:
        """Non-streaming ``status='incomplete'`` MUST raise to avoid bad tool calls."""
        from kiss.core.kiss_error import KISSError

        incomplete = json.dumps(
            {
                "id": "resp_incomplete",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_partial",
                        "name": "echo",
                        "arguments": '{"text":',
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = incomplete

        with pytest.raises(KISSError, match="incomplete|max_output_tokens"):
            m.generate_and_process_with_tools(
                function_map={"echo": _echo},
                tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
            )

    def test_mismatched_tool_result_name_does_not_reuse_wrong_call_id(
        self, capture_server: str
    ) -> None:
        """Submitting a result for an unknown function name must raise."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _tool_call_response_json(
            name="echo",
            arguments='{"text":"hello"}',
            call_id="call_echo",
        ).encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        with pytest.raises(KISSError, match="No pending function_call.*other"):
            m.add_function_results_to_conversation_and_return(
                [("other", {"result": "wrong result"})]
            )

    def test_streaming_response_incomplete_raises(
        self, capture_server: str
    ) -> None:
        """Streaming ``response.incomplete`` MUST raise."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _stream_sse_event(
            "response.incomplete",
            {
                "type": "response.incomplete",
                "sequence_number": 1,
                "response": {
                    "id": "resp_incomplete",
                    "object": "response",
                    "created_at": 0,
                    "model": "gpt-4o",
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [],
                    "usage": {
                        "input_tokens": 1,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 0,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 1,
                    },
                },
            },
        )
        with pytest.raises(KISSError, match="incomplete|max_output_tokens"):
            m.generate()


class TestReviewBugReproductions13:
    """Reproduce + verify fix for review #13 bugs."""

    def test_deepseek_fallback_strips_model_config_tools(
        self, capture_server: str
    ) -> None:
        """DeepSeek text fallback must not forward native tools from model_config."""
        m = OpenAICompatibleModel2(
            "deepseek/deepseek-r1",
            base_url=capture_server,
            api_key="k",
            model_config={"tools": _ECHO_TOOL_CHAT_SCHEMA},
        )
        m.initialize("please call echo")
        _CapturingHandler.next_response_body = _text_response_json(
            text=(
                '{"tool_calls": [{"name": "echo", '
                '"arguments": {"text": "hi"}}]}'
            )
        ).encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        body = _last_body()
        assert "tools" not in body or not body["tools"]

    def test_generate_without_tools_strips_model_config_tools(
        self, capture_server: str
    ) -> None:
        """Plain generate() must not forward native tools from model_config."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={"tools": _ECHO_TOOL_CHAT_SCHEMA},
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json("ok").encode()
        m.generate()
        body = _last_body()
        assert "tools" not in body

    def test_streaming_added_full_arguments_override_earlier_partial_delta(
        self, capture_server: str
    ) -> None:
        """Full arguments on output_item.added must repair earlier partial deltas."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 1,
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '{"text":"par',
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 2,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": '{"text":"final"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "parallel_tool_calls": True,
                        "tool_choice": "auto",
                        "tools": [],
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_1", "name": "echo", "arguments": {"text": "final"}}
        ]


class TestReviewBugReproductions14:
    """Reproducers for the four bugs reported by the 14th gpt-5.5 review."""

    def test_streaming_output_text_done_used_when_completed_output_empty(
        self, capture_server: str
    ) -> None:
        """``response.output_text.done`` is the sole final-text source.

        Some gateways emit ``.done`` events with the full final text but
        send an empty terminal ``response.output``. v2 must use the
        ``.done`` text rather than returning an empty string.
        """
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "text": "hello from done",
                    "logprobs": [],
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        text, _resp = m.generate()
        assert text == "hello from done"

    def test_streaming_refusal_done_used_when_completed_output_empty(
        self, capture_server: str
    ) -> None:
        """``response.refusal.done`` is the sole final-refusal source."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.refusal.done",
                {
                    "type": "response.refusal.done",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "refusal": "I cannot do that.",
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        text, _resp = m.generate()
        assert text == "I cannot do that."

    def test_streaming_output_item_done_empty_arguments_overrides_partial_delta(
        self, capture_server: str
    ) -> None:
        """Authoritative empty ``arguments=""`` overrides malformed delta.

        When the terminal ``output_item.done`` carries ``arguments=""``
        it MUST overwrite any partial / malformed argument string
        accumulated during streaming.  Otherwise the next turn replays
        an invalid ``function_call.arguments``.
        """
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_noargs",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '{"text":',
                },
            ),
            _stream_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": 3,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_noargs",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [{"id": "call_noargs", "name": "echo", "arguments": {}}]
        fc_items = [
            item
            for item in m.conversation
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]
        assert fc_items
        # Authoritative empty arguments must overwrite the partial
        # streamed ``{"text":`` prefix.
        assert fc_items[-1]["arguments"] == ""

    def test_function_call_missing_call_id_is_rejected(
        self, capture_server: str
    ) -> None:
        """Function_call without ``call_id`` raises (Responses-API contract)."""
        from kiss.core.kiss_error import KISSError

        body = json.dumps(
            {
                "id": "resp_bad",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "",
                        "name": "echo",
                        "arguments": '{"text":"hi"}',
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = body
        with pytest.raises(KISSError, match="call_id"):
            m.generate_and_process_with_tools(
                function_map={"echo": _echo},
                tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
            )


class TestReviewBugReproductions15:
    """Reproducer for the bug reported by the 15th gpt-5.5 review.

    Reviews 2-5 (dict-shaped response handling and openrouter embedding
    prefix) describe behavior that mirrors v1 — they are deliberate
    behavior parity, not regressions in v2.  Only Bug 1 (pending guard)
    is a real v2-only bug.
    """

    def test_generate_raises_if_parallel_tool_outputs_incomplete(
        self, capture_server: str
    ) -> None:
        """Pending-tool-call guard rejects new generate while outputs missing.

        The Responses API requires every model-produced ``function_call``
        to be paired with a ``function_call_output`` before the next
        request.  v2 must fail locally rather than send an invalid
        conversation.
        """
        from kiss.core.kiss_error import KISSError

        response_json = json.dumps(
            {
                "id": "resp_tc",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": '{"text":"a"}',
                    },
                    {
                        "type": "function_call",
                        "id": "fc_2",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": '{"text":"b"}',
                    },
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = response_json
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        # Only satisfy one of the two pending calls.
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "A"})]
        )
        # Snapshot the request count from the capture server.
        before = len(_CapturingHandler.captured_requests)
        with pytest.raises(KISSError, match="pending"):
            m.generate()
        # No new request was sent — the guard fired locally.
        assert len(_CapturingHandler.captured_requests) == before

    def test_generate_succeeds_after_all_pending_resolved(
        self, capture_server: str
    ) -> None:
        """After all pending tool calls are answered, generate() works again."""
        response_json = json.dumps(
            {
                "id": "resp_tc",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": '{"text":"a"}',
                    },
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = response_json
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "A"})]
        )
        # All pending resolved — next generate must succeed.
        _CapturingHandler.next_response_body = _text_response_json(
            "done"
        ).encode()
        text, _resp = m.generate()
        assert text == "done"


class TestReviewBugReproductions16:
    """Reproducer for the bug reported by the 16th gpt-5.5 review."""

    def test_trailing_fallback_rejects_mismatched_tool_result_name(
        self, capture_server: str
    ) -> None:
        """When pending queue is absent, trailing fallback must validate names.

        Restored / reconstructed conversations may carry prior
        ``function_call`` items without seeding
        ``_pending_function_calls``.  The trailing fallback must still
        refuse to pair a result with a mismatched function name.
        """
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
        )
        m.initialize("hi")
        m.conversation.append(
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_echo",
                "name": "echo",
                "arguments": '{"text":"hi"}',
            }
        )
        assert m._pending_function_calls == []
        with pytest.raises(
            KISSError,
            match=(
                "mismatch|No pending function_call|No prior"
                "|No unanswered function_call"
            ),
        ):
            m.add_function_results_to_conversation_and_return(
                [("other", {"result": "wrong result"})]
            )
        assert not any(
            isinstance(item, dict)
            and item.get("type") == "function_call_output"
            for item in m.conversation
        )


class TestReviewBugReproductions17:
    """Reproducers for the three bugs reported by the 17th gpt-5.5 review."""

    def test_streaming_output_text_done_invokes_token_callback(
        self, capture_server: str
    ) -> None:
        """Done-only text streams must still drive ``token_callback``."""
        tokens: list[str] = []
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "text": "hello from done",
                    "logprobs": [],
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        text, _resp = m.generate()
        assert text == "hello from done"
        assert tokens == ["hello from done"]

    def test_reasoning_summary_done_only_is_bracketed(
        self, capture_server: str
    ) -> None:
        """Done-only reasoning summary still emits True/False + token push."""
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
        frames = [
            _stream_sse_event(
                "response.reasoning_summary_text.done",
                {
                    "type": "response.reasoning_summary_text.done",
                    "sequence_number": 1,
                    "item_id": "rs_1",
                    "output_index": 0,
                    "summary_index": 0,
                    "text": "done-only reasoning",
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-5.5",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 1},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        m.generate()
        assert tokens == ["done-only reasoning"]
        assert thinking_events == [True, False]

    def test_generate_rejects_unanswered_function_call_even_if_pending_queue_lost(
        self, capture_server: str
    ) -> None:
        """Conversation-level guard catches unanswered function_call items."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _tool_call_response_json(
            name="echo",
            arguments='{"text":"hello"}',
            call_id="call_lost",
        ).encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        # Simulate conversation restore / process restart: the public
        # conversation survives, the private pending queue does not.
        assert any(
            isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("call_id") == "call_lost"
            for item in m.conversation
        )
        m._pending_function_calls = []
        before = len(_CapturingHandler.captured_requests)
        with pytest.raises(KISSError, match="function_call|pending|output"):
            m.generate()
        assert len(_CapturingHandler.captured_requests) == before


class TestReviewBugReproductions18:
    """Reproducers for the four bugs reported by the 18th gpt-5.5 review."""

    def test_streamed_text_preserved_when_completed_output_has_only_reasoning(
        self, capture_server: str
    ) -> None:
        """Streamed text must be replayed when final output lacks a message."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 1,
                    "content_index": 0,
                    "delta": "hello",
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "reasoning",
                                "id": "rs_1",
                                "summary": [],
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        text, _resp = m.generate()
        assert text == "hello"
        m.conversation.append(
            {"role": "user", "content": [{"type": "input_text", "text": "next"}]}
        )
        m.token_callback = None
        _CapturingHandler.next_response_body = _text_response_json(
            "done"
        ).encode()
        m.generate()
        serialized_input = json.dumps(_last_body()["input"])
        assert "hello" in serialized_input

    def test_streamed_function_call_inserted_at_original_output_index_when_final_omits_it(
        self, capture_server: str
    ) -> None:
        """A streamed function_call omitted from final output is replayed in order."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_first",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '{"text":"x"}',
                },
            ),
            _stream_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 3,
                    "item_id": "msg_1",
                    "output_index": 1,
                    "content_index": 0,
                    "delta": "after call",
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_1",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "after call",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_first", "name": "echo", "arguments": {"text": "x"}}
        ]
        assert content == "after call"
        fc_idx = next(
            i
            for i, item in enumerate(m.conversation)
            if isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("call_id") == "call_first"
        )
        msg_idx = next(
            i
            for i, item in enumerate(m.conversation)
            if isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role") == "assistant"
        )
        assert fc_idx < msg_idx

    def test_reasoning_summary_index_zero_is_not_treated_as_missing(
        self, capture_server: str
    ) -> None:
        """summary_index=0 must not fall back to content_index and duplicate text."""
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
        frames = [
            _stream_sse_event(
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "sequence_number": 1,
                    "item_id": "rs_1",
                    "output_index": 0,
                    "summary_index": 0,
                    "content_index": 1,
                    "delta": "plan",
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
                    "delta": "answer",
                },
            ),
            _stream_sse_event(
                "response.reasoning_summary_text.done",
                {
                    "type": "response.reasoning_summary_text.done",
                    "sequence_number": 3,
                    "item_id": "rs_1",
                    "output_index": 0,
                    "summary_index": 0,
                    "text": "plan",
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-5.5",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_1",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "answer",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 2,
                            "output_tokens_details": {"reasoning_tokens": 1},
                            "total_tokens": 3,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        text, _resp = m.generate()
        assert text == "answer"
        assert tokens == ["plan", "answer"]
        assert thinking_events == [True, False]

    def test_streamed_tool_preamble_preserved_when_final_output_has_only_function_call(
        self, capture_server: str
    ) -> None:
        """Streamed assistant text before a tool call must be replayed."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "I will call echo.",
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 2,
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 3,
                    "item_id": "fc_1",
                    "output_index": 1,
                    "delta": '{"text":"hi"}',
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "function_call",
                                "id": "fc_1",
                                "call_id": "call_1",
                                "name": "echo",
                                "arguments": '{"text":"hi"}',
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 2,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 3,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert content == "I will call echo."
        assert fcs == [
            {"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}
        ]
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "hi"})]
        )
        m.token_callback = None
        _CapturingHandler.next_response_body = _text_response_json(
            "done"
        ).encode()
        m.generate()
        serialized_input = json.dumps(_last_body()["input"])
        assert "I will call echo." in serialized_input


class TestReviewBugReproductions19:
    """Reproduce bugs found by the 19th gpt-5.5 review."""

    def test_arguments_delta_before_added_preserves_later_output_index(
        self, capture_server: str
    ) -> None:
        """Bug 1: arguments.delta first → output_item.added with real index 1 must re-key."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "I will call it.",
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "item_id": "fc_1",
                    "delta": '{"text":"x"}',
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 3,
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_1",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "I will call it.",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)

        fcs, content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        assert fcs == [{"id": "call_1", "name": "echo", "arguments": {"text": "x"}}]
        assert content == "I will call it."

        msg_idx = next(
            i
            for i, item in enumerate(m.conversation)
            if isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role") == "assistant"
        )
        fc_idx = next(
            i
            for i, item in enumerate(m.conversation)
            if isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("call_id") == "call_1"
        )
        # Original gateway order was message at index 0, function_call at
        # index 1.  Replay must preserve that order.
        assert msg_idx < fc_idx

    def test_done_text_fallback_does_not_drop_delta_only_parts(
        self, capture_server: str
    ) -> None:
        """Bug 2: streaming .done text fallback must not drop delta-only parts."""
        tokens: list[str] = []
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "hello ",
                },
            ),
            _stream_sse_event(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "sequence_number": 2,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 1,
                    "text": "world",
                    "logprobs": [],
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 2,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 3,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)

        text, _resp = m.generate()

        assert text == "hello world"
        assert tokens == ["hello ", "world"]

    def test_stream_metadata_patches_stale_completed_function_call_item(
        self, capture_server: str
    ) -> None:
        """Bug 3: stale terminal function_call must be patched with stream metadata."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "",
                        "name": "",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": 2,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_good",
                        "name": "echo",
                        "arguments": '{"text":"x"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "function_call",
                                "id": "fc_1",
                                "call_id": "",
                                "name": "",
                                "arguments": '{"text":"x"}',
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)

        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [{"id": "call_good", "name": "echo", "arguments": {"text": "x"}}]

        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "ok"})]
        )

        m.token_callback = None
        _CapturingHandler.next_response_body = _text_response_json("done").encode()
        m.generate()

        body = _last_body()
        fc_items = [
            item
            for item in body["input"]
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]
        assert fc_items
        assert fc_items[-1]["call_id"] == "call_good"
        assert fc_items[-1]["name"] == "echo"

    def test_restored_conversation_can_answer_non_trailing_function_call(
        self, capture_server: str
    ) -> None:
        """Bug 4: restored conversation must answer non-trailing function_call."""
        response_json = json.dumps(
            {
                "id": "resp_tc",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": '{"text":"x"}',
                    },
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "I called the tool.",
                                "annotations": [],
                            }
                        ],
                    },
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()

        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m.initialize("hi")
        _CapturingHandler.next_response_body = response_json

        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [{"id": "call_1", "name": "echo", "arguments": {"text": "x"}}]

        # Simulate restore / process restart: public conversation
        # survives, private pending queue is lost.
        m._pending_function_calls = []

        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "ok"})]
        )

        outputs = [
            item
            for item in m.conversation
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        assert outputs[-1]["call_id"] == "call_1"


class TestReviewBugReproductions20:
    """Reproduce bugs found by the 20th gpt-5.5 review."""

    def test_generate_rejects_orphan_function_call_output(
        self, capture_server: str
    ) -> None:
        """Bug 1: orphan function_call_output (no prior function_call) must be rejected."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
        )
        m.initialize("hi")

        m.conversation.append(
            {
                "type": "function_call_output",
                "call_id": "call_missing",
                "output": "orphan output",
            }
        )

        before = len(_CapturingHandler.captured_requests)

        with pytest.raises(
            KISSError, match="function_call_output|call_missing|prior"
        ):
            m.generate()

        assert len(_CapturingHandler.captured_requests) == before

    def test_stream_only_function_call_replayed_before_later_terminal_message(
        self, capture_server: str
    ) -> None:
        """Bug 2: stream-only function_call output_index must order vs terminal items."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 2,
                    "content_index": 0,
                    "delta": "after tool",
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 2,
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 3,
                    "item_id": "fc_1",
                    "output_index": 1,
                    "delta": '{"text":"x"}',
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_1",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "after tool",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]

        _CapturingHandler.next_response_body = b"".join(frames)

        fcs, content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        assert fcs == [
            {"id": "call_1", "name": "echo", "arguments": {"text": "x"}}
        ]
        assert content == "after tool"

        fc_idx = next(
            i
            for i, item in enumerate(m.conversation)
            if isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("call_id") == "call_1"
        )
        msg_idx = next(
            i
            for i, item in enumerate(m.conversation)
            if isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("id") == "msg_1"
        )

        assert fc_idx < msg_idx


class TestReviewBugReproductions21:
    """Reproduce bugs found by the 21st gpt-5.5 review."""

    def test_restored_parallel_tool_outputs_with_attachment_do_not_interleave(
        self, capture_server: str
    ) -> None:
        """Bug 1: restored parallel calls — attachment flush must wait for ALL outputs."""
        from kiss.core.models.model import encode_binary_attachment

        response_json = json.dumps(
            {
                "id": "resp_tc",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": '{"text":"a"}',
                    },
                    {
                        "type": "function_call",
                        "id": "fc_2",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": '{"text":"b"}',
                    },
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()

        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m.initialize("hi")

        _CapturingHandler.next_response_body = response_json
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        # Simulate process restart / restore.
        m._pending_function_calls = []

        img = encode_binary_attachment("image/png", b"\x89PNG\r\n\x1a\n")

        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": f"first {img}"})]
        )
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "second"})]
        )

        _CapturingHandler.next_response_body = _text_response_json("done").encode()
        m.generate()

        input_items = _last_body()["input"]

        idx_attachment_user = next(
            i
            for i, item in enumerate(input_items)
            if isinstance(item, dict)
            and item.get("role") == "user"
            and any(
                isinstance(p, dict) and p.get("type") == "input_image"
                for p in item.get("content", [])
            )
        )
        idx_call_b_output = next(
            i
            for i, item in enumerate(input_items)
            if isinstance(item, dict)
            and item.get("type") == "function_call_output"
            and item.get("call_id") == "call_b"
        )

        assert idx_call_b_output < idx_attachment_user

    def test_empty_refusal_message_not_resent(
        self, capture_server: str
    ) -> None:
        """Bug 2: empty refusal content parts must be dropped before resend."""
        refusal_empty = json.dumps(
            {
                "id": "resp_refusal_empty",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "refusal",
                                "refusal": "",
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 1,
                },
            }
        ).encode()

        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m.initialize("bad request")

        _CapturingHandler.next_response_body = refusal_empty
        text, _resp = m.generate()
        assert text == ""

        m.conversation.append(
            {"role": "user", "content": [{"type": "input_text", "text": "next"}]}
        )

        _CapturingHandler.next_response_body = _text_response_json("ok").encode()
        m.generate()

        body = _last_body()

        for item in body["input"]:
            if isinstance(item, dict) and item.get("role") == "assistant":
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        assert not (
                            isinstance(part, dict)
                            and part.get("type") == "refusal"
                            and not str(part.get("refusal", "")).strip()
                        ), "empty refusal parts must be dropped before resend"


class TestReviewBugReproductions22:
    """End-to-end reproducers for bugs reported by review 22."""

    def test_stream_only_function_call_before_unstreamed_terminal_message(
        self, capture_server: str
    ) -> None:
        """A streamed function_call at output_index=0 must replay before a terminal
        message whose item_id was never seen during streaming."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '{"text":"x"}',
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_1",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "after tool",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)

        fcs, content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        assert content == "after tool"
        assert fcs == [{"id": "call_1", "name": "echo", "arguments": {"text": "x"}}]

        fc_idx = next(
            i
            for i, item in enumerate(m.conversation)
            if isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("call_id") == "call_1"
        )
        msg_idx = next(
            i
            for i, item in enumerate(m.conversation)
            if isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("id") == "msg_1"
        )

        assert fc_idx < msg_idx

    def test_restored_after_partial_parallel_attachment_preserves_attachment(
        self, capture_server: str
    ) -> None:
        """Attachment buffered after first parallel tool output must survive restore."""
        from kiss.core.models.model import encode_binary_attachment

        response_json = json.dumps(
            {
                "id": "resp_tc",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": '{"text":"a"}',
                    },
                    {
                        "type": "function_call",
                        "id": "fc_2",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": '{"text":"b"}',
                    },
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()

        m1 = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m1.initialize("hi")
        _CapturingHandler.next_response_body = response_json
        m1.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        img = encode_binary_attachment("image/png", b"\x89PNG\r\n\x1a\n")

        m1.add_function_results_to_conversation_and_return(
            [("echo", {"result": f"first {img}"})]
        )

        assert m1._pending_tool_result_attachments, (
            "sanity: attachment is only in memory"
        )

        m2 = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m2.initialize("restored")
        m2.conversation = list(m1.conversation)
        m2._pending_function_calls = []
        m2._pending_tool_result_attachments = []

        m2.add_function_results_to_conversation_and_return(
            [("echo", {"result": "second"})]
        )

        _CapturingHandler.next_response_body = _text_response_json("done").encode()
        m2.generate()

        serialized = json.dumps(_last_body()["input"])
        assert "input_image" in serialized

    def test_deepseek_text_tool_call_ids_are_unique_across_turns(
        self, capture_server: str
    ) -> None:
        """DeepSeek text-based fallback must mint unique call_ids across turns."""
        m = OpenAICompatibleModel2(
            "deepseek/deepseek-r1",
            base_url=capture_server,
            api_key="k",
        )
        m.initialize("call echo")

        payload = '{"tool_calls": [{"name": "echo", "arguments": {"text": "hi"}}]}'

        _CapturingHandler.next_response_body = _text_response_json(
            text=payload
        ).encode()
        fcs1, _content1, _resp1 = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "hi"})]
        )

        m.conversation.append(
            {"role": "user", "content": [{"type": "input_text", "text": "again"}]}
        )

        _CapturingHandler.next_response_body = _text_response_json(
            text=payload
        ).encode()
        fcs2, _content2, _resp2 = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "hi again"})]
        )

        assert fcs1[0]["id"] != fcs2[0]["id"], (
            "DeepSeek fallback call_ids must be globally unique within conversation"
        )

        _CapturingHandler.next_response_body = _text_response_json("done").encode()
        m.generate()


class TestReviewBugReproductions23:
    """End-to-end reproducers for bugs reported by review 23."""

    def test_streaming_message_output_item_done_used_when_completed_output_empty(
        self, capture_server: str
    ) -> None:
        """`response.output_item.done` for a message item must be captured.

        When the gateway only emits the assistant text via
        ``output_item.done`` (no ``output_text.delta`` events) and the
        terminal ``response.completed.response.output`` is empty, v2 must
        still preserve and surface the message text.
        """
        tokens: list[str] = []
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "hello from item.done",
                                "annotations": [],
                            }
                        ],
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)

        text, _resp = m.generate()

        assert text == "hello from item.done"
        assert tokens == ["hello from item.done"]

    def test_streaming_added_full_arguments_override_valid_earlier_delta(
        self, capture_server: str
    ) -> None:
        """A full ``arguments`` payload on ``output_item.added`` must override
        any earlier delta-accumulated buffer, even if the buffer parses as
        valid JSON (e.g. delta=='{}', then item.arguments=='{"text":"final"}').
        """
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 1,
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": "{}",
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 2,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": '{"text":"final"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)

        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        assert fcs == [
            {"id": "call_1", "name": "echo", "arguments": {"text": "final"}}
        ]

    def test_stale_completed_function_call_without_id_is_removed_or_patched(
        self, capture_server: str
    ) -> None:
        """Stale terminal ``function_call`` item with empty ``call_id``/``name``
        must NOT remain in the conversation (it would cause the next
        generate() to fail with ``function_call item missing call_id``).
        """
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_good",
                        "name": "echo",
                        "arguments": '{"text":"x"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "",
                                "name": "",
                                "arguments": '{"text":"x"}',
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)

        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_good", "name": "echo", "arguments": {"text": "x"}}
        ]

        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "ok"})]
        )

        # Final generate uses non-streaming JSON body for inspection.
        m.token_callback = None
        _CapturingHandler.next_response_body = _text_response_json("done").encode()
        text, _resp = m.generate()

        assert text == "done"


class TestReviewBugReproductions24:
    """End-to-end reproducers for bugs reported by review 24."""

    def test_parallel_streaming_delta_before_added_does_not_collide(
        self, capture_server: str
    ) -> None:
        """A delta-first parallel call without ``output_index`` must NOT
        collide with a later call's real ``output_index``."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 1,
                    "item_id": "fc_b",
                    "delta": '{"text":"b"}',
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 2,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_a",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 3,
                    "item_id": "fc_a",
                    "output_index": 0,
                    "delta": '{"text":"a"}',
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 4,
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_b",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 5,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]

        _CapturingHandler.next_response_body = b"".join(frames)

        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        assert fcs == [
            {"id": "call_a", "name": "echo", "arguments": {"text": "a"}},
            {"id": "call_b", "name": "echo", "arguments": {"text": "b"}},
        ]

    def test_streaming_terminal_function_call_without_item_id_matches_by_call_id(
        self, capture_server: str
    ) -> None:
        """Terminal ``function_call`` without ``item_id`` but with a known
        ``call_id`` must merge with the streamed slot, not duplicate it."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_real",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "item_id": "fc_1",
                    "output_index": 1,
                    "delta": '{"text":"x"}',
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "call_real",
                                "name": "echo",
                                "arguments": '{"text":"final"}',
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]

        _CapturingHandler.next_response_body = b"".join(frames)

        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        assert fcs == [
            {"id": "call_real", "name": "echo", "arguments": {"text": "final"}}
        ]
        assert m._pending_function_calls == [
            {"name": "echo", "call_id": "call_real"}
        ]


class TestReviewBugReproductions25:
    """End-to-end reproducers for bugs reported by review 25."""

    def test_message_with_missing_content_not_resent(
        self, capture_server: str
    ) -> None:
        """Replayed role messages with absent/None content must be dropped."""
        bad_response = json.dumps(
            {
                "id": "resp_bad_msg",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 1,
                },
            }
        ).encode()

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")

        _CapturingHandler.next_response_body = bad_response
        m.generate()

        m.conversation.append(
            {"role": "user", "content": [{"type": "input_text", "text": "next"}]}
        )

        _CapturingHandler.next_response_body = _text_response_json("ok").encode()
        m.generate()

        body = _last_body()
        for item in body["input"]:
            assert not (
                isinstance(item, dict)
                and item.get("role") == "assistant"
                and item.get("content") is None
            )

    def test_streaming_added_full_arguments_plus_delta_does_not_duplicate(
        self, capture_server: str
    ) -> None:
        """Full ``arguments`` on ``output_item.added`` + subsequent delta
        must NOT produce a doubled string that fails JSON parsing."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": '{"text":"hi"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '{"text":"hi"}',
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]

        _CapturingHandler.next_response_body = b"".join(frames)

        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        assert fcs == [
            {"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}
        ]


class TestReviewBugReproductions26:
    """Review #26 — 4 new bugs in v2."""

    def test_native_reasoning_effort_prevents_default_xhigh(
        self, capture_server: str
    ) -> None:
        """gpt-5.5 default ``xhigh`` must not overwrite caller's native ``reasoning.effort``."""
        m = OpenAICompatibleModel2(
            "gpt-5.5",
            base_url=capture_server,
            api_key="k",
            model_config={"reasoning": {"summary": "auto", "effort": "low"}},
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json().encode()
        m.generate()

        body = _last_body()
        assert body["reasoning"] == {"summary": "auto", "effort": "low"}

    def test_deepseek_text_tool_call_ids_are_unique_across_turns(
        self, capture_server: str
    ) -> None:
        """DeepSeek text-based tool-call IDs must be unique across multiple turns."""
        payload = '{"tool_calls": [{"name": "echo", "arguments": {"text": "hi"}}]}'

        m = OpenAICompatibleModel2(
            "deepseek/deepseek-r1",
            base_url=capture_server,
            api_key="k",
        )
        m.initialize("call echo")

        _CapturingHandler.next_response_body = _text_response_json(text=payload).encode()
        fcs1, _, _ = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "hi"})]
        )

        m.conversation.append(
            {"role": "user", "content": [{"type": "input_text", "text": "again"}]}
        )

        _CapturingHandler.next_response_body = _text_response_json(text=payload).encode()
        fcs2, _, _ = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        assert fcs1[0]["id"] != fcs2[0]["id"]

    def test_output_item_done_real_output_index_reorders_provisional_slots(
        self, capture_server: str
    ) -> None:
        """``response.output_item.done`` with real output_index must re-key provisional slots."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_b",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": 2,
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_b",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": '{"text":"b"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 3,
                    "item": {
                        "type": "function_call",
                        "id": "fc_a",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": 4,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_a",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": '{"text":"a"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 5,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]

        _CapturingHandler.next_response_body = b"".join(frames)

        fcs, _, _ = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        assert [fc["id"] for fc in fcs] == ["call_a", "call_b"]

    def test_streaming_message_output_item_added_used_when_completed_output_empty(
        self, capture_server: str
    ) -> None:
        """``response.output_item.added`` for complete message must populate content."""
        tokens: list[str] = []
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "hello from added",
                                "annotations": [],
                            }
                        ],
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]

        _CapturingHandler.next_response_body = b"".join(frames)

        text, _ = m.generate()

        assert text == "hello from added"
        assert tokens == ["hello from added"]


class TestReviewBugReproductions27:
    """Review #27 — 3 new bugs (bug 1 not reproducible: parser already unique)."""

    def test_streaming_stale_identityless_terminal_function_call_does_not_duplicate(
        self, capture_server: str
    ) -> None:
        """Stale terminal function_call without identity must not create a bogus slot."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": 1,
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_good",
                        "name": "echo",
                        "arguments": '{"text":"x"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "",
                                "name": "",
                                "arguments": '{"text":"x"}',
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]

        _CapturingHandler.next_response_body = b"".join(frames)

        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        assert fcs == [
            {"id": "call_good", "name": "echo", "arguments": {"text": "x"}}
        ]

    def test_output_item_added_message_plus_delta_does_not_duplicate(
        self, capture_server: str
    ) -> None:
        """``output_item.added`` message text must not duplicate with subsequent deltas."""
        tokens: list[str] = []

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "hello",
                                "annotations": [],
                            }
                        ],
                    },
                },
            ),
            _stream_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 2,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "hello",
                    "logprobs": [],
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]

        _CapturingHandler.next_response_body = b"".join(frames)

        text, _resp = m.generate()

        assert text == "hello"
        assert tokens == ["hello"]

    def test_add_function_results_is_atomic_on_mismatched_batch(
        self, capture_server: str
    ) -> None:
        """Batch add_function_results must be atomic — no partial mutation on error."""
        from kiss.core.kiss_error import KISSError

        response_json = json.dumps(
            {
                "id": "resp_tc",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": '{"text":"a"}',
                    },
                    {
                        "type": "function_call",
                        "id": "fc_2",
                        "call_id": "call_b",
                        "name": "other_tool",
                        "arguments": "{}",
                    },
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()

        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m.initialize("hi")
        _CapturingHandler.next_response_body = response_json

        m.generate_and_process_with_tools(
            function_map={"echo": _echo, "other_tool": lambda: "x"},
            tools_schema=[
                *_ECHO_TOOL_CHAT_SCHEMA,
                {
                    "type": "function",
                    "function": {
                        "name": "other_tool",
                        "description": "other",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        )

        before_conversation = list(m.conversation)
        before_pending = list(m._pending_function_calls)
        before_attachments = list(m._pending_tool_result_attachments)

        with pytest.raises(KISSError):
            m.add_function_results_to_conversation_and_return(
                [
                    ("echo", {"result": "ok"}),
                    ("wrong_name", {"result": "bad"}),
                ]
            )

        assert m.conversation == before_conversation
        assert m._pending_function_calls == before_pending
        assert m._pending_tool_result_attachments == before_attachments


class TestReviewBugReproductions28:
    """Review #28 — 2 new bugs."""

    def test_parallel_added_without_indexes_then_done_reordered_preserves_both(
        self, capture_server: str
    ) -> None:
        """Parallel function_call items added without output_index must survive later re-keying."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_b",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 2,
                    "item": {
                        "type": "function_call",
                        "id": "fc_a",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": 3,
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_b",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": '{"text":"b"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": 4,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_a",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": '{"text":"a"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 5,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]

        _CapturingHandler.next_response_body = b"".join(frames)

        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        assert fcs == [
            {"id": "call_a", "name": "echo", "arguments": {"text": "a"}},
            {"id": "call_b", "name": "echo", "arguments": {"text": "b"}},
        ]

    def test_streamed_preamble_replayed_before_function_call_when_final_omits_message(
        self, capture_server: str
    ) -> None:
        """Stream-only assistant text must replay BEFORE the function_call sibling."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "I will call echo.",
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 2,
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 3,
                    "item_id": "fc_1",
                    "output_index": 1,
                    "delta": '{"text":"hi"}',
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "function_call",
                                "id": "fc_1",
                                "call_id": "call_1",
                                "name": "echo",
                                "arguments": '{"text":"hi"}',
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 2,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 3,
                        },
                    },
                },
            ),
        ]

        _CapturingHandler.next_response_body = b"".join(frames)

        fcs, content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        assert content == "I will call echo."
        assert fcs == [
            {"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}
        ]

        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "hi"})]
        )

        m.token_callback = None
        _CapturingHandler.next_response_body = _text_response_json("done").encode()
        m.generate()

        input_items = _last_body()["input"]

        assistant_idx = next(
            i
            for i, item in enumerate(input_items)
            if isinstance(item, dict)
            and item.get("role") == "assistant"
            and "I will call echo." in json.dumps(item)
        )
        fc_idx = next(
            i
            for i, item in enumerate(input_items)
            if isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("call_id") == "call_1"
        )
        output_idx = next(
            i
            for i, item in enumerate(input_items)
            if isinstance(item, dict)
            and item.get("type") == "function_call_output"
            and item.get("call_id") == "call_1"
        )

        assert assistant_idx < fc_idx < output_idx


class TestReviewBugReproductions29:
    """Review #29 — 2 new bugs (3 tests with refusal variant)."""

    def test_stop_is_not_forwarded_to_responses_api(
        self, capture_server: str
    ) -> None:
        """v1-style ``stop`` must be stripped before responses.create()."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={"stop": ["END"]},
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json("ok").encode()
        text, _resp = m.generate()
        assert text == "ok"
        body = _last_body()
        assert "stop" not in body

    def test_streaming_content_part_done_used_when_completed_output_empty(
        self, capture_server: str
    ) -> None:
        """``response.content_part.done`` text must surface when terminal output is empty."""
        tokens: list[str] = []
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "hello from content_part.done",
                        "annotations": [],
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        text, _resp = m.generate()
        assert text == "hello from content_part.done"
        assert tokens == ["hello from content_part.done"]

    def test_streaming_content_part_done_refusal_used_when_completed_output_empty(
        self, capture_server: str
    ) -> None:
        """Refusal in ``response.content_part.done`` must surface as content."""
        tokens: list[str] = []
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {
                        "type": "refusal",
                        "refusal": "I can't help with that.",
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        text, _resp = m.generate()
        assert text == "I can't help with that."
        assert tokens == ["I can't help with that."]


class TestReviewBugReproductions30:
    """Reproducing tests for review 30 (gpt-5.5) bugs."""

    def test_relocated_added_arguments_still_suppress_duplicate_delta(
        self, capture_server: str
    ) -> None:
        """Relocating a full-arguments slot must preserve args_from_added."""
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_b",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": '{"text":"b"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 2,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_a",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": '{"text":"a"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 3,
                    "item_id": "fc_b",
                    "output_index": 1,
                    "delta": '{"text":"b"}',
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_a", "name": "echo", "arguments": {"text": "a"}},
            {"id": "call_b", "name": "echo", "arguments": {"text": "b"}},
        ]

    def test_output_item_added_message_not_emitted_when_completed_has_final_message(
        self, capture_server: str
    ) -> None:
        """Provisional output_item.added message text must not leak via token_callback."""
        tokens: list[str] = []
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "stale",
                                "annotations": [],
                            }
                        ],
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_1",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "final",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        text, _resp = m.generate()
        assert text == "final"
        assert "stale" not in tokens

    def test_output_text_done_suffix_is_emitted_after_partial_delta(
        self, capture_server: str
    ) -> None:
        """If .done extends prior deltas, token_callback must see the suffix."""
        tokens: list[str] = []
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "hel",
                },
            ),
            _stream_sse_event(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "sequence_number": 2,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "text": "hello",
                    "logprobs": [],
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        text, _resp = m.generate()
        assert text == "hello"
        assert "".join(tokens) == "hello"


class TestReviewBugReproductions31:
    """Reproducing tests for review 31 (gpt-5.5) bugs."""

    def test_restored_function_call_missing_name_is_rejected(
        self, capture_server: str
    ) -> None:
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m.initialize("hi")
        m.conversation.append(
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "",
                "arguments": "{}",
            }
        )
        m.conversation.append(
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
            }
        )
        before = len(_CapturingHandler.captured_requests)
        with pytest.raises(KISSError, match="function_call.*name"):
            m.generate()
        assert len(_CapturingHandler.captured_requests) == before

    def test_restored_function_call_arguments_must_be_string(
        self, capture_server: str
    ) -> None:
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m.initialize("hi")
        m.conversation.append(
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "echo",
                "arguments": {"text": "hi"},
            }
        )
        m.conversation.append(
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
            }
        )
        before = len(_CapturingHandler.captured_requests)
        with pytest.raises(KISSError, match="function_call.*arguments"):
            m.generate()
        assert len(_CapturingHandler.captured_requests) == before

    def test_restored_function_call_output_output_must_be_string(
        self, capture_server: str
    ) -> None:
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m.initialize("hi")
        m.conversation.append(
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "echo",
                "arguments": "{}",
            }
        )
        m.conversation.append(
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": {"ok": True},
            }
        )
        before = len(_CapturingHandler.captured_requests)
        with pytest.raises(KISSError, match="function_call_output.*output"):
            m.generate()
        assert len(_CapturingHandler.captured_requests) == before

    def test_restored_function_call_output_missing_output_is_rejected(
        self, capture_server: str
    ) -> None:
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m.initialize("hi")
        m.conversation.append(
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "echo",
                "arguments": "{}",
            }
        )
        m.conversation.append(
            {
                "type": "function_call_output",
                "call_id": "call_1",
            }
        )
        before = len(_CapturingHandler.captured_requests)
        with pytest.raises(KISSError, match="function_call_output.*output"):
            m.generate()
        assert len(_CapturingHandler.captured_requests) == before

    def test_completed_response_suffix_emitted_after_partial_delta(
        self, capture_server: str
    ) -> None:
        tokens: list[str] = []
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "hel",
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_1",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "hello",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        text, _resp = m.generate()
        assert text == "hello"
        assert tokens == ["hel", "lo"]


class TestReviewBugReproductions32:
    """Reproducing tests for review 32 (gpt-5.5) bugs."""

    def test_output_item_done_new_item_colliding_with_provisional_slot_preserves_both(
        self, capture_server: str
    ) -> None:
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")

        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_b",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": 2,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_a",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": '{"text":"a"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": 3,
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_b",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": '{"text":"b"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_a", "name": "echo", "arguments": {"text": "a"}},
            {"id": "call_b", "name": "echo", "arguments": {"text": "b"}},
        ]

    def test_reasoning_done_suffix_is_emitted_after_partial_delta(
        self, capture_server: str
    ) -> None:
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

        frames = [
            _stream_sse_event(
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "sequence_number": 1,
                    "item_id": "rs_1",
                    "output_index": 0,
                    "summary_index": 0,
                    "delta": "pla",
                },
            ),
            _stream_sse_event(
                "response.reasoning_summary_text.done",
                {
                    "type": "response.reasoning_summary_text.done",
                    "sequence_number": 2,
                    "item_id": "rs_1",
                    "output_index": 0,
                    "summary_index": 0,
                    "text": "plan",
                },
            ),
            _stream_sse_event(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "sequence_number": 3,
                    "item_id": "msg_1",
                    "output_index": 1,
                    "content_index": 0,
                    "text": "answer",
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-5.5",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 2,
                            "output_tokens_details": {"reasoning_tokens": 1},
                            "total_tokens": 3,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        text, _resp = m.generate()
        assert text == "answer"
        assert tokens == ["pla", "n", "answer"]
        assert thinking_events == [True, False]

    def test_streaming_terminal_identityless_function_call_is_rejected(
        self, capture_server: str
    ) -> None:
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _stream_sse_event(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": 1,
                "response": {
                    "id": "r",
                    "object": "response",
                    "created_at": 0,
                    "model": "gpt-4o",
                    "output": [
                        {
                            "type": "function_call",
                            "arguments": '{"text":"x"}',
                        }
                    ],
                    "usage": {
                        "input_tokens": 1,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 1,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 2,
                    },
                },
            },
        )
        with pytest.raises(KISSError, match="function_call|call_id|name"):
            m.generate_and_process_with_tools(
                function_map={"echo": _echo},
                tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
            )


class TestReviewBugReproductions33:
    """Reproducing tests for review 33 (gpt-5.5) bugs."""

    def test_function_call_arguments_done_real_output_index_reorders_slots(
        self, capture_server: str
    ) -> None:
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_b",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 2,
                    "item": {
                        "type": "function_call",
                        "id": "fc_a",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": "",
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "sequence_number": 3,
                    "item_id": "fc_b",
                    "output_index": 1,
                    "arguments": '{"text":"b"}',
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "sequence_number": 4,
                    "item_id": "fc_a",
                    "output_index": 0,
                    "arguments": '{"text":"a"}',
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 5,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_a", "name": "echo", "arguments": {"text": "a"}},
            {"id": "call_b", "name": "echo", "arguments": {"text": "b"}},
        ]

    def test_move_tool_slot_preserves_args_from_added_for_relocated_occupant(
        self, capture_server: str
    ) -> None:
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_b",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": '{"text":"b"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 2,
                    "item": {
                        "type": "function_call",
                        "id": "fc_a",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": '{"text":"a"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": 3,
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_b",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": '{"text":"b"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 4,
                    "item_id": "fc_a",
                    "output_index": 0,
                    "delta": '{"text":"a"}',
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 5,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_a", "name": "echo", "arguments": {"text": "a"}},
            {"id": "call_b", "name": "echo", "arguments": {"text": "b"}},
        ]


class TestReviewBugReproductions34:
    """Reproducing tests for review 34 (gpt-5.5) bugs."""

    def test_identityless_terminal_function_call_does_not_override_streamed_call(
        self, capture_server: str
    ) -> None:
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_good",
                        "name": "echo",
                        "arguments": '{"text":"good"}',
                    },
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-4o",
                        "output": [
                            {
                                "type": "function_call",
                                "arguments": '{"text":"stale"}',
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [
            {"id": "call_good", "name": "echo", "arguments": {"text": "good"}}
        ]

    def test_chat_completions_n_is_not_forwarded_to_responses_api(
        self, capture_server: str
    ) -> None:
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={"n": 2},
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json("ok").encode()
        text, _resp = m.generate()
        assert text == "ok"
        body = _last_body()
        assert "n" not in body

    def test_legacy_functions_are_not_forwarded_to_responses_api(
        self, capture_server: str
    ) -> None:
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={
                "functions": [
                    {"name": "echo", "parameters": {"type": "object"}}
                ],
                "function_call": {"name": "echo"},
            },
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json("ok").encode()
        text, _resp = m.generate()
        assert text == "ok"
        body = _last_body()
        assert "functions" not in body
        assert "function_call" not in body

    def test_replayed_message_missing_role_is_not_forwarded_invalid(
        self, capture_server: str
    ) -> None:
        bad_response = json.dumps(
            {
                "id": "resp_bad_msg",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "hello",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()
        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m.initialize("hi")
        _CapturingHandler.next_response_body = bad_response
        text, _resp = m.generate()
        assert text == "hello"
        m.conversation.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "next"}],
            }
        )
        _CapturingHandler.next_response_body = _text_response_json("ok").encode()
        m.generate()
        body = _last_body()
        for item in body["input"]:
            assert not (
                isinstance(item, dict)
                and item.get("type") == "message"
                and not item.get("role")
            ), "message input items must not be replayed without role"


class TestReviewBugReproductions35:
    """Reproducing tests for review 35 (gpt-5.5) bugs."""

    def test_added_full_arguments_plus_chunked_duplicate_deltas_do_not_corrupt(
        self, capture_server: str
    ) -> None:
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        frames = [
            _stream_sse_event("response.output_item.added", {
                "type": "response.output_item.added",
                "sequence_number": 1, "output_index": 0,
                "item": {"type": "function_call", "id": "fc_1",
                         "call_id": "call_1", "name": "echo",
                         "arguments": '{"text":"hi"}'},
            }),
            _stream_sse_event("response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta",
                "sequence_number": 2, "item_id": "fc_1",
                "output_index": 0, "delta": '{"text"',
            }),
            _stream_sse_event("response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta",
                "sequence_number": 3, "item_id": "fc_1",
                "output_index": 0, "delta": ':"hi"}',
            }),
            _stream_sse_event("response.completed", {
                "type": "response.completed", "sequence_number": 4,
                "response": {"id": "r", "object": "response",
                    "created_at": 0, "model": "gpt-4o", "output": [],
                    "usage": {"input_tokens": 1,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 1,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 2}},
            }),
        ]
        _CapturingHandler.next_response_body = b"".join(frames)
        fcs, _content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert fcs == [{"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}]

    def test_unsupported_image_mime_is_dropped(
        self, capture_server: str
    ) -> None:
        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        att = Attachment(data=b"<svg></svg>", mime_type="image/svg+xml")
        m.initialize("look", attachments=[att])
        _CapturingHandler.next_response_body = _text_response_json("ok").encode()
        m.generate()
        body = _last_body()
        user_msg = next(
            item for item in body["input"]
            if isinstance(item, dict) and item.get("role") == "user"
        )
        parts = user_msg["content"]
        assert not any(
            isinstance(p, dict) and p.get("type") == "input_image"
            for p in parts
        )

    def test_chat_completions_logprobs_is_not_forwarded_to_responses_api(
        self, capture_server: str
    ) -> None:
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k",
            model_config={"logprobs": True},
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json("ok").encode()
        text, _resp = m.generate()
        assert text == "ok"
        body = _last_body()
        assert "logprobs" not in body


class TestReviewBugReproductions36:
    """Reproducing tests for review 36 (gpt-5.5) bugs."""

    def test_reasoning_done_suffix_after_text_is_rebracketed(
        self, capture_server: str
    ) -> None:
        """A late reasoning .done suffix must be emitted inside a thinking block."""
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

        frames = [
            _stream_sse_event(
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "sequence_number": 1,
                    "item_id": "rs_1",
                    "output_index": 0,
                    "summary_index": 0,
                    "delta": "pla",
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
                    "delta": "answer",
                },
            ),
            _stream_sse_event(
                "response.reasoning_summary_text.done",
                {
                    "type": "response.reasoning_summary_text.done",
                    "sequence_number": 3,
                    "item_id": "rs_1",
                    "output_index": 0,
                    "summary_index": 0,
                    "text": "plan",
                },
            ),
            _stream_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": {
                        "id": "r",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-5.5",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_1",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "answer",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 2,
                            "output_tokens_details": {"reasoning_tokens": 1},
                            "total_tokens": 3,
                        },
                    },
                },
            ),
        ]

        _CapturingHandler.next_response_body = b"".join(frames)

        text, _resp = m.generate()

        assert text == "answer"
        assert tokens == ["pla", "answer", "n"]
        # Late reasoning suffix "n" must be re-bracketed.
        assert thinking_events == [True, False, True, False]


class TestReviewBugReproductions37:
    def test_chat_completions_penalties_are_not_forwarded_to_responses_api(
        self, capture_server: str
    ) -> None:
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            model_config={
                "presence_penalty": 0.5,
                "frequency_penalty": 0.25,
            },
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _text_response_json("ok").encode()

        text, _resp = m.generate()

        assert text == "ok"
        body = _last_body()
        assert "presence_penalty" not in body
        assert "frequency_penalty" not in body

    def test_user_message_between_function_call_and_output_is_rejected(
        self, capture_server: str
    ) -> None:
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
        )
        m.initialize("hi")

        _CapturingHandler.next_response_body = _tool_call_response_json(
            name="echo",
            arguments='{"text":"hello"}',
            call_id="call_abc",
        ).encode()

        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        # Bad caller behavior / restored bad conversation: user message before tool output.
        m.conversation.append(
            {"role": "user", "content": [{"type": "input_text", "text": "new user turn"}]}
        )

        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "hello"})]
        )

        before = len(_CapturingHandler.captured_requests)
        with pytest.raises(KISSError, match="function_call|function_call_output|user"):
            m.generate()

        assert len(_CapturingHandler.captured_requests) == before

    def test_usage_extraction_handles_dict_shaped_response(
        self, capture_server: str
    ) -> None:
        m = OpenAICompatibleModel2(
            "openrouter/anthropic/claude-opus-4-6",
            base_url=capture_server,
            api_key="k",
        )

        resp = {
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {
                    "cached_tokens": 10,
                    "cache_write_tokens": 7,
                },
                "output_tokens": 5,
                "output_tokens_details": {
                    "reasoning_tokens": 2,
                },
            }
        }

        assert m.extract_input_output_token_counts_from_response(resp) == (
            83,
            5,
            10,
            7,
        )


class TestReviewBugReproductions38:
    """Reproducers for review-38 findings (dict-shaped Responses support)."""

    def test_generate_parses_dict_shaped_response_text(self) -> None:
        """Dict-shaped response text is extracted by ``generate()``."""
        from kiss.core.models.openai_compatible_model2 import (
            OpenAICompatibleModel2,
        )

        class _Responses:
            def __init__(self, response: Any) -> None:
                self.response = response
                self.last_kwargs: dict[str, Any] | None = None

            def create(self, **kwargs: Any) -> Any:
                self.last_kwargs = kwargs
                return self.response

        class _Client:
            def __init__(self, response: Any) -> None:
                self.responses = _Responses(response)

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url="http://unused/v1", api_key="k"
        )
        m.initialize("hi")
        m.client = _Client(
            {
                "id": "r",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "ok",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )
        text, _resp = m.generate()
        assert text == "ok"

    def test_generate_raises_on_dict_shaped_failed_response(self) -> None:
        """``status="failed"`` on a dict-shaped response raises KISSError."""
        from kiss.core.kiss_error import KISSError
        from kiss.core.models.openai_compatible_model2 import (
            OpenAICompatibleModel2,
        )

        class _Responses:
            def __init__(self, response: Any) -> None:
                self.response = response

            def create(self, **kwargs: Any) -> Any:
                return self.response

        class _Client:
            def __init__(self, response: Any) -> None:
                self.responses = _Responses(response)

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url="http://unused/v1", api_key="k"
        )
        m.initialize("hi")
        m.client = _Client(
            {
                "id": "r",
                "object": "response",
                "status": "failed",
                "error": {"message": "boom"},
                "output": [],
            }
        )
        with pytest.raises(KISSError, match="boom|failed"):
            m.generate()

    def test_tools_path_parses_dict_shaped_function_call(self) -> None:
        """``function_call`` items in a dict-shaped response are surfaced."""
        from kiss.core.models.openai_compatible_model2 import (
            OpenAICompatibleModel2,
        )

        schema = [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo.",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            }
        ]

        class _Responses:
            def __init__(self, response: Any) -> None:
                self.response = response

            def create(self, **kwargs: Any) -> Any:
                return self.response

        class _Client:
            def __init__(self, response: Any) -> None:
                self.responses = _Responses(response)

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url="http://unused/v1", api_key="k"
        )
        m.initialize("hi")
        m.client = _Client(
            {
                "id": "r",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": '{"text":"hi"}',
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )
        fcs, content, _resp = m.generate_and_process_with_tools(
            function_map={"echo": lambda text: text},
            tools_schema=schema,
        )
        assert content == ""
        assert fcs == [{"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}]
        # Follow-up function_call_output must not raise; conversation has the
        # prior dict-shaped function_call item.
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "hi"})]
        )


class TestReviewBugReproductions39:
    """Reproducers for review-39 findings."""

    def test_stream_without_terminal_completed_raises(
        self, capture_server: str
    ) -> None:
        """Stream ending without ``response.completed`` raises KISSError.

        The Responses API contract requires every successful stream to
        terminate with a ``response.completed`` event.  A stream that
        merely runs out (e.g. truncated HTTP body, proxy disconnect)
        must NOT be accepted as a successful generation, because the
        last event may be a partial text delta or partial tool-call
        arguments fragment.
        """
        from kiss.core.kiss_error import KISSError
        from kiss.core.models.openai_compatible_model2 import (
            OpenAICompatibleModel2,
        )

        tokens: list[str] = []
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
        )
        m.initialize("hi")

        _CapturingHandler.next_response_body = _stream_sse_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "partial",
                "logprobs": [],
            },
        )

        with pytest.raises(
            KISSError, match="completed|terminal|truncated|stream"
        ):
            m.generate()
