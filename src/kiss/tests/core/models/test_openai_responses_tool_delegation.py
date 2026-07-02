# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for v1's Responses-API tool-calling delegation.

:class:`OpenAICompatibleModel` (Chat Completions) historically STRIPPED
``reasoning_effort`` from tool-bearing requests because OpenAI's
``/v1/chat/completions`` rejects ``tools`` + ``reasoning_effort`` for
GPT-5.x / o-series reasoning models.  The tool-calling transport is now
migrated to ``/v1/responses`` (via an internal
:class:`OpenAICompatibleModel2` delegate) so every effort level —
including ``"xhigh"`` — survives during tool-using agent runs.

A real :class:`http.server.ThreadingHTTPServer` captures every request so
the tests assert on the exact JSON travelling over the wire.  No mocks,
patches, fakes, or test doubles are used — only the upstream HTTP endpoint
is replaced by the in-process capture server.

Contract verified here:

* Tool-bearing requests with ``reasoning_effort`` go to ``POST
  /v1/responses`` with ``reasoning.effort`` preserved (``xhigh``/``low``/…)
  and flat Responses-shape tools.
* The conversation stays in Chat-Completions format (hand-off compatible):
  assistant tool_calls messages and ``role="tool"`` results are stored in
  chat shape and converted to ``function_call`` / ``function_call_output``
  items per request.
* Raw Responses output items (reasoning items included) are replayed
  verbatim on follow-up turns.
* Auto-detection: delegation only kicks in for ``api.openai.com`` unless
  forced via ``model_config["use_responses_api"]``; the legacy stripping
  fallback is preserved for other endpoints.
* The no-tools ``generate()`` path still uses ``/v1/chat/completions``
  with ``reasoning_effort`` intact.
* Responses-shaped usage objects are extracted correctly.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.core.models.model import Attachment
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

# ---------------------------------------------------------------------------
# Capture server: serves both /chat/completions and /responses.
# ---------------------------------------------------------------------------


def _chat_response_json(text: str = "chat-ok") -> bytes:
    """Return a minimal /v1/chat/completions non-streaming JSON body."""
    return json.dumps(
        {
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5.5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }
    ).encode()


def _responses_text_json(text: str = "resp-ok") -> bytes:
    """Return a minimal /v1/responses JSON body with only text output."""
    return json.dumps(
        {
            "id": "resp_text",
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
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens": 6,
                "output_tokens_details": {"reasoning_tokens": 2},
                "total_tokens": 16,
            },
        }
    ).encode()


def _responses_tool_call_json(
    name: str = "echo",
    arguments: str = '{"text": "hello"}',
    call_id: str = "call_abc",
) -> bytes:
    """Return a /v1/responses JSON body: reasoning item + function_call."""
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
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                },
            ],
            "usage": {
                "input_tokens": 5,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 4,
                "output_tokens_details": {"reasoning_tokens": 3},
                "total_tokens": 9,
            },
        }
    ).encode()


def _sse(event: str, data: dict[str, Any]) -> bytes:
    """Format one SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _responses_tool_call_sse(
    name: str = "echo",
    args_text: str = '{"text": "hello"}',
    call_id: str = "call_stream",
) -> bytes:
    """Build a streaming /v1/responses SSE body carrying a function_call."""
    item = {
        "type": "function_call",
        "id": "fc_s1",
        "call_id": call_id,
        "name": name,
        "arguments": "",
    }
    done_item = dict(item)
    done_item["arguments"] = args_text
    return b"".join(
        [
            _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": item,
                },
            ),
            _sse(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "item_id": "fc_s1",
                    "output_index": 0,
                    "delta": args_text,
                },
            ),
            _sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "id": "resp_s",
                        "object": "response",
                        "created_at": 0,
                        "model": "gpt-5.5",
                        "parallel_tool_calls": True,
                        "tool_choice": "auto",
                        "tools": [],
                        "output": [done_item],
                        "usage": {
                            "input_tokens": 3,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 5,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 8,
                        },
                    },
                },
            ),
        ]
    )


class _DelegationHandler(BaseHTTPRequestHandler):
    """Captures POST bodies; serves /chat/completions and /responses."""

    captured_requests: list[dict[str, Any]] = []
    # FIFO of scripted /responses bodies; empty -> default text body.
    responses_queue: list[bytes] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {}
        self.__class__.captured_requests.append({"path": self.path, "body": body})

        streaming = bool(body.get("stream"))
        if self.path.endswith("/chat/completions"):
            payload = _chat_response_json()
            content_type = "application/json"
        else:  # /responses
            if self.__class__.responses_queue:
                payload = self.__class__.responses_queue.pop(0)
            elif streaming:
                payload = _responses_tool_call_sse()
            else:
                payload = _responses_text_json()
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
    _DelegationHandler.captured_requests = []
    _DelegationHandler.responses_queue = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DelegationHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()


def _last_request() -> dict[str, Any]:
    assert _DelegationHandler.captured_requests, "no request reached the server"
    return _DelegationHandler.captured_requests[-1]


def echo(text: str) -> str:
    """Echo back ``text`` (test-only tool stub).

    Args:
        text: The string to echo.

    Returns:
        The input string unchanged.
    """
    return text


def _make_model(
    capture_server: str,
    model_name: str = "gpt-5.5-xhigh",
    model_config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> OpenAICompatibleModel:
    """Build a v1 model pointed at the capture server with delegation forced."""
    config: dict[str, Any] = {"use_responses_api": True}
    config.update(model_config or {})
    return OpenAICompatibleModel(
        model_name,
        base_url=capture_server,
        api_key="test-key",
        model_config=config,
        **kwargs,
    )


class TestReasoningEffortPreservedWithTools:
    """tools + reasoning_effort must reach /v1/responses un-stripped."""

    def test_xhigh_preserved_on_responses_endpoint(
        self, capture_server: str
    ) -> None:
        """gpt-5.5-xhigh + tools -> /responses with reasoning.effort=xhigh."""
        m = _make_model(capture_server)
        m.initialize("hi")
        _DelegationHandler.responses_queue = [_responses_tool_call_json()]
        function_calls, _, _ = m.generate_and_process_with_tools({"echo": echo})
        req = _last_request()
        assert req["path"].endswith("/responses")
        body = req["body"]
        assert body["model"] == "gpt-5.5"
        assert body["reasoning"]["effort"] == "xhigh"
        assert "reasoning_effort" not in body
        assert "use_responses_api" not in body
        # Tools must be in the flat Responses shape.
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["name"] == "echo"
        assert function_calls == [
            {"id": "call_abc", "name": "echo", "arguments": {"text": "hello"}}
        ]

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_other_effort_levels_preserved(
        self, capture_server: str, effort: str
    ) -> None:
        """Explicit reasoning_effort levels survive tool-bearing requests."""
        m = _make_model(
            capture_server, "gpt-5.5", {"reasoning_effort": effort}
        )
        m.initialize("hi")
        m.generate_and_process_with_tools({"echo": echo})
        req = _last_request()
        assert req["path"].endswith("/responses")
        assert req["body"]["reasoning"]["effort"] == effort

    def test_conversation_stays_chat_format(self, capture_server: str) -> None:
        """The assistant turn is stored in Chat-Completions shape."""
        m = _make_model(capture_server)
        m.initialize("hi")
        _DelegationHandler.responses_queue = [_responses_tool_call_json()]
        m.generate_and_process_with_tools({"echo": echo})
        last = m.conversation[-1]
        assert last["role"] == "assistant"
        assert last["tool_calls"] == [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": json.dumps({"text": "hello"}),
                },
            }
        ]


class TestFollowUpTurns:
    """Tool results and prior raw output items must replay correctly."""

    def test_function_call_output_and_reasoning_replayed(
        self, capture_server: str
    ) -> None:
        """Turn 2 replays reasoning + function_call items and the result."""
        m = _make_model(capture_server)
        m.initialize("hi")
        _DelegationHandler.responses_queue = [
            _responses_tool_call_json(),
            _responses_text_json("done"),
        ]
        m.generate_and_process_with_tools({"echo": echo})
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "hello"})]
        )
        _, content, _ = m.generate_and_process_with_tools({"echo": echo})
        assert content == "done"
        body = _last_request()["body"]
        types = [
            it.get("type") for it in body["input"] if isinstance(it, dict)
        ]
        assert "reasoning" in types  # raw reasoning item replayed verbatim
        fc = next(it for it in body["input"] if it.get("type") == "function_call")
        assert fc["call_id"] == "call_abc"
        assert fc["id"] == "fc_1"
        out = next(
            it for it in body["input"] if it.get("type") == "function_call_output"
        )
        assert out["call_id"] == "call_abc"
        assert out["output"].startswith("hello")
        # The original user prompt is still present as a message item.
        user = next(
            it
            for it in body["input"]
            if isinstance(it, dict) and it.get("role") == "user"
        )
        assert "hi" in json.dumps(user["content"])

    def test_system_instruction_travels_as_system_message(
        self, capture_server: str
    ) -> None:
        """v1's system message converts to a Responses system message item."""
        m = _make_model(
            capture_server,
            model_config={"system_instruction": "be terse"},
        )
        m.initialize("hi")
        m.generate_and_process_with_tools({"echo": echo})
        body = _last_request()["body"]
        system = next(
            it
            for it in body["input"]
            if isinstance(it, dict) and it.get("role") == "system"
        )
        assert system["content"] == "be terse"
        # Not duplicated via the top-level ``instructions`` field.
        assert not body.get("instructions")


class TestEndpointSelection:
    """Delegation must be scoped to endpoints that support /v1/responses."""

    def test_non_openai_endpoint_keeps_stripping_fallback(
        self, capture_server: str
    ) -> None:
        """Without the flag, a non-OpenAI base_url stays on chat.completions."""
        m = OpenAICompatibleModel(
            "gpt-5.5-xhigh",
            base_url=capture_server,
            api_key="test-key",
        )
        m.initialize("hi")
        m.generate_and_process_with_tools({"echo": echo})
        req = _last_request()
        assert req["path"].endswith("/chat/completions")
        assert "reasoning_effort" not in req["body"]

    def test_openrouter_endpoint_preserves_reasoning_effort_with_tools(
        self, capture_server: str
    ) -> None:
        """An openrouter.ai base_url keeps tools + reasoning_effort on
        chat.completions un-stripped.

        OpenRouter's Chat Completions accepts the combination (including
        ``"xhigh"``) and translates the effort per provider, so — unlike
        other non-OpenAI gateways — the effort must survive on the wire.
        The capture server's URL carries an ``openrouter.ai`` path segment
        so the real endpoint-detection logic sees an OpenRouter base_url.
        """
        m = OpenAICompatibleModel(
            "openrouter/openai/gpt-5.5-xhigh",
            base_url=f"{capture_server}/openrouter.ai",
            api_key="test-key",
        )
        assert m.model_config.get("reasoning_effort") == "xhigh"
        m.initialize("hi")
        m.generate_and_process_with_tools({"echo": echo})
        req = _last_request()
        assert req["path"].endswith("/chat/completions")
        assert req["body"]["reasoning_effort"] == "xhigh"
        assert req["body"]["tools"], "tools must still be attached"

    def test_flag_false_forces_chat_completions(
        self, capture_server: str
    ) -> None:
        """use_responses_api=False disables delegation even with effort set."""
        m = _make_model(
            capture_server, model_config={"use_responses_api": False}
        )
        m.initialize("hi")
        m.generate_and_process_with_tools({"echo": echo})
        req = _last_request()
        assert req["path"].endswith("/chat/completions")
        assert "reasoning_effort" not in req["body"]
        assert "use_responses_api" not in req["body"]

    def test_openai_host_auto_delegates(self) -> None:
        """api.openai.com is auto-detected without any flag."""
        m = OpenAICompatibleModel(
            "gpt-5.5-xhigh",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        assert m._should_delegate_to_responses() is True

    def test_generate_without_tools_stays_on_chat_completions(
        self, capture_server: str
    ) -> None:
        """The no-tools path keeps chat.completions + reasoning_effort."""
        m = _make_model(capture_server)
        m.initialize("hi")
        content, _ = m.generate()
        assert content == "chat-ok"
        req = _last_request()
        assert req["path"].endswith("/chat/completions")
        assert req["body"]["reasoning_effort"] == "xhigh"
        assert "use_responses_api" not in req["body"]

    def test_no_reasoning_effort_stays_on_chat_completions(
        self, capture_server: str
    ) -> None:
        """Models without reasoning_effort keep the chat tool transport."""
        m = OpenAICompatibleModel(
            "gpt-4o",
            base_url=capture_server,
            api_key="test-key",
            model_config={"use_responses_api": True},
        )
        m.initialize("hi")
        m.generate_and_process_with_tools({"echo": echo})
        assert _last_request()["path"].endswith("/chat/completions")


class TestStreamingDelegation:
    """Streaming tool turns must flow through the delegate."""

    def test_streamed_function_call_parsed(self, capture_server: str) -> None:
        """token_callback set -> streamed /responses function_call parsed."""
        tokens: list[str] = []
        m = _make_model(capture_server, token_callback=tokens.append)
        m.initialize("hi")
        function_calls, _, _ = m.generate_and_process_with_tools({"echo": echo})
        req = _last_request()
        assert req["path"].endswith("/responses")
        assert req["body"]["stream"] is True
        assert req["body"]["reasoning"]["effort"] == "xhigh"
        assert function_calls == [
            {"id": "call_stream", "name": "echo", "arguments": {"text": "hello"}}
        ]


class TestUsageExtraction:
    """Responses-shaped usage must be extracted by v1."""

    def test_responses_usage_extracted(self, capture_server: str) -> None:
        """input/output/cached/reasoning token fields map correctly."""
        m = _make_model(capture_server)
        m.initialize("hi")
        _DelegationHandler.responses_queue = [_responses_text_json()]
        _, _, response = m.generate_and_process_with_tools({"echo": echo})
        counts = m.extract_input_output_token_counts_from_response(response)
        # input_tokens=10 with 4 cached -> (6, 6, 4, 0)
        assert counts == (6, 6, 4, 0)

    def test_chat_usage_still_extracted(self, capture_server: str) -> None:
        """Chat-shaped usage keeps working after a delegated turn."""
        m = _make_model(capture_server)
        m.initialize("hi")
        _DelegationHandler.responses_queue = [_responses_text_json()]
        m.generate_and_process_with_tools({"echo": echo})
        _, response = m.generate()  # chat.completions path
        counts = m.extract_input_output_token_counts_from_response(response)
        assert counts == (7, 3, 0, 0)


class TestHandoffConversationDelegated:
    """Anthropic-format hand-off history must convert then delegate."""

    def test_anthropic_tool_blocks_become_function_call_items(
        self, capture_server: str
    ) -> None:
        """tool_use/tool_result blocks map to function_call(+output) items."""
        m = _make_model(capture_server)
        m.initialize("hi")
        m.conversation = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "calling"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "echo",
                        "input": {"text": "hi"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "hi",
                    }
                ],
            },
        ]
        _DelegationHandler.responses_queue = [_responses_text_json()]
        m.generate_and_process_with_tools({"echo": echo})
        body = _last_request()["body"]
        fc = next(it for it in body["input"] if it.get("type") == "function_call")
        assert fc["call_id"] == "toolu_1"
        assert fc["name"] == "echo"
        assert json.loads(fc["arguments"]) == {"text": "hi"}
        out = next(
            it for it in body["input"] if it.get("type") == "function_call_output"
        )
        assert out == {
            "type": "function_call_output",
            "call_id": "toolu_1",
            "output": "hi",
        }

    def test_image_content_part_becomes_input_image(
        self, capture_server: str
    ) -> None:
        """Chat image_url user parts convert to input_image items."""
        m = _make_model(capture_server)
        att = Attachment(data=b"\x89PNG\r\n\x1a\n", mime_type="image/png")
        m.initialize("look", attachments=[att])
        _DelegationHandler.responses_queue = [_responses_text_json()]
        m.generate_and_process_with_tools({"echo": echo})
        body = _last_request()["body"]
        user = next(
            it
            for it in body["input"]
            if isinstance(it, dict) and it.get("role") == "user"
        )
        types = {p.get("type") for p in user["content"] if isinstance(p, dict)}
        assert "input_image" in types
        assert "input_text" in types
