# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: claude-fable-5 must actually reveal thinking tokens.

The user-reported bug this file locks in:

    "why are you not revealing thinking tokens, when you run a task
     using claude-fable-5?"

Two independent bugs in ``AnthropicModel._build_create_kwargs`` combined
to hide fable-5 thinking on every agentic turn:

1. **Missing ``display: "summarized"``.**  KISS sent
   ``thinking={"type": "adaptive"}``.  On Claude Fable 5 (and Mythos 5,
   Sonnet 5, Opus 4.8, Opus 4.7) ``thinking.display`` defaults to
   ``"omitted"``: the API returns thinking blocks with an EMPTY
   ``thinking`` field (encrypted signature only) and emits NO
   ``thinking_delta`` stream events, so ``thinking_callback`` never
   fires with any text.  Per the Anthropic docs
   (platform.claude.com/docs/en/build-with-claude/adaptive-thinking) the
   client must explicitly request
   ``thinking={"type": "adaptive", "display": "summarized"}``.

2. **Forced ``tool_choice={"type": "any"}``.**  For adaptive-thinking
   models with tools (KISSAgent always passes tools), KISS forced
   ``tool_choice={"type": "any"}``.  Tool use with thinking only
   supports ``tool_choice`` ``auto``/``none``; forcing tool use makes
   the API silently disable thinking for the request ("graceful
   thinking degradation"), so no thinking blocks are produced at all —
   verified against the live Anthropic API (``any`` → only
   ``tool_use`` blocks; ``auto`` → ``thinking`` + ``text`` +
   ``tool_use``).

Test strategy (no mocks, patches, or fakes):

* kwargs-level tests assert the wire request now carries
  ``display: "summarized"`` and no forced ``tool_choice``;
* an end-to-end SSE test drives a real ``anthropic`` client against a
  local ``ThreadingHTTPServer`` that (a) captures the exact JSON request
  body and (b) replays the adaptive-thinking stream shape the live API
  produces once the fix is in place, asserting that thinking text flows
  to ``thinking_callback``/``token_callback`` and to the ``JsonPrinter``
  event stream.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import anthropic
import pytest

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.server.json_printer import JsonPrinter

_FINISH_TOOL = {
    "name": "finish",
    "description": "Finish the task",
    "input_schema": {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
    },
}

_OPENAI_FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Finish the task",
        "parameters": {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        },
    },
}

_ADAPTIVE_MODELS = [
    "claude-fable-5",
    "claude-sonnet-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
]


class TestThinkingDisplaySummarized:
    """Adaptive-thinking requests must ask for summarized thinking text."""

    @pytest.mark.parametrize("name", _ADAPTIVE_MODELS)
    def test_adaptive_thinking_requests_summarized_display(self, name: str) -> None:
        """``thinking.display`` must be ``"summarized"`` for adaptive models.

        Without it, fable-5 defaults to ``display: "omitted"`` and returns
        thinking blocks whose ``thinking`` field is empty (signature only),
        with zero ``thinking_delta`` stream events — the user sees no
        thinking tokens at all.
        """
        m = AnthropicModel(name, api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        assert kwargs.get("thinking") == {
            "type": "adaptive",
            "display": "summarized",
        }, (name, kwargs.get("thinking"))

    def test_user_supplied_thinking_config_is_respected(self) -> None:
        """An explicit ``thinking`` in model_config must pass through unchanged."""
        m = AnthropicModel(
            "claude-fable-5",
            api_key="test-key",
            model_config={"thinking": {"type": "adaptive", "display": "omitted"}},
        )
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        assert kwargs.get("thinking") == {"type": "adaptive", "display": "omitted"}


class TestNoForcedToolChoiceWithThinking:
    """Forced tool use silently disables adaptive thinking — never send it."""

    @pytest.mark.parametrize("name", _ADAPTIVE_MODELS)
    def test_adaptive_thinking_does_not_force_tool_choice(self, name: str) -> None:
        """``tool_choice`` must be left at the API default (``auto``).

        ``tool_choice={"type": "any"}`` makes the Anthropic API silently
        drop thinking for adaptive-thinking models ("graceful thinking
        degradation"): the response contains only ``tool_use`` blocks and
        no thinking is ever revealed.
        """
        m = AnthropicModel(name, api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs(tools=[_FINISH_TOOL])
        assert "tool_choice" not in kwargs, (name, kwargs.get("tool_choice"))

    def test_non_thinking_model_still_forces_tool_use(self) -> None:
        """Models without thinking keep ``tool_choice=any`` for agentic turns."""
        m = AnthropicModel("claude-3-5-sonnet-20241022", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs(tools=[_FINISH_TOOL])
        assert "thinking" not in kwargs
        assert kwargs.get("tool_choice") == {"type": "any"}

    def test_user_supplied_tool_choice_is_respected(self) -> None:
        """An explicit ``tool_choice`` in model_config must pass through."""
        m = AnthropicModel(
            "claude-3-5-sonnet-20241022",
            api_key="test-key",
            model_config={"tool_choice": {"type": "auto"}},
        )
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs(tools=[_FINISH_TOOL])
        assert kwargs.get("tool_choice") == {"type": "auto"}


def _fable_thinking_tool_events() -> list[tuple[str, str]]:
    """SSE events replicating a live fable-5 adaptive+summarized+tools stream.

    Shape observed against the real API with
    ``thinking={"type": "adaptive", "display": "summarized"}`` and default
    ``tool_choice``: a thinking block streaming ``thinking_delta`` text,
    then a text block, then a ``tool_use`` block.
    """
    events: list[tuple[str, str]] = [
        (
            "message_start",
            json.dumps({
                "type": "message_start",
                "message": {
                    "id": "msg_fable",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-fable-5",
                    "stop_reason": None,
                    "usage": {"input_tokens": 25, "output_tokens": 0},
                },
            }),
        ),
        (
            "content_block_start",
            json.dumps({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            }),
        ),
        (
            "content_block_delta",
            json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "I'm calculating "},
            }),
        ),
        (
            "content_block_delta",
            json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "27 times 31."},
            }),
        ),
        (
            "content_block_delta",
            json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig_fable"},
            }),
        ),
        (
            "content_block_stop",
            json.dumps({"type": "content_block_stop", "index": 0}),
        ),
        (
            "content_block_start",
            json.dumps({
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            }),
        ),
        (
            "content_block_delta",
            json.dumps({
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "The answer is 837."},
            }),
        ),
        (
            "content_block_stop",
            json.dumps({"type": "content_block_stop", "index": 1}),
        ),
        (
            "content_block_start",
            json.dumps({
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_finish",
                    "name": "finish",
                    "input": {},
                },
            }),
        ),
        (
            "content_block_delta",
            json.dumps({
                "type": "content_block_delta",
                "index": 2,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"result": "837"}',
                },
            }),
        ),
        (
            "content_block_stop",
            json.dumps({"type": "content_block_stop", "index": 2}),
        ),
        (
            "message_delta",
            json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 40},
            }),
        ),
        ("message_stop", json.dumps({"type": "message_stop"})),
    ]
    return events


_CAPTURED_REQUESTS: list[dict[str, Any]] = []


class _FableHandler(BaseHTTPRequestHandler):
    """Captures the request body and replays the fable-5 thinking stream."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _CAPTURED_REQUESTS.append(json.loads(body))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event_type, data in _fable_thinking_tool_events():
            self.wfile.write(f"event: {event_type}\ndata: {data}\n\n".encode())
            self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture(scope="module")
def fable_server() -> Generator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FableHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


class TestFable5ThinkingEndToEnd:
    """End-to-end: fable-5 agentic turn must surface thinking tokens."""

    def test_agentic_turn_streams_thinking_to_callbacks(
        self, fable_server: str
    ) -> None:
        """Thinking text must reach thinking_callback/token_callback, and the
        wire request must carry ``display: "summarized"`` with no forced
        ``tool_choice``."""
        _CAPTURED_REQUESTS.clear()
        thinking_events: list[bool] = []
        tokens: list[str] = []

        m = AnthropicModel(
            "claude-fable-5",
            api_key="test-key",
            token_callback=tokens.append,
            thinking_callback=thinking_events.append,
        )
        m.client = anthropic.Anthropic(api_key="test-key", base_url=fable_server)
        m.conversation = [{"role": "user", "content": "What is 27*31? Then finish."}]

        tool_calls, content, _response = m.generate_and_process_with_tools(
            {}, tools_schema=[_OPENAI_FINISH_TOOL]
        )

        # The wire request must reveal thinking and must not force tool use.
        assert len(_CAPTURED_REQUESTS) == 1
        request = _CAPTURED_REQUESTS[0]
        assert request.get("thinking") == {
            "type": "adaptive",
            "display": "summarized",
        }, request.get("thinking")
        assert "tool_choice" not in request, request.get("tool_choice")

        # Thinking tokens must be revealed through the callbacks.
        assert thinking_events == [True, False], thinking_events
        streamed = "".join(tokens)
        assert "I'm calculating 27 times 31." in streamed, streamed
        assert "The answer is 837." in streamed, streamed

        # The tool call must still be extracted for the agent loop.
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "finish"

    def test_agentic_turn_streams_thinking_to_json_printer(
        self, fable_server: str
    ) -> None:
        """The JsonPrinter event stream must contain thinking_start,
        thinking_delta text, and thinking_end for a fable-5 agentic turn."""
        _CAPTURED_REQUESTS.clear()
        printer = JsonPrinter()
        printer._thread_local.task_id = "test-fable-thinking"
        printer.start_recording()

        m = AnthropicModel(
            "claude-fable-5",
            api_key="test-key",
            token_callback=printer.token_callback,
            thinking_callback=printer.thinking_callback,
        )
        m.client = anthropic.Anthropic(api_key="test-key", base_url=fable_server)
        m.conversation = [{"role": "user", "content": "What is 27*31? Then finish."}]

        m.generate_and_process_with_tools({}, tools_schema=[_OPENAI_FINISH_TOOL])

        recorded = printer.stop_recording()
        types = [e["type"] for e in recorded]
        assert types.count("thinking_start") == 1, types
        assert types.count("thinking_end") == 1, types

        start_idx = types.index("thinking_start")
        end_idx = types.index("thinking_end")
        thought = "".join(
            e["text"]
            for e in recorded[start_idx + 1 : end_idx]
            if e["type"] == "thinking_delta"
        )
        assert thought == "I'm calculating 27 times 31.", thought
