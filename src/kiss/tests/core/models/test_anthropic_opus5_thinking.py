# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: ``AnthropicModel`` must enable extended (adaptive)
thinking for ``claude-opus-5``.

The bug this test locks in:

    "When I use claude-opus-5 as the model for a task, the thinking
     tokens are not shown."

Root cause — ``claude-opus-5`` fell into the exact gap that previously
bit ``claude-fable-5`` (see test_anthropic_fable_thinking.py):

1. The ``MODEL_INFO.json`` entry for ``claude-opus-5`` carried no
   ``extended_thinking`` / ``adaptive_thinking`` flags, so the tri-state
   override in ``_supports_extended_thinking`` / ``_uses_adaptive_thinking``
   did not apply.
2. The legacy fallback heuristics only matched Claude *4.x* names:
   ``startswith(("claude-opus-4", "claude-sonnet-4", "claude-haiku-4"))``
   for extended thinking and the ``claude-opus-4-<minor>=6`` prefix rule
   for adaptive thinking.  ``claude-opus-5`` matched neither.

Consequences in ``AnthropicModel._build_create_kwargs``:

* no ``thinking`` request param was sent at all, so the API returned no
  thinking blocks / no ``thinking_delta`` SSE events and the thoughts
  panel stayed empty;
* the ``anthropic-beta: interleaved-thinking-2025-05-14`` extra header
  was never attached;
* ``tool_choice={"type": "any"}`` was force-set whenever tools were
  present, which by itself silently disables thinking on the API side;
* the ``max_tokens`` default was the plain 16384: the 64000-vs-65536
  selection lived inside the ``if _supports_extended_thinking(...)``
  block that opus-5 never entered, and the branch's own
  ``startswith("claude-opus-4")`` family test was version-blind anyway
  (it is now version-aware).

The fix (this test verifies every half):

1. ``MODEL_INFO.json`` now sets ``extended_thinking`` and
   ``adaptive_thinking`` to ``true`` for ``claude-opus-5``.
2. The fallback heuristics are version-aware: any
   ``claude-<family>-<major>`` name with family in
   opus/sonnet/haiku/fable and major >= 5 gets extended + adaptive
   thinking (the established Claude 4.x rules are preserved verbatim),
   so future families cannot regress the same way.
3. The ``max_tokens`` default is version-aware (Opus family -> 65536).

Includes a REAL end-to-end streaming test: a local
``ThreadingHTTPServer`` returns Anthropic-format SSE with a
``thinking_delta`` stream for ``claude-opus-5`` and the test asserts the
thinking text actually reaches the UI (``JsonPrinter`` events) and the
raw thinking callback.  No mocks, patches, or fakes of production code.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic
import pytest

from kiss.core.models.anthropic_model import (
    AnthropicModel,
    _supports_extended_thinking,
    _uses_adaptive_thinking,
)
from kiss.core.models.model_info import MODEL_INFO
from kiss.server.json_printer import JsonPrinter

_ANTHROPIC_FINISH_TOOL = {
    "name": "finish",
    "description": "Finish the task",
    "input_schema": {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
    },
}


class TestClaudeOpus5ThinkingConfig:
    """``claude-opus-5`` must be wired for adaptive extended thinking."""

    def test_model_info_declares_extended_thinking_flag(self) -> None:
        """The JSON catalog is the source of truth: flag must be True."""
        info = MODEL_INFO.get("claude-opus-5")
        assert info is not None, "claude-opus-5 missing from MODEL_INFO"
        assert info.extended_thinking is True, info.extended_thinking

    def test_model_info_declares_adaptive_thinking_flag(self) -> None:
        """opus-5 rejects thinking.type=enabled; adaptive must be flagged."""
        info = MODEL_INFO.get("claude-opus-5")
        assert info is not None, "claude-opus-5 missing from MODEL_INFO"
        assert info.adaptive_thinking is True, info.adaptive_thinking

    def test_supports_extended_thinking_helper_returns_true(self) -> None:
        assert _supports_extended_thinking("claude-opus-5") is True

    def test_uses_adaptive_thinking_helper_returns_true(self) -> None:
        assert _uses_adaptive_thinking("claude-opus-5") is True

    def test_build_kwargs_sets_thinking_adaptive_for_opus_5(self) -> None:
        """The wire request must carry adaptive thinking with summarized
        display — without ``display`` the API defaults to ``"omitted"``
        and returns signature-only (invisible) thinking blocks."""
        m = AnthropicModel("claude-opus-5", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        assert kwargs.get("thinking") == {
            "type": "adaptive",
            "display": "summarized",
        }, kwargs.get("thinking")

    def test_build_kwargs_attaches_interleaved_beta_for_opus_5(self) -> None:
        m = AnthropicModel("claude-opus-5", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        beta = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        assert "interleaved-thinking-2025-05-14" in beta, beta

    def test_build_kwargs_never_forces_tool_use_for_opus_5_tools(self) -> None:
        """Forced ``tool_choice=any`` silently disables adaptive thinking."""
        m = AnthropicModel("claude-opus-5", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs(tools=[_ANTHROPIC_FINISH_TOOL])
        assert "tool_choice" not in kwargs, kwargs.get("tool_choice")
        assert kwargs.get("thinking") == {
            "type": "adaptive",
            "display": "summarized",
        }, kwargs.get("thinking")

    def test_build_kwargs_uses_opus_max_tokens_default(self) -> None:
        """Opus-family thinking models default to 65536 max_tokens."""
        m = AnthropicModel("claude-opus-5", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        assert kwargs.get("max_tokens") == 65536, kwargs.get("max_tokens")

    def test_build_kwargs_respects_user_max_tokens(self) -> None:
        """A caller-supplied max_tokens must never be overridden."""
        m = AnthropicModel(
            "claude-opus-5", api_key="test-key", model_config={"max_tokens": 2048}
        )
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        assert kwargs.get("max_tokens") == 2048, kwargs.get("max_tokens")
        assert kwargs.get("thinking") == {
            "type": "adaptive",
            "display": "summarized",
        }, kwargs.get("thinking")

    def test_tiny_user_max_tokens_disables_enabled_thinking(self) -> None:
        """On the 4.x ``enabled``-thinking path, a user max_tokens too
        small for the 1024-token minimum budget must disable thinking."""
        m = AnthropicModel(
            "claude-haiku-4-5", api_key="test-key", model_config={"max_tokens": 512}
        )
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        assert "thinking" not in kwargs, kwargs.get("thinking")
        assert kwargs.get("max_tokens") == 512, kwargs.get("max_tokens")

    def test_non_opus_thinking_models_keep_64000_default(self) -> None:
        """The version-aware default keeps 64000 for non-Opus families."""
        for name in ("claude-sonnet-5", "claude-fable-5", "claude-sonnet-4-6"):
            m = AnthropicModel(name, api_key="test-key")
            m.conversation = [{"role": "user", "content": "ping"}]
            kwargs = m._build_create_kwargs()
            assert kwargs.get("max_tokens") == 64000, (name, kwargs.get("max_tokens"))

    def test_opus_4_families_keep_65536_default(self) -> None:
        """The old ``startswith("claude-opus-4")`` behaviour is preserved."""
        for name in ("claude-opus-4-1", "claude-opus-4-8"):
            m = AnthropicModel(name, api_key="test-key")
            m.conversation = [{"role": "user", "content": "ping"}]
            kwargs = m._build_create_kwargs()
            assert kwargs.get("max_tokens") == 65536, (name, kwargs.get("max_tokens"))


class TestVersionAwareHeuristicBranches:
    """Exhaustive branch coverage of the version-aware fallback heuristics.

    Names deliberately absent from ``MODEL_INFO`` exercise the pure
    name-parsing fallback (the tri-state override cannot apply).
    """

    @pytest.mark.parametrize(
        "name",
        [
            "gpt-4o",  # not a claude model at all
            "claude",  # too few segments
            "claude-opus",  # family without a version
            "claude-opus-x",  # malformed major version
            "claude-opus-04",  # zero-padded major is malformed
            "claude-opus-5-",  # empty trailing segment is malformed
            "claude-opus-5-junk",  # non-numeric trailing segment is malformed
            "claude-opus-5--20260101",  # empty segment before date is malformed
            "claude-opus-5-04",  # zero-padded minor is malformed
            "claude-opus-5-1-2",  # extra segment after the minor is malformed
            "claude-opus-5-20260301-99",  # non-final date segment is malformed
            "claude-opus-5-1-20260301-777",  # extra segment after the date
            "claude-opus-4-20250514-1",  # non-final date segment is malformed
            "claude-opus-4-20250514-20260101",  # double date is malformed
            "claude-3-5-sonnet-20241022",  # legacy claude-3 family
            "claude-3-haiku-20240307",  # legacy claude-3 family (alt shape)
            "claude-instant-1",  # pre-3 family, major < 4
        ],
    )
    def test_non_thinking_names(self, name: str) -> None:
        assert _supports_extended_thinking(name) is False, name
        assert _uses_adaptive_thinking(name) is False, name

    @pytest.mark.parametrize(
        "name",
        [
            "claude-opus-4",  # bare 4, no minor
            "claude-opus-4-1",  # 4.1 < 4.6
            "claude-opus-4-5",  # 4.5 < 4.6
            "claude-opus-4-1-20250805",  # dated snapshot keeps minor 1
            "claude-opus-4-20250514",  # dated snapshot of bare 4 (8-digit date)
            "claude-sonnet-4-6",  # sonnet 4.x never adaptive
            "claude-haiku-4-5",  # haiku 4.x never adaptive
            "claude-haiku-4-5-20251001",  # dated haiku snapshot
        ],
    )
    def test_extended_but_not_adaptive_4x(self, name: str) -> None:
        """Claude 4.x rules preserved: extended yes, adaptive only opus>=4.6."""
        assert _supports_extended_thinking(name) is True, name
        assert _uses_adaptive_thinking(name) is False, name

    @pytest.mark.parametrize(
        "name",
        [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
        ],
    )
    def test_opus_4_6_plus_is_adaptive(self, name: str) -> None:
        assert _supports_extended_thinking(name) is True, name
        assert _uses_adaptive_thinking(name) is True, name

    @pytest.mark.parametrize(
        "name",
        [
            "claude-opus-5",  # the reported bug
            "claude-opus-6",  # future major must not regress
            "claude-sonnet-6",  # not in MODEL_INFO: pure heuristic path
            "claude-haiku-5",  # not in MODEL_INFO: pure heuristic path
            "claude-fable-6",  # future fable major via heuristic
            "claude-opus-5-1",  # future 5.x minor
            "claude-opus-5-20260301",  # future dated 5 snapshot
        ],
    )
    def test_major_5_plus_is_adaptive(self, name: str) -> None:
        """Every modern-family Claude with major >= 5 thinks adaptively."""
        assert _supports_extended_thinking(name) is True, name
        assert _uses_adaptive_thinking(name) is True, name

    def test_explicit_false_flag_still_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The MODEL_INFO tri-state override stays the highest-priority
        source of truth: an explicit ``false`` must beat the heuristic."""
        info = MODEL_INFO["claude-opus-5"]
        monkeypatch.setattr(info, "extended_thinking", False)
        monkeypatch.setattr(info, "adaptive_thinking", False)
        assert _supports_extended_thinking("claude-opus-5") is False
        assert _uses_adaptive_thinking("claude-opus-5") is False
        m = AnthropicModel("claude-opus-5", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        assert "thinking" not in kwargs, kwargs.get("thinking")


def _opus5_thinking_events() -> list[tuple[str, str]]:
    """SSE pairs for a claude-opus-5 turn with a real thinking stream.

    Mirrors the wire shape Anthropic produces for adaptive-thinking
    models when ``display: "summarized"`` is requested: a thinking block
    with ``thinking_delta`` chunks (plus a trailing ``signature_delta``),
    followed by a text block.
    """
    events: list[tuple[str, str]] = []
    events.append((
        "message_start",
        json.dumps({
            "type": "message_start",
            "message": {
                "id": "msg_opus5",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-opus-5",
                "stop_reason": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }),
    ))
    events.append((
        "content_block_start",
        json.dumps({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        }),
    ))
    events.append((
        "content_block_delta",
        json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Pondering "},
        }),
    ))
    events.append((
        "content_block_delta",
        json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "the request."},
        }),
    ))
    events.append((
        "content_block_delta",
        json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig_opus5"},
        }),
    ))
    events.append((
        "content_block_stop",
        json.dumps({"type": "content_block_stop", "index": 0}),
    ))
    events.append((
        "content_block_start",
        json.dumps({
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        }),
    ))
    events.append((
        "content_block_delta",
        json.dumps({
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Four."},
        }),
    ))
    events.append((
        "content_block_stop",
        json.dumps({"type": "content_block_stop", "index": 1}),
    ))
    events.append((
        "message_delta",
        json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 9},
        }),
    ))
    events.append((
        "message_stop",
        json.dumps({"type": "message_stop"}),
    ))
    return events


def _opus5_signature_only_events() -> list[tuple[str, str]]:
    """SSE pairs where opus-5 decides not to think (signature-only block)."""
    events: list[tuple[str, str]] = []
    events.append((
        "message_start",
        json.dumps({
            "type": "message_start",
            "message": {
                "id": "msg_opus5_sig",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-opus-5",
                "stop_reason": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }),
    ))
    events.append((
        "content_block_start",
        json.dumps({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        }),
    ))
    events.append((
        "content_block_delta",
        json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig_only"},
        }),
    ))
    events.append((
        "content_block_stop",
        json.dumps({"type": "content_block_stop", "index": 0}),
    ))
    events.append((
        "content_block_start",
        json.dumps({
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        }),
    ))
    events.append((
        "content_block_delta",
        json.dumps({
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Quick answer."},
        }),
    ))
    events.append((
        "content_block_stop",
        json.dumps({"type": "content_block_stop", "index": 1}),
    ))
    events.append((
        "message_delta",
        json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 4},
        }),
    ))
    events.append((
        "message_stop",
        json.dumps({"type": "message_stop"}),
    ))
    return events


_RESPONSE_EVENTS: list[tuple[str, str]] = []
_LAST_REQUEST: dict[str, object] = {}


def _strip_thinking_events(
    events: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Drop every thinking-block SSE event from *events*.

    Mirrors the real Anthropic API: a request WITHOUT the ``thinking``
    param never receives thinking content blocks.  This keeps the
    end-to-end UI test coupled to the request gate — if the opus-5
    gating bug ever regressed (no ``thinking`` param sent), the served
    stream would carry no thinking events and the UI assertions would
    fail.
    """
    filtered: list[tuple[str, str]] = []
    thinking_indices: set[int] = set()
    for event_type, data in events:
        payload = json.loads(data)
        index = payload.get("index")
        if (
            event_type == "content_block_start"
            and payload.get("content_block", {}).get("type") == "thinking"
        ):
            thinking_indices.add(index)
            continue
        if index in thinking_indices:
            continue
        if isinstance(index, int):
            # Re-number so the surviving blocks stay contiguous — the
            # anthropic SDK accumulates blocks by index.
            payload["index"] = index - sum(1 for i in thinking_indices if i < index)
            data = json.dumps(payload)
        filtered.append((event_type, data))
    return filtered


class _AnthropicOpus5Handler(BaseHTTPRequestHandler):
    """Serves ``_RESPONSE_EVENTS`` and records the last request payload.

    Thinking-block events are served only when the request actually
    carries the ``thinking`` param, exactly like the real API.
    """

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        request_json = json.loads(body)
        _LAST_REQUEST.clear()
        _LAST_REQUEST["json"] = request_json
        _LAST_REQUEST["anthropic-beta"] = self.headers.get("anthropic-beta", "")
        events = _RESPONSE_EVENTS
        if "thinking" not in request_json:
            events = _strip_thinking_events(events)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event_type, data in events:
            self.wfile.write(f"event: {event_type}\ndata: {data}\n\n".encode())
            self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture(scope="module")
def anthropic_server() -> Generator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AnthropicOpus5Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _build_opus_5_model(server_url: str, printer: JsonPrinter) -> AnthropicModel:
    """Return an AnthropicModel for claude-opus-5 wired to the local server."""
    m = AnthropicModel(
        "claude-opus-5",
        api_key="test-key",
        token_callback=printer.token_callback,
        thinking_callback=printer.thinking_callback,
    )
    m.client = anthropic.Anthropic(api_key="test-key", base_url=server_url)
    m.conversation = [{"role": "user", "content": "What is 2+2?"}]
    return m


class TestOpus5EndToEndThinkingStream:
    """Real SSE streaming through a local server: thinking must reach the UI."""

    def test_thinking_text_reaches_ui_events(self, anthropic_server: str) -> None:
        """End-to-end bug reproduction: the thoughts panel must receive
        ``thinking_start`` / ``thinking_delta`` / ``thinking_end`` events
        carrying the streamed thinking text for claude-opus-5."""
        global _RESPONSE_EVENTS
        _RESPONSE_EVENTS = _opus5_thinking_events()

        printer = JsonPrinter()
        printer._thread_local.task_id = "test-opus5-thinking"
        printer.start_recording()
        m = _build_opus_5_model(anthropic_server, printer)
        m._create_message(m._build_create_kwargs())

        recorded = printer.stop_recording()
        types = [e["type"] for e in recorded]

        assert types.count("thinking_start") == 1, types
        assert types.count("thinking_end") == 1, types

        start_idx = types.index("thinking_start")
        end_idx = types.index("thinking_end")
        assert start_idx < end_idx, types
        thinking_deltas = [
            e for e in recorded[start_idx + 1 : end_idx] if e["type"] == "thinking_delta"
        ]
        thought = "".join(d["text"] for d in thinking_deltas)
        assert thought == "Pondering the request.", thought

        text_deltas = [e for e in recorded if e["type"] == "text_delta"]
        text = "".join(d.get("text", "") for d in text_deltas)
        assert text == "Four.", text

    def test_request_carries_thinking_param_and_beta_header(
        self, anthropic_server: str
    ) -> None:
        """The actual HTTP request for opus-5 must carry the adaptive
        thinking param and the interleaved-thinking beta header."""
        global _RESPONSE_EVENTS
        _RESPONSE_EVENTS = _opus5_thinking_events()

        printer = JsonPrinter()
        printer._thread_local.task_id = "test-opus5-request"
        printer.start_recording()
        m = _build_opus_5_model(anthropic_server, printer)
        m._create_message(m._build_create_kwargs())
        printer.stop_recording()

        payload = _LAST_REQUEST.get("json")
        assert isinstance(payload, dict), _LAST_REQUEST
        assert payload.get("thinking") == {
            "type": "adaptive",
            "display": "summarized",
        }, payload.get("thinking")
        assert payload.get("max_tokens") == 65536, payload.get("max_tokens")
        beta = _LAST_REQUEST.get("anthropic-beta", "")
        assert isinstance(beta, str)
        assert "interleaved-thinking-2025-05-14" in beta, beta

    def test_raw_thinking_callback_fires_true_then_false(
        self, anthropic_server: str
    ) -> None:
        """The raw thinking callback must fire True (on the first
        thinking_delta) then False (at block stop), and the thinking text
        must be delivered through the token callback in between."""
        global _RESPONSE_EVENTS
        _RESPONSE_EVENTS = _opus5_thinking_events()

        thinking_events: list[bool] = []
        tokens: list[str] = []

        m = AnthropicModel(
            "claude-opus-5",
            api_key="test-key",
            token_callback=tokens.append,
            thinking_callback=thinking_events.append,
        )
        m.client = anthropic.Anthropic(api_key="test-key", base_url=anthropic_server)
        m.conversation = [{"role": "user", "content": "hi"}]
        m._create_message(m._build_create_kwargs())

        assert thinking_events == [True, False], thinking_events
        joined = "".join(tokens)
        assert "Pondering the request." in joined, joined
        assert "Four." in joined, joined

    def test_signature_only_thinking_block_fires_no_callback(
        self, anthropic_server: str
    ) -> None:
        """A signature-only thinking block (opus-5 chose not to think)
        must NOT fire the thinking callback at all."""
        global _RESPONSE_EVENTS
        _RESPONSE_EVENTS = _opus5_signature_only_events()

        thinking_events: list[bool] = []
        tokens: list[str] = []

        m = AnthropicModel(
            "claude-opus-5",
            api_key="test-key",
            token_callback=tokens.append,
            thinking_callback=thinking_events.append,
        )
        m.client = anthropic.Anthropic(api_key="test-key", base_url=anthropic_server)
        m.conversation = [{"role": "user", "content": "hi"}]
        m._create_message(m._build_create_kwargs())

        assert thinking_events == [], thinking_events
        assert "Quick answer." in "".join(tokens)
