"""Integration test: AnthropicModel must NOT emit thinking UI events when
the streamed thinking block contains no thinking text (only signature deltas).

Reproduces the user-reported bug:

    "When I run a task with claude-opus-4-7 as the model, I mostly see
     'thinking' without other text in the thoughts panel."

claude-opus-4-7 uses ``thinking={"type": "adaptive"}``; when the model
decides not to think on a given turn, the streaming response still
contains a ``content_block_start`` with ``type: "thinking"`` followed by
only ``signature_delta`` events (no ``thinking_delta``) and a
``content_block_stop``.

Before the fix, ``AnthropicModel._create_message`` invoked
``thinking_callback(True)`` immediately on ``content_block_start`` and
``thinking_callback(False)`` on ``content_block_stop`` regardless of
whether any ``thinking_delta`` arrived in between.  The browser printer
then broadcast ``thinking_start`` and ``thinking_end`` events with no
``thinking_delta`` between them, so the thoughts panel showed the
"Thinking" label with no content.

The fix mirrors the one applied to ``ClaudeCodeModel``:

* Defer ``thinking_callback(True)`` until the first non-empty
  ``thinking_delta`` arrives.
* Only invoke ``thinking_callback(False)`` at ``content_block_stop`` if
  ``thinking_callback(True)`` was actually invoked.

Uses a real ThreadingHTTPServer that returns Anthropic-format SSE — no
mocks, patches, or fakes.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic
import pytest

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.core.models.anthropic_model import AnthropicModel

# ---------------------------------------------------------------------------
# Fake Anthropic streaming server — adaptive thinking with no thinking_delta
# ---------------------------------------------------------------------------


def _signature_only_thinking_events() -> list[tuple[str, str]]:
    """Build SSE event pairs where the thinking block has only signature deltas.

    This is the streaming shape Anthropic returns from claude-opus-4-7 with
    ``thinking={"type": "adaptive"}`` when the model decides not to think.
    """
    events: list[tuple[str, str]] = []
    events.append((
        "message_start",
        json.dumps({
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-opus-4-7",
                "stop_reason": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }),
    ))
    # Thinking block with ONLY a signature delta (no thinking_delta).
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
            "delta": {"type": "signature_delta", "signature": "sig_abc123"},
        }),
    ))
    events.append((
        "content_block_stop",
        json.dumps({"type": "content_block_stop", "index": 0}),
    ))
    # Real text answer.
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
            "delta": {"type": "text_delta", "text": "The answer is 42."},
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
            "usage": {"output_tokens": 12},
        }),
    ))
    events.append((
        "message_stop",
        json.dumps({"type": "message_stop"}),
    ))
    return events


def _real_thinking_events() -> list[tuple[str, str]]:
    """Build SSE pairs with a real (non-empty) thinking block.

    Used to confirm the fix does not regress normal extended-thinking output.
    """
    events: list[tuple[str, str]] = []
    events.append((
        "message_start",
        json.dumps({
            "type": "message_start",
            "message": {
                "id": "msg_real",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-opus-4-7",
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
    # signature_delta first, then real thinking_delta — exercises the
    # "deferred start" branch where thinking_callback(True) must fire on
    # the FIRST thinking_delta, not the earlier signature_delta.
    events.append((
        "content_block_delta",
        json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig_xyz"},
        }),
    ))
    events.append((
        "content_block_delta",
        json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Hmm, "},
        }),
    ))
    events.append((
        "content_block_delta",
        json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "let me see."},
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
            "delta": {"type": "text_delta", "text": "Done."},
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
            "usage": {"output_tokens": 5},
        }),
    ))
    events.append((
        "message_stop",
        json.dumps({"type": "message_stop"}),
    ))
    return events


# Module-level switch read by the request handler — flipped per-test before
# triggering an API call.  Avoids closures and global mocks.
_RESPONSE_EVENTS: list[tuple[str, str]] = []


class _AnthropicAdaptiveHandler(BaseHTTPRequestHandler):
    """Serves whichever event list ``_RESPONSE_EVENTS`` currently holds."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event_type, data in _RESPONSE_EVENTS:
            self.wfile.write(f"event: {event_type}\ndata: {data}\n\n".encode())
            self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture(scope="module")
def anthropic_server() -> Generator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AnthropicAdaptiveHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _build_opus_4_7_model(server_url: str, printer: BaseBrowserPrinter) -> AnthropicModel:
    """Return an AnthropicModel for claude-opus-4-7 wired to the fake server."""
    m = AnthropicModel(
        "claude-opus-4-7",
        api_key="test-key",
        token_callback=printer.token_callback,
        thinking_callback=printer.thinking_callback,
    )
    m.client = anthropic.Anthropic(api_key="test-key", base_url=server_url)
    m.conversation = [{"role": "user", "content": "What is 2+2?"}]
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpus47AdaptiveThinking:
    """Verify claude-opus-4-7 adaptive thinking does not show empty 'Thinking' panel."""

    def test_signature_only_thinking_block_emits_no_thinking_ui(
        self, anthropic_server: str
    ) -> None:
        """Bug reproduction: thinking block with only signature_delta must
        NOT produce thinking_start / thinking_end events.

        Before the fix, the user saw a "Thinking" label in the thoughts
        panel with no content because ``content_block_start`` for thinking
        immediately broadcast ``thinking_start`` even though no
        ``thinking_delta`` ever arrived.
        """
        global _RESPONSE_EVENTS
        _RESPONSE_EVENTS = _signature_only_thinking_events()

        printer = BaseBrowserPrinter()
        printer.start_recording()
        m = _build_opus_4_7_model(anthropic_server, printer)
        m._create_message(m._build_create_kwargs())

        recorded = printer.stop_recording()
        types = [e["type"] for e in recorded]

        assert "thinking_start" not in types, (
            f"Empty (signature-only) thinking block must not emit "
            f"thinking_start; got events: {types}"
        )
        assert "thinking_end" not in types, (
            f"Empty (signature-only) thinking block must not emit "
            f"thinking_end; got events: {types}"
        )
        assert "thinking_delta" not in types, (
            f"Empty thinking block must not emit thinking_delta; got: {types}"
        )

        # The real text answer must still be broadcast.
        text_deltas = [e for e in recorded if e["type"] == "text_delta"]
        full_text = "".join(e.get("text", "") for e in text_deltas)
        assert full_text == "The answer is 42.", full_text

    def test_real_thinking_block_still_streams_normally(
        self, anthropic_server: str
    ) -> None:
        """Real thinking_delta content must still produce thinking UI events.

        Also exercises the "signature_delta then thinking_delta" sequence
        to confirm the deferred start fires on the first thinking_delta,
        not the earlier signature_delta.
        """
        global _RESPONSE_EVENTS
        _RESPONSE_EVENTS = _real_thinking_events()

        printer = BaseBrowserPrinter()
        printer.start_recording()
        m = _build_opus_4_7_model(anthropic_server, printer)
        m._create_message(m._build_create_kwargs())

        recorded = printer.stop_recording()
        types = [e["type"] for e in recorded]

        assert types.count("thinking_start") == 1, types
        assert types.count("thinking_end") == 1, types

        start_idx = types.index("thinking_start")
        end_idx = types.index("thinking_end")
        between = recorded[start_idx + 1 : end_idx]
        thinking_deltas = [e for e in between if e["type"] == "thinking_delta"]
        thought = "".join(d["text"] for d in thinking_deltas)
        assert thought == "Hmm, let me see.", thought

        text_deltas = [e for e in recorded if e["type"] == "text_delta"]
        text = "".join(d.get("text", "") for d in text_deltas)
        assert text == "Done.", text

    def test_signature_only_block_skips_raw_thinking_callback(
        self, anthropic_server: str
    ) -> None:
        """Raw callback variant: no thinking_callback fires for signature-only blocks."""
        global _RESPONSE_EVENTS
        _RESPONSE_EVENTS = _signature_only_thinking_events()

        thinking_events: list[bool] = []
        tokens: list[str] = []

        m = AnthropicModel(
            "claude-opus-4-7",
            api_key="test-key",
            token_callback=tokens.append,
            thinking_callback=thinking_events.append,
        )
        m.client = anthropic.Anthropic(api_key="test-key", base_url=anthropic_server)
        m.conversation = [{"role": "user", "content": "hi"}]
        m._create_message(m._build_create_kwargs())

        assert thinking_events == [], thinking_events
        assert "The answer is 42." in "".join(tokens)

    def test_opus_4_7_uses_adaptive_thinking_config(self) -> None:
        """``_build_create_kwargs`` must request adaptive thinking for opus-4-7.

        This documents the precondition that triggers the original bug:
        adaptive thinking is what causes signature-only thinking blocks to
        be returned by the API.
        """
        m = AnthropicModel("claude-opus-4-7", api_key="test-key")
        m.conversation = [{"role": "user", "content": "hi"}]
        kwargs = m._build_create_kwargs()
        assert kwargs.get("thinking") == {"type": "adaptive"}, kwargs.get("thinking")
