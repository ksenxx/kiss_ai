"""Integration test: AnthropicModel must invoke thinking_callback.

Claude models with extended thinking send ``thinking_delta`` events during
streaming.  The ``thinking_callback`` must be invoked with ``True`` at the
start of a thinking block and ``False`` at the end so that the browser UI
routes thinking tokens to the thinking panel rather than the main text area.

Bug reproduction: without the fix, thinking tokens arrive at the
``BaseBrowserPrinter.token_callback`` while ``_current_block_type`` is
still ``""`` (not ``"thinking"``), causing them to be broadcast as
``text_delta`` events — thoughts appear outside the thinking panel.

Uses a real ThreadingHTTPServer that returns Anthropic-format SSE — no
mocks, patches, or fakes.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.core.models.anthropic_model import AnthropicModel

# ---------------------------------------------------------------------------
# Fake Anthropic streaming server
# ---------------------------------------------------------------------------


def _anthropic_sse_events() -> list[tuple[str, str]]:
    """Build (event_type, data) pairs for an Anthropic-format SSE stream.

    Simulates a response with a thinking block followed by a text block.
    """
    events: list[tuple[str, str]] = []

    # message_start
    events.append((
        "message_start",
        json.dumps({
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }),
    ))

    # thinking block
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
            "delta": {"type": "thinking_delta", "thinking": "Let me think"},
        }),
    ))
    events.append((
        "content_block_delta",
        json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": " about this."},
        }),
    ))
    events.append((
        "content_block_stop",
        json.dumps({"type": "content_block_stop", "index": 0}),
    ))

    # text block
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

    # message_delta + message_stop
    events.append((
        "message_delta",
        json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 20},
        }),
    ))
    events.append((
        "message_stop",
        json.dumps({"type": "message_stop"}),
    ))

    return events


class _AnthropicHandler(BaseHTTPRequestHandler):
    """Serves Anthropic-format SSE streaming responses with thinking blocks."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        for event_type, data in _anthropic_sse_events():
            self.wfile.write(f"event: {event_type}\ndata: {data}\n\n".encode())
            self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture(scope="module")
def anthropic_server() -> Generator[str]:
    """Start a fake Anthropic server and yield its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AnthropicHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnthropicThinkingCallback:
    """Verify that AnthropicModel invokes thinking_callback for thinking blocks."""

    def test_thinking_callback_fires(self, anthropic_server: str) -> None:
        """thinking_callback must receive True then False around thinking block."""
        tokens: list[str] = []
        thinking_events: list[bool] = []

        m = AnthropicModel(
            "claude-sonnet-4-20250514",
            api_key="test-key",
            token_callback=lambda t: tokens.append(t),
            thinking_callback=lambda s: thinking_events.append(s),
        )
        m.client = m.client = __import__("anthropic").Anthropic(
            api_key="test-key", base_url=anthropic_server
        )
        m.conversation = [{"role": "user", "content": "Think about this."}]

        kwargs = m._build_create_kwargs()
        m._create_message(kwargs)

        assert True in thinking_events, (
            "thinking_callback(True) was never called — "
            "thinking tokens leak as text_delta events"
        )
        assert False in thinking_events, (
            "thinking_callback(False) was never called — "
            "thinking panel will never close"
        )
        first_true = thinking_events.index(True)
        last_false = len(thinking_events) - 1 - thinking_events[::-1].index(False)
        assert first_true < last_false

        combined = "".join(tokens)
        assert "Let me think" in combined

    def test_browser_printer_routes_thinking_tokens_correctly(
        self, anthropic_server: str
    ) -> None:
        """Thinking tokens must be broadcast as thinking_delta, not text_delta.

        This is the core bug reproduction: without thinking_callback, the
        BaseBrowserPrinter never sets _current_block_type to 'thinking', so
        thinking tokens are broadcast as text_delta events — thoughts appear
        outside the thinking panel.
        """
        printer = BaseBrowserPrinter()
        printer.start_recording()

        m = AnthropicModel(
            "claude-sonnet-4-20250514",
            api_key="test-key",
            token_callback=printer.token_callback,
            thinking_callback=printer.thinking_callback,
        )
        m.client = __import__("anthropic").Anthropic(
            api_key="test-key", base_url=anthropic_server
        )
        m.conversation = [{"role": "user", "content": "Think about this."}]

        kwargs = m._build_create_kwargs()
        m._create_message(kwargs)

        recorded = printer.stop_recording()
        types = [e["type"] for e in recorded]

        # Must have thinking_start / thinking_end events
        assert "thinking_start" in types, f"No thinking_start — types: {types}"
        assert "thinking_end" in types, f"No thinking_end — types: {types}"

        # Thinking tokens must be thinking_delta, not text_delta
        start_idx = types.index("thinking_start")
        end_idx = types.index("thinking_end")
        between = recorded[start_idx + 1 : end_idx]
        thinking_deltas = [e for e in between if e["type"] == "thinking_delta"]
        assert thinking_deltas, (
            "No thinking_delta events between thinking_start/end — "
            "thinking tokens leaked as text_delta"
        )

        # Verify the thinking text content
        thought_text = "".join(d["text"] for d in thinking_deltas)
        assert "Let me think" in thought_text

        # No thinking content should be in text_delta events
        text_deltas = [e for e in recorded if e["type"] == "text_delta"]
        text_content = "".join(d.get("text", "") for d in text_deltas)
        assert "Let me think" not in text_content, (
            f"Thinking text leaked into text_delta: {text_content}"
        )

    def test_model_factory_passes_thinking_callback(self) -> None:
        """model() factory must pass thinking_callback to AnthropicModel."""
        thinking_events: list[bool] = []

        from kiss.core.models.model_info import model

        m = model(
            "claude-sonnet-4-20250514",
            token_callback=lambda t: None,
            thinking_callback=lambda s: thinking_events.append(s),
        )
        assert m.thinking_callback is not None, (
            "model() factory did not pass thinking_callback to AnthropicModel"
        )

    def test_gemini_model_factory_passes_thinking_callback(self) -> None:
        """model() factory must pass thinking_callback to GeminiModel."""
        thinking_events: list[bool] = []

        from kiss.core.models.model_info import model

        m = model(
            "gemini-2.5-flash",
            token_callback=lambda t: None,
            thinking_callback=lambda s: thinking_events.append(s),
        )
        assert m.thinking_callback is not None, (
            "model() factory did not pass thinking_callback to GeminiModel"
        )
