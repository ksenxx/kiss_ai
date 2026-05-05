"""Integration test: AnthropicModel must enable interleaved thinking so
between-tool-call reasoning is streamed as ``thinking`` blocks (routed to
the Thoughts panel) rather than as plain ``text`` blocks (which would
render in the main response area).

Reproduces the user-reported bug:

    "In the last task, you showed the model thinking tokens outside the
     Thoughts panel."

Diagnosis from the recorded event log of the offending task (which used
``claude-opus-4-7`` with ``thinking={"type": "adaptive"}``): between
tool calls the model emitted reasoning text such as

    "I have the core facts. Let me verify a couple more things..."
    "I have what I need. Quick reality check first..."

These were broadcast as ``text_delta`` events because Anthropic's API,
without the ``interleaved-thinking-2025-05-14`` beta header, returns
between-action reasoning as ``text`` content blocks rather than as
``thinking`` blocks.

The fix: when extended thinking is enabled in
``AnthropicModel._build_create_kwargs``, attach
``extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"}``
so the API emits reasoning as ``thinking`` blocks.

Uses a real ThreadingHTTPServer (no mocks/patches/fakes) that:
  * captures the inbound ``anthropic-beta`` header,
  * returns one ``thinking`` block followed by one ``text`` block via SSE.
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


def _two_block_events() -> list[tuple[str, str]]:
    """Build SSE pairs: a thinking block, then a text block.

    Mirrors the shape Anthropic returns once interleaved thinking is
    enabled, where reasoning is emitted as ``thinking_delta`` and the
    final answer as ``text_delta``.
    """
    return [
        (
            "message_start",
            json.dumps({
                "type": "message_start",
                "message": {
                    "id": "msg_il",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-opus-4-7",
                    "stop_reason": None,
                    "usage": {"input_tokens": 8, "output_tokens": 0},
                },
            }),
        ),
        (
            "content_block_start",
            json.dumps({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            }),
        ),
        (
            "content_block_delta",
            json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Reasoning step."},
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
                "delta": {"type": "text_delta", "text": "Final answer."},
            }),
        ),
        (
            "content_block_stop",
            json.dumps({"type": "content_block_stop", "index": 1}),
        ),
        (
            "message_delta",
            json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 10},
            }),
        ),
        (
            "message_stop",
            json.dumps({"type": "message_stop"}),
        ),
    ]


_CAPTURED_HEADERS: dict[str, str] = {}
_HEADERS_LOCK = threading.Lock()


class _CapturingAnthropicHandler(BaseHTTPRequestHandler):
    """Captures inbound ``anthropic-beta`` header and replies with two-block SSE."""

    def do_POST(self) -> None:  # noqa: N802 — required HTTP handler name
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        with _HEADERS_LOCK:
            _CAPTURED_HEADERS["anthropic-beta"] = self.headers.get(
                "anthropic-beta", ""
            )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event_type, data in _two_block_events():
            self.wfile.write(f"event: {event_type}\ndata: {data}\n\n".encode())
            self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the default per-request stderr logging during tests."""


@pytest.fixture(scope="module")
def anthropic_server() -> Generator[str]:
    """Start a real HTTP server returning the two-block SSE stream."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CapturingAnthropicHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _build_model(
    model_name: str, server_url: str, printer: BaseBrowserPrinter
) -> AnthropicModel:
    m = AnthropicModel(
        model_name,
        api_key="test-key",
        token_callback=printer.token_callback,
        thinking_callback=printer.thinking_callback,
    )
    m.client = anthropic.Anthropic(api_key="test-key", base_url=server_url)
    m.conversation = [{"role": "user", "content": "What is 2+2?"}]
    return m


class TestInterleavedThinkingEnabled:
    """Confirm AnthropicModel asks for interleaved thinking and routes
    reasoning to the Thoughts panel."""

    def test_build_kwargs_attaches_interleaved_beta_for_opus_4_7(self) -> None:
        """``_build_create_kwargs`` must add the interleaved-thinking
        beta header for ``claude-opus-4-7`` (adaptive thinking)."""
        m = AnthropicModel("claude-opus-4-7", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        beta = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        assert "interleaved-thinking-2025-05-14" in beta, (
            f"Expected 'interleaved-thinking-2025-05-14' in anthropic-beta "
            f"header for claude-opus-4-7, got: {beta!r}.  Without it, "
            f"between-tool-call reasoning is streamed as text and shown "
            f"outside the Thoughts panel."
        )

    def test_build_kwargs_attaches_interleaved_beta_for_sonnet_4(self) -> None:
        """The fix must also apply to the sonnet-4 family."""
        m = AnthropicModel("claude-sonnet-4-5", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        beta = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        assert "interleaved-thinking-2025-05-14" in beta, beta

    def test_build_kwargs_no_beta_for_non_thinking_models(self) -> None:
        """Models without thinking enabled must not gain the beta header."""
        m = AnthropicModel("claude-3-5-sonnet-20241022", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        beta = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        assert "interleaved-thinking" not in beta, beta

    def test_user_supplied_beta_header_is_preserved(self) -> None:
        """A user-supplied ``anthropic-beta`` header must be augmented,
        not replaced, by the interleaved-thinking token."""
        m = AnthropicModel(
            "claude-opus-4-7",
            api_key="test-key",
            model_config={
                "extra_headers": {"anthropic-beta": "fine-grained-tool-streaming-2025-05-14"},
            },
        )
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        beta = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        assert "fine-grained-tool-streaming-2025-05-14" in beta, beta
        assert "interleaved-thinking-2025-05-14" in beta, beta

    def test_live_request_sends_interleaved_beta_header(
        self, anthropic_server: str
    ) -> None:
        """End-to-end: the real HTTP request to Anthropic must carry the
        interleaved-thinking beta token, and reasoning must surface as
        thinking_* events on the printer (Thoughts panel)."""
        with _HEADERS_LOCK:
            _CAPTURED_HEADERS.clear()

        printer = BaseBrowserPrinter()
        printer.start_recording()
        m = _build_model("claude-opus-4-7", anthropic_server, printer)
        m._create_message(m._build_create_kwargs())
        recorded = printer.stop_recording()

        with _HEADERS_LOCK:
            captured_beta = _CAPTURED_HEADERS.get("anthropic-beta", "")
        assert "interleaved-thinking-2025-05-14" in captured_beta, (
            f"AnthropicModel did not send the interleaved-thinking beta "
            f"header to the server.  Captured anthropic-beta: "
            f"{captured_beta!r}"
        )

        types = [e["type"] for e in recorded]
        assert "thinking_start" in types, types
        assert "thinking_delta" in types, types
        assert "thinking_end" in types, types

        thinking_text = "".join(
            e.get("text", "") for e in recorded if e["type"] == "thinking_delta"
        )
        text_text = "".join(
            e.get("text", "") for e in recorded if e["type"] == "text_delta"
        )
        assert thinking_text == "Reasoning step.", thinking_text
        assert text_text == "Final answer.", text_text
