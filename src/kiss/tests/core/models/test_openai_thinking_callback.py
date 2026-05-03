"""Integration test: OpenAI-compatible models must invoke thinking_callback.

gpt-5.5 (and other OpenAI reasoning models) send ``reasoning_content``
in streaming deltas.  The ``thinking_callback`` must be invoked with
``True`` at the start of the reasoning block and ``False`` at the end
so that the webview renders thought tokens inside a thinking panel.

Uses a real ThreadingHTTPServer that returns SSE chunks with
``reasoning_content`` fields — no mocks, patches, or fakes.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel


def _sse_chunks_with_reasoning() -> list[str]:
    """Build SSE chunks that simulate reasoning_content followed by content."""
    chunks = []
    chunks.append(
        json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "gpt-5.5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "Let me think...",
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )
    )
    chunks.append(
        json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "gpt-5.5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": " Step 1 done."},
                        "finish_reason": None,
                    }
                ],
            }
        )
    )
    chunks.append(
        json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "gpt-5.5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "The answer is 42."},
                        "finish_reason": None,
                    }
                ],
            }
        )
    )
    chunks.append(
        json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "gpt-5.5",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
    )
    chunks.append(
        json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "gpt-5.5",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                },
            }
        )
    )
    return chunks


def _sse_chunks_reasoning_only() -> list[str]:
    """Reasoning tokens only, no normal content — callback must still close."""
    chunks = []
    chunks.append(
        json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "gpt-5.5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "Only thinking.",
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )
    )
    chunks.append(
        json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "gpt-5.5",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            }
        )
    )
    return chunks


class _ReasoningHandler(BaseHTTPRequestHandler):
    """Serves SSE streaming responses with reasoning_content."""

    mode: str = "reasoning_then_content"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        if self.mode == "reasoning_only":
            chunks = _sse_chunks_reasoning_only()
        else:
            chunks = _sse_chunks_with_reasoning()

        for chunk in chunks:
            self.wfile.write(f"data: {chunk}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture(scope="module")
def reasoning_server() -> Generator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ReasoningHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


class TestOpenAIThinkingCallback:
    """Verify that thinking_callback fires for OpenAI reasoning_content."""

    def test_thinking_callback_fires_for_reasoning_content(
        self, reasoning_server: str
    ) -> None:
        """thinking_callback must receive True then False around reasoning."""
        _ReasoningHandler.mode = "reasoning_then_content"
        tokens: list[str] = []
        thinking_events: list[bool] = []

        m = OpenAICompatibleModel(
            "gpt-5.5",
            base_url=reasoning_server,
            api_key="test-key",
            token_callback=lambda t: tokens.append(t),
            thinking_callback=lambda s: thinking_events.append(s),
        )
        m.initialize("Think about this.")

        def dummy() -> str:
            """Dummy tool."""
            return "ok"

        fc, content, response = m.generate_and_process_with_tools({"dummy": dummy})

        assert True in thinking_events, (
            "thinking_callback(True) was never called — "
            "thought tokens won't appear in the webview"
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
        assert "The answer is 42" in content

    def test_thinking_callback_fires_for_generate(
        self, reasoning_server: str
    ) -> None:
        """thinking_callback must also work via generate() (no tools)."""
        _ReasoningHandler.mode = "reasoning_then_content"
        thinking_events: list[bool] = []

        m = OpenAICompatibleModel(
            "gpt-5.5",
            base_url=reasoning_server,
            api_key="test-key",
            token_callback=lambda t: None,
            thinking_callback=lambda s: thinking_events.append(s),
        )
        m.initialize("Think about this.")
        content, response = m.generate()

        assert True in thinking_events
        assert False in thinking_events

    def test_thinking_callback_closes_when_reasoning_only(
        self, reasoning_server: str
    ) -> None:
        """Even if there's only reasoning (no content), callback must close."""
        _ReasoningHandler.mode = "reasoning_only"
        thinking_events: list[bool] = []

        m = OpenAICompatibleModel(
            "gpt-5.5",
            base_url=reasoning_server,
            api_key="test-key",
            token_callback=lambda t: None,
            thinking_callback=lambda s: thinking_events.append(s),
        )
        m.initialize("Only think.")

        def dummy() -> str:
            """Dummy."""
            return "ok"

        fc, content, response = m.generate_and_process_with_tools({"dummy": dummy})
        assert True in thinking_events
        assert False in thinking_events

    def test_model_factory_passes_thinking_callback(
        self, reasoning_server: str
    ) -> None:
        """model() factory must pass thinking_callback to OpenAI models."""
        _ReasoningHandler.mode = "reasoning_then_content"
        thinking_events: list[bool] = []

        from kiss.core.models.model_info import model

        m = model(
            "gpt-5.5",
            model_config={"base_url": reasoning_server, "api_key": "test-key"},
            token_callback=lambda t: None,
            thinking_callback=lambda s: thinking_events.append(s),
        )
        m.initialize("Think about this.")
        content, response = m.generate()

        assert True in thinking_events, (
            "model() factory did not pass thinking_callback to OpenAICompatibleModel"
        )
        assert False in thinking_events
