# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: capability probes survive vendor models that require ``stream=true``.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.scripts.test_update_models_stream_only``; the non-core tests remain there.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel


class _StreamOnlyChatHandler(BaseHTTPRequestHandler):
    """An OpenAI-compatible chat-completions endpoint that requires streaming.

    POST ``/v1/chat/completions`` with ``"stream": true`` returns a tiny
    SSE response (one content delta plus a final usage chunk). Without
    streaming it returns the same HTTP 400 / ``invalid_request_error``
    that Together AI returns for stream-only Qwen variants.
    """

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = {}

        if not body.get("stream"):
            payload = json.dumps(
                {
                    "error": {
                        "message": ('This model only supports streaming. Set "stream": true.'),
                        "type": "invalid_request_error",
                        "code": "stream_required",
                    }
                }
            ).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        tool_present = bool(body.get("tools"))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(chunk: dict[str, object]) -> None:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        if tool_present:
            send(
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "stream-only-test",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "calculator",
                                            "arguments": '{"expression": "2+3"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                    "usage": None,
                }
            )
            send(
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "stream-only-test",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            )
        else:
            send(
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "stream-only-test",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "hello"},
                            "finish_reason": None,
                        }
                    ],
                    "usage": None,
                }
            )
            send(
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "stream-only-test",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            )

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *_args: object, **_kwargs: object) -> None:
        """Silence the default per-request stderr logging."""


@contextmanager
def _stream_only_server() -> Iterator[str]:
    """Spin up the stream-only chat server on a random loopback port.

    Yields the OpenAI-compatible ``base_url`` (``http://127.0.0.1:<port>/v1``).
    """
    server = HTTPServer(("127.0.0.1", 0), _StreamOnlyChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_stream_only_server_rejects_request_without_token_callback() -> None:
    """Reproduces the bug: without a token_callback the request is
    non-streaming, and the stream-only server returns HTTP 400.
    """
    with _stream_only_server() as base_url:
        m = OpenAICompatibleModel(
            model_name="stream-only-test",
            base_url=base_url,
            api_key="dummy",
        )
        m.initialize("Say hello in one word.")
        with pytest.raises(Exception) as exc:
            m.generate()
        assert "stream" in str(exc.value).lower()


def test_stream_only_server_succeeds_with_token_callback() -> None:
    """With a token_callback registered, _stream_text sets ``stream=True`` and
    the stream-only server returns the expected SSE response — this is the
    behaviour the fix in ``update_models.test_generate`` relies on.
    """
    received: list[str] = []

    with _stream_only_server() as base_url:
        m = OpenAICompatibleModel(
            model_name="stream-only-test",
            base_url=base_url,
            api_key="dummy",
            token_callback=received.append,
        )
        m.initialize("Say hello in one word.")
        text, _ = m.generate()
    assert text == "hello"
    assert "".join(received) == "hello"
