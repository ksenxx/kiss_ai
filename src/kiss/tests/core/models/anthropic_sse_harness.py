# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A real local Anthropic endpoint for driving :class:`AnthropicModel`.

The ``anthropic`` SDK honours ``base_url``, so pointing a genuine client at
a ``ThreadingHTTPServer`` that speaks the real ``/v1/messages`` SSE wire
format exercises the whole stack — request shaping, the real SDK, real
sockets, real SSE parsing and the real abort watchdog — with no mocks,
patches or test doubles anywhere.

Every request body the server received is recorded, which is how a test
asserts what the adapter actually put on the wire.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any


def sse(event_type: str, payload: dict[str, Any]) -> bytes:
    """Render one Anthropic SSE event.

    Args:
        event_type: The SSE ``event:`` name (e.g. ``message_stop``).
        payload: The event object to serialise.

    Returns:
        The ``event: ...\\ndata: {...}\\n\\n`` bytes.
    """
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def text_message_stream(
    text: str = "ok", model_name: str = "claude-under-test"
) -> list[bytes]:
    """Render a complete assistant turn carrying a single text block.

    Args:
        text: The assistant text to stream.
        model_name: The model name echoed in ``message_start``.

    Returns:
        The SSE chunks of one full message, ready to write.
    """
    return [
        sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_harness",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model_name,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            },
        ),
        sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 4},
            },
        ),
        sse("message_stop", {"type": "message_stop"}),
    ]


class _Handler(BaseHTTPRequestHandler):
    """Records each request body and streams the server's scripted reply."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        """Record the request and write the scripted reply."""
        server: Any = self.server
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        with server.lock:
            server.requests.append(json.loads(raw or b"{}"))
            server.request_headers.append(
                {k.lower(): v for k, v in self.headers.items()}
            )
        body = b"".join(server.chunks)
        self.send_response(server.status)
        self.send_header("Content-Type", server.content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence the default stderr access log."""


class _DaemonServer(ThreadingHTTPServer):
    """Threading server whose request threads never block interpreter exit."""

    daemon_threads = True
    allow_reuse_address = True


class ScriptedAnthropicServer:
    """A real Anthropic ``/v1/messages`` endpoint answering a fixed stream.

    Use as a context manager; ``base_url`` is what the client under test
    receives, and ``requests`` accumulates every decoded request body.
    """

    def __init__(
        self,
        chunks: list[bytes] | None = None,
        status: int = 200,
        content_type: str = "text/event-stream",
    ) -> None:
        """Start the server on an ephemeral loopback port.

        Args:
            chunks: The bytes to answer every request with; defaults to
                the SSE chunks of a one-text-block assistant turn.
            status: The HTTP status code of every reply; a non-200 code
                turns the scripted reply into a real API error (give the
                error JSON as a single chunk).
            content_type: The Content-Type of every reply.
        """
        self._server = _DaemonServer(("127.0.0.1", 0), _Handler)
        self._server.chunks = chunks or text_message_stream()  # type: ignore[attr-defined]
        self._server.status = status  # type: ignore[attr-defined]
        self._server.content_type = content_type  # type: ignore[attr-defined]
        self._server.requests = []  # type: ignore[attr-defined]
        self._server.request_headers = []  # type: ignore[attr-defined]
        self._server.lock = threading.Lock()  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    @property
    def base_url(self) -> str:
        """Return the base URL to hand to an ``anthropic.Anthropic`` client."""
        return f"http://127.0.0.1:{self._server.server_port}"

    @property
    def requests(self) -> list[dict[str, Any]]:
        """Return a snapshot of the request bodies received so far."""
        with self._server.lock:  # type: ignore[attr-defined]
            return list(self._server.requests)  # type: ignore[attr-defined]

    @property
    def request_headers(self) -> list[dict[str, str]]:
        """Return a snapshot of the request headers received so far.

        Header names are lower-cased, one dict per request, in the same
        order as :attr:`requests`.
        """
        with self._server.lock:  # type: ignore[attr-defined]
            return list(self._server.request_headers)  # type: ignore[attr-defined]

    def stop(self) -> None:
        """Shut the server down and join its accept thread."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)

    def __enter__(self) -> ScriptedAnthropicServer:
        """Return the running server."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the server."""
        self.stop()
