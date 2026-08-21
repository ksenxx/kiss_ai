# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A real local OpenAI-compatible HTTP endpoint for end-to-end model tests.

The OpenAI SDK honours ``base_url``, so pointing a genuine
``OpenAICompatibleModel`` / ``OpenAICompatibleModel2`` at a
``ThreadingHTTPServer`` on ``127.0.0.1:0`` exercises the whole stack —
request shaping, the real SDK, real sockets, real SSE parsing, the real
streaming loops and the real abort watchdog — with no mocks, patches or
test doubles anywhere.

A test supplies a *responder*: a plain function from the received
request to the :class:`Reply` the server should send.  Replies are either
a JSON body or a list of pre-rendered SSE chunks, optionally followed by
holding the connection open (used to reproduce a wedged provider).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any

_HOLD_TIMEOUT = 60.0


@dataclass
class Request:
    """One request the scripted server received.

    Attributes:
        path: The request path (e.g. ``/v1/chat/completions``).
        body: The decoded JSON request body.
        connection_key: The client's ``host:port``, which identifies the
            TCP connection the request arrived on.  Counting distinct
            values is how connection-pool reuse is asserted.
        headers: The request headers, keys lower-cased — how a test
            asserts which credentials (``authorization``) and extra
            headers actually reached the endpoint.
    """

    path: str
    body: dict[str, Any]
    connection_key: str
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class Reply:
    """What the scripted server should answer with.

    Attributes:
        status: HTTP status code.
        json_body: A JSON body to send, or ``None`` when streaming.
        sse_chunks: Pre-rendered ``text/event-stream`` chunks, written and
            flushed one at a time.
        truncate_after: When set, only this many chunks are written and
            the connection is then dropped even though a full
            ``Content-Length`` was announced — a provider that crashes
            mid-stream.
        chunk_gate: When set, the handler waits for this event (bounded)
            after writing the first chunk, so a test can hold a stream
            open at an exact point without relying on sleeps.
        hold: When set, the handler keeps the connection open after
            writing ``sse_chunks``, emitting only SSE keep-alive comment
            lines (which the SDK filters out before yielding), until this
            event fires or the client hangs up — a provider that accepted
            the request and then went quiet.
    """

    status: int = 200
    json_body: dict[str, Any] | None = None
    sse_chunks: list[bytes] = field(default_factory=list)
    truncate_after: int | None = None
    chunk_gate: threading.Event | None = None
    hold: threading.Event | None = None


def chat_chunk(payload: dict[str, Any]) -> bytes:
    """Render one Chat Completions SSE chunk.

    Args:
        payload: The chunk object to serialise.

    Returns:
        The ``data: {...}\\n\\n`` bytes.
    """
    return f"data: {json.dumps(payload)}\n\n".encode()


def responses_event(event_type: str, payload: dict[str, Any]) -> bytes:
    """Render one Responses-API SSE event.

    Args:
        event_type: The SSE ``event:`` name (e.g. ``response.completed``).
        payload: The event object to serialise (its ``type`` is set to
            *event_type* when absent).

    Returns:
        The ``event: ...\\ndata: {...}\\n\\n`` bytes.
    """
    body = {"type": event_type, **payload}
    return f"event: {event_type}\ndata: {json.dumps(body)}\n\n".encode()


class _ScriptedHandler(BaseHTTPRequestHandler):
    """Answers every POST by asking the server's responder what to send."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        """Record the request and write the responder's reply."""
        server: Any = self.server
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        peer = self.connection.getpeername()
        request = Request(
            path=self.path,
            body=json.loads(raw or b"{}"),
            connection_key=f"{peer[0]}:{peer[1]}",
            headers={k.lower(): v for k, v in self.headers.items()},
        )
        with server.lock:
            server.requests.append(request)
        reply = server.responder(request)
        if reply.json_body is not None:
            self._write_json(reply)
            return
        self._write_sse(reply)

    def _write_json(self, reply: Reply) -> None:
        """Write a complete JSON reply, keeping the connection alive."""
        body = json.dumps(reply.json_body).encode()
        self.send_response(reply.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _write_sse(self, reply: Reply) -> None:
        """Write SSE chunks, optionally holding the connection open after."""
        self.send_response(reply.status)
        self.send_header("Content-Type", "text/event-stream")
        if reply.hold is None:
            body = b"".join(reply.sse_chunks)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._write_chunks(reply)
            if reply.truncate_after is not None:
                self.close_connection = True
            return
        server: Any = self.server
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self._write_chunks(reply)
        deadline = time.monotonic() + _HOLD_TIMEOUT
        while not reply.hold.wait(timeout=0.25):
            if time.monotonic() > deadline:
                return
            try:
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
            except OSError:
                server.client_disconnected.set()
                return

    def _write_chunks(self, reply: Reply) -> None:
        """Write each SSE chunk, pausing at ``chunk_gate`` after the first."""
        for index, chunk in enumerate(reply.sse_chunks):
            if reply.truncate_after is not None and index >= reply.truncate_after:
                return
            if index == 1 and reply.chunk_gate is not None:
                reply.chunk_gate.wait(timeout=_HOLD_TIMEOUT)
            self.wfile.write(chunk)
            self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence the default stderr access log."""


class _DaemonThreadingHTTPServer(ThreadingHTTPServer):
    """Threading server whose request threads never block interpreter exit."""

    daemon_threads = True
    allow_reuse_address = True


class ScriptedOpenAIServer:
    """A real OpenAI-compatible endpoint driven by a responder function.

    Use as a context manager; ``base_url`` is what the model under test
    receives.  ``requests`` accumulates every decoded request body, and
    ``connection_keys`` reports the distinct TCP connections used.
    """

    def __init__(self, responder: Callable[[Request], Reply]) -> None:
        """Start the server on an ephemeral loopback port.

        Args:
            responder: Called with each :class:`Request`; returns the
                :class:`Reply` to send.
        """
        self._server = _DaemonThreadingHTTPServer(
            ("127.0.0.1", 0), _ScriptedHandler
        )
        self._server.responder = responder  # type: ignore[attr-defined]
        self._server.requests = []  # type: ignore[attr-defined]
        self._server.lock = threading.Lock()  # type: ignore[attr-defined]
        self._server.client_disconnected = (  # type: ignore[attr-defined]
            threading.Event()
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    @property
    def base_url(self) -> str:
        """Return the ``/v1`` base URL to hand to a model under test."""
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    @property
    def requests(self) -> list[Request]:
        """Return a snapshot of the requests received so far."""
        with self._server.lock:  # type: ignore[attr-defined]
            return list(self._server.requests)  # type: ignore[attr-defined]

    @property
    def client_disconnected(self) -> threading.Event:
        """Return the event set when a held connection is dropped by the client.

        Writing to a socket the client has closed raises, which is the
        only server-side evidence that the model actually released the
        streamed response instead of stranding it.

        Returns:
            The event, set by the handler on the first failed write.
        """
        return self._server.client_disconnected  # type: ignore[attr-defined,no-any-return]

    @property
    def connection_keys(self) -> list[str]:
        """Return the distinct client connections seen, in first-use order."""
        seen: list[str] = []
        for request in self.requests:
            if request.connection_key not in seen:
                seen.append(request.connection_key)
        return seen

    def stop(self) -> None:
        """Shut the server down and join its accept thread."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)

    def __enter__(self) -> ScriptedOpenAIServer:
        """Return the running server."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Shut the server down."""
        self.stop()
