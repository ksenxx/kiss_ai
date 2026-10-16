# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared in-process HTTP test server that records every request.

Each test module supplies its own ``BaseHTTPRequestHandler`` that appends
a request dict to ``RecordingServer.requests`` and serves service-shaped
responses; :func:`serve_recording` runs the port-0 bind / daemon
``serve_forever`` / ``stop_http_server`` lifecycle so a module's fixture
is just ``yield from serve_recording(_Handler)``.  :func:`recording_server`
is the same lifecycle as a context manager for test bodies and for
fixtures that do more work around the server (``with recording_server(
_Handler) as server: ...``), where a ``for`` loop over the generator would
leave shutdown to garbage collection if the body raised.

``RecordingServer`` derives from the product's ``ThreadedHTTPServer``,
which is the stdlib ``ThreadingHTTPServer`` (per-request daemon threads)
plus ``SO_REUSEADDR`` on POSIX, so it is a drop-in replacement for the
private copies that subclassed either base.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from typing import Any, cast, overload
from urllib.parse import parse_qs, urlsplit

from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer, stop_http_server


def raw_post_status(port: int, headers: str, path: str = "/", body: bytes = b"") -> int:
    """Send a raw HTTP POST with custom headers and return the status code.

    Bypasses ``requests`` so malformed headers (a negative or non-numeric
    ``Content-Length``, an oversized body announcement) reach the server
    exactly as written.

    Args:
        port: Loopback port of the server under test.
        headers: Extra header lines, each terminated by ``\\r\\n``.
        path: Request path.
        body: Raw bytes sent after the headers (no ``Content-Length`` is
            added for them: the test writes whatever header it wants).

    Returns:
        The HTTP status code of the response's status line.
    """
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        request = f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n{headers}Connection: close\r\n\r\n"
        # latin-1, as ``http.server`` decodes header lines: a non-ASCII
        # header value reaches the server as the one character written.
        sock.sendall(request.encode("latin-1") + body)
        sock.settimeout(10)
        data = b""
        while b"\r\n" not in data:
            chunk = sock.recv(1024)
            if not chunk:
                break
            data += chunk
        return int(data.split(b" ")[1])


def start_serving(server: ThreadedHTTPServer) -> threading.Thread:
    """Serve *server* on a daemon thread; stop it with ``stop_http_server(server, thread)``.

    ``serve_forever`` polls its shutdown flag every 50 ms (the stdlib
    default of 0.5 s makes every teardown wait up to half a second).

    Args:
        server: A bound ``ThreadedHTTPServer`` (or subclass) not yet serving.

    Returns:
        The started thread, for ``stop_http_server``.
    """
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    return thread


@contextmanager
def serving[S: ThreadedHTTPServer](server: S) -> Iterator[S]:
    """Serve *server* on a daemon thread for the ``with`` body, then shut it down.

    Args:
        server: A bound ``ThreadedHTTPServer`` (or subclass) not yet serving.

    Yields:
        The running server.
    """
    thread = start_serving(server)
    try:
        yield server
    finally:
        stop_http_server(server, thread)


class RecordingServer(ThreadedHTTPServer):
    """HTTP server that records every request it handles.

    Every accepted connection gets a socket timeout, so a client that
    stalls mid-request cannot keep a handler thread blocked after the
    server is shut down.
    """

    connection_timeout = 60.0

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        super().__init__(address, handler)
        self.requests: list[dict[str, Any]] = []

    def get_request(self) -> tuple[Any, Any]:
        """Accept a connection and bound every read/write on it."""
        conn, addr = super().get_request()
        conn.settimeout(self.connection_timeout)
        return conn, addr

    @property
    def base_url(self) -> str:
        """``http://127.0.0.1:<port>`` of this server."""
        return f"http://127.0.0.1:{self.server_address[1]}"

    def header(self, name: str, index: int = -1) -> str:
        """Return a recorded request header (case-insensitive).

        Args:
            name: Header name.
            index: Which recorded request to inspect (default: last).

        Returns:
            The header value, or ``""`` when absent.
        """
        headers = self.requests[index]["headers"]
        return next((v for k, v in headers.items() if k.lower() == name.lower()), "")


@overload
def serve_recording(handler: type) -> Iterator[RecordingServer]: ...
@overload
def serve_recording[S: RecordingServer](handler: type, server_cls: type[S]) -> Iterator[S]: ...
def serve_recording(
    handler: type, server_cls: type[RecordingServer] = RecordingServer
) -> Iterator[RecordingServer]:
    """Run a recording server on an ephemeral loopback port for one fixture.

    Binds ``127.0.0.1:0``, serves on a daemon thread, yields the server,
    and shuts it down (``stop_http_server``) however the fixture exits.

    Args:
        handler: ``BaseHTTPRequestHandler`` subclass answering requests.
        server_cls: ``RecordingServer`` or a subclass adding extra knobs.

    Yields:
        The running server; ``server.server_address[1]`` is its port.
    """
    with serving(server_cls(("127.0.0.1", 0), handler)) as server:
        yield server


# :func:`serve_recording` as a context manager: ``with recording_server(_Handler) as server:``.
recording_server = contextmanager(serve_recording)


class JsonWebhookServer(RecordingServer):
    """Recording server answering every POST with one canned JSON body.

    Stands in for group-robot webhook endpoints (WeCom, DingTalk, ...)
    that accept a JSON POST and reply ``{"errcode": 0, ...}``.  Each
    request is recorded as ``{"path", "query", "json"}`` (``query`` is
    ``parse_qs`` of the URL query); a test changes ``response_body``
    before posting to simulate an API error.
    """

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        super().__init__(address, handler)
        self.response_body: dict[str, Any] = {"errcode": 0, "errmsg": "ok"}


class RecordingHandler(BaseHTTPRequestHandler):
    """Base for fake-service handlers: silent logging plus reply/body helpers.

    A test module's handler subclasses this and implements ``do_GET`` /
    ``do_POST`` with :meth:`_reply`, :meth:`_json` and :meth:`_read_body`
    instead of carrying its own copies of them.
    """

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence request logging."""

    def _reply(self, status: int, body: str, content_type: str = "application/json") -> None:
        """Send *body* (UTF-8) with *status* and a ``Content-Length``."""
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status: int, body: Any) -> None:
        """Send *body* as a JSON response."""
        self._reply(status, json.dumps(body))

    def _read_body(self) -> Any:
        """Return the request's JSON body: ``{}`` when empty, ``None`` when malformed."""
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            return None


class JsonWebhookHandler(RecordingHandler):
    """Handler for :class:`JsonWebhookServer`."""

    def do_POST(self) -> None:  # noqa: N802
        """Record the JSON body and reply with the server's ``response_body``."""
        server = cast(JsonWebhookServer, self.server)
        length = int(self.headers.get("Content-Length", 0))
        split = urlsplit(self.path)
        server.requests.append(
            {
                "path": split.path,
                "query": parse_qs(split.query),
                "json": json.loads(self.rfile.read(length).decode("utf-8")),
            }
        )
        self._json(200, server.response_body)


def serve_json_webhook() -> Iterator[JsonWebhookServer]:
    """Run a :class:`JsonWebhookServer` for one fixture (``yield from`` it)."""
    yield from serve_recording(JsonWebhookHandler, JsonWebhookServer)
