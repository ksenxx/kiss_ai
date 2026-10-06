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
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from typing import Any, cast, overload
from urllib.parse import parse_qs, urlsplit

from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer, stop_http_server


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
    server = server_cls(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        stop_http_server(server, thread)


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


class JsonWebhookHandler(BaseHTTPRequestHandler):
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
        payload = json.dumps(server.response_body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence request logging."""


def serve_json_webhook() -> Iterator[JsonWebhookServer]:
    """Run a :class:`JsonWebhookServer` for one fixture (``yield from`` it)."""
    yield from serve_recording(JsonWebhookHandler, JsonWebhookServer)
