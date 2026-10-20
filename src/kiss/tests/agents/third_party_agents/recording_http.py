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


def raw_post(port: int, headers: str, path: str = "/", body: bytes = b"") -> tuple[int, bytes]:
    """Send a raw HTTP POST with custom headers; return the status code and response body.

    Bypasses ``requests`` so malformed headers (a negative or non-numeric
    ``Content-Length``, an oversized body announcement) reach the server
    exactly as written.  The response is read up to its ``Content-Length``
    (none: just the header block) rather than to EOF, so a server that
    answers an oversized announcement and then drains the body it was
    promised (``drain_request_body``) sees this probe close at once.

    Args:
        port: Loopback port of the server under test.
        headers: Extra header lines, each terminated by ``\\r\\n``.
        path: Request path.
        body: Raw bytes sent after the headers (no ``Content-Length`` is
            added for them: the test writes whatever header it wants).

    Returns:
        ``(status code, response body bytes)``.
    """
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        request = f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n{headers}Connection: close\r\n\r\n"
        # latin-1, as ``http.server`` decodes header lines: a non-ASCII
        # header value reaches the server as the one character written.
        sock.sendall(request.encode("latin-1") + body)
        data = b""
        while b"\r\n\r\n" not in data and (chunk := sock.recv(4096)):
            data += chunk
        head, _, payload = data.partition(b"\r\n\r\n")
        length = _response_content_length(head)
        while len(payload) < length and (chunk := sock.recv(4096)):
            payload += chunk
    return int(head.split(b" ", 2)[1]), payload[:length]


def _response_content_length(head: bytes) -> int:
    """Return the ``Content-Length`` announced in a response header block (0 when absent)."""
    for line in head.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            return int(value)
    return 0


def raw_post_status(port: int, headers: str, path: str = "/", body: bytes = b"") -> int:
    """:func:`raw_post` returning only the HTTP status code."""
    return raw_post(port, headers, path, body)[0]


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
    """Recording server answering every POST/PUT with one canned JSON status and body.

    Stands in for JSON endpoints (group-robot webhooks, typing
    indicators, bot APIs) whose reply the test fixes in advance.  Each
    request is recorded as ``{"method", "path", "query", "headers",
    "json"}`` (``query`` is ``parse_qs`` of the URL query, ``json`` the
    decoded body: ``{}`` when empty, ``None`` when malformed); a test
    changes ``response_status`` / ``response_body`` before the request to
    simulate an API error.
    """

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        super().__init__(address, handler)
        self.response_status = 200
        self.response_body: dict[str, Any] = {"errcode": 0, "errmsg": "ok"}


class RecordingHandler(BaseHTTPRequestHandler):
    """Base for fake-service handlers: silent logging plus reply/body helpers.

    A test module's handler subclasses this and implements ``do_GET`` /
    ``do_POST`` with :meth:`_reply`, :meth:`_json`, :meth:`_read_body`,
    :meth:`bearer_ok` and :meth:`record_request` instead of carrying its
    own copies of them.
    """

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence request logging."""

    def bearer_ok(self, token: str) -> bool:
        """Return whether the request carries ``Authorization: Bearer <token>``."""
        return self.headers.get("Authorization") == f"Bearer {token}"

    def record_request(self, body: Any, **extra: Any) -> dict[str, Any]:
        """Append this request (method, path, query, authorization, *body*, *extra*).

        ``path`` is the URL path without the query string; ``query`` is the
        parsed query (``parse_qs``).  Returns the recorded entry.
        """
        split = urlsplit(self.path)
        entry: dict[str, Any] = {
            "method": self.command,
            "path": split.path,
            "query": parse_qs(split.query),
            "authorization": self.headers.get("Authorization", ""),
            "body": body,
            **extra,
        }
        cast(RecordingServer, self.server).requests.append(entry)
        return entry

    def _reply(
        self, status: int, body: str | bytes, content_type: str = "application/json"
    ) -> None:
        """Send *body* (a ``str`` is UTF-8 encoded) with *status* and a ``Content-Length``."""
        payload = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status: int, body: Any, content_type: str = "application/json") -> None:
        """Send *body* as a JSON response."""
        self._reply(status, json.dumps(body), content_type)

    def _read_raw(self) -> bytes:
        """Return the request body announced by ``Content-Length`` (``b""`` when absent)."""
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def _read_body(self) -> Any:
        """Return the request's JSON body: ``{}`` when empty, ``None`` when malformed."""
        raw = self._read_raw()
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            return None


class JsonWebhookHandler(RecordingHandler):
    """Handler for :class:`JsonWebhookServer`.

    A service emulator whose reply depends on the request subclasses
    this, calls :meth:`record` and answers itself.
    """

    def record(self) -> dict[str, Any]:
        """Append this request to the server's ``requests`` and return the entry."""
        split = urlsplit(self.path)
        entry = {
            "method": self.command,
            "path": split.path,
            "query": parse_qs(split.query),
            "headers": dict(self.headers),
            "json": self._read_body(),
        }
        cast(RecordingServer, self.server).requests.append(entry)
        return entry

    def do_POST(self) -> None:  # noqa: N802
        """Record the request; reply with the server's ``response_status`` / ``response_body``."""
        server = cast(JsonWebhookServer, self.server)
        self.record()
        self._json(server.response_status, server.response_body)

    do_PUT = do_POST  # noqa: N815


def serve_json_webhook() -> Iterator[JsonWebhookServer]:
    """Run a :class:`JsonWebhookServer` for one fixture (``yield from`` it)."""
    yield from serve_recording(JsonWebhookHandler, JsonWebhookServer)


class EyreHandler(RecordingHandler):
    """Urbit Eyre emulator: records every request and answers login, pokes and scries.

    Each request is recorded as ``{"method", "path", "body"}`` with the
    body as text.  ``POST /~/login`` and ``POST``/``PUT /~/channel/<uid>``
    answer 204, ``GET /~/scry/channels/...`` an empty channel map, and
    anything else 404.  A module serving a second service on the same
    port overrides a ``do_*`` method, calls :meth:`_record` first and
    ends with ``elif not self._eyre(): <its own 404>``.
    """

    def _record(self) -> None:
        """Record this request, body as text."""
        body = self._read_raw()
        cast(RecordingServer, self.server).requests.append(
            {"method": self.command, "path": self.path, "body": body.decode() if body else ""}
        )

    def _eyre(self) -> bool:
        """Answer the request if it targets an Eyre endpoint; return whether it did."""
        if self.command == "GET" and self.path.startswith("/~/scry/channels/"):
            self._json(200, {"channels": {}})
        elif self.command == "PUT" and self.path.startswith("/~/channel"):
            self.send_response(204)
            self.end_headers()
        elif self.command == "POST" and (
            self.path == "/~/login" or self.path.startswith("/~/channel")
        ):
            self.send_response(204)
            self.end_headers()
        else:
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        """Record the request and answer it as the ship, 404 off Eyre's endpoints."""
        self._record()
        if not self._eyre():
            self._json(404, {"error": "Not Found", "status": 404})

    do_PUT = do_GET  # noqa: N815
    do_POST = do_GET  # noqa: N815


class PhoneApiHandler(RecordingHandler):
    """Phone companion app emulator: ``/api/device/info`` and ``/api/sms/messages``.

    ``messages`` is the canned ``/api/sms/messages`` answer (a subclass
    sets its own).  Each request is recorded as ``{"method", "path",
    "query"}`` (``query`` is ``parse_qs`` of the URL query, blank values
    kept) so a test can check the ``since`` cursor the backend sends.
    """

    messages: list[dict[str, Any]] = []

    def do_GET(self) -> None:  # noqa: N802
        """Answer the two phone endpoints with canned JSON, anything else with 404."""
        split = urlsplit(self.path)
        cast(RecordingServer, self.server).requests.append(
            {
                "method": self.command,
                "path": split.path,
                "query": parse_qs(split.query, keep_blank_values=True),
            }
        )
        if split.path == "/api/device/info":
            self._json(200, {"device_name": "testphone"})
        elif split.path == "/api/sms/messages":
            self._json(200, {"messages": self.messages})
        else:
            self._json(404, {"error": "Not Found"})
