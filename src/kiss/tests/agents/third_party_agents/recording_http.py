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

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, overload

from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer, stop_http_server


class RecordingServer(ThreadedHTTPServer):
    """HTTP server that records every request it handles."""

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        super().__init__(address, handler)
        self.requests: list[dict[str, Any]] = []

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
