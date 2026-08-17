# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared helpers for channel backend polling and lifecycle management."""

from __future__ import annotations

import os
import queue
import sys
import threading
from collections.abc import Callable
from http.server import HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server with per-request threads and address reuse enabled."""

    daemon_threads = True
    allow_reuse_address = True


MAX_DRAIN_BYTES = 8 * 1024 * 1024
"""Upper bound on how much of a rejected request body is read and discarded.

Bounds :func:`drain_request_body` so a client claiming an enormous
``Content-Length`` cannot pin a handler thread; a genuinely oversized
sender past this bound still gets the connection reset, which is the
correct outcome for an abusive request.
"""


def drain_request_body(handler: Any, claimed_length: int | None) -> None:
    """Read and discard a request body the handler is not going to use.

    An HTTP handler that answers an error status (401/404/413/...)
    without reading the request body closes the connection with unread
    bytes in the socket's receive queue; the kernel then sends RST, the
    still-sending client's write fails, and the already-written response
    is discarded — the client sees ``ConnectionResetError`` instead of
    the status code.  Draining the (bounded) body lets the client finish
    sending and read the response.

    Call this AFTER writing the response.  A probe that claimed a
    ``Content-Length`` it never sends already has the status line by
    then: a probe that closes after reading it ends the drain with an
    immediate EOF, while one that instead reads until EOF stalls the
    handler only for the bounded per-read timeout below.  For a client
    mid-send of a large body the order is irrelevant: the small
    response fits the socket buffer, the drain unblocks the client's
    send, and the client then reads the response.

    Args:
        handler: The ``BaseHTTPRequestHandler`` serving the request.
        claimed_length: The parsed ``Content-Length``, or ``None`` when
            the header is missing or malformed (nothing can be safely
            read then — without a length the connection would block
            waiting for EOF the keep-alive client never sends).
    """
    if not claimed_length or claimed_length < 0:
        return
    try:
        handler.connection.settimeout(10.0)
        remaining = min(claimed_length, MAX_DRAIN_BYTES)
        while remaining > 0:
            chunk = handler.rfile.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
    except OSError:  # pragma: no cover — client gone mid-drain
        pass


def drain_queue_messages(
    message_queue: queue.Queue[dict[str, Any]],
    *,
    limit: int,
    keep: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Drain up to ``limit`` messages from a queue, optionally filtering.

    Args:
        message_queue: Queue containing message dicts.
        limit: Maximum number of kept messages to return.
        keep: Optional predicate deciding whether a drained message should be kept.

    Returns:
        The kept messages in dequeue order.
    """
    messages: list[dict[str, Any]] = []
    while len(messages) < limit:
        try:
            message = message_queue.get_nowait()
        except queue.Empty:
            break
        if keep is None or keep(message):
            messages.append(message)
    return messages


def stop_http_server(
    server: HTTPServer | None, server_thread: threading.Thread | None
) -> tuple[None, None]:
    """Shut down an embedded HTTP server and join its thread.

    Args:
        server: HTTP server instance to stop.
        server_thread: Background thread running ``serve_forever()``.

    Returns:
        ``(None, None)`` so callers can reset both attributes succinctly.
    """
    if server is not None:
        server.shutdown()
        server.server_close()
    if server_thread is not None:
        server_thread.join(timeout=5.0)
    return None, None


def is_headless_environment() -> bool:
    """Return True when running in a headless/Docker/Linux environment.

    Checks in order:
    1. KISS_HEADLESS env var (explicit override, "1"/"true"/"yes" → headless)
    2. Presence of /.dockerenv (running inside Docker)
    3. Linux with no $DISPLAY and no $WAYLAND_DISPLAY set
    """
    env = os.environ.get("KISS_HEADLESS", "").lower()
    if env in ("1", "true", "yes"):  # pragma: no branch
        return True
    if env in ("0", "false", "no"):  # pragma: no branch
        return False
    if Path("/.dockerenv").exists():  # pragma: no branch
        return True
    if sys.platform.startswith("linux"):  # pragma: no branch
        if (  # pragma: no branch
            not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
        ):
            return True
    return False
