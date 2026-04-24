"""Shared helpers for channel backend polling and lifecycle management."""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from collections.abc import Callable
from http.server import HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server with per-request threads and address reuse enabled."""

    daemon_threads = True
    allow_reuse_address = True


def wait_for_matching_message(
    *,
    poll: Callable[[], list[dict[str, Any]]],
    matches: Callable[[dict[str, Any]], bool],
    extract_text: Callable[[dict[str, Any]], str],
    timeout_seconds: float,
    poll_interval: float,
) -> str | None:
    """Wait for a message matching a predicate with timeout.

    Args:
        poll: Callable returning newly observed messages.
        matches: Predicate selecting the desired message.
        extract_text: Callable extracting the reply text from a matching message.
        timeout_seconds: Maximum time to wait.
        poll_interval: Delay between polls.

    Returns:
        Extracted reply text, or ``None`` on timeout.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        for message in poll():
            if matches(message):
                return extract_text(message)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(poll_interval, remaining))


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
