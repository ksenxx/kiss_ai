# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (C4): a bridge connection that never finishes its request retires.

Each accepted connection gets a handler thread that blocks in
``readline()`` before it authenticates or registers as a running call.
``ToolBridge._close`` joins the registered calls only (on purpose: a
tool that ignores the Stop must not hang the task), so a client that
connected and never sent its newline used to keep its handler thread,
its socket and a reference to the bridge's state alive for good after
the block ended.  The handler's socket I/O is now bounded by
``REQUEST_TIMEOUT_SECONDS``: the idle reader times out, logs, and its
socket is closed, while a request completed late is still answered with
the closed-bridge refusal (``test_audit1018_codex_bridge_stop``).
"""

from __future__ import annotations

import socket
import threading
import time

from kiss.core.models.codex_mcp_bridge import REQUEST_TIMEOUT_SECONDS, ToolBridge


def _handler_threads(before: set[threading.Thread]) -> list[threading.Thread]:
    return [
        t
        for t in threading.enumerate()
        if t not in before and "process_request_thread" in t.name
    ]


def test_idle_connections_do_not_keep_handler_threads_after_the_block() -> None:
    before = set(threading.enumerate())
    clients: list[socket.socket] = []
    bridge = ToolBridge([], lambda call: (call["name"], "unused"))
    try:
        with bridge:
            for _ in range(3):
                client = socket.create_connection(("127.0.0.1", bridge.port), timeout=5)
                client.sendall(b'{"op":')  # incomplete: no secret, no newline
                clients.append(client)
            deadline = time.monotonic() + 5
            while len(_handler_threads(before)) < 3 and time.monotonic() < deadline:
                time.sleep(0.01)
            handlers = _handler_threads(before)
            assert len(handlers) == 3
            left_at = time.monotonic()
        # Leaving the block does not wait for the idle readers...
        assert time.monotonic() - left_at < 2
        assert bridge.calls_in_flight == 0
        # ...but they retire by themselves once the socket bound elapses,
        # the clients still connected and still silent.
        deadline = time.monotonic() + 8
        for handler in handlers:
            handler.join(max(0.0, deadline - time.monotonic()))
        assert [t.is_alive() for t in handlers] == [False, False, False]
        assert 0 < REQUEST_TIMEOUT_SECONDS <= 5
        # Each client's connection was closed by the bridge: EOF, no reply.
        for client in clients:
            assert client.recv(1024) == b""
    finally:
        for client in clients:
            client.close()


def _trickle(client: socket.socket, stop: threading.Event) -> None:
    """Send a byte every 50 ms until *stop* or the connection ends (no newline, ever)."""
    while not stop.is_set():
        try:
            client.sendall(b" ")
        except OSError:
            return
        stop.wait(0.05)


def test_a_trickling_connection_retires_within_the_bound_too() -> None:
    """A client that keeps sending bytes without ever finishing its line
    would reset a per-read bound at every byte and keep its handler (and
    its growing line) alive for as long as it trickles, past the block.
    The bound covers the whole read: the handler retires within it, the
    client still trickling, and the client sees the connection end."""
    before = set(threading.enumerate())
    stop = threading.Event()
    bridge = ToolBridge([], lambda call: (call["name"], "unused"))
    client: socket.socket | None = None
    trickler: threading.Thread | None = None
    try:
        with bridge:
            client = socket.create_connection(("127.0.0.1", bridge.port), timeout=5)
            accepted_at = time.monotonic()
            trickler = threading.Thread(target=_trickle, args=(client, stop), daemon=True)
            trickler.start()
            deadline = accepted_at + 5
            while not _handler_threads(before) and time.monotonic() < deadline:
                time.sleep(0.01)
            (handler,) = _handler_threads(before)
            time.sleep(0.2)  # a few bytes in, the handler is still reading its line
            assert handler.is_alive()
            assert bridge.calls_in_flight == 0
            handler.join(REQUEST_TIMEOUT_SECONDS + 2)
            assert not handler.is_alive(), "the trickling client kept its handler alive"
            assert time.monotonic() - accepted_at >= REQUEST_TIMEOUT_SECONDS - 0.5
            trickler.join(5)
            assert not trickler.is_alive(), "the client's sends went on after the handler left"
    finally:
        stop.set()
        if trickler is not None:
            trickler.join(5)
        if client is not None:
            client.close()
