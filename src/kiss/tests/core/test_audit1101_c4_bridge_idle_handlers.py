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
``_BridgeHandler.timeout``: the idle reader times out, logs, and its
socket is closed, while a request completed late is still answered with
the closed-bridge refusal (``test_audit1018_codex_bridge_stop``).
"""

from __future__ import annotations

import socket
import threading
import time

from kiss.core.models.codex_mcp_bridge import ToolBridge, _BridgeHandler


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
        assert _BridgeHandler.timeout is not None and 0 < _BridgeHandler.timeout <= 5
        # Each client's connection was closed by the bridge: EOF, no reply.
        for client in clients:
            assert client.recv(1024) == b""
    finally:
        for client in clients:
            client.close()
