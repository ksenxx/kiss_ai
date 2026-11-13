# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``ChromeBridge.call``'s timeout bounds the socket write of the command too.

The timeout bounded the wait for the send lock and the wait for the
reply, but not ``conn.ws.send()`` in between: websockets' sync
connection writes with a blocking ``sendall`` that has no deadline, so
an authenticated extension that stops draining its socket (a frozen
service worker) filled the buffers and stranded the caller inside the
send for good, past any ``timeout``.  A watchdog now shuts the
connection's socket down at the deadline, which fails the blocked
write; the connection is dropped like any other failed send and a
fresh extension connection works afterwards.  The watchdog is cancelled
once the write returned, so a command that got on the wire in time
never has its connection cut later.

A real bridge on loopback and a real raw-socket peer (the handshake
done by hand, then never reading again, with tiny socket buffers on
both ends) are used; nothing is replaced.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import sys
import time
from collections.abc import Iterator
from typing import Any

import pytest
from websockets.frames import Frame, Opcode

from kiss.agents.sorcar.chrome_bridge import ChromeBridge, ChromeBridgeError
from kiss.tests.agents.sorcar.test_chrome_bridge import (
    ExtensionPeer,
    _call_in_thread,
    _eventually,
    _hello,
)

_BUFFER_BYTES = 4096
_PAYLOAD = "x" * (8 * 1024 * 1024)


class _StuckPeer:
    """An authenticated extension over a raw socket that stops reading after the handshake."""

    def __init__(self, bridge: ChromeBridge) -> None:
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _BUFFER_BYTES)
        self.sock.settimeout(5)
        self.sock.connect(("127.0.0.1", bridge.port))
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{bridge.port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: {key}\r\n\r\n"
            .encode()
        )
        headers = b""
        while not headers.endswith(b"\r\n\r\n"):
            headers += self.sock.recv(1)
        assert b"101 Switching Protocols" in headers
        nonce = self._receive_json()["nonce"]
        self._send_json(_hello(bridge.token, nonce))
        assert self._receive_json()["type"] == "hello_ok"

    def _receive_exact(self, size: int) -> bytes:
        data = b""
        while len(data) < size:
            part = self.sock.recv(size - len(data))
            assert part, "the bridge closed the socket"
            data += part
        return data

    def _receive_json(self) -> dict[str, Any]:
        first, second = self._receive_exact(2)
        assert first & 15 == Opcode.TEXT and not second & 128
        size = second & 127
        if size == 126:
            (size,) = struct.unpack("!H", self._receive_exact(2))
        elif size == 127:
            (size,) = struct.unpack("!Q", self._receive_exact(8))
        return dict(json.loads(self._receive_exact(size)))

    def _send_json(self, message: dict[str, Any]) -> None:
        self.sock.sendall(Frame(Opcode.TEXT, json.dumps(message).encode()).serialize(mask=True))

    def close(self) -> None:
        self.sock.close()


@pytest.fixture
def bridge() -> Iterator[ChromeBridge]:
    b = ChromeBridge(port=0, write_config=False)
    b.start()
    yield b
    b.stop()


def _answer(command: dict[str, Any]) -> Any:
    return "answered"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows loopback accepts a 256 MiB sendall at once whatever SO_SNDBUF and SO_RCVBUF "
    "say, so a write that cannot complete cannot be set up",
)
def test_a_send_the_extension_does_not_drain_fails_at_the_deadline(bridge: ChromeBridge) -> None:
    """A 0.5 s command whose 8 MiB frame cannot be written fails in time, not when the peer dies."""
    peer = _StuckPeer(bridge)
    thread = None
    try:
        assert bridge.wait_for_extension(5)
        conn = bridge._conn
        assert conn is not None
        conn.ws.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _BUFFER_BYTES)
        started = time.monotonic()
        thread, box = _call_in_thread(bridge, "evaluate", {"script": _PAYLOAD}, timeout=0.5)
        thread.join(5)
        assert not thread.is_alive(), "the send outlived its deadline (released only by the peer)"
        elapsed = time.monotonic() - started
        assert isinstance(box.get("error"), ChromeBridgeError), box
        assert "could not be sent within" in str(box["error"]), box
        assert elapsed < 3, elapsed
        assert conn.pending == {}
        assert not bridge.extension_connected  # the wedged peer was dropped
    finally:
        peer.close()
        if thread is not None:
            thread.join(10)
    # A later, healthy extension connection is untouched by the dead one's watchdog.
    fresh = ExtensionPeer(bridge.port, bridge.token, responder=_answer)
    try:
        assert bridge.wait_for_extension(5)
        assert bridge.call("list_tabs", timeout=5) == "answered"
    finally:
        fresh.close()


def test_a_peer_that_never_answers_does_not_hold_the_caller_for_the_close_handshake(
    bridge: ChromeBridge,
) -> None:
    """A small command the peer takes but never answers fails at its deadline, not after
    websockets' close handshake has waited its 10 s ``close_timeout`` for a close frame
    the peer does not send."""
    peer = _StuckPeer(bridge)
    thread = None
    try:
        assert bridge.wait_for_extension(5)
        started = time.monotonic()
        thread, box = _call_in_thread(bridge, "list_tabs", timeout=0.5)
        thread.join(5)
        assert not thread.is_alive(), "the call outlived its deadline (held by the close handshake)"
        elapsed = time.monotonic() - started
        assert isinstance(box.get("error"), ChromeBridgeError), box
        assert "did not answer" in str(box["error"]), box
        assert elapsed < 3, elapsed
        assert not bridge.extension_connected  # the silent peer was dropped
    finally:
        peer.close()
        if thread is not None:
            thread.join(10)
    fresh = ExtensionPeer(bridge.port, bridge.token, responder=_answer)
    try:
        assert bridge.wait_for_extension(5)
        assert bridge.call("list_tabs", timeout=5) == "answered"
    finally:
        fresh.close()


def test_replacing_a_peer_that_ignores_the_close_handshake_does_not_stall_the_new_one(
    bridge: ChromeBridge,
) -> None:
    """A newer extension connection replaces a wedged one and answers at once.

    Closing the replaced connection waits websockets' 10 s ``close_timeout``
    for a close frame this peer never sends; run on the new connection's
    reader thread, that wait left the healthy peer's replies undispatched,
    so a short command timed out and dropped the healthy connection too.
    """
    stuck = _StuckPeer(bridge)
    fresh = None
    try:
        assert bridge.wait_for_extension(5)
        replaced = bridge._conn
        # No readiness ping: it would wait a stalled reader out before the clock starts.
        fresh = ExtensionPeer(bridge.port, bridge.token, responder=_answer, wait_published=False)
        assert _eventually(lambda: bridge._conn is not None and bridge._conn is not replaced)
        started = time.monotonic()
        assert bridge.call("list_tabs", timeout=5) == "answered"
        assert time.monotonic() - started < 3
        assert bridge.extension_connected  # the healthy connection was kept
    finally:
        stuck.close()
        if fresh is not None:
            fresh.close()


def test_a_command_written_in_time_keeps_its_connection_past_the_deadline(
    bridge: ChromeBridge,
) -> None:
    """The watchdog of a send that returned is cancelled: nothing cuts the socket later."""
    peer = ExtensionPeer(bridge.port, bridge.token, responder=_answer)
    try:
        assert bridge.wait_for_extension(5)
        conn = bridge._conn
        assert conn is not None
        # Half the scripted peer's 1 MiB frame limit: a big but deliverable frame.
        assert bridge.call("evaluate", {"script": _PAYLOAD[: 512 * 1024]}, timeout=5) == "answered"
        assert bridge.call("list_tabs", timeout=0.3) == "answered"
        time.sleep(0.6)  # past that command's deadline
        assert bridge._conn is conn
        assert bridge.extension_connected
        assert bridge.call("list_tabs", timeout=5) == "answered"
    finally:
        peer.close()


if __name__ == "__main__":
    pytest.main([__file__])
