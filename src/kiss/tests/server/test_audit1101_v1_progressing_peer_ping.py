# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The watchdog ping must not drop a peer that is still reading a transcript.

Audit 2026-11-01 (V1 F1): ``RemoteAccessServer._ping_one_ws`` bounded
the ping write by an absolute ``_send_timeout`` and the pong by an
absolute ``_WS_PING_TIMEOUT``.  The ping frame shares the connection's
TCP stream with the transcript parts queued ahead of it, so its pong
cannot arrive before the peer has read those bytes; a peer on a slow
link that was taking the transcript the whole time was aborted once
either deadline passed, and the replay started over on its reconnect.
The sender itself had already moved to a no-progress policy
(``WebPrinter._send_while_draining``: the deadline restarts whenever
the peer's unsent bytes shrink); the ping and pong waits now follow
the same policy (``_await_while_draining``), and the watchdog tick
keeps at most one ping in flight per peer instead of waiting for a
slow one.

Real ``RemoteAccessServer`` on loopback (the plain ``ws://`` path a
loopback browser uses), a real ``websockets`` client on a socket that
reads eight of its receive buffers a second, so the server's unsent
bytes shrink only as the client reads.  The receive buffer is what
the kernel grants a loopback socket that asked for 8 KiB
(:func:`_loopback_rcvbuf`): Linux gives 16 KiB, macOS 320 KiB
whatever was asked, and the peer learns of the reading only through
the window updates that follow it, which macOS sends once most of
that buffer is free again.  The frame is 24 such buffers, so the
transfer outlasts the shortened deadlines (``_send_timeout`` on the
printer, the module's ``_WS_PING_TIMEOUT``) several times over while
the pong deadline keeps a margin over the eighth of a second the
client needs to read its own receive buffer once the server's queue
has drained.  The reader paces itself by the clock rather than by
sleeping between reads (:class:`_SlowSocket`): a late wake-up on a
loaded machine is caught up, not added to the time between two
window updates.  The frame goes out whole, not as ``part`` frames: a
part handed to the transport whenever the one before it has drained
refills the queue between two looks of the drain watch, which tells
a refill from a stall only over its production-length deadline, not
over the four looks the shortened one gives it.  No mocks.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import IsolatedAsyncioTestCase

from websockets.sync.client import connect

import kiss.server.web_server as ws_mod
from kiss.core.user_config import CONFIG_PATH, save_config
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.conftest import find_free_port


def _loopback_rcvbuf() -> int:
    """Return the receive buffer the kernel grants a loopback socket that asked for 8 KiB.

    Linux grants the request (doubled, for its own bookkeeping);
    macOS sizes a loopback connection's receive buffer from the route
    when it connects, twenty segments of the 16 KiB loopback MTU,
    whatever was asked.  A peer sees nothing of such a socket's
    reading until that much is queued at it, so the frame and the
    reader's pace are sized from this.
    """
    with socket.socket() as listener, socket.socket() as client:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
        client.connect(listener.getsockname())
        with listener.accept()[0]:
            return client.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)


class _SlowSocket(socket.socket):
    """A socket whose owner reads at most *rate* bytes a second, counted from its first read.

    A read takes what the pace has allowed since the last one, so a
    reader that woke up late catches up instead of falling behind.
    """

    def __init__(self, rate: int) -> None:
        super().__init__(socket.AF_INET, socket.SOCK_STREAM)
        self.rate = rate
        self.started = 0.0
        self.taken = 0

    def recv(self, size: int, flags: int = 0) -> bytes:
        if not self.started:
            self.started = time.monotonic()
        while (allowed := int((time.monotonic() - self.started) * self.rate) - self.taken) < 1:
            time.sleep(1 / 64)
        data = super().recv(min(size, allowed), flags)
        self.taken += len(data)
        return data


@unittest.skipIf(
    sys.platform == "win32",
    "Windows loopback takes a whole frame at once (a 256 MiB sendall into 4 KiB buffers "
    "returns immediately), so the server's unsent bytes never track a slow reader and the "
    "scenario cannot be set up",
)
class TestProgressingPeerPing(IsolatedAsyncioTestCase):
    """A peer reading eight receive buffers a second keeps its connection through a ping."""

    async def asyncSetUp(self) -> None:
        self.rcvbuf = _loopback_rcvbuf()
        self.payload = base64.b64encode(os.urandom(18 * self.rcvbuf)).decode("ascii")
        self.port = find_free_port()
        self._orig_config = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        save_config({"remote_password": ""})
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, work_dir=tempfile.mkdtemp(), use_tunnel=False,
        )
        self.server._printer._send_timeout = 0.5
        self._orig_pong_timeout = ws_mod._WS_PING_TIMEOUT
        ws_mod._WS_PING_TIMEOUT = 1.0
        await self.server.start_async()
        for task in (self.server._version_check_task, self.server._watchdog_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def asyncTearDown(self) -> None:
        ws_mod._WS_PING_TIMEOUT = self._orig_pong_timeout
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    def _read_slowly(
        self, results: list[str], authed: threading.Event, release: threading.Event,
    ) -> None:
        """Read one big frame, then wait for the heartbeat behind the pong.

        The connection stays open until *release* is set, so the test
        can look at the server's side of it first.
        """
        sock = _SlowSocket(8 * self.rcvbuf)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
        sock.connect(("127.0.0.1", self.port))
        try:
            with connect(
                f"ws://127.0.0.1:{self.port}/ws", sock=sock, ping_interval=None,
                compression=None, max_size=None, close_timeout=0.2,
            ) as client:
                client.send(json.dumps({"type": "auth", "password": ""}))
                while True:
                    wire = client.recv(timeout=20)
                    assert isinstance(wire, str), "the daemon sends text frames"
                    frame = json.loads(wire)
                    if frame.get("type") == "auth_ok":
                        authed.set()
                    elif frame.get("type") == "notice":
                        results.append("complete")
                    elif frame.get("type") == "heartbeat":
                        results.append("heartbeat")
                        release.wait(10)
                        return
        except Exception as exc:  # noqa: BLE001 - the failure mode under test
            results.append(type(exc).__name__)

    async def test_ping_during_a_slow_transfer_keeps_the_peer(self) -> None:
        results: list[str] = []
        authed, release = threading.Event(), threading.Event()
        reader = asyncio.create_task(
            asyncio.to_thread(self._read_slowly, results, authed, release),
        )
        self.addCleanup(release.set)
        self.assertTrue(await asyncio.to_thread(authed.wait, 10))
        assert self.server._ws_server is not None
        endpoint = next(iter(self.server._ws_server.connections))
        self.server._printer.broadcast({"type": "notice", "text": self.payload})
        await asyncio.sleep(0.3)  # the frame is queued, the reader is on its way
        started = time.monotonic()
        await asyncio.wait_for(self.server._watchdog_ping_clients(), 20)
        tick = time.monotonic() - started
        self.assertLess(tick, ws_mod._WS_PING_TIMEOUT + 1, "the tick waited for the slow ping")
        deadline = time.monotonic() + 20
        while len(results) < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        self.assertEqual(
            results, ["complete", "heartbeat"],
            "a peer reading the transcript was dropped",
        )
        self.assertFalse(endpoint.transport.is_closing())
        self.assertIn(endpoint, self.server._printer._remote_clients)
        release.set()
        await asyncio.wait_for(reader, 10)


if __name__ == "__main__":
    unittest.main()
