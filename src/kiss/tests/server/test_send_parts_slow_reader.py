# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A big frame reaches a slow reader in parts, whole; a reader that stops is still dropped.

``WebPrinter._timed_send`` used to bound every send by ``_send_timeout``
(30 s) as a whole, so a replayed transcript of a few MB took longer
than that on a slow network and the daemon dropped the browser
mid-frame, which then reconnected, asked for the transcript again and
was dropped again.  A frame above ``local_endpoint.PART_CHARS`` now
goes out as separate ``part`` frames (``local_endpoint.frame_parts``),
and the receiver joins them (``local_endpoint.FrameJoiner``;
``transport.js`` in the browser), and the deadline restarts whenever
the peer's socket took some of what is being sent
(``WebPrinter._send_while_draining``): a slow reader is never dropped
while it keeps reading, however slowly, it sees the parts land one by
one (its own stale check takes a long silence for a dead socket), a
peer that reads nothing is dropped after the same wait as before, and
a small frame still travels as it is.

Real ``websockets`` server and clients on loopback, real event loop,
no mocks.  The timeout is shortened through the printer's
``_send_timeout`` attribute.  The slow reader is a synchronous client
whose socket hands over at most :data:`_READ_BYTES` every
:data:`_READ_INTERVAL` seconds (with small socket buffers, so the
loopback cannot hold the frame for it): a steady 1.3 MB/s, so a
2.6 MB frame takes about four send timeouts to arrive.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import threading
import time
import unittest

from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.sync.client import connect as sync_connect

from kiss.agents.sorcar.local_endpoint import PART_CHARS, FrameJoiner, frame_parts
from kiss.server.web_server import WebPrinter

_SEND_TIMEOUT = 0.5
_BUFFER_BYTES = 64 * 1024
_READ_BYTES = 16 * 1024
_READ_INTERVAL = 0.0125
_READ_RATE = _READ_BYTES / _READ_INTERVAL
_READ_BURST = 16 * _READ_BYTES
_FRAME_CHARS = 40 * PART_CHARS


def _incompressible_frame(chars: int) -> str:
    """A JSON frame of about *chars* characters that deflate cannot shrink (random base64)."""
    body = base64.b64encode(os.urandom(chars * 3 // 4)).decode()
    return json.dumps({"type": "task_events", "events": [{"images": [{"b64": body}]}]})


def _small_buffers(sock: socket.socket) -> socket.socket:
    """Shrink *sock*'s send and receive buffers so the loopback cannot hold a frame.

    Set before the connection exists (a listening socket passes them on
    to the sockets it accepts), so the TCP window is sized by them.
    macOS sizes a connection's buffers itself (about 600 KB of a frame
    fit in the loopback there), so the transfer is slower than that
    buffer only because the frame is many times bigger.
    """
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _BUFFER_BYTES)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _BUFFER_BYTES)
    return sock


class _ThrottledSocket(socket.socket):
    """A TCP socket that reads :data:`_READ_BYTES` per :data:`_READ_INTERVAL` on average.

    A token bucket paced by the clock: bytes come due at
    :data:`_READ_RATE`, a read takes what is due (at most what the
    caller asked for, up to :data:`_READ_BURST` after an idle spell)
    and pauses first while less than :data:`_READ_BYTES` is due.
    Paced by elapsed time rather than by one pause per read because
    ``time.sleep`` can oversleep by far (macOS coalesces the timers of
    an idle machine to 100 ms), which would otherwise make the reader
    many times slower than designed.
    """

    def __init__(self, family: int = -1, kind: int = -1, proto: int = -1) -> None:
        super().__init__(family, kind, proto)
        self._due_from = time.monotonic()

    def recv(self, bufsize: int, flags: int = 0) -> bytes:
        """Read the bytes due by now, after a pause when fewer than :data:`_READ_BYTES` are."""
        due = self._due_now()
        if due < _READ_BYTES:
            time.sleep((_READ_BYTES - due) / _READ_RATE)
            due = self._due_now()
        data = super().recv(min(bufsize, max(int(due), _READ_BYTES)), flags)
        self._due_from += len(data) / _READ_RATE
        return data

    def _due_now(self) -> float:
        """Return the bytes due by now, at most :data:`_READ_BURST` after an idle spell."""
        self._due_from = max(self._due_from, time.monotonic() - _READ_BURST / _READ_RATE)
        return (time.monotonic() - self._due_from) * _READ_RATE


def _receive_slowly(
    port: int, compression: str | None, frames: int, received: list[str],
    arrivals: list[float], done: threading.Event, release: threading.Event,
) -> None:
    """Connect a throttled synchronous client; put its first *frames* joined frames in *received*.

    The time each wire message landed goes in *arrivals*.  Sets *done*
    once the frames are in and keeps the connection open until
    *release* is set, so the test can look at the peer before it closes.
    """
    sock = _small_buffers(_ThrottledSocket(socket.AF_INET, socket.SOCK_STREAM))
    sock.connect(("127.0.0.1", port))
    joiner = FrameJoiner()
    with sync_connect(
        f"ws://127.0.0.1:{port}/", sock=sock, compression=compression, max_size=None,
    ) as client:
        try:
            while len(received) < frames:
                message = client.recv(timeout=60)
                arrivals.append(time.monotonic())
                assert isinstance(message, str)
                whole = joiner.feed(json.loads(message), message)
                if whole is None:
                    assert len(message) <= PART_CHARS + 100, "a part must fit a part"
                else:
                    received.append(whole[1])
        finally:
            done.set()
        release.wait(60)


class _SlowReaderCase(unittest.IsolatedAsyncioTestCase):
    """A printer serving real loopback peers, with or without deflate."""

    compression: str | None = None
    # Whether the listener's buffers are shrunk (:func:`_small_buffers`);
    # a case that measures how far the sender runs ahead of a slow peer
    # keeps the kernel's own sizing, which grows for such a peer.
    small_buffers = True

    async def asyncSetUp(self) -> None:
        self.printer = WebPrinter()
        self.printer._loop = asyncio.get_running_loop()
        self.printer._send_timeout = _SEND_TIMEOUT
        self.registered: asyncio.Queue[ServerConnection] = asyncio.Queue()
        listener = socket.create_server(("127.0.0.1", 0))
        if self.small_buffers:
            _small_buffers(listener)
        self.server = await serve(
            self._handler, sock=listener,
            ping_interval=None, max_size=None, compression=self.compression,
        )
        self.port = listener.getsockname()[1]

    async def asyncTearDown(self) -> None:
        self.server.close()
        await self.server.wait_closed()

    # Whether the peers register as joining parts (every real client does).
    parts = True
    # Whether the peers register as loopback (``daemon_client``) peers,
    # which the printer does not pace, or as remote browsers.
    local = True

    async def _handler(self, ws: ServerConnection) -> None:
        """Register *ws* with the printer as a peer until it closes."""
        self.printer.add_client(ws, local=self.local, parts=self.parts)
        await self.registered.put(ws)
        try:
            await ws.wait_closed()
        finally:
            self.printer.remove_client(ws)

    async def _connect_paused(self) -> tuple[ClientConnection, ServerConnection]:
        """Open one asyncio peer; return (client side, server-side endpoint)."""
        client = await connect(
            f"ws://127.0.0.1:{self.port}/",
            ping_interval=None, max_size=None, compression=self.compression,
        )
        # Abort instead of a close handshake: a peer that paused reading
        # would otherwise make ``close()`` wait out its close timeout.
        self.addCleanup(client.transport.abort)
        endpoint = await asyncio.wait_for(self.registered.get(), 5)
        return client, endpoint

    async def _slow_reader_gets_big_frame(self) -> None:
        """The shared check: a 2.6 MB frame, then a small one, reach a throttled reader.

        In parts to a peer that joins them, whole to any other; the
        transfer outlasts the send timeout several times over either
        way, and the peer is kept because it keeps reading.
        """
        frame = _incompressible_frame(_FRAME_CHARS)
        after = json.dumps({"type": "after"})
        received: list[str] = []
        arrivals: list[float] = []
        done, release = threading.Event(), threading.Event()
        reader = asyncio.ensure_future(asyncio.to_thread(
            _receive_slowly, self.port, self.compression, 2, received, arrivals, done, release,
        ))
        try:
            endpoint = await asyncio.wait_for(self.registered.get(), 5)
            started = time.monotonic()
            self.printer._send_to_local_clients(frame)
            self.printer._send_to_local_clients(after)
            await asyncio.wait_for(asyncio.to_thread(done.wait, 60), 60)
            elapsed = time.monotonic() - started
            self.assertEqual([len(m) for m in received], [len(frame), len(after)])
            self.assertEqual(received, [frame, after])
            self.assertEqual(len(arrivals), (len(frame_parts(frame)) if self.parts else 1) + 1)
            self.assertGreater(
                elapsed, 2 * _SEND_TIMEOUT,
                "the transfer must outlast the send timeout to prove the point",
            )
            if self.parts:
                # The parts land one by one while the frame is still in
                # transit: the reader had most of it before the end came.
                self.assertLess(arrivals[0] - started, elapsed / 4)
                self.assertLess(arrivals[len(arrivals) // 2] - started, 3 * elapsed / 4)
            self.assertIn(endpoint, self.printer._local_clients, "a slow reader was dropped")
            self.assertFalse(endpoint.transport.is_closing())
        finally:
            release.set()
            await asyncio.wait_for(reader, 60)


class TestSlowReaderKeepsBigFrame(_SlowReaderCase):
    """Without deflate (test peers)."""

    async def test_slow_reader_receives_the_whole_frame(self) -> None:
        await self._slow_reader_gets_big_frame()

    async def test_stopped_reader_is_dropped_mid_frame(self) -> None:
        client, endpoint = await self._connect_paused()
        client.transport.pause_reading()
        self.printer._send_to_local_clients(_incompressible_frame(_FRAME_CHARS))
        deadline = time.monotonic() + 10
        while endpoint in self.printer._local_clients and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        self.assertNotIn(
            endpoint, self.printer._local_clients,
            "a peer that stopped reading must be dropped after the send timeout",
        )
        self.assertNotIn(endpoint, self.printer._pending_sends)
        self.assertTrue(endpoint.transport.is_closing(), "stuck peer's transport must close")

    async def test_small_frame_is_one_write(self) -> None:
        client, endpoint = await self._connect_paused()
        frame = _incompressible_frame(PART_CHARS // 2)
        self.printer._send_to_local_clients(frame)
        self.assertEqual(await asyncio.wait_for(client.recv(), 5), frame)
        self.assertIn(endpoint, self.printer._local_clients)

    async def test_parts_are_numbered_and_join_back(self) -> None:
        """The wire form of a big frame: numbered parts, the last without ``more``."""
        client, _endpoint = await self._connect_paused()
        frame = _incompressible_frame(2 * PART_CHARS + 10)
        self.printer._send_to_local_clients(frame)
        parts = [json.loads(await asyncio.wait_for(client.recv(), 5)) for _ in range(3)]
        self.assertEqual([p["seq"] for p in parts], [0, 1, 2])
        self.assertEqual([p.get("more") for p in parts], [True, True, None])
        self.assertEqual("".join(p["data"] for p in parts), frame)
        self.assertEqual([len(p["data"]) for p in parts[:2]], [PART_CHARS, PART_CHARS])
        joiner = FrameJoiner()
        self.assertEqual([joiner.feed(p) for p in parts[:2]], [None, None])
        self.assertEqual(joiner.feed(parts[2]), (json.loads(frame), frame))
        # A sequence whose start was missed is dropped until the next
        # frame starts; an ordinary frame passes through untouched.
        self.assertIsNone(joiner.feed(parts[1]))
        self.assertIsNone(joiner.feed(parts[2]))
        plain = '{"type": "x"}'
        self.assertEqual(joiner.feed({"type": "x"}, plain), ({"type": "x"}, plain))
        self.assertEqual([joiner.feed(p) for p in parts], [None, None, (json.loads(frame), frame)])


class TestSlowReaderKeepsBigFrameDeflate(_SlowReaderCase):
    """With permessage-deflate (browsers, the daemon links)."""

    compression = "deflate"

    async def test_slow_reader_receives_the_whole_frame(self) -> None:
        await self._slow_reader_gets_big_frame()


class TestPeerWithoutParts(_SlowReaderCase):
    """A peer that did not say it joins parts (an older client) gets big frames whole."""

    parts = False

    async def test_big_frame_arrives_whole(self) -> None:
        client, endpoint = await self._connect_paused()
        frame = _incompressible_frame(3 * PART_CHARS)
        self.printer._send_to_local_clients(frame)
        self.assertEqual(await asyncio.wait_for(client.recv(), 10), frame)
        self.assertIn(endpoint, self.printer._local_clients)
        self.assertNotIn(endpoint, self.printer._parts_clients)

    async def test_slow_reader_receives_the_whole_frame(self) -> None:
        """One 2.6 MB write outlasting the send timeout four times over: kept, as it drains."""
        await self._slow_reader_gets_big_frame()


if __name__ == "__main__":
    unittest.main()
