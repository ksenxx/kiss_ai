# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A sidebar panel's reply overtakes the transcripts queued ahead of it.

The CHATS rows a page opens on are a ``history`` reply of a few
kilobytes, yet a page over a slow link waited for them behind what the
``ready`` had queued first (replayed transcripts of several megabytes),
because a connection's frames went out strictly in queue order
(``WebPrinter._locked_send`` under its ``FifoSendLock``).  Now a reply
of one of :data:`~kiss.server.web_server.URGENT_REPLY_TYPES` takes the
lock ahead of the queue, and a big frame under way lets it through
between two of its parts; the parts of two big frames still never
interleave (the joiners drop a frame's gathered parts at another
frame's part): an urgent reply bigger than a part takes the ordinary
lane, in parts (A21-2, audit 2026-11-07: sent whole and ahead, a page
of fifty long rows gave a slow browser no complete frame for the
minutes it took, and its stale check dropped the connection), and a
frame that is not urgent still waits its turn.

Real ``websockets`` server and a throttled synchronous client on
loopback (the harness of ``test_send_parts_slow_reader``), no mocks.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import socket
import ssl
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import cast

from websockets.sync.client import connect as sync_connect

from kiss.agents.sorcar.local_endpoint import PART_CHARS, FrameJoiner, frame_parts
from kiss.server.web_server import (
    _PART_ROOM_BYTES,
    URGENT_REPLY_TYPES,
    FifoSendLock,
    RemoteAccessServer,
    _unsent_bytes,
)
from kiss.tests.conftest import posix_only
from kiss.tests.local_ws import _generate_self_signed_cert
from kiss.tests.server.conftest import no_verify_ssl
from kiss.tests.server.test_send_parts_slow_reader import (
    _FRAME_CHARS,
    _READ_RATE,
    _SEND_TIMEOUT,
    _incompressible_frame,
    _SlowReaderCase,
    _small_buffers,
    _ThrottledSocket,
)


def _receive_wire(
    port: int, frames: int, received: list[str], wire: list[tuple[float, str, object]],
    done: threading.Event, release: threading.Event,
) -> None:
    """A throttled synchronous client: the first *frames* joined frames go in *received*.

    Every wire message's landing time, kind (``part`` or the frame's
    type) and ``seq`` (a part's number, else ``None``) goes in *wire*.
    Sets *done* once the frames are in and holds the connection until
    *release*.
    """
    sock = _small_buffers(_ThrottledSocket(socket.AF_INET, socket.SOCK_STREAM))
    sock.connect(("127.0.0.1", port))
    joiner = FrameJoiner()
    with sync_connect(
        f"ws://127.0.0.1:{port}/", sock=sock, compression="deflate", max_size=None,
    ) as client:
        try:
            while len(received) < frames:
                message = client.recv(timeout=60)
                assert isinstance(message, str)
                decoded = json.loads(message)
                wire.append((time.monotonic(), str(decoded.get("type")), decoded.get("seq")))
                whole = joiner.feed(decoded, message)
                if whole is not None:
                    received.append(whole[1])
        finally:
            done.set()
        release.wait(60)


class TestUrgentReplyLane(_SlowReaderCase):
    """A ``history`` reply reaches a slow reader while a transcript is still in transit."""

    compression = "deflate"

    async def _run_reader(
        self, frames: int, received: list[str], wire: list[tuple[float, str, object]],
    ):
        """Start the throttled reader; return its (done, release, task)."""
        done, release = threading.Event(), threading.Event()
        task = asyncio.ensure_future(asyncio.to_thread(
            _receive_wire, self.port, frames, received, wire, done, release,
        ))
        return done, release, task

    async def test_history_reply_lands_between_the_parts_of_a_transcript(self) -> None:
        big_one = _incompressible_frame(_FRAME_CHARS)
        big_two = _incompressible_frame(_FRAME_CHARS // 4)
        after = json.dumps({"type": "after"})
        received: list[str] = []
        wire: list[tuple[float, str, object]] = []
        done, release, reader = await self._run_reader(4, received, wire)
        try:
            endpoint = await asyncio.wait_for(self.registered.get(), 5)
            self.printer.bind_conn("page", endpoint)
            started = time.monotonic()
            self.printer._send_to_local_clients(big_one)
            self.printer._send_to_local_clients(big_two)
            # The first part is on its way: now the page's rows are ready.
            while len(wire) < 2:
                await asyncio.sleep(0.01)
            self.printer.broadcast({"type": "history", "connId": "page", "sessions": []})
            self.printer._send_to_local_clients(after)
            history_sent = time.monotonic()
            await asyncio.wait_for(asyncio.to_thread(done.wait, 60), 60)
            ended = time.monotonic()
            kinds = [json.loads(m)["type"] for m in received]
            self.assertEqual(kinds, ["history", "task_events", "task_events", "after"])
            self.assertEqual(received[1:], [big_one, big_two, after])
            self.assertNotIn("connId", json.loads(received[0]))
            # On the wire: parts, the reply, the rest of the parts (the
            # first transcript's and then the second's, never mixed),
            # the plain frame last.
            wire_kinds = [kind for _at, kind, _seq in wire]
            history_at = wire_kinds.index("history")
            self.assertGreater(history_at, 0)
            self.assertLess(
                history_at, len(frame_parts(big_one)),
                "the reply must not wait for the whole first transcript",
            )
            self.assertEqual(
                wire_kinds,
                ["part"] * history_at + ["history"]
                + ["part"] * (len(frame_parts(big_one)) + len(frame_parts(big_two)) - history_at)
                + ["after"],
            )
            self.assertLess(
                wire[history_at][0] - history_sent, (ended - started) / 2,
                "the reply must land in a fraction of the transfer",
            )
            self.assertIn(endpoint, self.printer._local_clients)
        finally:
            release.set()
            await asyncio.wait_for(reader, 60)

    async def test_a_plain_small_frame_still_waits_its_turn(self) -> None:
        """Only the urgent types overtake: another small reply queues behind the transcript."""
        self.assertNotIn("models", URGENT_REPLY_TYPES)
        big = _incompressible_frame(_FRAME_CHARS // 2)
        received: list[str] = []
        wire: list[tuple[float, str, object]] = []
        done, release, reader = await self._run_reader(2, received, wire)
        try:
            endpoint = await asyncio.wait_for(self.registered.get(), 5)
            self.printer.bind_conn("page", endpoint)
            self.printer._send_to_local_clients(big)
            while len(wire) < 2:
                await asyncio.sleep(0.01)
            self.printer.broadcast({"type": "models", "connId": "page", "models": []})
            await asyncio.wait_for(asyncio.to_thread(done.wait, 60), 60)
            self.assertEqual(received[0], big)
            self.assertEqual(json.loads(received[1])["type"], "models")
            self.assertEqual(
                [kind for _at, kind, _seq in wire],
                ["part"] * len(frame_parts(big)) + ["models"],
            )
        finally:
            release.set()
            await asyncio.wait_for(reader, 60)

    async def test_an_urgent_reply_bigger_than_a_part_takes_the_ordinary_lane(self) -> None:
        """A ``history`` page bigger than a part (50 long rows run past 64 KB) goes
        out in parts behind the transcript, every part a complete frame the slow
        reader sees land (whole and ahead, it was one frame of minutes of silence
        to the browser's stale check); a small ``machines`` reply right behind it
        still overtakes both, and no two frames' parts interleave."""
        big = _incompressible_frame(_FRAME_CHARS // 2)
        rows = base64.b64encode(os.urandom(3 * PART_CHARS * 3 // 4)).decode()
        received: list[str] = []
        wire: list[tuple[float, str, object]] = []
        done, release, reader = await self._run_reader(3, received, wire)
        try:
            endpoint = await asyncio.wait_for(self.registered.get(), 5)
            self.printer.bind_conn("page", endpoint)
            self.printer._send_to_local_clients(big)
            while len(wire) < 2:
                await asyncio.sleep(0.01)
            self.printer.broadcast(
                {"type": "history", "connId": "page", "sessions": [{"title": rows}]},
            )
            self.printer.broadcast({"type": "machines", "connId": "page", "machines": []})
            await asyncio.wait_for(asyncio.to_thread(done.wait, 60), 60)
            decoded = [json.loads(m) for m in received]
            self.assertEqual([d["type"] for d in decoded], ["machines", "task_events", "history"])
            self.assertEqual(received[1], big)
            self.assertEqual(decoded[2]["sessions"], [{"title": rows}])
            # On the wire: the transcript's first parts, the small reply
            # as one frame, the transcript's remaining parts, then the
            # page's own parts, each a complete frame.
            page_parts = len(frame_parts(received[2]))
            self.assertGreater(page_parts, 1, "a page past one part")
            kinds = [kind for _at, kind, _seq in wire]
            machines_at = kinds.index("machines")
            self.assertGreater(machines_at, 0)
            self.assertEqual(
                [seq for _at, _kind, seq in wire],
                list(range(machines_at)) + [None]
                + list(range(machines_at, len(frame_parts(big)))) + list(range(page_parts)),
            )
        finally:
            release.set()
            await asyncio.wait_for(reader, 60)

    async def test_a_panel_reply_sent_direct_overtakes_the_transcript(self) -> None:
        """A ``machines`` reply written by ``RemoteAccessServer._reply_direct`` (the
        MACHINES panel's path, not ``broadcast``) lands between the parts too."""
        home = tempfile.mkdtemp(prefix="kiss-reply-direct-")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        server = RemoteAccessServer(
            local_endpoint_file=Path(home) / "sorcar-local.json", work_dir=home,
        )
        # The reader registers with the server's own printer.
        self.printer = server._printer
        self.printer._loop = asyncio.get_running_loop()
        self.printer._send_timeout = _SEND_TIMEOUT
        big = _incompressible_frame(_FRAME_CHARS // 2)
        received: list[str] = []
        wire: list[tuple[float, str, object]] = []
        done, release, reader = await self._run_reader(2, received, wire)
        try:
            endpoint = await asyncio.wait_for(self.registered.get(), 5)
            self.printer._send_to_local_clients(big)
            while len(wire) < 2:
                await asyncio.sleep(0.01)
            await server._reply_direct(
                endpoint, {"type": "machines", "machines": [], "checkedAt": 0}, "getMachines",
            )
            await asyncio.wait_for(asyncio.to_thread(done.wait, 60), 60)
            self.assertEqual(
                [json.loads(m)["type"] for m in received], ["machines", "task_events"],
            )
            self.assertEqual(received[1], big)
            kinds = [kind for _at, kind, _seq in wire]
            machines_at = kinds.index("machines")
            self.assertGreater(machines_at, 0)
            self.assertLess(machines_at, len(frame_parts(big)))
        finally:
            release.set()
            await asyncio.wait_for(reader, 60)


class TestSenderKeepsPaceWithThePeer(_SlowReaderCase):
    """The parts of a big frame never run far ahead of what a slow remote peer has taken.

    With the kernel's own buffer sizing (no :func:`_small_buffers`): it
    grows the send queue for a peer that reads slowly, which is what
    put hundreds of KB between an urgent reply and the wire.  A
    loopback peer (a ``daemon_client`` run) is not held back.
    """

    compression = "deflate"
    small_buffers = False
    local = False

    async def _unsent_while_sending(self, big: str) -> tuple[list[int], float]:
        """Send *big* to a throttled reader; return the sender's unsent bytes seen and the time."""
        received: list[str] = []
        wire: list[tuple[float, str, object]] = []
        done, release = threading.Event(), threading.Event()
        reader = asyncio.ensure_future(asyncio.to_thread(
            _receive_wire, self.port, 1, received, wire, done, release,
        ))
        try:
            endpoint = await asyncio.wait_for(self.registered.get(), 5)
            started = time.monotonic()
            self.printer._send_to_ws_clients(big)
            unsent: list[int] = []
            while not done.is_set():
                unsent.append(_unsent_bytes(endpoint.transport))
                await asyncio.sleep(0.005)
            ended = time.monotonic()
            self.assertEqual(received, [big])
            self.assertEqual(len(wire), len(frame_parts(big)))
            return unsent, ended - started
        finally:
            release.set()
            await asyncio.wait_for(reader, 60)

    async def test_a_later_part_waits_for_the_peer_to_take_the_earlier_ones(self) -> None:
        big = _incompressible_frame(_FRAME_CHARS)
        unsent, took = await self._unsent_while_sending(big)
        # Between two looks the peer has at most the part under way on
        # top of the room left for it; a sender that wrote the parts
        # as fast as the transport took them ran several hundred KB
        # ahead here.
        self.assertLessEqual(
            max(unsent), _PART_ROOM_BYTES + 2 * PART_CHARS,
            f"the sender ran {max(unsent)} bytes ahead of the peer",
        )
        # Pacing costs the transfer next to nothing: the peer's reading
        # rate sets its time.
        self.assertLess(took, 3 * len(big) / _READ_RATE + 1)

    # Running ahead shows in the kernel's send queue, which
    # ``_unsent_bytes`` reads on Linux and macOS only: the loop's write
    # buffer, all it has on Windows, holds about one part whether the
    # sender is paced or not (``websockets`` drains each part before
    # the next is written), so there the sibling's ceiling is never
    # exceeded and a sender that was held back would look the same.
    @posix_only("_unsent_bytes counts the kernel's send queue on Linux and macOS only")
    async def test_a_loopback_peer_is_not_held_back(self) -> None:
        """A ``daemon_client`` peer gets the parts as fast as the transport takes them."""
        self.local = True
        big = _incompressible_frame(_FRAME_CHARS)
        unsent, _took = await self._unsent_while_sending(big)
        self.assertGreater(
            max(unsent), _PART_ROOM_BYTES + 2 * PART_CHARS,
            "the sender was paced for a loopback peer",
        )


class TestUnsentBytesUnderTls(unittest.IsolatedAsyncioTestCase):
    """Under TLS the encrypted bytes the raw transport still holds are unsent too."""

    async def _accept(self, _reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.accepted.put_nowait(writer)

    @posix_only("_unsent_bytes counts the kernel's send queue on Linux and macOS only")
    async def test_the_count_holds_the_backlog_and_shrinks_as_the_peer_reads(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        _generate_self_signed_cert(tmp / "cert.pem", tmp / "key.pem")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(tmp / "cert.pem", tmp / "key.pem")
        self.accepted: asyncio.Queue[asyncio.StreamWriter] = asyncio.Queue()
        server = await asyncio.start_server(self._accept, "127.0.0.1", 0, ssl=context)
        self.addAsyncCleanup(server.wait_closed)
        self.addCleanup(server.close)
        port = server.sockets[0].getsockname()[1]
        reader, client = await asyncio.open_connection("127.0.0.1", port, ssl=no_verify_ssl())
        self.addCleanup(client.transport.abort)
        # The writer stays referenced: one that is collected closes its transport.
        writer = await asyncio.wait_for(self.accepted.get(), 10)
        transport = writer.transport
        self.addCleanup(transport.abort)
        # A peer that reads nothing, and a small kernel send buffer: all
        # but a sliver of what is written waits in the raw transport.
        # (A stream writer's transport reads too; its annotation is write-only.)
        receiving = cast(asyncio.Transport, client.transport)
        receiving.pause_reading()
        transport.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192)
        size = 4 * 1024 * 1024
        self.assertFalse(transport.is_closing())
        transport.write(b"x" * size)
        backlog = _unsent_bytes(transport)
        self.assertGreater(backlog, size // 2, "the raw transport's share is not counted")
        receiving.resume_reading()
        await asyncio.wait_for(reader.readexactly(size // 4), 30)
        self.assertLess(
            _unsent_bytes(transport), backlog - size // 8, "the peer's reading did not show",
        )


class TestFifoSendLockAhead(unittest.IsolatedAsyncioTestCase):
    """The lock's ``ahead`` lane: served first, in order, counted; FIFO for the rest."""

    async def test_ahead_waiters_go_first_and_are_counted(self) -> None:
        lock = FifoSendLock()
        await lock.acquire()
        order: list[str] = []

        async def take(name: str, ahead: bool) -> None:
            await lock.acquire(ahead)
            order.append(name)
            lock.release()

        tasks = [
            asyncio.ensure_future(take("first in line", False)),
            asyncio.ensure_future(take("second in line", False)),
        ]
        await asyncio.sleep(0)
        self.assertEqual(lock.ahead_waiting, 0)
        tasks.append(asyncio.ensure_future(take("urgent", True)))
        await asyncio.sleep(0)
        self.assertEqual(lock.ahead_waiting, 1)
        tasks.append(asyncio.ensure_future(take("later urgent", True)))
        await asyncio.sleep(0)
        self.assertEqual(lock.ahead_waiting, 2)
        lock.release()
        await asyncio.gather(*tasks)
        self.assertEqual(order, ["urgent", "later urgent", "first in line", "second in line"])
        self.assertEqual(lock.ahead_waiting, 0)

    async def test_let_ahead_serves_the_ahead_lane_then_the_owner_again(self) -> None:
        """The owner's ``let_ahead`` hands the lock to every ahead waiter, those that
        ask meanwhile too, and gets it back before the waiters in turn."""
        lock = FifoSendLock()
        await lock.acquire()
        order: list[str] = []
        second_urgent_asked = asyncio.Event()

        async def take(name: str, ahead: bool) -> None:
            await lock.acquire(ahead)
            order.append(name)
            if name == "urgent":
                # The lock is this waiter's: another urgent one asks now.
                asyncio.ensure_future(take("urgent meanwhile", True))
                await asyncio.sleep(0)
                second_urgent_asked.set()
            lock.release()

        in_line = asyncio.ensure_future(take("in line", False))
        urgent = asyncio.ensure_future(take("urgent", True))
        await asyncio.sleep(0)
        self.assertEqual(lock.ahead_waiting, 1)
        await lock.let_ahead()
        order.append("owner again")
        self.assertTrue(second_urgent_asked.is_set())
        self.assertEqual(lock.ahead_waiting, 0)
        lock.release()
        await asyncio.gather(in_line, urgent)
        await asyncio.sleep(0)
        self.assertEqual(order, ["urgent", "urgent meanwhile", "owner again", "in line"])

    async def test_a_cancelled_ahead_waiter_is_uncounted(self) -> None:
        lock = FifoSendLock()
        await lock.acquire()
        waiter = asyncio.ensure_future(lock.acquire(ahead=True))
        await asyncio.sleep(0)
        self.assertEqual(lock.ahead_waiting, 1)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertEqual(lock.ahead_waiting, 0)
        lock.release()
        # The cancelled waiter was skipped: the lock is free again.
        await asyncio.wait_for(lock.acquire(), 1)
        lock.release()


if __name__ == "__main__":
    unittest.main()
