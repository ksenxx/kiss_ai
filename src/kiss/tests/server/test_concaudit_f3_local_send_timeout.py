# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A local peer that stops reading must be dropped, not queued for forever.

Concurrency audit (F3, reviewer R3 finding 8): the printer's per-peer
send awaited the transport drain with no timeout while holding the
endpoint's FIFO send lock.  Local peers run without a ping watchdog, so
a peer that never read its socket kept the lock forever and every later
broadcast added one more pending future (and payload) to
``_pending_sends`` without bound.  ``WebPrinter._timed_send`` bounds the
send by ``_send_timeout``; on expiry the peer is removed, its pending
sends are cancelled and its transport aborted.

A peer that keeps reading, but slower than the broadcasts come, never
runs that timeout out: what bounds its queue is a budget
(``_max_send_backlog``), past which it is dropped the same way.

Real ``websockets`` server and clients on loopback, real event loop, no
mocks.  The timeout is shortened through the printer's ``_send_timeout``
attribute so the test does not wait 30 s.
"""

from __future__ import annotations

import asyncio
import threading
import time
import unittest
from concurrent.futures import Future as ConcurrentFuture
from functools import partial
from typing import Any

from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import ServerConnection, serve

from kiss.server.web_server import WebPrinter
from kiss.tests.server.pause import Pause

_PAYLOAD = "x" * 65536
"""64 KiB per message; a few hundred of them overflow both socket buffers."""

_MESSAGE_COUNT = 400


async def _take(client: ClientConnection, every: float, sizes: list[int]) -> None:
    """Take *client*'s messages, one each *every* seconds, noting their *sizes*."""
    async for message in client:
        sizes.append(len(message))
        await asyncio.sleep(every)


def _registers_outside(
    lock: Any, sender: threading.Thread, frame_locals: dict[str, Any],
) -> bool:
    """Whether *sender* registers a send's completion without holding *lock*."""
    return threading.get_ident() == sender.ident and not lock._is_owned()


class TestLocalSendTimeout(unittest.IsolatedAsyncioTestCase):
    """The bounded send drops a stuck peer and keeps a healthy one."""

    async def asyncSetUp(self) -> None:
        self.printer = WebPrinter()
        self.printer._loop = asyncio.get_running_loop()
        self.printer._send_timeout = 0.5
        self.registered: asyncio.Queue[ServerConnection] = asyncio.Queue()
        # No permessage-deflate: the repetitive payload would compress
        # to a few bytes and never fill the socket buffers.
        self.server = await serve(
            self._handler, "127.0.0.1", 0,
            ping_interval=None, max_size=None, compression=None,
        )
        self.port = next(iter(self.server.sockets)).getsockname()[1]

    async def asyncTearDown(self) -> None:
        self.server.close()
        await self.server.wait_closed()

    async def _handler(self, ws: ServerConnection) -> None:
        """Register *ws* with the printer as a local peer until it closes."""
        self.printer.add_client(ws, local=True)
        await self.registered.put(ws)
        try:
            await ws.wait_closed()
        finally:
            self.printer.remove_client(ws)

    async def _connect(self) -> tuple[ClientConnection, ServerConnection]:
        """Open one local peer; return (client side, server-side endpoint)."""
        client = await connect(
            f"ws://127.0.0.1:{self.port}/",
            ping_interval=None, max_size=None, compression=None,
        )
        # Abort instead of a close handshake: a peer that paused reading
        # would otherwise make ``close()`` wait out its close timeout.
        self.addCleanup(client.transport.abort)
        endpoint = await asyncio.wait_for(self.registered.get(), 5)
        return client, endpoint

    async def test_non_reading_peer_is_dropped_after_timeout(self) -> None:
        client, endpoint = await self._connect()
        # Stop reading at the transport level so the loopback buffers fill.
        client.transport.pause_reading()

        for _ in range(_MESSAGE_COUNT):
            self.printer._send_to_local_clients(_PAYLOAD)
        await asyncio.sleep(0.1)
        queued = len(self.printer._pending_sends.get(endpoint, ()))
        self.assertGreater(queued, 1, "peer never blocked the send queue")

        deadline = time.monotonic() + 5
        while endpoint in self.printer._local_clients and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        self.assertNotIn(
            endpoint, self.printer._local_clients,
            "a peer that stopped reading must be dropped after the send timeout",
        )
        self.assertNotIn(endpoint, self.printer._pending_sends)
        self.assertTrue(
            endpoint.transport.is_closing(), "stuck peer's transport must close",
        )
        # Later broadcasts no longer queue anything for the dropped peer.
        self.printer._send_to_local_clients(_PAYLOAD)
        await asyncio.sleep(0.05)
        self.assertNotIn(endpoint, self.printer._pending_sends)

    async def test_slow_reader_is_dropped_once_its_queue_is_over_budget(self) -> None:
        slow, slow_endpoint = await self._connect()
        fast, fast_endpoint = await self._connect()
        self.printer._send_timeout = 30  # the budget, not the timeout, drops it
        self.printer._max_send_backlog = 20 * len(_PAYLOAD)
        got: list[int] = []
        readers = [
            asyncio.create_task(_take(slow, 0.02, [])),
            asyncio.create_task(_take(fast, 0, got)),
        ]
        for reader in readers:
            self.addCleanup(reader.cancel)
        for _ in range(_MESSAGE_COUNT):
            self.printer._send_to_local_clients(_PAYLOAD)
            await asyncio.sleep(0.001)
        self.assertNotIn(slow_endpoint, self.printer._local_clients)
        self.assertNotIn(slow_endpoint, self.printer._pending_sends)
        await asyncio.wait_for(slow_endpoint.wait_closed(), 5)
        # The peer that keeps up got every payload, twenty budgets' worth:
        # a payload counts against the budget only until it is sent.
        deadline = time.monotonic() + 10
        while len(got) < _MESSAGE_COUNT and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        self.assertEqual(got, [len(_PAYLOAD)] * _MESSAGE_COUNT)
        self.assertIn(fast_endpoint, self.printer._local_clients)

    async def test_send_that_finished_is_not_counted_against_the_next(self) -> None:
        client, endpoint = await self._connect()
        self.printer._max_send_backlog = 0  # a peer with anything still queued is dropped
        first = threading.Thread(
            target=self.printer._send_to_local_clients, args=("one",), daemon=True,
        )
        # The sender is held where it registers its send's completion, if
        # it does that outside the printer's lock: the send finishes
        # meanwhile and must not look queued to the next one.
        with Pause(
            ConcurrentFuture.add_done_callback.__code__,
            partial(_registers_outside, self.printer._ws_lock, first),
        ):
            first.start()
            self.assertEqual(await asyncio.wait_for(client.recv(), 5), "one")
            await asyncio.sleep(0.05)
            self.printer._send_to_local_clients("two")
            await asyncio.sleep(0.05)
            self.assertIn(endpoint, self.printer._local_clients)
        await asyncio.to_thread(first.join, 5)
        self.assertEqual(await asyncio.wait_for(client.recv(), 5), "two")

    async def test_replay_the_peer_asked_for_does_not_count(self) -> None:
        # A replay goes to a slow link in parts, over minutes, and the page
        # asks again for one it lost with its connection: counted, a big
        # one would get its reader dropped again and again.
        client, endpoint = await self._connect()
        self.printer._max_send_backlog = 8 * 1024
        slot: ConcurrentFuture[str] = ConcurrentFuture()
        self.printer._schedule_send(endpoint, slot)  # what ``replay_snapshot`` reserves
        slot.set_result(_PAYLOAD)
        self.printer._send_to_local_clients("after")
        self.assertEqual(await asyncio.wait_for(client.recv(), 5), _PAYLOAD)
        self.assertEqual(await asyncio.wait_for(client.recv(), 5), "after")
        self.assertIn(endpoint, self.printer._local_clients)

    async def test_reading_peer_keeps_receiving(self) -> None:
        client, endpoint = await self._connect()

        self.printer._send_to_local_clients("hello")
        message = await asyncio.wait_for(client.recv(), 5)
        self.assertEqual(message, "hello")
        await asyncio.sleep(0.6)
        self.assertIn(endpoint, self.printer._local_clients)
        self.assertFalse(endpoint.transport.is_closing())


if __name__ == "__main__":
    unittest.main()
