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

Real ``websockets`` server and clients on loopback, real event loop, no
mocks.  The timeout is shortened through the printer's ``_send_timeout``
attribute so the test does not wait 30 s.
"""

from __future__ import annotations

import asyncio
import time
import unittest

from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import ServerConnection, serve

from kiss.server.web_server import WebPrinter

_PAYLOAD = "x" * 65536
"""64 KiB per message; a few hundred of them overflow both socket buffers."""

_MESSAGE_COUNT = 400


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
        self.printer.add_local_client(ws)
        await self.registered.put(ws)
        try:
            await ws.wait_closed()
        finally:
            self.printer.remove_local_client(ws)

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
