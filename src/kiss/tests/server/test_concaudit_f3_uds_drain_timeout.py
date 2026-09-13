# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A UDS peer that stops reading must be dropped, not queued for forever.

Concurrency audit (F3, reviewer R3 finding 8): ``WebPrinter._uds_send``
awaited ``writer.drain()`` with no timeout while holding the endpoint's
FIFO send lock.  UDS peers have no ping watchdog, so a peer that never
read its socket kept the lock forever and every later broadcast added
one more pending future (and payload) to ``_pending_sends`` without
bound.  The drain is now bounded; on expiry the peer is removed, its
pending sends are cancelled and its transport closed.

Real ``socketpair`` peers, real event loop, no mocks.  The timeout is
shortened through the printer's ``_uds_drain_timeout`` attribute so
the test does not wait 30 s.
"""

from __future__ import annotations

import asyncio
import socket
import time
import unittest

from kiss.server.web_server import WebPrinter

_PAYLOAD = "x" * 65536
"""64 KiB per message; ~40 of them overflow both socket buffers."""


class TestUdsDrainTimeout(unittest.IsolatedAsyncioTestCase):
    """The bounded drain drops a stuck peer and keeps a healthy one."""

    async def _connect(self, printer: WebPrinter) -> tuple[socket.socket, asyncio.StreamWriter]:
        """Register one socketpair peer with *printer*; return (peer, writer)."""
        server_sock, peer_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(peer_sock.close)
        _reader, writer = await asyncio.open_connection(sock=server_sock)
        self.addCleanup(writer.close)
        printer.add_uds_writer(writer)
        return peer_sock, writer

    async def test_non_reading_peer_is_dropped_after_timeout(self) -> None:
        printer = WebPrinter()
        printer._loop = asyncio.get_running_loop()
        printer._uds_drain_timeout = 0.5
        _peer, writer = await self._connect(printer)

        for _ in range(40):
            printer._send_to_uds_writers(_PAYLOAD)
        await asyncio.sleep(0.1)
        queued = len(printer._pending_sends.get(writer, ()))
        self.assertGreater(queued, 1, "peer never blocked the send queue")

        deadline = time.monotonic() + 5
        while writer in printer._uds_writers and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        self.assertNotIn(
            writer, printer._uds_writers,
            "a peer that stopped reading must be dropped after the drain timeout",
        )
        self.assertNotIn(writer, printer._pending_sends)
        self.assertTrue(writer.is_closing(), "stuck peer's transport must close")
        # Later broadcasts no longer queue anything for the dropped peer.
        printer._send_to_uds_writers(_PAYLOAD)
        await asyncio.sleep(0.05)
        self.assertNotIn(writer, printer._pending_sends)

    async def test_reading_peer_keeps_receiving(self) -> None:
        printer = WebPrinter()
        printer._loop = asyncio.get_running_loop()
        printer._uds_drain_timeout = 0.5
        peer, writer = await self._connect(printer)
        peer.setblocking(False)
        peer_reader, _peer_writer = await asyncio.open_connection(sock=peer)

        printer._send_to_uds_writers("hello")
        line = await asyncio.wait_for(peer_reader.readline(), 5)
        self.assertEqual(line, b"hello\n")
        self.assertIn(writer, printer._uds_writers)
        self.assertFalse(writer.is_closing())


if __name__ == "__main__":
    unittest.main()
