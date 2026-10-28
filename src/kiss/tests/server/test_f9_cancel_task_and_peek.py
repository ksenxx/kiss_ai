# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""F9: ``_cancel_task`` propagates its caller's own cancellation; the proactor peek owns its socket.

* ``_cancel_task(task)`` cancels *task* and awaits it; the
  ``CancelledError`` that await raises is normally the cancelled
  task's and is swallowed.  When the *awaiting* task is cancelled
  meanwhile (``stop_async`` cancelling a shutdown that is itself
  cancelling a watchdog), the same ``except`` used to swallow that
  outer cancellation too, and the caller ran on past the await
  instead of unwinding.
* On a loop without reader callbacks (Windows' proactor loop) the
  first-byte sniff runs in a worker thread on a blocking duplicate of
  the socket that the thread closes itself, so a cancelled caller
  never leaves the thread selecting on a descriptor that was closed
  and reused.  The thread half is exercised here over a real socket
  pair: a byte arrives, the peer closes first, nothing arrives.
"""

from __future__ import annotations

import asyncio
import socket
from unittest import IsolatedAsyncioTestCase, TestCase

from kiss.server import web_server


class TestCancelTaskPropagatesOwnCancellation(IsolatedAsyncioTestCase):
    """The awaiting task unwinds when it is cancelled during ``_cancel_task``."""

    async def test_outer_cancellation_is_not_swallowed(self) -> None:
        """An outer task cancelled while awaiting ``_cancel_task`` stops there."""
        inner_unwinding = asyncio.Event()
        ran_past = False

        async def _slow_to_unwind() -> None:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                inner_unwinding.set()
                await asyncio.sleep(0.3)
                raise

        async def _outer() -> None:
            nonlocal ran_past
            await web_server._cancel_task(inner)
            ran_past = True

        inner = asyncio.create_task(_slow_to_unwind())
        await asyncio.sleep(0)
        outer = asyncio.create_task(_outer())
        await inner_unwinding.wait()
        outer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await outer
        self.assertTrue(outer.cancelled(), "the outer task swallowed its own cancellation")
        self.assertFalse(ran_past, "the outer task ran on past _cancel_task")

    async def test_inner_cancellation_alone_is_swallowed(self) -> None:
        """Without an outer cancellation the helper returns normally."""
        inner = asyncio.create_task(asyncio.sleep(30))
        await asyncio.sleep(0)
        await web_server._cancel_task(inner)
        self.assertTrue(inner.cancelled())
        await web_server._cancel_task(None)


class TestPeekFirstByteBlocking(TestCase):
    """The thread half of the sniff peeks without consuming and closes its duplicate."""

    def _pair(self) -> tuple[socket.socket, socket.socket]:
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        return a, b

    def test_peeks_the_first_byte_and_leaves_it(self) -> None:
        server_side, peer = self._pair()
        peer.sendall(b"\x16abc")
        self.assertEqual(web_server._peek_first_byte_blocking(server_side), b"\x16")
        self.assertEqual(server_side.recv(4), b"\x16abc")

    def test_peer_closing_first_yields_empty(self) -> None:
        server_side, peer = self._pair()
        peer.close()
        self.assertEqual(web_server._peek_first_byte_blocking(server_side), b"")

    def test_nothing_arriving_times_out_to_empty(self) -> None:
        server_side, _peer = self._pair()
        saved = web_server._SNIFF_TIMEOUT_SECONDS
        web_server._SNIFF_TIMEOUT_SECONDS = 0.2
        self.addCleanup(setattr, web_server, "_SNIFF_TIMEOUT_SECONDS", saved)
        self.assertEqual(web_server._peek_first_byte_blocking(server_side), b"")
        # The original socket is untouched: still open and non-timing-out.
        self.assertIsNone(server_side.gettimeout())
