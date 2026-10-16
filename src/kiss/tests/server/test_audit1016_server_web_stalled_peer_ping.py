# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The watchdog ping of a peer that stopped reading must be bounded.

Concurrency audit 2026-10-16 (server-web): ``websockets``'
``Connection.ping()`` writes the ping frame and then waits for the
transport to drain before it returns the pong waiter.  A peer whose
socket buffers are full (a laptop asleep behind a half-open TCP
connection, a browser tab that paused reading) therefore held
``RemoteAccessServer._ping_one_ws`` in ``ping()`` forever: only the pong
wait and the heartbeat send were under ``asyncio.wait_for``.  Because
``_watchdog_ping_clients`` gathers every ping and the watchdog runs its
checks in sequence, one such peer froze the whole watchdog — tunnel
restarts, the URL-file rewrite, the IP-change restart and the pings of
every other client — for as long as the peer stayed stalled (the
library's own keep-alive is off, so nothing else ever dropped it).

Real ``RemoteAccessServer`` over TLS on loopback, a real ``websockets``
client whose transport pauses reading, no mocks.  The stall tolerance is
shortened through the printer's ``_send_timeout`` attribute so the test
does not wait 30 s.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import tempfile
import time
import unittest
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection

from kiss.core.user_config import CONFIG_PATH, save_config
from kiss.server.web_server import _WS_PING_TIMEOUT, RemoteAccessServer
from kiss.tests.server.conftest import find_free_port, no_verify_ssl

_PAYLOAD = base64.b64encode(os.urandom(192 * 1024)).decode("ascii")
"""256 KiB of incompressible text: permessage-deflate cannot shrink it."""


async def _flood(conn: ServerConnection, done: asyncio.Event) -> None:
    """Write *_PAYLOAD* to *conn* with the library's own ``send`` until it fails.

    ``send`` has no deadline: once the peer's buffers are full it blocks
    in the transport drain, which is exactly the state a ping must not
    inherit.  *done* is set when the connection is gone.
    """
    try:
        while True:
            await conn.send(_PAYLOAD)
    except Exception:
        pass
    finally:
        done.set()


class TestStalledPeerPing(IsolatedAsyncioTestCase):
    """A ping of a peer that stopped reading ends within the send timeout."""

    async def asyncSetUp(self) -> None:
        self.port = find_free_port()
        self._orig_config = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        save_config({"remote_password": ""})
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
            use_tunnel=False,
        )
        self.server._printer._send_timeout = 1.0
        await self.server.start_async()
        # The ping must end on its own account: stop the daemon's own
        # periodic senders (the version check's ``update_available``
        # broadcast, whose bounded send would abort the stalled peer and
        # so release a hung ping; the watchdog, which runs the very ping
        # under test).
        for task in (self.server._version_check_task, self.server._watchdog_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def _connect(self) -> tuple[object, ServerConnection]:
        """Open one authenticated browser peer; return (client, server endpoint)."""
        client = await connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=no_verify_ssl(), max_size=None,
        )
        # Abort instead of a close handshake: a peer that paused reading
        # would otherwise make ``close()`` wait out its close timeout.
        self.addCleanup(client.transport.abort)
        await client.send(json.dumps({"type": "auth", "password": ""}))
        reply = json.loads(await asyncio.wait_for(client.recv(), 5))
        self.assertEqual(reply.get("type"), "auth_ok", reply)
        assert self.server._ws_server is not None
        endpoints = [c for c in self.server._ws_server.connections]
        self.assertEqual(len(endpoints), 1)
        return client, endpoints[0]

    async def test_ping_of_stalled_peer_is_bounded_and_drops_it(self) -> None:
        client, endpoint = await self._connect()
        client.transport.pause_reading()  # type: ignore[attr-defined]

        gone = asyncio.Event()
        flood = asyncio.create_task(_flood(endpoint, gone))
        self.addCleanup(flood.cancel)
        # Wait until the server side is held in the drain for good: the
        # loopback socket buffers grow for a while before they are
        # full, so the transport must keep unsent bytes continuously,
        # not just for one sample.
        deadline = time.monotonic() + 15
        stalled_since = None
        while time.monotonic() < deadline:
            if endpoint.transport.get_write_buffer_size() == 0:
                stalled_since = None
            elif stalled_since is None:
                stalled_since = time.monotonic()
            elif time.monotonic() - stalled_since >= 1.0:
                break
            await asyncio.sleep(0.05)
        self.assertIsNotNone(stalled_since, "the peer never blocked the server's writes")

        # Bounded by the longest legitimate wait a bounded ping can make
        # (the pong timeout, should the ping frame itself squeeze
        # through); the unbounded ping (and ``close()``) hang for good.
        started = time.monotonic()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                self.server._watchdog_ping_clients(), _WS_PING_TIMEOUT + 5,
            )
        elapsed = time.monotonic() - started
        self.assertLess(
            elapsed, _WS_PING_TIMEOUT + 3,
            "the watchdog ping of a peer that stopped reading must end, "
            "not hang the watchdog",
        )
        await asyncio.wait_for(gone.wait(), 5)
        self.assertTrue(
            endpoint.transport.is_closing(), "stalled peer's transport must close",
        )
        self.assertNotIn(endpoint, self.server._printer._remote_clients)

    async def test_ping_of_reading_peer_delivers_heartbeat(self) -> None:
        client, endpoint = await self._connect()
        await asyncio.wait_for(self.server._watchdog_ping_clients(), 5)
        frame = json.loads(await asyncio.wait_for(client.recv(), 5))  # type: ignore[attr-defined]
        self.assertEqual(frame, {"type": "heartbeat"})
        self.assertFalse(endpoint.transport.is_closing())
        self.assertIn(endpoint, self.server._printer._remote_clients)


if __name__ == "__main__":
    unittest.main()
