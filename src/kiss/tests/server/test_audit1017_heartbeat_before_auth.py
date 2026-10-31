# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A watchdog ``heartbeat`` queued ahead of ``auth_ok`` is not a refusal.

The daemon's watchdog used to push its ``{"type": "heartbeat"}`` frame
at every socket whose WebSocket handshake had completed, authenticated
or not.  A tick landing between that handshake and the daemon's
``auth_ok`` queued a heartbeat ahead of it, and both Python clients
(``MachineLink._connect`` and ``local_endpoint.connect``) read exactly
one frame after their ``auth``: the link reported
``refused the connection: {'type': 'heartbeat'}`` and backed the
machine off for 30 s, a ``daemon_client`` run failed with
``rejected the local token``.

Two fixes, each with its own test against real sockets:

* the daemon heartbeats only sockets the printer has registered,
  which happens after ``auth_ok`` is written (a real in-process daemon
  with a 50 ms watchdog, a raw socket that waits several ticks before
  sending ``auth``);
* both clients skip heartbeats while waiting for the ``auth`` answer,
  so an older daemon that still heartbeats early is handled too (a
  stand-in daemon that answers ``auth`` with two heartbeats first),
  and the auth deadline still bounds a peer that sends nothing else.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import ssl
import tempfile
import time
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase

from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

import kiss.agents.sorcar.local_endpoint as local_endpoint
import kiss.agents.sorcar.machines as mach
from kiss.tests.local_ws import fake_daemon
from kiss.tests.server.test_machine_link_idle_heartbeats import (
    WATCHDOG_INTERVAL,
    TestMachineLinkIdleHeartbeats,
)


class TestDaemonHeartbeatsOnlyAuthenticatedSockets(TestCase):
    """The real daemon (50 ms watchdog) sends no heartbeat before ``auth_ok``."""

    def setUp(self) -> None:
        self.daemon = TestMachineLinkIdleHeartbeats("test_idle_link_survives_many_heartbeats")
        self.daemon.setUp()

    def tearDown(self) -> None:
        self.daemon.tearDown()

    def test_first_frame_after_a_late_auth_is_auth_ok(self) -> None:
        """Several watchdog ticks pass before ``auth``; ``auth_ok`` still comes first."""
        machine = self.daemon.machine
        ctx = ssl.create_default_context(cafile=str(machine.cert_file()))
        ctx.check_hostname = False
        with ws_connect(mach.ws_url(machine.url), ssl=ctx, max_queue=None) as ws:
            time.sleep(WATCHDOG_INTERVAL * 8)
            ws.send(json.dumps({"type": "auth", "link": True, "password": ""}))
            first = json.loads(ws.recv(timeout=5))
            self.assertEqual(first.get("type"), "auth_ok", first)
            # Once authenticated the socket does get the watchdog's
            # heartbeats (the browser transport's proof of life).
            second = json.loads(ws.recv(timeout=5))
            self.assertEqual(second, {"type": "heartbeat"})


async def _legacy_daemon(ws: ServerConnection) -> None:
    """Answer ``auth`` like a pre-fix daemon whose watchdog ticked twice first."""
    first = json.loads(await ws.recv())
    assert first.get("type") == "auth", first
    await ws.send(json.dumps({"type": "heartbeat"}))
    await ws.send(json.dumps({"type": "heartbeat"}))
    await ws.send(json.dumps({"type": "auth_ok", "local": True}))
    async for message in ws:
        cmd = json.loads(message)
        if cmd.get("type") == "ping":
            await ws.send(json.dumps({"type": "pong", "generation": cmd.get("generation")}))


async def _heartbeats_only(ws: ServerConnection) -> None:
    """Never answer ``auth``: heartbeat until the client gives up."""
    await ws.recv()
    try:
        while True:
            await ws.send(json.dumps({"type": "heartbeat"}))
            await asyncio.sleep(0.02)
    except ConnectionClosed:
        return


async def _heartbeat_flood(ws: ServerConnection) -> None:
    """Never answer ``auth``: heartbeat as fast as the socket takes them."""
    await ws.recv()
    try:
        while True:
            await ws.send(json.dumps({"type": "heartbeat"}))
            await asyncio.sleep(0)
    except ConnectionClosed:
        return


def _auth_reply_under_flood(endpoint_file: Path, timeout: float) -> float:
    """Send ``auth`` to the flooding stand-in; return how long ``recv_auth_reply`` waited."""
    endpoint = local_endpoint.read_endpoint(endpoint_file)
    assert endpoint is not None
    ws = ws_connect(
        endpoint.url, ssl=local_endpoint.client_ssl_context(endpoint),
        open_timeout=5.0, max_queue=None,
    )
    try:
        ws.send(json.dumps({"type": "auth", "token": "t"}))
        started = time.monotonic()
        try:
            local_endpoint.recv_auth_reply(ws, timeout=timeout)
        except TimeoutError:
            return time.monotonic() - started
        raise AssertionError("recv_auth_reply returned although only heartbeats arrived")
    finally:
        ws.close()


def _connect_and_ping(endpoint_file: Path) -> object:
    """Connect as a local client and return the answer to one ``ping``."""
    ws = local_endpoint.connect(endpoint_file, open_timeout=5.0)
    try:
        ws.send(json.dumps({"type": "ping", "generation": 7}))
        return json.loads(ws.recv(timeout=5))
    finally:
        ws.close()


def _machine_for(endpoint_file: Path) -> tuple[mach.Machine, Path]:
    """The stand-in daemon as a registered machine, with its pinned certificate."""
    endpoint = local_endpoint.read_endpoint(endpoint_file)
    assert endpoint is not None and endpoint.ca is not None
    return (
        mach.Machine("legacy", url="https://" + endpoint.url[len("wss://"):-len("/ws")]),
        Path(endpoint.ca),
    )


class TestClientsSkipHeartbeatsBeforeAuthOk(IsolatedAsyncioTestCase):
    """Both Python clients read past heartbeats to the ``auth`` answer."""

    def setUp(self) -> None:
        self.scratch = Path(tempfile.mkdtemp(prefix="kiss-hb-auth-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

    async def test_local_endpoint_connect_reaches_auth_ok(self) -> None:
        """``local_endpoint.connect`` returns a working socket despite the heartbeats."""
        async with fake_daemon(self.scratch, _legacy_daemon, answer_auth=False) as endpoint_file:
            # The sync client runs off the loop thread the stand-in serves on.
            reply = await asyncio.to_thread(_connect_and_ping, endpoint_file)
        self.assertEqual(reply, {"type": "pong", "generation": 7})

    async def test_machine_link_request_reaches_auth_ok(self) -> None:
        """``MachineLink.request`` answers over a link whose ``auth_ok`` came late."""
        async with fake_daemon(self.scratch, _legacy_daemon, answer_auth=False) as endpoint_file:
            machine, ca_file = _machine_for(endpoint_file)
            link = mach.MachineLink(machine, cert_file=ca_file)
            try:
                reply = await asyncio.to_thread(link.request, {"type": "ping"}, "pong", 5.0)
                self.assertEqual(reply["type"], "pong")
                self.assertEqual(link._health.retry_at, 0.0)
            finally:
                link.close()

    async def test_the_auth_deadline_still_bounds_a_peer_that_only_heartbeats(self) -> None:
        """Heartbeats alone do not keep either client waiting past its budget."""
        async with fake_daemon(self.scratch, _heartbeats_only, answer_auth=False) as endpoint_file:
            started = time.monotonic()
            with self.assertRaisesRegex(ConnectionError, "did not complete the local auth"):
                await asyncio.to_thread(local_endpoint.connect, endpoint_file, open_timeout=0.5)
            self.assertLess(time.monotonic() - started, 5.0)
            machine, ca_file = _machine_for(endpoint_file)
            link = mach.MachineLink(machine, cert_file=ca_file)
            started = time.monotonic()
            with self.assertRaisesRegex(mach.MachineError, "did not complete the handshake"):
                await asyncio.to_thread(link.request, {"type": "ping"}, "pong", 0.5)
            self.assertLess(time.monotonic() - started, 5.0)
            self.assertGreater(link._health.retry_at, 0.0)

    async def test_a_heartbeat_flood_cannot_extend_the_auth_deadline(self) -> None:
        """Heartbeats arriving faster than the per-frame floor still end at the deadline.

        ``recv_auth_reply`` used to give every heartbeat a fresh 1 ms
        read budget, so a peer sending them back to back kept the
        client reading past its deadline.
        """
        async with fake_daemon(self.scratch, _heartbeat_flood, answer_auth=False) as endpoint_file:
            took = await asyncio.to_thread(_auth_reply_under_flood, endpoint_file, 0.5)
        # The last ``recv`` wait may end a millisecond early on Windows
        # (its wait primitives round the timeout to the timer tick).
        self.assertGreaterEqual(took, 0.45, f"recv_auth_reply gave up after only {took:.3f}s")
        self.assertLess(took, 3.0, f"recv_auth_reply gave up only after {took:.2f}s")
