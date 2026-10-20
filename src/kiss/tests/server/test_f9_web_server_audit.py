# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end checks of the web_server.py audit fixes (F9, audits V1/V2).

Every test drives a real :class:`RemoteAccessServer` over TLS on
loopback with real ``websockets`` clients or raw sockets; nothing is
replaced.  ``tests/server/pause.py`` holds a daemon thread where a race
window must be widened.

* V1-1 — the watchdog's ``heartbeat`` is queued behind the frames
  already in a connection's send FIFO instead of being awaited under
  the pong budget: a peer that just answered the ping must not be
  aborted because a large replay is still draining ahead of the
  heartbeat.
* V1-2 — the receive queue is unbounded: a client whose frames pile up
  behind a slow command keeps answering pings (a paused transport
  held back the pong too, and the watchdog aborted a live client).
* V1-3 — a connection still sniffing its first byte, or in its TLS
  handshake, is closed with the listeners instead of lingering until
  its own timeout.
* V1-6 / V2-4 — the per-frame password check costs a ``stat`` unless
  ``config.json`` changed, and still sees a same-length rotation that
  lands a few milliseconds after the previous frame was served (the
  mtime of two such writes coincides on a jiffy-grained kernel; the
  inode tells them apart).
"""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future as ConcurrentFuture
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection

import kiss.server.web_server as ws
from kiss.core import tls_certs
from kiss.core.user_config import CONFIG_PATH, load_config, save_config
from kiss.server.server import AgentServer
from kiss.server.web_server import _SNIFF_TIMEOUT_SECONDS, RemoteAccessServer
from kiss.tests.server.conftest import find_free_port, no_verify_ssl
from kiss.tests.server.pause import Pause


class _LoopbackServerCase(IsolatedAsyncioTestCase):
    """A TLS daemon on 127.0.0.1 with its own periodic senders stopped."""

    password = ""

    async def asyncSetUp(self) -> None:
        self.port = find_free_port()
        self._orig_config = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        save_config({"remote_password": self.password})
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
            use_tunnel=False,
            local_endpoint_file=Path(tempfile.mkdtemp()) / "sorcar-local.json",
        )
        self.server._printer._send_timeout = 5.0
        await self.server.start_async()
        self.stopped = False
        # The watchdog runs the very ping under test and the version
        # check broadcasts an ``update_available`` the frame-order
        # assertions below would trip over.
        for task in (self.server._version_check_task, self.server._watchdog_task):
            if task is not None:
                await ws._cancel_task(task)
        self._orig_ping_timeout = ws._WS_PING_TIMEOUT

    async def asyncTearDown(self) -> None:
        ws._WS_PING_TIMEOUT = self._orig_ping_timeout
        if not self.stopped:
            await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def _connect(self, **headers: str) -> tuple[Any, ServerConnection]:
        """Open one authenticated browser peer; return (client, server endpoint)."""
        client = await connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=no_verify_ssl(), max_size=None,
            additional_headers=headers,
        )
        self.addCleanup(client.transport.abort)
        await client.send(json.dumps({"type": "auth", "password": self.password}))
        reply = json.loads(await asyncio.wait_for(client.recv(), 5))
        self.assertEqual(reply.get("type"), "auth_ok", reply)
        assert self.server._ws_server is not None
        endpoints = list(self.server._ws_server.connections)
        self.assertEqual(len(endpoints), 1)
        return client, endpoints[0]

    async def _recv(self, client: Any) -> dict[str, Any]:
        return dict(json.loads(await asyncio.wait_for(client.recv(), 10)))


class TestHeartbeatBehindTheFifo(_LoopbackServerCase):
    """V1-1: the heartbeat takes its FIFO place; the watchdog does not wait for it."""

    async def test_heartbeat_behind_a_held_send_slot_keeps_the_peer(self) -> None:
        client, endpoint = await self._connect()
        # A reserved replay slot (what ``replay_snapshot`` queues) holds
        # the connection's send FIFO until its payload is set.
        slot: ConcurrentFuture[str] = ConcurrentFuture()
        self.server._printer._schedule_send(endpoint, slot)
        ws._WS_PING_TIMEOUT = 0.5

        # The ping and pong complete; the tick returns while the slot
        # is still held, so it did not wait for the heartbeat.
        await asyncio.wait_for(self.server._watchdog_ping_clients(), 5)
        self.assertFalse(slot.done())
        # Longer than the pong budget the heartbeat used to be awaited
        # under: the peer must still be there.
        await asyncio.sleep(1.0)
        self.assertFalse(endpoint.transport.is_closing())
        self.assertIn(endpoint, self.server._printer._remote_clients)

        # The replay payload arrives first, the heartbeat behind it.
        slot.set_result(json.dumps({"type": "notice", "text": "replay"}))
        self.assertEqual(await self._recv(client), {"type": "notice", "text": "replay"})
        self.assertEqual(await self._recv(client), {"type": "heartbeat"})


class TestUnboundedReceiveQueue(_LoopbackServerCase):
    """V1-2: frames queued behind a slow command do not hold back the pong."""

    async def test_flood_behind_a_slow_command_keeps_the_pong(self) -> None:
        client, endpoint = await self._connect()
        # Hold this connection's handler inside the backend command it
        # forwarded to the executor; every frame the client sends
        # meanwhile waits in the library's receive queue.
        pause = Pause(
            AgentServer._handle_command.__code__,
            lambda frame_locals: frame_locals["cmd"].get("type") == "getConfig",
        )
        with pause:
            await client.send(json.dumps({"type": "getConfig"}))
            self.assertTrue(await asyncio.to_thread(pause.paused.wait, 10))
            for _ in range(64):
                await client.send(json.dumps({"type": "ping"}))
            await asyncio.sleep(0.5)  # the server has read them all
            ws._WS_PING_TIMEOUT = 1.0
            await asyncio.wait_for(self.server._watchdog_ping_clients(), 10)
            self.assertFalse(
                endpoint.transport.is_closing(),
                "a live client with queued frames was aborted for a missing pong",
            )
        # Released: the heartbeat (queued the moment the pong arrived)
        # and then every reply reach the client.
        types: list[str] = []
        while types.count("pong") < 64:
            types.append((await self._recv(client))["type"])
        self.assertIn("heartbeat", types)


class TestSniffingConnectionsCloseWithTheListeners(_LoopbackServerCase):
    """V1-3: a connection websockets does not know yet is closed on stop."""

    async def test_stop_closes_sniffing_and_handshaking_peers(self) -> None:
        peers: list[socket.socket] = []
        for first_bytes in (b"", b"\x16"):
            raw = socket.create_connection(("127.0.0.1", self.port), 5)
            raw.settimeout(5)
            self.addCleanup(raw.close)
            if first_bytes:
                raw.sendall(first_bytes)  # the server now waits for the ClientHello
            peers.append(raw)
        await asyncio.sleep(0.3)  # both accepted: one sniffing, one in its handshake
        self.assertEqual(len(self.server._sniffing), 2)

        started = time.monotonic()
        await asyncio.wait_for(self.server.stop_async(), 10)
        self.stopped = True
        for raw in peers:
            self.assertEqual(await asyncio.to_thread(raw.recv, 1), b"")
        self.assertLess(time.monotonic() - started, _SNIFF_TIMEOUT_SECONDS / 2)
        self.assertEqual(self.server._sniffing, set())


class TestStoredPasswordCache(_LoopbackServerCase):
    """V1-6 / V2-4: the per-frame check stats config.json and still sees every rotation."""

    password = "Correct-Horse-Battery-1"

    async def _remote(self) -> Any:
        """A password-authenticated client (forwarded, so not a loopback peer)."""
        client, _endpoint = await self._connect(**{"X-Forwarded-For": "203.0.113.9"})
        return client

    async def test_same_length_rotation_right_after_a_frame_is_seen(self) -> None:
        client = await self._remote()
        await client.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(client))["type"], "pong")
        stamp, _password = self.server._remote_password_cache
        self.assertIsNotNone(stamp)
        # Same length, written right behind the frame that was just
        # served: the file's mtime usually does not move.
        save_config({"remote_password": "Correct-Horse-Battery-2"})
        await client.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(client))["code"], "auth_failed")

    async def test_missing_config_refuses_the_frame(self) -> None:
        client = await self._remote()
        await client.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(client))["type"], "pong")
        self.assertIsNotNone(self.server._remote_password_cache[0])
        # A config.json that went away reads as "no password": remote
        # access is off and the cached value must not keep serving.
        CONFIG_PATH.unlink()
        await client.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(client))["code"], "auth_failed")
        self.assertEqual(self.server._remote_password_cache, (None, ""))

    async def test_an_older_refresh_cannot_keep_a_rotated_out_password(self) -> None:
        """Two refreshes interleaved: the cache never pairs an old password with the current stamp.

        An executor-side refresh (``_lan_urls`` calls the same method)
        has read the file and is held before it publishes; the
        password rotates; the loop's own refresh reads the new file
        and is held just before it publishes, where the old refresh is
        let finish first.  Whatever each publishes, the frame's reply
        and the next frame's must refuse the rotated-out password.
        """
        client = await self._remote()
        await client.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(client))["type"], "pong")
        # A same-password save gives the file a new inode, so the old
        # refresh really reads it instead of keeping the cached pair.
        save_config({"remote_password": self.password})
        cache_code = RemoteAccessServer._stored_remote_password.__code__
        load_code = load_config.__code__
        loop_tid = threading.get_ident()
        old_loaded = threading.Event()
        release_old = threading.Event()
        old_done = threading.Event()
        loop_loaded = False

        def old_refresh() -> None:
            try:
                self.server._stored_remote_password()
            finally:
                old_done.set()

        old = threading.Thread(target=old_refresh, name="old-password-refresh")

        def trace(frame: Any, event: str, _arg: Any) -> Any:
            nonlocal loop_loaded
            in_refresh = (
                event == "return" and frame.f_code is load_code
                and frame.f_back is not None and frame.f_back.f_code is cache_code
            )
            if threading.current_thread() is old and in_refresh:
                old_loaded.set()
                release_old.wait(10)
            elif threading.get_ident() == loop_tid:
                if in_refresh:
                    loop_loaded = True
                elif loop_loaded and event == "line" and frame.f_code is cache_code:
                    # The first line after the loop's read: let the old
                    # refresh publish its stale read first.
                    loop_loaded = False
                    release_old.set()
                    assert old_done.wait(10), "the old refresh did not finish"
            return trace

        threading.settrace_all_threads(trace)
        try:
            old.start()
            self.assertTrue(await asyncio.to_thread(old_loaded.wait, 5))
            save_config({"remote_password": "Correct-Horse-Battery-2"})
            await client.send(json.dumps({"type": "ping", "password": self.password}))
            self.assertEqual((await self._recv(client))["code"], "auth_failed")
        finally:
            threading.settrace_all_threads(None)
            release_old.set()
            await asyncio.to_thread(old.join, 5)
        # The refused frame closed the connection; the cache stays
        # coherent for the next client: only the new password is in force.
        self.assertEqual(self.server._stored_remote_password(), "Correct-Horse-Battery-2")
        self.password = "Correct-Horse-Battery-2"
        client = await self._remote()
        await client.send(json.dumps({"type": "ping", "password": "Correct-Horse-Battery-1"}))
        self.assertEqual((await self._recv(client))["code"], "auth_failed")


if __name__ == "__main__":
    unittest.main()


class TestPeerCertificateCheckOffTheLoop(_LoopbackServerCase):
    """V1-4: an enrolled peer's certificate check holds up no other connection.

    ``_peer_client_ca`` reads the enrolled CA files and verifies the
    peer's certificate against them; done on the event loop, every
    machine-link handshake froze every connection's frames for that
    long.  A CA is enrolled in the test home before the daemon starts,
    a peer connects with a certificate that CA issued, and its check is
    held: a browser's ping must still be answered meanwhile.
    """

    async def asyncSetUp(self) -> None:
        peer_tls = Path(tempfile.mkdtemp()) / "tls"
        self.peer_cert, self.peer_key = tls_certs.ensure_local_client_pair(peer_tls)
        trusted = ws._trusted_clients_dir()
        trusted.mkdir(parents=True, exist_ok=True)
        enrolled = trusted / "peer-under-test.pem"
        enrolled.write_bytes((peer_tls / tls_certs.CA_CERT_FILE).read_bytes())
        self.addCleanup(enrolled.unlink)
        await super().asyncSetUp()

    async def test_a_browser_is_answered_while_a_peer_s_certificate_is_checked(self) -> None:
        browser, _endpoint = await self._connect()
        peer_ssl = ssl.create_default_context()
        peer_ssl.check_hostname, peer_ssl.verify_mode = False, ssl.CERT_NONE
        peer_ssl.load_cert_chain(str(self.peer_cert), str(self.peer_key))
        with Pause(tls_certs.enrolled_client_ca.__code__, lambda local: True) as held:
            peer = await connect(f"wss://127.0.0.1:{self.port}/ws", ssl=peer_ssl, max_size=None)
            self.addCleanup(peer.transport.abort)
            await peer.send(json.dumps({"type": "auth"}))
            self.assertTrue(
                await asyncio.to_thread(held.paused.wait, 10), "the peer's certificate check",
            )
            started = time.monotonic()
            await browser.send(json.dumps({"type": "ping"}))
            self.assertEqual((await self._recv(browser))["type"], "pong")
            self.assertLess(time.monotonic() - started, 2.0, "the browser waited for the check")
        reply = json.loads(await asyncio.wait_for(peer.recv(), 10))
        self.assertEqual(reply, {"type": "auth_ok", "local": True})
