# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for the local-WSS endpoint review findings.

Every test drives a real ``RemoteAccessServer`` (or the real
``local_endpoint`` helpers against real files and sockets):

1. The private daemon (``start_private_async``) admits token clients
   only: an empty remote password does not open it to other users.
2. The per-IP password lockout never shuts out a client holding the
   local token (tunnelled visitors share ``127.0.0.1`` with it).
3. A listener bound to ``localhost`` publishes an endpoint URL whose
   address and port belong to the same bound socket.
4. Publication and ownership-checked removal of the endpoint file are
   serialised by one lock.
5. A non-``wss://`` endpoint record is treated as "no daemon", so
   ``connect`` raises ``ConnectionError`` instead of ``ValueError``.
6. ``local_endpoint.send`` gives up on a daemon that stopped reading.
7. Closing the listeners (the blocking shutdown path) removes the
   endpoint file.
8. A cancelled private startup leaves no listener and no endpoint file.
9. A listener on a single LAN address also binds the loopback alias so
   the endpoint names a loopback address local clients can use.
10. The endpoint lock wait is bounded: a stuck holder makes publication
    fail (and a startup roll back) instead of blocking the event loop.
11. The launcher's startup abort stops a private daemon whose startup
    completed before the cancel landed.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import shutil
import socket
import ssl
import tempfile
import threading
import time
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase

from websockets.asyncio.client import connect

import kiss.agents.sorcar.persistence as th
import kiss.core.vscode_config as vc
from kiss.agents.sorcar import local_endpoint
from kiss.agents.third_party_agents import _kiss_web_launcher as launcher
from kiss.core.file_lock import exclusive_file_lock
from kiss.server.web_server import (
    _AUTH_FAIL_MAX,
    RemoteAccessServer,
    _generate_self_signed_cert,
    _get_local_ips,
    _is_loopback_ip,
)
from kiss.tests.local_ws import connect_local, make_test_tls


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


class _ServerBase(IsolatedAsyncioTestCase):
    """Temp KISS state plus a server factory; subclasses start what they need."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-ep-fixes-")
        self.saved = _redirect_persistence(self.tmpdir)
        self._orig_cfg = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"
        self.certfile = Path(self.tmpdir) / "cert.pem"
        self.keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(self.certfile, self.keyfile)
        self.endpoint_file = Path(self.tmpdir) / "sorcar-local.json"
        self.servers: list[RemoteAccessServer] = []

    def _make_server(self, host: str = "127.0.0.1") -> RemoteAccessServer:
        server = RemoteAccessServer(
            host=host,
            port=0,
            certfile=str(self.certfile),
            keyfile=str(self.keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=self.endpoint_file,
        )
        self.servers.append(server)
        return server

    async def asyncTearDown(self) -> None:
        for server in self.servers:
            await server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        vc.CONFIG_DIR, vc.CONFIG_PATH = self._orig_cfg
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _password_attempt(self, password: str) -> dict:
        """One browser-style auth attempt; returns the daemon's first reply."""
        endpoint = local_endpoint.read_endpoint(self.endpoint_file)
        assert endpoint is not None
        async with connect(
            endpoint.url, ssl=local_endpoint.client_ssl_context(endpoint),
            compression=None,
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": password}))
            return dict(json.loads(await asyncio.wait_for(ws.recv(), 10)))


class TestPrivateDaemonIsLocalOnly(_ServerBase):
    async def test_password_auth_is_refused(self) -> None:
        server = self._make_server()
        await server.start_private_async()
        self.assertTrue(server.local_only)
        # The empty remote password admits any loopback peer on the
        # public daemon; the private daemon must refuse it ...
        reply = await self._password_attempt("")
        self.assertEqual(reply.get("type"), "error", reply)
        self.assertEqual(reply.get("code"), "auth_failed", reply)
        # ... while the token still works.
        ws = await connect_local(server)
        await ws.send(json.dumps({"type": "ping"}))
        self.assertEqual(json.loads(await asyncio.wait_for(ws.recv(), 10)).get("type"), "pong")
        await ws.close()


class TestLockoutDoesNotBlockLocalToken(_ServerBase):
    async def test_local_token_passes_a_locked_ip(self) -> None:
        server = self._make_server()
        await server.start_async()
        vc.save_config({"remote_password": "s3cret"})
        for i in range(_AUTH_FAIL_MAX):
            await self._password_attempt(f"wrong-{i}")
        self.assertGreater(server._auth_lock_remaining("127.0.0.1"), 0.0)
        # A browser from the locked IP is still told it is locked ...
        self.assertEqual((await self._password_attempt("s3cret")).get("type"), "auth_locked")
        # ... but the extension / Python client with the token gets in.
        ws = await connect_local(server, open_timeout=5.0)
        await ws.send(json.dumps({"type": "ping"}))
        self.assertEqual(json.loads(await asyncio.wait_for(ws.recv(), 10)).get("type"), "pong")
        await ws.close()


class TestEndpointNamesABoundSocket(_ServerBase):
    async def test_localhost_listener_publishes_reachable_url(self) -> None:
        server = self._make_server(host="localhost")
        await server.start_async()
        endpoint = local_endpoint.read_endpoint(self.endpoint_file)
        assert endpoint is not None
        bound = {
            (sock.family, sock.getsockname()[1]) for sock in server._ws_server.sockets
        }
        self.assertIn((socket.AF_INET, server.port), bound)
        self.assertEqual(endpoint.url, f"wss://127.0.0.1:{server.port}/ws")
        ws = await connect_local(server, open_timeout=5.0)
        await ws.close()

    async def test_lan_only_listener_publishes_loopback_alias(self) -> None:
        lan_ips = sorted(ip for ip in _get_local_ips() if not _is_loopback_ip(ip) and ":" not in ip)
        if not lan_ips:
            self.skipTest("no non-loopback IPv4 address on this host")
        server = self._make_server(host=lan_ips[0])
        await server.start_async()
        self.assertIsNotNone(server._ws_loopback_server, "loopback alias must be bound")
        endpoint = local_endpoint.read_endpoint(self.endpoint_file)
        assert endpoint is not None
        self.assertEqual(endpoint.url, f"wss://127.0.0.1:{server.port}/ws")
        ws = await connect_local(server, open_timeout=5.0)
        await ws.send(json.dumps({"type": "ping"}))
        self.assertEqual(json.loads(await asyncio.wait_for(ws.recv(), 10)).get("type"), "pong")
        await ws.close()

    async def test_stuck_endpoint_lock_fails_startup_and_rolls_back(self) -> None:
        saved = local_endpoint._LOCK_TIMEOUT
        local_endpoint._LOCK_TIMEOUT = 0.3
        release = threading.Event()

        def hold() -> None:
            with exclusive_file_lock(local_endpoint._lock_path(self.endpoint_file)):
                release.wait(10)

        holder = threading.Thread(target=hold)
        holder.start()
        try:
            await asyncio.sleep(0.05)
            server = self._make_server()
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                await server.start_private_async()
            self.assertLess(time.monotonic() - started, 5.0)
            self.assertIsNone(server._ws_server, "a failed publication must close the listener")
            self.assertFalse(self.endpoint_file.exists())
        finally:
            release.set()
            holder.join(5)
            local_endpoint._LOCK_TIMEOUT = saved

    async def test_launcher_abort_stops_a_completed_startup(self) -> None:
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        server = self._make_server()
        startup = asyncio.run_coroutine_threadsafe(server.start_private_async(), loop)
        startup.result(timeout=30)
        self.assertTrue(self.endpoint_file.exists())
        port = server.port
        private_dir = tempfile.mkdtemp(prefix="kiss-ep-private-")
        # The abort runs after startup already completed: cancel() is a
        # no-op, so the listener and endpoint must be torn down anyway.
        await asyncio.to_thread(
            launcher._abort_api_server_startup, loop, thread, server, startup, private_dir,
        )
        self.assertFalse(thread.is_alive(), "the private loop thread must stop")
        self.assertFalse(self.endpoint_file.exists(), "endpoint must be removed")
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
        self.servers.remove(server)

    async def test_close_listeners_removes_endpoint(self) -> None:
        server = self._make_server()
        await server.start_async()
        self.assertTrue(self.endpoint_file.exists())
        server._close_ws_listeners()
        self.assertFalse(self.endpoint_file.exists())

    async def test_cancelled_private_startup_leaves_nothing(self) -> None:
        for delay in (0.0, 0.002, 0.02, 0.1):
            server = self._make_server()
            task = asyncio.ensure_future(server.start_private_async())
            await asyncio.sleep(delay)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            if not task.cancelled():
                # Startup won the race: a complete daemon, cleaned up below.
                self.assertTrue(self.endpoint_file.exists())
                await server.stop_async()
                continue
            self.assertIsNone(server._ws_server, f"listener left after cancel at {delay}s")
            self.assertFalse(self.endpoint_file.exists(), f"endpoint left after cancel at {delay}s")


class TestEndpointFileHelpers(TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-ep-helpers-"))
        self.path = self.tmp / "sorcar-local.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _endpoint(self, url: str = "wss://127.0.0.1:1/ws") -> local_endpoint.LocalEndpoint:
        return local_endpoint.LocalEndpoint(url=url, token="t" * 64, ca=None, pid=1)

    def test_removal_waits_for_the_publication_lock(self) -> None:
        local_endpoint.write_endpoint(self.path, self._endpoint())
        lock = local_endpoint._lock_path(self.path)
        held = threading.Event()

        def hold() -> None:
            with exclusive_file_lock(lock):
                held.set()
                time.sleep(0.4)

        holder = threading.Thread(target=hold)
        holder.start()
        held.wait(5)
        started = time.monotonic()
        local_endpoint.remove_endpoint_if_owned(self.path, "t" * 64)
        elapsed = time.monotonic() - started
        holder.join()
        self.assertGreaterEqual(elapsed, 0.3, "removal must wait for the lock holder")
        self.assertFalse(self.path.exists())

    def test_non_wss_record_is_not_an_endpoint(self) -> None:
        self.path.write_text(json.dumps({
            "url": "ws://127.0.0.1:1/ws", "token": "t" * 64, "ca": None, "pid": 1,
        }))
        self.assertIsNone(local_endpoint.read_endpoint(self.path))
        with self.assertRaises(ConnectionError):
            local_endpoint.connect(self.path, open_timeout=2.0)


def _upgrade_only_server(tmp: Path) -> tuple[threading.Thread, int, Path, threading.Event]:
    """A TLS server that completes the WebSocket upgrade and then never reads.

    Returns the serving thread, the port, the CA file and the stop event.
    """
    certfile, keyfile, ca_file = make_test_tls(tmp)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile, keyfile)
    ctx.num_tickets = 0
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    listener.settimeout(0.2)
    port = listener.getsockname()[1]
    stop = threading.Event()

    def serve() -> None:
        try:
            while not stop.is_set():
                try:
                    raw, _ = listener.accept()
                except TimeoutError:
                    continue
                with ctx.wrap_socket(raw, server_side=True) as tls:
                    request = b""
                    while b"\r\n\r\n" not in request:
                        chunk = tls.recv(4096)
                        if not chunk:
                            break
                        request += chunk
                    key = ""
                    for line in request.decode("latin-1").split("\r\n"):
                        if line.lower().startswith("sec-websocket-key:"):
                            key = line.split(":", 1)[1].strip()
                    accept = base64.b64encode(hashlib.sha1(
                        (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode(),
                    ).digest()).decode()
                    tls.sendall(
                        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                        b"Connection: Upgrade\r\nSec-WebSocket-Accept: "
                        + accept.encode() + b"\r\n\r\n",
                    )
                    # Never read again: the client's writes pile up.
                    stop.wait(30)
        finally:
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return thread, port, ca_file, stop


class TestBoundedSend(TestCase):
    def test_send_times_out_when_the_daemon_stops_reading(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="kiss-ep-send-"))
        thread, port, ca_file, stop = _upgrade_only_server(tmp)
        path = tmp / "sorcar-local.json"
        local_endpoint.write_endpoint(path, local_endpoint.LocalEndpoint(
            url=f"wss://127.0.0.1:{port}/ws", token="t" * 64, ca=str(ca_file), pid=1,
        ))
        try:
            endpoint = local_endpoint.read_endpoint(path)
            assert endpoint is not None
            from websockets.sync.client import connect as sync_connect
            ws = sync_connect(
                endpoint.url, ssl=local_endpoint.client_ssl_context(endpoint),
                open_timeout=5.0, compression=None,
            )
            started = time.monotonic()
            with self.assertRaises(ConnectionError):
                local_endpoint.send(ws, "x" * (64 * 1024 * 1024), timeout=1.0)
            self.assertLess(time.monotonic() - started, 5.0)
            ws.close()
        finally:
            stop.set()
            thread.join(5)
            shutil.rmtree(tmp, ignore_errors=True)
