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
5. An endpoint record whose URL is not a WebSocket one (``ws://`` or
   ``wss://``) is treated as "no daemon", so ``connect`` raises
   ``ConnectionError`` instead of ``ValueError``.
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
import queue
import shutil
import socket
import ssl
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State
from websockets.sync.client import ClientConnection as SyncClientConnection
from websockets.sync.client import connect as sync_connect

import kiss.agents.sorcar.persistence as th
import kiss.core.user_config as vc
from kiss.agents.sorcar import local_endpoint
from kiss.agents.third_party_agents import _kiss_web_launcher as launcher
from kiss.core.file_lock import exclusive_file_lock
from kiss.server.sorcar import is_loopback_ip
from kiss.server.web_server import (
    _AUTH_FAIL_MAX,
    RemoteAccessServer,
    _bound_loopback,
    _get_local_ips,
)
from kiss.tests.local_ws import _generate_self_signed_cert, connect_local, make_test_tls
from kiss.tests.server.conftest import (
    FORWARDED_VISITOR,
    redirect_persistence,
    restore_persistence,
)
from kiss.tests.server.pause import Pause


async def _ping(ws: ClientConnection) -> None:
    """Send ``ping`` and wait for its ``pong``, skipping the daemon's own broadcasts.

    The startup PyPI check broadcasts ``update_available`` to every
    client the moment it completes, which on a loaded machine is just
    after this connection opened.
    """
    await ws.send(json.dumps({"type": "ping"}))
    deadline = asyncio.get_running_loop().time() + 10
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("no pong within 10 s")
        frame = json.loads(await asyncio.wait_for(ws.recv(), remaining))
        if frame.get("type") == "pong":
            return


class _ServerBase(IsolatedAsyncioTestCase):
    """Temp KISS state plus a server factory; subclasses start what they need."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-ep-fixes-")
        self.saved = redirect_persistence(self.tmpdir)
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
        th._close_thread_db()
        restore_persistence(self.saved)
        vc.CONFIG_DIR, vc.CONFIG_PATH = self._orig_cfg
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _password_attempt(self, password: str) -> dict:
        """One remote-browser-style auth attempt; returns the daemon's first reply.

        The client carries a forwarded client IP, as a cloudflared visitor
        does: a direct loopback browser would be admitted without a password.
        """
        endpoint = local_endpoint.read_endpoint(self.endpoint_file)
        assert endpoint is not None
        async with connect(
            endpoint.url, ssl=local_endpoint.client_ssl_context(endpoint),
            compression=None, additional_headers=FORWARDED_VISITOR,
        ) as ws:
            try:
                await ws.send(json.dumps({"type": "auth", "password": password}))
            except ConnectionClosed:
                # Refused before any frame was read (no password is set):
                # the refusal is already queued for the recv below.
                pass
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
        await _ping(ws)
        await ws.close()


class TestLockoutDoesNotBlockLocalToken(_ServerBase):
    async def test_local_token_passes_a_locked_ip(self) -> None:
        server = self._make_server()
        await server.start_async()
        vc.save_config({"remote_password": "s3cret"})
        for i in range(_AUTH_FAIL_MAX):
            await self._password_attempt(f"wrong-{i}")
        self.assertGreater(server._auth_lock_remaining(FORWARDED_VISITOR["X-Forwarded-For"]), 0.0)
        # A browser from the locked IP is still told it is locked ...
        self.assertEqual((await self._password_attempt("s3cret")).get("type"), "auth_locked")
        # ... but the extension / Python client with the token gets in.
        ws = await connect_local(server, open_timeout=5.0)
        await _ping(ws)
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
        self.assertEqual(endpoint.url, f"ws://127.0.0.1:{server.port}/ws")
        self.assertIsNone(local_endpoint.client_ssl_context(endpoint))
        ws = await connect_local(server, open_timeout=5.0)
        await ws.close()

    async def test_lan_only_listener_publishes_loopback_alias(self) -> None:
        lan_ips = sorted(ip for ip in _get_local_ips() if not is_loopback_ip(ip) and ":" not in ip)
        if not lan_ips:
            self.skipTest("no non-loopback IPv4 address on this host")
        server = self._make_server(host=lan_ips[0])
        await server.start_async()
        self.assertIsNotNone(server._ws_loopback_server, "loopback alias must be bound")
        endpoint = local_endpoint.read_endpoint(self.endpoint_file)
        assert endpoint is not None
        self.assertEqual(endpoint.url, f"ws://127.0.0.1:{server.port}/ws")
        ws = await connect_local(server, open_timeout=5.0)
        await _ping(ws)
        await ws.close()

    async def test_os_picked_port_moves_off_a_loopback_port_held_elsewhere(self) -> None:
        """The OS picks a port free on the LAN address only; when another
        socket holds ``127.0.0.1`` of it (seen under a loaded Linux run),
        an ephemeral-port server moves to a port it can share, while a
        caller-chosen port keeps it and waits for the watchdog."""
        lan_ips = sorted(ip for ip in _get_local_ips() if not is_loopback_ip(ip) and ":" not in ip)
        if not lan_ips:
            self.skipTest("no non-loopback IPv4 address on this host")
        server = self._make_server(host=lan_ips[0])
        await server.start_async()
        holder = socket.socket()
        self.addCleanup(holder.close)
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        held = holder.getsockname()[1]
        # Put the server where a port-0 bind could land: LAN:held free,
        # 127.0.0.1:held taken.
        for listener in (server._ws_loopback_server, server._ws_server):
            listener.close()
            await listener.wait_closed()
        server._ws_loopback_server = None
        server.port = held
        server._ws_server = await server._serve_wss(lan_ips[0])
        await server._bind_loopback_alias_or_repick_port(ephemeral=False)
        self.assertIsNone(server._ws_loopback_server)
        self.assertEqual(server.port, held)

        await server._bind_loopback_alias_or_repick_port(ephemeral=True)
        self.assertIsNotNone(server._ws_loopback_server, "loopback alias must be bound")
        self.assertNotEqual(server.port, held)
        self.assertEqual(_bound_loopback(server._ws_loopback_server)[1], server.port)
        self.assertEqual(server._ws_server.sockets[0].getsockname()[1], server.port)

    async def test_no_loopback_listener_publishes_no_endpoint(self) -> None:
        """A LAN listener whose ``127.0.0.1`` port another process holds
        publishes no local endpoint: a same-machine client dialling the
        LAN address is a non-loopback peer, refused as plaintext and
        never honoured with the token, so the ``ws://LAN:port/ws`` the
        fallback used to publish could only fail.  The watchdog's
        reclaim publishes the loopback endpoint, which then works."""
        lan_ips = sorted(ip for ip in _get_local_ips() if not is_loopback_ip(ip) and ":" not in ip)
        if not lan_ips:
            self.skipTest("no non-loopback IPv4 address on this host")
        server = self._make_server(host=lan_ips[0])
        await server.start_async()
        holder = socket.socket()
        self.addCleanup(holder.close)
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        held = holder.getsockname()[1]
        for listener in (server._ws_loopback_server, server._ws_server):
            listener.close()
            await listener.wait_closed()
        server._ws_loopback_server = None
        server.port = held
        server._ws_server = await server._serve_wss(lan_ips[0])
        await server._bind_loopback_alias_or_repick_port(ephemeral=False)
        self.assertIsNone(server._ws_loopback_server)
        # What ``start_async`` published names the closed port: a fresh
        # daemon in this situation has published nothing yet.
        self.endpoint_file.unlink()

        server._write_local_endpoint_file()
        self.assertIsNone(
            local_endpoint.read_endpoint(self.endpoint_file),
            "regression: an endpoint no same-machine client can use was published",
        )

        holder.close()
        await server._watchdog_reclaim_loopback()
        endpoint = local_endpoint.read_endpoint(self.endpoint_file)
        assert endpoint is not None
        self.assertEqual(endpoint.url, f"ws://127.0.0.1:{held}/ws")
        ws = await connect_local(server, open_timeout=5.0)
        await _ping(ws)
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

    def test_non_websocket_record_is_not_an_endpoint(self) -> None:
        self.path.write_text(json.dumps({
            "url": "http://127.0.0.1:1/ws", "token": "t" * 64, "ca": None, "pid": 1,
        }))
        self.assertIsNone(local_endpoint.read_endpoint(self.path))
        with self.assertRaises(ConnectionError):
            local_endpoint.connect(self.path, open_timeout=2.0)


def _upgrade_only_server(
    tmp: Path,
    drain: threading.Event | None = None,
    drained: queue.SimpleQueue[int] | None = None,
) -> tuple[threading.Thread, int, Path, threading.Event]:
    """A TLS server that completes the WebSocket upgrade and then never reads.

    With *drain*, it reads again once that event is set, to the end of
    the connection, and puts the number of bytes read on *drained*.

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
                    (drain or stop).wait(30)
                    if drained is not None:
                        drained.put(_read_to_the_end(tls))
        finally:
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return thread, port, ca_file, stop


def _read_to_the_end(tls: ssl.SSLSocket) -> int:
    """Read *tls* until the client ends the connection; return the number of bytes read."""
    count = 0
    tls.settimeout(20)
    try:
        while chunk := tls.recv(1 << 20):
            count += len(chunk)
    except OSError:  # a connection ended without a TLS close, or silent for 20 s
        pass
    return count


def _is_the_send_writer(frame_locals: dict[str, object]) -> bool:
    """Pause matcher at ``Thread._delete``: the thread ending is ``send``'s writer."""
    return getattr(frame_locals.get("self"), "name", "") == "sorcar-local-send"


def _the_first_call(_frame_locals: dict[str, object]) -> bool:
    """Pause matcher: hold the first thread that enters the function."""
    return True


def _send(ws: SyncClientConnection, text: str, timeout: float, errors: list[BaseException]) -> None:
    """Thread body: ``local_endpoint.send``, its error appended to *errors*."""
    try:
        local_endpoint.send(ws, text, timeout=timeout)
    except BaseException as exc:  # noqa: BLE001 - reported by the test
        errors.append(exc)


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

    def test_an_interrupted_send_releases_its_blocked_writer(self) -> None:
        """A Stop that lands while ``send`` waits for its writer must still
        close the socket underneath the writer.

        The writer holds the connection's protocol mutex while the
        daemon does not read, so the caller's cleanup ``ws.close()``
        would otherwise wait behind it for good (audit A09).
        """
        self._interrupt_send("call", threading.Thread.join.__code__, 30.0)

    def test_a_send_interrupted_as_its_writer_starts_releases_it(self) -> None:
        """The writer exists once ``Thread.start()`` returns: a Stop there is the same Stop."""
        self._interrupt_send("return", threading.Thread.start.__code__, 30.0)

    def test_a_send_interrupted_after_its_deadline_releases_its_writer(self) -> None:
        """A Stop between the timed-out wait and the timeout's own close of the socket."""
        self._interrupt_send("call", SyncClientConnection.close_socket.__code__, 0.5)

    def _open(
        self, drain: threading.Event, drained: queue.SimpleQueue[int],
    ) -> SyncClientConnection:
        """Connect to a server that reads nothing before *drain* and counts on *drained* then."""
        tmp = Path(tempfile.mkdtemp(prefix="kiss-ep-send-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        thread, port, ca_file, stop = _upgrade_only_server(tmp, drain, drained)
        self.addCleanup(thread.join, 5)
        self.addCleanup(stop.set)
        self.addCleanup(drain.set)
        ctx = ssl.create_default_context(cafile=str(ca_file))
        ws = sync_connect(f"wss://127.0.0.1:{port}/ws", ssl=ctx, open_timeout=5.0, compression=None)
        self.addCleanup(ws.close_socket)
        return ws

    def test_a_write_that_ended_is_no_timeout_while_its_thread_still_lives(self) -> None:
        """The deadline passes with the frame written and the writer thread not yet gone.

        Before: a writer thread still alive at the deadline was taken
        for a write still blocked: ``send`` closed the socket and
        reported a frame the daemon had as never sent, so a ``run``
        the daemon had started was reported as not started.
        """
        drain = threading.Event()
        drained: queue.SimpleQueue[int] = queue.SimpleQueue()
        ws = self._open(drain, drained)
        # ``Thread._delete`` is private and untyped: looked up, not named.
        delete = getattr(threading.Thread, "_delete")  # noqa: B009
        with Pause(delete.__code__, _is_the_send_writer) as held:
            local_endpoint.send(ws, "late", timeout=0.3)
            self.assertTrue(held.paused.is_set(), "the writer was not held at its exit")
        self.assertIs(ws.protocol.state, State.OPEN)
        drain.set()
        ws.close_socket()
        self.assertGreaterEqual(drained.get(timeout=30), len("late"))

    def test_a_write_that_ends_as_the_deadline_closes_the_socket_is_no_timeout(self) -> None:
        """The daemon reads the frame whole between the timed-out wait and the close.

        ``send`` returns: the frame is with the daemon.  The socket is
        closed by then, which the next read reports.  Before:
        ``ConnectionError``, the frame reported as never sent.
        """
        frame = "x" * (64 * 1024 * 1024)
        drain = threading.Event()
        drained: queue.SimpleQueue[int] = queue.SimpleQueue()
        ws = self._open(drain, drained)
        errors: list[BaseException] = []
        sender = threading.Thread(target=_send, args=(ws, frame, 0.5, errors), daemon=True)
        with Pause(SyncClientConnection.close_socket.__code__, _the_first_call) as held:
            sender.start()
            self.assertTrue(held.paused.wait(30), "the deadline's close was not reached")
            drain.set()  # the daemon reads again: the blocked write ends
            self._join_the_writer()
        sender.join(30)
        self.assertFalse(sender.is_alive())
        self.assertEqual(errors, [])
        with self.assertRaises(ConnectionClosed):
            ws.recv(timeout=5)
        self.assertGreaterEqual(drained.get(timeout=30), len(frame))

    def test_a_write_not_begun_when_the_last_wait_ends_is_a_timeout(self) -> None:
        """The writer gets no core before the deadline, its close and the wait after it are over.

        The frame was not written, and ``send`` says so.  The writer,
        once it runs, finds the socket closed: the daemon reads nothing.
        """
        drain = threading.Event()
        drained: queue.SimpleQueue[int] = queue.SimpleQueue()
        ws = self._open(drain, drained)
        with Pause(local_endpoint._write.__code__, _the_first_call) as held:
            with self.assertRaisesRegex(ConnectionError, "did not accept a 5-byte command"):
                local_endpoint.send(ws, "never", timeout=0.3)
            self.assertTrue(held.paused.is_set(), "the writer was not held ahead of its write")
        self._join_the_writer()
        drain.set()
        self.assertEqual(drained.get(timeout=30), 0)

    def _join_the_writer(self) -> None:
        """Wait for the end of ``send``'s writer thread."""
        for writer in threading.enumerate():
            if writer.name == "sorcar-local-send":
                writer.join(30)
                self.assertFalse(writer.is_alive(), "the write did not end")

    def _interrupt_send(self, at_event: str, at_code: object, timeout: float) -> None:
        """Raise a Stop in ``send`` at *at_event* of *at_code*; nothing may stay blocked."""
        tmp = Path(tempfile.mkdtemp(prefix="kiss-ep-send-"))
        thread, port, ca_file, stop = _upgrade_only_server(tmp)
        path = tmp / "sorcar-local.json"
        local_endpoint.write_endpoint(path, local_endpoint.LocalEndpoint(
            url=f"wss://127.0.0.1:{port}/ws", token="t" * 64, ca=str(ca_file), pid=1,
        ))

        def interrupt_the_join(frame: object, event: str, arg: object) -> None:
            # The KeyboardInterrupt of a Stop, delivered the moment the
            # caller's thread reaches that point (no sleeps, no patches).
            if event == at_event and getattr(frame, "f_code", None) is at_code:
                sys.setprofile(None)
                raise KeyboardInterrupt

        try:
            endpoint = local_endpoint.read_endpoint(path)
            assert endpoint is not None
            ws = sync_connect(
                endpoint.url, ssl=local_endpoint.client_ssl_context(endpoint),
                open_timeout=5.0, compression=None,
            )
            sys.setprofile(interrupt_the_join)
            try:
                with self.assertRaises(KeyboardInterrupt):
                    local_endpoint.send(ws, "x" * (64 * 1024 * 1024), timeout=timeout)
            finally:
                sys.setprofile(None)
            closer = threading.Thread(target=ws.close, daemon=True)
            closer.start()
            closer.join(5.0)
            self.assertFalse(
                closer.is_alive(), "ws.close() hung behind the writer the interrupt abandoned",
            )
            for writer in threading.enumerate():
                if writer.name == "sorcar-local-send":
                    writer.join(5.0)
                    self.assertFalse(
                        writer.is_alive(), "the abandoned writer is still blocked on the daemon",
                    )
        finally:
            stop.set()
            thread.join(5)
            shutil.rmtree(tmp, ignore_errors=True)
