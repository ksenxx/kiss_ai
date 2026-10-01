# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the explicit ``127.0.0.1`` listener of kiss-web.

Background
----------
On macOS a ``0.0.0.0:8787`` listener does not stop another process
from binding ``127.0.0.1:8787`` with ``SO_REUSEADDR``; loopback
connections then go to the more specific socket.  VS Code's
Remote-SSH port forwarding did exactly that when a remote machine
also ran kiss-web on 8787, so ``https://127.0.0.1:8787`` reached the
remote daemon (untrusted certificate) instead of the local one.

The fix (:meth:`RemoteAccessServer._bind_loopback_alias`) binds
``127.0.0.1:port`` explicitly beside the wildcard listener, warns
with the holder's name when another process already has it, and
retries from the watchdog until the address is reclaimed.

These tests drive ``_setup_server`` on a free port and use real
sockets and a real child process as the "VS Code" holder.  The
Linux branch of :meth:`RemoteAccessServer._wants_loopback_alias`
(``sys.platform != "darwin"``) and the ``lsof``-missing branch of
:func:`_describe_port_listeners` cannot be reached on a Mac without
faking the platform or PATH, so they are not exercised here; the
macOS-only tests are skipped elsewhere.
"""

from __future__ import annotations

import asyncio
import errno
import os
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
from unittest import IsolatedAsyncioTestCase, TestCase, skipUnless

import kiss.server.web_server as ws_mod
from kiss.server.web_server import RemoteAccessServer, _describe_port_listeners

_HOLDER_SCRIPT = """
import socket, sys, time
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", int(sys.argv[1])))
s.listen(1)
print("ready", flush=True)
time.sleep(60)
"""


def _free_port() -> int:
    """Pick a TCP port that is currently free on the loopback address."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _hold_loopback(port: int) -> subprocess.Popen[str]:
    """Start a child process that listens on ``127.0.0.1:port`` like VS Code does."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SCRIPT, str(port)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ready"
    return proc


def _try_bind_loopback(port: int) -> int | None:
    """Bind ``127.0.0.1:port`` with ``SO_REUSEADDR``; return the errno on failure."""
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            return exc.errno
    return None


def _https_status(port: int) -> int:
    """Return the HTTP status of ``GET /`` over TLS at ``127.0.0.1:port``."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with socket.create_connection(("127.0.0.1", port), timeout=5) as raw, \
            ctx.wrap_socket(raw, server_hostname="localhost") as tls:
        tls.sendall(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        status_line = tls.recv(64).split(b"\r\n", 1)[0]
    return int(status_line.split()[1])


class _LoopbackAliasTestBase(IsolatedAsyncioTestCase):
    """Create a wildcard server on a free port and always stop it."""

    async def asyncSetUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.port = _free_port()
        self.server = RemoteAccessServer(
            host="0.0.0.0",
            port=self.port,
            work_dir=self._tmpdir.name,
            local_endpoint_file=f"{self._tmpdir.name}/sorcar-local.json",
        )

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        self._tmpdir.cleanup()


@skipUnless(sys.platform == "darwin", "wildcard/specific bind overlap is BSD-only")
class TestLoopbackOwnedAtStartup(_LoopbackAliasTestBase):
    """A normal start owns ``127.0.0.1:port`` so nobody can shadow it."""

    async def test_loopback_bound_and_not_hijackable(self) -> None:
        """Both listeners are up, serve HTTPS, and block a VS Code-style bind."""
        await self.server._setup_server()
        self.assertIsNotNone(self.server._ws_server)
        self.assertIsNotNone(self.server._ws_loopback_server)
        self.assertEqual(_try_bind_loopback(self.port), errno.EADDRINUSE)
        self.assertEqual(await asyncio.to_thread(_https_status, self.port), 200)
        # A second call is a no-op once the alias is held.
        self.assertTrue(await self.server._bind_loopback_alias())
        await self.server._watchdog_reclaim_loopback()

    async def test_stop_releases_loopback(self) -> None:
        """Stopping the server frees ``127.0.0.1:port`` again."""
        await self.server._setup_server()
        await self.server.stop_async()
        self.assertIsNone(self.server._ws_loopback_server)
        self.assertIsNone(_try_bind_loopback(self.port))

    async def test_daemon_shutdown_closes_loopback_too(self) -> None:
        """The daemon lifecycle (``_serve_async``) stops the alias on shutdown.

        ``_serve_async`` returns when ``_shutdown_future`` resolves (the
        SIGTERM path); its cleanup must close the loopback listener too.
        """
        serve_task = asyncio.create_task(self.server._serve_async())
        while self.server._shutdown_future is None:
            await asyncio.sleep(0.02)
        self.assertIsNotNone(self.server._ws_loopback_server)
        self.assertEqual(await asyncio.to_thread(_https_status, self.port), 200)
        self.server._request_loop_shutdown()
        await asyncio.wait_for(serve_task, 10)
        await asyncio.wait_for(self.server._ws_loopback_server.wait_closed(), 5)
        self.assertIsNone(_try_bind_loopback(self.port))

    async def test_ip_change_restart_closes_loopback_too(self) -> None:
        """The IP-change restart must stop the alias, not only the wildcard listener.

        Otherwise the alias would keep answering on 127.0.0.1 while the
        daemon manager restarts the process.
        """
        await self.server._setup_server()
        self.server._last_ips = frozenset({"10.0.0.1"})
        restarted = False
        for _ in range(ws_mod._IP_CHANGE_DEBOUNCE_TICKS):
            restarted = self.server._watchdog_check_ip_change(frozenset({"10.0.0.2"}))
        self.assertTrue(restarted)
        await asyncio.wait_for(self.server._ws_loopback_server.wait_closed(), 5)
        await asyncio.wait_for(self.server._ws_server.wait_closed(), 5)
        self.assertIsNone(_try_bind_loopback(self.port))


@skipUnless(sys.platform == "darwin", "wildcard/specific bind overlap is BSD-only")
class TestLoopbackHeldByAnotherProcess(_LoopbackAliasTestBase):
    """Start while a foreign process holds ``127.0.0.1:port``; reclaim later."""

    async def test_warns_then_reclaims_when_holder_exits(self) -> None:
        """The wildcard listener starts, the holder is named, and the watchdog reclaims."""
        holder = _hold_loopback(self.port)
        try:
            with self.assertLogs("kiss.server.web_server", level="WARNING") as logs:
                await self.server._setup_server()
            self.assertIsNotNone(self.server._ws_server)
            self.assertIsNone(self.server._ws_loopback_server)
            warning = "\n".join(logs.output)
            self.assertIn(f"127.0.0.1:{self.port} is bound by another process", warning)
            self.assertIn(f"[{holder.pid}]", warning)
            self.assertIn("Stop Forwarding Port", warning)
            # Still held: the watchdog retry fails quietly (no new warning
            # is asserted; the bind simply stays unavailable).
            self.assertFalse(await self.server._bind_loopback_alias())
            self.assertIsNone(self.server._ws_loopback_server)
        finally:
            holder.kill()
            holder.wait()
        with self.assertLogs("kiss.server.web_server", level="INFO") as logs:
            await self.server._watchdog_reclaim_loopback()
        self.assertIsNotNone(self.server._ws_loopback_server)
        self.assertIn(f"Reclaimed 127.0.0.1:{self.port}", "\n".join(logs.output))
        self.assertEqual(_try_bind_loopback(self.port), errno.EADDRINUSE)
        self.assertEqual(await asyncio.to_thread(_https_status, self.port), 200)


class TestSpecificHostNeedsNoAlias(IsolatedAsyncioTestCase):
    """A server bound to a specific address never adds a loopback alias."""

    async def test_loopback_host_has_no_alias(self) -> None:
        """``host="127.0.0.1"`` leaves ``_ws_loopback_server`` unset on every platform."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server = RemoteAccessServer(
                host="127.0.0.1",
                port=_free_port(),
                work_dir=tmpdir,
                local_endpoint_file=f"{tmpdir}/sorcar-local.json",
            )
            try:
                await server._setup_server()
                self.assertFalse(server._wants_loopback_alias())
                self.assertIsNone(server._ws_loopback_server)
                self.assertTrue(await server._bind_loopback_alias())
                await server._watchdog_reclaim_loopback()
                self.assertIsNone(server._ws_loopback_server)
            finally:
                await server.stop_async()


class TestDescribePortListeners(TestCase):
    """``_describe_port_listeners`` names foreign listeners and skips ourselves."""

    @skipUnless(shutil.which("lsof"), "_describe_port_listeners reports nothing without lsof")
    def test_names_foreign_holder_and_skips_own_process(self) -> None:
        """A child holding the port is reported as ``name[pid]``; a free port gives ``""``."""
        port = _free_port()
        holder = _hold_loopback(port)
        try:
            described = _describe_port_listeners(port)
        finally:
            holder.kill()
            holder.wait()
        self.assertIn(f"[{holder.pid}]", described)
        self.assertNotIn(f"[{os.getpid()}]", described)
        with socket.socket() as own:
            own.bind(("127.0.0.1", 0))
            own.listen(1)
            self.assertEqual(_describe_port_listeners(own.getsockname()[1]), "")
        self.assertEqual(_describe_port_listeners(_free_port()), "")
