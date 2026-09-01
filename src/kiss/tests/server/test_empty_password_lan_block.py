# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: an empty ``remote_password`` means localhost-only access.

Security hardening being locked in: the daemon binds ``0.0.0.0`` by
default, and :meth:`ServerApi.authenticate` accepts an empty submitted
password when the configured ``remote_password`` is empty.  Before the
fix, that combination let ANY machine on the LAN open the webapp with
no credentials at all — the empty-password state only disabled the
public cloudflared tunnel, not direct LAN connections.

The fix, exercised here over real TLS sockets against a live
``RemoteAccessServer``:

1. ``RemoteAccessServer._process_request`` — which every parsed HTTP
   request AND the ``/ws`` WebSocket upgrade pass through — answers
   **403** to any non-loopback TCP peer while the configured password
   is empty.  The one request kind answered before the parser runs,
   the HEAD health check (``_HeadAwareServerConnection``), applies the
   same rule via ``_head_health_response``.
2. ``ServerApi.authenticate`` re-loads the password for every auth
   attempt and re-applies the localhost-only rule, so clearing (or
   changing) the password while a peer awaits credentials takes
   effect immediately — a stale snapshot is never honoured.
3. Loopback clients keep the historical behaviour: full HTTP access
   and successful empty-password authentication.
4. With a NON-empty password configured, LAN peers are served and
   authenticate normally, so the gate does not regress password-
   protected setups (where cloudflared, a loopback peer, relays real
   remote visitors anyway).
5. Clearing the password also tears the tunnel infrastructure down:
   the watchdog terminates a live cloudflared and withdraws the
   advertised URL, and startup terminates an orphaned cloudflared
   left by a previous instance instead of letting it keep serving.

Unreachable-branch note (documented instead of mocked, per testing
policy): ``_connection_peer_is_loopback``'s fail-closed arm for a
missing / empty ``remote_address`` cannot be reached end-to-end —
every real TCP WebSocket connection has a peer address — and would
require a test double to trigger, so it is intentionally not covered
here.

Every test that needs a non-loopback route to this host discovers the
machine's own LAN IP and connects to it; on a loopback-only machine
those tests skip.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import websockets
from websockets.asyncio.client import connect

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server import web_server as ws_mod
from kiss.server.web_server import RemoteAccessServer

_PASSWORD = "correct-horse-battery-staple"
_PASSWORD2 = "a-different-password"
_PASSWORD3 = "yet-another-password"


def _pick_free_port() -> int:
    """Return an OS-assigned free TCP port on this host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("0.0.0.0", 0))
        return int(sock.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    """Permissive SSL context for the dev self-signed cert."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _lan_ip() -> str | None:
    """Return a non-loopback IP of this machine, or None if it has none.

    Uses the connected-UDP-socket trick (no packet is actually sent) to
    learn the source address the OS would use for outbound traffic.
    Connecting to that address from this same machine produces a real
    TCP connection whose peer address is non-loopback — exactly what a
    LAN client would look like to the server.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))  # TEST-NET-1; nothing is sent.
            ip = str(sock.getsockname()[0])
    except OSError:
        return None
    if ipaddress.ip_address(ip).is_loopback:
        return None
    return ip


class TestEmptyPasswordLanBlock(IsolatedAsyncioTestCase):
    """Empty ``remote_password`` must refuse every non-loopback peer."""

    async def asyncSetUp(self) -> None:
        """Start a real server on all interfaces with an EMPTY password."""
        self._port = _pick_free_port()
        self._orig_config: str | None = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self._server = RemoteAccessServer(
            host="0.0.0.0",
            port=self._port,
            work_dir=tempfile.mkdtemp(),
            use_tunnel=False,
        )
        await self._server.start_async()

    async def asyncTearDown(self) -> None:
        """Stop the server; restore the config even if the stop fails."""
        try:
            await self._server.stop_async()
        finally:
            if self._orig_config is not None:
                CONFIG_PATH.write_text(self._orig_config)
            elif CONFIG_PATH.exists():
                CONFIG_PATH.unlink()

    def _require_lan_ip(self) -> str:
        """Return this machine's non-loopback IP or skip the test."""
        ip = _lan_ip()
        if ip is None:
            self.skipTest("machine has no non-loopback interface")
        return ip

    async def _http_status(self, host: str, path: str) -> int:
        """GET ``https://host:port{path}`` and return the HTTP status."""
        url = f"https://{host}:{self._port}{path}"

        def fetch() -> int:
            try:
                with urllib.request.urlopen(
                    url, context=_no_verify_ssl(), timeout=10,
                ) as resp:
                    return int(resp.status)
            except urllib.error.HTTPError as exc:
                return int(exc.code)

        return await asyncio.to_thread(fetch)

    async def _head_status_line(self, host: str) -> str:
        """Send a raw HEAD request over TLS; return the status line.

        HEAD requests are answered by ``_HeadAwareServerConnection``
        BEFORE the websockets HTTP parser runs, so ``urllib`` (which
        insists on a parseable body-less response) is bypassed in
        favour of a raw socket.
        """

        def probe() -> str:
            with socket.create_connection((host, self._port), 10) as raw:
                with _no_verify_ssl().wrap_socket(
                    raw, server_hostname=host,
                ) as tls:
                    tls.sendall(
                        b"HEAD /api/jobs HTTP/1.1\r\n"
                        b"Host: kiss\r\n\r\n"
                    )
                    return tls.recv(1024).split(b"\r\n")[0].decode()

        return await asyncio.to_thread(probe)

    async def _ws_connect(self, host: str) -> Any:
        """Open a WSS connection to /ws via *host*."""
        return await connect(
            f"wss://{host}:{self._port}/ws", ssl=_no_verify_ssl(),
        )

    async def _auth_reply(self, host: str, password: str) -> str | None:
        """Complete a WS upgrade + one ``auth`` frame; return reply type.

        Returns the server's first message ``type``, or ``None`` when
        the connection attempt or handshake failed outright.
        """
        try:
            async with await self._ws_connect(host) as ws:
                with contextlib.suppress(Exception):
                    await ws.send(
                        json.dumps({"type": "auth", "password": password})
                    )
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                return str(json.loads(raw).get("type"))
        except Exception:
            return None

    async def test_lan_http_requests_are_403(self) -> None:
        """Every parsed HTTP path must answer 403 to a non-loopback peer."""
        ip = self._require_lan_ip()
        for path in ("/", "/trajectories", "/api/jobs", "/media/api.js"):
            status = await self._http_status(ip, path)
            self.assertEqual(
                status, 403,
                f"GET {path} from LAN peer {ip} must be 403 while the "
                f"remote password is empty; got {status}",
            )

    async def test_lan_head_request_is_403(self) -> None:
        """The pre-parser HEAD health check must also refuse LAN peers."""
        ip = self._require_lan_ip()
        self.assertIn("403", await self._head_status_line(ip))
        # Loopback HEAD keeps answering 200 — cloudflared's origin
        # health checks arrive from loopback and must never break.
        self.assertIn("200", await self._head_status_line("127.0.0.1"))

    async def test_lan_ws_upgrade_is_refused(self) -> None:
        """The /ws upgrade itself must be rejected with HTTP 403."""
        ip = self._require_lan_ip()
        with self.assertRaises(websockets.exceptions.InvalidStatus) as cm:
            await self._ws_connect(ip)
        self.assertEqual(cm.exception.response.status_code, 403)

    async def test_localhost_keeps_full_access(self) -> None:
        """Loopback clients still get the page and empty-password auth."""
        self.assertEqual(await self._http_status("127.0.0.1", "/"), 200)
        self.assertEqual(
            await self._auth_reply("127.0.0.1", ""), "auth_ok",
            "An empty password must still authenticate a loopback "
            "client when no remote_password is configured.",
        )

    async def test_lan_allowed_once_password_is_set(self) -> None:
        """Setting a password re-enables LAN access (with auth)."""
        ip = self._require_lan_ip()
        save_config({"remote_password": _PASSWORD})
        self.assertEqual(await self._http_status(ip, "/"), 200)
        self.assertIn("200", await self._head_status_line(ip))
        self.assertEqual(await self._auth_reply(ip, _PASSWORD), "auth_ok")
        self.assertEqual(
            await self._auth_reply(ip, "wrong-guess"), "auth_required",
        )

    async def test_password_cleared_mid_handshake_refuses_lan_peer(
        self,
    ) -> None:
        """Clearing the password mid-handshake evicts a LAN peer.

        A LAN peer admitted while a password was set must NOT be
        authenticated by the empty-vs-empty compare (or by the stale
        old password) once the password is cleared: ``authenticate``
        re-loads the config for every attempt.
        """
        ip = self._require_lan_ip()
        save_config({"remote_password": _PASSWORD})
        async with await self._ws_connect(ip) as ws:
            save_config({"remote_password": ""})
            with contextlib.suppress(Exception):
                await ws.send(
                    json.dumps({"type": "auth", "password": _PASSWORD})
                )
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            self.assertEqual(
                json.loads(raw).get("type"), "error",
                "A non-loopback peer must be refused once the "
                "configured password is cleared mid-handshake.",
            )
            with self.assertRaises(websockets.exceptions.ConnectionClosed):
                await asyncio.wait_for(ws.recv(), timeout=5)

    async def test_password_change_mid_handshake_applies_immediately(
        self,
    ) -> None:
        """A mid-handshake password CHANGE invalidates the old password.

        The compare must use the password re-loaded for EVERY attempt:
        the password is changed once before the first attempt (whose
        stale pre-change password earns ``auth_required``) and AGAIN
        between the attempts (so a reload done only at the first
        attempt would reject the retry).  On loopback, clearing the
        password makes the empty probe succeed.
        """
        save_config({"remote_password": _PASSWORD})
        async with await self._ws_connect("127.0.0.1") as ws:
            save_config({"remote_password": _PASSWORD2})
            await ws.send(
                json.dumps({"type": "auth", "password": _PASSWORD})
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            self.assertEqual(json.loads(raw).get("type"), "auth_required")
            save_config({"remote_password": _PASSWORD3})
            await ws.send(
                json.dumps({"type": "auth", "password": _PASSWORD3})
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            self.assertEqual(json.loads(raw).get("type"), "auth_ok")
        async with await self._ws_connect("127.0.0.1") as ws:
            save_config({"remote_password": ""})
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            self.assertEqual(
                json.loads(raw).get("type"), "auth_ok",
                "Loopback empty-password auth must work as soon as "
                "the password is cleared.",
            )

    async def test_pre_handshake_gate_refuses_lan_peer(self) -> None:
        """``authenticate``'s pre-recv gate refuses a non-loopback peer.

        Through the live server this branch is shadowed by the 403
        upgrade refusal, so drive the same real handshake —
        :meth:`ServerApi.authenticate` bound to the running
        ``RemoteAccessServer`` — over a dedicated real TCP WebSocket
        from the LAN address with the password already empty: the gate
        must fire before any credential frame is read.
        """
        ip = self._require_lan_ip()
        api = self._server._server_api
        results: list[bool] = []

        async def handshake(ws: Any) -> None:
            results.append(await api.authenticate(ws))

        async with websockets.serve(handshake, "0.0.0.0", 0) as srv:
            port = srv.sockets[0].getsockname()[1]  # type: ignore[index]
            async with connect(f"ws://{ip}:{port}/") as ws:
                # The gate refuses the peer as soon as the handshake
                # starts — possibly before this frame is sent, in which
                # case the send fails on the already-closed socket.
                with contextlib.suppress(Exception):
                    await ws.send(
                        json.dumps({"type": "auth", "password": ""})
                    )
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                self.assertEqual(json.loads(raw).get("type"), "error")
                with self.assertRaises(
                    websockets.exceptions.ConnectionClosed,
                ):
                    await asyncio.wait_for(ws.recv(), timeout=5)
        self.assertEqual(results, [False])

    async def test_watchdog_kills_live_tunnel_when_password_cleared(
        self,
    ) -> None:
        """The watchdog terminates a live tunnel once the password is empty.

        Refusing (re)starts is not enough: a running cloudflared keeps
        the public URL resolving to this server over loopback, where
        the empty password authenticates.  A real subprocess stands in
        for the tracked tunnel process; the config is already empty.
        """
        server = self._server
        proc = subprocess.Popen(
            ["sleep", "300"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        # Redirect the pidfile: terminating the tracked proc unlinks
        # it, and this test must not touch a real deployment's
        # cloudflared bookkeeping.
        orig_pidfile = ws_mod._CLOUDFLARED_PIDFILE
        ws_mod._CLOUDFLARED_PIDFILE = (
            Path(tempfile.mkdtemp()) / "cloudflared.pid"
        )
        try:
            server.use_tunnel = True
            server._tunnel_proc = proc
            server._tunnel_metrics_port = 1
            server._active_url = "https://stale.trycloudflare.com"
            # Pre-mark the local URL as already posted so withdrawing
            # the tunnel URL does not post to the production ntfy
            # discovery topic from a test.
            server._last_posted_url = server._local_url
            await server._check_and_restart_tunnel()
            self.assertIsNotNone(
                proc.poll(),
                "The tracked tunnel process must be terminated when "
                "the remote password is empty.",
            )
            self.assertIsNone(server._tunnel_proc)
            self.assertEqual(server._active_url, server._local_url)
        finally:
            ws_mod._CLOUDFLARED_PIDFILE = orig_pidfile
            server.use_tunnel = False
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)

    async def test_startup_orphan_tunnel_terminated(self) -> None:
        """An orphaned cloudflared is killed when the password is empty.

        A previous instance deliberately leaves its cloudflared alive
        (so the public URL survives restarts).  A successor starting
        with an empty password must terminate it — instead of adopting
        it or leaving it to forward public traffic — and remove the
        pidfile.  A real process running a copy of ``sleep`` named
        ``cloudflared`` satisfies the identity check that guards the
        kill against recycled PIDs.
        """
        tmp = Path(tempfile.mkdtemp())
        sleep_bin = shutil.which("sleep")
        assert sleep_bin is not None
        fake = tmp / "cloudflared"
        shutil.copy(sleep_bin, fake)
        fake.chmod(0o755)
        proc = subprocess.Popen(
            [str(fake), "300"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Reap the child the moment SIGTERM lands, so the helper's
        # is-alive polling sees the death instead of a zombie.
        reaper = threading.Thread(target=proc.wait, daemon=True)
        reaper.start()
        # Redirect the pidfile so the test never overwrites or unlinks
        # a real deployment's cloudflared bookkeeping.
        orig_pidfile = ws_mod._CLOUDFLARED_PIDFILE
        ws_mod._CLOUDFLARED_PIDFILE = tmp / "cloudflared.pid"
        try:
            ws_mod._save_cloudflared_pidfile(
                proc.pid, 45678, "https://stale.trycloudflare.com",
            )
            await asyncio.to_thread(ws_mod._terminate_orphan_cloudflared)
            reaper.join(timeout=10)
            self.assertIsNotNone(
                proc.poll(),
                "The orphaned cloudflared must be terminated when the "
                "remote password is empty.",
            )
            self.assertIsNone(
                ws_mod._load_cloudflared_pidfile(),
                "The cloudflared pidfile must be removed.",
            )
        finally:
            ws_mod._CLOUDFLARED_PIDFILE = orig_pidfile
            if proc.poll() is None:
                proc.kill()
            shutil.rmtree(tmp, ignore_errors=True)
