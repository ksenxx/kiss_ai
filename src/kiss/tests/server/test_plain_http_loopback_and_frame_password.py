"""E2E: one port speaks plain HTTP to loopback and TLS to everyone; a
browser on the daemon's own machine needs no password; every frame of a
password-authenticated client must carry the password.

Covers, against a live ``RemoteAccessServer`` on ``0.0.0.0``:

* ``http://127.0.0.1:PORT/`` and ``ws://127.0.0.1:PORT/ws`` are served
  without TLS; ``https://`` / ``wss://`` on the same port still work
  (the first byte of each connection decides); a plain-HTTP request
  from a non-loopback peer is refused with 400 before any byte of the
  app is served; a HEAD health check works on both transports.
* The endpoint file names a ``ws://`` URL and local clients connect to
  it without a TLS context; the URL file's ``loopback`` is ``http://``.
* With ``remote_password`` configured, a DIRECT loopback peer (no
  forwarded client IP) is admitted on its ``auth`` frame without a
  password, gets no ``local`` privileges, and its later frames need no
  password either (a ``password`` key it sends is dropped unread); a
  loopback peer carrying ``Cf-Connecting-Ip`` / ``X-Forwarded-For`` (a
  cloudflared visitor) and a LAN peer still need the password; a
  ``local_only`` daemon still refuses a loopback browser.
* For a password-authenticated (forwarded) client the ``auth`` frame is
  checked against the stored password; a later frame without the
  password (or with a wrong one) is answered with ``error`` /
  ``auth_failed`` and the socket is closed; a frame with the password
  is served.  Rotating the password on disk shuts an authenticated
  session out at its next command.
* ``configData`` carries ``remote_password`` (the Settings panel shows
  it read-only) and ``saveConfig`` cannot change it.  (The browser side
  — ``ws://`` on an ``http:`` page, the password repeated in every
  frame — is covered by ``webapp/test/transportFramePassword.test.js``.)
* A local-token client whose upgrade request has neither ``User-Agent``
  nor ``Origin`` (the retired VS Code extension's hand-written client)
  is sent a ``notice`` pointing at the web app after ``auth_ok``; the
  ``websockets`` client (which sends ``User-Agent``) is not.

Unreachable branch (documented, not mocked): a peer that connects and
sends nothing for ``_SNIFF_TIMEOUT_SECONDS`` (10 s) is dropped; the
suite does not wait that long.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import socket
import ssl
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect
from websockets.typing import Origin

from kiss.agents.sorcar import local_endpoint
from kiss.core.brand import PRODUCT_NAME
from kiss.core.remote_password import generate_password
from kiss.core.user_config import CONFIG_PATH, load_config, save_config
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.conftest import find_free_port, no_verify_ssl


def _lan_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))
            ip = str(sock.getsockname()[0])
    except OSError:
        return None
    return None if ipaddress.ip_address(ip).is_loopback else ip


class _ServerCase(IsolatedAsyncioTestCase):
    password = ""

    async def asyncSetUp(self) -> None:
        self._port = find_free_port()
        self._orig_config = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        save_config({"remote_password": self.password})
        self._endpoint_file = Path(tempfile.mkdtemp()) / "sorcar-local.json"
        self._server = RemoteAccessServer(
            host="0.0.0.0", port=self._port, work_dir=tempfile.mkdtemp(),
            use_tunnel=False, local_endpoint_file=self._endpoint_file,
        )
        await self._server.start_async()

    async def asyncTearDown(self) -> None:
        try:
            await self._server.stop_async()
        finally:
            if self._orig_config is not None:
                CONFIG_PATH.write_text(self._orig_config)
            elif CONFIG_PATH.exists():
                CONFIG_PATH.unlink()

    def _get(self, url: str) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(url, context=no_verify_ssl(), timeout=10) as resp:
                return int(resp.status), resp.read()
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read()

    def _raw(self, host: str, request: bytes, tls: bool) -> bytes:
        with socket.create_connection((host, self._port), 10) as raw:
            if not tls:
                raw.sendall(request)
                return raw.recv(4096)
            with no_verify_ssl().wrap_socket(raw, server_hostname=host) as sock:
                sock.sendall(request)
                return sock.recv(4096)

    async def _recv(self, ws: Any) -> dict[str, Any]:
        """The next frame that is not the daemon's unsolicited ``update_available`` broadcast."""
        while True:
            reply = dict(json.loads(await asyncio.wait_for(ws.recv(), timeout=10)))
            if reply["type"] != "update_available":
                return reply

    def _token(self) -> str:
        endpoint = local_endpoint.read_endpoint(self._endpoint_file)
        assert endpoint is not None
        return endpoint.token


class TestPlainHttpLoopback(_ServerCase):
    """The listener serves plain HTTP on loopback and TLS everywhere."""

    async def test_plain_and_tls_on_one_port(self) -> None:
        """GET / works over http:// and https:// at the same port."""
        for scheme in ("http", "https"):
            status, body = await asyncio.to_thread(
                self._get, f"{scheme}://127.0.0.1:{self._port}/",
            )
            self.assertEqual(status, 200, scheme)
            self.assertIn(b"<html", body.lower(), scheme)

    async def test_head_health_check_on_both_transports(self) -> None:
        """Cloudflared's HEAD probe is answered with 200 plain or TLS."""
        head = b"HEAD / HTTP/1.1\r\nHost: kiss\r\n\r\n"
        for tls in (False, True):
            reply = await asyncio.to_thread(self._raw, "127.0.0.1", head, tls)
            self.assertTrue(reply.startswith(b"HTTP/1.1 200"), (tls, reply))

    async def test_tls_request_and_close_notify_in_one_write(self) -> None:
        """TLS 1.3 may deliver the request and the EOF right behind the handshake.

        Both then reach the connection before ``_sniff`` has handed the
        transport to websockets; they must be replayed in order, so the
        HEAD is still answered.
        """
        reply = await asyncio.to_thread(self._tls_head_then_close_notify, "127.0.0.1")
        self.assertTrue(reply.startswith(b"HTTP/1.1 200"), reply)

    def _tls_head_then_close_notify(self, host: str) -> bytes:
        incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
        tls = no_verify_ssl().wrap_bio(incoming, outgoing, server_hostname=host)
        with socket.create_connection((host, self._port), 10) as raw:
            while True:
                try:
                    tls.do_handshake()
                    break
                except ssl.SSLWantReadError:
                    raw.sendall(outgoing.read())
                    incoming.write(raw.recv(65536))
            # The client's Finished is still in ``outgoing``: send it, the
            # request and the close_notify as one TCP write.
            tls.write(b"HEAD / HTTP/1.1\r\nHost: kiss\r\n\r\n")
            with contextlib.suppress(ssl.SSLWantReadError):
                tls.unwrap()
            raw.sendall(outgoing.read())
            reply = b""
            while b"\r\n\r\n" not in reply:
                data = raw.recv(65536)
                if not data:
                    break
                incoming.write(data)
                with contextlib.suppress(ssl.SSLError):
                    while True:
                        reply += tls.read(65536)
            return reply

    async def test_plain_http_from_lan_peer_is_refused(self) -> None:
        """A non-loopback peer gets 400 before the app is served in the clear."""
        ip = _lan_ip()
        if ip is None:
            self.skipTest("machine has no non-loopback interface")
        reply = await asyncio.to_thread(
            self._raw, ip, b"GET / HTTP/1.1\r\nHost: kiss\r\n\r\n", False,
        )
        self.assertTrue(reply.startswith(b"HTTP/1.1 400"), reply)
        self.assertIn(b"use https://", reply)
        # TLS from the same peer is served (no password: 403 is the gate).
        status, _ = await asyncio.to_thread(self._get, f"https://{ip}:{self._port}/")
        self.assertEqual(status, 403)

    async def test_urls_and_endpoint_file_are_plain(self) -> None:
        """loopbackUrl is http:// and local clients use ws:// with no TLS."""
        self.assertEqual(self._server._loopback_url, f"http://127.0.0.1:{self._port}")
        endpoint = local_endpoint.read_endpoint(self._endpoint_file)
        assert endpoint is not None
        self.assertEqual(endpoint.url, f"ws://127.0.0.1:{self._port}/ws")
        self.assertIsNone(local_endpoint.client_ssl_context(endpoint))
        async with connect(endpoint.url) as ws:
            await ws.send(json.dumps({"type": "auth", "token": endpoint.token}))
            reply = await self._recv(ws)
        self.assertEqual(reply, {"type": "auth_ok", "local": True})
        # A legacy wss:// endpoint keeps a TLS context; junk is refused.
        wss = local_endpoint.LocalEndpoint(
            url=f"wss://127.0.0.1:{self._port}/ws", token="t", ca=None, pid=0,
        )
        self.assertIsNotNone(local_endpoint.client_ssl_context(wss))
        junk = self._endpoint_file.with_name("junk.json")
        junk.write_text(json.dumps({"url": "https://x/ws", "token": "t"}))
        self.assertIsNone(local_endpoint.read_endpoint(junk))

    async def test_empty_password_loopback_browser_is_served(self) -> None:
        """With no password set, a loopback browser authenticates with "" and bare frames pass."""
        async with connect(f"ws://127.0.0.1:{self._port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": False})
            await ws.send(json.dumps({"type": "ping"}))
            self.assertEqual((await self._recv(ws))["type"], "pong")


class TestLoopbackNeedsNoPassword(_ServerCase):
    """The web app opened at a loopback address on the daemon's machine needs no password."""

    password = "Correct-Horse-Battery-1"

    def _origin(self, host: str = "127.0.0.1") -> Origin:
        """The ``Origin`` a browser sends for the page served at *host*."""
        return Origin(f"http://{host}:{self._port}")

    async def test_loopback_browser_auth_without_password(self) -> None:
        """``auth`` with "" from the page at 127.0.0.1 gets ``auth_ok`` (not local).

        Bare frames are served.
        """
        async with connect(f"ws://127.0.0.1:{self._port}/ws", origin=self._origin()) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": False})
            await ws.send(json.dumps({"type": "getConfig"}))
            while (reply := await self._recv(ws))["type"] != "configData":
                pass
            self.assertEqual(reply["config"]["remote_password"], self.password)
            # A stale password the browser still repeats is dropped unread.
            await ws.send(json.dumps({"type": "ping", "password": "stale-wrong"}))
            self.assertEqual((await self._recv(ws))["type"], "pong")
        # The page at http://localhost:<port> is the same browser.
        async with connect(
            f"ws://127.0.0.1:{self._port}/ws", origin=self._origin("localhost"),
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": False})

    async def test_wrong_password_from_loopback_browser_is_still_admitted(self) -> None:
        """What the ``auth`` frame carries is irrelevant for the loopback page (and not counted)."""
        for _ in range(6):  # more than the brute-force lockout allows
            async with connect(f"ws://127.0.0.1:{self._port}/ws", origin=self._origin()) as ws:
                await ws.send(json.dumps({"type": "auth", "password": "wrong"}))
                self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": False})

    async def test_loopback_program_and_foreign_page_still_need_the_password(self) -> None:
        """No ``Origin`` (a program on this machine) or a foreign one: password required.

        A foreign ``Origin`` is a proxied page.
        """
        for origin in (None, Origin("https://proxy.example"), Origin("http://10.0.0.5:8787")):
            async with connect(f"ws://127.0.0.1:{self._port}/ws", origin=origin) as ws:
                await ws.send(json.dumps({"type": "auth", "password": ""}))
                self.assertEqual((await self._recv(ws))["type"], "auth_required", origin)
                await ws.send(json.dumps({"type": "auth", "password": self.password}))
                self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": False}, origin)

    async def test_forwarded_visitor_on_loopback_still_needs_the_password(self) -> None:
        """A cloudflared visitor arrives on loopback with a forwarded IP: password required."""
        for header in ("Cf-Connecting-Ip", "X-Forwarded-For"):
            async with connect(
                f"ws://127.0.0.1:{self._port}/ws",
                origin=self._origin(),
                additional_headers={header: "203.0.113.9"},
            ) as ws:
                await ws.send(json.dumps({"type": "auth", "password": ""}))
                self.assertEqual((await self._recv(ws))["type"], "auth_required", header)
                await ws.send(json.dumps({"type": "auth", "password": "wrong"}))
                reply = await self._recv(ws)
                self.assertEqual(reply["type"], "error", header)
                self.assertEqual(reply["code"], "auth_failed", header)

    async def test_lan_peer_still_needs_the_password(self) -> None:
        """A non-loopback peer is not a browser on this machine."""
        ip = _lan_ip()
        if ip is None:
            self.skipTest("machine has no non-loopback interface")
        async with connect(f"wss://{ip}:{self._port}/ws", ssl=no_verify_ssl()) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            self.assertEqual((await self._recv(ws))["type"], "auth_required")
            await ws.send(json.dumps({"type": "auth", "password": self.password}))
            self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": False})

    async def test_local_token_still_gets_local_privileges(self) -> None:
        """The token path is unchanged: ``auth_ok`` with ``local: true``."""
        async with connect(f"ws://127.0.0.1:{self._port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "token": self._token()}))
            self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": True})


class TestLocalOnlyDaemonUnchanged(IsolatedAsyncioTestCase):
    """A ``local_only`` daemon admits token clients only, loopback browser or not."""

    async def test_loopback_browser_is_refused(self) -> None:
        orig = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        save_config({"remote_password": "Correct-Horse-Battery-1"})
        endpoint_file = Path(tempfile.mkdtemp()) / "sorcar-local.json"
        server = RemoteAccessServer(
            host="127.0.0.1", port=0, work_dir=tempfile.mkdtemp(),
            use_tunnel=False, local_endpoint_file=endpoint_file,
        )
        await server.start_private_async()
        try:
            endpoint = local_endpoint.read_endpoint(endpoint_file)
            assert endpoint is not None
            origin = Origin(f"http://127.0.0.1:{server.port}")
            async with connect(endpoint.url, origin=origin) as ws:
                await ws.send(json.dumps({"type": "auth", "password": ""}))
                reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                self.assertEqual(reply["type"], "error")
                self.assertEqual(reply["code"], "auth_failed")
            async with connect(endpoint.url) as ws:
                await ws.send(json.dumps({"type": "auth", "token": endpoint.token}))
                reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                self.assertEqual(reply, {"type": "auth_ok", "local": True})
        finally:
            await server.stop_async()
            if orig is not None:
                CONFIG_PATH.write_text(orig)
            elif CONFIG_PATH.exists():
                CONFIG_PATH.unlink()


class TestVsCodeNotice(_ServerCase):
    """The retired VS Code extension's client is told to use the web app."""

    async def test_token_client_without_user_agent_and_origin_gets_the_notice(self) -> None:
        """No ``User-Agent`` and no ``Origin`` on the upgrade: ``notice`` follows ``auth_ok``."""
        async with connect(
            f"ws://127.0.0.1:{self._port}/ws", user_agent_header=None,
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "token": self._token()}))
            self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": True})
            notice = await self._recv(ws)
        self.assertEqual(notice["type"], "notice")
        self.assertEqual(
            notice["text"],
            f"VS Code is no longer supported by {PRODUCT_NAME}. "
            f"Use the web app instead: http://127.0.0.1:{self._port}",
        )

    async def test_other_clients_get_no_notice(self) -> None:
        """A ``User-Agent`` (websockets client), an ``Origin`` (browser) or no token: no notice."""
        url = f"ws://127.0.0.1:{self._port}/ws"
        clients: tuple[dict[str, Any], ...] = (
            {},  # the websockets client's default User-Agent: Python/...
            {"user_agent_header": None, "additional_headers": {"Origin": "http://x"}},
        )
        for kwargs in clients:
            async with connect(url, **kwargs) as ws:
                await ws.send(json.dumps({"type": "auth", "token": self._token()}))
                self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": True})
                await ws.send(json.dumps({"type": "ping"}))
                self.assertEqual((await self._recv(ws))["type"], "pong", kwargs)
        # A loopback browser admitted without a token is not the extension.
        async with connect(
            f"ws://127.0.0.1:{self._port}/ws", user_agent_header=None,
            origin=Origin(f"http://127.0.0.1:{self._port}"),
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": False})
            await ws.send(json.dumps({"type": "ping"}))
            self.assertEqual((await self._recv(ws))["type"], "pong")


class TestFramePassword(_ServerCase):
    """Every frame of a password-authenticated client carries the password.

    The clients here connect from 127.0.0.1 with ``X-Forwarded-For`` set,
    as a cloudflared visitor does, so they are NOT direct loopback peers.
    """

    password = "Correct-Horse-Battery-1"
    _forwarded = {"X-Forwarded-For": "203.0.113.9"}

    async def _authed(self, scheme: str = "ws") -> Any:
        ssl = no_verify_ssl() if scheme == "wss" else None
        ws = await connect(
            f"{scheme}://127.0.0.1:{self._port}/ws", ssl=ssl,
            additional_headers=self._forwarded,
        )
        await ws.send(json.dumps({"type": "auth", "password": self.password}))
        self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": False})
        return ws

    async def test_auth_checks_the_stored_password(self) -> None:
        """The password in config.json, as install.sh wrote it, decides the handshake."""
        self.assertEqual(load_config()["remote_password"], self.password)
        async with connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=no_verify_ssl(),
            additional_headers=self._forwarded,
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": "wrong"}))
            self.assertEqual((await self._recv(ws))["type"], "auth_required")
        ws = await self._authed("wss")
        await ws.close()

    async def test_frame_with_password_is_served(self) -> None:
        ws = await self._authed()
        await ws.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(ws))["type"], "pong")
        await ws.close()

    async def test_frame_without_password_closes_socket(self) -> None:
        for bad in ({}, {"password": "wrong"}, {"password": 42}):
            ws = await self._authed()
            await ws.send(json.dumps({"type": "ping", **bad}))
            reply = await self._recv(ws)
            self.assertEqual(reply["type"], "error", bad)
            self.assertEqual(reply["code"], "auth_failed", bad)
            with self.assertRaises(Exception):
                await asyncio.wait_for(ws.recv(), timeout=10)
            await ws.close()

    async def test_rotated_password_shuts_session_out(self) -> None:
        """install.sh re-run (new password on disk) stops the old session at its next frame."""
        ws = await self._authed()
        await ws.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(ws))["type"], "pong")
        new_password = generate_password()
        save_config({"remote_password": new_password})
        await ws.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(ws))["code"], "auth_failed")
        await ws.close()
        self.password = new_password
        ws = await self._authed()
        await ws.send(json.dumps({"type": "ping", "password": new_password}))
        self.assertEqual((await self._recv(ws))["type"], "pong")
        await ws.close()

    async def test_config_carries_the_password_but_cannot_set_it(self) -> None:
        """configData shows the password; saveConfig with another one is ignored."""
        ws = await self._authed()
        await ws.send(json.dumps({"type": "getConfig", "password": self.password}))
        while (reply := await self._recv(ws))["type"] != "configData":
            pass
        self.assertEqual(reply["config"]["remote_password"], self.password)
        await ws.send(json.dumps({
            "type": "saveConfig", "password": self.password,
            "config": {"remote_password": "other", "max_budget": 7},
        }))
        while (reply := await self._recv(ws))["type"] != "configData":
            pass
        self.assertEqual(reply["config"]["remote_password"], self.password)
        self.assertEqual(reply["config"]["max_budget"], 7)
        self.assertEqual(load_config()["remote_password"], self.password)
        await ws.close()
