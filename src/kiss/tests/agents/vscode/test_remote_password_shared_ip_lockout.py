# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: the remote-password prompt must not be globally
suppressed by one visitor's failed guesses.

Bug being locked in
-------------------

The public WSS port is reached through the local ``cloudflared``
tunnel, which connects to the server over **loopback**.  The auth
brute-force lockout is keyed on the client IP.  When that key was the
raw TCP peer address, *every* tunnel visitor collapsed onto a single
``127.0.0.1`` bucket, so as few as :data:`_AUTH_FAIL_MAX` wrong guesses
from ONE actor (or one user fat-fingering the password across
reconnects) locked out **everyone**.  A locked IP is refused with
``auth_locked`` (or a bare close) instead of ``auth_required``, and the
webapp's password modal only opens on ``auth_required`` — so from a
fresh visitor's point of view "the remote webapp does not ask for the
remote password".

The fix keys the lockout on the *real* client IP that cloudflared
forwards in the upgrade request (``Cf-Connecting-Ip`` /
``X-Forwarded-For``), trusted only for loopback TCP peers.  A different
visitor therefore keeps getting the password prompt even while another
IP is locked out.

These tests use the real :class:`RemoteAccessServer` over a real TLS
WebSocket — no mocks.  cloudflared is simulated by connecting over
loopback and setting the ``Cf-Connecting-Ip`` header ourselves (exactly
what cloudflared does on the upgrade request).
"""

from __future__ import annotations

import asyncio
import json
import ssl
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import (
    _AUTH_FAIL_MAX,
    RemoteAccessServer,
    _forwarded_client_ip,
    _is_loopback_ip,
)


def _conn(**attrs: Any) -> ServerConnection:
    """Build a stand-in ``ServerConnection`` exposing only the attributes
    (``remote_address`` / ``request``) that the IP helpers read."""
    return cast(ServerConnection, SimpleNamespace(**attrs))


def _find_free_port() -> int:
    """Return a free TCP port on localhost."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _no_verify_ssl() -> ssl.SSLContext:
    """Return a client SSL context that skips certificate verification."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class TestForwardedIpHelpers(unittest.TestCase):
    """Unit tests for the per-client-IP helper functions."""

    def test_is_loopback_ipv4(self) -> None:
        self.assertTrue(_is_loopback_ip("127.0.0.1"))
        self.assertTrue(_is_loopback_ip("127.5.6.7"))

    def test_is_loopback_ipv6_and_mapped(self) -> None:
        self.assertTrue(_is_loopback_ip("::1"))
        self.assertTrue(_is_loopback_ip("::ffff:127.0.0.1"))

    def test_public_ip_is_not_loopback(self) -> None:
        self.assertFalse(_is_loopback_ip("203.0.113.7"))
        self.assertFalse(_is_loopback_ip("::ffff:203.0.113.7"))

    def test_malformed_ip_is_not_loopback(self) -> None:
        self.assertFalse(_is_loopback_ip(""))
        self.assertFalse(_is_loopback_ip("not-an-ip"))

    def test_forwarded_prefers_cf_connecting_ip(self) -> None:
        ws = _conn(
            request=SimpleNamespace(
                headers=Headers(
                    {
                        "Cf-Connecting-Ip": "203.0.113.9",
                        "X-Forwarded-For": "198.51.100.2, 10.0.0.1",
                    }
                )
            )
        )
        self.assertEqual(_forwarded_client_ip(ws), "203.0.113.9")

    def test_forwarded_falls_back_to_xff_first_hop(self) -> None:
        ws = _conn(
            request=SimpleNamespace(
                headers=Headers({"X-Forwarded-For": "198.51.100.2, 10.0.0.1"})
            )
        )
        self.assertEqual(_forwarded_client_ip(ws), "198.51.100.2")

    def test_forwarded_empty_when_no_headers(self) -> None:
        self.assertEqual(_forwarded_client_ip(_conn(request=None)), "")
        ws = _conn(request=SimpleNamespace(headers=Headers({})))
        self.assertEqual(_forwarded_client_ip(ws), "")


class TestClientIpBucketing(unittest.TestCase):
    """``_client_ip`` trusts the forwarded header only for loopback peers."""

    def setUp(self) -> None:
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=0, work_dir=tempfile.mkdtemp()
        )

    def _fake_ws(self, peer: str, cf_ip: str | None) -> ServerConnection:
        headers = Headers({"Cf-Connecting-Ip": cf_ip} if cf_ip else {})
        return _conn(
            remote_address=(peer, 12345),
            request=SimpleNamespace(headers=headers),
        )

    def test_loopback_peer_uses_forwarded_ip(self) -> None:
        ws = self._fake_ws("127.0.0.1", "203.0.113.9")
        self.assertEqual(self.server._client_ip(ws), "203.0.113.9")

    def test_loopback_peer_without_header_uses_peer(self) -> None:
        ws = self._fake_ws("127.0.0.1", None)
        self.assertEqual(self.server._client_ip(ws), "127.0.0.1")

    def test_non_loopback_peer_ignores_forwarded_ip(self) -> None:
        # A direct (non-tunnel) peer must not be able to spoof the header.
        ws = self._fake_ws("203.0.113.50", "10.0.0.1")
        self.assertEqual(self.server._client_ip(ws), "203.0.113.50")

    def test_unknown_peer_returns_placeholder(self) -> None:
        ws = _conn(remote_address=None, request=None)
        self.assertEqual(self.server._client_ip(ws), "?")


class TestSharedIpLockoutRegression(IsolatedAsyncioTestCase):
    """One visitor's failed guesses must not suppress another's prompt."""

    async def asyncSetUp(self) -> None:
        import kiss.agents.sorcar.persistence as _persistence

        self._saved_persistence = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        self._persistence_dir = Path(tempfile.mkdtemp(prefix="kiss_lockout_"))
        _persistence._KISS_DIR = self._persistence_dir
        _persistence._DB_PATH = self._persistence_dir / "sorcar.db"
        _persistence._db_conn = None

        self.port = _find_free_port()
        self._orig_config = (
            CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        )
        save_config({"remote_password": "correct-horse"})

        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, work_dir=tempfile.mkdtemp()
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

        import kiss.agents.sorcar.persistence as _persistence

        if _persistence._db_conn is not None:
            try:
                _persistence._db_conn.close()
            except Exception:
                pass
            _persistence._db_conn = None
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_persistence

    def _url(self) -> str:
        return f"wss://127.0.0.1:{self.port}/ws"

    async def _first_handshake_frame(self, client_ip: str, password: str) -> str:
        """Connect as *client_ip*, send one ``auth`` frame, return the
        server's first response ``type`` (or ``"closed"``).

        A locked-out IP is refused with an ``auth_locked`` frame followed
        immediately by a server close, so the ``auth`` ``send`` may race
        the close — both the send and the recv are therefore tolerant of
        a connection that is already closing.
        """
        headers = {"Cf-Connecting-Ip": client_ip}
        async with connect(
            self._url(), ssl=_no_verify_ssl(), additional_headers=headers
        ) as ws:
            try:
                await ws.send(json.dumps({"type": "auth", "password": password}))
            except Exception:
                pass
            try:
                resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                return str(resp.get("type"))
            except Exception:
                return "closed"

    async def _make_failures(self, client_ip: str, guesses: int) -> None:
        """Drive *guesses* wrong non-empty guesses from *client_ip*."""
        headers = {"Cf-Connecting-Ip": client_ip}
        made = 0
        while made < guesses:
            async with connect(
                self._url(), ssl=_no_verify_ssl(), additional_headers=headers
            ) as ws:
                # First (non-retry) wrong guess.
                await ws.send(json.dumps({"type": "auth", "password": "nope"}))
                try:
                    await asyncio.wait_for(ws.recv(), timeout=5)
                except Exception:
                    return
                made += 1
                if made >= guesses:
                    return
                # Retry wrong guess on the same socket.
                await ws.send(json.dumps({"type": "auth", "password": "nope2"}))
                try:
                    await asyncio.wait_for(ws.recv(), timeout=5)
                except Exception:
                    pass
                made += 1

    async def test_attacker_lockout_is_not_global(self) -> None:
        # Sanity: a fresh visitor is prompted for the password.
        self.assertEqual(
            await self._first_handshake_frame("198.51.100.7", ""),
            "auth_required",
        )

        # Attacker IP burns through the failure budget.
        await self._make_failures("203.0.113.99", _AUTH_FAIL_MAX + 1)

        # The attacker's own IP is now locked out.
        self.assertEqual(
            await self._first_handshake_frame("203.0.113.99", ""),
            "auth_locked",
        )

        # A DIFFERENT visitor must still be asked for the password.
        self.assertEqual(
            await self._first_handshake_frame("198.51.100.7", ""),
            "auth_required",
        )

    async def test_correct_password_still_authenticates(self) -> None:
        self.assertEqual(
            await self._first_handshake_frame("198.51.100.8", "correct-horse"),
            "auth_ok",
        )


if __name__ == "__main__":
    unittest.main()
