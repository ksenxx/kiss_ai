# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: local (127.0.0.1) and LAN URLs accompany the
Cloudflare URL.

Wherever the server reports the active (Cloudflare tunnel) URL — the
``remote_url`` WebSocket broadcast rendered by the settings panel and
the welcome page, and the ``~/.kiss/remote-url.json`` file polled by
the VS Code extension — it must also report the ``https://127.0.0.1:PORT``
URL for the local machine and the ``https://<lan-ip>:PORT`` URLs for
other devices on the LAN.

LAN URLs are advertised only when a LAN client could actually connect:
the server must be bound to a non-loopback host AND a
``remote_password`` must be configured (``_process_request`` answers
non-loopback peers 403 when the password is empty).
"""

from __future__ import annotations

import asyncio
import json
import ssl
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase

from websockets.asyncio.client import connect

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import (
    _IP_CHANGE_DEBOUNCE_TICKS,
    _URL_FILE,
    RemoteAccessServer,
    _save_url_file,
)
from kiss.tests.conftest import TRANSIENT_REPLACE_READ_ERRORS
from kiss.tests.server.test_web_server import _find_free_port


def _no_verify_ssl() -> ssl.SSLContext:
    """SSL context that skips certificate verification (self-signed)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class TestSaveUrlFileLocalLan(TestCase):
    """`_save_url_file` persists the loopback and LAN URLs."""

    def test_all_fields_written(self) -> None:
        """local, tunnel, loopback and lan all land in the JSON file."""
        with tempfile.TemporaryDirectory() as td:
            url_file = Path(td) / "remote-url.json"
            _save_url_file(
                url_file,
                "https://localhost:9999",
                "https://x.trycloudflare.com",
                "https://127.0.0.1:9999",
                ["https://192.168.1.5:9999"],
            )
            data = json.loads(url_file.read_text())
            self.assertEqual(data["local"], "https://localhost:9999")
            self.assertEqual(data["tunnel"], "https://x.trycloudflare.com")
            self.assertEqual(data["loopback"], "https://127.0.0.1:9999")
            self.assertEqual(data["lan"], ["https://192.168.1.5:9999"])

    def test_optional_fields_omitted(self) -> None:
        """Absent tunnel/loopback/lan leave their keys out (legacy shape)."""
        with tempfile.TemporaryDirectory() as td:
            url_file = Path(td) / "remote-url.json"
            _save_url_file(url_file, "https://localhost:9999")
            data = json.loads(url_file.read_text())
            self.assertEqual(data, {"local": "https://localhost:9999"})

    def test_empty_lan_list_omitted(self) -> None:
        """An empty LAN list must not create an empty ``lan`` key."""
        with tempfile.TemporaryDirectory() as td:
            url_file = Path(td) / "remote-url.json"
            _save_url_file(
                url_file, "https://localhost:9999", None,
                "https://127.0.0.1:9999", [],
            )
            data = json.loads(url_file.read_text())
            self.assertNotIn("lan", data)
            self.assertEqual(data["loopback"], "https://127.0.0.1:9999")


class _LiveServerCase(IsolatedAsyncioTestCase):
    """Base: a live server with config + URL-file backup/restore."""

    host = "127.0.0.1"
    password = ""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": self.password})
        self._url_backup: bytes | None = None
        if _URL_FILE.is_file():
            self._url_backup = _URL_FILE.read_bytes()
        self.server = RemoteAccessServer(
            host=self.host,
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
        if self._url_backup is not None:
            _URL_FILE.write_bytes(self._url_backup)
        else:
            _URL_FILE.unlink(missing_ok=True)

    async def _auth_ws(self, ws: Any) -> None:
        """Authenticate the WebSocket with this class's password."""
        await ws.send(json.dumps({"type": "auth", "password": self.password}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert resp["type"] == "auth_ok", resp

    async def _recv_remote_url(self, ws: Any) -> dict[str, Any]:
        """Read events until a ``remote_url`` event arrives."""
        for _ in range(10):
            ev: dict[str, Any] = json.loads(
                await asyncio.wait_for(ws.recv(), timeout=5)
            )
            if ev.get("type") == "remote_url":
                return ev
        self.fail("no remote_url event received")


class TestLoopbackOnlyServer(_LiveServerCase):
    """Loopback bind + empty password: loopback URL yes, LAN URLs no."""

    async def test_broadcast_has_loopback_and_empty_lan(self) -> None:
        """remote_url carries loopbackUrl; lanUrls is [] (LAN would 403)."""
        self.server._active_url = "https://lan-test.trycloudflare.com"
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await self._auth_ws(ws)
            await ws.send(json.dumps({"type": "getWelcomeSuggestions"}))
            ev = await self._recv_remote_url(ws)
            self.assertEqual(ev["url"], "https://lan-test.trycloudflare.com")
            self.assertEqual(
                ev["loopbackUrl"], f"https://127.0.0.1:{self.port}"
            )
            self.assertEqual(ev["lanUrls"], [])

    async def test_url_file_has_loopback_no_lan(self) -> None:
        """Startup writes loopback; lan is omitted on a loopback bind."""
        data = json.loads(_URL_FILE.read_text())
        self.assertEqual(data["local"], f"https://localhost:{self.port}")
        self.assertEqual(data["loopback"], f"https://127.0.0.1:{self.port}")
        self.assertNotIn("lan", data)

    async def test_watchdog_rewrite_keeps_loopback(self) -> None:
        """A watchdog re-write of a deleted URL file keeps the new keys."""
        _URL_FILE.unlink()
        self.server._watchdog_check_url_file()
        data = json.loads(_URL_FILE.read_text())
        self.assertEqual(data["loopback"], f"https://127.0.0.1:{self.port}")

    async def test_localhost_host_string_also_gated(self) -> None:
        """A host given as the name ``localhost`` yields no LAN URLs."""
        srv = RemoteAccessServer(
            host="localhost", port=self.port + 1, work_dir=tempfile.mkdtemp(),
        )
        self.assertEqual(srv._lan_urls(), [])


class TestLanCapableServer(_LiveServerCase):
    """0.0.0.0 bind + password: LAN URLs are advertised everywhere."""

    host = "0.0.0.0"
    password = "lan-urls-test-pw"

    def _expected_lan(self) -> list[str]:
        return [
            f"https://{ip}:{self.port}"
            for ip in sorted(self.server._last_ips)
        ]

    async def test_broadcast_includes_loopback_and_lan(self) -> None:
        """getWelcomeSuggestions' remote_url has loopbackUrl and lanUrls."""
        self.server._active_url = "https://lan-test.trycloudflare.com"
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await self._auth_ws(ws)
            await ws.send(json.dumps({"type": "getWelcomeSuggestions"}))
            ev = await self._recv_remote_url(ws)
            self.assertEqual(ev["url"], "https://lan-test.trycloudflare.com")
            self.assertEqual(
                ev["loopbackUrl"], f"https://127.0.0.1:{self.port}"
            )
            self.assertEqual(ev["lanUrls"], self._expected_lan())

    async def test_url_file_has_lan_when_ips_exist(self) -> None:
        """Startup writes the lan list when the host has routable IPs."""
        data = json.loads(_URL_FILE.read_text())
        self.assertEqual(data["loopback"], f"https://127.0.0.1:{self.port}")
        expected = self._expected_lan()
        if expected:
            self.assertEqual(data["lan"], expected)
        else:
            self.assertNotIn("lan", data)

    async def test_lan_urls_empty_password_gate(self) -> None:
        """Clearing remote_password stops LAN URLs from being advertised."""
        save_config({"remote_password": ""})
        try:
            self.assertEqual(self.server._lan_urls(), [])
        finally:
            save_config({"remote_password": self.password})

    async def test_lan_urls_uses_cached_ips(self) -> None:
        """_lan_urls prefers the probed IP snapshot (no live socket probe)."""
        self.assertTrue(self.server._ips_probed)
        old = self.server._last_ips
        try:
            self.server._last_ips = frozenset({"10.9.8.7"})
            self.assertEqual(
                self.server._lan_urls(), [f"https://10.9.8.7:{self.port}"]
            )
        finally:
            self.server._last_ips = old

    async def test_lan_urls_empty_before_first_probe(self) -> None:
        """Before the first off-thread probe, no LAN URLs are advertised.

        Regression: `_lan_urls` used to fall back to a live
        `_get_local_ips()` socket probe, which blocked the asyncio
        event loop when a client connected during the setup window.
        """
        old = self.server._last_ips
        try:
            self.server._ips_probed = False
            self.server._last_ips = frozenset({"192.168.3.3"})
            self.assertEqual(self.server._lan_urls(), [])
        finally:
            self.server._ips_probed = True
            self.server._last_ips = old

    async def test_ip_change_republishes_urls_in_tunnel_mode(self) -> None:
        """A debounced IP change re-writes the URL file and re-broadcasts.

        Regression: the tunnel-mode branch of
        ``_watchdog_check_ip_change`` used to only log, leaving stale
        LAN URLs in ``remote-url.json`` and on every open panel.
        """
        self.server.use_tunnel = True
        self.server._active_url = "https://republish.trycloudflare.com"
        self.server._last_ips = frozenset({"192.168.77.1"})
        new_ips = frozenset({"192.168.77.2"})
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await self._auth_ws(ws)
            restarted = False
            for _ in range(_IP_CHANGE_DEBOUNCE_TICKS):
                restarted = self.server._watchdog_check_ip_change(new_ips)
            self.assertFalse(restarted)
            self.assertEqual(self.server._last_ips, new_ips)
            ev = await self._recv_remote_url(ws)
            self.assertEqual(ev["url"], "https://republish.trycloudflare.com")
            self.assertTrue(ev["tunnelActive"])
            self.assertEqual(
                ev["lanUrls"], [f"https://192.168.77.2:{self.port}"]
            )
        data = await self._wait_for_lan_in_url_file(
            [f"https://192.168.77.2:{self.port}"]
        )
        self.assertEqual(
            data["tunnel"], "https://republish.trycloudflare.com"
        )

    async def test_empty_baseline_adoption_republishes_urls(self) -> None:
        """Adopting the first non-empty IP baseline republishes URLs."""
        self.server._last_ips = frozenset()
        new_ips = frozenset({"192.168.88.5"})
        restarted = self.server._watchdog_check_ip_change(new_ips)
        self.assertFalse(restarted)
        await self._wait_for_lan_in_url_file(
            [f"https://192.168.88.5:{self.port}"]
        )

    async def _wait_for_lan_in_url_file(
        self, expected_lan: list[str]
    ) -> dict[str, Any]:
        """Poll the URL file (written off-thread) for the expected lan list.

        Returns:
            The parsed URL-file JSON once its ``lan`` list matches.
        """
        deadline = asyncio.get_event_loop().time() + 5
        data: dict[str, Any] = {}
        while asyncio.get_event_loop().time() < deadline:
            try:
                data = json.loads(_URL_FILE.read_text())
            except TRANSIENT_REPLACE_READ_ERRORS:
                data = {}  # the writer's ``os.replace`` is mid-swap (Windows)
            if data.get("lan") == expected_lan:
                return data
            await asyncio.sleep(0.02)
        self.fail(f"URL file never gained lan={expected_lan}: {data}")

    async def test_republish_write_failure_is_logged_not_raised(self) -> None:
        """A failing URL-file write in a republish is logged, not fatal.

        Points the server at an unwritable location and re-runs the
        republish path; the broadcast must still go out and no
        exception may surface.  The location is a path *under a regular
        file*, which no OS lets ``mkdir`` create (a POSIX-only ``/proc``
        path is just ``C:\\proc`` on Windows, where it gets created).
        """
        old_url_file = self.server._url_file
        self.server._last_ips = frozenset()
        not_a_dir = Path(self.server.work_dir) / "not-a-dir"
        not_a_dir.write_text("")
        try:
            self.server._url_file = not_a_dir / "remote-url.json"
            with self.assertLogs(
                "kiss.server.web_server", level="WARNING"
            ) as logs:
                self.server._watchdog_check_ip_change(
                    frozenset({"192.168.99.9"})
                )
                deadline = asyncio.get_event_loop().time() + 5
                while asyncio.get_event_loop().time() < deadline:
                    if any(
                        "URL-file re-write failed" in line
                        for line in logs.output
                    ):
                        break
                    await asyncio.sleep(0.02)
            self.assertTrue(
                any("URL-file re-write failed" in line for line in logs.output)
            )
        finally:
            self.server._url_file = old_url_file
