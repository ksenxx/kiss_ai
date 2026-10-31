# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: the daemon knows whether the standalone app's popup window is connected.

The macOS ``kiss-app`` window (``kiss.server.standalone_app``) connects
to the daemon as ``/ws?appWindow=1``.  The daemon counts those
connections and answers ``GET /api/app-window`` with ``{"open": ...}``,
which ``kiss-app`` asks before it sends a running browser a reopen (a
browser without a window would answer that with an empty browser
window).  Other clients do not count, and a reconnect that overlaps
the old connection (a new one opens before the old one closes) never
makes the daemon believe the window is gone.  On macOS the last
connection closing with the page gone (close code 1001) also arms the
quit check for the brand browser under HOME, which the quit test here
exercises with a stand-in browser under a temp HOME; a connection
closing any other way (the daemon's keepalive timed out) does not arm
it, as the page is still there.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import ClientConnection, connect

import kiss.core.user_config as vc
from kiss.server import standalone_app
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.server.conftest import (
    find_free_port,
    no_verify_ssl,
    redirect_persistence,
    restore_persistence,
)


class TestAppWindowConnection(IsolatedAsyncioTestCase):
    """``/api/app-window`` follows the popup's WebSocket connections."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-app-window-")
        self.saved = redirect_persistence(self.tmpdir)
        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"
        self._orig_home = os.environ.get("HOME")
        os.environ["HOME"] = self.tmpdir

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = find_free_port()
        self.ctx = no_verify_ssl()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()
        self._sockets: list[ClientConnection] = []

    async def asyncTearDown(self) -> None:
        for ws in self._sockets:
            await ws.close()
        await self.server.stop_async()
        restore_persistence(self.saved)
        vc.CONFIG_DIR = self._orig_cfg_dir
        vc.CONFIG_PATH = self._orig_cfg_path
        if self._orig_home is None:
            del os.environ["HOME"]
        else:
            os.environ["HOME"] = self._orig_home
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(self, query: str = "") -> ClientConnection:
        """Open and authenticate a WebSocket connection to ``/ws`` + *query*."""
        ws = await connect(f"wss://127.0.0.1:{self.port}/ws{query}", ssl=self.ctx)
        self._sockets.append(ws)
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        self.assertEqual(resp["type"], "auth_ok")
        return ws

    async def _close(self, ws: ClientConnection) -> None:
        """Close *ws* and give the daemon a moment to notice."""
        await ws.close()
        self._sockets.remove(ws)
        await asyncio.sleep(0.3)

    async def _open(self) -> bool | None:
        """What kiss-app learns: :func:`standalone_app.ask_app_window_open` on the daemon.

        Asked at the daemon's plain-HTTP loopback URL from its URL file,
        as kiss-app does (it opens the probe itself, bypassing any proxy).
        """
        url = standalone_app.read_loopback_url(Path(self.tmpdir) / "remote-url.json")
        self.assertIsNotNone(url)
        return await asyncio.to_thread(standalone_app.ask_app_window_open, str(url))

    async def test_popup_connections_are_counted(self) -> None:
        """Open while a marked connection lives; other clients do not count, nor an overlap."""
        self.assertIs(await self._open(), False)
        other = await self._connect()
        self.assertIs(await self._open(), False, "an ordinary client is not the popup")
        popup = await self._connect("?appWindow=1")
        self.assertIs(await self._open(), True)
        # A reconnect whose new connection opens before the old closes.
        popup2 = await self._connect("?appWindow=1")
        await self._close(popup)
        self.assertIs(await self._open(), True, "the popup is still connected")
        await self._close(popup2)
        self.assertIs(await self._open(), False)
        await self._close(other)
        self.assertIs(await self._open(), False)

    @unittest.skipUnless(platform.system() == "Darwin", "the daemon quits the browser on macOS")
    async def test_the_popup_going_away_quits_the_windowless_browser(self) -> None:
        """The page gone (close code 1001) quits a windowless browser; a drop does not.

        The browser is a stand-in running from the brand bundle under
        this test's HOME (it has no window, like a real one whose popup
        closed).  The popup's connection closing with any other code
        (the daemon's keepalive timed out) leaves it alone: the page is
        still there, maybe minimized, and connects again.
        """
        bundle = standalone_app.macos_bundle_path()
        exe = bundle / "Contents" / "MacOS" / "Chromium"
        exe.parent.mkdir(parents=True)
        # argv[0] is the executable's path, as for a real browser (ps
        # shows it, which is how the daemon finds the browser's pid).
        exe.write_text('#!/bin/bash\nexec -a "$0" sleep 300\n')
        exe.chmod(0o755)
        browser = subprocess.Popen([str(exe)])
        try:
            for _ in range(50):
                if standalone_app.macos_bundle_pid(bundle) == browser.pid:
                    break
                await asyncio.sleep(0.1)
            self.assertEqual(standalone_app.macos_bundle_pid(bundle), browser.pid)
            popup = await self._connect("?appWindow=1")
            self._sockets.remove(popup)
            await popup.close(code=1000)
            await asyncio.sleep(standalone_app.APP_WINDOW_QUIT_GRACE_SECS + 1)
            self.assertIsNone(browser.poll(), "a dropped connection quits nothing")
            self.assertIs(await self._open(), False)
            popup = await self._connect("?appWindow=1")
            self._sockets.remove(popup)
            await popup.close(code=standalone_app.APP_WINDOW_GONE_CLOSE_CODE)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + standalone_app.APP_WINDOW_QUIT_GRACE_SECS + 5
            while browser.poll() is None and loop.time() < deadline:
                await asyncio.sleep(0.2)
            self.assertEqual(browser.returncode, -signal.SIGTERM)
        finally:
            if browser.poll() is None:
                browser.kill()
            browser.wait()

    async def test_a_daemon_that_does_not_answer_gives_no_verdict(self) -> None:
        """kiss-app gets None (not False) from a URL that serves no JSON state."""
        self.assertIsNone(standalone_app.ask_app_window_open("http://127.0.0.1:9/", timeout=1))
        self.assertIsNone(
            standalone_app.ask_app_window_open(f"https://127.0.0.1:{self.port}/nothing/", timeout=5)
            if False
            else standalone_app.ask_app_window_open(f"http://127.0.0.1:{self.port}/", timeout=5)
        )
