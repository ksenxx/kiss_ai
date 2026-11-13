# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: which promptlets the remote web app's page holds, and which follow the login.

The page (``GET /``) is served before any login, to whoever reaches
the port: it embeds the promptlets shipped with the package and none
of the user's own (``MY_INJECTION.md``).  Those reach the page with
``tricksData``, the daemon's answer to the ``ready`` of a page that
has authenticated.

Neither list is frozen at daemon start (``RemoteAccessServer`` once
rendered ``media/chat.html`` in ``__init__`` and served those bytes
for every later ``GET /``): both files are re-read, so the Inject
panel agrees with the daemon's ghost-text ``trick`` completions
(``tricks.read_tricks`` on every keystroke).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import ClientConnection, connect

from kiss.core.user_config import save_config
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.conftest import find_free_port, no_verify_ssl

_PASSWORD = "Correct-Horse-Battery-1"
_BUNDLED = "Bundled trick: read the error message."
_BUNDLED_NEW = "Bundled trick: bisect the regression."
_TRICK_A = "Alpha trick: run the tests first."
_TRICK_B = "Bravo trick: profile before optimising."


def _tricks_file(*tricks: str) -> str:
    """Return the text of a promptlet file holding *tricks*."""
    return "".join(f"## Trick\n\n{trick}\n\n" for trick in tricks)


class _RecordingPrinter:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class TestServedTricksFollowFileEdits(IsolatedAsyncioTestCase):
    """Edit the promptlet files while a password-protected daemon runs; reload the page."""

    async def asyncSetUp(self) -> None:
        self.port = find_free_port()
        self._saved_env = {
            k: os.environ.get(k) for k in ("KISS_HOME", "KISS_INJECTIONS_PATH")
        }
        kiss_dir = Path(tempfile.mkdtemp(prefix="kiss_html_fresh_")) / ".kiss"
        kiss_dir.mkdir(parents=True)
        self.my_injection = kiss_dir / "MY_INJECTION.md"
        self.my_injection.write_text(_tricks_file(_TRICK_A), encoding="utf-8")
        self.bundled = kiss_dir / "bundled_INJECTIONS.md"
        self.bundled.write_text(_tricks_file(_BUNDLED), encoding="utf-8")
        os.environ["KISS_HOME"] = str(kiss_dir)
        os.environ["KISS_INJECTIONS_PATH"] = str(self.bundled)
        save_config({"remote_password": _PASSWORD})  # in the home set just above
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    async def _page(self) -> str:
        """Return the page a tunnel visitor who has not logged in is served."""
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", self.port, ssl=no_verify_ssl(),
            server_hostname="localhost",
        )
        writer.write(
            b"GET / HTTP/1.1\r\nHost: localhost\r\n"
            b"Cf-Connecting-Ip: 203.0.113.9\r\nConnection: close\r\n\r\n"
        )
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(), timeout=15)
        writer.close()
        return raw.split(b"\r\n\r\n", 1)[1].decode("utf-8")

    async def _ready_tricks(self, ws: ClientConnection) -> tuple[list[str], int]:
        """Send the ``ready`` of a page load; return the promptlets and the user's count."""
        await ws.send(json.dumps({"type": "ready", "tabId": "t1", "password": _PASSWORD}))
        while True:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            if event["type"] == "tricksData":
                return event["tricks"], event["userCount"]

    def _daemon_trick_completions(self, query: str) -> list[str]:
        printer = _RecordingPrinter()
        backend = self.server._agent_server
        saved = backend.printer
        backend.printer = printer  # type: ignore[assignment]
        try:
            backend._complete_seq_latest[""] = 1
            backend._complete(query, 1)
        finally:
            backend.printer = saved
        comps = next(e for e in printer.events if e["type"] == "completions")
        return [c["text"] for c in comps["completions"] if c["type"] == "trick"]

    async def test_page_served_before_login_holds_no_personal_promptlet(self) -> None:
        """The page of a visitor who never logs in carries the bundled promptlets only."""
        body = await self._page()
        self.assertFalse(_TRICK_A in body, "the page discloses MY_INJECTION.md before any login")
        self.assertIn("window.__MY_TRICKS_COUNT__ = 0;", body)
        match = re.search(r"window\.__TRICKS__\s*=\s*(\[.*?\]);", body, re.DOTALL)
        assert match is not None, "TRICKS_JSON placeholder was not substituted"
        self.assertEqual(json.loads(match.group(1)), [_BUNDLED])

        # A package upgrade under the running daemon: the next page load follows it.
        self.bundled.write_text(_tricks_file(_BUNDLED_NEW), encoding="utf-8")
        self.assertIn(json.dumps([_BUNDLED_NEW]), await self._page())

    async def test_personal_promptlets_follow_the_login(self) -> None:
        """``ready`` from a logged-in page gets the user's promptlets, re-read on every load."""
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=no_verify_ssl(),
            additional_headers={"Cf-Connecting-Ip": "203.0.113.9"},
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": _PASSWORD}))
            self.assertEqual(json.loads(await ws.recv())["type"], "auth_ok")
            self.assertEqual(await self._ready_tricks(ws), ([_TRICK_A, _BUNDLED], 1))

            # The user edits the file while the daemon keeps running.
            self.my_injection.write_text(_tricks_file(_TRICK_B, _TRICK_A), encoding="utf-8")
            # The daemon's own completion source already sees the edit …
            self.assertEqual(self._daemon_trick_completions("Bravo"), [_TRICK_B])
            # … so the next page load must agree with it.
            self.assertEqual(
                await self._ready_tricks(ws), ([_TRICK_B, _TRICK_A, _BUNDLED], 2),
            )
