# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``retry_after`` of an ``auth_locked`` refusal is when the lock really ends.

An IP is locked while it has :data:`_AUTH_FAIL_MAX` failed logins
within the last :data:`_AUTH_FAIL_WINDOW` seconds, so the lock ends
when the oldest of those failures leaves the window.  The refusal used
to report a full minute after the LAST failure instead: with failures
spread over 40 s the page counted down 60 s for a lock that ended
after 20.

A real ``RemoteAccessServer`` answers real WSS handshakes.  Four
earlier failures (40, 30, 20 and 10 s ago) are seeded into the
server's failure log; the fifth is a real wrong guess.  Time passing
is simulated by moving the logged failures back.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
import time
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.core.user_config import CONFIG_PATH, save_config
from kiss.server.web_server import _AUTH_FAIL_MAX, RemoteAccessServer
from kiss.tests.server.conftest import FORWARDED_VISITOR, find_free_port, no_verify_ssl

_PASSWORD = "correct-horse-battery-staple"
_VISITOR_IP = FORWARDED_VISITOR["X-Forwarded-For"]


class TestLockoutRetryAfter(IsolatedAsyncioTestCase):
    """A locked visitor is told the real time left on its lock."""

    async def asyncSetUp(self) -> None:
        self._port = find_free_port()
        self._orig_config: str | None = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": _PASSWORD})
        self._server = RemoteAccessServer(
            host="127.0.0.1", port=self._port, work_dir=tempfile.mkdtemp(), use_tunnel=False,
        )
        await self._server.start_async()

    async def asyncTearDown(self) -> None:
        await self._server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def _auth(self, password: str) -> dict[str, Any]:
        """Open one WSS connection, send one ``auth`` frame, return the first reply."""
        async with connect(
            f"wss://127.0.0.1:{self._port}/ws",
            ssl=no_verify_ssl(),
            additional_headers=FORWARDED_VISITOR,
        ) as ws:
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"type": "auth", "password": password}))
            reply: dict[str, Any] = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            return reply

    def _age_failures(self, seconds: float) -> None:
        """Make every logged failure of the visitor *seconds* older."""
        failures = self._server._auth_failures
        failures[_VISITOR_IP] = [t - seconds for t in failures[_VISITOR_IP]]

    async def test_retry_after_is_when_the_oldest_failure_leaves_the_window(self) -> None:
        now = time.monotonic()
        self._server._auth_failures[_VISITOR_IP] = [
            now - age for age in (40.0, 30.0, 20.0, 10.0)
        ]
        self.assertEqual(_AUTH_FAIL_MAX, 5)

        locked = await self._auth("wrong-guess")
        self.assertEqual(locked.get("type"), "auth_locked")
        retry_after = locked["retry_after"]
        self.assertIn(retry_after, (20, 21), "the lock ends 60 s after the oldest failure")

        # Two seconds before retry_after the visitor is still locked out...
        self._age_failures(retry_after - 2)
        self.assertEqual((await self._auth(_PASSWORD)).get("type"), "auth_locked")
        # ...and once retry_after has passed the right password gets in.
        self._age_failures(2.5)
        self.assertEqual((await self._auth(_PASSWORD)).get("type"), "auth_ok")
