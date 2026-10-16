# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The per-IP auth lockout does not shut out the password-free loopback browser.

Audit 2026-10-16 (server-runner F2).  A web app page opened at
``http://127.0.0.1:<port>`` on the daemon's own machine is admitted
without a password (``_is_loopback_browser``).  The brute-force lockout
is keyed by client IP, and every peer that reaches loopback without a
forwarded client IP shares the ``127.0.0.1`` bucket, so wrong guesses
from one such peer locked the owner's own browser out for the whole
window although it would never have been password-checked.  Real
:class:`RemoteAccessServer` over a real TLS WebSocket; no mocks.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.core.user_config import CONFIG_PATH, save_config
from kiss.server.web_server import _AUTH_FAIL_MAX, RemoteAccessServer
from kiss.tests.server.conftest import find_free_port, no_verify_ssl


class TestLockedLoopbackBrowser(IsolatedAsyncioTestCase):
    """A locked ``127.0.0.1`` still admits the loopback browser page."""

    async def asyncSetUp(self) -> None:
        import kiss.agents.sorcar.persistence as _persistence

        self._saved_persistence = (
            _persistence._DB_PATH, _persistence._db_conn, _persistence._KISS_DIR,
        )
        self._persistence_dir = Path(tempfile.mkdtemp(prefix="kiss_lockout_"))
        _persistence._KISS_DIR = self._persistence_dir
        _persistence._DB_PATH = self._persistence_dir / "history.db"
        _persistence._db_conn = None

        self.port = find_free_port()
        self._orig_config = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        save_config({"remote_password": "correct-horse"})
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
        import kiss.agents.sorcar.persistence as _persistence

        _persistence._close_db()
        (
            _persistence._DB_PATH, _persistence._db_conn, _persistence._KISS_DIR,
        ) = self._saved_persistence

    def _url(self) -> str:
        return f"wss://127.0.0.1:{self.port}/ws"

    async def _first_reply(self, frame: dict[str, object], origin: str | None) -> str:
        """Connect from loopback (no forwarded IP), send *frame*, return the reply ``type``."""
        async with connect(self._url(), ssl=no_verify_ssl(), origin=origin) as ws:  # type: ignore[arg-type]
            try:
                await ws.send(json.dumps(frame))
            except Exception:
                pass
            try:
                resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                return str(resp.get("type"))
            except Exception:
                return "closed"

    async def _lock_loopback(self) -> None:
        """Guess wrong passwords from a loopback program until ``127.0.0.1`` is locked."""
        for _ in range(_AUTH_FAIL_MAX + 2):
            reply = await self._first_reply({"type": "auth", "password": "nope"}, None)
            if reply == "auth_locked":
                return
        self.fail("the loopback bucket never locked")

    async def test_loopback_browser_is_admitted_while_its_ip_is_locked(self) -> None:
        await self._lock_loopback()
        # Locked: a loopback program guessing again is still refused ...
        self.assertEqual(
            await self._first_reply({"type": "auth", "password": "correct-horse"}, None),
            "auth_locked",
        )
        # ... but the browser page served at 127.0.0.1 needs no password and gets in.
        self.assertEqual(
            await self._first_reply({"type": "auth"}, f"http://127.0.0.1:{self.port}"),
            "auth_ok",
        )
        # A page relayed from a public origin is still locked out.
        self.assertEqual(
            await self._first_reply({"type": "auth"}, "https://tunnel.example.com"),
            "auth_locked",
        )


if __name__ == "__main__":
    unittest.main()
