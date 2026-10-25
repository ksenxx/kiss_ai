# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A private (``local_only``) daemon admits no loopback browser, locked or not.

Audit 2026-10-25 (server-main): ``ServerApi.authenticate`` refuses
every non-token peer of a private daemon before any other admission
in its handshake loop, but its rate-limited branch (a source IP
still locked out after too many failed logins) called
``_admit_loopback_browser`` without that check, so a browser page at
``http://127.0.0.1`` would have been admitted to the private daemon
exactly while its IP was locked.  The two branches now share the
check (it lives in ``_admit_loopback_browser``).

A private daemon records no auth failures of its own (every non-token
frame is refused before the failure accounting), so the lock-out is
driven through the real per-IP counter (``_record_auth_failure``, the
method the public handshake calls) on the real :class:`RemoteAccessServer`
over a real TLS WebSocket; nothing is replaced.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.server.web_server import _AUTH_FAIL_MAX, RemoteAccessServer
from kiss.tests.server.conftest import no_verify_ssl, redirect_persistence, restore_persistence


class TestPrivateDaemonLockout(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-private-lockout-"))
        self._saved = redirect_persistence(self.tmp)
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            local_endpoint_file=self.tmp / "sorcar-local.json",
            work_dir=str(self.tmp),
        )
        await self.server.start_private_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        restore_persistence(self._saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def _first_reply(self, frame: dict[str, object], origin: str | None) -> str:
        """Connect from loopback, send *frame*, return the reply ``type`` (or ``closed``)."""
        url = f"wss://127.0.0.1:{self.server.port}/ws"
        async with connect(url, ssl=no_verify_ssl(), origin=origin) as ws:  # type: ignore[arg-type]
            await ws.send(json.dumps(frame))
            try:
                resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                return str(resp.get("type"))
            except Exception:
                return "closed"

    async def test_locked_private_daemon_still_refuses_the_loopback_browser(self) -> None:
        origin = f"http://127.0.0.1:{self.server.port}"
        # Unlocked: the browser page is refused (a private daemon admits
        # tokens only) and the local token gets in.
        self.assertEqual(await self._first_reply({"type": "auth"}, origin), "error")
        self.assertEqual(
            await self._first_reply({"type": "auth", "token": self.server.local_token}, origin),
            "auth_ok",
        )
        for _ in range(_AUTH_FAIL_MAX):
            self.server._record_auth_failure("127.0.0.1")
        self.assertGreater(self.server._auth_lock_remaining("127.0.0.1"), 0.0)
        # Locked: the browser page is STILL refused ...
        self.assertEqual(await self._first_reply({"type": "auth"}, origin), "auth_locked")
        # ... and the local token still gets in.
        self.assertEqual(
            await self._first_reply({"type": "auth", "token": self.server.local_token}, origin),
            "auth_ok",
        )


if __name__ == "__main__":
    unittest.main()
