# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 6: a malformed client field must not kill the connection.

The websocket handler's message loop wraps dispatch in one ``except
Exception``; before the fix, ANY malformed client field that raises
(e.g. an unhashable ``tabId``) EXITED the loop and tore down the whole
authenticated WebSocket connection over a single bad message.

Historically this file also reproduced the same connection-teardown
bug through failing hunk-reject writes in the interactive diff/merge
review; that review workflow (``mergeAction``, ``_merge_states``,
``web_merge.py``) has been removed from the server, so only the
malformed-field reproduction remains.

These tests use a real ``RemoteAccessServer`` with real wss://
connections — no mocks.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import ClientConnection, connect

import kiss.agents.sorcar.persistence as th
import kiss.core.user_config as vc
from kiss.server.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
)
from kiss.tests.server.conftest import (
    find_free_port,
    no_verify_ssl,
    redirect_persistence,
    restore_persistence,
)


class TestMalformedFieldConnectionRobustness(IsolatedAsyncioTestCase):
    """A malformed client field must degrade gracefully, never drop the client."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt6-rejfail-")
        self.saved = redirect_persistence(self.tmpdir)
        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)

        self.port = find_free_port()
        self.url = f"wss://127.0.0.1:{self.port}/ws"
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
            try:
                await ws.close()
            except Exception:
                pass
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        restore_persistence(self.saved)
        vc.CONFIG_DIR = self._orig_cfg_dir
        vc.CONFIG_PATH = self._orig_cfg_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect_ok(self) -> ClientConnection:
        """Open + successfully authenticate one WSS connection."""
        ws = await connect(self.url, ssl=self.ctx)
        self._sockets.append(ws)
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        self.assertEqual(resp["type"], "auth_ok")
        return ws

    async def _assert_connection_alive(self, ws: ClientConnection) -> None:
        """Probe liveness: an activeTasksQuery must get a direct reply."""
        await ws.send(json.dumps({"type": "activeTasksQuery"}))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            ev = json.loads(raw)
            if ev.get("type") == "activeTasksResponse":
                return
        self.fail("no activeTasksResponse received")  # pragma: no cover

    async def test_unhashable_tab_id_does_not_kill_connection(self) -> None:
        """A malformed client field that raises (unhashable tabId) must be
        contained per-message, not tear down the authenticated session."""
        ws = await self._connect_ok()
        await ws.send(json.dumps({
            "type": "ready", "tabId": {"x": 1}, "restoredTabs": [],
        }))
        await asyncio.sleep(0.3)
        try:
            await self._assert_connection_alive(ws)
        except Exception as exc:  # noqa: BLE001
            self.fail(
                "BUG: a malformed tabId killed the whole WebSocket "
                f"connection ({type(exc).__name__}: {exc})",
            )


if __name__ == "__main__":
    unittest.main()
