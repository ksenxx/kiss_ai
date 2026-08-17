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
import socket
import ssl
import tempfile
import time
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import ClientConnection, connect

import kiss.agents.sorcar.persistence as th
import kiss.core.vscode_config as vc
from kiss.server.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
)


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    """Point the sorcar persistence layer at a per-test directory."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    """Undo :func:`_redirect_persistence`."""
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _find_free_port() -> int:
    """Return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _no_verify_ssl() -> ssl.SSLContext:
    """Return an SSL client context that skips certificate verification."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class TestMalformedFieldConnectionRobustness(IsolatedAsyncioTestCase):
    """A malformed client field must degrade gracefully, never drop the client."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt6-rejfail-")
        self.saved = _redirect_persistence(self.tmpdir)
        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)

        self.port = _find_free_port()
        self.url = f"wss://127.0.0.1:{self.port}/ws"
        self.ctx = _no_verify_ssl()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=Path(self.tmpdir) / "sorcar.sock",
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
        _restore_persistence(self.saved)
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
