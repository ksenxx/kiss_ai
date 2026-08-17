# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for server-side tab persistence across disconnects.

All frontend clients are mirror copies of each other: every client
shows the same tabs and their contents, so a tab is GLOBAL server
state that no single connection owns.  A client disconnect (browser
window closed, laptop lid shut, network blip) therefore must NOT tear
down any tab's backend state — the tab persists server-side until an
explicit ``closeTab`` command arrives from some client.

The tests pin this contract end-to-end against a real running
:class:`RemoteAccessServer` over real ``wss://`` connections (no
mocks):

* a connection that touches a tab and then drops leaves the tab's
  ``AgentState`` fully intact, and dispatches no ``closeTab``;
* an explicit ``closeTab`` command still tears the tab down.
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
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import ClientConnection, connect

import kiss.agents.sorcar.persistence as th
import kiss.core.vscode_config as vc
from kiss.server import agent_state
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


class TestTabPersistence(IsolatedAsyncioTestCase):
    """Tabs are global server state: only ``closeTab`` disposes them."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-tab-persistence-")
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
        # Observe every backend command dispatch (and still forward
        # it) so the tests can assert that no ``closeTab`` is ever
        # issued on behalf of a mere disconnect.
        self.dispatched: list[dict[str, Any]] = []
        orig_run_cmd = self.server._run_cmd

        async def recording_run_cmd(cmd: dict[str, Any]) -> None:
            self.dispatched.append(dict(cmd))
            await orig_run_cmd(cmd)

        self.server._run_cmd = recording_run_cmd  # type: ignore[assignment]
        await self.server.start_async()
        self._sockets: list[ClientConnection] = []

    async def asyncTearDown(self) -> None:
        for ws in self._sockets:
            try:
                await ws.close()
            except Exception:
                pass
        await self.server.stop_async()
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
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

    async def _wait_for_event(
        self, ws: ClientConnection, wanted: str, timeout: float = 6.0,
    ) -> dict | None:
        """Return the first event of type *wanted* within *timeout*."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError:
                return None
            ev = json.loads(raw)
            if ev.get("type", "") == wanted:
                return dict(ev)
        return None

    def _close_tabs_dispatched(self) -> list[str]:
        """Return the tab ids of every dispatched ``closeTab`` command."""
        return [
            str(cmd.get("tabId", ""))
            for cmd in list(self.dispatched)
            if cmd.get("type") == "closeTab"
        ]

    async def test_disconnect_does_not_tear_down_idle_tab(self) -> None:
        """A dropped connection leaves the touched tab's state intact."""
        tab_id = "tab-persist-idle"
        agent_state.register(
            agent_state.AgentState(
                f"task-{tab_id}", tab_id=tab_id, server_owned=True,
            ),
        )

        ws = await self._connect_ok()
        await ws.send(json.dumps({
            "type": "ready", "tabId": tab_id, "restoredTabs": [],
        }))
        self.assertIsNotNone(await self._wait_for_event(ws, "focusInput"))
        await ws.close()

        # Give any (wrongly) armed teardown ample time to fire.
        await asyncio.sleep(1.0)
        state = agent_state.find_by_tab(tab_id)
        self.assertIsNotNone(
            state,
            "a client disconnect must not tear down the shared tab — "
            "tabs are global server state under the mirror-clients "
            "model and persist until an explicit closeTab",
        )
        assert state is not None
        self.assertFalse(state.frontend_closed)
        self.assertEqual(
            self._close_tabs_dispatched(), [],
            "no closeTab may be dispatched on behalf of a disconnect",
        )

    async def test_explicit_close_tab_tears_down_idle_tab(self) -> None:
        """An explicit ``closeTab`` command still disposes the tab."""
        tab_id = "tab-explicit-close-idle"
        agent_state.register(
            agent_state.AgentState(
                f"task-{tab_id}", tab_id=tab_id, server_owned=True,
            ),
        )

        ws = await self._connect_ok()
        await ws.send(json.dumps({
            "type": "ready", "tabId": tab_id, "restoredTabs": [],
        }))
        self.assertIsNotNone(await self._wait_for_event(ws, "focusInput"))

        await ws.send(json.dumps({"type": "closeTab", "tabId": tab_id}))

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if agent_state.find_by_tab(tab_id) is None:
                break
            await asyncio.sleep(0.05)
        self.assertIsNone(
            agent_state.find_by_tab(tab_id),
            "an explicit closeTab must still dispose the idle tab's "
            "backend state",
        )

if __name__ == "__main__":
    unittest.main()
