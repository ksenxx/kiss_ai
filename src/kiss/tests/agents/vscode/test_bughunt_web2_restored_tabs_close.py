# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt round 2 (rewritten): restored tabs persist across reloads.

Tabs are global server state shared by all (mirror-copy) clients: no
connection owns a tab, so neither the disconnect of the connection
that claimed a tab via ``ready.tabId`` nor the disconnect of a
later connection that re-claimed it via ``ready.restoredTabs`` may
tear the tab's backend state down.  Teardown happens only through an
explicit ``closeTab`` command.

The test below drives a real :class:`RemoteAccessServer` over real
``wss://`` connections (no mocks) and asserts that, after a
reload-style sequence (connection A claims the tab and drops;
connection B restores it and drops too), the backend
:class:`AgentState` for both the restored tab and the tab claimed via
``ready.tabId`` simply persist.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import ssl
import tempfile
from pathlib import Path
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


class TestRestoredTabPersistence(IsolatedAsyncioTestCase):
    """Restored tabs persist across disconnects like any other tab."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt-web2-")
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

    async def test_restored_tab_persists_across_disconnects(self) -> None:
        """A tab claimed directly or via ``restoredTabs`` survives drops.

        Sequence (a page reload): connection A claims ``tab-x`` and
        drops; connection B reconnects with
        ``ready.restoredTabs=[tab-x]`` (plus its own ``tab-other``)
        and then drops too.  Neither disconnect may tear down either
        tab's backend state: tabs are global server state and persist
        until an explicit ``closeTab``.
        """
        for tab_id in ("tab-x", "tab-other"):
            agent_state.register(
                agent_state.AgentState(
                    f"task-{tab_id}", tab_id=tab_id, server_owned=True,
                ),
            )

        ws1 = await self._connect_ok()
        await ws1.send(json.dumps({"type": "ready", "tabId": "tab-x"}))
        await asyncio.sleep(0.5)
        await ws1.close()
        await asyncio.sleep(1.0)
        state_x = agent_state.find_by_tab("tab-x")
        self.assertIsNotNone(
            state_x,
            "conn A's disconnect must not tear down the tab it claimed "
            "via ready.tabId",
        )

        ws2 = await self._connect_ok()
        await ws2.send(json.dumps({
            "type": "ready",
            "tabId": "tab-other",
            "restoredTabs": [{"tabId": "tab-x", "chatId": ""}],
        }))
        await asyncio.sleep(0.5)
        await ws2.close()
        await asyncio.sleep(1.0)

        for tab_id in ("tab-x", "tab-other"):
            state = agent_state.find_by_tab(tab_id)
            self.assertIsNotNone(
                state,
                f"conn B's disconnect tore down {tab_id!r} — tabs are "
                "global server state and must persist until an explicit "
                "closeTab",
            )
            assert state is not None
            self.assertFalse(
                state.frontend_closed,
                f"{tab_id!r} must not be flagged frontend_closed by a "
                "mere disconnect",
            )
