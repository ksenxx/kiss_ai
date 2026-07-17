# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt round 2: ``restoredTabs`` escape the deferred-close protocol.

``RemoteAccessServer._ws_handler``'s ``finally`` block documents the
deferred-disposal contract: it schedules a deferred ``closeTab`` "for
every tab id this connection touched", so a browser that goes away
for good cannot leak backend ``_RunningAgentState`` entries.  Tab ids
are collected per-connection in ``tabs_seen`` by
``_dispatch_client_command`` — but only from each command's own
``tabId`` field.

``_handle_ready`` *also* re-claims every entry of the ``ready``
command's ``restoredTabs`` list: it cancels the tab's pending
deferred close and dispatches a ``resumeSession`` for it.  Those tab
ids are therefore very much "touched" by the connection, yet they are
never recorded in ``tabs_seen``.  Consequence: after a page reload
(connection A drops → deferred close armed; connection B reconnects
and re-claims the tab via ``restoredTabs`` → pending close cancelled),
connection B's own disconnect never re-arms the deferred close for
the restored tab.  The backend state for that tab is then leaked
forever — no ``closeTab`` is ever issued for it again.

The test below drives a real :class:`RemoteAccessServer` over real
``wss://`` connections (no mocks) and asserts that, after the
reload-style reconnect sequence, dropping the second connection arms
a deferred close for the restored tab exactly like it does for the
tab the connection claimed via ``ready.tabId``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import ssl
import tempfile
import time
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


class TestRestoredTabDeferredClose(IsolatedAsyncioTestCase):
    """Restored tabs must re-enter the deferred-close protocol."""

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

    def _pending_tabs(self) -> set[str]:
        """Snapshot the server's armed deferred-close tab ids."""
        with self.server._pending_tab_closes_lock:
            return set(self.server._pending_tab_closes)

    async def _wait_pending(
        self, tab_id: str, present: bool, timeout: float = 5.0,
    ) -> bool:
        """Poll until *tab_id*'s pending-close presence matches *present*."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (tab_id in self._pending_tabs()) == present:
                return True
            await asyncio.sleep(0.05)
        return False

    async def test_restored_tab_close_rearmed_on_disconnect(self) -> None:
        """A tab re-claimed via ``restoredTabs`` must be re-armed for close.

        Sequence (a page reload): connection A claims ``tab-x`` and
        drops (deferred close armed); connection B reconnects with
        ``ready.restoredTabs=[tab-x]`` (pending close cancelled) and
        then drops too.  B touched ``tab-x``, so B's disconnect must
        re-arm the deferred close — otherwise the backend state for
        ``tab-x`` is never disposed.
        """
        ws1 = await self._connect_ok()
        await ws1.send(json.dumps({"type": "ready", "tabId": "tab-x"}))
        # Wait until the server has processed the command (its tabId
        # lands in the connection's tabs_seen) before dropping.
        await asyncio.sleep(0.5)
        await ws1.close()
        self.assertTrue(
            await self._wait_pending("tab-x", present=True),
            "sanity: conn A's disconnect should arm a deferred close "
            "for the tab it claimed via ready.tabId",
        )

        ws2 = await self._connect_ok()
        await ws2.send(json.dumps({
            "type": "ready",
            "tabId": "tab-other",
            "restoredTabs": [{"tabId": "tab-x", "chatId": ""}],
        }))
        self.assertTrue(
            await self._wait_pending("tab-x", present=False),
            "sanity: conn B's ready(restoredTabs=[tab-x]) should cancel "
            "the pending deferred close for tab-x",
        )
        await ws2.close()

        self.assertTrue(
            await self._wait_pending("tab-other", present=True),
            "sanity: conn B's disconnect should arm a deferred close "
            "for the tab it claimed via ready.tabId",
        )
        self.assertIn(
            "tab-x",
            self._pending_tabs(),
            "BUG: conn B re-claimed tab-x via ready.restoredTabs but its "
            "disconnect did not re-arm the deferred closeTab for tab-x — "
            "the restored tab's backend state is leaked forever",
        )
