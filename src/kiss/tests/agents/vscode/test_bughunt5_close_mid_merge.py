# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt iteration 5: deferred tab close mid-merge-review leaks the tab.

When a browser disconnects, :meth:`RemoteAccessServer._ws_handler`
arms a deferred ``closeTab`` for every tab the connection touched.
If the grace timer fires while a merge review is STILL IN FLIGHT,
``_fire_pending_tab_close`` silently pops the server-side
``_WebMergeState`` and dispatches only ``closeTab``.  The backend
``_close_tab`` sees ``is_merging=True`` (a busy lifecycle flag), so it
merely flips ``frontend_closed=True`` and defers disposal until "the
merge ends" — but the merge can now NEVER end: the web merge state is
gone, every future ``mergeAction`` returns early, ``all-done`` is
never dispatched and ``_finish_merge`` never runs.

Result: the backend tab is stuck ``is_merging=True`` forever (agent
and ``_RunningAgentState`` leak, the per-tab merge artifact directory
is never cleaned, a pending worktree is never presented/released, and
the stuck flag can block other tabs' merge actions via busy guards).

This test drives a real :class:`RemoteAccessServer` over a real
``wss://`` connection (no mocks): it opens a merge review through the
real backend ``_start_merge_session`` path (which sets ``is_merging``
and registers the web-side ``_WebMergeState`` via the broadcast hook),
claims the tab from a browser connection, drops the connection, and
asserts that once the (shortened) close grace elapses the backend tab
is fully disposed rather than leaked in ``is_merging`` limbo.
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
from functools import partial
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import ClientConnection, connect

import kiss.agents.sorcar.persistence as th
import kiss.agents.vscode.vscode_config as vc
import kiss.agents.vscode.web_server as web_server_module
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.web_server import (
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


def _build_pending_merge(work_dir: Path) -> Path:
    """Create real files plus a pending-merge.json with a 2-hunk file.

    Returns:
        Path to the written ``pending-merge.json`` manifest.
    """
    current = work_dir / "f.txt"
    base = work_dir / "f_base.txt"
    lines = "".join(f"line{i}\n" for i in range(20))
    current.write_text(lines.replace("line2\n", "edit2\n"))
    base.write_text(lines)
    manifest = {
        "branch": "HEAD",
        "files": [
            {
                "name": "f.txt",
                "base": str(base),
                "current": str(current),
                "target": str(current),
                "hunks": [
                    {"bs": 2, "bc": 1, "cs": 2, "cc": 1},
                    {"bs": 10, "bc": 1, "cs": 10, "cc": 1},
                ],
            }
        ],
    }
    merge_json = work_dir / "pending-merge.json"
    merge_json.write_text(json.dumps(manifest))
    return merge_json


class TestCloseTabMidMergeReview(IsolatedAsyncioTestCase):
    """The deferred tab close must end the merge lifecycle, not leak it."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt5-closemerge-")
        self.saved = _redirect_persistence(self.tmpdir)
        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"
        # Shorten the deferred-close grace window so the test does not
        # wait 10 real seconds for the timer to fire.
        self._orig_grace = web_server_module._TAB_CLOSE_GRACE
        web_server_module._TAB_CLOSE_GRACE = 0.3

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
        self._tab_ids: list[str] = []

    async def asyncTearDown(self) -> None:
        for ws in self._sockets:
            try:
                await ws.close()
            except Exception:
                pass
        await self.server.stop_async()
        web_server_module._TAB_CLOSE_GRACE = self._orig_grace
        for tab_id in self._tab_ids:
            _RunningAgentState.running_agent_states.pop(tab_id, None)
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

    async def _start_review(self, tab_id: str) -> Path:
        """Open a real merge review on *tab_id* through the backend path."""
        self._tab_ids.append(tab_id)
        work = Path(self.tmpdir) / f"work-{tab_id}"
        work.mkdir(exist_ok=True)
        merge_json = _build_pending_merge(work)
        vs = self.server._vscode_server
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, vs._get_tab, tab_id)
        started = await loop.run_in_executor(
            None,
            partial(
                vs._start_merge_session,
                str(merge_json),
                tab_id=tab_id,
                work_dir=str(work),
            ),
        )
        self.assertTrue(started, "merge session failed to start")
        tab = _RunningAgentState.running_agent_states[tab_id]
        self.assertTrue(tab.is_merging)
        with self.server._merge_states_lock:
            self.assertIn(tab_id, self.server._merge_states)
        return work

    async def test_deferred_close_mid_review_ends_merge_and_disposes(self) -> None:
        """Dropping the browser mid-review must not leak an is_merging tab."""
        tab_id = "tab-close-mid-merge"
        await self._start_review(tab_id)

        # A browser claims the tab (receives the replayed review), then
        # disconnects without resolving the remaining hunks.
        ws = await self._connect_ok()
        await ws.send(json.dumps({
            "type": "ready", "tabId": tab_id, "restoredTabs": [],
        }))
        replay = await self._wait_for_event(ws, "merge_started")
        self.assertIsNotNone(replay, "in-flight review was not replayed")
        await ws.close()

        # After the grace window the deferred close fires.  The merge
        # lifecycle must END (all-done -> _finish_merge) so the closed
        # tab is disposed instead of leaking in is_merging limbo.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if tab_id not in _RunningAgentState.running_agent_states:
                break
            await asyncio.sleep(0.1)
        leaked = _RunningAgentState.running_agent_states.get(tab_id)
        self.assertIsNone(
            leaked,
            "BUG: deferred close mid-merge-review leaked the backend tab "
            f"forever (is_merging={getattr(leaked, 'is_merging', None)}, "
            f"frontend_closed={getattr(leaked, 'frontend_closed', None)}) — "
            "the web merge state was popped so all-done/_finish_merge can "
            "never run and _dispose_if_closed never fires",
        )
        with self.server._merge_states_lock:
            self.assertNotIn(tab_id, self.server._merge_states)

    async def test_explicit_close_tab_mid_review_ends_merge(self) -> None:
        """An explicit web ``closeTab`` mid-review must end the merge too.

        The web UI lets the user close a chat tab at any time, sending
        ``closeTab`` over the still-open WebSocket.  That destroys the
        only UI that could ever finish the review, so the server must
        end the merge lifecycle (all-done) instead of leaving the
        backend tab in ``is_merging`` limbo until the whole connection
        eventually drops.
        """
        tab_id = "tab-explicit-close-merge"
        await self._start_review(tab_id)

        ws = await self._connect_ok()
        await ws.send(json.dumps({
            "type": "ready", "tabId": tab_id, "restoredTabs": [],
        }))
        self.assertIsNotNone(await self._wait_for_event(ws, "merge_started"))

        # The user closes the chat tab in the web UI; the connection
        # itself stays open (no deferred-close timer is involved).
        await ws.send(json.dumps({"type": "closeTab", "tabId": tab_id}))

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            with self.server._merge_states_lock:
                state_gone = tab_id not in self.server._merge_states
            if state_gone and tab_id not in _RunningAgentState.running_agent_states:
                break
            await asyncio.sleep(0.1)
        leaked = _RunningAgentState.running_agent_states.get(tab_id)
        self.assertIsNone(
            leaked,
            "BUG: explicit closeTab mid-merge-review left the backend tab "
            f"stuck (is_merging={getattr(leaked, 'is_merging', None)}, "
            f"frontend_closed={getattr(leaked, 'frontend_closed', None)})",
        )
        with self.server._merge_states_lock:
            self.assertNotIn(
                tab_id, self.server._merge_states,
                "BUG: web merge state leaked after explicit closeTab",
            )

    async def test_reconnect_within_grace_keeps_review_alive(self) -> None:
        """A reload within the grace window must keep tab + review intact."""
        tab_id = "tab-reload-mid-merge"
        await self._start_review(tab_id)

        ws = await self._connect_ok()
        await ws.send(json.dumps({
            "type": "ready", "tabId": tab_id, "restoredTabs": [],
        }))
        self.assertIsNotNone(await self._wait_for_event(ws, "merge_started"))
        await ws.close()

        # Reconnect immediately (well within the grace window).
        ws2 = await self._connect_ok()
        await ws2.send(json.dumps({
            "type": "ready", "tabId": tab_id, "restoredTabs": [],
        }))
        self.assertIsNotNone(
            await self._wait_for_event(ws2, "merge_started"),
            "review must be replayed to the reconnecting client",
        )
        # Give any (incorrectly surviving) close timer a chance to fire.
        await asyncio.sleep(1.0)
        tab = _RunningAgentState.running_agent_states.get(tab_id)
        self.assertIsNotNone(tab, "tab must survive a reload within grace")
        assert tab is not None
        self.assertTrue(tab.is_merging)
        with self.server._merge_states_lock:
            self.assertIn(tab_id, self.server._merge_states)


if __name__ == "__main__":
    unittest.main()
