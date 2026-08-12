# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the server-canonical shared tab registry.

Every connected client — VS Code webviews and remote web apps — must
show the SAME set of tabs with the same titles, order and chat
bindings.  The daemon owns the canonical tab registry, mutates it on
``openTab`` / ``closeTab`` / ``run`` / ``resumeSession`` / ``ready``
and broadcasts a full ``tabs_state`` snapshot to ALL clients after
every mutation.  The registry persists across daemon restarts.

These tests drive a real :class:`RemoteAccessServer` over real
``wss://`` connections (no mocks) and pin that contract:

* a tab opened by one client appears on every other client;
* a tab closed by one client disappears everywhere;
* a chat resumed in a tab binds + titles that tab for every client,
  and the replayed transcript reaches every client identically;
* a client that connects late receives the full canonical snapshot
  and the transcripts of every bound tab;
* the registry survives a daemon restart;
* a legacy client's ``restoredTabs`` are adopted only into an EMPTY
  registry (one-time migration);
* concurrent mutations from two clients never corrupt the registry.
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


def _tab_ids(snapshot: dict[str, Any]) -> list[str]:
    """Return the ordered tab ids of a ``tabs_state`` event."""
    return [t.get("tabId", "") for t in snapshot.get("tabs", [])]


def _tab_entry(snapshot: dict[str, Any], tab_id: str) -> dict[str, Any]:
    """Return the entry for *tab_id* in a ``tabs_state`` event."""
    for t in snapshot.get("tabs", []):
        if t.get("tabId") == tab_id:
            return dict(t)
    raise AssertionError(f"tab {tab_id!r} not in snapshot {snapshot!r}")


class TabMirroringBase(IsolatedAsyncioTestCase):
    """Shared harness: one real daemon, N real WSS clients."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-tab-mirror-")
        self.saved = _redirect_persistence(self.tmpdir)
        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        self.certfile = Path(self.tmpdir) / "cert.pem"
        self.keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(self.certfile, self.keyfile)
        self.ctx = _no_verify_ssl()
        self._sockets: list[ClientConnection] = []
        self.server: RemoteAccessServer | None = None
        await self._start_server()

    async def _start_server(self) -> None:
        """Start a fresh daemon over the test's persistence dir."""
        self.port = _find_free_port()
        self.url = f"wss://127.0.0.1:{self.port}/ws"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(self.certfile),
            keyfile=str(self.keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=Path(self.tmpdir) / "sorcar.sock",
        )
        await self.server.start_async()

    async def _stop_server(self) -> None:
        """Stop the daemon and drop all client sockets."""
        for ws in self._sockets:
            try:
                await ws.close()
            except Exception:
                pass
        self._sockets.clear()
        if self.server is not None:
            await self.server.stop_async()
            self.server = None

    async def asyncTearDown(self) -> None:
        await self._stop_server()
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
        ws = await connect(self.url, ssl=self.ctx, max_size=64 * 1024 * 1024)
        self._sockets.append(ws)
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        self.assertEqual(resp["type"], "auth_ok")
        return ws

    async def _send(self, ws: ClientConnection, cmd: dict[str, Any]) -> None:
        """Send one JSON command."""
        await ws.send(json.dumps(cmd))

    async def _ready(
        self,
        ws: ClientConnection,
        restored: list[dict[str, str]] | None = None,
    ) -> None:
        """Send a ``ready`` handshake for *ws*."""
        await self._send(ws, {
            "type": "ready", "tabId": "", "restoredTabs": restored or [],
        })

    async def _wait_for_event(
        self,
        ws: ClientConnection,
        wanted: str,
        timeout: float = 8.0,
        pred: Any = None,
    ) -> dict[str, Any] | None:
        """Return the first *wanted* event (matching *pred*) or ``None``."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError:
                return None
            ev = json.loads(raw)
            if ev.get("type", "") == wanted and (pred is None or pred(ev)):
                return dict(ev)
        return None

    async def _wait_for_snapshot_with(
        self,
        ws: ClientConnection,
        present: set[str] | None = None,
        absent: set[str] | None = None,
        timeout: float = 8.0,
    ) -> dict[str, Any] | None:
        """Return the first ``tabs_state`` whose tab set matches."""
        def _pred(ev: dict[str, Any]) -> bool:
            ids = set(_tab_ids(ev))
            if present is not None and not present <= ids:
                return False
            if absent is not None and ids & absent:
                return False
            return True

        return await self._wait_for_event(
            ws, "tabs_state", timeout=timeout, pred=_pred,
        )

    def _seed_chat(
        self, task: str, events: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """Record a finished chat in the DB; return (task_id, chat_id)."""
        task_id, chat_id = th._add_task(task)
        for ev in events:
            th._append_chat_event(ev, task_id=task_id)
        th._save_task_result(task_id, "success: seeded")
        return task_id, chat_id


class TestTabMirroring(TabMirroringBase):
    """Two clients must always converge on the same tab set."""

    async def test_open_tab_mirrors_to_other_client(self) -> None:
        """(a) A tab opened on client A appears on client B."""
        ws_a = await self._connect_ok()
        ws_b = await self._connect_ok()
        await self._ready(ws_a)
        await self._ready(ws_b)

        await self._send(ws_a, {
            "type": "openTab", "tabId": "tab-a1", "title": "hello tab",
        })
        snap_b = await self._wait_for_snapshot_with(ws_b, present={"tab-a1"})
        self.assertIsNotNone(
            snap_b,
            "client B never received a tabs_state snapshot containing the "
            "tab client A opened — the tab sets diverge",
        )
        assert snap_b is not None
        self.assertEqual(_tab_entry(snap_b, "tab-a1")["title"], "hello tab")
        snap_a = await self._wait_for_snapshot_with(ws_a, present={"tab-a1"})
        self.assertIsNotNone(snap_a, "the opening client must also converge")

    async def test_close_tab_mirrors_to_other_client(self) -> None:
        """(b) A tab closed on client B disappears on client A."""
        ws_a = await self._connect_ok()
        ws_b = await self._connect_ok()
        await self._ready(ws_a)
        await self._ready(ws_b)

        await self._send(ws_a, {
            "type": "openTab", "tabId": "tab-x", "title": "doomed",
        })
        self.assertIsNotNone(
            await self._wait_for_snapshot_with(ws_b, present={"tab-x"}),
        )
        await self._send(ws_b, {"type": "closeTab", "tabId": "tab-x"})
        snap_a = await self._wait_for_snapshot_with(ws_a, absent={"tab-x"})
        self.assertIsNotNone(
            snap_a,
            "client A never learned that client B closed the tab",
        )

    async def test_resume_binds_titles_and_mirrors_transcript(self) -> None:
        """(c) Chat resumed on A: B sees binding, title and transcript."""
        seeded_events = [
            {"type": "prompt", "text": "seeded prompt"},
            {"type": "text", "text": "seeded answer"},
        ]
        _task_id, chat_id = self._seed_chat("Seeded task", seeded_events)

        ws_a = await self._connect_ok()
        ws_b = await self._connect_ok()
        await self._ready(ws_a)
        await self._ready(ws_b)

        await self._send(ws_a, {
            "type": "openTab", "tabId": "tab-c", "title": "new chat",
        })
        self.assertIsNotNone(
            await self._wait_for_snapshot_with(ws_b, present={"tab-c"}),
        )
        await self._send(ws_a, {
            "type": "resumeSession", "chatId": chat_id, "tabId": "tab-c",
        })

        def _bound(ev: dict[str, Any]) -> bool:
            try:
                entry = _tab_entry(ev, "tab-c")
            except AssertionError:
                return False
            return entry.get("chatId") == chat_id

        snap_b = await self._wait_for_event(
            ws_b, "tabs_state", pred=_bound,
        )
        self.assertIsNotNone(
            snap_b,
            "client B never saw tab-c bound to the resumed chat",
        )
        assert snap_b is not None
        self.assertEqual(
            _tab_entry(snap_b, "tab-c")["title"], "Seeded task",
        )

        replay_b = await self._wait_for_event(
            ws_b, "task_events", pred=lambda ev: ev.get("tabId") == "tab-c",
        )
        self.assertIsNotNone(
            replay_b,
            "client B never received the replayed transcript for the "
            "shared tab — tab contents diverge",
        )
        assert replay_b is not None
        got_types = [e.get("type") for e in replay_b.get("events", [])]
        self.assertIn("prompt", got_types)

    async def test_run_registers_titles_and_binds_tab(self) -> None:
        """A ``run`` mutation registers + titles + binds the tab for all."""
        ws_a = await self._connect_ok()
        ws_b = await self._connect_ok()
        await self._ready(ws_a)
        await self._ready(ws_b)

        await self._send(ws_a, {
            "type": "run",
            "prompt": "Do the mirrored thing",
            "model": "definitely-not-a-real-model",
            "workDir": self.tmpdir,
            "tabId": "tab-run",
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": False,
        })

        def _bound(ev: dict[str, Any]) -> bool:
            try:
                entry = _tab_entry(ev, "tab-run")
            except AssertionError:
                return False
            return bool(entry.get("chatId"))

        snap_b = await self._wait_for_event(ws_b, "tabs_state", pred=_bound)
        self.assertIsNotNone(
            snap_b,
            "client B never saw the tab client A started a run in",
        )
        assert snap_b is not None
        self.assertEqual(
            _tab_entry(snap_b, "tab-run")["title"],
            "Do the mirrored thing",
        )

    async def test_late_client_receives_full_state_and_transcripts(
        self,
    ) -> None:
        """(d) A third client connecting late reconstructs everything."""
        seeded_events = [
            {"type": "prompt", "text": "late prompt"},
            {"type": "text", "text": "late answer"},
        ]
        _task_id, chat_id = self._seed_chat("Late task", seeded_events)

        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._send(ws_a, {
            "type": "openTab", "tabId": "tab-l1", "title": "first",
        })
        await self._send(ws_a, {
            "type": "openTab", "tabId": "tab-l2", "title": "second",
        })
        await self._send(ws_a, {
            "type": "resumeSession", "chatId": chat_id, "tabId": "tab-l2",
        })
        self.assertIsNotNone(await self._wait_for_event(
            ws_a, "task_events",
            pred=lambda ev: ev.get("tabId") == "tab-l2",
        ))

        ws_c = await self._connect_ok()
        await self._ready(ws_c)
        snap_c = await self._wait_for_snapshot_with(
            ws_c, present={"tab-l1", "tab-l2"},
        )
        self.assertIsNotNone(
            snap_c,
            "the late client never received the canonical tab snapshot",
        )
        assert snap_c is not None
        self.assertEqual(_tab_entry(snap_c, "tab-l2")["chatId"], chat_id)
        replay_c = await self._wait_for_event(
            ws_c, "task_events",
            pred=lambda ev: ev.get("tabId") == "tab-l2",
        )
        self.assertIsNotNone(
            replay_c,
            "the late client never received the bound tab's transcript",
        )
        assert replay_c is not None
        texts = [e.get("text") for e in replay_c.get("events", [])]
        self.assertIn("late prompt", texts)

    async def test_registry_survives_daemon_restart(self) -> None:
        """(e) Tabs persist under KISS_HOME across a daemon restart."""
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._send(ws_a, {
            "type": "openTab", "tabId": "tab-p1", "title": "persist me",
        })
        self.assertIsNotNone(
            await self._wait_for_snapshot_with(ws_a, present={"tab-p1"}),
        )

        await self._stop_server()
        await self._start_server()

        ws_b = await self._connect_ok()
        await self._ready(ws_b)
        snap = await self._wait_for_snapshot_with(ws_b, present={"tab-p1"})
        self.assertIsNotNone(
            snap,
            "the restarted daemon lost the canonical tab registry",
        )
        assert snap is not None
        self.assertEqual(_tab_entry(snap, "tab-p1")["title"], "persist me")

    async def test_legacy_restored_tabs_merge_only_into_empty_registry(
        self,
    ) -> None:
        """(f) restoredTabs are adopted once, then the registry wins."""
        _task_id, chat_id = self._seed_chat(
            "Legacy chat", [{"type": "prompt", "text": "legacy"}],
        )
        ws_a = await self._connect_ok()
        await self._send(ws_a, {
            "type": "ready", "tabId": "legacy-1",
            "restoredTabs": [
                {"tabId": "legacy-1", "chatId": chat_id,
                 "title": "Legacy chat"},
            ],
        })
        snap_a = await self._wait_for_snapshot_with(
            ws_a, present={"legacy-1"},
        )
        self.assertIsNotNone(
            snap_a,
            "an empty registry must adopt a legacy client's restoredTabs",
        )

        ws_b = await self._connect_ok()
        await self._send(ws_b, {
            "type": "ready", "tabId": "legacy-2",
            "restoredTabs": [
                {"tabId": "legacy-2", "chatId": chat_id,
                 "title": "Other legacy"},
            ],
        })
        snap_b = await self._wait_for_event(ws_b, "tabs_state")
        self.assertIsNotNone(snap_b)
        assert snap_b is not None
        self.assertIn("legacy-1", _tab_ids(snap_b))
        self.assertNotIn(
            "legacy-2", _tab_ids(snap_b),
            "a non-empty registry must NOT merge a second client's "
            "legacy tabs — the canonical registry wins",
        )

    async def test_concurrent_mutations_do_not_corrupt_registry(self) -> None:
        """(g) Interleaved mutations from two clients stay consistent."""
        ws_a = await self._connect_ok()
        ws_b = await self._connect_ok()
        await self._ready(ws_a)
        await self._ready(ws_b)

        async def _mutate(ws: ClientConnection, prefix: str) -> None:
            for i in range(20):
                await self._send(ws, {
                    "type": "openTab",
                    "tabId": f"{prefix}-{i}",
                    "title": f"{prefix} {i}",
                })
            for i in range(0, 20, 2):
                await self._send(ws, {
                    "type": "closeTab", "tabId": f"{prefix}-{i}",
                })

        await asyncio.gather(_mutate(ws_a, "ca"), _mutate(ws_b, "cb"))

        expected_present = {
            f"{p}-{i}" for p in ("ca", "cb") for i in range(1, 20, 2)
        }
        expected_absent = {
            f"{p}-{i}" for p in ("ca", "cb") for i in range(0, 20, 2)
        }
        ws_c = await self._connect_ok()
        await self._ready(ws_c)
        snap = await self._wait_for_snapshot_with(
            ws_c, present=expected_present, absent=expected_absent,
            timeout=12.0,
        )
        self.assertIsNotNone(
            snap,
            "concurrent open/close mutations corrupted the registry",
        )
        assert snap is not None
        ids = _tab_ids(snap)
        self.assertEqual(
            len(ids), len(set(ids)), f"duplicate tab ids in registry: {ids}",
        )


if __name__ == "__main__":
    unittest.main()
