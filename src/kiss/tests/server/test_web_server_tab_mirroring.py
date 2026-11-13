# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the server-canonical shared tab registry.

Every connected client — every open chat page, local or remote — must
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
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import ClientConnection, connect

import kiss.agents.sorcar.persistence as th
import kiss.core.user_config as vc
from kiss.server import agent_state
from kiss.server.tab_registry import OpenTabOutcome, TabRegistry
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import is_root, posix_only
from kiss.tests.local_ws import _generate_self_signed_cert, open_local_connection
from kiss.tests.server.conftest import (
    find_free_port,
    no_verify_ssl,
    redirect_persistence,
    restore_persistence,
)


def _tab_ids(snapshot: dict[str, Any]) -> list[str]:
    """Return the ordered tab ids of a ``tabs_state`` event."""
    return [t.get("tabId", "") for t in snapshot.get("tabs", [])]


def _tab_entry(snapshot: dict[str, Any], tab_id: str) -> dict[str, Any]:
    """Return the entry for *tab_id* in a ``tabs_state`` event."""
    for t in snapshot.get("tabs", []):
        if t.get("tabId") == tab_id:
            return dict(t)
    raise AssertionError(f"tab {tab_id!r} not in snapshot {snapshot!r}")


def _for_tab_t(event: dict[str, Any]) -> bool:
    """Whether *event* is addressed to the tab ``tab-t``."""
    return event.get("tabId") == "tab-t"


def _tab_t_has_no_pin(snapshot: dict[str, Any]) -> bool:
    """Whether a ``tabs_state`` lists ``tab-t`` with no task it is pinned to."""
    return any(
        t.get("tabId") == "tab-t" and not t.get("taskId") for t in snapshot.get("tabs", [])
    )


class TabMirroringBase(IsolatedAsyncioTestCase):
    """Shared harness: one real daemon, N real WSS clients."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-tab-mirror-")
        self.saved = redirect_persistence(self.tmpdir)
        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        self.certfile = Path(self.tmpdir) / "cert.pem"
        self.keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(self.certfile, self.keyfile)
        self.ctx = no_verify_ssl()
        self._sockets: list[ClientConnection] = []
        self.server: RemoteAccessServer | None = None
        await self._start_server()

    async def _start_server(self) -> None:
        """Start a fresh daemon over the test's persistence dir."""
        self.port = find_free_port()
        self.url = f"wss://127.0.0.1:{self.port}/ws"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(self.certfile),
            keyfile=str(self.keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
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
        th._close_thread_db()
        restore_persistence(self.saved)
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
        tab_id: str = "",
    ) -> None:
        """Send a ``ready`` handshake for *ws*.

        *tab_id* names the tab the client shows: the only one whose
        transcript the daemon replays at once (every other bound tab
        is answered with ``replay_deferred``).
        """
        await self._send(ws, {
            "type": "ready", "tabId": tab_id, "restoredTabs": restored or [],
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
        th._save_task_result("success: seeded", task_id)
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
            "isParallel": False,
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
        await self._ready(ws_c, tab_id="tab-l2")
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

    async def test_neighbour_task_answers_only_the_surface_that_asked(self) -> None:
        """(h) Reading back through a chat grows one surface's thread only.

        Two surfaces show tab T on the chat's newest task B.  Surface A
        asks for the task before B.  The answer names the task it is
        adjacent to and reaches A alone: spliced into the other
        surface's transcript too, it would sit where THAT thread ends,
        which is another task as soon as the two have read to different
        depths, and the tasks in between could never be loaded there.
        """
        task_a, chat_id = self._seed_chat(
            "Task A", [{"type": "prompt", "text": "prompt A"}],
        )
        task_b, _ = th._add_task("Task B", chat_id=chat_id)
        th._append_chat_event(
            {"type": "prompt", "text": "prompt B"}, task_id=task_b,
        )
        th._save_task_result("success: seeded", task_b)

        def _on_tab(ev: dict[str, Any]) -> bool:
            return bool(ev.get("tabId") == "tab-t")

        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._send(ws_a, {
            "type": "openTab", "tabId": "tab-t", "title": "new chat",
        })
        await self._send(ws_a, {
            "type": "resumeSession", "chatId": chat_id,
            "taskId": task_b, "tabId": "tab-t",
        })
        self.assertIsNotNone(
            await self._wait_for_event(ws_a, "task_events", pred=_on_tab),
        )
        ws_b = await self._connect_ok()
        await self._ready(ws_b, tab_id="tab-t")
        self.assertIsNotNone(
            await self._wait_for_event(ws_b, "task_events", pred=_on_tab),
        )

        await self._send(ws_a, {
            "type": "getAdjacentTask", "tabId": "tab-t",
            "taskId": task_b, "direction": "prev",
        })
        asked = await self._wait_for_event(ws_a, "adjacent_task_events")
        assert asked is not None, "the surface that asked got no answer"
        self.assertEqual(asked["task_id"], task_a)
        self.assertEqual(
            asked["adjacent_to"], task_b,
            "the answer must name the task it is adjacent to",
        )

        # B asks for its own neighbour: the first answer on its socket
        # is that one, not the task A scrolled to.
        await self._send(ws_b, {
            "type": "getAdjacentTask", "tabId": "tab-t",
            "taskId": task_b, "direction": "next",
        })
        own = await self._wait_for_event(ws_b, "adjacent_task_events")
        assert own is not None, "the second surface got no answer of its own"
        self.assertEqual(
            (own["direction"], own["task_id"], own["adjacent_to"]),
            ("next", None, task_b),
            "a neighbour one surface asked for reached another surface",
        )


class TestTabMirroringReviewFixes(TabMirroringBase):
    """Pins for the five defects found by the read-only review."""

    async def test_ready_replay_preserves_selected_task(self) -> None:
        """[1] A tab resumed to an OLD task must survive any ``ready``.

        Chat C has an older task A and a newer task B.  A client
        resumes tab T to task A; a later ``ready`` from ANY client
        replays the registry's tabs and must keep showing task A —
        not silently switch every client's tab T to the latest task B.
        """
        task_a, chat_id = self._seed_chat(
            "Old task A", [{"type": "prompt", "text": "old prompt A"}],
        )
        task_b, chat_b = th._add_task("New task B", chat_id=chat_id)
        th._append_chat_event(
            {"type": "prompt", "text": "new prompt B"}, task_id=task_b,
        )
        th._save_task_result("success: seeded", task_b)
        self.assertEqual(chat_b, chat_id)

        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._send(ws_a, {
            "type": "openTab", "tabId": "tab-t", "title": "new chat",
        })
        await self._send(ws_a, {
            "type": "resumeSession", "chatId": chat_id,
            "taskId": task_a, "tabId": "tab-t",
        })
        replay_a = await self._wait_for_event(
            ws_a, "task_events",
            pred=lambda ev: ev.get("tabId") == "tab-t",
        )
        self.assertIsNotNone(replay_a)
        assert replay_a is not None
        texts_a = [e.get("text") for e in replay_a.get("events", [])]
        self.assertIn("old prompt A", texts_a)

        # ANY client's ready must not lose the selected task.
        ws_c = await self._connect_ok()
        await self._ready(ws_c, tab_id="tab-t")
        replay_c = await self._wait_for_event(
            ws_c, "task_events",
            pred=lambda ev: ev.get("tabId") == "tab-t",
        )
        self.assertIsNotNone(
            replay_c, "ready never replayed the bound tab",
        )
        assert replay_c is not None
        texts_c = [e.get("text") for e in replay_c.get("events", [])]
        self.assertIn(
            "old prompt A", texts_c,
            "ready replay switched the tab away from the task the "
            "user selected",
        )
        self.assertNotIn(
            "new prompt B", texts_c,
            "ready replay silently switched the tab to the chat's "
            "latest task",
        )

    async def test_ready_replay_of_a_running_chat_shows_the_running_task(self) -> None:
        """A pin on an older task ends when the chat runs.

        Tab T was resumed to the finished task A of chat C, and C runs
        task B since.  A ``ready`` brings B's transcript with the
        running state: sent A's, the page shows a finished task as
        running and takes B's output into it.  That holds from B's
        admission on, before its history row exists.  The pin goes on
        every surface, so the tab is B's from then on; a click that
        names A all the same (from a history row that does not say
        yet that the chat runs) is answered with B's transcript too.
        """
        task_a, chat_id = self._seed_chat(
            "Old task A", [{"type": "prompt", "text": "old prompt A"}],
        )

        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._send(ws_a, {"type": "openTab", "tabId": "tab-t", "title": "new chat"})
        await self._send(ws_a, {
            "type": "resumeSession", "chatId": chat_id, "taskId": task_a, "tabId": "tab-t",
        })
        pinned = await self._wait_for_event(ws_a, "task_events", pred=_for_tab_t)
        assert pinned is not None
        self.assertEqual(pinned["task_id"], task_a, "an idle chat's pin is honoured")

        release = threading.Event()
        worker = threading.Thread(target=release.wait, daemon=True)
        worker.start()
        try:
            # What a run in tab T registers on admission; its history
            # row comes after its setup (``_cmd_run``).
            run_b = agent_state.AgentState(
                "provisional-key-of-b", tab_id="tab-t", chat_id=chat_id, server_owned=True,
                task_thread=worker, is_task_active=True,
            )
            run_b.last_user_prompt = "prompt B"
            agent_state.register(run_b)
            ws_c = await self._connect_ok()
            await self._ready(ws_c, tab_id="tab-t")
            frames: list[dict[str, Any]] = []
            while not frames or frames[-1].get("type") != "task_events":
                frame = json.loads(await asyncio.wait_for(ws_c.recv(), timeout=8))
                if _for_tab_t(frame):
                    frames.append(frame)
            unpinned = await self._wait_for_event(
                ws_a, "tabs_state", pred=_tab_t_has_no_pin,
            )
            task_b, _ = th._add_task("prompt B", chat_id=chat_id)
            agent_state.rekey(run_b, str(task_b))
            await self._send(ws_a, {
                "type": "resumeSession", "chatId": chat_id, "taskId": task_a, "tabId": "tab-t",
            })
            clicked = await self._wait_for_event(ws_a, "task_events", pred=_for_tab_t)
        finally:
            release.set()
            worker.join(timeout=5)
        self.assertEqual(frames[-1]["task"], "prompt B", "a running chat replayed a finished task")
        self.assertTrue(
            any(f.get("type") == "status" and f.get("running") for f in frames),
            "the tab is not told its chat runs",
        )
        self.assertIsNotNone(unpinned, "the other surface still has the tab pinned to task A")
        assert clicked is not None, "the click was not answered"
        self.assertEqual(clicked["task_id"], task_b, "a click replayed a finished task")

    async def test_subagent_close_broadcasts_to_all_clients(self) -> None:
        """[2] Closing a sub-agent tab must mirror to every client.

        Sub-agent tabs are not in the registry, so their close cannot
        mirror via ``tabs_state``; the daemon must broadcast a
        canonical ``closeSubagentTab`` event instead.
        """
        ws_a = await self._connect_ok()
        ws_b = await self._connect_ok()
        await self._ready(ws_a)
        await self._ready(ws_b)

        await self._send(ws_a, {
            "type": "closeTab", "tabId": "tab-parent__sub_9",
        })
        closed_b = await self._wait_for_event(
            ws_b, "closeSubagentTab",
            pred=lambda ev: ev.get("tab_id") == "tab-parent__sub_9",
        )
        self.assertIsNotNone(
            closed_b,
            "one client closing a sub-agent tab never reached the "
            "other clients — the tab sets diverge",
        )

    async def test_run_broadcasts_task_text_to_all_clients(self) -> None:
        """[3] A run from ANY origin mirrors the task-panel text.

        The common server-side run path must broadcast ``setTaskText``
        so ``run``-originated tasks behave exactly like remote-web
        ``submit`` ones.
        """
        ws_a = await self._connect_ok()
        ws_b = await self._connect_ok()
        await self._ready(ws_a)
        await self._ready(ws_b)

        await self._send(ws_a, {
            "type": "run",
            "prompt": "Mirror this task text",
            "model": "definitely-not-a-real-model",
            "workDir": self.tmpdir,
            "tabId": "tab-tt",
            "useWorktree": False,
            "isParallel": False,
            "autoCommit": False,
        })
        stt_b = await self._wait_for_event(
            ws_b, "setTaskText",
            pred=lambda ev: ev.get("tabId") == "tab-tt",
        )
        self.assertIsNotNone(
            stt_b,
            "the run's task text never reached the other client",
        )
        assert stt_b is not None
        self.assertEqual(stt_b.get("text"), "Mirror this task text")

    async def test_api_run_task_text_arrives_after_tab_adoption(self) -> None:
        """[3b] A run that CREATES its tab re-echoes ``setTaskText``.

        ``sorcar.run`` (the Python client) submits with a synthetic
        ``api-…`` tab id no client has yet, so the submit-ack
        ``setTaskText`` reaches every client BEFORE the ``tabs_state``
        snapshot that registers the tab — clients drop a task-panel
        echo for a tab they have not adopted, and the tab then showed
        its transcript without the fixed task panel at the top.  The
        daemon must broadcast the task text again AFTER the
        registration so an already-connected client can apply it to
        the freshly adopted tab.
        """
        ws_b = await self._connect_ok()
        await self._ready(ws_b)

        # Submit exactly like ``kiss.server.sorcar.run``: raw
        # newline-delimited JSON over the daemon's local endpoint, with a
        # synthetic ``api-…`` tab no client knows about.
        tab_id = "api-panel-e2e"
        reader, writer = await open_local_connection(self.server)
        try:
            writer.write(json.dumps({
                "type": "run",
                "prompt": "Panel me please",
                "tabId": tab_id,
                "model": "definitely-not-a-real-model",
                "workDir": self.tmpdir,
                "useWorktree": False,
                "isParallel": False,
                "autoCommit": False,
            }).encode("utf-8") + b"\n")
            await writer.drain()

            # Client B must observe, IN ORDER: a ``tabs_state``
            # snapshot containing the tab, then a ``setTaskText`` for
            # it — only that one can land on the adopted tab.
            adopted = False
            stt_after_adoption: dict[str, Any] | None = None
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                remaining = max(0.05, deadline - time.monotonic())
                try:
                    raw = await asyncio.wait_for(
                        ws_b.recv(), timeout=remaining,
                    )
                except TimeoutError:
                    break
                ev = json.loads(raw)
                etype = ev.get("type", "")
                if etype == "tabs_state" and tab_id in _tab_ids(ev):
                    adopted = True
                elif (
                    etype == "setTaskText"
                    and ev.get("tabId") == tab_id
                    and adopted
                ):
                    stt_after_adoption = dict(ev)
                    break
            self.assertTrue(
                adopted,
                "the api tab was never registered in the shared registry",
            )
            self.assertIsNotNone(
                stt_after_adoption,
                "no setTaskText for the api tab arrived after the tab "
                "was adopted — the fixed task panel stays hidden",
            )
            assert stt_after_adoption is not None
            self.assertEqual(
                stt_after_adoption.get("text"), "Panel me please",
            )
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def test_ready_replay_carries_prompt_before_history_row(
        self,
    ) -> None:
        """[3c] A reconnect during task setup still repaints the panel.

        A run spends its whole setup (worktree, SEA, agent
        script) BEFORE allocating its ``task_history`` row, so a client
        that connects in that window replays the run through
        ``_replay_session``'s pre-history-row branch.  That branch used
        to broadcast ``task_events`` with ``task: ""`` — blanking the
        fixed task panel of the freshly adopted tab.  It must carry the
        submitted prompt (stamped on the agent state at submit time).
        """
        from kiss.core import config as kiss_config
        from kiss.core.models.model_info import get_available_models

        # The run must survive ``_run_task``'s model validation to
        # reach its (slow) SEA import, so a REAL catalog model
        # is needed.  On a keyless machine, fake one provider
        # credential on the daemon-shared config; the run is stopped
        # while still blocked in the tools import, so no LLM is ever
        # actually called.
        cfg = kiss_config.DEFAULT_CONFIG
        saved_key = cfg.OPENAI_API_KEY
        if not get_available_models():
            cfg.OPENAI_API_KEY = "sk-kiss-e2e-fake"
        try:
            available = get_available_models()
            self.assertTrue(available)
            model = available[0]
            # A SEA that blocks its import (interruptibly, with
            # a 60 s failsafe) holds the run in the pre-history-row
            # setup window while the late client connects.
            tools_py = Path(self.tmpdir) / "slow_agent.py"
            tools_py.write_text(
                """
import time
from kiss.agents.seas.base.base_sea import BaseSea
for _ in range(1200):
    time.sleep(0.05)
class Sea(BaseSea):
    def tools(self, tools):
        return tools + []
""",
                encoding="utf-8",
            )
            tab_id = "api-panel-window"
            ws_a = await self._connect_ok()
            await self._ready(ws_a)
            reader, writer = await open_local_connection(self.server)
            try:
                writer.write(json.dumps({
                    "type": "run",
                    "prompt": "Panel me during setup",
                    "tabId": tab_id,
                    "model": model,
                    "workDir": self.tmpdir,
                    "seaPath": str(tools_py),
                    "useWorktree": False,
                    "isParallel": False,
                    "autoCommit": False,
                }).encode("utf-8") + b"\n")
                await writer.drain()
                # The tab is registered + chat-bound synchronously
                # before the worker thread starts; wait for the
                # snapshot so the late client's ready is guaranteed to
                # find the binding.
                self.assertIsNotNone(
                    await self._wait_for_snapshot_with(
                        ws_a, present={tab_id},
                    ),
                )

                ws_b = await self._connect_ok()
                await self._ready(ws_b, tab_id=tab_id)
                replay = await self._wait_for_event(
                    ws_b, "task_events",
                    pred=lambda ev: ev.get("tabId") == tab_id,
                )
                self.assertIsNotNone(
                    replay,
                    "the late client never received the running tab's "
                    "replay",
                )
                assert replay is not None
                self.assertEqual(
                    replay.get("task"),
                    "Panel me during setup",
                    "the pre-history-row replay blanked the task panel "
                    "text",
                )
            finally:
                # Stop the task while it is still blocked in the tools
                # import (the daemon force-interrupts a non-cooperating
                # setup after ~1 s) and wait for the terminal status so
                # teardown never overlaps a live worker — and, on a
                # machine with real keys, no LLM call is ever made.
                try:
                    writer.write(json.dumps({
                        "type": "stop", "tabId": tab_id,
                    }).encode("utf-8") + b"\n")
                    await writer.drain()
                    await self._wait_for_event(
                        ws_a, "status", timeout=15.0,
                        pred=lambda ev: (
                            ev.get("tabId") == tab_id
                            and not ev.get("running", True)
                        ),
                    )
                except Exception:
                    pass
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
        finally:
            cfg.OPENAI_API_KEY = saved_key

    @unittest.skipIf(is_root(), "root bypasses directory permissions")
    @posix_only("chmod-based directory permission denial")
    async def test_persist_failure_is_flushed_at_shutdown(self) -> None:
        """[4] Live mirroring survives an unwritable KISS dir, and the
        registry is re-persisted at shutdown once the dir is writable.
        """
        kiss_dir = Path(self.tmpdir) / ".kiss"
        tabs_file = kiss_dir / "tabs.json"

        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._send(ws_a, {
            "type": "openTab", "tabId": "tab-w1", "title": "persisted",
        })
        self.assertIsNotNone(
            await self._wait_for_snapshot_with(ws_a, present={"tab-w1"}),
        )
        self.assertIn("tab-w1", tabs_file.read_text(encoding="utf-8"))

        os.chmod(kiss_dir, 0o500)
        try:
            await self._send(ws_a, {
                "type": "openTab", "tabId": "tab-w2", "title": "in memory",
            })
            snap = await self._wait_for_snapshot_with(
                ws_a, present={"tab-w1", "tab-w2"},
            )
            self.assertIsNotNone(
                snap,
                "a persistence failure must not break live mirroring",
            )
            self.assertNotIn(
                "tab-w2", tabs_file.read_text(encoding="utf-8"),
            )
        finally:
            os.chmod(kiss_dir, 0o755)

        await self._stop_server()
        self.assertIn(
            "tab-w2", tabs_file.read_text(encoding="utf-8"),
            "the daemon shut down without re-persisting the tabs that "
            "failed to persist while the dir was unwritable",
        )

    async def test_open_tab_rejected_notifies_originating_client(
        self,
    ) -> None:
        """[5] A cap-rejected ``openTab`` must answer the client.

        Otherwise the originating client keeps a permanently local,
        snapshot-immune tab no other client ever sees.
        """
        await self._stop_server()
        kiss_dir = Path(self.tmpdir) / ".kiss"
        entries = [
            {"tabId": f"seed-{i}", "chatId": "", "title": f"seed {i}",
             "workDir": ""}
            for i in range(512)
        ]
        (kiss_dir / "tabs.json").write_text(
            json.dumps({"tabs": entries}), encoding="utf-8",
        )
        await self._start_server()

        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        self.assertIsNotNone(
            await self._wait_for_snapshot_with(ws_a, present={"seed-0"}),
        )
        await self._send(ws_a, {
            "type": "openTab", "tabId": "tab-overflow", "title": "one too many",
        })
        rejected = await self._wait_for_event(
            ws_a, "openTabRejected",
            pred=lambda ev: ev.get("tabId") == "tab-overflow",
        )
        self.assertIsNotNone(
            rejected,
            "the rejected openTab was silently dropped — the client "
            "keeps a permanently local tab",
        )

    async def test_open_and_close_acks_ride_the_snapshot_that_reflects_them(
        self,
    ) -> None:
        """A18-1: the ``ack`` of an ``openTab`` / ``closeTab`` is echoed
        in the ``acks`` of the first ``tabs_state`` taken after the
        daemon processed that command.

        Snapshots other activity publishes meanwhile carry no such
        token, so a client keeps its optimistic open / close until the
        snapshot that answers it (a count of unrelated snapshots, or
        elapsed time, is no acknowledgment).  Every snapshot carries
        the ``acks`` list (empty when it answers nobody): a page tells
        a daemon that echoes acknowledgments from one that does not.
        """
        ws_a = await self._connect_ok()
        ws_b = await self._connect_ok()
        await self._ready(ws_a)
        await self._ready(ws_b)
        baseline = await self._wait_for_event(ws_b, "tabs_state")
        self.assertIsNotNone(baseline)
        assert baseline is not None
        self.assertEqual(
            baseline.get("acks"), [],
            "every tabs_state carries an acks list, empty when it answers no command",
        )

        def _acking(token: str) -> Any:
            return lambda ev: token in ev.get("acks", [])

        # B's open (sent without a token, as an older page would)
        # publishes a snapshot that acknowledges nobody; the snapshot
        # that acknowledges A's open was taken after it and lists the
        # tab.  (The sender may ship the tab before the token when
        # the two opens overlap; the listing alone also lifts the
        # page's shield, so the order is free.)
        await self._send(ws_b, {
            "type": "openTab", "tabId": "tab-b1", "title": "theirs",
        })
        await self._send(ws_a, {
            "type": "openTab", "tabId": "tab-a1", "title": "mine",
            "ack": "a-open-1",
        })
        opened = await self._wait_for_event(
            ws_a, "tabs_state", pred=_acking("a-open-1"),
        )
        self.assertIsNotNone(opened, "no snapshot acknowledged A's openTab")
        assert opened is not None
        self.assertIn(
            "tab-a1", _tab_ids(opened),
            "the acknowledging snapshot must reflect the open it answers",
        )

        # The close's token rides a snapshot that omits the tab.
        await self._send(ws_a, {
            "type": "closeTab", "tabId": "tab-a1", "ack": "a-close-1",
        })
        closed = await self._wait_for_event(
            ws_a, "tabs_state", pred=_acking("a-close-1"),
        )
        self.assertIsNotNone(closed, "no snapshot acknowledged A's closeTab")
        assert closed is not None
        self.assertNotIn("tab-a1", _tab_ids(closed))
        # A token answers once: the snapshot B's close publishes
        # carries neither of A's.
        await self._send(ws_b, {"type": "closeTab", "tabId": "tab-b1"})
        later = await self._wait_for_snapshot_with(ws_a, absent={"tab-b1"})
        assert later is not None
        self.assertEqual(later.get("acks"), [])

    async def test_replayed_open_does_not_reopen_a_tab_closed_meanwhile(
        self,
    ) -> None:
        """A page's transport re-sends the batch it flushed on a
        connection that died before the daemon's ``pong`` confirmed it
        (media/transport.js, at-least-once), so an ``openTab`` the
        daemon already processed can arrive a second time, with the
        same ``ack`` token, after another surface closed the tab.  The
        replay must not reopen it: the daemon answers with a snapshot
        that echoes the token and omits the tab, which is what the
        replaying page needs to drop its shielded copy.  A fresh token
        for the same id is a new open, as before.
        """
        ws_a = await self._connect_ok()
        ws_b = await self._connect_ok()
        await self._ready(ws_a)
        await self._ready(ws_b)
        self.assertIsNotNone(await self._wait_for_event(ws_b, "tabs_state"))

        def _acking(token: str) -> Any:
            return lambda ev: token in ev.get("acks", [])

        open_cmd = {
            "type": "openTab", "tabId": "tab-a1", "title": "mine",
            "ack": "a-open-1",
        }
        await self._send(ws_a, open_cmd)
        opened = await self._wait_for_event(
            ws_a, "tabs_state", pred=_acking("a-open-1"),
        )
        assert opened is not None
        self.assertIn("tab-a1", _tab_ids(opened))
        self.assertIsNotNone(
            await self._wait_for_snapshot_with(ws_b, present={"tab-a1"}),
        )
        # A's connection dies before the pong; the user closes the tab
        # on B meanwhile.
        await ws_a.close()
        await self._send(ws_b, {
            "type": "closeTab", "tabId": "tab-a1", "ack": "b-close-1",
        })
        closed = await self._wait_for_event(
            ws_b, "tabs_state", pred=_acking("b-close-1"),
        )
        assert closed is not None
        self.assertNotIn("tab-a1", _tab_ids(closed))

        # A's next connection replays the open, token and all.
        ws_a2 = await self._connect_ok()
        await self._send(ws_a2, open_cmd)
        answer = await self._wait_for_event(
            ws_a2, "tabs_state", pred=_acking("a-open-1"),
        )
        self.assertIsNotNone(
            answer,
            "the replayed open was not answered: the page keeps shielding "
            "a tab the registry does not have",
        )
        assert answer is not None
        self.assertNotIn(
            "tab-a1", _tab_ids(answer),
            "REGRESSION: a replayed open (same ack token) reopened the tab "
            "another surface closed meanwhile",
        )
        self.assertIsNone(
            await self._wait_for_snapshot_with(
                ws_b, present={"tab-a1"}, timeout=1.0,
            ),
            "the closed tab came back on the other surface",
        )
        on_disk = json.loads(
            (Path(self.tmpdir) / ".kiss" / "tabs.json").read_text(encoding="utf-8"),
        )
        self.assertNotIn("tab-a1", [t["tabId"] for t in on_disk["tabs"]])

        # A genuine new open of the same id (a fresh token) is opened.
        await self._send(ws_a2, dict(open_cmd, ack="a-open-2"))
        reopened = await self._wait_for_event(
            ws_a2, "tabs_state", pred=_acking("a-open-2"),
        )
        assert reopened is not None
        self.assertIn("tab-a1", _tab_ids(reopened))


@unittest.skipIf(is_root(), "root bypasses directory permissions")
class TestTabRegistryPersistenceFailure(unittest.TestCase):
    """[4] The registry must survive persistence failures loudly."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-tab-reg-")
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.reg_dir = Path(self.tmpdir) / "reg"
        self.reg_dir.mkdir()
        self.path = self.reg_dir / "tabs.json"

    def tearDown(self) -> None:
        os.chmod(self.reg_dir, 0o755)

    @posix_only("chmod-based directory permission denial")
    def test_mutations_survive_and_recover_from_unwritable_dir(self) -> None:
        """Failed persists log ONE error, keep serving the in-memory
        state, and re-persist on the next mutation / flush."""
        reg = TabRegistry(self.path)
        self.assertIs(reg.open_tab("t1", "one"), OpenTabOutcome.OPENED)
        self.assertIn("t1", self.path.read_text(encoding="utf-8"))

        os.chmod(self.reg_dir, 0o500)
        with self.assertLogs("kiss.server.tab_registry", level="ERROR") as cm:
            self.assertIs(reg.open_tab("t2", "two"), OpenTabOutcome.OPENED)
            self.assertIs(reg.open_tab("t3", "three"), OpenTabOutcome.OPENED)
        errors = [r for r in cm.records if r.levelname == "ERROR"]
        self.assertEqual(
            len(errors), 1,
            "consecutive persistence failures must log loudly ONCE, "
            f"got {len(errors)} error records",
        )
        # Live state still serves every mutation.
        ids = [e["tabId"] for e in reg.snapshot()]
        self.assertEqual(ids, ["t1", "t2", "t3"])
        self.assertNotIn("t2", self.path.read_text(encoding="utf-8"))

        # Recovery via the next mutation once the dir is writable.
        os.chmod(self.reg_dir, 0o755)
        self.assertIs(reg.open_tab("t4", "four"), OpenTabOutcome.OPENED)
        on_disk = self.path.read_text(encoding="utf-8")
        for tab in ("t1", "t2", "t3", "t4"):
            self.assertIn(tab, on_disk)

    def test_flush_persists_dirty_state(self) -> None:
        """``flush()`` re-persists state whose last mutation failed."""
        reg = TabRegistry(self.path)
        self.assertIs(reg.open_tab("t1", "one"), OpenTabOutcome.OPENED)
        os.chmod(self.reg_dir, 0o500)
        self.assertIs(reg.open_tab("t2", "two"), OpenTabOutcome.OPENED)
        os.chmod(self.reg_dir, 0o755)
        reg.flush()
        on_disk = self.path.read_text(encoding="utf-8")
        self.assertIn("t1", on_disk)
        self.assertIn("t2", on_disk)
        # A clean flush is a no-op.
        reg.flush()
        self.assertIn("t2", self.path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
