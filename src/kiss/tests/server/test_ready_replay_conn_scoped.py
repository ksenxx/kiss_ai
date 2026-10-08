# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a client's ``ready`` replays transcripts to it alone.

Every (re)connecting chat webview sends ``ready``; the daemon answers
by resuming every chat-bound registry tab so the client can rebuild
its transcripts.  Those replays used to be broadcast, so opening one
more panel made every other connected window receive — and re-render
from scratch — megabytes of transcripts it already showed.

These tests drive a real :class:`RemoteAccessServer` over real
``wss://`` connections (the harness of
``test_web_server_tab_mirroring``) and pin the contract:

* the ``task_events`` a ``ready`` triggers reach ONLY the connection
  that sent the ``ready``;
* a ``ready`` naming ``singleTabId`` (an editor-tab panel mirroring one
  registry tab) receives that tab's transcript and no other;
* a user's history click (``resumeSession`` without ``replayConnId``)
  still reaches every client — pinned by
  ``test_resume_binds_titles_and_mirrors_transcript`` in the mirroring
  suite, which this file relies on rather than duplicates;
* the conn-scoped delivery still re-presents a tab's worktree to the
  daemon: after a daemon restart, the first ``ready`` restores the
  worktree directory tracked for a tab whose transcript created one;
* the persisted sub-agent tabs a parent's replay reopens follow the
  same scope: another client's ``ready`` neither re-sends their
  transcripts nor re-opens them in windows that already show them;
* the one exception: a ``ready`` whose legacy ``restoredTabs`` seed an
  EMPTY registry announces tabs that are new to every client, and
  their replays are broadcast so no window is left with empty tabs.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from websockets.asyncio.client import ClientConnection

import kiss.agents.sorcar.persistence as th
from kiss.tests.server.test_web_server_tab_mirroring import (
    TabMirroringBase,
)


def _seed_subagent_row(parent_task_id: str, chat_id: str, text: str) -> str:
    """Record a finished sub-agent of *parent_task_id*; return its task id."""
    task_id, _ = th._add_task("sub-agent task", chat_id=chat_id)
    th._append_chat_event({"type": "text", "text": text}, task_id=task_id)
    th._save_task_extra(
        {"subagent": {"parent_task_id": parent_task_id}}, task_id=task_id,
    )
    th._save_task_result(task_id, "success: sub done")
    return task_id


class TestReadyReplayConnScoped(TabMirroringBase):
    """``ready``-driven replays are private to the requesting client."""

    async def _collect(
        self, ws: ClientConnection, wanted: set[str], timeout: float = 1.5,
    ) -> list[dict[str, Any]]:
        """Return every event of a *wanted* type *ws* gets within *timeout*."""
        got: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return got
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError:
                return got
            ev = json.loads(raw)
            if ev.get("type") in wanted:
                got.append(dict(ev))

    async def _bind_tab(
        self, ws: ClientConnection, tab_id: str, chat_id: str,
    ) -> None:
        """Open *tab_id* on *ws* and resume *chat_id* in it (history click)."""
        await self._send(ws, {
            "type": "openTab", "tabId": tab_id, "title": "new chat",
        })
        self.assertIsNotNone(
            await self._wait_for_snapshot_with(ws, present={tab_id}),
        )
        await self._send(ws, {
            "type": "resumeSession", "chatId": chat_id, "tabId": tab_id,
        })
        self.assertIsNotNone(await self._wait_for_event(
            ws, "task_events", pred=lambda ev: ev.get("tabId") == tab_id,
        ))

    async def test_ready_replay_reaches_only_the_reconnecting_client(
        self,
    ) -> None:
        """Client A already shows tab-a; B's ``ready`` must not repaint A."""
        _task_id, chat_id = self._seed_chat("Seeded task", [
            {"type": "prompt", "text": "seeded prompt"},
            {"type": "text", "text": "seeded answer"},
        ])
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._bind_tab(ws_a, "tab-a", chat_id)

        ws_b = await self._connect_ok()
        await self._ready(ws_b)
        replay_b = await self._wait_for_event(
            ws_b, "task_events", pred=lambda ev: ev.get("tabId") == "tab-a",
        )
        self.assertIsNotNone(
            replay_b, "the (re)connecting client never got tab-a's transcript",
        )
        assert replay_b is not None
        self.assertNotIn(
            "connId", replay_b,
            "the routing stamp must not leak to the client",
        )
        texts = [e.get("text") for e in replay_b.get("events", [])]
        self.assertIn("seeded prompt", texts)

        # A, which already shows tab-a, must not receive the replay B's
        # ready triggered: that broadcast is what made every open window
        # rebuild every transcript whenever one more panel connected.
        leaked = await self._wait_for_event(
            ws_a, "task_events", timeout=1.5,
        )
        self.assertIsNone(
            leaked,
            f"client A received a replay triggered by client B's ready: "
            f"{leaked!r}",
        )

    async def test_single_tab_ready_replays_only_the_mirrored_tab(
        self,
    ) -> None:
        """An editor-tab panel (``singleTabId``) gets its tab, nothing else."""
        _t1, chat_1 = self._seed_chat("First chat", [
            {"type": "prompt", "text": "first prompt"},
        ])
        _t2, chat_2 = self._seed_chat("Second chat", [
            {"type": "prompt", "text": "second prompt"},
        ])
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._bind_tab(ws_a, "tab-1", chat_1)
        await self._bind_tab(ws_a, "tab-2", chat_2)

        ws_panel = await self._connect_ok()
        await self._send(ws_panel, {
            "type": "ready", "tabId": "tab-2", "singleTabId": "tab-2",
            "restoredTabs": [],
        })
        replay = await self._wait_for_event(
            ws_panel, "task_events",
        )
        self.assertIsNotNone(replay, "the panel never got its own transcript")
        assert replay is not None
        self.assertEqual(replay.get("tabId"), "tab-2")
        texts = [e.get("text") for e in replay.get("events", [])]
        self.assertIn("second prompt", texts)
        foreign = await self._wait_for_event(
            ws_panel, "task_events", timeout=1.5,
        )
        self.assertIsNone(
            foreign,
            f"a single-tab panel received another tab's transcript: "
            f"{foreign!r}",
        )

    async def test_scoped_replay_restores_worktree_tracking_after_restart(
        self,
    ) -> None:
        """The conn-scoped replay still records the tab's worktree dir."""
        wt_dir = f"{self.tmpdir}/wt/feature-x"
        _task_id, chat_id = self._seed_chat("Worktree task", [
            {"type": "prompt", "text": "change things in a worktree"},
            {"type": "worktree_created", "worktreeDir": wt_dir},
            {"type": "text", "text": "done"},
        ])
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._bind_tab(ws_a, "tab-w", chat_id)
        assert self.server is not None
        self.assertEqual(
            self.server._printer.worktree_dir_for_tab("tab-w"), wt_dir,
        )

        # A restarted daemon knows nothing about the worktree until the
        # first ready replays the tab's transcript — to that client only.
        await self._stop_server()
        await self._start_server()
        assert self.server is not None
        self.assertEqual(self.server._printer.worktree_dir_for_tab("tab-w"), "")

        ws_b = await self._connect_ok()
        await self._ready(ws_b)
        self.assertIsNotNone(await self._wait_for_event(
            ws_b, "task_events", pred=lambda ev: ev.get("tabId") == "tab-w",
        ))
        self.assertEqual(
            self.server._printer.worktree_dir_for_tab("tab-w"), wt_dir,
            "the conn-scoped replay skipped the worktree tracking",
        )

    async def test_persisted_subagent_tabs_follow_the_replay_scope(
        self,
    ) -> None:
        """B's ready reopens the parent's sub-agent tabs for B only."""
        parent_id, chat_id = self._seed_chat("Parent task", [
            {"type": "prompt", "text": "fan out"},
        ])
        sub_id = _seed_subagent_row(parent_id, chat_id, "child transcript")
        sub_tab = f"tab-p__sub_{sub_id}"

        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._bind_tab(ws_a, "tab-p", chat_id)
        # A's history click (broadcast) opened the child tab on A.
        self.assertIsNotNone(await self._wait_for_event(
            ws_a, "task_events", pred=lambda ev: ev.get("tabId") == sub_tab,
        ))

        ws_b = await self._connect_ok()
        await self._ready(ws_b)
        kinds = {"task_events", "openSubagentTab"}
        got_b = await self._collect(ws_b, kinds)
        self.assertEqual(
            sorted(
                str(ev.get("tabId")) for ev in got_b
                if ev.get("type") == "task_events"
            ),
            sorted(["tab-p", sub_tab]),
            "B must receive the parent's and the child's transcripts",
        )
        self.assertEqual(
            [
                ev.get("tab_id") for ev in got_b
                if ev.get("type") == "openSubagentTab"
            ],
            [sub_tab],
            "B must be told to open the child tab",
        )
        self.assertEqual(
            await self._collect(ws_a, kinds), [],
            "B's ready re-sent the child tab / transcript to A",
        )

    async def test_legacy_seeding_ready_broadcasts_the_replays(
        self,
    ) -> None:
        """Tabs a ready seeds into an empty registry are replayed to all."""
        _task_id, chat_id = self._seed_chat("Legacy chat", [
            {"type": "prompt", "text": "legacy prompt"},
        ])
        ws_a = await self._connect_ok()
        await self._ready(ws_a)

        ws_b = await self._connect_ok()
        await self._ready(ws_b, restored=[
            {"tabId": "tab-leg", "chatId": chat_id, "title": "Legacy chat"},
        ])
        # A learns of tab-leg from the snapshot and, since it never
        # asks for a transcript on its own, must get the replay too.
        self.assertIsNotNone(
            await self._wait_for_snapshot_with(ws_a, present={"tab-leg"}),
        )
        replay_a = await self._wait_for_event(
            ws_a, "task_events", pred=lambda ev: ev.get("tabId") == "tab-leg",
        )
        self.assertIsNotNone(
            replay_a,
            "the client that adopted the seeded tab was left with an "
            "empty transcript",
        )
        assert replay_a is not None
        texts = [e.get("text") for e in replay_a.get("events", [])]
        self.assertIn("legacy prompt", texts)

    async def test_legacy_seeding_by_single_tab_panel_replays_every_tab(
        self,
    ) -> None:
        """A seeding editor panel's ``singleTabId`` must not hide the others."""
        _t1, chat_1 = self._seed_chat("Legacy one", [
            {"type": "prompt", "text": "legacy one"},
        ])
        _t2, chat_2 = self._seed_chat("Legacy two", [
            {"type": "prompt", "text": "legacy two"},
        ])
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        # A's ready must finish its registry sync before the panel
        # seeds the registry: two concurrent readies legitimately
        # replay the seeded tabs to A twice (A's own conn-scoped sync
        # plus the panel's broadcast), which is not what this test
        # measures.  The empty snapshot is the sync's last step.
        self.assertIsNotNone(await self._wait_for_snapshot_with(ws_a))

        ws_panel = await self._connect_ok()
        await self._send(ws_panel, {
            "type": "ready", "tabId": "tab-2", "singleTabId": "tab-2",
            "restoredTabs": [
                {"tabId": "tab-1", "chatId": chat_1, "title": "Legacy one"},
                {"tabId": "tab-2", "chatId": chat_2, "title": "Legacy two"},
            ],
        })
        self.assertIsNotNone(await self._wait_for_snapshot_with(
            ws_a, present={"tab-1", "tab-2"},
        ))
        got_a = await self._collect(ws_a, {"task_events"}, timeout=3.0)
        self.assertEqual(
            sorted(str(ev.get("tabId")) for ev in got_a),
            ["tab-1", "tab-2"],
            "every tab the panel seeded must be replayed to the other client",
        )
