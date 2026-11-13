# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a client's ``ready`` replays transcripts to it alone.

Every (re)connecting chat page sends ``ready``; the daemon answers
by resuming the tab the client shows (``tabId``) so it can rebuild that
transcript, and DEFERS every other chat-bound registry tab: the client
gets the tab's running state and a ``replay_deferred`` offer, and asks
for the transcript (a ``scoped`` ``resumeSession``) when the user
opens the tab.  Those replays used to be broadcast, so opening one more
panel made every other connected window receive — and re-render from
scratch — megabytes of transcripts it already showed; and every tab
used to be replayed, so a laptop with twenty tabs pulled every
transcript on each reconnect.

These tests drive a real :class:`RemoteAccessServer` over real
``wss://`` connections (the harness of
``test_web_server_tab_mirroring``) and pin the contract:

* the ``task_events`` a ``ready`` triggers reach ONLY the connection
  that sent the ``ready``;
* only the tab named by the ``ready`` is replayed; the other bound
  tabs are answered with ``replay_deferred`` (plus their status), and
  a ``scoped`` ``resumeSession`` fetches one of them for that
  connection alone;
* a user's history click (``resumeSession`` without ``replayConnId``)
  still reaches every client — pinned by
  ``test_resume_binds_titles_and_mirrors_transcript`` in the mirroring
  suite, which this file relies on rather than duplicates;
* the conn-scoped replay of a tab whose task runs in a worktree keeps
  the worktree directory the daemon tracks for the tab;
* the persisted sub-agent tabs a parent's replay reopens follow the
  same scope: another client's ``ready`` neither re-sends their
  transcripts nor re-opens them in windows that already show them;
* the one exception: a ``ready`` whose legacy ``restoredTabs`` seed an
  EMPTY registry announces tabs that are new to every client, and
  their replays reach every connection so no window is left with
  empty tabs — as refreshes of what the registry binds when each
  replay runs: a tab another surface closed or rebound meanwhile
  stays closed, or keeps the chat it was rebound to (V8-F1).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from websockets.asyncio.client import ClientConnection

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import init_repo
from kiss.tests.server.pause import Pause
from kiss.tests.server.test_web_server_tab_mirroring import (
    TabMirroringBase,
)


def _bound_to(tab_id: str, chat_id: str) -> Any:
    """A ``tabs_state`` predicate: the snapshot binds *tab_id* to *chat_id*."""
    return lambda ev: any(
        row.get("tabId") == tab_id and row.get("chatId") == chat_id
        for row in ev.get("tabs", [])
    )


def _seed_subagent_row(parent_task_id: str, chat_id: str, text: str) -> str:
    """Record a finished sub-agent of *parent_task_id*; return its task id."""
    task_id, _ = th._add_task("sub-agent task", chat_id=chat_id)
    th._append_chat_event({"type": "text", "text": text}, task_id=task_id)
    th._save_task_extra(
        {"subagent": {"parent_task_id": parent_task_id}}, task_id=task_id,
    )
    th._save_task_result("success: sub done", task_id)
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
        await self._ready(ws_b, tab_id="tab-a")
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

    async def test_ready_defers_every_tab_but_the_one_on_screen(
        self,
    ) -> None:
        """Only the named tab is replayed; the other is offered, then fetched."""
        task_1, chat_1 = self._seed_chat("First chat", [
            {"type": "prompt", "text": "first prompt"},
        ])
        _t2, chat_2 = self._seed_chat("Second chat", [
            {"type": "prompt", "text": "second prompt"},
        ])
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._bind_tab(ws_a, "tab-1", chat_1)
        await self._bind_tab(ws_a, "tab-2", chat_2)

        ws_b = await self._connect_ok()
        await self._ready(ws_b, tab_id="tab-2")
        got = await self._collect(
            ws_b, {"task_events", "replay_deferred", "status"}, timeout=2.0,
        )
        replays = [ev for ev in got if ev.get("type") == "task_events"]
        self.assertEqual(
            [ev.get("tabId") for ev in replays], ["tab-2"],
            "the ready must replay the tab on screen and nothing else",
        )
        deferred = [ev for ev in got if ev.get("type") == "replay_deferred"]
        self.assertEqual(len(deferred), 1, got)
        self.assertEqual(deferred[0].get("tabId"), "tab-1")
        self.assertEqual(deferred[0].get("chatId"), chat_1)
        self.assertEqual(deferred[0].get("taskId"), "")
        # Unpinned, the offer stands for the chat's latest task.
        self.assertEqual(deferred[0].get("latestTaskId"), task_1)
        self.assertNotIn("connId", deferred[0])
        # The deferred tab's state is settled for the client: it is
        # idle, so the page's statusKnown rule may retire it.
        statuses = [
            ev for ev in got
            if ev.get("type") == "status" and ev.get("tabId") == "tab-1"
        ]
        self.assertTrue(statuses, got)
        self.assertFalse(statuses[-1].get("running"))

        # Opening the deferred tab: a scoped resume fetches its
        # transcript for B alone; A, which shows it, is not repainted.
        await self._send(ws_b, {
            "type": "resumeSession", "chatId": chat_1, "tabId": "tab-1",
            "scoped": True,
        })
        fetched = await self._wait_for_event(
            ws_b, "task_events", pred=lambda ev: ev.get("tabId") == "tab-1",
        )
        self.assertIsNotNone(fetched, "the scoped fetch never arrived")
        assert fetched is not None
        self.assertEqual(fetched.get("task_id"), task_1)
        self.assertIn(
            "first prompt",
            [e.get("text") for e in fetched.get("events", [])],
        )
        leaked = await self._wait_for_event(ws_a, "task_events", timeout=1.5)
        self.assertIsNone(leaked, f"the scoped fetch reached client A: {leaked!r}")

    async def test_ready_naming_no_tab_defers_everything(self) -> None:
        """A fresh page (no tab on screen yet) gets offers only."""
        _t1, chat_1 = self._seed_chat("Only chat", [
            {"type": "prompt", "text": "only prompt"},
        ])
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._bind_tab(ws_a, "tab-1", chat_1)

        ws_b = await self._connect_ok()
        await self._ready(ws_b)
        got = await self._collect(
            ws_b, {"task_events", "replay_deferred"}, timeout=2.0,
        )
        self.assertEqual(
            [(ev.get("type"), ev.get("tabId")) for ev in got],
            [("replay_deferred", "tab-1")],
        )

    async def test_deferred_tab_keeps_its_pinned_task(self) -> None:
        """The offer names the task the tab is pinned to, and the fetch loads it."""
        old_task, chat_id = self._seed_chat("Pinned chat", [
            {"type": "prompt", "text": "older prompt"},
        ])
        th._add_task("newer task", chat_id=chat_id)
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._send(ws_a, {
            "type": "openTab", "tabId": "tab-pin", "title": "new chat",
        })
        self.assertIsNotNone(
            await self._wait_for_snapshot_with(ws_a, present={"tab-pin"}),
        )
        await self._send(ws_a, {
            "type": "resumeSession", "chatId": chat_id, "taskId": old_task,
            "tabId": "tab-pin",
        })
        self.assertIsNotNone(await self._wait_for_event(
            ws_a, "task_events", pred=lambda ev: ev.get("tabId") == "tab-pin",
        ))

        ws_b = await self._connect_ok()
        await self._ready(ws_b, tab_id="elsewhere")
        offer = await self._wait_for_event(ws_b, "replay_deferred")
        self.assertIsNotNone(offer)
        assert offer is not None
        self.assertEqual(offer.get("taskId"), old_task)
        self.assertNotIn("latestTaskId", offer, "a pinned offer names its pin alone")
        await self._send(ws_b, {
            "type": "resumeSession", "chatId": offer["chatId"],
            "taskId": offer["taskId"], "tabId": "tab-pin", "scoped": True,
        })
        fetched = await self._wait_for_event(ws_b, "task_events")
        assert fetched is not None
        self.assertEqual(fetched.get("task_id"), old_task)

    async def test_scoped_replay_keeps_running_worktree_tracked(
        self,
    ) -> None:
        """The conn-scoped replay of a running worktree task keeps its worktree dir."""
        repo = f"{self.tmpdir}/repo"
        init_repo(repo)
        task_id, chat_id = self._seed_chat("Worktree task", [
            {"type": "prompt", "text": "change things in a worktree"},
        ])
        agent = ChatSorcarAgent("agent")
        agent._enter_work_dir(repo)
        agent.create_worktree()
        wt_work_dir = str(agent._wt_work_dir)
        state = AgentState(
            str(task_id), agent=agent, chat_id=chat_id, tab_id="tab-w",
            is_task_active=True,
        )
        agent_state.register(state)
        try:
            ws_a = await self._connect_ok()
            await self._ready(ws_a)
            await self._bind_tab(ws_a, "tab-w", chat_id)
            assert self.server is not None
            self.assertEqual(
                self.server._printer.worktree_dir_for_tab("tab-w"), wt_work_dir,
            )

            # B's ready replays the tab to B alone; the replay drops and
            # re-creates the daemon's per-tab state on the way.
            ws_b = await self._connect_ok()
            await self._ready(ws_b, tab_id="tab-w")
            self.assertIsNotNone(await self._wait_for_event(
                ws_b, "task_events", pred=lambda ev: ev.get("tabId") == "tab-w",
            ))
            self.assertEqual(
                self.server._printer.worktree_dir_for_tab("tab-w"), wt_work_dir,
                "the conn-scoped replay dropped the worktree tracking",
            )
        finally:
            state.is_task_active = False
            agent_state.unregister(state.task_id, state)
            agent.discard()

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
        # A's history click (broadcast) opened the child tab on A and
        # offered its transcript instead of sending it.
        self.assertIsNotNone(await self._wait_for_event(
            ws_a, "replay_deferred", pred=lambda ev: ev.get("tabId") == sub_tab,
        ))

        ws_b = await self._connect_ok()
        await self._ready(ws_b, tab_id="tab-p")
        kinds = {"task_events", "openSubagentTab", "replay_deferred"}
        got_b = await self._collect(ws_b, kinds)
        # The parent's transcript is replayed; the child's tab is
        # announced and its transcript offered, not sent.
        self.assertEqual(
            [
                str(ev.get("tabId")) for ev in got_b
                if ev.get("type") == "task_events"
            ],
            ["tab-p"],
            "B must receive the parent's transcript and no other",
        )
        self.assertEqual(
            [
                ev.get("tab_id") for ev in got_b
                if ev.get("type") == "openSubagentTab"
            ],
            [sub_tab],
            "B must be told to open the child tab",
        )
        offers = [ev for ev in got_b if ev.get("type") == "replay_deferred"]
        self.assertEqual(
            [(ev.get("tabId"), ev.get("taskId")) for ev in offers],
            [(sub_tab, sub_id)],
            "the child's transcript must be offered",
        )
        self.assertEqual(
            await self._collect(ws_a, kinds), [],
            "B's ready re-sent the child tab / transcript to A",
        )
        # Opening the child tab fetches its transcript for B alone.
        await self._send(ws_b, {
            "type": "resumeSession", "chatId": chat_id, "taskId": sub_id,
            "tabId": sub_tab, "scoped": True,
        })
        fetched = await self._wait_for_event(
            ws_b, "task_events", pred=lambda ev: ev.get("tabId") == sub_tab,
        )
        assert fetched is not None
        self.assertIn(
            "child transcript",
            [e.get("text") for e in fetched.get("events", [])],
        )
        self.assertEqual(
            await self._collect(ws_a, {"task_events", "openSubagentTab"}), [],
            "the child's scoped fetch reached A",
        )

    async def test_deferred_parent_announces_its_subagent_tabs(self) -> None:
        """A deferred parent's children are announced (offered) as well."""
        parent_id, chat_id = self._seed_chat("Parent task", [
            {"type": "prompt", "text": "fan out"},
        ])
        sub_id = _seed_subagent_row(parent_id, chat_id, "child transcript")
        sub_tab = f"tab-p__sub_{sub_id}"
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._bind_tab(ws_a, "tab-p", chat_id)

        ws_b = await self._connect_ok()
        await self._ready(ws_b, tab_id="elsewhere")
        got_b = await self._collect(
            ws_b, {"task_events", "openSubagentTab", "replay_deferred"},
        )
        self.assertEqual(
            [ev.get("tabId") for ev in got_b if ev.get("type") == "task_events"],
            [],
        )
        # Each child with the task that started it: a page tells by it
        # which of its sub-agent tabs this answer names.
        self.assertEqual(
            [
                (ev.get("tab_id"), ev.get("parent_task_id")) for ev in got_b
                if ev.get("type") == "openSubagentTab"
            ],
            [(sub_tab, parent_id)],
        )
        self.assertEqual(
            sorted(
                str(ev.get("tabId")) for ev in got_b
                if ev.get("type") == "replay_deferred"
            ),
            sorted(["tab-p", sub_tab]),
        )

    async def test_deferred_offer_names_the_task_whose_subagents_it_announces(
        self,
    ) -> None:
        """The child of a task the chat replaced is not announced; asked by name, it is done.

        A page away while the chat moved on still shows the replaced
        task, its child's tab running.  The offer names the task it
        stands for, so the page can tell it is another one and ask
        about the child, where it used to keep the spinner until the
        user opened the chat.
        """
        old_task, chat_id = self._seed_chat("Parent task", [
            {"type": "prompt", "text": "fan out"},
        ])
        sub_id = _seed_subagent_row(old_task, chat_id, "child transcript")
        sub_tab = f"tab-p__sub_{sub_id}"
        new_task, _ = th._add_task("Follow-up task", chat_id=chat_id)
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._bind_tab(ws_a, "tab-p", chat_id)

        ws_b = await self._connect_ok()
        await self._ready(ws_b, tab_id="elsewhere")
        got_b = await self._collect(
            ws_b, {"task_events", "openSubagentTab", "replay_deferred"},
        )
        self.assertEqual(
            [(ev.get("type"), ev.get("tabId"), ev.get("latestTaskId")) for ev in got_b],
            [("replay_deferred", "tab-p", new_task)],
            "the offer must name the follow-up, which fanned out nothing",
        )
        await self._send(ws_b, {
            "type": "resumeSession", "taskId": sub_id, "tabId": sub_tab,
            "scoped": True,
        })
        answer = await self._wait_for_event(
            ws_b, "openSubagentTab", pred=lambda ev: ev.get("tab_id") == sub_tab,
        )
        assert answer is not None, "the child asked about by name was not answered"
        self.assertIs(answer.get("isDone"), True)
        self.assertEqual(answer.get("parent_task_id"), old_task)

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

    def _hold_seeded_replay(self, tab_id: str, chat_id: str) -> Pause:
        """Hold the executor at the replay a seeding ``ready`` queued for *tab_id*.

        The ready chose what to replay from a snapshot of the registry;
        the hold widens the window before the replay runs, in which
        another surface can close or rebind the tab.
        """
        return Pause(AgentServer._handle_command.__code__, lambda local: (
            local["cmd"].get("type") == "resumeSession"
            and local["cmd"].get("tabId") == tab_id
            and local["cmd"].get("chatId") == chat_id
        ))

    async def test_a_seeding_ready_does_not_reopen_a_tab_closed_meanwhile(
        self,
    ) -> None:
        """V8-F1 (audit 2026-11-08): a tab closed after the ready chose to replay it
        stays closed on every surface; the replay publishes nothing."""
        _task_id, chat_id = self._seed_chat("adopted chat", [
            {"type": "prompt", "text": "adopted transcript"},
        ])
        ws_a = await self._connect_ok()
        ws_b = await self._connect_ok()
        assert self.server is not None
        registry = self.server._agent_server.tab_registry
        tab = "tab-adopted"
        with self._hold_seeded_replay(tab, chat_id) as pause:
            await self._ready(ws_a, restored=[
                {"tabId": tab, "chatId": chat_id, "title": "adopted chat"},
            ], tab_id=tab)
            self.assertTrue(await asyncio.to_thread(pause.paused.wait, 15))
            self.assertTrue(registry.has_tab(tab))
            await self._send(ws_b, {"type": "closeTab", "tabId": tab})
            self.assertIsNotNone(
                await self._wait_for_snapshot_with(ws_b, absent={tab}),
            )
            self.assertFalse(registry.has_tab(tab))
            pause.release()
            reopened = await self._wait_for_snapshot_with(
                ws_b, present={tab}, timeout=2.0,
            )
            self.assertIsNone(
                reopened, f"the seeded replay reopened the closed tab: {reopened}",
            )
            self.assertFalse(registry.has_tab(tab))

    async def test_a_seeding_ready_does_not_rebind_a_tab_navigated_meanwhile(
        self,
    ) -> None:
        """V8-F1 (audit 2026-11-08): a tab rebound after the ready chose to replay
        it keeps the newer chat everywhere, and the ready's client gets THAT
        chat's transcript, not the stale one."""
        _old_task, old_chat = self._seed_chat("old adopted chat", [
            {"type": "prompt", "text": "old transcript"},
        ])
        _new_task, new_chat = self._seed_chat("new user-selected chat", [
            {"type": "prompt", "text": "new transcript"},
        ])
        ws_a = await self._connect_ok()
        ws_b = await self._connect_ok()
        assert self.server is not None
        registry = self.server._agent_server.tab_registry
        tab = "tab-navigated"
        with self._hold_seeded_replay(tab, old_chat) as pause:
            await self._ready(ws_a, restored=[
                {"tabId": tab, "chatId": old_chat, "title": "old adopted chat"},
            ], tab_id=tab)
            self.assertTrue(await asyncio.to_thread(pause.paused.wait, 15))
            await self._send(ws_b, {
                "type": "resumeSession", "tabId": tab, "chatId": new_chat,
            })
            self.assertIsNotNone(await self._wait_for_event(
                ws_b, "tabs_state", pred=_bound_to(tab, new_chat),
            ))
            pause.release()
            stale = await self._wait_for_event(
                ws_b, "tabs_state", pred=_bound_to(tab, old_chat), timeout=2.0,
            )
            self.assertIsNone(
                stale, f"the seeded replay rebound the tab to its old chat: {stale}",
            )
            self.assertEqual(
                [chat for t, chat, _task in registry.bound_tabs() if t == tab],
                [new_chat],
            )
            # The ready's client resumed from the stale snapshot: what it
            # gets is the chat the tab shows now, never the old one.
            old_replay_a = await self._wait_for_event(
                ws_a, "task_events", timeout=2.0,
                pred=lambda ev: ev.get("tabId") == tab and ev.get("chat_id") == old_chat,
            )
            self.assertIsNone(
                old_replay_a, "the ready's client received the stale chat's transcript",
            )
