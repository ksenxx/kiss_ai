# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: loading a parent task opens a sub-agent tab for
each of its sub-agents that runs, and closes the tabs of the finished ones.

Spec
----
1. When :meth:`AgentServer._replay_session` loads a PARENT
   ``task_history`` row (no ``extra.subagent`` blob) by ``task_id``,
   it broadcasts the parent's ``task_events`` and then, for every
   sub-agent row whose ``extra.subagent.parent_task_id`` matches the
   parent's id:

   a. RUNNING: an ``openSubagentTab`` event with a deterministic
      ``tab_id = f"{parent_tab_id}__sub_{sub_task_id}"`` (re-loads are
      idempotent on the frontend's ``openSubagentTab`` handler) and a
      ``replay_deferred`` offer of its transcript, which a client
      fetches by resuming the sub-agent tab by name.

   b. FINISHED: a ``subagentDone`` naming that tab id and the
      sub-agent's task, addressed like the rest of the replay; no tab
      is announced, offered or attached.  A running sub-agent below a
      finished one is still announced, under the nearest tab that is
      open.

2. A client resuming a SUB-AGENT row by name gets that one tab
   (running) or its ``subagentDone`` (finished) — no fan-out of
   sibling sub-agents.

3. Sub-tabs are emitted in ``task_history.id`` ASC order
   (the order in which the parent enqueued them).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import redirect_persistence, restore_persistence


def _make_server() -> tuple[AgentServer, list[dict]]:
    """Create a AgentServer whose broadcasts go into an in-memory list."""
    server = AgentServer()
    thread = server._orphan_sweep_thread
    if thread is not None:
        thread.join(timeout=30)
        assert not thread.is_alive(), "orphan sweep did not finish"
    events: list[dict] = []
    lock = threading.Lock()

    real_broadcast = JsonPrinter.broadcast

    def capture(event: dict) -> None:
        ev = server.printer._inject_task_id(event)
        with server.printer._lock:
            server.printer._record_event(ev)
        with lock:
            events.append(ev)

    server.printer.broadcast = capture  # type: ignore[assignment]
    _ = real_broadcast
    return server, events


def _seed_parent(chat_id: str, description: str = "parent task") -> str:
    """Insert a parent ``task_history`` row + one persisted event."""
    task_id, _ = th._add_task(description, chat_id=chat_id)
    th._append_chat_event(
        {"type": "text_delta", "text": "parent-event"},
        task_id=task_id,
    )
    th._save_task_extra(
        {
            "model": "test-model",
            "work_dir": "/tmp",
            "version": "test",
            "tokens": 0,
            "cost": 0.0,
            "is_parallel": True,
            "is_worktree": False,
        },
        task_id=task_id,
    )
    return task_id


def _seed_subagent(
    *, parent_task_id: str, chat_id: str, description: str, event_text: str,
) -> str:
    """Insert a sub-agent ``task_history`` row + one persisted event."""
    task_id, _ = th._add_task(description, chat_id=chat_id)
    th._append_chat_event(
        {"type": "text_delta", "text": event_text},
        task_id=task_id,
    )
    th._save_task_extra(
        {
            "model": "test-model",
            "work_dir": "/tmp",
            "version": "test",
            "tokens": 0,
            "cost": 0.0,
            "is_parallel": False,
            "is_worktree": False,
            "subagent": {"parent_task_id": parent_task_id},
        },
        task_id=task_id,
    )
    return task_id


class TestLoadParentOpensSubagentTabs:
    """Loading a parent task opens one sub-agent tab per running
    sub-agent row and tells the finished ones over."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)
        self.running: list[agent_state.AgentState] = []

    def teardown_method(self) -> None:
        for state in self.running:
            agent_state.unregister(state.task_id, state)
        th._close_thread_db()
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _seed_running(self, parent_task_id: str, chat_id: str, description: str, text: str) -> str:
        """Insert a sub-agent row of *parent_task_id* whose task is running."""
        task_id = _seed_subagent(
            parent_task_id=parent_task_id, chat_id=chat_id,
            description=description, event_text=text,
        )
        state = agent_state.AgentState(
            task_id, chat_id=chat_id, tab_id=f"owner-{task_id}", parent_task_id=parent_task_id,
            server_owned=True, is_task_active=True,
        )
        agent_state.register(state)
        self.running.append(state)
        return task_id

    def test_parent_load_emits_open_subagent_tab_and_events_per_subagent(
        self,
    ) -> None:
        chat_id = "chat-parent-multi"
        parent_id = _seed_parent(chat_id)
        sub_ids = [
            self._seed_running(
                parent_id, chat_id, f"Sub-task {i}: do work {i}", f"subagent-{i}-event",
            )
            for i in range(3)
        ]
        server, events = _make_server()
        parent_tab_id = "tab-parent-history-click"

        server._replay_session(
            chat_id=chat_id, tab_id=parent_tab_id, task_id=parent_id,
        )

        parent_task_events = [
            e for e in events
            if e.get("type") == "task_events"
            and e.get("tabId") == parent_tab_id
        ]
        assert len(parent_task_events) == 1
        assert parent_task_events[0]["task_id"] == parent_id

        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(opens) == 3, f"got opens={opens}"

        expected_sub_tab_ids = [
            f"{parent_tab_id}__sub_{sid}" for sid in sub_ids
        ]
        assert [o["tab_id"] for o in opens] == expected_sub_tab_ids

        for i, op in enumerate(opens):
            assert op["isSubagentTab"] is True
            assert not {"isDone", "reopened", "closedHistory", "incarnation"} & op.keys()
            assert op["description"].startswith(f"Sub-task {i}")
            assert op["taskIndex"] == i
            assert op["parent_tab_id"] == parent_tab_id
        assert not [e for e in events if e.get("type") == "subagentDone"]

        # The history click sends no sub-agent transcript: each tab gets
        # an offer, fetched when the user opens it.
        assert not [
            e for e in events
            if e.get("type") == "task_events"
            and e.get("tabId") in expected_sub_tab_ids
        ]
        offers = [e for e in events if e.get("type") == "replay_deferred"]
        assert [o["tabId"] for o in offers] == expected_sub_tab_ids
        assert [o["taskId"] for o in offers] == [str(s) for s in sub_ids]

        parent_idx = events.index(parent_task_events[0])
        for op in opens:
            assert events.index(op) > parent_idx
        for op, offer in zip(opens, offers, strict=True):
            assert events.index(op) < events.index(offer)

        for i, sub_id in enumerate(sub_ids):
            del events[:]
            server._replay_session(
                chat_id=chat_id, tab_id=expected_sub_tab_ids[i], task_id=sub_id,
            )
            fetched = [e for e in events if e.get("type") == "task_events"]
            assert len(fetched) == 1
            assert fetched[0]["tabId"] == expected_sub_tab_ids[i]
            assert fetched[0]["task_id"] == sub_id
            assert any(
                ev.get("type") == "text_delta"
                and ev.get("text") == f"subagent-{i}-event"
                for ev in fetched[0]["events"]
            )

    def test_finished_subagents_are_told_over_and_get_no_tab(self) -> None:
        """A finished child's ``subagentDone`` stands in for its announcement.

        The walk below the parent and a client asking for the tab by
        name both answer with it, addressed like the replay, and attach
        nothing to the tab.
        """
        chat_id = "chat-parent-finished"
        parent_id = _seed_parent(chat_id)
        sub_ids = [
            _seed_subagent(
                parent_task_id=parent_id, chat_id=chat_id,
                description=f"Sub-task {i}", event_text=f"sub-{i}-event",
            )
            for i in range(2)
        ]
        server, events = _make_server()
        sub_tabs = [f"tab-parent__sub_{sid}" for sid in sub_ids]

        server._replay_session(
            chat_id=chat_id, tab_id="tab-parent", task_id=parent_id,
            conn_id="conn-1", replay_id="replay-1",
        )

        kinds = [e.get("type") for e in events]
        assert "openSubagentTab" not in kinds and "replay_deferred" not in kinds
        done = [e for e in events if e.get("type") == "subagentDone"]
        assert [(e["tab_id"], e["task_id"]) for e in done] == list(zip(sub_tabs, sub_ids))
        assert {(e["connId"], e["replayId"]) for e in done} == {("conn-1", "replay-1")}
        assert kinds.index("task_events") < kinds.index("subagentDone")

        del events[:]
        server._replay_session(
            chat_id=chat_id, tab_id=sub_tabs[0], task_id=sub_ids[0], conn_id="conn-1",
        )
        assert [(e["type"], e["tab_id"], e["task_id"], e["connId"]) for e in events] == [
            ("subagentDone", sub_tabs[0], sub_ids[0], "conn-1"),
        ]
        for tab in sub_tabs:
            assert server.printer.tasks_for_tab(tab) == []
            assert tab not in server._tab_opened_task_ids

    def test_running_grandchild_of_a_finished_child_opens_under_the_parent_tab(self) -> None:
        """A sub-agent that outlived its parent hangs below the nearest tab that is open."""
        chat_id = "chat-parent-orphan"
        parent_id = _seed_parent(chat_id)
        child = _seed_subagent(
            parent_task_id=parent_id, chat_id=chat_id,
            description="finished child", event_text="child-event",
        )
        grand = self._seed_running(child, chat_id, "running grandchild", "grand-event")
        server, events = _make_server()
        child_tab = f"tab-parent__sub_{child}"
        grand_tab = f"{child_tab}__sub_{grand}"

        server._replay_session(chat_id=chat_id, tab_id="tab-parent", task_id=parent_id)

        frames = [
            (e["type"], e.get("tab_id") or e.get("tabId"))
            for e in events
            if e.get("type") in ("openSubagentTab", "subagentDone", "replay_deferred")
        ]
        assert frames == [
            ("subagentDone", child_tab),
            ("openSubagentTab", grand_tab),
            ("replay_deferred", grand_tab),
        ]
        (opened,) = [e for e in events if e.get("type") == "openSubagentTab"]
        assert (opened["parent_tab_id"], opened["parent_task_id"]) == ("tab-parent", child)
        assert server.printer.tasks_for_tab(grand_tab) == [grand]
        assert server.printer.tasks_for_tab(child_tab) == []

    def test_history_click_of_a_wide_fan_out_sends_no_sub_agent_transcript(
        self,
    ) -> None:
        """Ninety sub-agents' transcripts stay on disk until their tabs open.

        Replaying them all with the parent kept the daemon's event loop
        sending about fifty megabytes per viewer for some twelve seconds.
        """
        chat_id = "chat-wide-fan-out"
        parent_id = _seed_parent(chat_id)
        big = "x" * 20_000
        for i in range(90):
            self._seed_running(parent_id, chat_id, f"sub {i}", big)
        server, events = _make_server()

        server._replay_session(chat_id=chat_id, tab_id="tab-wide", task_id=parent_id)

        kinds = [e.get("type") for e in events]
        assert kinds.count("openSubagentTab") == 90
        assert kinds.count("replay_deferred") == 90
        assert kinds.count("task_events") == 1
        sent = sum(len(json.dumps(e)) for e in events)
        assert sent < 200_000, f"{sent} bytes broadcast for 1.8 MB of sub-agent transcripts"

    def test_parent_load_with_no_subagents_emits_only_parent_events(
        self,
    ) -> None:
        chat_id = "chat-parent-no-subs"
        parent_id = _seed_parent(chat_id)
        server, events = _make_server()

        server._replay_session(
            chat_id=chat_id, tab_id="tab-parent", task_id=parent_id,
        )

        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert opens == []
        task_events = [e for e in events if e.get("type") == "task_events"]
        assert len(task_events) == 1
        assert task_events[0]["tabId"] == "tab-parent"

    def test_loading_subagent_row_does_not_open_sibling_subagents(
        self,
    ) -> None:
        """Resuming a running sub-agent's row by name must NOT
        recursively spawn its siblings: only the named tab is announced.
        """
        chat_id = "chat-parent-sib"
        parent_id = _seed_parent(chat_id)
        sub_a = self._seed_running(parent_id, chat_id, "Sub-task A", "sub-a-event")
        self._seed_running(parent_id, chat_id, "Sub-task B", "sub-b-event")
        server, events = _make_server()

        server._replay_session(
            chat_id=chat_id, tab_id="tab-direct-sub", task_id=sub_a,
        )

        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(opens) == 1
        assert opens[0]["tab_id"] == "tab-direct-sub"
        assert opens[0]["description"] == "Sub-task A"

    def test_reloading_parent_uses_stable_sub_tab_ids(self) -> None:
        """Re-clicking the same parent task twice produces identical
        sub-tab ids each time so the frontend's idempotent
        ``openSubagentTab`` handler updates existing tabs instead
        of stacking duplicates.
        """
        chat_id = "chat-parent-reload"
        parent_id = _seed_parent(chat_id)
        sub_ids = [
            self._seed_running(parent_id, chat_id, f"Sub-task {i}", f"sub-{i}-event")
            for i in range(2)
        ]
        server, events = _make_server()
        parent_tab_id = "tab-reload"

        server._replay_session(
            chat_id=chat_id, tab_id=parent_tab_id, task_id=parent_id,
        )
        first_opens = [
            e["tab_id"] for e in events
            if e.get("type") == "openSubagentTab"
        ]
        assert first_opens == [
            f"{parent_tab_id}__sub_{sid}" for sid in sub_ids
        ]

        events.clear()
        server._replay_session(
            chat_id=chat_id, tab_id=parent_tab_id, task_id=parent_id,
        )
        second_opens = [
            e["tab_id"] for e in events
            if e.get("type") == "openSubagentTab"
        ]
        assert second_opens == first_opens
