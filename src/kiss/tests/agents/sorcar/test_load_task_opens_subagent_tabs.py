# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: loading a parent task from the history sidebar
also reopens every persisted sub-agent fanned out by the parent's
``run_parallel`` call, each in its own sub-agent tab.

Spec
----
1. When :meth:`VSCodeServer._replay_session` loads a PARENT
   ``task_history`` row (no ``extra.subagent`` blob) by ``task_id``,
   it broadcasts the parent's ``task_events`` (existing behavior) and
   then, for every persisted sub-agent row whose
   ``extra.subagent.parent_task_id`` matches the parent's id, fires:

   a. An ``openSubagentTab`` event with a deterministic
      ``tab_id = f"{parent_tab_id}__sub_{sub_task_id}"`` so re-loads
      are idempotent on the frontend's ``openSubagentTab`` handler.

   b. A ``task_events`` event routed to that sub tab id so the
      sub-agent tab's output panel is populated with its persisted
      events.

2. When the loaded row is itself a SUB-AGENT row (clicked directly
   from history), the existing single-tab subagent-conversion path
   runs unchanged — no extra fan-out of sibling sub-agents.

3. Sub-tabs are emitted in ``task_history.id`` ASC order
   (the order in which the parent enqueued them).
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.agents.vscode.server import VSCodeServer


def _redirect(tmpdir: str) -> tuple[Path, object, Path]:
    """Redirect the persistence DB to a temp dir; return saved state."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer whose broadcasts go into an in-memory list."""
    server = VSCodeServer()
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


def _seed_parent(chat_id: str, description: str = "parent task") -> int:
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
    *, parent_task_id: int, chat_id: str, description: str, event_text: str,
) -> int:
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
    """Loading a parent task opens one sub-agent tab per persisted
    sub-agent row, populated with that row's events."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_parent_load_emits_open_subagent_tab_and_events_per_subagent(
        self,
    ) -> None:
        chat_id = "chat-parent-multi"
        parent_id = _seed_parent(chat_id)
        sub_ids = [
            _seed_subagent(
                parent_task_id=parent_id,
                chat_id=chat_id,
                description=f"Sub-task {i}: do work {i}",
                event_text=f"subagent-{i}-event",
            )
            for i in range(3)
        ]
        server, events = _make_server()
        parent_tab_id = "tab-parent-history-click"

        server._replay_session(
            chat_id=chat_id, tab_id=parent_tab_id, task_id=parent_id,
        )

        # Parent ``task_events`` was broadcast exactly once and is
        # routed to the parent tab.
        parent_task_events = [
            e for e in events
            if e.get("type") == "task_events"
            and e.get("tabId") == parent_tab_id
        ]
        assert len(parent_task_events) == 1
        assert parent_task_events[0]["task_id"] == parent_id

        # One ``openSubagentTab`` per persisted sub-agent row.
        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(opens) == 3, f"got opens={opens}"

        # Deterministic sub-tab ids derived from parent tab id +
        # sub-agent task id.
        expected_sub_tab_ids = [
            f"{parent_tab_id}__sub_{sid}" for sid in sub_ids
        ]
        assert [o["tab_id"] for o in opens] == expected_sub_tab_ids

        # Each open carries the row's description as taken from the
        # ``task`` column and signals sub-agent styling + completion.
        for i, op in enumerate(opens):
            assert op["isSubagentTab"] is True
            # Sub-agents not currently in ``running_agents`` ⇒ done.
            assert op["isDone"] is True
            assert op["description"].startswith(f"Sub-task {i}")
            assert op["taskIndex"] == i
            assert op["parent_tab_id"] == parent_tab_id

        # One ``task_events`` per sub-agent tab, routed to its sub
        # tab id and carrying the sub-agent's persisted event.
        sub_task_events = [
            e for e in events
            if e.get("type") == "task_events"
            and e.get("tabId") in expected_sub_tab_ids
        ]
        assert len(sub_task_events) == 3
        for i, te in enumerate(sub_task_events):
            assert te["tabId"] == expected_sub_tab_ids[i]
            assert te["task_id"] == sub_ids[i]
            assert any(
                ev.get("type") == "text_delta"
                and ev.get("text") == f"subagent-{i}-event"
                for ev in te["events"]
            )

        # Ordering: parent's ``task_events`` comes before any sub-agent
        # ``openSubagentTab`` so the frontend renders the parent first.
        parent_idx = events.index(parent_task_events[0])
        for op in opens:
            assert events.index(op) > parent_idx
        # Each sub-tab's ``openSubagentTab`` precedes its
        # ``task_events`` so the tab exists when events arrive.
        for op, te in zip(opens, sub_task_events, strict=True):
            assert events.index(op) < events.index(te)

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
        """Clicking a sub-agent row directly must NOT recursively
        spawn its siblings — the existing subagent-conversion path
        styles only the clicked tab.
        """
        chat_id = "chat-parent-sib"
        parent_id = _seed_parent(chat_id)
        sub_a = _seed_subagent(
            parent_task_id=parent_id,
            chat_id=chat_id,
            description="Sub-task A",
            event_text="sub-a-event",
        )
        _seed_subagent(
            parent_task_id=parent_id,
            chat_id=chat_id,
            description="Sub-task B",
            event_text="sub-b-event",
        )
        server, events = _make_server()

        server._replay_session(
            chat_id=chat_id, tab_id="tab-direct-sub", task_id=sub_a,
        )

        # Exactly one ``openSubagentTab`` (for the clicked sub-agent
        # row itself, NOT one per sibling).
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
            _seed_subagent(
                parent_task_id=parent_id,
                chat_id=chat_id,
                description=f"Sub-task {i}",
                event_text=f"sub-{i}-event",
            )
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
