# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: a sub-agent ``task_history`` row reopened from
the history sidebar is treated like a regular task except that the
resulting tab is styled as a sub-agent tab (``isSubagentTab=True``,
purple accent) and the frontend suppresses adjacent-task
loading on it.

Spec
----
1. ``ChatSorcarAgent`` sub-agents persist a minimal payload —
   ``extra.subagent = {"parent_task_id": <parent's task_history.id>}``
   — into their own ``task_history`` row.  Presence of the
   ``subagent`` key implies the row is a sub-agent.

2. ``VSCodeServer._get_history`` filters sub-agent rows out of the
   listing entirely — they are an internal implementation detail of
   the parent's ``run_parallel`` tool call and the user-facing
   history tab should only show parent tasks.

3. ``VSCodeServer._replay_session``, when the loaded row carries an
   ``extra.subagent`` blob:

   a. Broadcasts ``openSubagentTab`` for the freshly allocated tab
      with ``description`` derived from the row's own ``task``
      column and ``isDone`` derived from
      :attr:`ChatSorcarAgent.running_agents` membership of the
      sub-agent's own task id.

   b. Does NOT invoke ``_reattach_running_chat`` — sub-agents share
      the parent's chat_id but run as threads inside the parent's
      executor; rebinding the parent's ``_RunningAgentState`` would
      steal it.

   c. Still broadcasts ``task_events`` so persisted history shows.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
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


def _seed_subagent_row(
    *,
    parent_task_id: int,
    chat_id: str,
    description: str,
) -> int:
    """Insert a sub-agent task_history row + one persisted event.

    Returns the inserted task id.
    """
    task_id, _ = th._add_task(description, chat_id=chat_id)
    th._append_chat_event(
        {"type": "text_delta", "text": "subagent-history-event"},
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


class TestHistorySurfacesSubagentMetadata:
    """``_get_history`` must omit sub-agent rows entirely; regular
    rows pass through unchanged."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_subagent_row_is_excluded_from_history_listing(self) -> None:
        chat_id = "chat-parent-1"
        parent_id, _ = th._add_task("parent task", chat_id=chat_id)
        task_id = _seed_subagent_row(
            parent_task_id=parent_id,
            chat_id=chat_id,
            description="Sub-task 3: research X",
        )
        server, events = _make_server()
        server._get_history(query=None, offset=0, generation=0)
        hist_events = [e for e in events if e.get("type") == "history"]
        assert len(hist_events) == 1
        sessions = hist_events[0]["sessions"]
        # The sub-agent row must NOT appear in the user-facing history.
        assert all(s.get("task_id") != task_id for s in sessions)
        # The parent row still appears.
        assert any(s.get("task_id") == parent_id for s in sessions)

    def test_regular_row_has_no_subagent_flag(self) -> None:
        task_id, _ = th._add_task("regular task", chat_id="chat-reg-1")
        th._save_task_extra(
            {"model": "m", "work_dir": "/", "version": "v"},
            task_id=task_id,
        )
        server, events = _make_server()
        server._get_history(query=None, offset=0, generation=0)
        hist = [e for e in events if e.get("type") == "history"][0]
        match = [
            s for s in hist["sessions"] if s.get("task_id") == task_id
        ]
        assert len(match) == 1
        assert "is_subagent" not in match[0]
        assert "parent_task_id" not in match[0]


class TestReplaySessionOpensSubagentTab:
    """``_replay_session`` on a sub-agent row converts the new tab
    into a sub-agent tab."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_replay_subagent_row_emits_open_subagent_then_events(
        self,
    ) -> None:
        chat_id = "chat-parent-2"
        parent_id, _ = th._add_task("parent task", chat_id=chat_id)
        task_id = _seed_subagent_row(
            parent_task_id=parent_id,
            chat_id=chat_id,
            description="Sub-task A",
        )
        server, events = _make_server()
        new_tab_id = "tab-history-click"

        server._replay_session(
            chat_id=chat_id, tab_id=new_tab_id, task_id=task_id,
        )

        # 1. ``openSubagentTab`` was broadcast for the new tab id.
        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(opens) == 1, f"events={events}"
        op = opens[0]
        assert op["tab_id"] == new_tab_id
        assert op["isSubagentTab"] is True
        # Description is taken from the row's own ``task`` column.
        assert op["description"] == "Sub-task A"
        # Sub-agent has completed (not in running_agents) so isDone=True.
        assert op["isDone"] is True

        # 2. ``task_events`` was broadcast after, routed to new tab.
        replays = [e for e in events if e.get("type") == "task_events"]
        assert len(replays) == 1
        assert replays[0]["tabId"] == new_tab_id
        assert replays[0]["chat_id"] == chat_id
        # The persisted event must be present.
        assert any(
            ev.get("type") == "text_delta"
            and ev.get("text") == "subagent-history-event"
            for ev in replays[0]["events"]
        )

        # 3. ``openSubagentTab`` is emitted BEFORE ``task_events`` so
        # the frontend tab is already a sub-agent tab when events
        # arrive.
        open_idx = events.index(opens[0])
        replay_idx = events.index(replays[0])
        assert open_idx < replay_idx

    def test_replay_subagent_does_not_invoke_reattach_running_chat(
        self,
    ) -> None:
        """A sub-agent row must NOT rebind the parent's
        ``_RunningAgentState``.  The parent's tab state holds the
        running thread; rebinding it to the sub-agent's new tab
        would steal the parent's tab.
        """
        chat_id = "chat-parent-3"
        parent_id, _ = th._add_task("parent task", chat_id=chat_id)
        task_id = _seed_subagent_row(
            parent_task_id=parent_id,
            chat_id=chat_id,
            description="Sub-task B",
        )
        server, _events = _make_server()

        # Simulate the parent agent still running under its own tab.
        parent_tab = server._get_tab("tab-parent")
        parent_tab.agent = WorktreeSorcarAgent("Sorcar VS Code")
        parent_tab.agent._chat_id = chat_id
        parent_tab.is_task_active = True
        parent_thread = threading.Thread(
            target=lambda: threading.Event().wait(0.01),
            daemon=True,
        )
        parent_tab.task_thread = parent_thread
        parent_thread.start()

        server._replay_session(
            chat_id=chat_id,
            tab_id="tab-history-click",
            task_id=task_id,
        )

        # Parent tab MUST remain keyed under "tab-parent" — not
        # rebound to "tab-history-click".
        assert "tab-parent" in server._running_agent_states
        assert server._running_agent_states["tab-parent"] is parent_tab
        # Replay is a VIEW operation: the viewer tab must NOT get its
        # own ``_RunningAgentState`` registry entry (no agent runs
        # there), and in particular the parent's state must not have
        # been rebound under the new tab id.
        assert "tab-history-click" not in server._running_agent_states
        parent_thread.join(timeout=1)
