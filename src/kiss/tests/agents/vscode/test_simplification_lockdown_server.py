# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Characterization (lockdown) tests for server simplifications.

Pins the externally-observable behavior of ``kiss.server.server`` and
``kiss.server.merge_flow`` on top of the task-keyed
``kiss.server.agent_state`` registry:

- ``_get_history`` extra-JSON numeric coercion (garbage -> zero
  defaults, numeric strings round-trip).
- ``_get_running_task_ids`` / ``_overlay_live_metrics`` read live
  metrics from the agent-state registry by task id.
- the busy predicate shared by ``_close_tab`` / ``_dispose_if_closed``
  (deferred tab disposal).
- persisted sub-agent rows are reported done when no live agent is
  registered (``_open_persisted_subagent_tabs``).
- ``_emit_pending_worktree`` is a silent no-op when ``use_worktree``
  is off.
- unknown worktree actions are refused with the exact
  ``Unknown action: <name>`` message (direct call + command dispatch).
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _redirect(tmpdir: str):
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


class _DbTestBase:
    """Shared per-test DB redirection (copied convention from
    ``test_favorite_task.py``)."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


def _register_state(
    task_id: str,
    *,
    agent: WorktreeSorcarAgent | None = None,
    tab_id: str = "",
    is_task_active: bool = False,
) -> agent_state.AgentState:
    """Register and return a server-owned AgentState for a test."""
    st = agent_state.AgentState(
        task_id,
        agent=agent,
        tab_id=tab_id,
        server_owned=True,
        is_task_active=is_task_active,
    )
    agent_state.register(st)
    return st


def _history_sessions(server: VSCodeServer, events: list[dict]) -> list[dict]:
    server._get_history(query=None)
    hist = [e for e in events if e.get("type") == "history"]
    assert len(hist) == 1
    sessions = hist[0]["sessions"]
    assert isinstance(sessions, list)
    return sessions


def _session_by_title(sessions: list[dict], title: str) -> dict:
    matches = [s for s in sessions if s["title"] == title]
    assert len(matches) == 1, f"expected exactly one row titled {title!r}"
    return matches[0]


class TestGetHistoryExtraCoercion(_DbTestBase):
    """``_get_history`` coerces extra-JSON metrics field by field."""

    def test_valid_numbers_round_trip(self) -> None:
        task_id, _ = th._add_task("valid numbers")
        th._save_task_extra(
            {
                "tokens": 123,
                "cost": 1.5,
                "steps": 7,
                "startTs": 1690000000000,
                "endTs": 1700000000000,
            },
            task_id=task_id,
        )
        server, events = _make_server()

        s = _session_by_title(_history_sessions(server, events), "valid numbers")

        assert s["tokens"] == 123
        assert isinstance(s["tokens"], int)
        assert s["cost"] == 1.5
        assert isinstance(s["cost"], float)
        assert s["steps"] == 7
        assert s["startTs"] == 1690000000000
        assert s["endTs"] == 1700000000000

    def test_numeric_strings_are_coerced(self) -> None:
        task_id, _ = th._add_task("numeric strings")
        th._save_task_extra(
            {"tokens": "123", "cost": "2.5", "steps": "9", "endTs": "1700000000001"},
            task_id=task_id,
        )
        server, events = _make_server()

        s = _session_by_title(_history_sessions(server, events), "numeric strings")

        assert s["tokens"] == 123
        assert isinstance(s["tokens"], int)
        assert s["cost"] == 2.5
        assert isinstance(s["cost"], float)
        assert s["steps"] == 9
        assert s["endTs"] == 1700000000001

    def test_garbage_values_yield_zero_defaults(self) -> None:
        task_id, _ = th._add_task("garbage extras")
        th._save_task_extra(
            {
                "tokens": "abc",
                "cost": None,
                "steps": "junk",
                "endTs": "bad",
                "startTs": "bad",
            },
            task_id=task_id,
        )
        server, events = _make_server()

        s = _session_by_title(_history_sessions(server, events), "garbage extras")

        assert s["tokens"] == 0
        assert isinstance(s["tokens"], int)
        assert s["cost"] == 0.0
        assert isinstance(s["cost"], float)
        assert s["steps"] == 0
        assert s["endTs"] == 0
        assert s["startTs"] > 0

    def test_mixed_valid_and_garbage_rows_coexist(self) -> None:
        good_id, _ = th._add_task("good row")
        th._save_task_extra({"tokens": 11, "cost": 0.25, "steps": 3}, task_id=good_id)
        bad_id, _ = th._add_task("bad row")
        th._save_task_extra(
            {"tokens": "abc", "cost": "xyz", "steps": None}, task_id=bad_id,
        )
        server, events = _make_server()

        sessions = _history_sessions(server, events)
        good = _session_by_title(sessions, "good row")
        bad = _session_by_title(sessions, "bad row")

        assert (good["tokens"], good["cost"], good["steps"]) == (11, 0.25, 3)
        assert (bad["tokens"], bad["cost"], bad["steps"]) == (0, 0.0, 0)


class TestUnknownWorktreeAction(_DbTestBase):
    """The merge/discard verb ladder refuses unknown actions."""

    def _arm_pending_worktree(self, tab_id: str, task_id: str) -> None:
        """Register a state whose agent holds a pending worktree."""
        agent = WorktreeSorcarAgent("lockdown worktree agent")
        agent._wt = GitWorktree(
            repo_root=Path(self.tmpdir),
            branch="kiss-wt-test",
            original_branch="main",
            wt_dir=Path(self.tmpdir) / "wt",
            baseline_commit=None,
        )
        st = _register_state(task_id, agent=agent, tab_id=tab_id)
        st.use_worktree = True

    def test_direct_call_returns_unknown_action_message(self) -> None:
        server, _ = _make_server()
        self._arm_pending_worktree("tab-unknown-action", "task-ua")

        result = server._handle_worktree_action(
            "frobnicate", "tab-unknown-action",
        )

        assert result == {
            "success": False,
            "message": "Unknown action: frobnicate",
        }

    def test_command_dispatch_broadcasts_worktree_result(self) -> None:
        server, events = _make_server()
        self._arm_pending_worktree("tab-dispatch", "task-disp")

        server._handle_command({
            "type": "worktreeAction",
            "action": "frobnicate",
            "tabId": "tab-dispatch",
        })

        results = [e for e in events if e.get("type") == "worktree_result"]
        assert len(results) == 1
        assert results[0] == {
            "type": "worktree_result",
            "tabId": "tab-dispatch",
            "success": False,
            "message": "Unknown action: frobnicate",
        }

    def test_guard_ordering_worktree_mode_checked_first(self) -> None:
        """Without ``use_worktree`` even an unknown action gets the
        mode-disabled message, pinning the guard ordering."""
        server, _ = _make_server()
        st = _register_state("task-no-wt", tab_id="tab-no-wt")
        st.use_worktree = False

        result = server._handle_worktree_action("frobnicate", "tab-no-wt")

        assert result == {
            "success": False,
            "message": "Worktree mode is not enabled",
        }


class TestRegistryLiveMetrics(_DbTestBase):
    """Live task ids and metrics come from the agent-state registry."""

    def test_get_running_task_ids_tracks_alive_threads(self) -> None:
        server, _ = _make_server()
        st = _register_state("7", tab_id="tab-live")

        release = threading.Event()
        worker = threading.Thread(target=release.wait, daemon=True)
        worker.start()
        st.task_thread = worker
        try:
            assert server._get_running_task_ids() == {"7"}
        finally:
            release.set()
            worker.join(timeout=5)

    def test_get_running_task_ids_ignores_dead_threads(self) -> None:
        server, _ = _make_server()
        st = _register_state("7", tab_id="tab-dead")

        worker = threading.Thread(target=lambda: None, daemon=True)
        worker.start()
        worker.join(timeout=5)
        st.task_thread = worker

        assert server._get_running_task_ids() == set()

    def test_overlay_live_metrics_reads_registered_agent(self) -> None:
        server, _ = _make_server()
        agent = WorktreeSorcarAgent("lockdown metrics agent")
        agent.total_tokens_used = 555
        agent.budget_used = 1.25
        agent.total_steps = 9
        agent.model_name = "test-model"
        st = _register_state("7", agent=agent, tab_id="tab-overlay")
        st.use_worktree = True
        st.use_parallel = False
        st.auto_commit_mode = False

        matched: dict = {"tokens": 0, "cost": 0.0, "steps": 0}
        server._overlay_live_metrics(matched, "7")
        assert matched == {
            "tokens": 555,
            "cost": 1.25,
            "steps": 9,
            "model": "test-model",
            "is_worktree": True,
            "is_parallel": False,
            "auto_commit_mode": False,
        }

    def test_overlay_live_metrics_ignores_unregistered_task(self) -> None:
        server, _ = _make_server()
        agent = WorktreeSorcarAgent("lockdown unmatched agent")
        agent.total_tokens_used = 100
        _register_state("7", agent=agent, tab_id="tab-unmatched")

        unmatched: dict = {"tokens": 0, "cost": 0.0, "steps": 0}
        server._overlay_live_metrics(unmatched, "42")

        assert unmatched == {"tokens": 0, "cost": 0.0, "steps": 0}


class TestDeferredTabDisposal(_DbTestBase):
    """The busy predicate defers disposal; idle close disposes."""

    def test_close_during_active_task_keeps_state(self) -> None:
        server, _ = _make_server()
        st = _register_state("t-busy", tab_id="tab-busy", is_task_active=True)

        server._handle_command({"type": "closeTab", "tabId": "tab-busy"})

        assert st.frontend_closed is True
        assert agent_state.get("t-busy") is st

    def test_close_during_merge_keeps_state(self) -> None:
        server, _ = _make_server()
        st = _register_state("t-merging", tab_id="tab-merging")
        st.is_merging = True

        server._handle_command({"type": "closeTab", "tabId": "tab-merging"})

        assert st.frontend_closed is True
        assert agent_state.get("t-merging") is st

    def test_close_with_live_thread_keeps_state(self) -> None:
        server, _ = _make_server()
        st = _register_state("t-thread", tab_id="tab-thread")
        release = threading.Event()
        worker = threading.Thread(target=release.wait, daemon=True)
        worker.start()
        st.task_thread = worker
        try:
            server._handle_command({"type": "closeTab", "tabId": "tab-thread"})

            assert st.frontend_closed is True
            assert agent_state.get("t-thread") is st
        finally:
            release.set()
            worker.join(timeout=5)

    def test_close_idle_tab_disposes_immediately(self) -> None:
        server, _ = _make_server()
        _register_state("t-idle", tab_id="tab-idle")

        server._handle_command({"type": "closeTab", "tabId": "tab-idle"})

        assert agent_state.get("t-idle") is None

    def test_dispose_if_closed_after_task_end(self) -> None:
        server, _ = _make_server()
        st = _register_state(
            "t-deferred", tab_id="tab-deferred", is_task_active=True,
        )
        server._handle_command({"type": "closeTab", "tabId": "tab-deferred"})
        assert agent_state.get("t-deferred") is st

        st.is_task_active = False
        server._dispose_if_closed("tab-deferred")

        assert agent_state.get("t-deferred") is None

    def test_dispose_if_closed_noop_when_frontend_open(self) -> None:
        server, _ = _make_server()
        st = _register_state("t-open", tab_id="tab-open")

        server._dispose_if_closed("tab-open")

        assert agent_state.get("t-open") is st


class TestEmitPendingWorktreeNoop(_DbTestBase):
    """``_emit_pending_worktree`` is silent without ``use_worktree``."""

    def test_emit_pending_worktree_noop_without_use_worktree(self) -> None:
        server, events = _make_server()
        st = _register_state("t-no-wt", tab_id="tab-no-worktree")
        st.use_worktree = False
        before = len(events)

        server._emit_pending_worktree("tab-no-worktree")

        assert len(events) == before


class TestPersistedSubagentIsDone(_DbTestBase):
    """Persisted sub-agent rows replay as done when not running."""

    def test_open_persisted_subagent_tabs_marks_done(self) -> None:
        parent_id, chat_id = th._add_task("parent task")
        sub_id, _ = th._add_task(
            "sub task",
            chat_id=chat_id,
            extra={"subagent": {"parent_task_id": parent_id}},
        )
        assert agent_state.get(sub_id) is None
        server, events = _make_server()

        server._open_persisted_subagent_tabs(
            parent_task_id=parent_id, parent_tab_id="tab-parent",
        )

        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(opens) == 1
        assert opens[0]["tab_id"] == f"tab-parent__sub_{sub_id}"
        assert opens[0]["parent_tab_id"] == "tab-parent"
        assert opens[0]["description"] == "sub task"
        assert opens[0]["isDone"] is True
        replays = [e for e in events if e.get("type") == "task_events"]
        assert len(replays) == 1
        assert replays[0]["task_id"] == sub_id
        assert replays[0]["tabId"] == f"tab-parent__sub_{sub_id}"

    def test_no_subagent_rows_no_broadcast(self) -> None:
        parent_id, _ = th._add_task("childless parent")
        server, events = _make_server()

        server._open_persisted_subagent_tabs(
            parent_task_id=parent_id, parent_tab_id="tab-solo",
        )

        assert [e for e in events if e.get("type") == "openSubagentTab"] == []
