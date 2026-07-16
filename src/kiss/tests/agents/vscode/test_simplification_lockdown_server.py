# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Characterization (lockdown) tests for planned server simplifications.

Pins the CURRENT externally-observable behavior of
``kiss.server.server`` and ``kiss.server.merge_flow``
so the refactorings catalogued in ``tmp/findings-4.md`` cannot
silently change semantics:

- A2: live-task-id preference (``agent._last_task_id`` over
  ``tab.task_history_id``) in ``_get_running_task_ids`` /
  ``_overlay_live_metrics``.
- A3: the tab-busy predicate shared by ``_close_tab`` /
  ``_dispose_if_closed`` (deferred tab disposal).
- A4: ``_get_history`` extra-JSON numeric coercion (garbage -> zero
  defaults, numeric strings round-trip).
- A5: persisted sub-agent rows are reported done when no live agent
  is registered (``_open_persisted_subagent_tabs``).
- C1: ``_ensure_wt_agent`` returns exactly ``tab.agent``.
- C2: ``_emit_pending_worktree`` is a silent no-op when
  ``use_worktree`` is off.
- C6: unknown worktree actions are refused with the exact
  ``Unknown action: <name>`` message (direct call + command dispatch).
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
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
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


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
    """A4: ``_get_history`` coerces extra-JSON metrics field by field."""

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
        # Garbage startTs keeps the row-timestamp fallback (ms since
        # epoch from the INSERT time, hence > 0).
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
    """C6: the merge/discard verb ladder refuses unknown actions."""

    def _arm_pending_worktree(self, server: VSCodeServer, tab_id: str) -> None:
        """Give *tab_id* a real agent holding a pending worktree snapshot."""
        tab = server._get_tab(tab_id)
        tab.use_worktree = True
        assert tab.agent is not None
        tab.agent._wt = GitWorktree(
            repo_root=Path(self.tmpdir),
            branch="kiss-wt-test",
            original_branch="main",
            wt_dir=Path(self.tmpdir) / "wt",
            baseline_commit=None,
        )

    def test_direct_call_returns_unknown_action_message(self) -> None:
        server, _ = _make_server()
        self._arm_pending_worktree(server, "tab-unknown-action")

        result = server._handle_worktree_action(
            "frobnicate", "tab-unknown-action",
        )

        assert result == {
            "success": False,
            "message": "Unknown action: frobnicate",
        }

    def test_command_dispatch_broadcasts_worktree_result(self) -> None:
        server, events = _make_server()
        self._arm_pending_worktree(server, "tab-dispatch")

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
        server._get_tab("tab-no-wt")

        result = server._handle_worktree_action("frobnicate", "tab-no-wt")

        assert result == {
            "success": False,
            "message": "Worktree mode is not enabled",
        }


class TestLiveTaskIdFallback(_DbTestBase):
    """A2: prefer ``agent._last_task_id``, fall back to ``task_history_id``."""

    def test_get_running_task_ids_prefers_agent_last_task_id(self) -> None:
        server, _ = _make_server()
        tab = server._get_tab("tab-live")
        assert tab.agent is not None
        tab.agent._last_task_id = "7"
        tab.task_history_id = "42"

        release = threading.Event()
        worker = threading.Thread(target=release.wait, daemon=True)
        worker.start()
        tab.task_thread = worker
        try:
            assert server._get_running_task_ids() == {"7"}

            tab.agent._last_task_id = None
            assert server._get_running_task_ids() == {"42"}
        finally:
            release.set()
            worker.join(timeout=5)

    def test_get_running_task_ids_ignores_dead_threads(self) -> None:
        server, _ = _make_server()
        tab = server._get_tab("tab-dead")
        assert tab.agent is not None
        tab.agent._last_task_id = "7"
        tab.task_history_id = "42"

        worker = threading.Thread(target=lambda: None, daemon=True)
        worker.start()
        worker.join(timeout=5)
        tab.task_thread = worker

        assert server._get_running_task_ids() == set()

    def test_overlay_live_metrics_prefers_agent_last_task_id(self) -> None:
        server, _ = _make_server()
        tab = server._get_tab("tab-overlay")
        agent = tab.agent
        assert agent is not None
        agent._last_task_id = "7"
        tab.task_history_id = "42"
        agent.total_tokens_used = 555
        agent.budget_used = 1.25
        agent.total_steps = 9
        agent.model_name = "test-model"
        tab.use_worktree = True
        tab.use_parallel = False
        tab.auto_commit_mode = False

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

        # The fallback id must NOT match while the agent id is set.
        unmatched: dict = {"tokens": 0, "cost": 0.0, "steps": 0}
        server._overlay_live_metrics(unmatched, "42")
        assert unmatched == {"tokens": 0, "cost": 0.0, "steps": 0}

    def test_overlay_live_metrics_falls_back_to_task_history_id(self) -> None:
        server, _ = _make_server()
        tab = server._get_tab("tab-fallback")
        agent = tab.agent
        assert agent is not None
        agent._last_task_id = None
        tab.task_history_id = "42"
        agent.total_tokens_used = 100
        agent.budget_used = 0.5
        agent.total_steps = 4
        agent.model_name = "fallback-model"
        tab.use_worktree = False
        tab.use_parallel = True
        tab.auto_commit_mode = False

        session: dict = {"tokens": 0, "cost": 0.0, "steps": 0}
        server._overlay_live_metrics(session, "42")

        assert session == {
            "tokens": 100,
            "cost": 0.5,
            "steps": 4,
            "model": "fallback-model",
            "is_worktree": False,
            "is_parallel": True,
            "auto_commit_mode": False,
        }


class TestDeferredTabDisposal(_DbTestBase):
    """A3: the busy predicate defers disposal; idle close disposes."""

    def test_close_during_active_task_keeps_state(self) -> None:
        server, _ = _make_server()
        tab = server._get_tab("tab-busy")
        tab.is_task_active = True

        server._handle_command({"type": "closeTab", "tabId": "tab-busy"})

        assert tab.frontend_closed is True
        assert "tab-busy" in _RunningAgentState.running_agent_states

    def test_close_during_merge_keeps_state(self) -> None:
        server, _ = _make_server()
        tab = server._get_tab("tab-merging")
        tab.is_merging = True

        server._handle_command({"type": "closeTab", "tabId": "tab-merging"})

        assert tab.frontend_closed is True
        assert "tab-merging" in _RunningAgentState.running_agent_states

    def test_close_with_live_thread_keeps_state(self) -> None:
        server, _ = _make_server()
        tab = server._get_tab("tab-thread")
        release = threading.Event()
        worker = threading.Thread(target=release.wait, daemon=True)
        worker.start()
        tab.task_thread = worker
        try:
            server._handle_command({"type": "closeTab", "tabId": "tab-thread"})

            assert tab.frontend_closed is True
            assert "tab-thread" in _RunningAgentState.running_agent_states
        finally:
            release.set()
            worker.join(timeout=5)

    def test_close_idle_tab_disposes_immediately(self) -> None:
        server, _ = _make_server()
        server._get_tab("tab-idle")

        server._handle_command({"type": "closeTab", "tabId": "tab-idle"})

        assert "tab-idle" not in _RunningAgentState.running_agent_states

    def test_dispose_if_closed_after_task_end(self) -> None:
        server, _ = _make_server()
        tab = server._get_tab("tab-deferred")
        tab.is_task_active = True
        server._handle_command({"type": "closeTab", "tabId": "tab-deferred"})
        assert "tab-deferred" in _RunningAgentState.running_agent_states

        tab.is_task_active = False
        server._dispose_if_closed("tab-deferred")

        assert "tab-deferred" not in _RunningAgentState.running_agent_states

    def test_dispose_if_closed_noop_when_frontend_open(self) -> None:
        server, _ = _make_server()
        server._get_tab("tab-open")

        server._dispose_if_closed("tab-open")

        assert "tab-open" in _RunningAgentState.running_agent_states


class TestMergeFlowDelegates(_DbTestBase):
    """C1/C2: thin merge-flow delegates keep their exact semantics."""

    def test_ensure_wt_agent_returns_tab_agent(self) -> None:
        server, _ = _make_server()
        tab = server._get_tab("tab-wt-agent")

        assert server._ensure_wt_agent(tab) is tab.agent

        tab.agent = None
        assert server._ensure_wt_agent(tab) is None

    def test_emit_pending_worktree_noop_without_use_worktree(self) -> None:
        server, events = _make_server()
        tab = server._get_tab("tab-no-worktree")
        assert tab.use_worktree is False
        before = len(events)

        server._emit_pending_worktree("tab-no-worktree")

        assert len(events) == before  # no broadcast, no exception


class TestPersistedSubagentIsDone(_DbTestBase):
    """A5: persisted sub-agent rows replay as done when not running."""

    def test_open_persisted_subagent_tabs_marks_done(self) -> None:
        parent_id, chat_id = th._add_task("parent task")
        sub_id, _ = th._add_task(
            "sub task",
            chat_id=chat_id,
            extra={"subagent": {"parent_task_id": parent_id}},
        )
        assert sub_id not in ChatSorcarAgent.running_agents
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

