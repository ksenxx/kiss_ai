# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for two refactoring bugs found during review.

Bug 1 — stale ``cmd["_state_key"]`` after mid-run re-key: the printer
bridge (``agent_task_allocated``) re-keys the run's ``AgentState`` from
the server-minted uuid to the persisted ``task_history`` row id.  The
``finally`` of ``_run_task`` used to re-resolve the state from
``cmd["_state_key"]`` — a miss after the re-key — silently skipping the
whole end-of-run cleanup.  The tab's state then kept its dead
``task_thread`` forever, so every subsequent ``run`` on the tab was
queued as steering for the finished task and dropped.

Bug 2 — ``_cmd_run`` destroyed an in-flight merge: a thread-less
state with ``is_merging=True`` was unregistered and replaced by a fresh
state, orphaning the review.  The run must be refused instead.

Both tests drive the real :class:`VSCodeServer` command dispatch; the
first stubs only the inner task body while performing the exact
registry re-key the printer bridge performs mid-run.
"""

from __future__ import annotations

import threading
import time
import unittest

from kiss.server import agent_state
from kiss.server.server import VSCodeServer


class _Harness(unittest.TestCase):
    """Shared server harness with captured broadcasts."""

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.events: list[dict] = []
        self._evt_lock = threading.Lock()

        def capture(ev: dict) -> None:
            with self._evt_lock:
                self.events.append(ev)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        for state in agent_state.snapshot():
            th = state.task_thread
            if th is not None:
                th.join(timeout=2)
        agent_state.agent_states.clear()

    def _events_of(self, ev_type: str) -> list[dict]:
        with self._evt_lock:
            return [e for e in self.events if e.get("type") == ev_type]


class TestFinallyCleanupSurvivesRekey(_Harness):
    """The end-of-run cleanup must find the state after a re-key."""

    def test_second_run_starts_after_rekeyed_first_run_finishes(self) -> None:
        tab_id = "tab-rekey"
        started = threading.Event()
        release = threading.Event()
        runs: list[dict] = []
        persisted_ids = iter(["task-101", "task-202"])

        def fake_inner(cmd: dict) -> None:
            state = self.server._resolve_run_state(cmd)
            state.is_task_active = True
            # Mirror what JsonPrinter.agent_task_allocated does the
            # moment the task_history row id is allocated mid-run.
            agent_state.rekey(state, next(persisted_ids))
            runs.append(cmd)
            started.set()
            release.wait(timeout=10)

        self.server._run_task_inner = fake_inner  # type: ignore[assignment]

        self.server._handle_command({
            "type": "run",
            "prompt": "first task",
            "model": "test-model",
            "tabId": tab_id,
        })
        assert started.wait(timeout=3), "first task did not start"

        state = agent_state.find_by_tab(tab_id)
        assert state is not None
        assert state.task_id == "task-101", "state must be re-keyed"

        release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if state.task_thread is None:
                break
            time.sleep(0.05)
        assert state.task_thread is None, (
            "BUG: _run_task's finally missed the re-keyed state and "
            "skipped cleanup — task_thread was never cleared"
        )
        assert state.stop_event is None
        assert state.user_answer_queue is None
        assert not state.is_task_active
        assert not state.is_running_non_wt

        status_off = [
            e for e in self._events_of("status") if e.get("running") is False
        ]
        assert status_off, "finish must broadcast status running:false"

        started.clear()
        release.clear()
        self.server._handle_command({
            "type": "run",
            "prompt": "second task",
            "model": "test-model",
            "tabId": tab_id,
        })
        assert started.wait(timeout=3), (
            "BUG: the second run on the tab was swallowed as steering "
            "for the finished task instead of starting a new task"
        )
        assert len(runs) == 2
        release.set()


class TestRunRefusedDuringMerge(_Harness):
    """A run on a tab with a merge in progress must be refused."""

    def test_run_does_not_destroy_merging_state(self) -> None:
        tab_id = "tab-merge"
        review = agent_state.AgentState(
            "task-review",
            chat_id="chat-review",
            tab_id=tab_id,
            server_owned=True,
        )
        review.is_merging = True
        agent_state.register(review)

        ran = threading.Event()

        def fake_inner(cmd: dict) -> None:
            ran.set()

        self.server._run_task_inner = fake_inner  # type: ignore[assignment]

        self.server._handle_command({
            "type": "run",
            "prompt": "should be refused",
            "model": "test-model",
            "tabId": tab_id,
        })

        errors = [
            e
            for e in self._events_of("error")
            if "while a merge is in progress" in str(e.get("text", ""))
        ]
        assert errors, (
            "BUG: run during a merge must broadcast the "
            "merge-in-progress error"
        )
        assert not ran.wait(timeout=0.5), (
            "BUG: run during a merge must not start a task"
        )
        assert agent_state.get("task-review") is review, (
            "BUG: _cmd_run unregistered the merging tab's state"
        )
        assert review.is_merging


if __name__ == "__main__":
    unittest.main()


class TestFinishedBridgeLeavesServerOwnedActive(unittest.TestCase):
    """``agent_task_finished`` must not deactivate server-owned states.

    The task runner still does persistence / autocommit / worktree
    post-processing after ``ChatSorcarAgent.run`` returns; flipping
    ``is_task_active`` in the bridge would let a concurrent
    merge/discard race that post-processing (the old per-tab registry
    kept the tab active until the runner's own lifecycle ended).
    """

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_server_owned_state_stays_active_and_registered(self) -> None:
        from kiss.server.json_printer import JsonPrinter

        printer = JsonPrinter()
        agent = object()
        state = agent_state.AgentState(
            "task-po",
            agent=agent,  # type: ignore[arg-type]
            tab_id="tab-po",
            server_owned=True,
            is_task_active=True,
        )
        agent_state.register(state)

        printer.agent_task_finished(agent, "task-po")

        assert state.is_task_active, (
            "BUG: the bridge deactivated a server-owned state while "
            "the task runner still owns its lifecycle"
        )
        assert agent_state.get("task-po") is state

    def test_bridge_owned_state_is_deactivated_and_removed(self) -> None:
        from kiss.server.json_printer import JsonPrinter

        printer = JsonPrinter()
        agent = object()
        state = agent_state.AgentState(
            "task-sub",
            agent=agent,  # type: ignore[arg-type]
            parent_task_id="task-parent",
            is_task_active=True,
        )
        agent_state.register(state)

        printer.agent_task_finished(agent, "task-sub")

        assert not state.is_task_active
        assert agent_state.get("task-sub") is None
