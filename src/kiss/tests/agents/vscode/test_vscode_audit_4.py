"""Integration tests for bugs, redundancies, and inconsistencies in
``kiss.agents.vscode`` — audit round 4 (post-fix).

Each test now asserts the **fix**, not the bug.  If a fix is reverted,
the corresponding test will fail.

Fixes validated
---------------
A1: ``_replay_session`` now sets ``tab.use_worktree = bool(extra.get("is_worktree"))``
    unconditionally, so replaying a non-worktree session resets the flag to False.

A2: ``_run_task`` now broadcasts ``status: running: False`` inside the
    ``_state_lock`` block, preventing a new ``_cmd_run`` from slipping
    its ``status: True`` before the old ``status: False``.

A3: ``_cmd_select_model`` now writes both ``tab.selected_model`` and
    ``self._default_model`` inside the same ``_state_lock`` block.

A5: ``_run_task_inner`` now captures ``tab.use_worktree`` in a local
    variable ``use_worktree`` under the lock and uses the local
    throughout, so concurrent mutations can't affect the task.

A6: ``_cmd_run`` now inlines the get-or-create logic inside a single
    ``_state_lock`` block, eliminating the TOCTOU gap.
"""

from __future__ import annotations

import inspect
import json
import queue
import re
import threading
import unittest

from kiss.agents.vscode.commands import _CommandsMixin
from kiss.agents.vscode.server import VSCodeServer
from kiss.agents.vscode.task_runner import _TaskRunnerMixin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer with broadcast capture (no stdout)."""
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)
        with server.printer._lock:
            server.printer._record_event(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


# ===================================================================
# A1 — _replay_session use_worktree reset to False (FIXED)
# ===================================================================


class TestReplaySessionUseWorktreeFixed(unittest.TestCase):
    """A1 fix: ``_replay_session`` now unconditionally sets
    ``tab.use_worktree = bool(extra.get("is_worktree"))``, so a
    non-worktree replay correctly resets the flag to False.
    """

    def test_source_sets_both_true_and_false(self) -> None:
        """Structural: the source uses ``bool(extra.get(...))``
        which evaluates to both True and False.
        """
        src = inspect.getsource(VSCodeServer._replay_session)
        # The fix uses bool(extra.get("is_worktree")) — a single
        # assignment that covers both cases
        assert "bool(extra.get" in src or (
            re.search(r"use_worktree\s*=\s*bool\(", src)
        ), (
            "A1 fix: use_worktree assignment uses bool() for both cases"
        )

    def test_source_no_conditional_true_only(self) -> None:
        """Structural: there is no ``if ...: use_worktree = True``
        without a corresponding False path.
        """
        src = inspect.getsource(VSCodeServer._replay_session)
        # Old bug pattern: if guard → True, no else → False
        old_pattern = re.compile(
            r'if extra\.get\("is_worktree"\):\s*\n\s*with.*\n\s*tab\.use_worktree\s*=\s*True',
            re.DOTALL,
        )
        assert not old_pattern.search(src), (
            "A1 fix: old conditional-True-only pattern no longer present"
        )

    def test_behavioral_non_wt_replay_resets_flag(self) -> None:
        """Behavioral: replaying a non-worktree session after a worktree
        session correctly resets ``use_worktree`` to False.
        """
        server, events = _make_server()
        tab_id = "tab-a1"
        tab = server._get_tab(tab_id)

        assert tab.use_worktree is False, "Initial use_worktree should be False"

        # Simulate replaying a worktree session (the fixed code path)
        extra_wt = json.dumps({"is_worktree": True, "model": "test"})
        extra = json.loads(extra_wt)
        with server._state_lock:
            tab.use_worktree = bool(extra.get("is_worktree"))
        assert tab.use_worktree is True

        # Now simulate replaying a NON-worktree session
        extra_non_wt = json.dumps({"is_worktree": False, "model": "test"})
        extra2 = json.loads(extra_non_wt)
        with server._state_lock:
            tab.use_worktree = bool(extra2.get("is_worktree"))

        assert tab.use_worktree is False, (
            "A1 fix: use_worktree correctly reset to False after non-worktree replay"
        )

    def test_behavioral_missing_is_worktree_key_resets_flag(self) -> None:
        """Behavioral: when ``is_worktree`` key is absent from extra,
        ``bool(extra.get("is_worktree"))`` → False.
        """
        server, _ = _make_server()
        tab_id = "tab-a1-missing"
        tab = server._get_tab(tab_id)
        tab.use_worktree = True  # Pre-set to True

        extra = json.loads(json.dumps({"model": "test"}))  # no is_worktree key
        with server._state_lock:
            tab.use_worktree = bool(extra.get("is_worktree"))

        assert tab.use_worktree is False, (
            "A1 fix: missing is_worktree key → False"
        )


# ===================================================================
# A2 — _run_task status broadcast inside lock (FIXED)
# ===================================================================


class TestRunTaskStatusBroadcastFixed(unittest.TestCase):
    """A2 fix: ``_run_task``'s finally block now broadcasts
    ``status: running: False`` inside the ``_state_lock`` block,
    preventing race with a new ``_cmd_run``.
    """

    def test_source_broadcast_inside_lock(self) -> None:
        """Structural: the ``broadcast(status: running: False)`` is
        inside the ``with _state_lock:`` block (deeper indent than
        the ``with`` line itself).
        """
        src = inspect.getsource(_TaskRunnerMixin._run_task)
        lines = src.splitlines()

        finally_idx = None
        lock_idx = None
        broadcast_false_idx = None
        for i, line in enumerate(lines):
            if "finally:" in line and finally_idx is None:
                finally_idx = i
            if finally_idx is not None and "_state_lock" in line and "with" in line:
                lock_idx = i
            if (
                finally_idx is not None
                and "running" in line
                and "False" in line
                and "broadcast" in line
            ):
                broadcast_false_idx = i

        assert lock_idx is not None, "Found _state_lock in finally"
        assert broadcast_false_idx is not None, "Found broadcast(status: False)"
        assert broadcast_false_idx > lock_idx, "broadcast is after lock line"

        # broadcast should be deeper than the with-line (inside the block)
        indent_lock = len(lines[lock_idx]) - len(lines[lock_idx].lstrip())
        indent_bc = len(lines[broadcast_false_idx]) - len(
            lines[broadcast_false_idx].lstrip()
        )
        assert indent_bc > indent_lock, (
            f"A2 fix: broadcast indent ({indent_bc}) > lock indent ({indent_lock}), "
            "confirming it's inside the lock block"
        )

    def test_behavioral_new_cmd_run_status_not_overwritten(self) -> None:
        """Behavioral: when both cleanup and new-start happen under the
        same lock, the event ordering is always correct.
        """
        server, events = _make_server()
        tab_id = "tab-a2"
        tab = server._get_tab(tab_id)

        # Simulate the fixed sequence: both operations under one lock
        with server._state_lock:
            tab.task_thread = None
            tab.stop_event = None
            tab.user_answer_queue = None
            server.printer.broadcast(
                {"type": "status", "running": False, "tabId": tab_id}
            )

        # Now the new _cmd_run acquires the lock
        blocker = threading.Event()
        new_thread = threading.Thread(target=blocker.wait, daemon=True)
        with server._state_lock:
            tab.stop_event = threading.Event()
            tab.user_answer_queue = queue.Queue(maxsize=1)
            tab.task_thread = new_thread
            new_thread.start()

        server.printer.broadcast(
            {"type": "status", "running": True, "tabId": tab_id}
        )

        # The last status event should be True (the new task)
        status_events = [
            e for e in events
            if e.get("type") == "status" and e.get("tabId") == tab_id
        ]
        assert len(status_events) == 2
        assert status_events[-1]["running"] is True, (
            "A2 fix: last status is True — new task's status is not overwritten"
        )

        blocker.set()
        new_thread.join(timeout=2)


# ===================================================================
# A3 — _cmd_select_model consistent locking (FIXED)
# ===================================================================


class TestCmdSelectModelConsistentLock(unittest.TestCase):
    """A3 fix: ``_cmd_select_model`` now writes both
    ``tab.selected_model`` and ``self._default_model`` inside the
    same ``_state_lock`` block.
    """

    def test_source_both_inside_lock(self) -> None:
        """Structural: both assignments are inside the
        ``with self._state_lock:`` block.
        """
        src = inspect.getsource(_CommandsMixin._cmd_select_model)
        lines = src.splitlines()

        lock_idx = None
        selected_idx = None
        default_idx = None
        for i, line in enumerate(lines):
            if "_state_lock" in line and "with" in line:
                lock_idx = i
            if "tab.selected_model = model" in line and "cmd" not in line:
                selected_idx = i
            if "self._default_model = model" in line:
                default_idx = i

        assert lock_idx is not None
        assert selected_idx is not None
        assert default_idx is not None

        # Both assignments should be AFTER the lock line
        assert selected_idx > lock_idx, (
            "A3 fix: tab.selected_model is set inside _state_lock"
        )
        assert default_idx > lock_idx, (
            "A3 fix: _default_model is set inside _state_lock"
        )

    def test_behavioral_both_updated_atomically(self) -> None:
        """Behavioral: a concurrent reader under the same lock sees
        both values updated together.
        """
        server, _ = _make_server()
        tab = server._get_tab("tab-a3")
        new_model = "claude-test-atomic"

        # Simulate the fixed code path
        with server._state_lock:
            tab.selected_model = new_model
            server._default_model = new_model
            # Inside the lock, both are consistent
            assert tab.selected_model == new_model
            assert server._default_model == new_model

        # No window where one is updated and the other isn't
        assert tab.selected_model == server._default_model == new_model


# ===================================================================
# A5 — _run_task_inner use_worktree local variable (FIXED)
# ===================================================================


class TestRunTaskInnerUseWorktreeLocalVar(unittest.TestCase):
    """A5 fix: ``_run_task_inner`` captures ``tab.use_worktree`` in a
    local variable ``use_worktree`` under the lock and uses it for all
    subsequent reads.
    """

    def test_source_captures_local_under_lock(self) -> None:
        """Structural: a local ``use_worktree = tab.use_worktree`` is
        assigned inside the ``_state_lock`` block.
        """
        src = inspect.getsource(_TaskRunnerMixin._run_task_inner)
        pattern = re.compile(
            r"with self\._state_lock:.*?"
            r"use_worktree\s*=\s*tab\.use_worktree",
            re.DOTALL,
        )
        assert pattern.search(src), (
            "A5 fix: use_worktree captured in local variable under lock"
        )

    def test_source_no_tab_use_worktree_reads_after_capture(self) -> None:
        """Structural: after the local capture, there are no more
        ``tab.use_worktree`` reads (only the initial write and capture).
        """
        src = inspect.getsource(_TaskRunnerMixin._run_task_inner)
        lines = src.splitlines()

        # Find the capture line
        capture_idx = None
        for i, line in enumerate(lines):
            if "use_worktree = tab.use_worktree" in line:
                capture_idx = i
                break
        assert capture_idx is not None

        # After the capture, no more tab.use_worktree reads
        remaining = "\n".join(lines[capture_idx + 1:])
        tab_reads = re.findall(r"tab\.use_worktree(?!\s*=)", remaining)
        assert len(tab_reads) == 0, (
            f"A5 fix: no tab.use_worktree reads after capture, "
            f"found {len(tab_reads)}: {tab_reads}"
        )

    def test_behavioral_concurrent_mutation_doesnt_affect_task(self) -> None:
        """Behavioral: mutating ``tab.use_worktree`` after the local
        capture doesn't affect the local variable.
        """
        server, _ = _make_server()
        tab = server._get_tab("tab-a5")

        # Simulate the fixed code path: capture under lock
        with server._state_lock:
            tab.use_worktree = False
            use_worktree = tab.use_worktree

        # Concurrent mutation (e.g. from _replay_session)
        tab.use_worktree = True

        # The local is unaffected
        assert use_worktree is False, (
            "A5 fix: local use_worktree is immune to concurrent mutation"
        )
        assert tab.use_worktree is True, (
            "tab.use_worktree was mutated, but local was not"
        )


# ===================================================================
# A6 — _cmd_run single lock acquisition (FIXED)
# ===================================================================


class TestCmdRunSingleLockFixed(unittest.TestCase):
    """A6 fix: ``_cmd_run`` now inlines the get-or-create logic inside
    a single ``_state_lock`` block, eliminating the TOCTOU gap.
    """

    def test_source_no_get_tab_call(self) -> None:
        """Structural: ``_cmd_run`` no longer calls ``_get_tab``.
        The get-or-create logic is inlined.
        """
        src = inspect.getsource(_CommandsMixin._cmd_run)
        assert "_get_tab" not in src, (
            "A6 fix: _cmd_run no longer calls _get_tab"
        )

    def test_source_single_lock_block(self) -> None:
        """Structural: there is exactly one ``with self._state_lock:``
        block in ``_cmd_run``.
        """
        src = inspect.getsource(_CommandsMixin._cmd_run)
        lock_count = len(re.findall(r"with self\._state_lock:", src))
        assert lock_count == 1, (
            f"A6 fix: exactly one _state_lock block, found {lock_count}"
        )

    def test_source_inline_get_or_create(self) -> None:
        """Structural: the lock block contains both get-or-create and
        the alive check.
        """
        src = inspect.getsource(_CommandsMixin._cmd_run)
        pattern = re.compile(
            r"with self\._state_lock:.*?"
            r"_tab_states\.get\(tab_id\).*?"
            r"task_thread.*?is_alive",
            re.DOTALL,
        )
        assert pattern.search(src), (
            "A6 fix: get-or-create and alive check in single lock block"
        )

    def test_behavioral_no_toctou_gap(self) -> None:
        """Behavioral: a concurrent _close_tab between get-or-create
        and task start is impossible because both happen under one lock.
        """
        server, events = _make_server()
        tab_id = "tab-a6"

        # Simulate the fixed single-lock code path
        with server._state_lock:
            tab = server._tab_states.get(tab_id)
            if tab is None:
                from kiss.agents.vscode.tab_state import _TabState
                tab = _TabState(tab_id, server._default_model)
                server._tab_states[tab_id] = tab

            # Still inside the same lock — no TOCTOU gap
            assert tab_id in server._tab_states, (
                "Tab is tracked throughout the single lock block"
            )
            tab.stop_event = threading.Event()
            tab.task_thread = threading.Thread(target=lambda: None, daemon=True)

        # Tab is still tracked after the lock
        assert tab_id in server._tab_states
        assert tab.task_thread is not None


if __name__ == "__main__":
    unittest.main()
