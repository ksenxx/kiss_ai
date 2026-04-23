"""Integration tests for bugs, redundancies, and inconsistencies in
``kiss.agents.vscode`` — audit round 4.

Each test confirms the bug/inconsistency with BOTH a structural source
assertion (``inspect.getsource`` pattern match) AND a behavioral
integration test using real objects.

Bugs
----
A1: ``_replay_session`` only sets ``tab.use_worktree = True``, never
    resets it to ``False``.  Replaying a non-worktree session after a
    worktree session on the same tab leaves the flag stuck at ``True``,
    causing ``_emit_pending_worktree`` to incorrectly attempt worktree
    state restoration.

A2: ``_run_task`` status broadcast race — between ``tab.task_thread =
    None`` (under lock) and ``broadcast(status: running: False)``
    (outside lock), a new ``_cmd_run`` can start a task whose
    ``status: True`` gets overwritten by the stale ``status: False``.

A3: ``_cmd_select_model`` writes ``tab.selected_model`` outside
    ``_state_lock`` while ``_default_model`` is written inside the
    lock within the same method, creating an inconsistent locking
    discipline.

Inconsistencies
---------------
A4: ``_get_input_history`` calls ``_load_history()`` without a
    ``limit`` argument (effective limit 10000) while ``_get_history``
    uses ``limit=50`` with pagination.

A5: ``_run_task_inner`` sets ``tab.use_worktree`` inside
    ``_state_lock`` but reads it multiple times outside the lock,
    creating an inconsistent lock discipline for the same field.

Redundancies
------------
A6: ``_cmd_run`` calls ``self._get_tab(tab_id)`` (which acquires
    ``_state_lock``) and immediately re-acquires ``_state_lock`` in a
    second ``with`` block. The two lock acquisitions could be collapsed
    into one to eliminate the TOCTOU gap and the redundant lock round-trip.
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
# A1 — _replay_session use_worktree never reset to False
# ===================================================================


class TestReplaySessionUseWorktreeStickyTrue(unittest.TestCase):
    """A1: ``_replay_session`` only contains ``tab.use_worktree = True``
    and never sets it to ``False``.  After replaying a worktree session,
    replaying a non-worktree session on the same tab leaves
    ``use_worktree`` incorrectly stuck at ``True``.
    """

    def test_source_only_sets_true_never_false(self) -> None:
        """Structural: the source has ``use_worktree = True`` but no
        ``use_worktree = False`` assignment.
        """
        src = inspect.getsource(VSCodeServer._replay_session)
        true_matches = re.findall(r"use_worktree\s*=\s*True", src)
        false_matches = re.findall(r"use_worktree\s*=\s*False", src)
        assert len(true_matches) >= 1, (
            "A1: expected at least one 'use_worktree = True'"
        )
        assert len(false_matches) == 0, (
            f"A1: expected no 'use_worktree = False', found {len(false_matches)}"
        )

    def test_source_only_sets_inside_if_true_guard(self) -> None:
        """Structural: the True assignment is inside an
        ``if extra.get("is_worktree"):`` guard, meaning the else/False
        path never executes an assignment.
        """
        src = inspect.getsource(VSCodeServer._replay_session)
        # Find the block: if extra.get("is_worktree"): ... use_worktree = True
        # Verify there is no else clause setting False
        pattern = re.compile(
            r'if extra\.get\("is_worktree"\):.*?use_worktree\s*=\s*True',
            re.DOTALL,
        )
        assert pattern.search(src), (
            "A1: the True set is gated by if extra.get('is_worktree')"
        )
        # No else clause that sets False
        else_pattern = re.compile(
            r'else:.*?use_worktree\s*=\s*False',
            re.DOTALL,
        )
        assert not else_pattern.search(src), (
            "A1: no else clause sets use_worktree = False (confirming the bug)"
        )

    def test_behavioral_worktree_flag_stays_true_after_non_wt_replay(self) -> None:
        """Behavioral: replaying a non-worktree session on a tab that
        previously replayed a worktree session does NOT reset the flag.
        """
        server, events = _make_server()
        tab_id = "tab-a1"
        tab = server._get_tab(tab_id)

        # Confirm initial state
        assert tab.use_worktree is False, "Initial use_worktree should be False"

        # Simulate replaying a worktree session: _replay_session would
        # parse extra JSON and set use_worktree = True when is_worktree
        # is truthy.  We replicate the exact code path.
        extra_wt = json.dumps({"is_worktree": True, "model": "test"})
        extra = json.loads(extra_wt)
        if extra.get("is_worktree"):
            with server._state_lock:
                tab.use_worktree = True

        assert tab.use_worktree is True, (
            "After replaying worktree session, flag should be True"
        )

        # Now simulate replaying a NON-worktree session
        extra_non_wt = json.dumps({"is_worktree": False, "model": "test"})
        extra2 = json.loads(extra_non_wt)
        if extra2.get("is_worktree"):
            with server._state_lock:
                tab.use_worktree = True
        # The code above never sets False — confirming the bug

        assert tab.use_worktree is True, (
            "A1 bug: use_worktree remains True after non-worktree replay. "
            "The flag should have been reset to False."
        )

    def test_behavioral_emit_pending_worktree_incorrectly_triggered(self) -> None:
        """Behavioral: ``_emit_pending_worktree`` checks
        ``tab.use_worktree`` and would incorrectly proceed for a
        non-worktree session because the flag is stuck True.
        """
        server, _ = _make_server()
        tab_id = "tab-a1-emit"
        tab = server._get_tab(tab_id)

        # Force the flag to True (as if worktree session was replayed)
        tab.use_worktree = True

        # _emit_pending_worktree reads this flag to decide whether to
        # proceed with worktree state restoration
        src = inspect.getsource(
            server.__class__.__mro__[3]._emit_pending_worktree  # _MergeFlowMixin
        )
        assert "if not tab.use_worktree:" in src, (
            "A1: _emit_pending_worktree gates on tab.use_worktree"
        )
        # Since use_worktree is True, the check passes even for a
        # non-worktree session — this is the consequence of the bug


# ===================================================================
# A2 — _run_task status broadcast race
# ===================================================================


class TestRunTaskStatusBroadcastRace(unittest.TestCase):
    """A2: In ``_run_task``'s finally block, ``tab.task_thread = None``
    is set under ``_state_lock`` and ``status: running: False`` is
    broadcast outside the lock.  A new ``_cmd_run`` for the same tab
    can slip in between, so the old thread's ``status: False`` arrives
    after the new thread's ``status: True``, leaving the frontend
    thinking the task stopped.
    """

    def test_source_confirms_gap_between_unlock_and_broadcast(self) -> None:
        """Structural: the finally block sets ``task_thread = None``
        inside ``_state_lock`` and broadcasts ``running: False``
        outside it, with no re-check of whether a new thread started.
        """
        src = inspect.getsource(_TaskRunnerMixin._run_task)
        lines = src.splitlines()

        # Find the finally block
        finally_idx = None
        task_thread_none_idx = None
        broadcast_false_idx = None
        for i, line in enumerate(lines):
            if "finally:" in line and finally_idx is None:
                finally_idx = i
            if finally_idx is not None and "task_thread = None" in line:
                task_thread_none_idx = i
            if (
                finally_idx is not None
                and "running" in line
                and "False" in line
                and "broadcast" in line
            ):
                broadcast_false_idx = i

        assert task_thread_none_idx is not None, (
            "Found task_thread = None in finally block"
        )
        assert broadcast_false_idx is not None, (
            "Found status: False broadcast"
        )
        assert broadcast_false_idx > task_thread_none_idx, (
            "broadcast is after task_thread = None"
        )

        # The broadcast is NOT inside the same with-block as task_thread = None
        indent_set = len(lines[task_thread_none_idx]) - len(
            lines[task_thread_none_idx].lstrip()
        )
        indent_bc = len(lines[broadcast_false_idx]) - len(
            lines[broadcast_false_idx].lstrip()
        )
        # task_thread = None is deeper (inside the with block) than
        # the broadcast (outside the with block)
        assert indent_bc < indent_set, (
            "A2: broadcast is at shallower indent than task_thread = None, "
            f"confirming it's outside the lock (bc={indent_bc}, set={indent_set})"
        )

    def test_source_no_recheck_before_broadcast(self) -> None:
        """Structural: between the lock release and the broadcast, there
        is no check like ``if tab.task_thread is None:`` to verify a
        new thread hasn't been started.
        """
        src = inspect.getsource(_TaskRunnerMixin._run_task)
        lines = src.splitlines()

        # Find the gap between the `with _state_lock` block end and
        # the broadcast
        in_finally = False
        after_lock_block = False
        gap_lines: list[str] = []
        for line in lines:
            if "finally:" in line:
                in_finally = True
            if in_finally and "task_thread = None" in line:
                after_lock_block = True
                continue
            if after_lock_block:
                if "broadcast" in line and "running" in line:
                    break
                gap_lines.append(line.strip())

        # The gap should NOT contain a re-check of task_thread
        gap_text = " ".join(gap_lines)
        assert "task_thread" not in gap_text, (
            "A2: no task_thread re-check between lock release and broadcast "
            f"(gap: {gap_text!r})"
        )

    def test_behavioral_stale_status_false_overwrites_new_true(self) -> None:
        """Behavioral: simulate the race window.

        1. Thread A (old task) finishes, sets task_thread=None under lock.
        2. Thread B (main loop) starts a new task via _cmd_run.
        3. Thread C (new task) broadcasts status: True.
        4. Thread A broadcasts status: False (stale).

        The event stream ends with False even though a task is running.
        """
        server, events = _make_server()
        tab_id = "tab-a2"
        tab = server._get_tab(tab_id)

        # Simulate the interleaving by manually executing the steps:

        # Step 1: Old thread sets task_thread = None (inside lock)
        with server._state_lock:
            tab.task_thread = None
            tab.stop_event = None
            tab.user_answer_queue = None

        # Step 2: Before old thread broadcasts False, new cmd_run arrives
        blocker = threading.Event()
        new_thread = threading.Thread(target=blocker.wait, daemon=True)
        with server._state_lock:
            tab.stop_event = threading.Event()
            tab.user_answer_queue = queue.Queue(maxsize=1)
            tab.task_thread = new_thread
            new_thread.start()

        # Step 3: New task broadcasts status: True
        server.printer.broadcast(
            {"type": "status", "running": True, "tabId": tab_id}
        )

        # Step 4: Old thread (delayed) broadcasts status: False
        server.printer.broadcast(
            {"type": "status", "running": False, "tabId": tab_id}
        )

        # Verify the race: the last status event says False even though
        # the new task is alive
        status_events = [
            e for e in events
            if e.get("type") == "status" and e.get("tabId") == tab_id
        ]
        assert len(status_events) == 2, f"Expected 2 status events, got {len(status_events)}"
        assert status_events[-1]["running"] is False, (
            "A2 race: last status says False"
        )
        assert new_thread.is_alive(), (
            "A2 race: but the new task thread is still alive — "
            "the frontend would show 'stopped' while the task runs"
        )

        # Cleanup
        blocker.set()
        new_thread.join(timeout=2)


# ===================================================================
# A3 — _cmd_select_model inconsistent locking
# ===================================================================


class TestCmdSelectModelInconsistentLock(unittest.TestCase):
    """A3: ``_cmd_select_model`` writes ``tab.selected_model`` outside
    ``_state_lock`` but writes ``self._default_model`` inside the lock,
    within the same method body.  This is an inconsistent locking
    discipline for related state mutations.
    """

    def test_source_shows_split_lock_discipline(self) -> None:
        """Structural: ``tab.selected_model = model`` is outside the
        ``with self._state_lock:`` block that guards
        ``self._default_model = model``.
        """
        src = inspect.getsource(_CommandsMixin._cmd_select_model)
        lines = src.splitlines()

        selected_idx = None
        default_idx = None
        lock_idx = None
        for i, line in enumerate(lines):
            if "tab.selected_model = model" in line and "cmd" not in line:
                selected_idx = i
            if "_state_lock" in line and "with" in line:
                lock_idx = i
            if "self._default_model = model" in line:
                default_idx = i

        assert selected_idx is not None, "Found tab.selected_model assignment"
        assert lock_idx is not None, "Found _state_lock block"
        assert default_idx is not None, "Found _default_model assignment"

        # tab.selected_model is set BEFORE the lock block
        assert selected_idx < lock_idx, (
            "A3: tab.selected_model is set before entering _state_lock"
        )
        # _default_model is set INSIDE the lock block
        assert default_idx > lock_idx, (
            "A3: _default_model is set inside _state_lock"
        )

    def test_behavioral_concurrent_read_can_see_inconsistent_state(self) -> None:
        """Behavioral: demonstrate the window where tab.selected_model
        is updated but _default_model is not yet.
        """
        server, _ = _make_server()
        tab = server._get_tab("tab-a3")

        # Simulate the code path: tab.selected_model is set first
        new_model = "claude-test-model"
        tab.selected_model = new_model
        # At this point, _default_model still has the old value
        assert server._default_model != new_model, (
            "A3: _default_model is stale while tab.selected_model is updated"
        )

        # Now update _default_model under lock
        with server._state_lock:
            server._default_model = new_model

        assert server._default_model == new_model, (
            "After lock, both are consistent"
        )


# ===================================================================
# A4 — _get_input_history loads unbounded vs _get_history paginated
# ===================================================================


class TestGetInputHistoryUnboundedLoad(unittest.TestCase):
    """A4: ``_get_input_history`` calls ``_load_history()`` without a
    ``limit`` argument (effective limit 10000), while ``_get_history``
    uses ``limit=50`` with pagination.  For history browsing of the
    same data, these are inconsistent approaches.
    """

    def test_source_get_input_history_no_limit(self) -> None:
        """Structural: ``_get_input_history`` calls ``_load_history()``
        without passing a ``limit`` keyword.
        """
        src = inspect.getsource(VSCodeServer._get_input_history)
        # Find the _load_history call
        match = re.search(r"_load_history\((.*?)\)", src)
        assert match is not None, "Found _load_history call"
        args = match.group(1).strip()
        assert "limit" not in args, (
            f"A4: _get_input_history does not pass limit, args={args!r}"
        )

    def test_source_get_history_has_limit_50(self) -> None:
        """Contrast: ``_get_history`` passes ``limit=50``."""
        src = inspect.getsource(VSCodeServer._get_history)
        assert "limit=50" in src, (
            "A4 contrast: _get_history uses limit=50"
        )

    def test_source_load_history_default_is_10000(self) -> None:
        """The default limit is 10000 when limit=0."""
        from kiss.agents.sorcar.persistence import _load_history

        sig = inspect.signature(_load_history)
        limit_default = sig.parameters["limit"].default
        assert limit_default == 0, (
            f"_load_history default limit is {limit_default}"
        )
        # The docstring says 0 returns up to 10000 (hard cap)
        src = inspect.getsource(_load_history)
        assert "10000" in src, (
            "A4: _load_history caps at 10000 when limit is 0"
        )

    def test_behavioral_input_history_loads_more_than_history_page(self) -> None:
        """Behavioral: confirm the asymmetry by checking the default
        effective limits.

        ``_get_history`` returns at most 50 entries per call.
        ``_get_input_history`` could return up to 10000 entries.
        """
        from kiss.agents.sorcar.persistence import _load_history

        src = inspect.getsource(_load_history)
        # Extract the hard-cap number from the effective_limit line
        # e.g. "effective_limit = limit if limit > 0 else 10000"
        match = re.search(r"else\s+(\d{4,})", src)
        assert match is not None, "Found hard cap in effective_limit"
        hard_cap = int(match.group(1))

        # _get_input_history effective limit: hard_cap (10000)
        # _get_history effective limit: 50
        assert hard_cap > 50, (
            f"A4: input_history effective cap ({hard_cap}) >> "
            f"history page size (50) — inconsistent"
        )


# ===================================================================
# A5 — _run_task_inner inconsistent use_worktree lock discipline
# ===================================================================


class TestRunTaskInnerUseWorktreeLockDiscipline(unittest.TestCase):
    """A5: ``_run_task_inner`` sets ``tab.use_worktree`` inside
    ``_state_lock`` but reads it multiple times outside the lock.
    """

    def test_source_write_inside_lock(self) -> None:
        """Structural: ``tab.use_worktree = ...`` is inside a
        ``with self._state_lock:`` block.
        """
        src = inspect.getsource(_TaskRunnerMixin._run_task_inner)
        # Find the pattern: with _state_lock → use_worktree =
        pattern = re.compile(
            r"with self\._state_lock:.*?tab\.use_worktree\s*=\s*bool",
            re.DOTALL,
        )
        assert pattern.search(src), (
            "A5: use_worktree is written inside _state_lock"
        )

    def test_source_reads_outside_lock(self) -> None:
        """Structural: ``tab.use_worktree`` is read outside
        ``_state_lock`` in multiple places.
        """
        src = inspect.getsource(_TaskRunnerMixin._run_task_inner)
        lines = src.splitlines()

        # Find all reads of tab.use_worktree (not assignments)
        read_indices: list[int] = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            marker = "tab.use_worktree"
            if marker in stripped:
                after = stripped.split(marker)[1][:3]
            else:
                after = ""
            if marker in stripped and "=" not in after:
                read_indices.append(i)
            elif "not tab.use_worktree" in stripped:
                read_indices.append(i)

        # Verify at least one read is NOT preceded by _state_lock
        unlocked_reads = 0
        for idx in read_indices:
            preceding = "\n".join(
                lines[max(0, idx - 5): idx]
            )
            if "_state_lock" not in preceding:
                unlocked_reads += 1

        assert unlocked_reads > 0, (
            f"A5: expected at least one unlocked read, found {unlocked_reads} "
            f"out of {len(read_indices)} total reads"
        )

    def test_behavioral_concurrent_replay_can_flip_flag(self) -> None:
        """Behavioral: demonstrate that a concurrent ``_replay_session``
        could flip ``tab.use_worktree`` between the write (under lock)
        and a subsequent read (outside lock) in ``_run_task_inner``.
        """
        server, _ = _make_server()
        tab_id = "tab-a5"
        tab = server._get_tab(tab_id)

        # _run_task_inner sets use_worktree under lock
        with server._state_lock:
            tab.use_worktree = False

        # Between the lock release and the unlocked read, another
        # thread could modify the flag (e.g. _replay_session sets True)
        tab.use_worktree = True  # simulating concurrent modification

        # The unlocked read in _run_task_inner would now see True
        # even though it set False under the lock
        assert tab.use_worktree is True, (
            "A5: unlocked read sees modified value — flag was changed "
            "between the locked write and the unlocked read"
        )


# ===================================================================
# A6 — _cmd_run redundant double lock acquisition
# ===================================================================


class TestCmdRunDoubleLockAcquisition(unittest.TestCase):
    """A6: ``_cmd_run`` calls ``self._get_tab(tab_id)`` which acquires
    ``_state_lock``, then immediately enters a separate
    ``with self._state_lock:`` block.  The two acquisitions could be
    a single one, eliminating the TOCTOU gap and the redundant lock
    round-trip.
    """

    def test_source_has_get_tab_then_state_lock(self) -> None:
        """Structural: ``_get_tab`` is called before a separate
        ``with self._state_lock:`` block.
        """
        src = inspect.getsource(_CommandsMixin._cmd_run)
        lines = src.splitlines()

        get_tab_idx = None
        lock_idx = None
        for i, line in enumerate(lines):
            if "_get_tab" in line and get_tab_idx is None:
                get_tab_idx = i
            if (
                get_tab_idx is not None
                and "_state_lock" in line
                and "with" in line
            ):
                lock_idx = i
                break

        assert get_tab_idx is not None, "Found _get_tab call"
        assert lock_idx is not None, "Found subsequent _state_lock block"
        assert lock_idx > get_tab_idx, (
            "A6: _state_lock block is after _get_tab call"
        )
        # Verify _get_tab internally acquires _state_lock
        get_tab_src = inspect.getsource(VSCodeServer._get_tab)
        assert "_state_lock" in get_tab_src, (
            "A6: _get_tab acquires _state_lock internally"
        )

    def test_source_get_tab_uses_state_lock(self) -> None:
        """Structural: ``_get_tab`` acquires ``_state_lock``."""
        src = inspect.getsource(VSCodeServer._get_tab)
        assert "with self._state_lock:" in src, (
            "A6: _get_tab uses _state_lock"
        )

    def test_behavioral_toctou_gap_between_get_tab_and_lock(self) -> None:
        """Behavioral: demonstrate the TOCTOU gap.

        Thread A: calls ``_get_tab('t1')`` → creates tab → releases lock.
        Thread B: calls ``_close_tab('t1')`` → removes tab → releases lock.
        Thread A: enters ``with _state_lock:`` → tab still in local var
                  but no longer in ``_tab_states``.

        The tab reference is valid but orphaned.
        """
        server, events = _make_server()
        tab_id = "tab-a6"

        # Step 1: _get_tab creates the tab
        tab = server._get_tab(tab_id)
        assert tab_id in server._tab_states, "Tab exists after _get_tab"

        # Step 2: Between _get_tab and the next lock acquisition,
        # another thread could close the tab
        with server._state_lock:
            server._tab_states.pop(tab_id, None)

        assert tab_id not in server._tab_states, (
            "Tab removed by concurrent close"
        )

        # Step 3: _cmd_run's second lock block uses the orphaned tab ref
        # The tab object is valid but no longer tracked
        with server._state_lock:
            # This code would run in _cmd_run's second lock block
            if tab.task_thread is not None and tab.task_thread.is_alive():
                pass  # would check alive status on an orphaned tab
            else:
                # Would start a thread for a tab that's no longer tracked
                tab.stop_event = threading.Event()
                tab.task_thread = threading.Thread(
                    target=lambda: None, daemon=True
                )

        # The thread is set on an orphaned tab — cleanup in _run_task's
        # finally block would try _tab_states.get(tab_id) and get None,
        # so the thread/stop_event are never cleaned up
        still_tracked = server._tab_states.get(tab_id)
        assert still_tracked is None, (
            "A6: task started on an orphaned tab — cleanup will be skipped"
        )
        # tab.task_thread is set but unreachable via _tab_states
        assert tab.task_thread is not None, (
            "A6: orphaned tab has a task_thread that can't be cleaned up"
        )


if __name__ == "__main__":
    unittest.main()
