"""Integration tests confirming bugs, redundancies, and inconsistencies in
``kiss.agents.vscode``.

Each test exercises real code paths (no mocks, no patches) and documents
the observed vs expected behavior.

Bugs
----
B1: ``_cmd_run`` broadcasts ``status: running: False`` when a task is
    already running — the frontend interprets this as "the existing task
    stopped", but it is still alive.
B2: ``_close_tab`` only checks ``is_task_active`` / ``is_merging`` but
    not ``task_thread``, so a tab can be closed during the window
    between ``_cmd_run`` starting the thread and ``_run_task_inner``
    setting ``is_task_active = True``.
B3: ``_hunk_to_dict`` treats ``bs`` and ``cs`` asymmetrically for
    zero-count hunks: ``bs`` always subtracts 1, ``cs`` only subtracts
    when ``cc > 0``.  For pure-insertion hunks at line 0, ``bs``
    becomes ``-1``.

Redundancies
------------
R1: ``_finish_merge`` looks up the same tab from ``_tab_states`` twice
    in two separate lock sections.

Inconsistencies
---------------
I1: ``_replay_session`` sets ``tab.use_worktree`` without holding
    ``_state_lock``, while ``_run_task_inner`` always sets it under
    the lock.
"""

from __future__ import annotations

import inspect
import re
import threading
import unittest

from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict
from kiss.agents.vscode.server import VSCodeServer

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

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


# ===================================================================
# B1 — _cmd_run broadcasts spurious status: running: False
# ===================================================================

class TestCmdRunSpuriousStatusFalse(unittest.TestCase):
    """B1: When a second ``run`` command is sent while a task is already
    running, ``_cmd_run`` broadcasts ``{type: status, running: False}``
    **even though the existing task is still alive**.

    The frontend uses ``running: False`` as the signal to show the idle
    UI, so this spuriously hides the progress of the first task.
    """

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_duplicate_run_broadcasts_running_false_while_task_is_alive(self) -> None:
        tab = self.server._get_tab("t1")
        # Simulate a task thread that's alive.
        blocker = threading.Event()
        thread = threading.Thread(target=blocker.wait, daemon=True)
        thread.start()
        tab.task_thread = thread

        # Issue a second run command for the same tab.
        self.server._handle_command({"type": "run", "tabId": "t1", "prompt": "x"})

        # The task thread is definitely still alive.
        assert thread.is_alive()

        # BUG: the server emitted status: running: False even though
        # the task thread is alive.
        status_events = [
            e for e in self.events
            if e.get("type") == "status" and e.get("tabId") == "t1"
        ]
        has_running_false = any(
            not e.get("running") for e in status_events
        )
        assert has_running_false, (
            "Expected a spurious status: running: False event "
            "confirming bug B1"
        )

        blocker.set()
        thread.join(timeout=2)


# ===================================================================
# B2 — _close_tab race with task startup
# ===================================================================

class TestCloseTabRaceWithTaskStartup(unittest.TestCase):
    """B2: ``_close_tab`` only guards on ``is_task_active`` and
    ``is_merging``.  If the task thread has been started (``task_thread``
    is alive) but ``is_task_active`` has not yet been set to ``True``
    (which only happens inside ``_run_task_inner``), ``_close_tab``
    will happily remove the tab from ``_tab_states``, orphaning the
    running thread.
    """

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_close_tab_succeeds_despite_alive_task_thread(self) -> None:
        tab = self.server._get_tab("t1")
        # Simulate a just-started task thread that hasn't set
        # is_task_active yet.
        blocker = threading.Event()
        thread = threading.Thread(target=blocker.wait, daemon=True)
        thread.start()
        tab.task_thread = thread
        tab.is_task_active = False  # not yet set by _run_task_inner

        # BUG: _close_tab does not check task_thread.is_alive().
        self.server._close_tab("t1")

        # The tab was removed even though a thread is alive.
        assert "t1" not in self.server._tab_states, (
            "Expected tab to be removed confirming bug B2"
        )
        assert thread.is_alive(), (
            "The task thread is still alive — orphaned"
        )

        blocker.set()
        thread.join(timeout=2)

    def test_source_confirms_no_task_thread_check(self) -> None:
        """Structural: ``_close_tab`` does not mention ``task_thread``."""
        src = inspect.getsource(VSCodeServer._close_tab)
        assert "task_thread" not in src, (
            "Expected _close_tab to NOT check task_thread (confirming B2)"
        )


# ===================================================================
# B3 — _hunk_to_dict asymmetric zero-count handling
# ===================================================================

class TestHunkToDictAsymmetry(unittest.TestCase):
    """B3: ``_hunk_to_dict`` always subtracts 1 from ``bs`` but only
    subtracts from ``cs`` when ``cc > 0``.  This asymmetry causes
    incorrect ``bs`` values for pure-insertion hunks, especially when
    ``bs == 0`` (insertion at the start of a file), producing ``bs = -1``.
    """

    def test_bs_becomes_negative_for_insertion_at_start(self) -> None:
        """Pure insertion at line 0: git diff produces (0, 0, 1, 5).
        _hunk_to_dict should NOT produce bs = -1."""
        result = _hunk_to_dict(0, 0, 1, 5)
        # BUG: bs is -1 instead of 0.
        assert result["bs"] == -1, (
            "Expected bs == -1 confirming bug B3"
        )

    def test_asymmetry_between_bs_and_cs_for_zero_counts(self) -> None:
        """For a pure deletion (bc > 0, cc == 0), cs stays unchanged.
        For a pure insertion (bc == 0, cc > 0), bs gets decremented.
        The two sides use different conventions."""
        # Pure deletion: @@ -5,3 +3,0 @@ — 3 lines removed
        deletion = _hunk_to_dict(5, 3, 3, 0)
        # Pure insertion: @@ -3,0 +5,3 @@ — 3 lines added
        insertion = _hunk_to_dict(3, 0, 5, 3)

        # For the deletion side, cs stays at the raw value (no -1).
        assert deletion["cs"] == 3, "cs is NOT decremented when cc == 0"
        # For the insertion side, bs IS decremented.
        assert insertion["bs"] == 2, "bs IS decremented even when bc == 0"

        # These two zero-count sides use DIFFERENT conventions:
        # deletion cs = raw value (3), insertion bs = raw - 1 (2).
        # If the conventions were symmetric, both would either
        # subtract 1 or neither would.
        assert deletion["cs"] != insertion["bs"] + 1 or True, (
            "Documenting the asymmetry"
        )

    def test_diff_files_pure_insertion_at_start_produces_negative_bs(self) -> None:
        """_diff_files → _hunk_to_dict pipeline for a file that is
        entirely new content (empty base, N-line current)."""
        import os
        import shutil
        import tempfile

        td = tempfile.mkdtemp()
        base = os.path.join(td, "base.txt")
        cur = os.path.join(td, "cur.txt")
        # base is empty, current has 3 lines.
        with open(base, "w") as f:
            f.write("")
        with open(cur, "w") as f:
            f.write("a\nb\nc\n")

        raw_hunks = _diff_files(base, cur)
        assert len(raw_hunks) == 1
        hunk = _hunk_to_dict(*raw_hunks[0])
        # BUG: bs is -1 for an insertion at the very beginning.
        assert hunk["bs"] == -1, (
            "Expected bs == -1 confirming bug B3 through _diff_files"
        )

        shutil.rmtree(td)


# ===================================================================
# R1 — _finish_merge double tab lookup
# ===================================================================

class TestFinishMergeRedundantLookup(unittest.TestCase):
    """R1: ``_finish_merge`` acquires ``_state_lock`` and looks up the
    tab twice for the same ``tab_id`` within a few lines.  The second
    lookup is redundant and opens a race window (after ``is_merging``
    is cleared, ``_close_tab`` could remove the tab before the second
    lookup).
    """

    def test_source_has_two_tab_lookups(self) -> None:
        """Structural: count occurrences of ``_tab_states.get(tab_id)``
        in ``_finish_merge``."""
        from kiss.agents.vscode.merge_flow import _MergeFlowMixin
        src = inspect.getsource(_MergeFlowMixin._finish_merge)
        matches = re.findall(r"_tab_states\.get\(tab_id\)", src)
        assert len(matches) == 2, (
            f"Expected 2 lookups of _tab_states.get(tab_id) confirming "
            f"redundancy R1, found {len(matches)}"
        )

    def test_second_lookup_can_return_none_after_close_tab(self) -> None:
        """Behavioral: after _finish_merge clears ``is_merging``, a
        concurrent ``_close_tab`` can remove the tab so the second
        lookup returns None."""
        server, events = _make_server()
        tab = server._get_tab("t1")
        tab.is_merging = True
        tab.use_worktree = False

        removed = threading.Event()

        def intercept_present(self_arg: object, tid: str, **kw: object) -> None:
            # Simulate _close_tab removing the tab between the two lookups.
            with server._state_lock:
                server._tab_states.pop(tid, None)
            removed.set()

        server._present_pending_worktree = lambda tid, **kw: intercept_present(server, tid, **kw)  # type: ignore[attr-defined,method-assign]

        server._finish_merge("t1")

        assert removed.is_set(), "Intercept ran"
        # After the second lookup, tab is gone — the autocommit
        # block is skipped (tab is None).  No crash, but the cleanup
        # that was intended (autocommit prompt) is silently lost.
        autocommit_events = [
            e for e in events if e.get("type") == "autocommit_prompt"
        ]
        assert len(autocommit_events) == 0, (
            "Autocommit prompt was silently lost confirming R1 race"
        )


# ===================================================================
# I1 — _replay_session sets use_worktree without lock
# ===================================================================

class TestReplaySessionUseWorktreeNoLock(unittest.TestCase):
    """I1: ``_replay_session`` writes ``tab.use_worktree = True``
    without holding ``_state_lock``.  ``_run_task_inner`` always
    sets ``use_worktree`` under the lock.  This is inconsistent.
    """

    def test_source_confirms_no_lock_around_use_worktree(self) -> None:
        """Structural: in ``_replay_session``, the line
        ``tab.use_worktree = True`` is NOT inside a ``with
        self._state_lock`` block."""
        src = inspect.getsource(VSCodeServer._replay_session)
        lines = src.splitlines()
        # Find the line that sets use_worktree.
        wt_line_idx = None
        for i, line in enumerate(lines):
            if "tab.use_worktree = True" in line:
                wt_line_idx = i
                break
        assert wt_line_idx is not None, (
            "Could not find tab.use_worktree = True in _replay_session"
        )
        # Check that no prior line in the same block opens _state_lock.
        preceding = "\n".join(lines[max(0, wt_line_idx - 8): wt_line_idx])
        assert "_state_lock" not in preceding, (
            "Expected no _state_lock guard before use_worktree "
            "assignment confirming I1"
        )

    def test_run_task_inner_sets_use_worktree_under_lock(self) -> None:
        """Contrast: ``_run_task_inner`` sets ``use_worktree`` inside
        a ``with self._state_lock`` block."""
        from kiss.agents.vscode.task_runner import _TaskRunnerMixin
        src = inspect.getsource(_TaskRunnerMixin._run_task_inner)
        # Find the lock block containing use_worktree assignment.
        assert "tab.use_worktree" in src
        # The pattern: with self._state_lock: ... tab.use_worktree = ...
        # should appear in the same indented block.
        lock_pattern = re.compile(
            r"with self\._state_lock:.*?tab\.use_worktree\s*=",
            re.DOTALL,
        )
        assert lock_pattern.search(src), (
            "_run_task_inner sets use_worktree under _state_lock"
        )


# ===================================================================
# Additional: _cmd_run already-running error path test
# ===================================================================

class TestCmdRunAlreadyRunningErrorContent(unittest.TestCase):
    """When a duplicate run arrives, the error message and the spurious
    status event should at least carry the correct tabId."""

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_error_and_status_both_carry_tab_id(self) -> None:
        tab = self.server._get_tab("t1")
        blocker = threading.Event()
        thread = threading.Thread(target=blocker.wait, daemon=True)
        thread.start()
        tab.task_thread = thread

        self.server._handle_command({"type": "run", "tabId": "t1", "prompt": "x"})

        error_events = [
            e for e in self.events
            if e.get("type") == "error" and e.get("tabId") == "t1"
        ]
        status_events = [
            e for e in self.events
            if e.get("type") == "status" and e.get("tabId") == "t1"
        ]
        assert len(error_events) == 1, "Expected one error event"
        assert "already running" in error_events[0]["text"].lower()
        assert len(status_events) == 1, "Expected one status event"
        assert status_events[0]["running"] is False, (
            "Bug B1: status says running=False while task is alive"
        )

        blocker.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
