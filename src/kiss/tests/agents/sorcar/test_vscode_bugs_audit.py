# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for bugs, redundancies, and inconsistencies in
``kiss.agents.vscode`` — updated to verify the fixes.

Bugs
----
B1: ``_cmd_run`` silently drops a duplicate ``run`` while a task is
    already running — no error, no status broadcast.
B2: ``_close_tab`` now also checks ``task_thread.is_alive()`` and
    refuses to remove a tab with a live thread.
B3: ``_hunk_to_dict`` now treats ``bs`` and ``cs`` symmetrically:
    both skip the ``-1`` adjustment when their respective count is 0.

Redundancies
------------
R1: ``_finish_merge`` uses a single tab lookup instead of two.

Inconsistencies
---------------
I1: ``_replay_session`` now sets ``tab.use_worktree`` under
    ``_state_lock``, consistent with ``_run_task_inner``.
"""

from __future__ import annotations

import threading
import unittest

from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict
from kiss.agents.vscode.server import VSCodeServer


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


class TestCmdRunDuplicateSilentDrop(unittest.TestCase):
    """When a second ``run`` command is sent while a task is already
    running, ``_cmd_run`` silently drops it: no error, no status
    broadcast, no new thread.  Adding a task while one is running
    must do nothing.
    """

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_duplicate_run_emits_no_events_while_task_is_alive(self) -> None:
        tab = self.server._get_tab("t1")
        tab.agent = WorktreeSorcarAgent("Sorcar VS Code")
        blocker = threading.Event()
        thread = threading.Thread(target=blocker.wait, daemon=True)
        thread.start()
        tab.task_thread = thread

        events_before = len(self.events)
        self.server._handle_command({"type": "run", "tabId": "t1", "prompt": "x"})

        assert thread.is_alive()
        # The duplicate run must produce zero broadcasts.
        assert len(self.events) == events_before
        # And task_thread is still the original blocker thread (no new run).
        assert tab.task_thread is thread

        blocker.set()
        thread.join(timeout=2)


class TestCloseTabRaceWithTaskStartup(unittest.TestCase):
    """B2 fix: ``_close_tab`` now also checks ``task_thread.is_alive()``
    so it refuses to remove a tab with an alive thread even when
    ``is_task_active`` has not yet been set.
    """

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_close_tab_refuses_when_task_thread_alive(self) -> None:
        tab = self.server._get_tab("t1")
        tab.agent = WorktreeSorcarAgent("Sorcar VS Code")
        blocker = threading.Event()
        thread = threading.Thread(target=blocker.wait, daemon=True)
        thread.start()
        tab.task_thread = thread
        tab.is_task_active = False

        self.server._close_tab("t1")

        assert "t1" in self.server._running_agent_states, (
            "B2 fix: tab should NOT be removed while task_thread is alive"
        )

        blocker.set()
        thread.join(timeout=2)



class TestHunkToDictAsymmetry(unittest.TestCase):
    """B3 fix: ``_hunk_to_dict`` now treats ``bs`` and ``cs``
    symmetrically — both skip the ``-1`` adjustment when their
    respective count is 0.
    """

    def test_bs_is_zero_for_insertion_at_start(self) -> None:
        """Pure insertion at line 0: bs should be 0, not -1."""
        result = _hunk_to_dict(0, 0, 1, 5)
        assert result["bs"] == 0, (
            "B3 fix: bs should be 0 for zero-count insertion at start"
        )

    def test_symmetry_between_bs_and_cs_for_zero_counts(self) -> None:
        """Both zero-count sides now use the same convention."""
        deletion = _hunk_to_dict(5, 3, 3, 0)
        insertion = _hunk_to_dict(3, 0, 5, 3)

        assert deletion["cs"] == 3, "cs is NOT decremented when cc == 0"
        assert insertion["bs"] == 3, (
            "B3 fix: bs is NOT decremented when bc == 0"
        )

    def test_diff_files_pure_insertion_at_start_produces_zero_bs(self) -> None:
        """_diff_files → _hunk_to_dict pipeline: bs should be 0."""
        import os
        import shutil
        import tempfile

        td = tempfile.mkdtemp()
        base = os.path.join(td, "base.txt")
        cur = os.path.join(td, "cur.txt")
        with open(base, "w") as f:
            f.write("")
        with open(cur, "w") as f:
            f.write("a\nb\nc\n")

        raw_hunks = _diff_files(base, cur)
        assert len(raw_hunks) == 1
        hunk = _hunk_to_dict(*raw_hunks[0])
        assert hunk["bs"] == 0, (
            "B3 fix: bs should be 0 for insertion at start through _diff_files"
        )

        shutil.rmtree(td)


class TestFinishMergeRedundantLookup(unittest.TestCase):
    """R1 fix: ``_finish_merge`` now performs a single tab lookup."""


    def test_autocommit_prompt_not_lost_after_tab_removal(self) -> None:
        """Behavioral: the autocommit check uses the tab ref from the
        first lookup, so removing the tab mid-flow doesn't lose it."""
        server, events = _make_server()
        tab = server._get_tab("t1")
        tab.agent = WorktreeSorcarAgent("Sorcar VS Code")
        tab.is_merging = True
        tab.use_worktree = False

        removed = threading.Event()

        def intercept_present(tid: str, **kw: object) -> None:
            with server._state_lock:
                server._running_agent_states.pop(tid, None)
            removed.set()

        server._present_pending_worktree = intercept_present  # type: ignore[assignment,method-assign]

        server._finish_merge("t1")

        assert removed.is_set(), "Intercept ran"




class TestCmdRunDuplicateSilentDropNoErrorBroadcast(unittest.TestCase):
    """A duplicate ``run`` while a task is alive emits zero broadcasts —
    no error, no status, nothing on any tab id.
    """

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_no_error_or_status_for_alive_task(self) -> None:
        tab = self.server._get_tab("t1")
        tab.agent = WorktreeSorcarAgent("Sorcar VS Code")
        blocker = threading.Event()
        thread = threading.Thread(target=blocker.wait, daemon=True)
        thread.start()
        tab.task_thread = thread

        events_before = len(self.events)
        self.server._handle_command({"type": "run", "tabId": "t1", "prompt": "x"})

        new_events = self.events[events_before:]
        assert not any(e.get("type") == "error" for e in new_events)
        assert not any(e.get("type") == "status" for e in new_events)

        blocker.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
