# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 5: double Merge click runs the worktree merge twice (BUG-5E-1).

``_handle_worktree_action`` guards a worktree merge/discard with
``_check_worktree_busy`` and then sets ``tab.is_merging = True`` under
``_state_lock``.  But the busy check inspects only
``tab.is_task_active`` and ``_any_non_wt_running()`` — it never checks
``tab.is_merging`` itself.  So a second ``worktreeAction`` command that
arrives while the first merge is still in flight (a double click on the
Merge button, or one click from each of two connected clients) passes
the guard, queues behind ``repo_lock``, and re-runs ``wt.merge()`` /
``wt.discard()`` on the already-merged worktree.  Additionally,
whichever thread finishes first clears ``is_merging`` in its ``finally``
while the other thread is still merging, reopening the window for a
non-worktree task to start writing to the main tree mid-merge.

The test makes the race deterministic by blocking the agent's
``merge()`` until the second command has been issued.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server.server import VSCodeServer


class _BlockingMergeAgent(WorktreeSorcarAgent):
    """Real agent whose merge() blocks until released and counts calls."""

    def __init__(self, repo_root: Path) -> None:
        super().__init__("Bughunt5 blocking-merge agent")
        # A pending worktree: ``_wt_pending`` / ``_repo_root`` etc. are
        # properties derived from ``self._wt``.
        self._wt = GitWorktree(
            repo_root=repo_root,
            branch="kiss/wt-bughunt5",
            original_branch="main",
            wt_dir=repo_root / "wt",
            baseline_commit=None,
        )
        self.merge_entered = threading.Event()
        self.merge_release = threading.Event()
        self.merge_calls = 0
        self._calls_lock = threading.Lock()

    def merge(self) -> str:
        """Simulate a slow merge (LLM commit-message + git squash-merge)."""
        with self._calls_lock:
            self.merge_calls += 1
        self.merge_entered.set()
        self.merge_release.wait(timeout=30)
        # A real successful merge releases the worktree.
        self._wt = None
        return "Successfully merged worktree branch."


class TestDoubleMergeClick(unittest.TestCase):
    """A second merge click during an in-flight merge must be refused."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt5-dblmerge-")
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

        def capture(event: dict[str, Any]) -> None:
            with self._events_lock:
                self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_second_concurrent_merge_click_is_refused(self) -> None:
        tab_id = "wt-tab"
        agent = _BlockingMergeAgent(Path(self.tmpdir))
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = True
        tab.agent = agent

        results: dict[str, dict[str, Any]] = {}

        def first_click() -> None:
            results["first"] = self.server._handle_worktree_action(
                "merge", tab_id,
            )

        t1 = threading.Thread(target=first_click, daemon=True)
        t1.start()
        assert agent.merge_entered.wait(timeout=30), (
            "first merge never started"
        )

        # First merge is now in flight (inside wt.merge(), holding
        # repo_lock, tab.is_merging=True).  The user double-clicks:
        def second_click() -> None:
            results["second"] = self.server._handle_worktree_action(
                "merge", tab_id,
            )

        t2 = threading.Thread(target=second_click, daemon=True)
        t2.start()
        # The second click must be refused promptly — it must NOT
        # queue behind the in-flight merge.
        t2.join(timeout=5)
        was_queued = t2.is_alive()
        agent.merge_release.set()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not was_queued, (
            "BUG: the second merge click passed _check_worktree_busy "
            "(is_merging is never checked) and queued behind the "
            "in-flight merge instead of being refused"
        )
        assert agent.merge_calls == 1, (
            f"BUG: wt.merge() ran {agent.merge_calls} times — the second "
            "click re-merged the already-merged worktree"
        )
        assert results["first"]["success"] is True
        assert results["second"]["success"] is False, (
            f"second click reported {results['second']!r} instead of a "
            "busy refusal"
        )

    def test_merge_click_during_hunk_review_is_refused(self) -> None:
        """While ``is_merging`` is held by a merge review session, a
        ``worktreeAction`` must be refused instead of mutating git
        state under the open review."""
        tab_id = "wt-tab-2"
        agent = _BlockingMergeAgent(Path(self.tmpdir))
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = True
        tab.agent = agent
        with self.server._state_lock:
            tab.is_merging = True

        result = self.server._handle_worktree_action("merge", tab_id)

        assert result["success"] is False, (
            "BUG: worktreeAction ran during an open merge review "
            f"(result={result!r})"
        )
        assert agent.merge_calls == 0, (
            "BUG: wt.merge() mutated git state while the merge review "
            "was still open"
        )


if __name__ == "__main__":
    unittest.main()
