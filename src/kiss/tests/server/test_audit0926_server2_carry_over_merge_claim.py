# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: the pre-run carry-over merge must fence off direct runs.

Since f7d726827, a tab holding a pending worktree whose next prompt
runs directly on the main tree merges that worktree BEFORE the run
starts (``_run_task_inner``, ``agent._retire_previous_worktree()``)
whenever the main tree has no tracked edits.  That merge stashes,
checks out and squash-merges the main working tree, but it ran with
neither ``is_merging`` (the tab is a non-worktree run) nor a main-tree
claim published.  The non-worktree admission gate only refuses on
those two signals, so a direct task started on ANOTHER tab while the
merge was in flight was admitted and wrote the very tree the merge was
rewriting.

Interleaving reproduced here: the retirement is held part-way (its
first ``repo_lock(repo)`` acquisition) by keeping that real lock on the
test thread; a direct run is then started on a third tab.  It must be
refused.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from kiss.agents.sorcar.git_worktree import repo_lock
from kiss.server import agent_state
from kiss.tests.server import test_worktree_repo_aware_busy_guard as _guard
from kiss.tests.server.test_worktree_no_autocommit_branch import (
    _list_kiss_wt_branches,
)
from kiss.tests.server.test_worktree_repo_aware_busy_guard import (
    _WT_TAB,
    _RepoAwareGuardBase,
)

_THIRD_TAB = "third-tab"


def _thread_in_function(thread: threading.Thread, name: str) -> bool:
    """True when *thread*'s live stack contains a frame of function *name*."""
    frame = sys._current_frames().get(thread.ident or -1)
    while frame is not None:
        if frame.f_code.co_name == name:
            return True
        frame = frame.f_back
    return False


class TestCarryOverMergeFencesDirectRuns(_RepoAwareGuardBase):

    def test_direct_run_on_other_tab_refused_during_carry_over_merge(
        self,
    ) -> None:
        helpers = _guard.TestMergeNotBlockedByUntouchedMainTree
        helpers._strand_then_let_occupant_settle(self)  # type: ignore[arg-type]
        repo = Path(self.repo)
        lock = repo_lock(repo)
        lock.acquire()
        worker = threading.Thread(
            target=helpers._run_direct_task_on_wt_tab,
            args=(self,),
            kwargs={"auto_commit": True},
            daemon=True,
        )
        try:
            worker.start()
            deadline = time.monotonic() + 30
            while not _thread_in_function(worker, "_retire_previous_worktree"):
                assert worker.is_alive(), "the carry-over merge never started"
                assert time.monotonic() < deadline, "the retirement never began"
                time.sleep(0.01)

            self.server._run_task_inner({
                "prompt": "direct task on another tab",
                "workDir": self.repo,
                "tabId": _THIRD_TAB,
                "useWorktree": False,
                "autoCommit": False,
                "model": "",
            })
        finally:
            lock.release()
            worker.join(timeout=60)
        assert not worker.is_alive()

        refusals = [
            e for e in self.events
            if e["type"] == "error" and e.get("tabId") == _THIRD_TAB
        ]
        assert any("in progress" in e.get("text", "") for e in refusals), (
            "BUG: a direct run was admitted onto the main tree while the "
            f"carry-over merge was rewriting it: {self._types()}"
        )
        # The carry-over itself still merged and the tab's own run ran.
        assert (repo / "agent_out.txt").exists()
        assert (repo / "direct_out.txt").exists()
        assert _list_kiss_wt_branches(self.repo) == []
        with self.server._state_lock:
            assert self.server._main_tree_claim_reason(repo) is None, (
                "the merge's main-tree claim must be released afterwards"
            )
            third = agent_state.find_by_tab(_THIRD_TAB)
            assert third is None or not third.is_running_non_wt
        state = agent_state.find_by_tab(_WT_TAB)
        assert state is not None and state.agent is not None
        assert not state.agent._wt_pending
