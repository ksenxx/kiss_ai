# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: a worktree run's carry-over merge must fence off direct runs.

A tab holding a pending worktree whose next prompt runs in a NEW
worktree used to leave the retirement of the old one to the agent:
``WorktreeSorcarAgent.run`` -> ``_try_setup_worktree`` ->
``_retire_previous_worktree`` -> ``_do_merge`` stashes, checks out and
squash-merges the main working tree.  That merge ran with neither
``is_merging`` (the tab's state only flags merges started by
``merge_flow``) nor a main-tree claim, and the non-worktree admission
gate in ``_run_task_inner`` only refuses on those two signals, so a
direct task started on ANOTHER tab while the merge was in flight was
admitted and wrote the very tree the merge was rewriting.

Interleaving reproduced here: the carry-over retirement is held
part-way by keeping the real ``repo_lock(repo)`` on the test thread
(the merge, and the agent-side setup that would run it, both take
it); a direct run is then started on a third tab.  It must be refused.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from kiss.agents.sorcar.git_worktree import repo_lock
from kiss.server import agent_state
from kiss.tests.server import test_worktree_repo_aware_busy_guard as _guard
from kiss.tests.server.test_audit0926_server2_carry_over_merge_claim import (
    _thread_in_function,
)
from kiss.tests.server.test_worktree_no_autocommit_branch import (
    _list_kiss_wt_branches,
    _patch_parent_run_create_file,
)
from kiss.tests.server.test_worktree_repo_aware_busy_guard import (
    _OTHER_TAB,
    _WT_TAB,
    _RepoAwareGuardBase,
)

_THIRD_TAB = "third-tab"


class TestWorktreeRunCarryOverFencesDirectRuns(_RepoAwareGuardBase):

    def test_direct_run_refused_during_worktree_run_carry_over_merge(
        self,
    ) -> None:
        helpers = _guard.TestMergeNotBlockedByUntouchedMainTree
        helpers._strand_then_let_occupant_settle(self)  # type: ignore[arg-type]
        # The occupant finishes: nothing runs on the main tree any more,
        # so nothing blocks the carry-over merge of the next run.
        with self.server._state_lock:
            other = agent_state.find_by_tab(_OTHER_TAB)
            assert other is not None
            other.is_running_non_wt = False
            other.is_task_active = False
        repo = Path(self.repo)
        stranded = agent_state.find_by_tab(_WT_TAB)
        assert stranded is not None and stranded.agent is not None
        assert stranded.agent._wt_pending

        lock = repo_lock(repo)
        lock.acquire()
        worker = threading.Thread(
            target=self._run_worktree_task_with_changes, daemon=True,
        )
        try:
            worker.start()
            deadline = time.monotonic() + 30
            while not (
                _thread_in_function(worker, "_retire_previous_worktree")
                or _thread_in_function(worker, "_try_setup_worktree")
            ):
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
            f"worktree run's carry-over merge was rewriting it: {self._types()}"
        )
        # The carry-over merged and the new worktree run merged too.
        assert (repo / "agent_out.txt").exists()
        assert _list_kiss_wt_branches(self.repo) == []
        with self.server._state_lock:
            assert self.server._main_tree_claim_reason(repo) is None, (
                "the merge's main-tree claim must be released afterwards"
            )
            third = agent_state.find_by_tab(_THIRD_TAB)
            assert third is None or not third.is_running_non_wt


class TestWorktreeRunOutsideGitKeepsCarryOverPending(_RepoAwareGuardBase):

    def test_non_git_work_dir_leaves_previous_worktree_pending(self) -> None:
        """A worktree-mode run whose workDir is not in a git repository
        never enters worktree setup (the agent runs it directly), so it
        must not retire the tab's pending worktree: the worktree stays
        pending and the user is offered Merge/Discard for it."""
        helpers = _guard.TestMergeNotBlockedByUntouchedMainTree
        helpers._strand_then_let_occupant_settle(self)  # type: ignore[arg-type]
        with self.server._state_lock:
            other = agent_state.find_by_tab(_OTHER_TAB)
            assert other is not None
            other.is_running_non_wt = False
            other.is_task_active = False
        stranded = agent_state.find_by_tab(_WT_TAB)
        assert stranded is not None and stranded.agent is not None
        agent = stranded.agent
        assert agent._wt_pending
        branches = _list_kiss_wt_branches(self.repo)
        assert branches
        plain_dir = Path(self.tmpdir) / "plain-dir"
        plain_dir.mkdir()

        _patch_parent_run_create_file(None)
        self.server._run_task_inner({
            "prompt": "task outside any git repository",
            "workDir": str(plain_dir),
            "tabId": _WT_TAB,
            "useWorktree": True,
            "autoCommit": False,
            "model": "",
        })

        assert agent._wt_pending, (
            "BUG: a run that never enters worktree setup retired the "
            f"tab's pending worktree: {self._types()}"
        )
        assert _list_kiss_wt_branches(self.repo) == branches
        assert any(
            e["type"] == "worktree_done" and e.get("tabId") == _WT_TAB
            for e in self.events
        ), self._types()
        assert not (Path(self.repo) / "agent_out.txt").exists()
        with self.server._state_lock:
            assert self.server._main_tree_claim_reason(Path(self.repo)) is None
