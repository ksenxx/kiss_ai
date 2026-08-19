# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: agent worktrees leak when the main tree is busy.

Bug report: the repository accumulates orphaned ``kiss/wt-*`` branches,
leftover ``.kiss-worktrees/<slug>`` directories and stale
``branch.kiss/*`` git-config sections that nothing ever cleans up.

Root cause — three code sites bail out on the process-global
``_any_non_wt_running()`` guard and nothing ever retries the cleanup:

1. :meth:`_MergeFlowMixin._handle_worktree_action` with
   ``internal=True`` (the post-task auto-merge / auto-discard path)
   returns ``{"success": False, "message": "Another tab is running a
   task on the main working tree..."}`` instead of discarding.
2. :meth:`_MergeFlowMixin._present_pending_worktree` skips its
   empty-worktree auto-discard branch when ``non_wt_busy``.
3. :meth:`_TaskRunnerMixin._run_task_inner` clears ``tab.agent._wt``
   when a non-worktree task is running.  ``_wt`` is the **only**
   in-memory handle to the worktree, so the directory, the branch and
   its ``branch.<name>.*`` config section leak forever.

Bailing out is defensible for a *merge* (which mutates the main
working tree another task is writing).  It is **not** defensible for
cleanup of a worktree that has no changes: removing
``.kiss-worktrees/<slug>`` and deleting an unmerged, empty branch
touches neither the main working tree's files nor its HEAD, so there
is no reason to leak.

Every test here is end to end: a real temp git repo, a real
:class:`VSCodeServer`, and a real ``server._run_task_inner(...)``
call.  The only substitution is the harness' existing
``_patch_parent_run_create_file`` stub, which replaces the LLM-driven
agent body with a deterministic one — production wiring is untouched.

:class:`TestWorktreeLeakWhenMainTreeBusy` FAILS on the buggy code —
one test per leaked artifact (branch, directory, git config), plus one
that runs two tasks on the same tab to hit leak site 3.
:class:`TestWorktreeCleanupWhenMainTreeIdle` is its non-regression
twin: the very same scenario with an idle main tree cleans up
perfectly, which proves the failures above are caused by the guard and
not by the harness.
"""

from __future__ import annotations

from pathlib import Path

from kiss.server import agent_state
from kiss.tests.server.test_worktree_no_autocommit_branch import (
    _list_kiss_wt_branches,
    _patch_parent_run_create_file,
    _run_git,
    _WorktreeNoAutocommitBase,
)

#: Tab that runs the worktree task under test.
_WT_TAB = "wt-tab"

#: Tab standing in for another VS Code tab that is running a plain
#: (non-worktree) task directly on the main working tree.
_BUSY_TAB = "busy-main-tree-tab"


def _worktree_dirs(repo: str) -> list[str]:
    """Names of the leftover directories under ``<repo>/.kiss-worktrees``."""
    root = Path(repo) / ".kiss-worktrees"
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir())


def _registered_worktrees(repo: str) -> list[str]:
    """Paths git itself still tracks as worktrees of *repo*.

    The first entry is always the main working tree, so anything
    beyond it is a linked worktree that was never removed.
    """
    result = _run_git(repo, "worktree", "list", "--porcelain")
    return [
        line.split(" ", 1)[1].strip()
        for line in result.stdout.splitlines()
        if line.startswith("worktree ")
    ]


def _kiss_branch_config(repo: str) -> list[str]:
    """``branch.kiss/*`` config entries still present in *repo*.

    ``git branch -D`` removes the matching config section; entries
    surviving the task therefore mean the branch was never deleted (or
    was deleted in a way that orphaned its config).
    """
    result = _run_git(repo, "config", "--get-regexp", r"^branch\.kiss/")
    return [line for line in result.stdout.splitlines() if line.strip()]


class _MainTreeBusyBase(_WorktreeNoAutocommitBase):
    """Runs a no-change worktree task and exposes the leaked artifacts."""

    #: Whether another tab is hammering the main working tree.
    main_tree_busy: bool = True

    def _mark_main_tree_busy(self, busy: bool) -> None:
        """Flip the "another tab owns the main working tree" state.

        Registered exactly the way production does it — the server
        registers a server-owned :class:`agent_state.AgentState` for
        the tab, and ``_run_task_inner`` sets ``is_running_non_wt =
        True`` on it for the whole duration of a non-worktree task.
        Setting the flag here is the faithful steady-state equivalent
        of that other task being mid-flight.
        """
        with self.server._state_lock:
            state = agent_state.find_by_tab(_BUSY_TAB)
            if state is None:
                state = agent_state.AgentState(
                    "task-" + _BUSY_TAB,
                    tab_id=_BUSY_TAB,
                    server_owned=True,
                )
                agent_state.register(state)
            state.is_task_active = busy
            state.is_running_non_wt = busy
        assert self.server._any_non_wt_running() is busy

    def _run_no_change_worktree_task(self, count: int = 1) -> None:
        """Drive *count* consecutive worktree tasks that change nothing.

        All of them run on the same tab, which is what a user does
        when they keep chatting in one VS Code tab.  The main tree is
        busy for the whole sequence when :attr:`main_tree_busy` is set,
        and idle again once it finishes — after that point nothing
        legitimately blocks cleanup.
        """
        if self.main_tree_busy:
            self._mark_main_tree_busy(True)

        self._original_run = _patch_parent_run_create_file(None)
        for i in range(count):
            self.server._run_task_inner({
                "prompt": f"worktree task {i} that changes nothing",
                "workDir": self.repo,
                "tabId": _WT_TAB,
                "useWorktree": True,
                "autoCommit": True,
                "model": "",
            })

        if self.main_tree_busy:
            self._mark_main_tree_busy(False)

    def _diagnostics(self) -> str:
        """Human-readable dump of every artifact the task left behind."""
        return (
            f"branches={_list_kiss_wt_branches(self.repo)} "
            f"worktree_dirs={_worktree_dirs(self.repo)} "
            f"git_worktrees={_registered_worktrees(self.repo)} "
            f"branch_config={_kiss_branch_config(self.repo)} "
            f"events={self._types()}"
        )


class TestWorktreeLeakWhenMainTreeBusy(_MainTreeBusyBase):
    """REPRODUCES THE BUG — cleanup is skipped and never retried."""

    main_tree_busy = True

    def test_no_change_worktree_task_does_not_leak_branch_when_main_tree_busy(
        self,
    ) -> None:
        """A no-change worktree task must not leave a branch behind.

        The worktree is empty, so there is nothing to merge and
        nothing that could clash with the concurrent non-worktree
        task; the branch should be deleted just as it is when the main
        tree is idle.
        """
        self._run_no_change_worktree_task()

        assert "worktree_created" in self._types(), self._types()

        branches = _list_kiss_wt_branches(self.repo)
        assert branches == [], (
            "BUG: a kiss/wt-* branch leaked because another tab was "
            "running a non-worktree task while this empty worktree "
            f"task finished.  {self._diagnostics()}"
        )

    def test_worktree_directory_removed_when_main_tree_busy(self) -> None:
        """No ``.kiss-worktrees`` directory and no linked git worktree
        may survive the task."""
        self._run_no_change_worktree_task()

        leftovers = _worktree_dirs(self.repo)
        assert leftovers == [], (
            "BUG: .kiss-worktrees still holds directories after the "
            f"task.  {self._diagnostics()}"
        )

        registered = _registered_worktrees(self.repo)
        assert len(registered) == 1, (
            "BUG: git still tracks a linked worktree; only the main "
            f"working tree should remain.  {self._diagnostics()}"
        )

    def test_branch_config_section_not_leaked(self) -> None:
        """``git config --get-regexp '^branch\\.kiss/'`` must be empty.

        Every leaked branch drags a ``branch.kiss/wt-*.*`` config
        section with it; those accumulate in ``.git/config`` forever.
        """
        self._run_no_change_worktree_task()

        entries = _kiss_branch_config(self.repo)
        assert entries == [], (
            "BUG: stale branch.kiss/* git config entries survived the "
            f"task: {entries}.  {self._diagnostics()}"
        )

    def test_second_task_on_same_tab_does_not_orphan_the_first_worktree(
        self,
    ) -> None:
        """Two no-change worktree tasks in one tab must leave nothing.

        This is leak site 3.  At the top of ``_run_task_inner`` the
        second task sees the first one's still-pending worktree and,
        because the main tree is busy, executes::

            tab.agent._merge_conflict_warning = ...
            tab.agent._wt = None

        ``_wt`` is the *only* in-memory handle to that worktree — the
        branch name, the directory and the original branch are all
        properties derived from it — so clearing it strands the first
        worktree beyond the reach of every later cleanup path.  Two
        tasks therefore leak two sets of artifacts, and the first set
        is unreachable forever.
        """
        self._run_no_change_worktree_task(count=2)

        tab = agent_state.find_by_tab(_WT_TAB)
        handle = None if tab is None or tab.agent is None else tab.agent._wt
        branches = _list_kiss_wt_branches(self.repo)
        dirs = _worktree_dirs(self.repo)

        assert branches == [] and dirs == [], (
            "BUG: two consecutive no-change worktree tasks leaked "
            f"{len(branches)} branch(es) and {len(dirs)} directory "
            "(the first one is now unreachable because _wt was "
            f"cleared: handle={handle!r}).  {self._diagnostics()}"
        )


class TestWorktreeCleanupWhenMainTreeIdle(_MainTreeBusyBase):
    """NON-REGRESSION — the identical scenario with an idle main tree.

    This whole class must pass on the buggy code.  It pins the
    behaviour the tests above are asking for and proves the harness
    itself is sound: the only difference from
    :class:`TestWorktreeLeakWhenMainTreeBusy` is the busy flag.
    """

    main_tree_busy = False

    def test_no_change_worktree_task_leaves_nothing_behind(self) -> None:
        """Branch, directory, git registration and config all cleaned up."""
        self._run_no_change_worktree_task()

        assert "worktree_created" in self._types(), self._types()

        assert _list_kiss_wt_branches(self.repo) == [], self._diagnostics()
        assert _worktree_dirs(self.repo) == [], self._diagnostics()
        assert len(_registered_worktrees(self.repo)) == 1, self._diagnostics()
        assert _kiss_branch_config(self.repo) == [], self._diagnostics()

    def test_two_no_change_tasks_leave_nothing_behind(self) -> None:
        """The same for two back-to-back tasks on one tab."""
        self._run_no_change_worktree_task(count=2)

        assert _list_kiss_wt_branches(self.repo) == [], self._diagnostics()
        assert _worktree_dirs(self.repo) == [], self._diagnostics()
        assert len(_registered_worktrees(self.repo)) == 1, self._diagnostics()
        assert _kiss_branch_config(self.repo) == [], self._diagnostics()


if __name__ == "__main__":  # pragma: no cover
    import unittest
    unittest.main()
