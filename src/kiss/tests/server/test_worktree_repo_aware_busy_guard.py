# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: the main-tree-busy guard must be repository-aware.

Bug report: with two tasks running in worktree mode, finishing one of
them is refused with::

    Another tab is running a task on the main working tree.
    Wait for it to finish before merging.

even though *no* task is touching the main working tree — both tasks
live in their own ``.kiss-worktrees/<slug>`` directories.

Root cause: ``VSCodeServer._any_non_wt_running()`` returned True when
ANY tab had ``is_running_non_wt`` set, without knowing *which* main
working tree that task occupies.  A non-worktree task running in a
different repository, in a non-git directory, or inside a linked
``.kiss-worktrees`` worktree (e.g. a sub-task submitted through the
daemon API, whose ``useWorktree`` defaults to False and whose
``work_dir`` is the parent task's worktree) therefore blocked every
worktree merge everywhere.

The fix records the resolved main-repo root on the tab
(``non_wt_repo_root``) when a non-worktree task starts, and the guard
only blocks actions on that same repository.  The reverse admission
gate (a worktree merge in progress refusing a new non-worktree task)
became repository-aware the same way (``_wt_merge_on_repo``).

One occupancy case must KEEP blocking (gpt-5.6-sol review finding):
a non-worktree task whose toplevel is the very worktree directory a
merge or discard would remove.  Distinct worktrees stay non-blocking,
but deleting a running task's working directory out from under it
loses work, so ``_check_worktree_busy`` compares against the pending
``wt_dir`` too, and ``_wt_merge_on_repo`` refuses to admit a task
into a worktree that is mid-merge.

Every test drives the real ``VSCodeServer`` post-task flow against a
real temp git repo; only the LLM agent body and the LLM commit-message
generator are replaced with deterministic functions.
"""

from __future__ import annotations

from pathlib import Path

import kiss.agents.sorcar.commit_message as _commit_message_module
from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.tests.server.test_worktree_no_autocommit_branch import (
    _init_repo,
    _list_kiss_wt_branches,
    _patch_parent_run_create_file,
    _WorktreeNoAutocommitBase,
)

#: Tab that runs the worktree task under test.
_WT_TAB = "wt-tab"

#: Tab standing in for another tab running a plain (non-worktree) task.
_OTHER_TAB = "other-task-tab"


def _fake_commit_message(
    diff_text: str,
    user_prompt: str | None = None,
    task_result: str | None = None,
) -> str:
    """Deterministic stand-in for the LLM commit-message generator."""
    del diff_text, user_prompt, task_result
    return "test: deterministic commit message"


class _RepoAwareGuardBase(_WorktreeNoAutocommitBase):
    """Harness: second repo + busy-tab helpers + offline commit messages."""

    def setUp(self) -> None:
        super().setUp()
        self.other_repo = str(Path(self.tmpdir) / "other-repo")
        Path(self.other_repo).mkdir(parents=True, exist_ok=True)
        _init_repo(self.other_repo)
        self._orig_gen = _commit_message_module.generate_commit_message_from_diff
        _commit_message_module.generate_commit_message_from_diff = (  # type: ignore[assignment]
            _fake_commit_message
        )

    def tearDown(self) -> None:
        _commit_message_module.generate_commit_message_from_diff = (  # type: ignore[assignment]
            self._orig_gen
        )
        super().tearDown()

    def _mark_non_wt_task(self, repo_root: Path | None) -> None:
        """Register a tab running a non-worktree task on *repo_root*.

        Mirrors exactly what ``_run_task_inner`` records for a running
        non-worktree task: ``is_running_non_wt = True`` plus the
        resolved repo root of its ``work_dir`` (``None`` when the task
        is not inside any git repository).
        """
        with self.server._state_lock:
            state = agent_state.find_by_tab(_OTHER_TAB)
            if state is None:
                state = AgentState(
                    "other-task-key", tab_id=_OTHER_TAB, server_owned=True,
                )
                agent_state.register(state)
            state.is_task_active = True
            state.is_running_non_wt = True
            state.non_wt_repo_root = repo_root.resolve() if repo_root else None

    def _run_worktree_task_with_changes(self) -> None:
        """Run one worktree task (autoCommit on) that creates a file."""
        self._original_run = _patch_parent_run_create_file("agent_out.txt")
        self.server._run_task_inner({
            "prompt": "worktree task with changes",
            "workDir": self.repo,
            "tabId": _WT_TAB,
            "useWorktree": True,
            "autoCommit": True,
            "model": "",
        })

    def _worktree_results(self) -> list[dict]:
        return [e for e in self.events if e["type"] == "worktree_result"]


class TestMergeNotBlockedByUnrelatedTasks(_RepoAwareGuardBase):
    """REPRODUCES THE BUG — unrelated tasks must not block the merge."""

    def test_merge_succeeds_while_task_runs_in_other_repo(self) -> None:
        """A non-worktree task in a *different* repository never
        touches this repo's main working tree, so the post-task
        auto-merge must go through."""
        self._mark_non_wt_task(Path(self.other_repo))
        self._run_worktree_task_with_changes()

        results = self._worktree_results()
        assert results and results[-1]["success"], (
            "BUG: the auto-merge was refused although the other task "
            f"runs in a different repository: {results}"
        )
        assert (Path(self.repo) / "agent_out.txt").exists(), (
            "the merged file must land in the main working tree"
        )
        assert _list_kiss_wt_branches(self.repo) == [], (
            "the merged task branch must be deleted"
        )

    def test_merge_succeeds_while_task_runs_inside_linked_worktree(
        self,
    ) -> None:
        """The reported scenario: the other task's ``work_dir`` is a
        linked ``.kiss-worktrees`` worktree of the SAME repo.  Its
        ``git rev-parse --show-toplevel`` is the worktree directory,
        not the main tree, so it must not block the merge."""
        linked_wt = Path(self.repo) / ".kiss-worktrees" / "kiss_wt-other"
        self._mark_non_wt_task(linked_wt)
        self._run_worktree_task_with_changes()

        results = self._worktree_results()
        assert results and results[-1]["success"], (
            "BUG: 'Another tab is running a task on the main working "
            "tree' although that task runs inside a linked worktree: "
            f"{results}"
        )
        assert (Path(self.repo) / "agent_out.txt").exists()

    def test_merge_succeeds_while_task_runs_outside_any_repo(self) -> None:
        """A non-worktree task in a non-git directory records no repo
        root at all and can never occupy a main working tree."""
        self._mark_non_wt_task(None)
        self._run_worktree_task_with_changes()

        results = self._worktree_results()
        assert results and results[-1]["success"], (
            "BUG: a task outside any git repository blocked the "
            f"merge: {results}"
        )

    def test_user_merge_click_succeeds_while_task_runs_in_other_repo(
        self,
    ) -> None:
        """The user-initiated ``worktreeAction merge`` goes through
        ``_check_worktree_busy`` — the exact surface that produced the
        reported message — and must not be refused either."""
        # Strand a pending worktree first: the same-main-tree guard
        # legitimately refuses the post-task auto-merge.
        self._mark_non_wt_task(Path(self.repo))
        self._run_worktree_task_with_changes()
        wt_state = agent_state.find_by_tab(_WT_TAB)
        agent = wt_state.agent if wt_state is not None else None
        assert agent is not None and agent._wt_pending, (
            "precondition: the worktree must still be pending"
        )

        # The same-repo task ends; a task in ANOTHER repo is running.
        self._mark_non_wt_task(Path(self.other_repo))
        self.events.clear()
        result = self.server._handle_worktree_action("merge", _WT_TAB)

        assert result["success"], (
            "BUG: the user's Merge click was refused with "
            f"{result['message']!r} although the running task is in a "
            "different repository"
        )
        assert (Path(self.repo) / "agent_out.txt").exists()


class TestMergeBlockedByTaskInsideOwnWorktree(_RepoAwareGuardBase):
    """A task running INSIDE the pending worktree still blocks its merge.

    Merging (and discarding) removes the worktree directory; doing so
    while another tab's non-worktree task runs inside that directory
    would delete the running task's working tree (gpt-5.6-sol review
    finding).  Only the SAME worktree blocks — a different linked
    worktree of the same repo does not (covered above).
    """

    def test_user_merge_refused_while_task_runs_inside_this_worktree(
        self,
    ) -> None:
        # Strand a pending worktree first (same-main-tree guard
        # legitimately refuses the post-task auto-merge).
        self._mark_non_wt_task(Path(self.repo))
        self._run_worktree_task_with_changes()
        wt_state = agent_state.find_by_tab(_WT_TAB)
        agent = wt_state.agent if wt_state is not None else None
        assert agent is not None and agent._wt_pending, (
            "precondition: the worktree must still be pending"
        )
        wt_dir = agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()

        # The main-tree task ends; a task INSIDE this worktree starts
        # (a linked worktree's `git rev-parse --show-toplevel` is the
        # worktree directory itself).
        self._mark_non_wt_task(wt_dir)
        result = self.server._handle_worktree_action("merge", _WT_TAB)

        assert not result["success"], (
            "merging must be refused while a task runs inside the "
            f"worktree the merge would remove: {result}"
        )
        assert "worktree" in result["message"], result
        assert wt_dir.exists(), (
            "the occupied worktree directory must not be removed"
        )

    def test_user_discard_refused_while_task_runs_inside_this_worktree(
        self,
    ) -> None:
        self._mark_non_wt_task(Path(self.repo))
        self._run_worktree_task_with_changes()
        wt_state = agent_state.find_by_tab(_WT_TAB)
        agent = wt_state.agent if wt_state is not None else None
        assert agent is not None and agent._wt_pending
        wt_dir = agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()

        self._mark_non_wt_task(wt_dir)
        result = self.server._handle_worktree_action("discard", _WT_TAB)

        assert not result["success"], (
            "discarding must be refused while a task runs inside the "
            f"worktree the discard would remove: {result}"
        )
        assert wt_dir.exists(), (
            "the occupied worktree directory must not be removed"
        )

    def test_internal_auto_merge_refused_while_worktree_occupied(
        self,
    ) -> None:
        """The post-task auto-merge (``internal=True``) must honor the
        same occupancy guard."""
        self._mark_non_wt_task(Path(self.repo))
        self._run_worktree_task_with_changes()
        wt_state = agent_state.find_by_tab(_WT_TAB)
        agent = wt_state.agent if wt_state is not None else None
        assert agent is not None and agent._wt_pending
        wt_dir = agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()

        self._mark_non_wt_task(wt_dir)
        result = self.server._handle_worktree_action(
            "merge", _WT_TAB, internal=True,
        )

        assert not result["success"], (
            "the internal auto-merge must be refused while a task "
            f"runs inside the worktree: {result}"
        )
        assert wt_dir.exists()


class TestMergeStillBlockedOnSameMainTree(_RepoAwareGuardBase):
    """NON-REGRESSION — a task really writing this main tree blocks."""

    def test_merge_refused_while_task_runs_on_same_main_tree(self) -> None:
        """The guard must keep refusing when the non-worktree task
        occupies the very main working tree the merge would write."""
        self._mark_non_wt_task(Path(self.repo))
        self._run_worktree_task_with_changes()

        results = self._worktree_results()
        assert results and not results[-1]["success"], (
            "a task on the SAME main tree must still block the merge: "
            f"{results}"
        )
        assert "main working tree" in results[-1]["message"]
        assert not (Path(self.repo) / "agent_out.txt").exists(), (
            "the merge must not have touched the busy main tree"
        )
        assert _list_kiss_wt_branches(self.repo), (
            "the unmerged branch must be preserved for a later merge"
        )


class TestNonWorktreeTaskAdmission(_RepoAwareGuardBase):
    """The reverse gate: a worktree merge only blocks tasks on its repo."""

    def _mark_wt_merge_in_progress(self, repo_root: str) -> None:
        """Register a tab mid-way through a worktree merge on *repo_root*."""
        with self.server._state_lock:
            tab = agent_state.find_by_tab(_OTHER_TAB)
            if tab is None:
                tab = AgentState(
                    "other-task-key", tab_id=_OTHER_TAB, server_owned=True,
                )
                agent_state.register(tab)
        agent = WorktreeSorcarAgent("merge-holder")
        agent._wt = GitWorktree(
            repo_root=Path(repo_root),
            branch="kiss/wt-merge-holder",
            original_branch="main",
            wt_dir=Path(repo_root) / ".kiss-worktrees" / "kiss_wt-merge-holder",
            baseline_commit=None,
        )
        with self.server._state_lock:
            tab.agent = agent
            tab.use_worktree = True
            tab.is_merging = True

    def _start_non_wt_task(self, work_dir: str) -> None:
        self._original_run = _patch_parent_run_create_file("direct_out.txt")
        self.server._run_task_inner({
            "prompt": "direct task",
            "workDir": work_dir,
            "tabId": _WT_TAB,
            "useWorktree": False,
            "autoCommit": False,
            "model": "",
        })

    def test_task_in_other_repo_not_refused_by_worktree_merge(self) -> None:
        """A worktree merge in repo A must not refuse a direct task in
        repo B."""
        self._mark_wt_merge_in_progress(self.repo)
        self._start_non_wt_task(self.other_repo)

        errors = [e for e in self.events if e["type"] == "error"]
        assert not any(
            "worktree merge is in progress" in e.get("text", "")
            for e in errors
        ), f"BUG: the unrelated task was refused: {errors}"
        assert (Path(self.other_repo) / "direct_out.txt").exists(), (
            "the direct task must actually have run"
        )

    def test_task_inside_merging_worktree_refused(self) -> None:
        """A task whose ``work_dir`` is INSIDE the worktree being
        merged must be refused: the merge removes that directory
        (gpt-5.6-sol review finding).  The worktree's own toplevel is
        the worktree directory, not the main repo, so the repo-root
        comparison alone would wrongly admit it."""
        repo = Path(self.repo)
        branch = "kiss/wt-merge-holder-real"
        wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
        assert GitWorktreeOps.create(repo, branch, wt_dir), (
            "precondition: a real linked worktree must exist"
        )

        with self.server._state_lock:
            tab = agent_state.find_by_tab(_OTHER_TAB)
            if tab is None:
                tab = AgentState(
                    "other-task-key", tab_id=_OTHER_TAB, server_owned=True,
                )
                agent_state.register(tab)
        agent = WorktreeSorcarAgent("merge-holder")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        )
        with self.server._state_lock:
            tab.agent = agent
            tab.use_worktree = True
            tab.is_merging = True

        self._start_non_wt_task(str(wt_dir))

        errors = [e for e in self.events if e["type"] == "error"]
        assert any(
            "worktree merge is in progress" in e.get("text", "")
            for e in errors
        ), (
            "BUG: a task was admitted into the very worktree a merge "
            f"is about to remove: {errors}"
        )
        assert not (wt_dir / "direct_out.txt").exists()

    def test_task_on_same_repo_still_refused_by_worktree_merge(self) -> None:
        """NON-REGRESSION: a direct task on the repo being merged is
        still refused."""
        self._mark_wt_merge_in_progress(self.repo)
        self._start_non_wt_task(self.repo)

        errors = [e for e in self.events if e["type"] == "error"]
        assert any(
            "worktree merge is in progress" in e.get("text", "")
            for e in errors
        ), f"a direct task on the merging repo must be refused: {self._types()}"
        assert not (Path(self.repo) / "direct_out.txt").exists()


class TestNonWtRepoRootRecording(_RepoAwareGuardBase):
    """End-to-end: the running non-worktree task records its repo root."""

    def test_running_direct_task_records_and_clears_repo_root(self) -> None:
        """While a direct task runs, its tab must expose the resolved
        repo root to the guard; after it ends both fields are clear."""
        observed: dict[str, object] = {}
        server = self.server
        repo = Path(self.repo)
        other = Path(self.other_repo)
        original = _patch_parent_run_create_file(None)
        self._original_run = original

        def probing_run(self_agent: object, **kwargs: object) -> str:
            tab = agent_state.find_by_tab(_WT_TAB)
            assert tab is not None
            with server._state_lock:
                observed["flag"] = tab.is_running_non_wt
                observed["root"] = tab.non_wt_repo_root
                observed["busy_same"] = server._any_non_wt_running(repo)
                observed["busy_other"] = server._any_non_wt_running(other)
            return "success: true\nsummary: stub\n"

        parent = type(self)._patched_parent()
        setattr(parent, "run", probing_run)  # noqa: B010 — mypy: "type" has no attr "run"

        self.server._run_task_inner({
            "prompt": "direct task",
            "workDir": self.repo,
            "tabId": _WT_TAB,
            "useWorktree": False,
            "autoCommit": False,
            "model": "",
        })

        assert observed["flag"] is True
        assert observed["root"] == repo.resolve()
        assert observed["busy_same"] is True, (
            "the guard must see the direct task on its own repo"
        )
        assert observed["busy_other"] is False, (
            "the guard must NOT see it from an unrelated repo"
        )
        tab = agent_state.find_by_tab(_WT_TAB)
        assert tab is not None
        assert tab.is_running_non_wt is False
        assert tab.non_wt_repo_root is None

    @classmethod
    def _patched_parent(cls) -> type:
        """The direct parent class whose ``run`` the harness stubs."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent
        return SorcarAgent.__mro__[1]


if __name__ == "__main__":  # pragma: no cover
    import unittest
    unittest.main()
