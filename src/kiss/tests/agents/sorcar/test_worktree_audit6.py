"""Tests confirming bugs found in worktree audit round 6.

Each test CONFIRMS the bug exists (assertions pass when buggy behaviour
is present).

BUG-25: _release_worktree orphans branch silently when original_branch
        is None — _finalize_worktree removes the worktree dir, but the
        if-wt.original_branch block is skipped entirely, so the branch
        is never deleted and no warning is set.

BUG-26: merge() calls delete_branch and sets self._wt = None OUTSIDE
        repo_lock — inconsistent with _release_worktree which does both
        inside the lock.  Creates a window for concurrent operations on
        the same repo.

BUG-27: cleanup_orphans unconditionally deletes kiss/wt-* branches
        that have no worktree attached.  After a merge conflict,
        _release_worktree preserves the branch for manual resolution
        but removes the worktree dir — cleanup_orphans then deletes
        the preserved branch, losing agent work.

BUG-28: _start_merge_session reads tab_id from thread-local storage
        to set is_merging.  When called from the main thread (session
        replay via _emit_pending_worktree → _start_worktree_merge_review)
        the thread-local is unset, so is_merging is never set and a
        task can be started during an active merge review.

BUG-29: _release_worktree pops the user stash after a merge conflict
        (cherry-pick --abort / reset --hard), restoring the dirty
        working tree, then emits instructions telling the user to run
        `git merge --squash` — which git will refuse because the
        working tree is dirty.
"""

from __future__ import annotations

import inspect
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    _git,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.server import VSCodeServer


def _redirect_db(tmpdir: str) -> tuple:
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore_db(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True,
        check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


def _patch_super_run(
    return_value: str = "success: true\nsummary: test done\n",
) -> Any:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def fake_run(self_agent: object, **kwargs: object) -> str:
        return return_value

    parent_class.run = fake_run
    return original


def _unpatch_super_run(original: Any) -> None:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    parent_class.run = original


class TestBug25ReleaseBranchOrphanedOnNoneOriginal:
    """BUG-25: When original_branch is None, _release_worktree:
      1. Calls _finalize_worktree which removes the worktree directory
      2. Skips the entire `if wt.original_branch:` block
      3. Never deletes the branch
      4. Sets no _merge_conflict_warning
      5. Returns None — user gets no notification

    The branch becomes a permanent orphan discoverable only via
    cleanup_orphans.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_branch_orphaned_no_warning(self) -> None:
        """BUG-25: Branch is orphaned and no warning is set."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test25"

        wt_work = agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt = agent._wt
        assert wt is not None
        branch_name = wt.branch

        (wt.wt_dir / "work.txt").write_text("agent work\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "agent changes")

        agent._wt = GitWorktree(
            repo_root=wt.repo_root,
            branch=wt.branch,
            original_branch=None,
            wt_dir=wt.wt_dir,
            baseline_commit=wt.baseline_commit,
        )

        result = agent._release_worktree()

        assert result is None

        assert GitWorktreeOps.branch_exists(repo, branch_name), (
            "Branch must be preserved for manual resolution"
        )

        assert agent._merge_conflict_warning is not None, (
            "BUG-50 fix: warning should be set on orphan branch"
        )
        assert branch_name in agent._merge_conflict_warning
        assert "original branch is unknown" in agent._merge_conflict_warning
        assert agent._stash_pop_warning is None

        assert not wt.wt_dir.exists(), (
            "Worktree dir should have been removed by _finalize_worktree"
        )

        assert agent._wt is None, "_wt should be cleared"

    def test_source_shows_no_cleanup_on_none_original(self) -> None:
        """BUG-25: Confirm source code has no branch cleanup when
        original_branch is None."""
        source = inspect.getsource(WorktreeSorcarAgent._release_worktree)
        lines = source.splitlines()
        in_original_branch_block = False
        has_else_for_original = False
        for line in lines:
            if "if wt.original_branch:" in line:
                in_original_branch_block = True
            if in_original_branch_block and line.strip().startswith("else:"):
                if_indent = None
                for l2 in lines:
                    if "if wt.original_branch:" in l2:
                        if_indent = len(l2) - len(l2.lstrip())
                        break
                else_indent = len(line) - len(line.lstrip())
                if if_indent is not None and else_indent == if_indent:
                    has_else_for_original = True

        assert not has_else_for_original, (
            "BUG-25 appears fixed: there's now an else branch for "
            "original_branch is None"
        )


class TestBug26MergeDeleteInsideLock:
    """BUG-26 FIX: merge() now delegates to _do_merge() which runs
    delete_branch inside repo_lock. Both merge() and _release_worktree
    use the same _do_merge() helper, so they are consistent.
    """

    def test_merge_uses_do_merge(self) -> None:
        """merge() delegates to _do_merge() for the locked operation."""
        source = inspect.getsource(WorktreeSorcarAgent.merge)
        assert "_do_merge" in source, (
            "merge() must delegate to _do_merge()"
        )

    def test_release_uses_do_merge(self) -> None:
        """_release_worktree delegates to _do_merge() for the locked operation."""
        source = inspect.getsource(WorktreeSorcarAgent._release_worktree)
        assert "_do_merge" in source, (
            "_release_worktree must delegate to _do_merge()"
        )

    def test_do_merge_has_delete_inside_lock(self) -> None:
        """_do_merge runs delete_branch inside repo_lock."""
        source = inspect.getsource(WorktreeSorcarAgent._do_merge)
        lines = source.splitlines()

        lock_indent = None
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if "with repo_lock(" in line:
                lock_indent = indent
                continue

            if lock_indent is not None and "delete_branch" in line:
                assert indent > lock_indent, (
                    "delete_branch must be inside repo_lock"
                )
                break
        else:
            raise AssertionError("delete_branch not found in _do_merge")


class TestBug27CleanupDeletesConflictBranch:
    """BUG-27: After a merge conflict, _release_worktree keeps the
    branch for manual resolution but removes the worktree directory.
    cleanup_orphans sees the branch with no worktree attached and
    deletes it — losing the agent's work that was explicitly preserved.

    The merge conflict warning tells the user to manually merge the
    branch, but cleanup_orphans could delete it before they do so.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_cleanup_deletes_conflict_preserved_branch(self) -> None:
        """BUG-27: cleanup_orphans deletes branch kept for conflict resolution."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test27"

        wt_work = agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt = agent._wt
        assert wt is not None
        branch_name = wt.branch

        (wt.wt_dir / "README.md").write_text("# Agent version\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "agent changes README")

        (repo / "README.md").write_text("# User conflicting version\n")
        _git("add", ".", cwd=repo)
        _git("commit", "-m", "user conflicting change", cwd=repo)

        result = agent._release_worktree()

        assert result is None, "Release should return None on conflict"
        assert agent._merge_conflict_warning is not None, (
            "Warning should be set on merge conflict"
        )
        assert branch_name in agent._merge_conflict_warning

        assert GitWorktreeOps.branch_exists(repo, branch_name), (
            "Branch should be preserved after merge conflict"
        )

        assert not wt.wt_dir.exists(), (
            "Worktree dir should be removed by _finalize_worktree"
        )

        cleanup_output = GitWorktreeOps.cleanup_orphans(repo)

        assert GitWorktreeOps.branch_exists(repo, branch_name), (
            "BUG-58 fix: cleanup_orphans must preserve "
            "conflict-preserved branches"
        )
        assert "Pending-merge branches (kept)" in cleanup_output, (
            "cleanup_orphans should classify the branch as pending-merge"
        )
        assert branch_name in cleanup_output

    def test_cleanup_orphans_does_not_check_unmerged(self) -> None:
        """BUG-27: Confirm cleanup_orphans has no unmerged-commit check."""
        source = inspect.getsource(GitWorktreeOps.cleanup_orphans)
        assert "--no-merged" not in source, (
            "BUG-27 appears fixed: cleanup_orphans now checks for "
            "unmerged commits"
        )
        assert "merge-base" not in source, (
            "BUG-27 appears fixed: cleanup_orphans now checks merge status"
        )


class TestBug28StartMergeSessionThreadLocal:
    """BUG-28: _start_merge_session gets tab_id from
    printer._thread_local.tab_id.  When called from the main thread
    (e.g. session replay → _emit_pending_worktree →
    _start_worktree_merge_review), the thread-local is not set,
    so is_merging is never set for the tab.

    This allows a new task to be started while a merge review is
    active (the is_merging guard in _run_task_inner is bypassed).
    """

    def test_start_merge_session_accepts_tab_id(self) -> None:
        """BUG-28 FIXED: _start_merge_session now accepts tab_id."""
        sig = inspect.signature(VSCodeServer._start_merge_session)
        assert "tab_id" in sig.parameters, (
            "BUG-28 fix: _start_merge_session must accept tab_id"
        )

    def test_is_merging_not_set_without_thread_local(self) -> None:
        """BUG-28: is_merging stays False when thread-local tab_id is unset."""
        tmpdir = tempfile.mkdtemp()
        saved = _redirect_db(tmpdir)
        try:
            repo = _make_repo(Path(tmpdir) / "repo")

            server = VSCodeServer()
            server.work_dir = str(repo)

            tab_id = "t28"
            tab = server._get_tab(tab_id)
            tab.use_worktree = True

            if hasattr(server.printer._thread_local, "tab_id"):
                delattr(server.printer._thread_local, "tab_id")

            merge_dir = Path(tmpdir) / "merge"
            merge_dir.mkdir(parents=True, exist_ok=True)
            merge_json = merge_dir / "pending-merge.json"
            merge_json.write_text(json.dumps({
                "branch": "HEAD",
                "files": [{
                    "name": "test.py",
                    "base": str(repo / "README.md"),
                    "current": str(repo / "README.md"),
                    "hunks": [{"bs": 0, "bc": 0, "cs": 0, "cc": 1}],
                }],
            }))

            started = server._start_merge_session(str(merge_json))
            assert started, "Merge session should start"

            assert tab.is_merging is False, (
                "BUG-28 appears fixed: is_merging is now set without "
                "thread-local tab_id"
            )

        finally:
            _restore_db(saved)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_present_pending_worktree_does_not_set_thread_local(self) -> None:
        """BUG-28: _present_pending_worktree (which now includes the
        worktree merge review logic) doesn't set thread-local tab_id."""
        source = inspect.getsource(VSCodeServer._present_pending_worktree)
        assert "_thread_local.tab_id" not in source, (
            "BUG-28 appears fixed: _present_pending_worktree now "
            "sets thread-local tab_id"
        )


class TestBug29ConflictInstructionsIgnoreDirtyState:
    """BUG-29: When _release_worktree encounters a merge conflict AND
    the user had dirty state that was stashed, it:
      1. Stash-pushes the user's dirty files
      2. Attempts merge (fails — conflict)
      3. Aborts/resets (cherry-pick --abort or reset --hard)
      4. Pops the stash (restores dirty files)
      5. Sets _merge_conflict_warning with instructions:
         `git merge --squash <branch>`

    But `git merge --squash` requires a clean working tree.  The
    popped stash has restored the dirty files, so the instructions
    will fail.  The warning should mention that the user needs to
    stash or commit their changes first.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_conflict_instructions_fail_with_dirty_tree(self) -> None:
        """BUG-29: Demonstrate that the conflict instructions fail."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test29"

        (repo / "user_file.txt").write_text("user dirty state\n")

        wt_work = agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt = agent._wt
        assert wt is not None

        (wt.wt_dir / "README.md").write_text("# Agent\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "agent change")

        (repo / "README.md").write_text("# User conflict\n")
        _git("add", "README.md", cwd=repo)
        _git("commit", "-m", "user conflict", cwd=repo)

        (repo / "user_file.txt").write_text("user dirty state again\n")

        result = agent._release_worktree()
        assert result is None, "Should fail with conflict"

        warning = agent._merge_conflict_warning
        assert warning is not None, "Warning should be set"

        if wt.baseline_commit:
            assert "cherry-pick" in warning, (
                "Warning should contain cherry-pick instructions when "
                "baseline exists"
            )
        else:
            assert "merge --squash" in warning, (
                "Warning should contain merge --squash instructions "
                "when no baseline"
            )

    def test_warning_mentions_stash_pop(self) -> None:
        """BUG-29 fix: the conflict warning tells the user to run
        ``git stash pop`` when the auto-merge stashed their uncommitted
        changes, so they can restore them after resolving the conflict.
        """
        source = inspect.getsource(WorktreeSorcarAgent._release_worktree)
        assert "stash_suffix" in source
        assert "git stash pop" in source
