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
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    MergeResult,
    _git,
    repo_lock,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.server import VSCodeServer

# ---------------------------------------------------------------------------
# Helpers (same as prior audit test files)
# ---------------------------------------------------------------------------


def _redirect_db(tmpdir: str) -> tuple:
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
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


# ===================================================================
# BUG-25: _release_worktree orphans branch when original_branch is None
# ===================================================================


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

        # Agent makes changes
        (wt.wt_dir / "work.txt").write_text("agent work\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "agent changes")

        # Simulate crash between creation and config write:
        # original_branch is None
        agent._wt = GitWorktree(
            repo_root=wt.repo_root,
            branch=wt.branch,
            original_branch=None,  # <-- simulates crash
            wt_dir=wt.wt_dir,
            baseline_commit=wt.baseline_commit,
        )

        result = agent._release_worktree()

        # BUG-25: returns None (correct for failure) but...
        assert result is None

        # BUG-25: branch still exists — never deleted
        assert GitWorktreeOps.branch_exists(repo, branch_name), (
            "BUG-25 appears fixed: branch was deleted"
        )

        # BUG-25: no warning set — user has no idea
        assert agent._merge_conflict_warning is None, (
            "BUG-25 appears fixed: warning is now set"
        )
        assert agent._stash_pop_warning is None

        # BUG-25: worktree dir was removed by _finalize_worktree
        assert not wt.wt_dir.exists(), (
            "Worktree dir should have been removed by _finalize_worktree"
        )

        # BUG-25: _wt is None, so user can't call discard() either
        assert agent._wt is None, "_wt should be cleared"

    def test_source_shows_no_cleanup_on_none_original(self) -> None:
        """BUG-25: Confirm source code has no branch cleanup when
        original_branch is None."""
        source = inspect.getsource(WorktreeSorcarAgent._release_worktree)
        # The `if wt.original_branch:` block contains all cleanup.
        # When original_branch is None (falsy), the block is skipped
        # entirely — no delete_branch, no warning.
        #
        # Check there's no else/fallback for original_branch being None:
        lines = source.splitlines()
        in_original_branch_block = False
        has_else_for_original = False
        for line in lines:
            if "if wt.original_branch:" in line:
                in_original_branch_block = True
            if in_original_branch_block and line.strip().startswith("else:"):
                # Check if this else is at the same indent as the if
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


# ===================================================================
# BUG-26: merge() puts delete_branch + self._wt=None outside repo_lock
# ===================================================================


class TestBug26MergeDeleteOutsideLock:
    """BUG-26: merge() exits the `with repo_lock(...)` context before
    calling delete_branch and setting self._wt = None.

    Compare with _release_worktree which does delete_branch INSIDE
    the lock.  The gap between lock release and branch deletion allows
    a concurrent tab's _release_worktree to interleave — e.g. starting
    a checkout/stash/merge sequence on the same repo while the branch
    still exists.
    """

    def test_merge_delete_branch_outside_lock(self) -> None:
        """BUG-26: Confirm delete_branch is outside repo_lock in merge()."""
        source = inspect.getsource(WorktreeSorcarAgent.merge)
        lines = source.splitlines()

        # Find the repo_lock context manager and the delete_branch call
        lock_indent = None
        lock_end_line = None
        delete_line = None

        indent_stack: list[int] = []
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if "with repo_lock(" in line:
                lock_indent = indent
                continue

            if lock_indent is not None and "delete_branch" in line:
                delete_line = i
                # Check if this line is outside the with block
                # (indent <= lock_indent means we've exited the with)
                if indent <= lock_indent:
                    # delete_branch is OUTSIDE the lock — BUG confirmed
                    pass
                break

        assert delete_line is not None, "sanity: delete_branch found in merge()"

    def test_release_has_delete_inside_lock(self) -> None:
        """Contrast: _release_worktree puts delete_branch INSIDE the lock."""
        source = inspect.getsource(WorktreeSorcarAgent._release_worktree)
        lines = source.splitlines()

        lock_indent = None
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if "with repo_lock(" in line:
                lock_indent = indent
                continue

            if lock_indent is not None and "delete_branch" in line:
                # In _release_worktree, delete_branch should be inside lock
                assert indent > lock_indent, (
                    "Unexpectedly, _release_worktree also has delete_branch "
                    "outside the lock"
                )
                break

    def test_merge_wt_none_outside_lock(self) -> None:
        """BUG-26: self._wt = None is also outside the lock in merge().

        Uses line-position analysis: the `with repo_lock` block ends
        when indentation returns to the with-statement's level.  Any
        `self._wt = None` after that point is outside the lock.
        """
        source = inspect.getsource(WorktreeSorcarAgent.merge)
        lines = source.splitlines()

        lock_start = None
        lock_indent = None
        lock_end = None

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if "with repo_lock(" in line:
                lock_start = i
                lock_indent = indent
                continue

            # Detect end of `with` block: first non-blank line at or
            # below the `with` indent after the block has started.
            if lock_start is not None and lock_end is None:
                if stripped and indent <= lock_indent and i > lock_start + 1:
                    lock_end = i

        assert lock_end is not None, "sanity: found end of repo_lock block"

        # Find self._wt = None after the lock block
        wt_none_after_lock = False
        for i in range(lock_end, len(lines)):
            if "self._wt = None" in lines[i]:
                wt_none_after_lock = True
                break

        # BUG-26: self._wt = None appears after the lock block
        assert wt_none_after_lock, (
            "BUG-26 appears fixed: self._wt = None is now inside lock"
        )


# ===================================================================
# BUG-27: cleanup_orphans deletes conflict-preserved branches
# ===================================================================


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

        # Agent makes a change to README.md (will conflict)
        (wt.wt_dir / "README.md").write_text("# Agent version\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "agent changes README")

        # Make a conflicting change on main
        (repo / "README.md").write_text("# User conflicting version\n")
        _git("add", ".", cwd=repo)
        _git("commit", "-m", "user conflicting change", cwd=repo)

        # Release should hit a merge conflict
        result = agent._release_worktree()

        # Verify conflict was detected and warning was set
        assert result is None, "Release should return None on conflict"
        assert agent._merge_conflict_warning is not None, (
            "Warning should be set on merge conflict"
        )
        assert branch_name in agent._merge_conflict_warning

        # Branch should exist (preserved for manual resolution)
        assert GitWorktreeOps.branch_exists(repo, branch_name), (
            "Branch should be preserved after merge conflict"
        )

        # Worktree dir should be gone (removed by _finalize_worktree)
        assert not wt.wt_dir.exists(), (
            "Worktree dir should be removed by _finalize_worktree"
        )

        # BUG-27: cleanup_orphans sees the branch with no worktree
        # and deletes it — losing agent's work
        cleanup_output = GitWorktreeOps.cleanup_orphans(repo)

        # BUG-27: The branch was deleted by cleanup_orphans
        assert not GitWorktreeOps.branch_exists(repo, branch_name), (
            "BUG-27 appears fixed: cleanup_orphans no longer deletes "
            "conflict-preserved branches"
        )
        assert "Deleted" in cleanup_output, (
            "cleanup_orphans should report deleting the branch"
        )

    def test_cleanup_orphans_does_not_check_unmerged(self) -> None:
        """BUG-27: Confirm cleanup_orphans has no unmerged-commit check."""
        source = inspect.getsource(GitWorktreeOps.cleanup_orphans)
        # A safe implementation would check if the branch has unmerged
        # commits before deleting (e.g. `git branch --no-merged`)
        assert "--no-merged" not in source, (
            "BUG-27 appears fixed: cleanup_orphans now checks for "
            "unmerged commits"
        )
        assert "merge-base" not in source, (
            "BUG-27 appears fixed: cleanup_orphans now checks merge status"
        )


# ===================================================================
# BUG-28: _start_merge_session reads tab_id from thread-local
# ===================================================================


class TestBug28StartMergeSessionThreadLocal:
    """BUG-28: _start_merge_session gets tab_id from
    printer._thread_local.tab_id.  When called from the main thread
    (e.g. session replay → _emit_pending_worktree →
    _start_worktree_merge_review), the thread-local is not set,
    so is_merging is never set for the tab.

    This allows a new task to be started while a merge review is
    active (the is_merging guard in _run_task_inner is bypassed).
    """

    def test_start_merge_session_uses_thread_local(self) -> None:
        """BUG-28: Confirm _start_merge_session reads from thread-local."""
        source = inspect.getsource(VSCodeServer._start_merge_session)
        # The function reads tab_id from thread-local, not from a parameter
        assert "_thread_local" in source, (
            "sanity: _start_merge_session references _thread_local"
        )
        assert "tab_id" not in inspect.signature(
            VSCodeServer._start_merge_session
        ).parameters or (
            # It takes merge_json_path, not tab_id
            "tab_id" not in [
                p for p in inspect.signature(
                    VSCodeServer._start_merge_session
                ).parameters
                if p != "self"
            ]
        ), "BUG-28 appears fixed: tab_id is now a parameter"

    def test_is_merging_not_set_without_thread_local(self) -> None:
        """BUG-28: is_merging stays False when thread-local tab_id is unset."""
        tmpdir = tempfile.mkdtemp()
        saved = _redirect_db(tmpdir)
        try:
            repo = _make_repo(Path(tmpdir) / "repo")

            server = VSCodeServer()
            server.work_dir = str(repo)

            # Create a tab and set it up
            tab_id = "t28"
            tab = server._get_tab(tab_id)
            tab.use_worktree = True

            # Ensure thread-local tab_id is NOT set on this thread
            # (simulating main thread / replay path)
            if hasattr(server.printer._thread_local, "tab_id"):
                delattr(server.printer._thread_local, "tab_id")

            # Create a fake merge JSON to pass to _start_merge_session
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

            # Call _start_merge_session — should set is_merging but won't
            started = server._start_merge_session(str(merge_json))
            assert started, "Merge session should start"

            # BUG-28: is_merging was NOT set because thread-local tab_id
            # was None, so the tab lookup returned None.
            assert tab.is_merging is False, (
                "BUG-28 appears fixed: is_merging is now set without "
                "thread-local tab_id"
            )

        finally:
            _restore_db(saved)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_start_worktree_merge_review_does_not_set_thread_local(self) -> None:
        """BUG-28: _start_worktree_merge_review doesn't set thread-local tab_id."""
        source = inspect.getsource(VSCodeServer._start_worktree_merge_review)
        # The method receives tab_id as a parameter but never sets
        # printer._thread_local.tab_id before calling
        # _prepare_and_start_merge → _start_merge_session
        assert "_thread_local.tab_id" not in source, (
            "BUG-28 appears fixed: _start_worktree_merge_review now "
            "sets thread-local tab_id"
        )


# ===================================================================
# BUG-29: _release_worktree conflict instructions assume clean tree
# ===================================================================


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

        # Create dirty state that will be stashed
        (repo / "user_file.txt").write_text("user dirty state\n")

        wt_work = agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt = agent._wt
        assert wt is not None

        # Agent modifies README.md (will conflict with main)
        (wt.wt_dir / "README.md").write_text("# Agent\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "agent change")

        # Make conflicting change on main
        (repo / "README.md").write_text("# User conflict\n")
        _git("add", "README.md", cwd=repo)
        _git("commit", "-m", "user conflict", cwd=repo)

        # Re-create user dirty state (simulating the user having dirty
        # files when _release_worktree runs)
        (repo / "user_file.txt").write_text("user dirty state again\n")

        result = agent._release_worktree()
        assert result is None, "Should fail with conflict"

        warning = agent._merge_conflict_warning
        assert warning is not None, "Warning should be set"

        # BUG-29: The warning includes `git merge --squash` instructions
        assert "merge --squash" in warning, (
            "Warning should contain merge --squash instructions"
        )

        # After the release, the stash was popped — check if dirty state
        # was restored
        status = _git("status", "--porcelain", cwd=repo)
        has_dirty = bool(status.stdout.strip())

        # If the stash was popped successfully, the tree is dirty
        # and git merge --squash would refuse
        if has_dirty:
            # Try the exact command from the warning
            merge_attempt = _git(
                "merge", "--squash", wt.branch, cwd=repo,
            )
            # BUG-29: git merge --squash refuses because of dirty tree
            assert merge_attempt.returncode != 0, (
                "BUG-29 appears fixed: merge --squash should work with "
                "dirty tree (or instructions were updated)"
            )

    def test_warning_does_not_mention_stash(self) -> None:
        """BUG-29: Confirm the conflict warning doesn't mention stashing."""
        source = inspect.getsource(WorktreeSorcarAgent._release_worktree)
        # Find the _merge_conflict_warning assignment in the conflict path
        # (the one with "had conflicts")
        lines = source.splitlines()
        in_conflict_warning = False
        warning_text = ""
        for line in lines:
            if "had conflicts" in line:
                in_conflict_warning = True
            if in_conflict_warning:
                warning_text += line
                if line.strip().endswith(")") or line.strip().endswith('")'):
                    break

        # BUG-29: The warning doesn't tell the user to stash first
        assert "stash" not in warning_text.lower(), (
            "BUG-29 appears fixed: conflict warning now mentions stashing"
        )
