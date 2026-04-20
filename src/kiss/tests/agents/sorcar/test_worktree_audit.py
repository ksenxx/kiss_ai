"""Audit tests for worktree mode: confirming bugs and inconsistencies.

Each test targets a specific bug or inconsistency found during code review.
Tests are labeled BUG-N or INCONSISTENCY-N for traceability.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    _git,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

# ---------------------------------------------------------------------------
# Helpers
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
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True, check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True, check=True,
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


# ---------------------------------------------------------------------------
# BUG-1: commit_all doesn't check git commit return code
# ---------------------------------------------------------------------------


class TestBug1CommitAllIgnoresFailure:
    """commit_all returns True even when 'git commit' fails.

    If ``git commit`` fails (e.g. pre-commit hook rejection, permissions),
    ``commit_all`` still returns True because it doesn't check the commit
    subprocess's return code. This means callers believe the commit
    succeeded when it didn't.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_commit_all_returns_true_when_commit_fails(self) -> None:
        """BUG-1: commit_all returns True even when git commit fails.

        We install a pre-commit hook that always rejects commits, then
        call commit_all. It stages changes and reports True (committed),
        but the commit was actually rejected by the hook.
        """
        # Install a pre-commit hook that rejects all commits
        hooks_dir = self.repo / ".git" / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        # Make a change
        (self.repo / "new_file.txt").write_text("content")

        # commit_all should detect this isn't committed
        result = GitWorktreeOps.commit_all(self.repo, "test commit")

        # BUG: returns True (thinks commit succeeded) but commit was rejected
        assert result is True, (
            "commit_all returns True because it doesn't check git commit rc"
        )

        # Verify the commit was actually rejected (nothing was committed)
        log = _git("log", "--oneline", cwd=self.repo)
        commit_count = len(log.stdout.strip().splitlines())
        assert commit_count == 1, (
            "Only the initial commit should exist; the hook rejected the new one"
        )


# ---------------------------------------------------------------------------
# BUG-2: _check_merge_conflict doesn't account for uncommitted changes
# ---------------------------------------------------------------------------


class TestBug2ConflictCheckMissesUncommitted:
    """_check_merge_conflict returns False even when uncommitted worktree
    changes would conflict with the original branch.

    Since _check_merge_conflict compares the worktree *branch* (not
    working tree) against the original branch, and the agent's changes
    aren't committed yet, the check always reports "no conflict".
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_conflict_check_misses_uncommitted_worktree_changes(self) -> None:
        """BUG-2: would_merge_conflict returns False for uncommitted conflicts.

        Steps:
        1. Agent creates a worktree (branch forked from original)
        2. Modify README.md in the worktree (uncommitted)
        3. Modify README.md on the original branch (committed)
        4. would_merge_conflict compares branch vs original — but branch
           has no commits, so it's comparing original with itself → False
        """
        agent = WorktreeSorcarAgent("test")
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None

        # 1. Modify README.md in worktree (uncommitted)
        (wt_dir / "README.md").write_text("worktree conflicting change\n")

        # 2. Modify README.md on original branch (committed)
        (self.repo / "README.md").write_text("main conflicting change\n")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-m", "conflict on main", cwd=self.repo)

        # 3. Check: would_merge_conflict compares branch (no commits)
        #    vs original (has new commit). Since the branch has no commits
        #    beyond the fork point, git sees no tree difference.
        wt = agent._wt
        assert wt is not None
        has_conflict = GitWorktreeOps.would_merge_conflict(
            wt.repo_root, wt.original_branch, wt.branch,  # type: ignore[arg-type]
        )

        # BUG: This returns False because branch has no commits
        # (the conflicting change is uncommitted in the worktree)
        assert has_conflict is False, (
            "would_merge_conflict returns False because uncommitted "
            "changes aren't on the branch yet"
        )

        # But after auto-commit and merge, there IS a real conflict
        GitWorktreeOps.commit_all(wt_dir, "commit worktree changes")
        actual_conflict = GitWorktreeOps.would_merge_conflict(
            wt.repo_root, wt.original_branch, wt.branch,  # type: ignore[arg-type]
        )
        assert actual_conflict is True, (
            "After committing, the conflict is detectable"
        )

        agent.discard()


# ---------------------------------------------------------------------------
# BUG-3: merge() conflict instructions suggest wrong merge strategy
# ---------------------------------------------------------------------------


class TestBug3MergeInstructionsInconsistency:
    """merge() conflict instructions tell user to run 'git merge' but
    the agent uses 'git merge --squash'. Following the manual instructions
    would produce a merge commit instead of a squash commit.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_conflict_instructions_say_merge_not_squash(self) -> None:
        """BUG-3: Conflict instructions use 'git merge' not 'git merge --squash'.

        The agent merges using squash_merge_branch (git merge --squash),
        but the conflict recovery instructions tell the user to run
        'git merge <branch>' (a regular merge), which would create a
        merge commit — a different history shape than the agent intended.
        """
        agent = WorktreeSorcarAgent("test")
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "README.md").write_text("worktree change\n")
        GitWorktreeOps.stage_all(wt_dir)
        GitWorktreeOps.commit_all(wt_dir, "wt conflict")

        # Create conflicting change on original
        (self.repo / "README.md").write_text("main change\n")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-m", "main conflict", cwd=self.repo)

        msg = agent.merge()
        assert "Merge conflict" in msg

        # BUG: Instructions say "git merge" but agent uses squash merge
        branch = agent._wt_branch
        assert branch is not None
        assert f"git merge {branch}" in msg
        assert "git merge --squash" not in msg, (
            "Instructions should say 'git merge --squash' to match "
            "agent behavior, but they say 'git merge'"
        )

        agent.discard()


# ---------------------------------------------------------------------------
# INCONSISTENCY-1: _release_worktree duplicates _finalize_worktree logic
# ---------------------------------------------------------------------------


class TestInconsistency1DuplicateFinalization:
    """_release_worktree duplicates the exact same auto-commit + remove +
    prune logic that _finalize_worktree provides. If one is updated
    without the other, they'll diverge.
    """

    def test_release_and_finalize_have_duplicate_code(self) -> None:
        """INCONSISTENCY-1: _release_worktree inlines _finalize_worktree logic.

        Both methods do:
            if wt.wt_dir.exists():
                self._auto_commit_worktree()
                GitWorktreeOps.remove(wt.repo_root, wt.wt_dir)
            GitWorktreeOps.prune(wt.repo_root)

        _release_worktree should call _finalize_worktree() instead.
        """
        import inspect

        finalize_src = inspect.getsource(WorktreeSorcarAgent._finalize_worktree)
        release_src = inspect.getsource(WorktreeSorcarAgent._release_worktree)

        # Both contain the same worktree removal pattern
        assert "self._auto_commit_worktree()" in finalize_src
        assert "self._auto_commit_worktree()" in release_src
        assert "GitWorktreeOps.remove(" in finalize_src
        assert "GitWorktreeOps.remove(" in release_src
        assert "GitWorktreeOps.prune(" in finalize_src
        assert "GitWorktreeOps.prune(" in release_src

        # But _release_worktree doesn't call _finalize_worktree
        assert "_finalize_worktree" not in release_src, (
            "_release_worktree duplicates _finalize_worktree instead of "
            "calling it"
        )


# ---------------------------------------------------------------------------
# BUG-4: merge() after conflict still allows re-call (no clear error)
# ---------------------------------------------------------------------------


class TestBug4MergeRetryAfterConflict:
    """After merge() returns a conflict message, self._wt remains set.
    Calling merge() again repeats the entire finalize + squash-merge cycle
    without telling the user it's a retry or that the worktree dir is gone.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_merge_can_be_called_twice_after_conflict(self) -> None:
        """BUG-4: merge() doesn't clear self._wt on conflict, so it can
        be called again. The second call tries the full cycle again
        (finalize + squash merge), but the worktree dir is already gone.
        """
        agent = WorktreeSorcarAgent("test")
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "README.md").write_text("worktree change\n")
        GitWorktreeOps.stage_all(wt_dir)
        GitWorktreeOps.commit_all(wt_dir, "wt conflict")

        (self.repo / "README.md").write_text("main change\n")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-m", "main conflict", cwd=self.repo)

        msg1 = agent.merge()
        assert "Merge conflict" in msg1
        assert agent._wt_pending  # still pending after conflict

        # The worktree dir is already removed by _finalize_worktree
        assert not wt_dir.exists()

        # But merge() can be called again (no RuntimeError)
        msg2 = agent.merge()
        assert "Merge conflict" in msg2  # same conflict

        # Eventually discard to clean up
        agent.discard()
