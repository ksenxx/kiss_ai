# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests verifying fixes for bugs found in worktree audit round 4.

Each test verifies that the bug has been fixed:

BUG-12: squash_merge_from_baseline now checks commit return code
BUG-13 / BUG-18 (obsolete): covered the auto-release merge on the next
         task (``_release_worktree``), removed with the auto-merge; a
         pending worktree is now committed and kept for review instead
BUG-14: _new_chat surfaces the pending _merge_conflict_warning
BUG-15: concurrent worktree operations use per-repo locking
BUG-16: _finalize_worktree preserves worktree when the commit fails
BUG-17 (obsolete): covered _save_untracked_base, removed together with
         the interactive diff/merge review workflow
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    MergeResult,
    repo_lock,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _WorktreeCleanupOutcome
from kiss.tests.conftest import seed_worktree


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
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True,
        check=True,
    )
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


class TestBug12SquashMergeFromBaselineChecksCommit:
    """squash_merge_from_baseline returns MERGE_FAILED when git commit fails."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        hook = self.repo / ".git" / "hooks" / "pre-commit"
        if hook.exists():
            hook.unlink()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_merge_failed_when_commit_rejected(self) -> None:
        """FIX: squash_merge_from_baseline returns MERGE_FAILED when
        the commit is rejected by a pre-commit hook.
        """
        repo = self.repo

        wt_dir = repo / ".kiss-worktrees" / "test_wt"
        assert GitWorktreeOps.create(repo, "kiss/wt-test", wt_dir)

        (wt_dir / "dirty.txt").write_text("dirty")
        GitWorktreeOps.commit_all(wt_dir, "baseline")
        baseline = GitWorktreeOps.head_sha(wt_dir)
        assert baseline is not None

        (wt_dir / "agent.txt").write_text("agent work")
        GitWorktreeOps.commit_all(wt_dir, "agent work")

        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        hooks_dir = repo / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        result = GitWorktreeOps.squash_merge_from_baseline(
            repo, "kiss/wt-test", baseline,
        )

        hook.unlink()

        assert result == MergeResult.MERGE_FAILED, (
            "squash_merge_from_baseline should return MERGE_FAILED "
            "when commit is rejected by pre-commit hook"
        )

    def test_full_flow_does_not_delete_branch_on_commit_failure(self) -> None:
        """FIX: The full merge() flow does NOT delete the source branch
        when squash_merge_from_baseline's commit is rejected.
        """
        repo = self.repo
        saved = _redirect_db(self.tmpdir)
        orig = _patch_super_run()
        try:
            agent = ChatSorcarAgent("test")
            agent._chat_id = "test-chat-12"

            wt_dir = repo / ".kiss-worktrees" / "test_wt"
            assert GitWorktreeOps.create(repo, "kiss/wt-test12", wt_dir)
            GitWorktreeOps.save_original_branch(repo, "kiss/wt-test12", "main")

            (wt_dir / "dirty.txt").write_text("dirty")
            GitWorktreeOps.commit_all(wt_dir, "baseline")
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None
            GitWorktreeOps.save_baseline_commit(repo, "kiss/wt-test12", baseline)

            (wt_dir / "work.txt").write_text("important work")
            GitWorktreeOps.commit_all(wt_dir, "important agent work")

            seed_worktree(agent, GitWorktree(
                repo_root=repo,
                branch="kiss/wt-test12",
                original_branch="main",
                wt_dir=wt_dir,
                baseline_commit=baseline,
            ))

            hooks_dir = repo / ".git" / "hooks"
            hooks_dir.mkdir(parents=True, exist_ok=True)
            hook = hooks_dir / "pre-commit"
            hook.write_text("#!/bin/sh\nexit 1\n")
            hook.chmod(0o755)

            msg = agent.merge()
            hook.unlink()

            assert "Successfully merged" not in msg, (
                "merge() should NOT report success when commit fails"
            )
            assert GitWorktreeOps.branch_exists(repo, "kiss/wt-test12"), (
                "branch must be preserved when commit fails"
            )
        finally:
            hook_path = repo / ".git" / "hooks" / "pre-commit"
            if hook_path.exists():
                hook_path.unlink()
            _unpatch_super_run(orig)
            _restore_db(saved)


class TestBug15RepoLockIsPerRepoAndReentrant:
    """Concurrent worktree operations on one repo share a re-entrant lock."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.saved = _redirect_db(self.tmpdir)
        self.orig = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.orig)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_repo_lock_exists_and_serializes(self) -> None:
        """FIX: repo_lock returns a per-repo threading.Lock that
        serializes concurrent operations.
        """
        repo = self.repo
        lock1 = repo_lock(repo)
        lock2 = repo_lock(repo)

        assert lock1 is lock2, (
            "repo_lock must return the same lock for the same repo"
        )

        other = Path(self.tmpdir) / "other"
        other.mkdir()
        lock3 = repo_lock(other)
        assert lock3 is not lock1, (
            "repo_lock must return different locks for different repos"
        )

        assert lock1.acquire(blocking=False)
        try:
            assert lock1.acquire(blocking=False), (
                "repo_lock must be re-entrant (RLock)"
            )
            lock1.release()
        finally:
            lock1.release()


class TestBug16FinalizePreservesWorktreeOnCommitFailure:
    """_finalize_worktree returns False and keeps the worktree directory
    when auto-commit is rejected by a pre-commit hook.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.saved = _redirect_db(self.tmpdir)
        self.orig = _patch_super_run()

    def teardown_method(self) -> None:
        hook = self.repo / ".git" / "hooks" / "pre-commit"
        if hook.exists():
            hook.unlink()
        _unpatch_super_run(self.orig)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


    def test_finalize_returns_true_on_success(self) -> None:
        """Regression: _finalize_worktree returns True on normal success."""
        repo = self.repo
        agent = ChatSorcarAgent("test")
        agent._chat_id = "test-chat-16b"

        wt_dir = repo / ".kiss-worktrees" / "test_wt16b"
        assert GitWorktreeOps.create(repo, "kiss/wt-test16b", wt_dir)
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-test16b", "main")

        (wt_dir / "work.txt").write_text("work")
        GitWorktreeOps.commit_all(wt_dir, "committed work")

        seed_worktree(agent, GitWorktree(
            repo_root=repo,
            branch="kiss/wt-test16b",
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        ))

        result = agent._finalize_worktree()

        assert result is _WorktreeCleanupOutcome.COMMITTED_AND_REMOVED, (
            "_finalize_worktree should report COMMITTED_AND_REMOVED on successful cleanup"
        )
        assert not wt_dir.exists(), (
            "Worktree directory should be removed on success"
        )

    def test_merge_reports_autocommit_failure(self) -> None:
        """FIX: merge() reports auto-commit failure instead of proceeding."""
        repo = self.repo
        agent = ChatSorcarAgent("test")
        agent._chat_id = "test-chat-16c"

        wt_dir = repo / ".kiss-worktrees" / "test_wt16c"
        assert GitWorktreeOps.create(repo, "kiss/wt-test16c", wt_dir)
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-test16c", "main")

        (wt_dir / "important.txt").write_text("critical work")

        hooks_dir = repo / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        seed_worktree(agent, GitWorktree(
            repo_root=repo,
            branch="kiss/wt-test16c",
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        ))

        msg = agent.merge()
        hook.unlink()

        assert "Cannot merge" in msg or "auto-commit" in msg.lower(), (
            "merge() should report auto-commit failure"
        )
        assert wt_dir.exists(), (
            "Worktree should be preserved when auto-commit fails"
        )
