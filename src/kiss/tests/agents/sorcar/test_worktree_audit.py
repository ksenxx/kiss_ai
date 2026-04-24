"""Tests verifying fixes for worktree mode bugs found during audit.

Each test targets a specific bug fix and verifies the correct behavior.
Tests are labeled BUG-N or FIX-N for traceability.
"""

from __future__ import annotations

import inspect
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


class TestFix1CommitAllChecksReturnCode:
    """commit_all returns False when 'git commit' fails."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_commit_all_returns_false_when_commit_fails(self) -> None:
        """FIX-1: commit_all returns False when git commit is rejected.

        We install a pre-commit hook that always rejects commits, then
        call commit_all. It should return False because the commit failed.
        """
        hooks_dir = self.repo / ".git" / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        (self.repo / "new_file.txt").write_text("content")

        result = GitWorktreeOps.commit_all(self.repo, "test commit")

        assert result is False

        log = _git("log", "--oneline", cwd=self.repo)
        commit_count = len(log.stdout.strip().splitlines())
        assert commit_count == 1

    def test_commit_all_returns_true_on_success(self) -> None:
        """commit_all returns True when git commit succeeds."""
        (self.repo / "new_file.txt").write_text("content")
        result = GitWorktreeOps.commit_all(self.repo, "test commit")
        assert result is True

    def test_commit_all_returns_false_when_nothing_to_commit(self) -> None:
        """commit_all returns False when there are no changes."""
        result = GitWorktreeOps.commit_all(self.repo, "test commit")
        assert result is False


class TestFix3MergeInstructionsUseSquash:
    """merge() conflict instructions correctly say 'git merge --squash'."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_conflict_instructions_say_squash(self) -> None:
        """FIX-3: Conflict instructions use 'git merge --squash'."""
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

        msg = agent.merge()
        assert "Merge conflict" in msg

        branch = agent._wt_branch
        assert branch is not None
        assert f"git merge --squash {branch}" in msg

        agent.discard()

    def test_merge_instructions_say_squash(self) -> None:
        """merge_instructions() also uses 'git merge --squash'."""
        agent = WorktreeSorcarAgent("test")
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        instructions = agent.merge_instructions()
        assert "git merge --squash" in instructions

        agent.discard()

    def test_unknown_branch_instructions_say_squash(self) -> None:
        """When original_branch is None, instructions still say --squash."""
        agent = WorktreeSorcarAgent("test")
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        from kiss.agents.sorcar.git_worktree import GitWorktree
        assert agent._wt is not None
        agent._wt = GitWorktree(
            repo_root=agent._wt.repo_root,
            branch=agent._wt.branch,
            original_branch=None,
            wt_dir=agent._wt.wt_dir,
        )

        msg = agent.merge()
        assert "git merge --squash" in msg

        agent._wt = GitWorktree(
            repo_root=agent._wt.repo_root,
            branch=agent._wt.branch,
            original_branch="main",
            wt_dir=agent._wt.wt_dir,
        )
        agent.discard()


class TestFixInc1ReleaseCallsFinalize:
    """_release_worktree now calls _finalize_worktree() instead of
    duplicating its logic.
    """

    def test_release_calls_finalize(self) -> None:
        """FIX-INC1: _release_worktree delegates to _finalize_worktree."""
        release_src = inspect.getsource(
            WorktreeSorcarAgent._release_worktree
        )
        assert "_finalize_worktree" in release_src

        assert "GitWorktreeOps.remove(" not in release_src
        assert "GitWorktreeOps.prune(" not in release_src


class TestFix4MergeRetrySkipsFinalize:
    """After merge() returns a conflict, calling merge() again skips
    _finalize_worktree since the worktree dir is already gone, and
    goes straight to the squash merge retry.
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

    def test_merge_retry_after_conflict_is_safe(self) -> None:
        """FIX-4: merge() retries cleanly after a conflict."""
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
        assert agent._wt_pending
        assert not wt_dir.exists()

        msg2 = agent.merge()
        assert "Merge conflict" in msg2

        agent.discard()

    def test_merge_retry_succeeds_after_conflict_resolved(self) -> None:
        """FIX-4: merge() succeeds on retry after user resolves conflict."""
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

        _git("reset", "--hard", "HEAD~1", cwd=self.repo)

        msg2 = agent.merge()
        assert "Successfully merged" in msg2
        assert not agent._wt_pending
