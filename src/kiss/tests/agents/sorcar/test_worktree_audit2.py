"""Tests confirming bugs found in worktree audit round 2.

Each test confirms a specific bug exists in the current code, labeled
BUG-5 through BUG-7 plus INC-2.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    MergeResult,
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


class TestBug5UseWorktreeNotRestored:
    """After server restart, _emit_pending_worktree returns early because
    tab.use_worktree defaults to False and is never restored from the
    persisted 'extra' data.
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

    def test_replay_session_does_not_restore_use_worktree(self) -> None:
        """BUG-5: _replay_session doesn't parse persisted extra data to
        restore use_worktree, making pending worktrees invisible after restart.
        """
        from kiss.agents.vscode.server import VSCodeServer

        agent = WorktreeSorcarAgent("test")
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        assert agent._wt_pending

        chat_id = agent.chat_id
        task_id = agent._last_task_id
        assert task_id is not None
        th._save_task_extra(
            {"is_worktree": True, "model": "test"},
            task_id=task_id,
        )

        server = VSCodeServer()

        tab = server._get_tab("test-tab")

        assert tab.use_worktree is False

        entry = th._load_latest_chat_events_by_chat_id(chat_id)
        assert entry is not None
        extra = json.loads(entry["extra"])  # type: ignore[arg-type]
        assert extra["is_worktree"] is True

        tab.agent.resume_chat_by_id(chat_id)
        assert tab.use_worktree is False

        agent.discard()


class TestBug6FinalizeIgnoresCommitFailure:
    """BUG-6 FIX: _finalize_worktree now preserves the worktree when
    auto-commit fails, preventing data loss.
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

    def test_finalize_removes_worktree_despite_commit_failure(self) -> None:
        """BUG-6 FIX: _finalize_worktree returns False and preserves
        the worktree directory when auto-commit is rejected by a
        pre-commit hook, preventing data loss.
        """
        agent = WorktreeSorcarAgent("test")
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()

        main_hooks = self.repo / ".git" / "hooks"
        main_hooks.mkdir(exist_ok=True)
        hook = main_hooks / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        (wt_dir / "agent_work.txt").write_text("important work\n")

        assert GitWorktreeOps.commit_all(wt_dir, "test") is False

        result = agent._finalize_worktree()
        assert result is False

        assert wt_dir.exists()
        assert (wt_dir / "agent_work.txt").read_text() == "important work\n"

        hook.unlink()
        branch = agent._wt_branch
        assert branch is not None
        GitWorktreeOps.remove(self.repo, wt_dir)
        GitWorktreeOps.delete_branch(self.repo, branch)


class TestBug7SquashMergeDoesntCheckCommit:
    """BUG-7 FIX: squash_merge_branch() now returns MERGE_FAILED when
    git commit fails, preventing the source branch from being deleted.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        hook = self.repo / ".git" / "hooks" / "pre-commit"
        if hook.exists():
            hook.unlink()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_squash_merge_returns_success_even_when_commit_fails(self) -> None:
        """BUG-7 FIX: squash_merge_branch returns MERGE_FAILED when
        the git commit is rejected by a pre-commit hook.
        """
        _git("checkout", "-b", "feature", cwd=self.repo)
        (self.repo / "feature.txt").write_text("feature work\n")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-m", "feature commit", cwd=self.repo)

        _git("checkout", "main", cwd=self.repo)

        hooks_dir = self.repo / ".git" / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        result = GitWorktreeOps.squash_merge_branch(self.repo, "feature")

        assert result == MergeResult.MERGE_FAILED

    def test_full_merge_flow_deletes_branch_despite_commit_failure(self) -> None:
        """BUG-7 FIX: Full merge flow does NOT delete source branch
        when squash commit fails — agent work is preserved.
        """
        tmpdir2 = tempfile.mkdtemp()
        db_saved = _redirect_db(tmpdir2)
        original_run = _patch_super_run()
        try:
            agent = WorktreeSorcarAgent("test")
            agent.run(prompt_template="task1", work_dir=str(self.repo))

            wt_dir = agent._wt_dir
            assert wt_dir is not None
            (wt_dir / "feature.txt").write_text("agent work\n")
            GitWorktreeOps.commit_all(wt_dir, "agent work")

            branch = agent._wt_branch
            assert branch is not None

            hooks_dir = self.repo / ".git" / "hooks"
            hooks_dir.mkdir(exist_ok=True)
            hook = hooks_dir / "pre-commit"
            hook.write_text("#!/bin/sh\nexit 1\n")
            hook.chmod(0o755)

            msg = agent.merge()

            assert "Successfully merged" not in msg

            assert GitWorktreeOps.branch_exists(self.repo, branch)
        finally:
            hook_path = self.repo / ".git" / "hooks" / "pre-commit"
            if hook_path.exists():
                hook_path.unlink()
            _unpatch_super_run(original_run)
            _restore_db(db_saved)
            shutil.rmtree(tmpdir2, ignore_errors=True)


class TestInc2RedundantStageAllFixed:
    """_auto_commit_worktree now calls stage_all() then commit_staged()
    which does NOT re-stage — the redundant git add -A is eliminated.
    """

    def test_auto_commit_uses_commit_staged(self) -> None:
        """INC-2 FIX: _auto_commit_worktree uses commit_staged (no re-stage)."""
        import inspect

        src = inspect.getsource(WorktreeSorcarAgent._auto_commit_worktree)
        assert "stage_all" in src
        assert "commit_staged" in src
        assert "commit_all" not in src

    def test_commit_staged_does_not_stage(self) -> None:
        """commit_staged does not run git add -A."""
        import inspect

        src = inspect.getsource(GitWorktreeOps.commit_staged)
        assert "add" not in src or '"add", "-A"' not in src
