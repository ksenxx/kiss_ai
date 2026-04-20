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


# ---------------------------------------------------------------------------
# BUG-5: use_worktree not restored after server restart
# ---------------------------------------------------------------------------


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

        # Persist task with is_worktree=True in the extra data
        chat_id = agent.chat_id
        task_id = agent._last_task_id
        assert task_id is not None
        th._save_task_extra(
            {"is_worktree": True, "model": "test"},
            task_id=task_id,
        )

        # Simulate server restart: create fresh server + tab state
        server = VSCodeServer()

        tab = server._get_tab("test-tab")

        # BUG: use_worktree is False after restart — never restored
        assert tab.use_worktree is False

        # Even though the persisted extra says is_worktree=True
        entry = th._load_latest_chat_events_by_chat_id(chat_id)
        assert entry is not None
        extra = json.loads(entry["extra"])  # type: ignore[arg-type]
        assert extra["is_worktree"] is True

        # _emit_pending_worktree returns immediately (doesn't detect pending wt)
        tab.agent.resume_chat_by_id(chat_id)
        # use_worktree is still False, so _emit_pending_worktree skips everything
        assert tab.use_worktree is False

        # Clean up worktree
        agent.discard()


# ---------------------------------------------------------------------------
# BUG-6: _finalize_worktree ignores auto-commit failure
# ---------------------------------------------------------------------------


class TestBug6FinalizeIgnoresCommitFailure:
    """_finalize_worktree removes the worktree even if auto-commit fails,
    silently losing the agent's uncommitted work.
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
        """BUG-6: Worktree removed even though auto-commit failed.

        The BUG-1 fix made commit_all() return False on failure, but
        _finalize_worktree still ignores this and removes the worktree,
        losing the agent's uncommitted changes.
        """
        agent = WorktreeSorcarAgent("test")
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()

        # Worktrees share hooks with the main repo
        main_hooks = self.repo / ".git" / "hooks"
        main_hooks.mkdir(exist_ok=True)
        hook = main_hooks / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        # Create a file that the agent "worked on"
        (wt_dir / "agent_work.txt").write_text("important work\n")

        # Verify the commit would fail
        assert GitWorktreeOps.commit_all(wt_dir, "test") is False

        # _finalize_worktree ignores the failure and removes the worktree
        agent._finalize_worktree()

        # BUG: worktree is gone, agent's work is lost
        assert not wt_dir.exists()

        # The branch exists but has no commit with agent_work.txt
        branch = agent._wt_branch
        assert branch is not None
        show = _git("show", f"{branch}:agent_work.txt", cwd=self.repo)
        # The file was never committed, so git show fails
        assert show.returncode != 0

        # Clean up
        hook.unlink()
        GitWorktreeOps.delete_branch(self.repo, branch)


# ---------------------------------------------------------------------------
# BUG-7: squash_merge_branch doesn't check commit return code
# ---------------------------------------------------------------------------


class TestBug7SquashMergeDoesntCheckCommit:
    """squash_merge_branch() returns SUCCESS even when git commit fails,
    causing the caller to delete the source branch while changes are
    only staged (not committed).
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        # Remove pre-commit hook if it exists
        hook = self.repo / ".git" / "hooks" / "pre-commit"
        if hook.exists():
            hook.unlink()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_squash_merge_returns_success_even_when_commit_fails(self) -> None:
        """BUG-7: squash_merge_branch returns SUCCESS even when the
        git commit --no-edit fails (e.g. pre-commit hook rejection).
        """
        # Create a feature branch with changes
        _git("checkout", "-b", "feature", cwd=self.repo)
        (self.repo / "feature.txt").write_text("feature work\n")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-m", "feature commit", cwd=self.repo)

        # Switch back to main
        _git("checkout", "main", cwd=self.repo)

        # Install a pre-commit hook that rejects commits
        hooks_dir = self.repo / ".git" / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        # squash_merge_branch should return CONFLICT or some failure,
        # but it returns SUCCESS because it doesn't check commit return code
        result = GitWorktreeOps.squash_merge_branch(self.repo, "feature")

        # BUG: returns SUCCESS even though commit failed
        assert result == MergeResult.SUCCESS

        # The changes are staged but NOT committed
        diff_cached = _git("diff", "--cached", "--quiet", cwd=self.repo)
        assert diff_cached.returncode != 0, "Changes still staged (not committed)"

        # Verify the commit was actually rejected
        log = _git("log", "--oneline", cwd=self.repo)
        commit_count = len(log.stdout.strip().splitlines())
        assert commit_count == 1, "No new commit was created"

        # In the real flow, the caller would now delete the feature branch:
        # GitWorktreeOps.delete_branch(self.repo, "feature")
        # leaving changes stuck in staging with source branch gone.

    def test_full_merge_flow_deletes_branch_despite_commit_failure(self) -> None:
        """BUG-7: Full merge flow deletes source branch even though
        squash commit failed, leaving changes orphaned in staging area.
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

            # Install hook ONLY on main repo (worktree uses same hooks)
            hooks_dir = self.repo / ".git" / "hooks"
            hooks_dir.mkdir(exist_ok=True)
            hook = hooks_dir / "pre-commit"
            hook.write_text("#!/bin/sh\nexit 1\n")
            hook.chmod(0o755)

            msg = agent.merge()

            # BUG: merge reports success despite commit failure
            assert "Successfully merged" in msg

            # But branch is deleted
            assert not GitWorktreeOps.branch_exists(self.repo, branch)

            # Changes are staged but not committed
            diff_cached = _git("diff", "--cached", "--quiet", cwd=self.repo)
            assert diff_cached.returncode != 0

            # Clean up staged changes
            _git("reset", "--hard", "HEAD", cwd=self.repo)
        finally:
            hook_path = self.repo / ".git" / "hooks" / "pre-commit"
            if hook_path.exists():
                hook_path.unlink()
            _unpatch_super_run(original_run)
            _restore_db(db_saved)
            shutil.rmtree(tmpdir2, ignore_errors=True)


# ---------------------------------------------------------------------------
# INC-2: Redundant stage_all in _auto_commit_worktree
# ---------------------------------------------------------------------------


class TestInc2RedundantStageAllFixed:
    """_auto_commit_worktree now calls stage_all() then commit_staged()
    which does NOT re-stage — the redundant git add -A is eliminated.
    """

    def test_auto_commit_uses_commit_staged(self) -> None:
        """INC-2 FIX: _auto_commit_worktree uses commit_staged (no re-stage)."""
        import inspect

        src = inspect.getsource(WorktreeSorcarAgent._auto_commit_worktree)
        # Uses stage_all to stage once
        assert "stage_all" in src
        # Uses commit_staged (not commit_all) so no redundant git add -A
        assert "commit_staged" in src
        assert "commit_all" not in src

    def test_commit_staged_does_not_stage(self) -> None:
        """commit_staged does not run git add -A."""
        import inspect

        src = inspect.getsource(GitWorktreeOps.commit_staged)
        assert "add" not in src or '"add", "-A"' not in src
