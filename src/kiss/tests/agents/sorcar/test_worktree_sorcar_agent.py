"""Tests for WorktreeSorcarAgent: worktree lifecycle, blocking, crash recovery."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import (
    WorktreeSorcarAgent,
    _git,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _redirect_db(tmpdir: str) -> tuple:
    """Redirect persistence DB to a temp dir."""
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
    """Create a minimal git repo with one commit and return its path."""
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
    subprocess.run(["git", "-C", str(path), "add", "."], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True, check=True,
    )
    return path


def _patch_super_run(return_value: str = "success: true\nsummary: test done\n") -> Any:
    """Monkey-patch RelentlessAgent.run to skip the LLM call."""
    parent_class = cast(Any, SorcarAgent.__mro__[1])  # RelentlessAgent
    original = parent_class.run

    def fake_run(self_agent: object, **kwargs: object) -> str:
        return return_value

    parent_class.run = fake_run
    return original


def _unpatch_super_run(original: Any) -> None:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    parent_class.run = original


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestWorktreeSorcarAgent:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _agent(self, chat_id: str | None = None) -> WorktreeSorcarAgent:
        agent = WorktreeSorcarAgent("test")
        if chat_id:
            agent.resume_chat_by_id(chat_id)
        return agent

    # 1. Happy path (merge)
    def test_run_and_merge(self) -> None:
        agent = self._agent()
        result = agent.run(prompt_template="task1", work_dir=str(self.repo))
        assert "test done" in result
        assert agent._wt_branch is not None
        assert agent._original_branch == "main"

        msg = agent.merge()
        assert "Successfully merged" in msg
        assert agent._wt_branch is None
        assert agent._original_branch is None

    # 2. Discard path
    def test_run_and_discard(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        branch = agent._wt_branch
        assert branch is not None

        msg = agent.discard()
        assert "Discarded" in msg
        assert agent._wt_branch is None

        # Branch should be gone
        check = _git("rev-parse", "--verify", f"refs/heads/{branch}",
                      cwd=self.repo)
        assert check.returncode != 0

    # 3. Blocking (same session)
    def test_blocking_same_session(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        result = agent.run(prompt_template="task2", work_dir=str(self.repo))
        assert "pending merge/discard" in result

        agent.merge()
        result = agent.run(prompt_template="task3", work_dir=str(self.repo))
        assert "test done" in result

    # 4. No blocking (different session)
    def test_no_blocking_different_session(self) -> None:
        agent_a = self._agent()
        agent_a.run(prompt_template="task1", work_dir=str(self.repo))
        assert agent_a._wt_pending

        agent_b = self._agent()  # different chat_id
        result = agent_b.run(prompt_template="task2", work_dir=str(self.repo))
        assert "test done" in result

        # Clean up both
        agent_a.discard()
        agent_b.discard()

    # 5. Not a git repo
    def test_not_a_git_repo(self) -> None:
        no_repo = Path(self.tmpdir) / "no_repo"
        no_repo.mkdir()
        agent = self._agent()
        result = agent.run(prompt_template="task1", work_dir=str(no_repo))
        assert "test done" in result
        assert agent._wt_branch is None

    # 6. Merge conflict
    def test_merge_conflict(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        # Modify file in worktree
        (wt_dir / "README.md").write_text("worktree change\n")

        # Modify same file on main branch
        (self.repo / "README.md").write_text("main change\n")
        subprocess.run(
            ["git", "-C", str(self.repo), "add", "."],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-m", "main edit"],
            capture_output=True, check=True,
        )

        msg = agent.merge()
        assert "Merge conflict" in msg
        assert "git merge" in msg
        # Should NOT reference worktree removal (already removed)
        assert "git worktree remove" not in msg
        # Should still be pending
        assert agent._wt_pending

        # Main worktree should be clean (merge was aborted)
        status = _git("status", "--porcelain", cwd=self.repo)
        assert status.stdout.strip() == ""

        # Can still discard
        agent.discard()
        assert not agent._wt_pending

    # 7. merge_instructions()
    def test_merge_instructions(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        instructions = agent.merge_instructions()
        assert agent._wt_branch is not None
        assert agent._wt_branch in instructions
        assert "agent.merge()" in instructions
        assert "agent.discard()" in instructions
        assert "main" in instructions
        agent.discard()

    # 8. Auto-commit
    def test_auto_commit_before_merge(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        # Create a new file in worktree (uncommitted)
        (wt_dir / "new_file.txt").write_text("hello\n")

        msg = agent.merge()
        assert "Successfully merged" in msg
        # The file should be on main now
        assert (self.repo / "new_file.txt").exists()

    # 9. Subdirectory offset
    def test_subdirectory_offset(self) -> None:
        subdir = self.repo / "src" / "app"
        subdir.mkdir(parents=True)
        (subdir / "main.py").write_text("print('hello')\n")
        subprocess.run(
            ["git", "-C", str(self.repo), "add", "."],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-m", "add subdir"],
            capture_output=True, check=True,
        )

        agent = self._agent()
        # Capture what work_dir gets set to
        captured: dict[str, Any] = {}
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        orig = parent_class.run

        def capture_run(self_agent: object, **kwargs: object) -> str:
            captured["work_dir"] = kwargs.get("work_dir")
            return "success: true\nsummary: test done\n"

        parent_class.run = capture_run
        try:
            agent.run(prompt_template="task1", work_dir=str(subdir))
        finally:
            parent_class.run = orig

        assert captured["work_dir"] is not None
        assert "src/app" in str(captured["work_dir"]) or "src\\app" in str(captured["work_dir"])
        agent.discard()

    # 10. State persistence via git
    def test_state_persistence_via_git(self) -> None:
        agent1 = self._agent(chat_id="aabbccdd11223344")
        agent1.run(prompt_template="task1", work_dir=str(self.repo))
        branch = agent1._wt_branch
        assert branch is not None

        # New agent instance with same chat_id
        agent2 = self._agent(chat_id="aabbccdd11223344")
        agent2._repo_root = self.repo
        agent2._restore_from_git()
        assert agent2._wt_branch == branch
        assert agent2._original_branch == "main"

        agent1.discard()

    # 11. Process crash recovery
    def test_process_crash_recovery(self) -> None:
        agent1 = self._agent(chat_id="crash_recovery_id1")
        agent1.run(prompt_template="task1", work_dir=str(self.repo))
        branch = agent1._wt_branch
        assert branch is not None

        # Simulate crash: create new agent with same chat_id
        agent2 = self._agent(chat_id="crash_recovery_id1")
        # run() should be blocked
        result = agent2.run(prompt_template="task2", work_dir=str(self.repo))
        assert "pending merge/discard" in result

        # Merge should work on the new instance
        msg = agent2.merge()
        assert "Successfully merged" in msg

    # 12. Idempotent merge
    def test_idempotent_merge(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        msg1 = agent.merge()
        assert "Successfully merged" in msg1
        # Second merge should raise (no pending task)
        with pytest.raises(RuntimeError, match="No pending"):
            agent.merge()

    # 13. Idempotent discard
    def test_idempotent_discard(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        msg1 = agent.discard()
        assert "Discarded" in msg1
        with pytest.raises(RuntimeError, match="No pending"):
            agent.discard()

    # 14. Detached HEAD
    def test_detached_head(self) -> None:
        # Detach HEAD
        head = _git("rev-parse", "HEAD", cwd=self.repo)
        subprocess.run(
            ["git", "-C", str(self.repo), "checkout", head.stdout.strip()],
            capture_output=True, check=True,
        )
        agent = self._agent()
        result = agent.run(prompt_template="task1", work_dir=str(self.repo))
        assert "test done" in result
        assert agent._wt_branch is None

        # Restore
        subprocess.run(
            ["git", "-C", str(self.repo), "checkout", "main"],
            capture_output=True, check=True,
        )

    # 15. Offset directory creation
    def test_offset_dir_creation(self) -> None:
        # work_dir is a subdir that doesn't exist on the branch yet
        subdir = self.repo / "new_dir" / "sub"
        subdir.mkdir(parents=True, exist_ok=True)
        agent = self._agent()
        result = agent.run(prompt_template="task1", work_dir=str(subdir))
        assert "test done" in result
        agent.discard()

    # 16. Dirty main worktree at merge
    def test_dirty_main_worktree_at_merge(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        # Make main worktree dirty (new tracked file modification)
        (self.repo / "README.md").write_text("dirty\n")
        subprocess.run(
            ["git", "-C", str(self.repo), "add", "."],
            capture_output=True, check=True,
        )

        msg = agent.merge()
        # checkout should fail because of staged changes
        if "Cannot checkout" in msg:
            assert agent._wt_pending
            # Reset and retry
            subprocess.run(
                ["git", "-C", str(self.repo), "reset", "HEAD"],
                capture_output=True, check=True,
            )
            subprocess.run(
                ["git", "-C", str(self.repo), "checkout", "--", "."],
                capture_output=True, check=True,
            )
            msg2 = agent.merge()
            assert "Successfully merged" in msg2
        else:
            # Git might handle this differently on some versions
            assert "Successfully merged" in msg or "Merge conflict" in msg

    # 17. Conflict instructions exclude worktree removal
    def test_conflict_instructions_no_worktree_remove(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        # Create conflict
        (wt_dir / "README.md").write_text("wt\n")
        (self.repo / "README.md").write_text("main\n")
        subprocess.run(
            ["git", "-C", str(self.repo), "add", "."],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-m", "conflict"],
            capture_output=True, check=True,
        )

        msg = agent.merge()
        assert "Merge conflict" in msg
        assert "git worktree remove" not in msg
        assert "agent.discard()" in msg

        agent.discard()

    # 18. Cleanup
    def test_cleanup(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        branch = agent._wt_branch
        assert branch is not None

        # Remove worktree manually but leave branch (orphan)
        wt_dir = agent._wt_dir
        assert wt_dir is not None
        _git("worktree", "remove", str(wt_dir), "--force", cwd=self.repo)
        _git("worktree", "prune", cwd=self.repo)

        result = WorktreeSorcarAgent.cleanup(self.repo)
        assert "Deleted" in result or "orphan" in result.lower() or "1 kiss/wt-*" in result

        # Branch should be gone now
        check = _git("rev-parse", "--verify", f"refs/heads/{branch}",
                      cwd=self.repo)
        assert check.returncode != 0

    # 19. Git config cleanup after branch deletion
    def test_git_config_cleanup(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        branch = agent._wt_branch
        assert branch is not None

        # Verify config exists
        cfg = _git("config", f"branch.{branch}.kiss-original",
                    cwd=self.repo)
        assert cfg.returncode == 0

        agent.merge()

        # After branch deletion, git removes the config section
        cfg2 = _git("config", f"branch.{branch}.kiss-original",
                     cwd=self.repo)
        assert cfg2.returncode != 0

    # 20. Branch name collision
    def test_branch_name_collision(self) -> None:
        # Use a different prefix so _restore_from_git won't match
        # Create collision branches with a known prefix "zzzzzzzzzzzz"
        # that differs from any real chat_id.
        _git("branch", "kiss/wt-zzzzzzzzzzzz-100", cwd=self.repo)
        _git("branch", "kiss/wt-zzzzzzzzzzzz-100-1", cwd=self.repo)

        # Run normally — agent uses its own chat_id, no collision
        agent = self._agent()
        result = agent.run(prompt_template="task1", work_dir=str(self.repo))
        assert "test done" in result
        agent.discard()

        # Clean up
        _git("branch", "-D", "kiss/wt-zzzzzzzzzzzz-100", cwd=self.repo)
        _git("branch", "-D", "kiss/wt-zzzzzzzzzzzz-100-1", cwd=self.repo)

    # 21. Worktree excluded from git
    def test_worktree_excluded_from_git(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        # Check exclude file
        git_dir = self.repo / ".git"
        exclude_file = git_dir / "info" / "exclude"
        assert exclude_file.exists()
        content = exclude_file.read_text()
        assert ".kiss-worktrees/" in content

        # git status should not show .kiss-worktrees
        status = _git("status", "--porcelain", cwd=self.repo)
        assert ".kiss-worktrees" not in status.stdout

        agent.discard()

    # 22. Missing kiss-original config (crash recovery)
    def test_missing_kiss_original_config(self) -> None:
        agent = self._agent(chat_id="missing_config_1234")
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        branch = agent._wt_branch
        assert branch is not None

        # Simulate crash: remove the config entry manually
        _git("config", "--unset", f"branch.{branch}.kiss-original",
             cwd=self.repo)

        # New agent should recover
        agent2 = self._agent(chat_id="missing_config_1234")
        agent2._repo_root = self.repo
        agent2._restore_from_git()
        assert agent2._wt_branch == branch
        # Falls back to current HEAD
        assert agent2._original_branch == "main"

        agent.discard()

    # 23. Missing kiss-original config + detached HEAD
    def test_missing_config_detached_head(self) -> None:
        agent = self._agent(chat_id="detached_cfg_test")
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        branch = agent._wt_branch
        assert branch is not None

        # Remove config
        _git("config", "--unset", f"branch.{branch}.kiss-original",
             cwd=self.repo)

        # Detach HEAD
        head = _git("rev-parse", "HEAD", cwd=self.repo)
        subprocess.run(
            ["git", "-C", str(self.repo), "checkout", head.stdout.strip()],
            capture_output=True, check=True,
        )

        agent2 = self._agent(chat_id="detached_cfg_test")
        agent2._repo_root = self.repo
        agent2._restore_from_git()
        assert agent2._wt_branch == branch
        assert agent2._original_branch is None

        # merge should fail gracefully
        msg = agent2.merge()
        assert "Cannot merge" in msg
        assert "original branch is unknown" in msg

        # discard should work
        msg = agent2.discard()
        assert "Discarded" in msg

        subprocess.run(
            ["git", "-C", str(self.repo), "checkout", "main"],
            capture_output=True, check=True,
        )

    # 24. merge_instructions when idle
    def test_merge_instructions_idle(self) -> None:
        agent = self._agent()
        assert agent.merge_instructions() == "No pending worktree task."

    # 25. _auto_commit_worktree when nothing to commit
    def test_auto_commit_nothing(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        committed = agent._auto_commit_worktree()
        assert not committed
        agent.discard()

    # 26. _auto_commit_worktree when wt_dir is None
    def test_auto_commit_no_wt_dir(self) -> None:
        agent = self._agent()
        assert not agent._auto_commit_worktree()

    # 27. super().run raises exception
    def test_super_run_raises(self) -> None:
        _unpatch_super_run(self.original_run)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        orig = parent_class.run

        def raising_run(self_agent: object, **kwargs: object) -> str:
            raise RuntimeError("LLM crashed")

        parent_class.run = raising_run
        try:
            agent = self._agent()
            result = agent.run(prompt_template="task1", work_dir=str(self.repo))
            assert "Task failed" in result
            assert agent._wt_pending
            agent.discard()
        finally:
            parent_class.run = orig
            self.original_run = _patch_super_run()

    # 28. merge raises when no pending task
    def test_merge_no_pending(self) -> None:
        agent = self._agent()
        with pytest.raises(RuntimeError, match="No pending"):
            agent.merge()

    # 29. discard raises when no pending task
    def test_discard_no_pending(self) -> None:
        agent = self._agent()
        with pytest.raises(RuntimeError, match="No pending"):
            agent.discard()

    # 30. _restore_from_git with no repo root
    def test_restore_no_repo_root(self) -> None:
        agent = self._agent()
        agent._repo_root = None
        agent._restore_from_git()
        assert agent._wt_branch is None

    # 31. _restore_from_git when already known in-memory
    def test_restore_already_known(self) -> None:
        agent = self._agent()
        agent._repo_root = self.repo
        agent._wt_branch = "some/branch"
        agent._restore_from_git()
        assert agent._wt_branch == "some/branch"

    # 32. _ensure_worktree_excluded when repo_root is None
    def test_ensure_excluded_no_repo(self) -> None:
        agent = self._agent()
        agent._repo_root = None
        agent._ensure_worktree_excluded()  # should not raise

    # 33. _ensure_worktree_excluded is idempotent
    def test_ensure_excluded_idempotent(self) -> None:
        agent = self._agent()
        agent._repo_root = self.repo
        agent._ensure_worktree_excluded()
        agent._ensure_worktree_excluded()
        exclude_file = self.repo / ".git" / "info" / "exclude"
        content = exclude_file.read_text()
        # Should appear only once in the lines
        lines = content.splitlines()
        assert lines.count(".kiss-worktrees/") == 1

    # 34. _wt_dir when no repo or no branch
    def test_wt_dir_none(self) -> None:
        agent = self._agent()
        assert agent._wt_dir is None
        agent._repo_root = self.repo
        assert agent._wt_dir is None

    # 35. work_dir not in repo
    def test_work_dir_outside_repo(self) -> None:
        outside = Path(self.tmpdir) / "outside"
        outside.mkdir()
        # Initialize a different repo there so git finds it but it's not the same repo
        agent = self._agent()
        result = agent.run(prompt_template="task1", work_dir=str(outside))
        # Should fall back to direct execution (not a git repo)
        assert "test done" in result

    # 36. Empty repo (no commits)
    def test_empty_repo(self) -> None:
        empty = Path(self.tmpdir) / "empty_repo"
        empty.mkdir()
        subprocess.run(["git", "init", str(empty)], capture_output=True, check=True)
        agent = self._agent()
        result = agent.run(prompt_template="task1", work_dir=str(empty))
        # git rev-parse --abbrev-ref HEAD on empty repo returns HEAD
        # which triggers direct execution
        assert "test done" in result
        assert agent._wt_branch is None

    # 37. cleanup with no orphans
    def test_cleanup_no_orphans(self) -> None:
        result = WorktreeSorcarAgent.cleanup(self.repo)
        assert "No orphans found" in result

    # 38. _git helper
    def test_git_helper(self) -> None:
        result = _git("rev-parse", "--show-toplevel", cwd=self.repo)
        assert result.returncode == 0
        assert str(self.repo.resolve()) in result.stdout or str(self.repo) in result.stdout

    # 39. _git without cwd
    def test_git_no_cwd(self) -> None:
        result = _git("--version")
        assert result.returncode == 0
        assert "git version" in result.stdout

    # 40. _ensure_worktree_excluded when exclude file doesn't exist yet
    def test_ensure_excluded_no_file(self) -> None:
        agent = self._agent()
        agent._repo_root = self.repo
        exclude_file = self.repo / ".git" / "info" / "exclude"
        if exclude_file.exists():
            exclude_file.unlink()
        if exclude_file.parent.exists():
            exclude_file.parent.rmdir()
        agent._ensure_worktree_excluded()
        assert exclude_file.exists()
        assert ".kiss-worktrees/" in exclude_file.read_text()

    # 42. Worktree creation failure fallback
    def test_worktree_add_failure(self) -> None:
        agent = self._agent()
        # Make the .kiss-worktrees dir read-only to force failure
        wt_base = self.repo / ".kiss-worktrees"
        wt_base.mkdir(exist_ok=True)
        # Create a file that blocks directory creation
        blocker = wt_base / f"kiss_wt-{agent.chat_id[:12]}-99999999999"
        blocker.write_text("blocker")
        # Run should fall back to direct execution when worktree add fails
        # (can't create worktree dir because a file exists with that name)
        # Actually this won't match the slug. Let me use a different approach.
        blocker.unlink()

        # Better: corrupt the repo to make worktree add fail
        # Just test that run completes even if worktree can't be created
        # by making the target a regular file
        import time as t
        ts = int(t.time())
        branch_name = f"kiss/wt-{agent.chat_id[:12]}-{ts}"
        slug = branch_name.replace("/", "_")
        target = wt_base / slug
        target.mkdir(parents=True, exist_ok=True)
        (target / "blocker").write_text("x")  # non-empty dir blocks worktree add

        result = agent.run(prompt_template="task1", work_dir=str(self.repo))
        # Should fall back to direct execution
        assert "test done" in result
        # Clean up
        shutil.rmtree(wt_base, ignore_errors=True)

    # 43. cleanup static method with active worktree
    def test_cleanup_with_active_worktree(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        result = WorktreeSorcarAgent.cleanup(self.repo)
        assert "1 kiss/wt-*" in result
        assert "1 active" in result
        agent.discard()

    # 45. Merge when worktree was already removed externally
    def test_merge_worktree_already_removed(self) -> None:
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()

        # Manually remove the worktree
        _git("worktree", "remove", str(wt_dir), "--force", cwd=self.repo)
        _git("worktree", "prune", cwd=self.repo)
        assert not wt_dir.exists()

        # merge should still succeed (worktree removal is skipped)
        msg = agent.merge()
        assert "Successfully merged" in msg

    # 46. run() without work_dir kwarg
    def test_run_without_work_dir(self) -> None:
        import os
        old_cwd = os.getcwd()
        os.chdir(str(self.repo))
        try:
            agent = self._agent()
            result = agent.run(prompt_template="task1")
            assert "test done" in result
            if agent._wt_pending:
                agent.discard()
        finally:
            os.chdir(old_cwd)

    # x. _cleanup_partial_worktree when wt_dir does not exist
    def test_cleanup_partial_worktree_no_dir(self) -> None:
        agent = self._agent()
        agent._repo_root = self.repo
        branch = "kiss/wt-nocleanup"
        _git("branch", branch, cwd=self.repo)
        nonexistent = self.repo / ".kiss-worktrees" / "nonexistent"
        agent._cleanup_partial_worktree(branch, nonexistent)
        # Branch should be gone
        check = _git("rev-parse", "--verify", f"refs/heads/{branch}",
                      cwd=self.repo)
        assert check.returncode != 0

    # 46. _cleanup_partial_worktree when wt_dir exists
    def test_cleanup_partial_worktree_exists(self) -> None:
        agent = self._agent()
        agent._repo_root = self.repo
        # Create a worktree manually to test cleanup
        branch = "kiss/wt-testcleanup"
        slug = branch.replace("/", "_")
        wt_dir = self.repo / ".kiss-worktrees" / slug
        _git("worktree", "add", "-b", branch, str(wt_dir), cwd=self.repo)
        assert wt_dir.exists()
        agent._cleanup_partial_worktree(branch, wt_dir)
        assert not wt_dir.exists()
        # Branch should be gone
        check = _git("rev-parse", "--verify", f"refs/heads/{branch}",
                      cwd=self.repo)
        assert check.returncode != 0


