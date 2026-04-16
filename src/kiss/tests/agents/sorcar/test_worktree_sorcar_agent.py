"""Tests for WorktreeSorcarAgent: worktree lifecycle, blocking, crash recovery."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

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

    # 2. Discard path

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

    # 5. Not a git repo
    def test_not_a_git_repo(self) -> None:
        no_repo = Path(self.tmpdir) / "no_repo"
        no_repo.mkdir()
        agent = self._agent()
        result = agent.run(prompt_template="task1", work_dir=str(no_repo))
        assert "test done" in result
        assert agent._wt_branch is None

    # 6. Merge conflict

    # 7. merge_instructions()

    # 8. Auto-commit

    # 9. Subdirectory offset

    # 10. State persistence via git

    # 11. Process crash recovery

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

    # 15. Offset directory creation

    # 16. Dirty main worktree at merge

    # 17. Conflict instructions exclude worktree removal

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

    # 20. Branch name collision

    # 21. Worktree excluded from git

    # 22. Missing kiss-original config (crash recovery)

    # 23. Missing kiss-original config + detached HEAD
    def test_missing_config_detached_head(self) -> None:
        agent = self._agent(chat_id="1000")
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

        agent2 = self._agent(chat_id="1000")
        agent2._restore_from_git(self.repo)
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

    # 29. discard raises when no pending task

    # 30. _restore_from_git with no repo root — tested via run() on non-repo

    # 31. _restore_from_git when already known in-memory

    # 32. ensure_excluded with no repo — tested via run() fallback

    # 33. ensure_excluded is idempotent

    # 34. _wt_dir when no _wt

    # 35. work_dir not in repo

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

    # 39. _git without cwd
    def test_git_no_cwd(self) -> None:
        result = _git("--version")
        assert result.returncode == 0
        assert "git version" in result.stdout

    # 40. ensure_excluded when exclude file doesn't exist yet
    def test_ensure_excluded_no_file(self) -> None:
        exclude_file = self.repo / ".git" / "info" / "exclude"
        if exclude_file.exists():
            exclude_file.unlink()
        if exclude_file.parent.exists():
            exclude_file.parent.rmdir()
        GitWorktreeOps.ensure_excluded(self.repo)
        assert exclude_file.exists()
        assert ".kiss-worktrees/" in exclude_file.read_text()

    # 42. Worktree creation failure fallback
    def test_worktree_add_failure(self) -> None:
        agent = self._agent()
        # Make the .kiss-worktrees dir read-only to force failure
        wt_base = self.repo / ".kiss-worktrees"
        wt_base.mkdir(exist_ok=True)
        # Create a file that blocks directory creation
        blocker = wt_base / f"kiss_wt-{agent.chat_id}-99999999999"
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
        branch_name = f"kiss/wt-{agent.chat_id}-{ts}"
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

    # x. cleanup_partial when wt_dir does not exist
    def test_cleanup_partial_worktree_no_dir(self) -> None:
        branch = "kiss/wt-nocleanup"
        _git("branch", branch, cwd=self.repo)
        nonexistent = self.repo / ".kiss-worktrees" / "nonexistent"
        GitWorktreeOps.cleanup_partial(self.repo, branch, nonexistent)
        # Branch should be gone
        check = _git("rev-parse", "--verify", f"refs/heads/{branch}",
                      cwd=self.repo)
        assert check.returncode != 0

    # 46. cleanup_partial when wt_dir exists


class TestCliAgentSelection:
    """Verify that main() selects the correct agent based on CLI flags."""

    def test_main_arg_parser_has_agent_type_flags(self) -> None:
        import inspect

        from kiss.agents.sorcar.cli_helpers import _build_arg_parser

        src = inspect.getsource(_build_arg_parser)
        assert "--use-worktree" in src
        assert "--use-chat" in src
        assert "--base-sorcar" not in src

    def test_main_source_creates_worktree_agent_with_flag(self) -> None:
        import inspect
        src = inspect.getsource(
            __import__("kiss.agents.sorcar.worktree_sorcar_agent", fromlist=["main"]).main
        )
        assert 'if args.use_worktree:' in src
        assert 'WorktreeSorcarAgent(' in src

    def test_main_source_creates_stateful_agent_with_flag(self) -> None:
        import inspect
        src = inspect.getsource(
            __import__("kiss.agents.sorcar.worktree_sorcar_agent", fromlist=["main"]).main
        )
        assert 'elif args.use_chat:' in src
        assert 'StatefulSorcarAgent(' in src

    def test_main_source_defaults_to_sorcar_agent(self) -> None:
        import inspect
        src = inspect.getsource(
            __import__("kiss.agents.sorcar.worktree_sorcar_agent", fromlist=["main"]).main
        )
        assert 'SorcarAgent("Sorcar Agent")' in src

    def test_main_source_guards_merge_prompt_with_isinstance(self) -> None:
        import inspect
        src = inspect.getsource(
            __import__("kiss.agents.sorcar.worktree_sorcar_agent", fromlist=["main"]).main
        )
        assert 'isinstance(agent, WorktreeSorcarAgent)' in src

    def test_main_source_uses_discover_repo_for_cleanup(self) -> None:
        import inspect
        src = inspect.getsource(
            __import__("kiss.agents.sorcar.worktree_sorcar_agent", fromlist=["main"]).main
        )
        assert "discover_repo" in src


