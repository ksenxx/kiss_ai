"""Regression tests for WorktreeSorcarAgent refactoring.

These tests capture current PUBLIC API behavior that must be preserved
when executing PLAN.md.  They focus on contract boundaries, edge cases
in merge/manual_merge/discard, and observable git state — i.e. things
the refactoring into GitWorktree/GitWorktreeOps is likely to break.

Every test is independent and uses real git repos (no mocks).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import GitWorktree, _git
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.kiss_error import KISSError

# ---------------------------------------------------------------------------
# Helpers (same as main test file)
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
# Tests
# ---------------------------------------------------------------------------


class TestWorktreeRegression:
    """Regression tests for WorktreeSorcarAgent refactoring safety."""

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

    # -----------------------------------------------------------------------
    # 2. manual_merge auto-commits before merging (removed)
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 3. discard when worktree directory already removed
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 4. merge conflict: main worktree is clean after abort
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 5. Branch naming convention is preserved
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 6. Worktree directory naming convention
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 7. Git config key format for original branch
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 8. merge() uses --no-edit flag (no editor prompt)
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 9. manual_merge unstages changes (reset HEAD) on success
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 10. merge conflict preserves _wt_branch for discard
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 11. run() result format: task output + separator + merge instructions
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 12. _delete_branch fallback: -d fails → -D
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 13. Multiple cleanup calls with mixed orphans
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 14. Worktree prune is called during merge path
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 15. Worktree prune is called during discard path
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 16. run() with KISSError re-raises (not caught)
    # -----------------------------------------------------------------------
    def test_run_kiss_error_reraises(self) -> None:
        """KISSError from super().run() is re-raised, not wrapped."""
        _unpatch_super_run(self.original_run)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        orig = parent_class.run

        def raising_run(self_agent: object, **kwargs: object) -> str:
            raise KISSError("budget exceeded")

        parent_class.run = raising_run
        try:
            agent = self._agent()
            with pytest.raises(KISSError, match="budget exceeded"):
                agent.run(prompt_template="t", work_dir=str(self.repo))
            # Worktree should still exist (cleanup is caller's job)
            assert agent._wt_pending
            agent.discard()
        finally:
            parent_class.run = orig
            self.original_run = _patch_super_run()

    # -----------------------------------------------------------------------
    # 17. _wt_pending property correctness
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 18. Restore picks latest branch when multiple exist
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 19. Merge unknown original → graceful error with manual instructions
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 20. manual_merge uses --no-commit --no-ff flags
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 21. manual_merge deletes branch on success (no conflict)
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 22. auto_commit creates a commit with all files staged
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 23. Worktree is a proper git branch off current HEAD
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 24. Full lifecycle: run → conflict merge → discard → run again
    # -----------------------------------------------------------------------
    def test_full_lifecycle_conflict_then_new_task(self) -> None:
        """After merge conflict + discard, a new task can run in the same
        chat session."""
        agent = self._agent()

        # Task 1
        agent.run(prompt_template="t1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        # Force conflict
        (wt_dir / "README.md").write_text("wt\n")
        (self.repo / "README.md").write_text("main\n")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-m", "conflict", cwd=self.repo)

        msg = agent.merge()
        assert "Merge conflict" in msg

        agent.discard()
        assert not agent._wt_pending

        # Task 2 should work
        result = agent.run(prompt_template="t2", work_dir=str(self.repo))
        assert "test done" in result
        assert agent._wt_pending
        agent.merge()

    # -----------------------------------------------------------------------
    # 25. cleanup deletes config section for orphaned branches
    # -----------------------------------------------------------------------
