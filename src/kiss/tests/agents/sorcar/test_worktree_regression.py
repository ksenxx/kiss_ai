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
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import (
    WorktreeSorcarAgent,
    _git,
)
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
    # 1. manual_merge with conflicts — branch is preserved, state is cleared
    # -----------------------------------------------------------------------
    def test_manual_merge_conflict_preserves_branch(self) -> None:
        """After manual_merge with conflict, the task branch is NOT deleted
        so the user can still reference it.  But agent state IS cleared."""
        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        branch = agent._wt_branch
        assert wt_dir is not None and branch is not None

        # Create conflicting changes
        (wt_dir / "README.md").write_text("worktree\n")
        (self.repo / "README.md").write_text("main\n")
        subprocess.run(
            ["git", "-C", str(self.repo), "add", "."],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-m", "main edit"],
            capture_output=True,
            check=True,
        )

        msg = agent.manual_merge()
        assert "conflicts" in msg.lower()

        # Agent state is cleared (no longer pending)
        assert agent._wt_branch is None
        assert agent._original_branch is None

        # But the BRANCH still exists in git (not deleted on conflict)
        check = _git(
            "rev-parse", "--verify", f"refs/heads/{branch}", cwd=self.repo
        )
        assert check.returncode == 0, "Branch should be preserved on conflict"

        # Clean up conflict state and branch
        _git("merge", "--abort", cwd=self.repo)
        _git("branch", "-D", branch, cwd=self.repo)

    # -----------------------------------------------------------------------
    # 2. manual_merge auto-commits before merging
    # -----------------------------------------------------------------------
    def test_manual_merge_auto_commits_uncommitted(self) -> None:
        """Uncommitted changes in worktree are committed before manual merge."""
        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        (wt_dir / "new.txt").write_text("uncommitted\n")

        msg = agent.manual_merge()
        assert "ready for review" in msg

        # The file should be in the working tree (unstaged after reset HEAD)
        assert (self.repo / "new.txt").exists()

    # -----------------------------------------------------------------------
    # 3. discard when worktree directory already removed
    # -----------------------------------------------------------------------
    def test_discard_worktree_already_removed(self) -> None:
        """discard() works even if the worktree directory was already deleted."""
        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        branch = agent._wt_branch
        assert wt_dir is not None and branch is not None

        # Manually remove worktree
        _git("worktree", "remove", str(wt_dir), "--force", cwd=self.repo)
        _git("worktree", "prune", cwd=self.repo)

        msg = agent.discard()
        assert "Discarded" in msg
        assert agent._wt_branch is None

        # Branch gone
        check = _git(
            "rev-parse", "--verify", f"refs/heads/{branch}", cwd=self.repo
        )
        assert check.returncode != 0

    # -----------------------------------------------------------------------
    # 4. merge conflict: main worktree is clean after abort
    # -----------------------------------------------------------------------
    def test_merge_conflict_leaves_clean_worktree(self) -> None:
        """After a merge conflict, the merge is aborted and the main worktree
        has no uncommitted changes."""
        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        (wt_dir / "README.md").write_text("wt\n")
        (self.repo / "README.md").write_text("main\n")
        subprocess.run(
            ["git", "-C", str(self.repo), "add", "."],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-m", "conflict"],
            capture_output=True,
            check=True,
        )

        msg = agent.merge()
        assert "Merge conflict" in msg

        # Main worktree is clean
        status = _git("status", "--porcelain", cwd=self.repo)
        assert status.stdout.strip() == ""

        # Current branch is the original branch
        head = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=self.repo)
        assert head.stdout.strip() == "main"

        agent.discard()

    # -----------------------------------------------------------------------
    # 5. Branch naming convention is preserved
    # -----------------------------------------------------------------------
    def test_branch_name_format(self) -> None:
        """Branch name must be kiss/wt-{chat_id[:12]}-{timestamp}."""
        agent = self._agent(chat_id="abcdef123456789xyz")
        agent.run(prompt_template="t", work_dir=str(self.repo))
        branch = agent._wt_branch
        assert branch is not None
        assert branch.startswith("kiss/wt-abcdef123456-")
        # The rest is a unix timestamp (digits only)
        suffix = branch.split("-", 2)[-1]
        assert suffix.isdigit()
        agent.discard()

    # -----------------------------------------------------------------------
    # 6. Worktree directory naming convention
    # -----------------------------------------------------------------------
    def test_worktree_dir_naming(self) -> None:
        """Worktree dir: repo/.kiss-worktrees/{branch with / replaced by _}."""
        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        branch = agent._wt_branch
        assert wt_dir is not None and branch is not None
        expected_slug = branch.replace("/", "_")
        # Resolve both to handle macOS /var → /private/var symlink
        assert wt_dir.resolve() == (
            self.repo / ".kiss-worktrees" / expected_slug
        ).resolve()
        assert wt_dir.exists()
        agent.discard()

    # -----------------------------------------------------------------------
    # 7. Git config key format for original branch
    # -----------------------------------------------------------------------
    def test_git_config_key_format(self) -> None:
        """Original branch is stored as branch.{name}.kiss-original."""
        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))
        branch = agent._wt_branch
        assert branch is not None

        cfg = _git(
            "config", f"branch.{branch}.kiss-original", cwd=self.repo
        )
        assert cfg.returncode == 0
        assert cfg.stdout.strip() == "main"
        agent.discard()

    # -----------------------------------------------------------------------
    # 8. merge() uses --no-edit flag (no editor prompt)
    # -----------------------------------------------------------------------
    def test_merge_fast_forwards_when_possible(self) -> None:
        """merge() uses --no-edit; when no divergence, git fast-forwards."""
        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        (wt_dir / "file.txt").write_text("data\n")

        msg = agent.merge()
        assert "Successfully merged" in msg

        # The file is on main after merge
        assert (self.repo / "file.txt").exists()
        assert (self.repo / "file.txt").read_text() == "data\n"

    # -----------------------------------------------------------------------
    # 9. manual_merge unstages changes (reset HEAD) on success
    # -----------------------------------------------------------------------
    def test_manual_merge_unstages_on_success(self) -> None:
        """After successful manual_merge, changes should be unstaged (not in
        index) so the user can selectively stage in Source Control."""
        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        (wt_dir / "review.txt").write_text("for review\n")

        msg = agent.manual_merge()
        assert "ready for review" in msg

        # Changes should NOT be staged (git diff --cached should be empty)
        cached = _git("diff", "--cached", "--quiet", cwd=self.repo)
        assert cached.returncode == 0, "Changes should be unstaged"

        # But they should exist as working tree modifications
        status = _git("status", "--porcelain", cwd=self.repo)
        assert "review.txt" in status.stdout

    # -----------------------------------------------------------------------
    # 10. merge conflict preserves _wt_branch for discard
    # -----------------------------------------------------------------------
    def test_merge_conflict_preserves_state_for_discard(self) -> None:
        """After merge conflict, _wt_branch is preserved so discard() works."""
        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        (wt_dir / "README.md").write_text("wt\n")
        (self.repo / "README.md").write_text("main\n")
        subprocess.run(
            ["git", "-C", str(self.repo), "add", "."],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-m", "conflict"],
            capture_output=True,
            check=True,
        )

        msg = agent.merge()
        assert "Merge conflict" in msg
        # State preserved
        assert agent._wt_branch is not None
        assert agent._original_branch is not None

        # Can discard
        msg = agent.discard()
        assert "Discarded" in msg
        assert agent._wt_branch is None

    # -----------------------------------------------------------------------
    # 11. run() result format: task output + separator + merge instructions
    # -----------------------------------------------------------------------
    def test_run_result_format(self) -> None:
        """run() appends merge instructions after '---' separator."""
        agent = self._agent()
        result = agent.run(prompt_template="t", work_dir=str(self.repo))
        parts = result.split("\n\n---\n")
        assert len(parts) == 2
        assert "test done" in parts[0]
        assert "agent.merge()" in parts[1]
        assert "agent.discard()" in parts[1]
        agent.discard()

    # -----------------------------------------------------------------------
    # 12. _delete_branch fallback: -d fails → -D
    # -----------------------------------------------------------------------
    def test_delete_branch_force_fallback(self) -> None:
        """When -d fails (unmerged branch), -D is used as fallback."""
        agent = self._agent()
        agent._repo_root = self.repo

        # Create a branch with a commit not on main (unmerged)
        _git("checkout", "-b", "unmerged-test", cwd=self.repo)
        (self.repo / "extra.txt").write_text("extra\n")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-m", "unmerged commit", cwd=self.repo)
        _git("checkout", "main", cwd=self.repo)

        # -d would fail because unmerged, but _delete_branch uses -D fallback
        agent._delete_branch("unmerged-test")
        check = _git(
            "rev-parse", "--verify", "refs/heads/unmerged-test", cwd=self.repo
        )
        assert check.returncode != 0

    # -----------------------------------------------------------------------
    # 13. Multiple cleanup calls with mixed orphans
    # -----------------------------------------------------------------------
    def test_cleanup_multiple_orphans(self) -> None:
        """cleanup() deletes all orphaned kiss/wt-* branches."""
        # Create orphan branches (no worktrees)
        _git("branch", "kiss/wt-orphan1-100", cwd=self.repo)
        _git("branch", "kiss/wt-orphan2-200", cwd=self.repo)

        result = WorktreeSorcarAgent.cleanup(self.repo)
        assert "Deleted" in result

        # Both should be gone
        for b in ["kiss/wt-orphan1-100", "kiss/wt-orphan2-200"]:
            check = _git(
                "rev-parse", "--verify", f"refs/heads/{b}", cwd=self.repo
            )
            assert check.returncode != 0

    # -----------------------------------------------------------------------
    # 14. Worktree prune is called during merge path
    # -----------------------------------------------------------------------
    def test_merge_prunes_worktree_bookkeeping(self) -> None:
        """merge() cleans up git worktree bookkeeping (prune)."""
        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))

        # Before merge: worktree is listed
        wt_list = _git("worktree", "list", "--porcelain", cwd=self.repo)
        assert ".kiss-worktrees" in wt_list.stdout

        agent.merge()

        # After merge: no kiss worktrees listed
        wt_list = _git("worktree", "list", "--porcelain", cwd=self.repo)
        assert ".kiss-worktrees" not in wt_list.stdout

    # -----------------------------------------------------------------------
    # 15. Worktree prune is called during discard path
    # -----------------------------------------------------------------------
    def test_discard_prunes_worktree_bookkeeping(self) -> None:
        """discard() cleans up git worktree bookkeeping (prune)."""
        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))

        agent.discard()

        wt_list = _git("worktree", "list", "--porcelain", cwd=self.repo)
        assert ".kiss-worktrees" not in wt_list.stdout

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
    def test_wt_pending_reflects_branch_state(self) -> None:
        """_wt_pending is True iff _wt_branch is not None."""
        agent = self._agent()
        assert not agent._wt_pending

        agent.run(prompt_template="t", work_dir=str(self.repo))
        assert agent._wt_pending

        agent.merge()
        assert not agent._wt_pending

    # -----------------------------------------------------------------------
    # 18. Restore picks latest branch when multiple exist
    # -----------------------------------------------------------------------
    def test_restore_picks_latest_branch(self) -> None:
        """When multiple kiss/wt-{chat_id}* branches exist,
        _restore_from_git picks the lexicographically last one (latest)."""
        chat_id = "multi_branch_test"
        agent = self._agent(chat_id=chat_id)
        prefix = f"kiss/wt-{chat_id[:12]}"

        # Create two branches manually
        _git("branch", f"{prefix}-1000", cwd=self.repo)
        _git(
            "config",
            f"branch.{prefix}-1000.kiss-original",
            "main",
            cwd=self.repo,
        )
        _git("branch", f"{prefix}-2000", cwd=self.repo)
        _git(
            "config",
            f"branch.{prefix}-2000.kiss-original",
            "main",
            cwd=self.repo,
        )

        agent._repo_root = self.repo
        agent._restore_from_git()
        assert agent._wt_branch == f"{prefix}-2000"

        # Clean up
        _git("branch", "-D", f"{prefix}-1000", cwd=self.repo)
        _git("branch", "-D", f"{prefix}-2000", cwd=self.repo)

    # -----------------------------------------------------------------------
    # 19. Merge unknown original → graceful error with manual instructions
    # -----------------------------------------------------------------------
    def test_merge_unknown_original_branch(self) -> None:
        """merge() with None original_branch returns a helpful error."""
        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))
        branch = agent._wt_branch
        assert branch is not None

        agent._original_branch = None
        msg = agent.merge()
        assert "Cannot merge" in msg
        assert "original branch is unknown" in msg
        assert branch in msg  # should reference the task branch

        # Restore for cleanup
        agent._original_branch = "main"
        agent.discard()

    # -----------------------------------------------------------------------
    # 20. manual_merge uses --no-commit --no-ff flags
    # -----------------------------------------------------------------------
    def test_manual_merge_no_commit_no_ff(self) -> None:
        """manual_merge creates NO commit — changes are left for user review."""
        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        (wt_dir / "manual.txt").write_text("manual\n")

        # Record commit count before
        log_before = _git("log", "--oneline", cwd=self.repo)
        count_before = len(log_before.stdout.strip().splitlines())

        agent.manual_merge()

        # No new commit should have been created on main
        log_after = _git("log", "--oneline", cwd=self.repo)
        count_after = len(log_after.stdout.strip().splitlines())
        assert count_after == count_before

    # -----------------------------------------------------------------------
    # 21. manual_merge deletes branch on success (no conflict)
    # -----------------------------------------------------------------------
    def test_manual_merge_deletes_branch_on_success(self) -> None:
        """On successful manual_merge (no conflicts), the task branch is
        deleted."""
        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))
        branch = agent._wt_branch
        wt_dir = agent._wt_dir
        assert branch is not None and wt_dir is not None

        (wt_dir / "m.txt").write_text("m\n")
        agent.manual_merge()

        check = _git(
            "rev-parse", "--verify", f"refs/heads/{branch}", cwd=self.repo
        )
        assert check.returncode != 0, "Branch should be deleted on success"

    # -----------------------------------------------------------------------
    # 22. auto_commit creates a commit with all files staged
    # -----------------------------------------------------------------------
    def test_auto_commit_stages_all_files(self) -> None:
        """_auto_commit_worktree does git add -A (tracks new + modified)."""
        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        # Create new file AND modify existing
        (wt_dir / "new.txt").write_text("new\n")
        (wt_dir / "README.md").write_text("modified\n")

        committed = agent._auto_commit_worktree()
        assert committed is True

        # Verify both are committed
        show = _git("show", "--name-only", "--format=", "HEAD", cwd=wt_dir)
        assert "new.txt" in show.stdout
        assert "README.md" in show.stdout

        agent.discard()

    # -----------------------------------------------------------------------
    # 23. Worktree is a proper git branch off current HEAD
    # -----------------------------------------------------------------------
    def test_worktree_branch_based_on_current_head(self) -> None:
        """The worktree branch starts from the same commit as the main branch."""
        # Add a second commit so we have a non-trivial history
        (self.repo / "file2.txt").write_text("v2\n")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-m", "second commit", cwd=self.repo)

        main_head = _git("rev-parse", "HEAD", cwd=self.repo).stdout.strip()

        agent = self._agent()
        agent.run(prompt_template="t", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        wt_head = _git("rev-parse", "HEAD", cwd=wt_dir).stdout.strip()
        assert wt_head == main_head

        agent.discard()

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
    def test_cleanup_removes_config_for_orphans(self) -> None:
        """cleanup() also removes the branch.{name}.* config section."""
        branch = "kiss/wt-cfgclean-100"
        _git("branch", branch, cwd=self.repo)
        _git(
            "config",
            f"branch.{branch}.kiss-original",
            "main",
            cwd=self.repo,
        )

        # Config exists
        cfg = _git("config", f"branch.{branch}.kiss-original", cwd=self.repo)
        assert cfg.returncode == 0

        WorktreeSorcarAgent.cleanup(self.repo)

        # Config gone
        cfg = _git("config", f"branch.{branch}.kiss-original", cwd=self.repo)
        assert cfg.returncode != 0
