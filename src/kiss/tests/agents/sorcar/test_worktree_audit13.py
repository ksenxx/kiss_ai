"""Audit 13: Integration tests for further bugs and redundancies.

BUG-63: ``discard()`` lies about success when the main repo cannot
    checkout the original branch.  The code returns
    ``"Discarded branch '<X>'"`` with a checkout warning, but also
    unconditionally calls ``delete_branch(repo, wt.branch)`` which
    silently fails because git refuses to delete the currently
    checked-out branch.  The branch persists in git but the user
    believes it was discarded.

RED-7: Several methods in ``GitWorktreeOps`` are dead code — never
    called by any production module, only possibly referenced in
    tests.  They add attack surface and maintenance burden without
    providing any value:

    * ``apply_branch_to_working_tree``
    * ``would_merge_conflict``
    * ``branch_diff_files``
    * ``merge_branch`` (non-squash variant; ``squash_merge_branch`` is
      used instead)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo(tmp_path: Path, name: str = "repo") -> Path:
    """Create a minimal git repo with one commit."""
    repo = tmp_path / name
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"],
        cwd=repo,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"],
        cwd=repo,
        capture_output=True,
    )
    (repo / "init.txt").write_text("init\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        capture_output=True,
    )
    return repo


# ===================================================================
# BUG-63: discard() lies when delete_branch silently fails
# ===================================================================


class TestBug63DiscardLiesOnDeleteFailure:
    """BUG-63: ``discard()`` tells the user the branch was discarded
    even when ``delete_branch`` silently fails because the main repo
    is currently on that branch.

    This happens when:
        1. ``wt.original_branch`` is None (checkout skipped), OR
        2. ``checkout(wt.original_branch)`` fails (dirty tree,
           missing branch, etc.)

    In both cases, the main repo remains on ``wt.branch``.  git
    refuses ``branch -d`` / ``-D`` on the currently-checked-out
    branch, so the branch persists.  The user sees
    ``"Discarded branch 'X'"`` (possibly with a checkout warning)
    but believes the branch is gone.

    FIX: ``delete_branch`` returns ``bool``; ``discard()`` checks the
    return value and surfaces a clear failure message when the
    branch could not be removed.
    """

    def test_discard_with_none_original_branch_lies(
        self,
        tmp_path: Path,
    ) -> None:
        """When original_branch is None, no checkout is attempted.
        The main repo remains on the kiss/wt-* branch, so
        delete_branch fails — but the message still says 'Discarded'.
        """
        repo = _make_repo(tmp_path)

        # Simulate the setup: main repo is on the kiss/wt-* branch
        # directly (no separate worktree).  Equivalent to a crash
        # scenario where the main repo got switched to the branch.
        branch = "kiss/wt-bug63a-1"
        subprocess.run(
            ["git", "checkout", "-b", branch],
            cwd=repo,
            capture_output=True,
        )

        # Build agent state pointing at a non-existent worktree dir
        # so `remove` is a no-op.
        agent = WorktreeSorcarAgent("test")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch=None,  # checkout skipped
            wt_dir=repo / ".kiss-worktrees" / "nonexistent",
            baseline_commit=None,
        )

        msg = agent.discard()

        # BUG-63: the branch still exists because we are on it,
        # yet the message claims discard success.
        branch_still_exists = GitWorktreeOps.branch_exists(repo, branch)

        # With the fix, the message must WARN about the failure.
        assert branch_still_exists, (
            "Setup sanity check: branch should still exist because "
            "git cannot delete the currently checked-out branch"
        )
        # FIX assertion: message must NOT claim straight success.
        lower = msg.lower()
        assert ("could not" in lower) or ("failed" in lower) or ("still exists" in lower), (
            "BUG-63: discard() claimed success but branch still "
            "exists.  Message must surface the failure.  Got:\n"
            f"{msg}"
        )

    def test_discard_with_failed_checkout_warns_about_branch(
        self,
        tmp_path: Path,
    ) -> None:
        """When checkout fails AND delete_branch also fails, the
        user must be warned about the branch persisting, not just
        the checkout issue.
        """
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug63b-1"
        # Set up: main is on the kiss branch (simulates post-crash state)
        subprocess.run(
            ["git", "checkout", "-b", branch],
            cwd=repo,
            capture_output=True,
        )
        # original_branch references a nonexistent branch so checkout fails
        agent = WorktreeSorcarAgent("test")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="nonexistent-branch",
            wt_dir=repo / ".kiss-worktrees" / "nonexistent",
            baseline_commit=None,
        )

        msg = agent.discard()

        # The branch should still exist (can't delete current)
        assert GitWorktreeOps.branch_exists(repo, branch)
        # Message must mention that delete failed / branch kept
        lower = msg.lower()
        assert "could not" in lower or "failed" in lower or ("still exists" in lower), (
            f"BUG-63: message does not warn that branch remains.\n{msg}"
        )

    def test_discard_normal_case_still_works(
        self,
        tmp_path: Path,
    ) -> None:
        """Regression: a clean discard must still succeed cleanly."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug63c-1"
        wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")

        agent = WorktreeSorcarAgent("test")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        )

        msg = agent.discard()
        assert "Discarded branch" in msg
        # No failure warnings in happy path
        lower = msg.lower()
        assert "could not" not in lower, (
            f"Regression: clean discard message contains failure warning.  Got:\n{msg}"
        )
        assert not GitWorktreeOps.branch_exists(repo, branch)

    def test_delete_branch_returns_bool(self) -> None:
        """The FIX requires ``delete_branch`` to return a bool so
        discard() can check the outcome.  This test confirms the
        signature change.
        """
        import inspect

        sig = inspect.signature(GitWorktreeOps.delete_branch)
        ret = sig.return_annotation
        # After the fix, return annotation should be ``bool`` (may
        # be the class or the string "bool" under PEP 563).
        assert ret is bool or ret == "bool", (
            f"BUG-63 FIX: delete_branch must return bool, got {ret!r}"
        )


# ===================================================================
# RED-7: Dead-code removal
# ===================================================================


class TestRed7DeadCodeRemoved:
    """RED-7: several ``GitWorktreeOps`` methods are never called
    by production code:

    * ``apply_branch_to_working_tree``
    * ``would_merge_conflict``
    * ``branch_diff_files``
    * ``merge_branch`` (non-squash; ``squash_merge_branch`` is used)

    Dead code adds maintenance burden, expands attack surface, and
    confuses readers.  FIX: remove them.
    """

    def test_apply_branch_to_working_tree_removed(self) -> None:
        assert not hasattr(GitWorktreeOps, "apply_branch_to_working_tree"), (
            "RED-7: dead code — apply_branch_to_working_tree is unused in production; remove it."
        )

    def test_would_merge_conflict_removed(self) -> None:
        assert not hasattr(GitWorktreeOps, "would_merge_conflict"), (
            "RED-7: dead code — would_merge_conflict is unused; remove it."
        )

    def test_branch_diff_files_removed(self) -> None:
        assert not hasattr(GitWorktreeOps, "branch_diff_files"), (
            "RED-7: dead code — branch_diff_files is unused; remove it."
        )

    def test_non_squash_merge_branch_removed(self) -> None:
        assert not hasattr(GitWorktreeOps, "merge_branch"), (
            "RED-7: dead code — merge_branch (non-squash) is "
            "unused; squash_merge_branch is used instead."
        )

    def test_still_used_methods_remain(self) -> None:
        """Regression sanity: methods that ARE used must remain."""
        for name in (
            "squash_merge_branch",
            "squash_merge_from_baseline",
            "stash_if_dirty",
            "stash_pop",
            "checkout",
            "create",
            "remove",
            "prune",
            "commit_staged",
            "has_uncommitted_changes",
            "delete_branch",
            "branch_exists",
            "load_original_branch",
            "save_original_branch",
            "load_baseline_commit",
            "save_baseline_commit",
            "copy_dirty_state",
            "head_sha",
            "current_branch",
            "discover_repo",
            "cleanup_partial",
            "cleanup_orphans",
            "ensure_excluded",
            "find_pending_branch",
            "stage_all",
            "staged_diff",
            "unstaged_files",
            "staged_files",
        ):
            assert hasattr(GitWorktreeOps, name), (
                f"Regression: still-used method {name!r} missing from GitWorktreeOps"
            )
