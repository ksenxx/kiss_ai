# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests locking down behavior before simplification.

Covers the exact code paths touched by the sorcar-misc simplification:

* ``GitWorktreeOps.copy_dirty_state`` rename handling (old-path removal
  for plain files, directories, and symlinks in the target worktree).
* ``WorktreeSorcarAgent._finalize_worktree`` /
  ``_preserve_pending_worktree_for_review`` (auto-commit, late-arriver
  retry, worktree removal) — exercised via real temp git repos on the
  no-LLM paths (clean worktree, or ``auto_commit_enabled=False``).
* ``WorktreeSorcarAgent.merge`` / ``discard`` / ``_release_worktree``
  happy paths.
* ``_manual_merge_cmd`` / ``_merge_fix_steps`` /
  ``_reject_interactive_only_flags`` pure helpers.

No mocks, patches, or fakes: every test drives real git repositories
created under ``tmp_path``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    strip_worktree_suffix,
)
from kiss.agents.sorcar.worktree_sorcar_agent import (
    WorktreeSorcarAgent,
    _manual_merge_cmd,
    _merge_fix_steps,
    _reject_interactive_only_flags,
)


def _run_git(repo: Path, *args: str) -> str:
    """Run git in *repo*, asserting success, returning stdout."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _init_repo(tmp_path: Path) -> Path:
    """Create a git repo with one commit and return its root."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test")
    _run_git(repo, "config", "commit.gpgsign", "false")
    (repo / "a.txt").write_text("hello\n")
    _run_git(repo, "add", "-A")
    _run_git(repo, "commit", "-m", "initial")
    return repo


def _make_worktree(repo: Path, slug: str = "kiss_wt-test-1") -> tuple[str, Path]:
    """Create a real agent-style worktree; return (branch, wt_dir)."""
    branch = f"kiss/wt-{slug}"
    wt_dir = repo / ".kiss-worktrees" / slug
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    return branch, wt_dir


def _make_agent(repo: Path, branch: str, wt_dir: Path) -> WorktreeSorcarAgent:
    """Build a WorktreeSorcarAgent with a pending worktree state."""
    agent = WorktreeSorcarAgent("regr-test")
    agent._wt = GitWorktree(
        repo_root=repo,
        branch=branch,
        original_branch=GitWorktreeOps.current_branch(repo),
        wt_dir=wt_dir,
        baseline_commit=None,
    )
    return agent


class TestCopyDirtyStateRename:
    """Rename old-path removal branches in ``copy_dirty_state``."""

    def test_rename_removes_old_file_and_copies_new(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _run_git(repo, "mv", "a.txt", "b.txt")
        _branch, wt_dir = _make_worktree(repo)
        assert (wt_dir / "a.txt").is_file()
        assert GitWorktreeOps.copy_dirty_state(repo, wt_dir) is True
        assert not (wt_dir / "a.txt").exists()
        assert (wt_dir / "b.txt").read_text() == "hello\n"

    def test_rename_old_path_is_directory_in_worktree(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _run_git(repo, "mv", "a.txt", "b.txt")
        _branch, wt_dir = _make_worktree(repo)
        # Simulate the worktree's old path being a directory.
        (wt_dir / "a.txt").unlink()
        (wt_dir / "a.txt").mkdir()
        (wt_dir / "a.txt" / "junk").write_text("x")
        assert GitWorktreeOps.copy_dirty_state(repo, wt_dir) is True
        assert not (wt_dir / "a.txt").exists()
        assert (wt_dir / "b.txt").read_text() == "hello\n"

    def test_rename_old_path_is_broken_symlink_in_worktree(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        _run_git(repo, "mv", "a.txt", "b.txt")
        _branch, wt_dir = _make_worktree(repo)
        (wt_dir / "a.txt").unlink()
        (wt_dir / "a.txt").symlink_to(wt_dir / "no-such-target")
        assert GitWorktreeOps.copy_dirty_state(repo, wt_dir) is True
        assert not (wt_dir / "a.txt").is_symlink()
        assert not (wt_dir / "a.txt").exists()
        assert (wt_dir / "b.txt").read_text() == "hello\n"

    def test_rename_old_path_symlink_to_dir_in_worktree(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        _run_git(repo, "mv", "a.txt", "b.txt")
        _branch, wt_dir = _make_worktree(repo)
        target = wt_dir / "realdir"
        target.mkdir()
        (wt_dir / "a.txt").unlink()
        (wt_dir / "a.txt").symlink_to(target)
        assert GitWorktreeOps.copy_dirty_state(repo, wt_dir) is True
        assert not (wt_dir / "a.txt").is_symlink()
        assert not (wt_dir / "a.txt").exists()
        # The symlink target itself must be untouched.
        assert target.is_dir()

    def test_clean_repo_returns_false(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _branch, wt_dir = _make_worktree(repo)
        assert GitWorktreeOps.copy_dirty_state(repo, wt_dir) is False


class TestFinalizeWorktree:
    """No-LLM paths of ``_finalize_worktree``."""

    def test_clean_worktree_is_removed(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        branch, wt_dir = _make_worktree(repo)
        agent = _make_agent(repo, branch, wt_dir)
        assert agent._finalize_worktree() is True
        assert not wt_dir.exists()
        assert GitWorktreeOps.branch_exists(repo, branch)

    def test_dirty_worktree_no_auto_commit_is_preserved(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        branch, wt_dir = _make_worktree(repo)
        agent = _make_agent(repo, branch, wt_dir)
        agent.auto_commit_enabled = False
        (wt_dir / "new.txt").write_text("work in progress\n")
        assert agent._finalize_worktree() is False
        assert wt_dir.exists()
        assert (wt_dir / "new.txt").read_text() == "work in progress\n"
        assert GitWorktreeOps.has_uncommitted_changes(wt_dir)

    def test_missing_worktree_dir_prunes_and_succeeds(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        branch, wt_dir = _make_worktree(repo)
        agent = _make_agent(repo, branch, wt_dir)
        import shutil

        shutil.rmtree(wt_dir)
        assert agent._finalize_worktree() is True


class TestPreserveForReview:
    """``_preserve_pending_worktree_for_review`` end-to-end."""

    def test_no_pending_worktree_returns_false(self, tmp_path: Path) -> None:
        agent = WorktreeSorcarAgent("regr-test")
        assert agent._preserve_pending_worktree_for_review() is False

    def test_dirty_worktree_preserved_uncommitted_no_auto_commit(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        branch, wt_dir = _make_worktree(repo)
        agent = _make_agent(repo, branch, wt_dir)
        # fixer3-F14: with auto-commit disabled the preserve path must
        # honor the ``--no-auto-commit`` contract — never force-commit
        # the user's reviewable changes via the late-arriver retry.
        # The worktree directory is preserved intact for manual review.
        agent.auto_commit_enabled = False
        agent._pending_review = True
        (wt_dir / "partial.txt").write_text("partial work\n")
        assert agent._preserve_pending_worktree_for_review() is True
        assert agent._wt is None
        assert agent._pending_review is False
        # Worktree dir survives with the change still uncommitted.
        assert wt_dir.exists()
        porcelain = _run_git(wt_dir, "status", "--porcelain")
        assert "partial.txt" in porcelain
        assert "partial.txt" not in _run_git(repo, "ls-files")
        # No forced "late-arriving" commit landed on the branch.
        log = _run_git(repo, "log", "-1", "--format=%s", branch)
        assert "late-arriving" not in log

    def test_clean_worktree_is_just_cleaned_up(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        branch, wt_dir = _make_worktree(repo)
        agent = _make_agent(repo, branch, wt_dir)
        agent._pending_review = True
        assert agent._preserve_pending_worktree_for_review() is True
        assert agent._wt is None
        assert agent._pending_review is False
        assert not wt_dir.exists()


class TestMergeDiscardRelease:
    """Happy paths of merge / discard / _release_worktree (no LLM)."""

    def test_merge_success(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        original = GitWorktreeOps.current_branch(repo)
        branch, wt_dir = _make_worktree(repo)
        agent = _make_agent(repo, branch, wt_dir)
        (wt_dir / "agent.txt").write_text("agent work\n")
        assert GitWorktreeOps.commit_all(wt_dir, "agent: did work") is True
        msg = agent.merge()
        assert "Successfully merged" in msg
        assert agent._wt is None
        assert (repo / "agent.txt").read_text() == "agent work\n"
        assert not GitWorktreeOps.branch_exists(repo, branch)
        assert GitWorktreeOps.current_branch(repo) == original
        # Squash-merge commit reuses the branch HEAD message.
        assert _run_git(repo, "log", "-1", "--format=%s").strip() == (
            "agent: did work"
        )

    def test_merge_without_pending_worktree_raises(self, tmp_path: Path) -> None:
        agent = WorktreeSorcarAgent("regr-test")
        with pytest.raises(RuntimeError):
            agent.merge()

    def test_merge_unknown_original_branch(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        branch, wt_dir = _make_worktree(repo)
        agent = _make_agent(repo, branch, wt_dir)
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch=None,
            wt_dir=wt_dir,
            baseline_commit=None,
        )
        msg = agent.merge()
        assert "Cannot merge" in msg
        assert f"git merge --squash {branch}" in msg

    def test_discard(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        original = GitWorktreeOps.current_branch(repo)
        branch, wt_dir = _make_worktree(repo)
        agent = _make_agent(repo, branch, wt_dir)
        (wt_dir / "junk.txt").write_text("junk\n")
        msg = agent.discard()
        assert f"Discarded branch '{branch}'" in msg
        assert agent._wt is None
        assert not wt_dir.exists()
        assert not GitWorktreeOps.branch_exists(repo, branch)
        assert not (repo / "junk.txt").exists()
        assert GitWorktreeOps.current_branch(repo) == original

    def test_discard_without_pending_worktree_raises(
        self, tmp_path: Path
    ) -> None:
        agent = WorktreeSorcarAgent("regr-test")
        with pytest.raises(RuntimeError):
            agent.discard()

    def test_release_worktree_merges_and_returns_original(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        original = GitWorktreeOps.current_branch(repo)
        branch, wt_dir = _make_worktree(repo)
        agent = _make_agent(repo, branch, wt_dir)
        (wt_dir / "released.txt").write_text("released\n")
        assert GitWorktreeOps.commit_all(wt_dir, "agent: released work") is True
        assert agent._release_worktree() == original
        assert agent._wt is None
        assert agent._merge_conflict_warning is None
        assert (repo / "released.txt").read_text() == "released\n"
        assert not GitWorktreeOps.branch_exists(repo, branch)

    def test_release_worktree_none_pending(self, tmp_path: Path) -> None:
        agent = WorktreeSorcarAgent("regr-test")
        assert agent._release_worktree() is None

    def test_release_worktree_no_auto_commit_preserves(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        branch, wt_dir = _make_worktree(repo)
        agent = _make_agent(repo, branch, wt_dir)
        agent.auto_commit_enabled = False
        (wt_dir / "keep.txt").write_text("keep\n")
        assert agent._release_worktree() is None
        assert agent._wt is None
        assert agent._merge_conflict_warning is not None
        assert str(wt_dir) in agent._merge_conflict_warning
        assert wt_dir.exists()
        assert (wt_dir / "keep.txt").read_text() == "keep\n"


class TestPureHelpers:
    """Pure-function helpers in worktree_sorcar_agent / git_worktree."""

    def _wt(self, baseline: str | None) -> GitWorktree:
        return GitWorktree(
            repo_root=Path("/repo"),
            branch="kiss/wt-x",
            original_branch="main",
            wt_dir=Path("/repo/.kiss-worktrees/kiss_wt-x"),
            baseline_commit=baseline,
        )

    def test_manual_merge_cmd_no_baseline(self) -> None:
        assert _manual_merge_cmd(self._wt(None)) == (
            "git merge --squash kiss/wt-x"
        )

    def test_manual_merge_cmd_with_baseline(self) -> None:
        assert _manual_merge_cmd(self._wt("abc123")) == (
            "git cherry-pick --no-commit abc123..kiss/wt-x"
        )

    def test_merge_fix_steps(self) -> None:
        # ``-D`` (force): after a squash merge the branch is never an
        # ancestor of the original branch, so ``-d`` would always
        # refuse — see test_worktree_manual_fix_steps.py.
        steps = _merge_fix_steps(self._wt(None), "    git commit\n")
        assert steps == (
            "    cd /repo\n"
            "    git checkout main\n"
            "    git merge --squash kiss/wt-x\n"
            "    git commit\n"
            "    git branch -D kiss/wt-x"
        )

    def test_reject_interactive_only_flags_ok(self) -> None:
        _reject_interactive_only_flags(["sorcar", "-t", "do stuff"])

    def test_reject_interactive_only_flags_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _reject_interactive_only_flags(
                ["sorcar", "--worktree", "-t", "x", "--worktree", "--auto-commit"]
            )
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        # Deduplicated, order-preserving flag list.
        assert "--worktree, --auto-commit" in err

    def test_strip_worktree_suffix(self) -> None:
        assert strip_worktree_suffix(
            "/u/p/.kiss-worktrees/kiss_wt-123/src"
        ) == "/u/p"
        assert strip_worktree_suffix("/u/p/src") == "/u/p/src"
        assert strip_worktree_suffix("") == ""
        assert strip_worktree_suffix(
            ".kiss-worktrees/kiss_wt-123/src"
        ) == "."
