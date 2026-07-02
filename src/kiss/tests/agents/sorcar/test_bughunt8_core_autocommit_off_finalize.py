# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt pass #8: ``auto_commit_enabled=False`` contract of
``WorktreeSorcarAgent._finalize_worktree``.

The ``auto_commit_enabled`` attribute docstring (in
``WorktreeSorcarAgent.__init__``) promises:

    When False, ``_auto_commit_worktree`` is a no-op so the worktree's
    uncommitted changes are preserved for manual review (and
    ``_finalize_worktree`` returns False, keeping the worktree
    directory in place).

Before the fix, ``_finalize_worktree``'s "late-arriver retry"
(``GitWorktreeOps.commit_all``) ran unconditionally, so with
``auto_commit_enabled=False`` and a dirty worktree the user's
uncommitted changes were silently committed with the generic
"kiss: auto-commit late-arriving changes" message, the worktree
directory was removed, and the release path went on to squash-merge
the branch — exactly the behavior ``--no-auto-commit`` exists to
prevent.

Real git repo, real worktree, real files — no mocks.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    _git(path, "config", "user.email", "t@t.com")
    _git(path, "config", "user.name", "T")
    _git(path, "config", "commit.gpgsign", "false")
    (path / "seed.txt").write_text("seed\n")
    _git(path, "add", "seed.txt")
    _git(path, "commit", "-q", "-m", "seed")
    return path


def _setup_agent(tmp: Path, slug: str) -> tuple[Path, Path, str, WorktreeSorcarAgent]:
    """Create repo + worktree and an agent tracking that worktree."""
    repo = _make_repo(tmp / "repo")
    branch = f"kiss/wt-bughunt8-{slug}"
    wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    _git(wt_dir, "config", "user.email", "t@t.com")
    _git(wt_dir, "config", "user.name", "T")
    agent = WorktreeSorcarAgent("bughunt8")
    agent._wt = GitWorktree(
        repo_root=repo,
        branch=branch,
        original_branch="main",
        wt_dir=wt_dir,
        baseline_commit=None,
    )
    return repo, wt_dir, branch, agent


class TestAutoCommitOffFinalizePreservesWorktree(unittest.TestCase):
    """``auto_commit_enabled=False`` + dirty worktree must preserve it."""

    def test_finalize_worktree_returns_false_and_preserves_dirty_worktree(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, wt_dir, branch, agent = _setup_agent(Path(tmp), "finalize")
            agent.auto_commit_enabled = False
            (wt_dir / "uncommitted.txt").write_text("manual review me\n")
            branch_sha_before = _git(repo, "rev-parse", branch)

            finalized = agent._finalize_worktree()

            assert finalized is False, (
                "auto_commit_enabled=False with uncommitted changes must "
                "make _finalize_worktree return False (contract in the "
                "auto_commit_enabled attribute docstring)"
            )
            assert wt_dir.exists(), (
                "worktree directory must be kept in place for manual review"
            )
            assert GitWorktreeOps.has_uncommitted_changes(wt_dir), (
                "uncommitted changes must be preserved, not committed away"
            )
            branch_sha_after = _git(repo, "rev-parse", branch)
            assert branch_sha_before == branch_sha_after, (
                "no commit may be created on the task branch when "
                "auto-commit is disabled"
            )

    def test_release_worktree_end_to_end_does_not_merge_or_delete(
        self,
    ) -> None:
        """End-to-end release path (used by ``new_chat`` / next-task
        boot): with auto-commit off and a dirty worktree, nothing may
        be committed, merged, or removed."""
        with tempfile.TemporaryDirectory() as tmp:
            repo, wt_dir, branch, agent = _setup_agent(Path(tmp), "release")
            agent.auto_commit_enabled = False
            (wt_dir / "uncommitted.txt").write_text("manual review me\n")
            main_sha_before = _git(repo, "rev-parse", "main")

            released = agent._release_worktree()

            assert released is None, (
                "release must not report success when the worktree was "
                "preserved for manual review"
            )
            assert wt_dir.exists(), (
                "worktree directory must survive _release_worktree when "
                "auto-commit is disabled and changes are uncommitted"
            )
            assert GitWorktreeOps.has_uncommitted_changes(wt_dir), (
                "uncommitted changes must be preserved, not committed away"
            )
            assert _git(repo, "rev-parse", "main") == main_sha_before, (
                "the original branch must not receive a squash-merge of "
                "changes the user chose not to auto-commit"
            )
            assert (wt_dir / "uncommitted.txt").exists()

    def test_finalize_worktree_auto_commit_off_clean_worktree_cleans_up(
        self,
    ) -> None:
        """Regression guard: with auto-commit off but NOTHING
        uncommitted, finalize still cleans up the worktree normally."""
        with tempfile.TemporaryDirectory() as tmp:
            repo, wt_dir, branch, agent = _setup_agent(Path(tmp), "clean")
            agent.auto_commit_enabled = False

            finalized = agent._finalize_worktree()

            assert finalized is True
            assert not wt_dir.exists()

    def test_finalize_worktree_auto_commit_on_still_commits(self) -> None:
        """Regression guard: default auto-commit-on behavior unchanged
        (dirty worktree is committed and the directory removed).

        Uses the same LLM-free fallback pattern as
        ``test_cli_default_modes.py`` — the commit-message generator
        raises, so ``auto_commit_changes`` falls back to its generic
        message.
        """
        import kiss.core.kiss_agent as kiss_agent_mod

        with tempfile.TemporaryDirectory() as tmp:
            repo, wt_dir, branch, agent = _setup_agent(Path(tmp), "on")
            agent.auto_commit_enabled = True
            (wt_dir / "agent_work.txt").write_text("agent output\n")
            branch_sha_before = _git(repo, "rev-parse", branch)

            saved = kiss_agent_mod.KISSAgent

            class _RaisingAgent:
                def __init__(self, *_a: object, **_kw: object) -> None:
                    pass

                def run(self, *_a: object, **_kw: object) -> str:
                    raise RuntimeError("no LLM in test")

            kiss_agent_mod.KISSAgent = _RaisingAgent  # type: ignore[misc, assignment]
            try:
                finalized = agent._finalize_worktree()
            finally:
                kiss_agent_mod.KISSAgent = saved  # type: ignore[misc]

            assert finalized is True
            assert not wt_dir.exists()
            assert _git(repo, "rev-parse", branch) != branch_sha_before


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
