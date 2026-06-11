# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""BUGHUNT-4: a failed baseline commit must not leave inconsistent state.

``_try_setup_worktree`` copies the user's dirty state into the new
worktree and commits it as a baseline with ``--no-verify``.  But
``--no-verify`` only skips the ``pre-commit`` and ``commit-msg`` hooks
— a failing ``prepare-commit-msg`` hook (or a stale ``index.lock``,
missing committer identity, ...) still makes the commit fail.  The
old code silently continued with ``baseline_commit = None`` while the
user's dirty files sat UNCOMMITTED in the worktree, so they were later
auto-committed as (and attributed to) agent work and squash-merged
back into the original branch — duplicating the user's edits.

The fix honors ``run()``'s documented fallback contract ("Any git
command fails during setup" → direct execution): clean up the
half-created worktree/branch and return ``None``.

These integration tests use real on-disk git repos (no mocks).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import _git
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _make_repo(path: Path) -> Path:
    """Create a git repo on branch ``main`` with one initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True,
        check=True,
    )
    for key, val in (("user.email", "t@t.com"), ("user.name", "T")):
        subprocess.run(
            ["git", "-C", str(path), "config", key, val],
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


def _install_failing_prepare_commit_msg_hook(repo: Path) -> None:
    """Install a prepare-commit-msg hook that rejects every commit."""
    hooks = repo / ".git" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    hook = hooks / "prepare-commit-msg"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)


def _kiss_branches(repo: Path) -> list[str]:
    """All kiss/wt-* branch names in *repo*."""
    result = _git(
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/heads/kiss/wt-*",
        cwd=repo,
    )
    return [b for b in result.stdout.strip().splitlines() if b]


class TestBaselineCommitFailure:
    """A baseline commit rejected by a hook must trigger clean fallback."""

    def test_setup_falls_back_and_leaves_no_partial_state(self) -> None:
        """When the baseline commit fails, _try_setup_worktree must NOT
        hand out a worktree whose uncommitted user dirty state would be
        attributed to the agent — it must clean up and return None."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            _install_failing_prepare_commit_msg_hook(repo)
            # User dirty state that needs a baseline commit.
            (repo / "README.md").write_text("# Test\nuser edit\n")

            agent = WorktreeSorcarAgent("bh4-baseline-fail")
            result = agent._try_setup_worktree(repo, None)

            assert result is None, (
                "expected graceful fallback (None) when the baseline "
                f"commit fails, got a worktree work dir: {result}"
            )
            assert agent._wt is None
            assert _kiss_branches(repo) == [], (
                "partial worktree branch left behind after the "
                "baseline commit failed"
            )
            wt_root = repo / ".kiss-worktrees"
            leftovers = (
                [p.name for p in wt_root.iterdir()] if wt_root.is_dir() else []
            )
            assert leftovers == [], (
                f"partial worktree dirs left behind: {leftovers}"
            )
            # The user's dirty edit in the main tree is untouched.
            assert (repo / "README.md").read_text() == "# Test\nuser edit\n"

    def test_clean_main_tree_is_unaffected_by_failing_hook(self) -> None:
        """With a CLEAN main tree no baseline commit is needed, so the
        failing hook must not prevent worktree creation."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            _install_failing_prepare_commit_msg_hook(repo)

            agent = WorktreeSorcarAgent("bh4-baseline-clean")
            result = agent._try_setup_worktree(repo, None)

            assert result is not None
            assert agent._wt is not None
            assert agent._wt.baseline_commit is None
