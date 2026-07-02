# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8: merge commit message when a FILE shares the branch name.

``GitWorktreeOps._merge_commit_message`` reads the worktree branch's
HEAD message with ``git log -1 --format=%B <branch>``.  Without a
``--`` end-of-revisions terminator, git refuses the command with
``fatal: ambiguous argument '<branch>': both revision and filename``
whenever the user's repo contains a file whose path equals the branch
name.  The merge then silently degrades to the synthetic
``"kiss: merged from <branch>"`` placeholder instead of the agent's
meaningful per-task commit message — contradicting the documented
contract that the fallback fires "only when the branch's HEAD message
cannot be read (corrupt branch, git error)".

These tests drive the real :mod:`kiss.agents.sorcar.git_worktree`
operations against actual on-disk git repositories (no mocks).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    MergeResult,
    _git,
)

_BRANCH = "kisswt-msgtest"

_AGENT_MSG = (
    "feat: add feature.txt\n"
    "\n"
    "Detailed body describing exactly what the agent changed and why."
)


def _make_repo(path: Path) -> Path:
    """Create a git repo with one initial commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"], check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"], check=True,
    )
    return path


def _setup_branch_and_ambiguous_file(repo: Path) -> None:
    """Create branch ``_BRANCH`` with one agent commit, plus a tracked
    file named exactly ``_BRANCH`` on main."""
    wt_dir = repo / ".kiss-worktrees" / "kiss_wt-msgtest"
    assert GitWorktreeOps.create(repo, _BRANCH, wt_dir)
    subprocess.run(
        ["git", "-C", str(wt_dir), "config", "user.email", "t@t.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(wt_dir), "config", "user.name", "T"], check=True,
    )
    (wt_dir / "feature.txt").write_text("hello\n")
    _git("add", "feature.txt", cwd=wt_dir)
    commit = _git("commit", "-m", _AGENT_MSG, cwd=wt_dir)
    assert commit.returncode == 0, commit.stderr

    # A user file on main whose name equals the branch name makes
    # ``git log <branch>`` ambiguous (revision vs. filename).
    (repo / _BRANCH).write_text("user file that shadows the branch name\n")
    _git("add", _BRANCH, cwd=repo)
    commit = _git("commit", "-m", "add file shadowing branch name", cwd=repo)
    assert commit.returncode == 0, commit.stderr


def _head_message(repo: Path) -> str:
    """Return the full commit message of HEAD in *repo*."""
    result = _git("log", "-1", "--format=%B", "HEAD", "--", cwd=repo)
    return result.stdout.rstrip()


class TestMergeMessageWithBranchNamedFile:
    """The merge commit must carry the branch HEAD message even when a
    tracked file shares the branch's name."""

    def test_squash_merge_branch_uses_head_message(self) -> None:
        """``squash_merge_branch`` merge message == branch HEAD message."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            _setup_branch_and_ambiguous_file(repo)

            result = GitWorktreeOps.squash_merge_branch(repo, _BRANCH)

            assert result == MergeResult.SUCCESS
            assert _head_message(repo) == _AGENT_MSG

    def test_squash_merge_from_baseline_uses_head_message(self) -> None:
        """``squash_merge_from_baseline`` merge message == branch HEAD
        message."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            baseline = GitWorktreeOps.head_sha(repo)
            assert baseline is not None
            _setup_branch_and_ambiguous_file(repo)

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, _BRANCH, baseline
            )

            assert result == MergeResult.SUCCESS
            assert _head_message(repo) == _AGENT_MSG
