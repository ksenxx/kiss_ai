# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the squash-merge commit message used by worktree merges.

When the worktree agent finishes a task, it auto-commits its changes
on the worktree branch (with an LLM-generated message), then
squash-merges that branch into the user's original branch.  The
merge commit on the original branch should carry the worktree
branch's HEAD commit message verbatim — NOT git's auto-generated
``"Squashed commit of the following:"`` text and NOT a synthetic
``"kiss: merged from <branch>"`` placeholder.

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


def _make_repo(path: Path) -> Path:
    """Create a git repo with one initial commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        check=True,
    )
    return path


def _create_worktree(repo: Path, branch: str) -> Path:
    """Create a worktree at repo/.kiss-worktrees/<slug>."""
    slug = branch.replace("/", "_")
    wt_dir = repo / ".kiss-worktrees" / slug
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    # Inherit user config in the worktree.
    subprocess.run(
        ["git", "-C", str(wt_dir), "config", "user.email", "t@t.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(wt_dir), "config", "user.name", "T"],
        check=True,
    )
    return wt_dir


def _head_message(repo: Path) -> str:
    """Return the full commit message of HEAD in *repo*."""
    result = _git("log", "-1", "--format=%B", "HEAD", cwd=repo)
    return result.stdout.rstrip()


class TestMergeCommitMessageFromBaseline:
    """squash_merge_from_baseline uses the worktree branch's last
    commit message as the merge commit message on the original
    branch.
    """

    def test_uses_branch_head_message(self) -> None:
        """The merge commit message equals the worktree branch HEAD
        commit message (subject + body) when the agent made one
        commit on the branch.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-merge-msg-1"
            wt_dir = _create_worktree(repo, branch)
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            (wt_dir / "feature.txt").write_text("hello\n")
            _git("add", "feature.txt", cwd=wt_dir)
            agent_msg = (
                "feat: add feature.txt\n"
                "\n"
                "This is the detailed body of the agent's commit "
                "describing exactly what changed and why."
            )
            commit = _git("commit", "-m", agent_msg, cwd=wt_dir)
            assert commit.returncode == 0, commit.stderr

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, branch, baseline
            )
            assert result == MergeResult.SUCCESS

            assert _head_message(repo) == agent_msg

    def test_uses_last_of_multiple_commits(self) -> None:
        """When the agent made multiple commits, the merge commit
        message equals the LAST commit's message (branch HEAD).
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-merge-msg-2"
            wt_dir = _create_worktree(repo, branch)
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            (wt_dir / "a.txt").write_text("a\n")
            _git("add", "a.txt", cwd=wt_dir)
            _git("commit", "-m", "first commit on branch", cwd=wt_dir)

            (wt_dir / "b.txt").write_text("b\n")
            _git("add", "b.txt", cwd=wt_dir)
            last_msg = "second and final commit on branch"
            _git("commit", "-m", last_msg, cwd=wt_dir)

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, branch, baseline
            )
            assert result == MergeResult.SUCCESS
            assert _head_message(repo) == last_msg

    def test_no_squashed_commit_header(self) -> None:
        """The merge commit message must NOT start with
        ``Squashed commit of the following:`` — that is git's default
        ``--no-edit`` output, which the user explicitly does not
        want.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-merge-msg-3"
            wt_dir = _create_worktree(repo, branch)
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            (wt_dir / "c.txt").write_text("c\n")
            _git("add", "c.txt", cwd=wt_dir)
            _git("commit", "-m", "agent: add c.txt", cwd=wt_dir)

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, branch, baseline
            )
            assert result == MergeResult.SUCCESS
            msg = _head_message(repo)
            assert not msg.startswith("Squashed commit of the following")
            assert "kiss: merged from " not in msg


class TestMergeCommitMessageLegacy:
    """squash_merge_branch (legacy path, no baseline) also uses the
    worktree branch's HEAD commit message.
    """

    def test_uses_branch_head_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-merge-msg-legacy"
            wt_dir = _create_worktree(repo, branch)

            (wt_dir / "x.txt").write_text("x\n")
            _git("add", "x.txt", cwd=wt_dir)
            agent_msg = (
                "fix: legacy path message\n"
                "\n"
                "Body explaining the legacy merge."
            )
            _git("commit", "-m", agent_msg, cwd=wt_dir)

            result = GitWorktreeOps.squash_merge_branch(repo, branch)
            assert result == MergeResult.SUCCESS
            assert _head_message(repo) == agent_msg

    def test_no_squashed_commit_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-merge-msg-legacy-2"
            wt_dir = _create_worktree(repo, branch)

            (wt_dir / "y.txt").write_text("y\n")
            _git("add", "y.txt", cwd=wt_dir)
            _git("commit", "-m", "agent: y", cwd=wt_dir)

            result = GitWorktreeOps.squash_merge_branch(repo, branch)
            assert result == MergeResult.SUCCESS
            msg = _head_message(repo)
            assert not msg.startswith("Squashed commit of the following")


class TestMergeCommitMessageHelper:
    """The _merge_commit_message helper falls back to the
    ``kiss: merged from <branch>`` synthetic when the branch HEAD
    cannot be resolved (so ``git commit`` is never invoked with an
    empty message).
    """

    def test_fallback_on_missing_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            msg = GitWorktreeOps._merge_commit_message(repo, "no/such-branch")
            assert msg == "kiss: merged from no/such-branch"

    def test_returns_branch_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-helper"
            wt_dir = _create_worktree(repo, branch)
            (wt_dir / "z.txt").write_text("z\n")
            _git("add", "z.txt", cwd=wt_dir)
            _git("commit", "-m", "helper test commit\n\nbody", cwd=wt_dir)

            msg = GitWorktreeOps._merge_commit_message(repo, branch)
            assert msg == "helper test commit\n\nbody"
