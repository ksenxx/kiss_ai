# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A commit message longer than one execve argument allows must still commit.

Linux caps a single ``argv`` string at ``MAX_ARG_STRLEN`` (128 KiB).
``git_worktree._ensure_task_metadata`` appends the full user prompt and the
full task result to the merge commit message with no size cap, so a
pasted log is enough to exceed it.  ``git commit -m <message>`` then
fails in ``Popen`` with ``OSError(E2BIG)`` before git starts — and on
the merge path that error escaped mid-merge, leaving the main tree
with a staged squash.  The message now travels on stdin (``-F -``).

Real git repositories on disk, no mocks.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps, MergeResult, _git

HUGE_BODY = "x" * 140_000


def _make_repo(path: Path) -> Path:
    """Create a git repo with one initial commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "T"], check=True)
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "initial"], check=True)
    return path


def _head_message(repo: Path) -> str:
    """Return the full commit message of HEAD in *repo*."""
    return _git("log", "-1", "--format=%B", "HEAD", cwd=repo).stdout.rstrip()


def test_commit_staged_accepts_a_140kb_message() -> None:
    """``commit_staged`` commits a message far above ``MAX_ARG_STRLEN``."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(Path(tmp) / "repo")
        (repo / "f.txt").write_text("hello\n")
        GitWorktreeOps.stage_all(repo)
        message = "subject\n\nUser prompt:\n" + HUGE_BODY
        assert GitWorktreeOps.commit_staged(repo, message) is True
        head = _head_message(repo)
        assert head.startswith("subject\n\nUser prompt:\n")
        assert HUGE_BODY in head
        assert not GitWorktreeOps.has_staged_changes(repo)


def test_commit_staged_with_nul_byte_fails_cleanly() -> None:
    """A NUL in the message is a failed commit (``False``), not a ``ValueError``."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(Path(tmp) / "repo")
        (repo / "f.txt").write_text("hello\n")
        GitWorktreeOps.stage_all(repo)
        assert GitWorktreeOps.commit_staged(repo, "subject\n\nbad\0byte") is False
        assert _head_message(repo) == "initial"


def test_squash_merge_with_140kb_prompt_commits_and_leaves_tree_clean() -> None:
    """The merge path commits a huge trailer instead of raising mid-merge."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(Path(tmp) / "repo")
        wt_dir = repo / ".kiss-worktrees" / "wt"
        assert GitWorktreeOps.create(repo, "kiss/wt", wt_dir)
        subprocess.run(["git", "-C", str(wt_dir), "config", "user.email", "t@t.com"], check=True)
        subprocess.run(["git", "-C", str(wt_dir), "config", "user.name", "T"], check=True)
        (wt_dir / "new.txt").write_text("from worktree\n")
        assert GitWorktreeOps.commit_all(wt_dir, "worktree change")

        result = GitWorktreeOps.squash_merge_branch(
            repo, "kiss/wt", user_prompt=HUGE_BODY, task_result="done",
        )
        assert result == MergeResult.SUCCESS
        assert (repo / "new.txt").read_text() == "from worktree\n"
        assert HUGE_BODY in _head_message(repo)
        # Nothing staged or modified is left behind (the worktree
        # directory itself is untracked, as in any repo).
        status = _git("status", "--porcelain", "--untracked-files=no", cwd=repo).stdout
        assert status == "", status
