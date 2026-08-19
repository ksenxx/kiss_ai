# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for workflow bugs in worktree and non-worktree modes.

Confirms four bugs, each demonstrated by a test that fails before the fix
and passes after:

BUG 1 — ``is_task_active`` leaks True on early return
    When ``_run_task_inner`` rejects a non-worktree task because a
    worktree merge is in progress on another tab, ``is_task_active``
    is set to True but the early return bypasses the finally block
    that resets it.  The tab is then permanently stuck: merge/discard
    is blocked by ``_check_worktree_busy``.

BUG 2 — ``stash_pop`` loses staging state
    ``stash_pop`` uses plain ``git stash pop`` without ``--index``,
    so user's carefully staged changes lose their staged/unstaged
    distinction after the auto-stash → merge → auto-pop cycle.

BUG 3 — ``_auto_commit_worktree`` crashes when LLM is unavailable
    ``_generate_commit_message`` calls the LLM with no fallback.  If
    the LLM API is unreachable, the exception propagates uncaught,
    preventing worktree finalization and blocking all subsequent tasks
    on that agent.

BUG 4 — ``_close_tab`` orphans pending worktrees
    Closing a tab with a pending worktree drops the in-memory
    reference without auto-merging.  The worktree directory and branch
    persist in git, and ``cleanup_orphans`` skips them because they
    have ``kiss-original`` config.

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _git


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "checkout", "-b", "main"],
        capture_output=True, check=True,
    )
    (path / "init.txt").write_text("init\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        capture_output=True, check=True,
    )
    return path


class TestBug2StashPopLosesStagingState:
    """stash_pop should preserve staged vs unstaged distinction."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stash_pop_preserves_index(self) -> None:
        """After stash → pop, staged modifications should remain staged."""
        repo = self.repo

        (repo / "f.txt").write_text("line1\n")
        (repo / "g.txt").write_text("line1\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", "."],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "two files"],
            capture_output=True, check=True,
        )

        (repo / "f.txt").write_text("line1\nline2\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", "f.txt"],
            capture_output=True, check=True,
        )
        (repo / "g.txt").write_text("line1\nline2\n")

        cached = _git("diff", "--cached", "--name-only", cwd=repo)
        assert "f.txt" in cached.stdout
        unstaged = _git("diff", "--name-only", cwd=repo)
        assert "g.txt" in unstaged.stdout

        did_stash = GitWorktreeOps.stash_if_dirty(repo)
        assert did_stash

        ok = GitWorktreeOps.stash_pop(repo)
        assert ok

        cached_after = _git("diff", "--cached", "--name-only", cwd=repo)
        assert "f.txt" in cached_after.stdout, (
            "BUG 2: stash_pop lost staging state — f.txt is no longer "
            "in the index after stash → pop"
        )
        unstaged_after = _git("diff", "--name-only", cwd=repo)
        assert "g.txt" in unstaged_after.stdout
