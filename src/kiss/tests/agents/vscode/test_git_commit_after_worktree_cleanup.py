# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: settings-panel "Git Commit" works after worktree cleanup.

Bug reported by the user:

    "in the last task when I pressed 'git commit' it says
    'Not a git repository.'"

Repro:

1.  A worktree task runs and finishes (merge or discard).  The
    worktree directory under ``<repo>/.kiss-worktrees/kiss_wt-*`` is
    removed by ``git worktree remove`` as part of cleanup.
2.  The frontend tab's ``workDir`` was captured from the agent's
    ``extra.work_dir`` (the worktree path) during the task and is
    NOT reset after worktree cleanup.
3.  The user presses the settings-panel "Git Commit" button.  The
    frontend posts ``autocommitAction`` with that now-stale
    ``workDir`` (the deleted ``.kiss-worktrees/kiss_wt-…`` path).
4.  ``_handle_autocommit_action`` runs ``git -C <stale>`` which
    fails (the directory no longer exists) and the user sees a
    misleading "Not a git repository." error even though their
    main working tree IS a git repo with uncommitted changes
    waiting to be committed.

Fix: when ``work_dir`` points under a ``.kiss-worktrees/kiss_wt-*``
segment that no longer exists (or contains no ``.git`` link), strip
that segment and act on the equivalent path inside the parent repo
(``_stale_worktree_fallback`` in ``useful_tools.py``).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kiss.server.server import VSCodeServer


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    _run_git(repo, "add", "seed.txt")
    _run_git(repo, "commit", "-q", "-m", "seed")


class TestGitCommitAfterWorktreeCleanup(unittest.TestCase):
    """Settings-panel "Git Commit" must work with a stale worktree path."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-stale-wt-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        _init_repo(self.repo)

        # Simulate a removed worktree directory.  The stale path lives
        # under ``<repo>/.kiss-worktrees/kiss_wt-…`` and does NOT exist
        # on disk — exactly what the frontend captures during a task
        # whose worktree was cleaned up after the task ended.
        self.stale_wt_dir = str(
            Path(self.repo) / ".kiss-worktrees" / "kiss_wt-1781574606-49147541",
        )

        self.server = VSCodeServer()
        self.server.work_dir = self.repo
        self.events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        # Dirty the main working tree so a real commit is possible.
        Path(self.repo, "edited.txt").write_text("dirty content\n")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _last_done_event(self) -> dict[str, Any]:
        for ev in reversed(self.events):
            if ev.get("type") == "autocommit_done":
                return ev
        raise AssertionError(
            f"No autocommit_done event captured. Events: {self.events}",
        )

    def test_commit_via_stale_worktree_path_commits_in_parent_repo(self) -> None:
        """Settings-panel ``commit`` action with a stale ``.kiss-worktrees/kiss_wt-…``
        ``workDir`` must commit the dirty files in the parent repo, not
        report "Not a git repository."."""
        before_head = _run_git(self.repo, "rev-parse", "HEAD").stdout.strip()

        self.server._handle_autocommit_action(
            "commit", tab_id="t-stale", work_dir=self.stale_wt_dir,
        )

        done = self._last_done_event()
        assert "Not a git repository" not in done.get("message", ""), (
            f"BUG: stale worktree path reported 'Not a git repository': {done}"
        )
        assert done.get("success") is True, f"commit should succeed: {done}"
        assert done.get("committed") is True, f"should have committed: {done}"

        # HEAD must have advanced in the parent repo.
        after_head = _run_git(self.repo, "rev-parse", "HEAD").stdout.strip()
        assert after_head != before_head, (
            f"HEAD did not advance: before={before_head} after={after_head}"
        )

        # Working tree must now be clean.
        status = _run_git(self.repo, "status", "--porcelain")
        assert status.stdout.strip() == "", (
            f"working tree still dirty: {status.stdout!r}"
        )


if __name__ == "__main__":
    unittest.main()
