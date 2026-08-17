# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: auto-commit resolves a cleaned-up worktree path.

After a worktree task finishes, cleanup removes its directory under
``<repo>/.kiss-worktrees/kiss_wt-*``.  A delayed automatic commit may
still carry that now-stale worktree path.  Passing it directly to
``git -C`` would report "Not a git repository" even though the parent
repository has changes waiting to be committed.

When ``work_dir`` points under a removed ``.kiss-worktrees/kiss_wt-*``
segment (or one without a valid ``.git`` link), the server must strip
that segment and act on the corresponding parent-repository path via
``_stale_worktree_fallback`` in ``useful_tools.py``.
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


class TestAutoCommitAfterWorktreeCleanup(unittest.TestCase):
    """Automatic commit must recover from a stale worktree path."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-stale-wt-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        _init_repo(self.repo)

        self.stale_wt_dir = str(
            Path(self.repo) / ".kiss-worktrees" / "kiss_wt-1781574606-49147541",
        )

        self.server = VSCodeServer()
        self.server.work_dir = self.repo
        self.events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

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

    def test_autocommit_via_stale_worktree_path_uses_parent_repo(self) -> None:
        """A stale path below ``.kiss-worktrees/kiss_wt-…``
        must commit the dirty files in the parent repo, not
        report "Not a git repository."."""
        before_head = _run_git(self.repo, "rev-parse", "HEAD").stdout.strip()

        self.server._autocommit_changes(
            "t-stale", work_dir=self.stale_wt_dir,
        )

        done = self._last_done_event()
        assert "Not a git repository" not in done.get("message", ""), (
            f"BUG: stale worktree path reported 'Not a git repository': {done}"
        )
        assert done.get("success") is True, f"commit should succeed: {done}"
        assert done.get("committed") is True, f"should have committed: {done}"

        after_head = _run_git(self.repo, "rev-parse", "HEAD").stdout.strip()
        assert after_head != before_head, (
            f"HEAD did not advance: before={before_head} after={after_head}"
        )

        status = _run_git(self.repo, "status", "--porcelain")
        assert status.stdout.strip() == "", (
            f"working tree still dirty: {status.stdout!r}"
        )


if __name__ == "__main__":
    unittest.main()
