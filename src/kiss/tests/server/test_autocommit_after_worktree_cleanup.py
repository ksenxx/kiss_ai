# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: the Git Commit button resolves a cleaned-up worktree path.

After a worktree task merges or is discarded, cleanup removes its
directory under ``<repo>/.kiss-worktrees/kiss_wt-*``.  A tab whose
working directory still carries that now-stale worktree path may press
Git Commit (``merge_flow._commit_main_tree``).  Passing the path
directly to ``git -C`` would report "Not a git repository" even though
the parent repository has changes waiting to be committed.

When ``work_dir`` points under a removed ``.kiss-worktrees/kiss_wt-*``
segment, the server must strip that segment and act on the
corresponding parent-repository path (``remap_vanished_worktree``).
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kiss.server.server import VSCodeServer
from kiss.tests.server.conftest import init_repo, run_git


class TestManualCommitAfterWorktreeCleanup(unittest.TestCase):
    """The manual Git Commit must recover from a stale worktree path."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-stale-wt-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        init_repo(self.repo)

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

    def test_commit_via_stale_worktree_path_uses_parent_repo(self) -> None:
        """A stale path below ``.kiss-worktrees/kiss_wt-…``
        must commit the dirty files in the parent repo, not
        report "Not a git repository."."""
        before_head = run_git(self.repo, "rev-parse", "HEAD").stdout.strip()

        self.server._commit_main_tree(
            "t-stale", work_dir=self.stale_wt_dir,
        )

        done = self._last_done_event()
        assert done.get("manual") is True, f"Git Commit must be flagged manual: {done}"
        assert done.get("tabId") == "t-stale", f"wrong tab: {done}"
        assert "Not a git repository" not in done.get("message", ""), (
            f"BUG: stale worktree path reported 'Not a git repository': {done}"
        )
        assert done.get("success") is True, f"commit should succeed: {done}"
        assert done.get("committed") is True, f"should have committed: {done}"

        after_head = run_git(self.repo, "rev-parse", "HEAD").stdout.strip()
        assert after_head != before_head, (
            f"HEAD did not advance: before={before_head} after={after_head}"
        )

        status = run_git(self.repo, "status", "--porcelain")
        assert status.stdout.strip() == "", (
            f"working tree still dirty: {status.stdout!r}"
        )


if __name__ == "__main__":
    unittest.main()
