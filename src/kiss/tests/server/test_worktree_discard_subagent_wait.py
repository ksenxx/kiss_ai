# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Work-loss audit: the pool guard.

``worktree_pool.discard_all`` must preserve a spare an external writer
put content into (mirroring the reclaim pass), while still removing
clean spares.  Real git repository and worktrees; no mocks.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar import worktree_pool
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.tests.server.test_worktree_ignored_file_rescue import _make_repo


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in *cwd* capturing output."""
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, check=False,
    )


class TestDiscardAllPreservesDirtySpare:
    """worktree_pool.discard_all mirrors the reclaim spare guard."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-pool-guard-")
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        worktree_pool.discard_all()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dirty_spare_preserved_clean_spare_removed(self) -> None:
        """A spare with external content survives discard_all."""
        assert worktree_pool.prewarm(self.repo)
        spare = worktree_pool.take_spare(self.repo)
        assert spare is not None
        branch, wt_dir = spare
        # Put it back so discard_all owns it again.
        worktree_pool._spares[worktree_pool._repo_key(self.repo)] = spare
        (wt_dir / "external-writer.txt").write_text("do not destroy\n")
        worktree_pool.discard_all()
        assert wt_dir.is_dir(), (
            "discard_all destroyed a spare carrying external content"
        )
        assert GitWorktreeOps.branch_exists(self.repo, branch)
        # A clean spare is still removed as before.
        (wt_dir / "external-writer.txt").unlink()
        worktree_pool._spares[worktree_pool._repo_key(self.repo)] = spare
        worktree_pool.discard_all()
        assert not wt_dir.exists()
        assert not GitWorktreeOps.branch_exists(self.repo, branch)
