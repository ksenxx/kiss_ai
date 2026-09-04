# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""One spare-content probe shared by every path that may destroy a spare.

Redundancy found by the 2026-09-02 sorcar-infra audit: the four-part
"does this pool spare hold foreign content?" test (dirty status,
ignored files present, ignored files unenumerable, commits unique to
the branch) was copied verbatim into ``worktree_pool.take_spare``,
``worktree_pool.discard_all`` and the spare branch of
``GitWorktreeOps.reclaim_orphaned_worktrees``.  Three copies of a
safety predicate drift; it now lives once in
:meth:`GitWorktreeOps.spare_has_content`.  These tests pin the shared
predicate and prove the three consumers agree on every case, using
real git repositories only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from kiss.agents.sorcar import worktree_pool
from kiss.agents.sorcar.git_worktree import GitWorktreeOps


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
    ).stdout


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(path)], capture_output=True, check=True)
    _git(path, "config", "user.email", "t@t.com")
    _git(path, "config", "user.name", "T")
    (path / "README.md").write_text("# repo\n")
    (path / ".gitignore").write_text("*.log\n")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "initial")
    return path


class TestSpareHasContent:
    def setup_method(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmp) / "repo")
        worktree_pool.discard_all()
        assert worktree_pool.prewarm(self.repo)
        self.branch, self.wt_dir = worktree_pool._spares[
            worktree_pool._repo_key(self.repo)
        ]

    def teardown_method(self) -> None:
        worktree_pool.discard_all()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _all_consumers_preserve(self) -> None:
        """take_spare, discard_all and reclaim must all refuse to destroy."""
        assert worktree_pool.take_spare(self.repo) is None
        assert self.wt_dir.is_dir()
        # Re-pool the same spare so discard_all sees it too.
        worktree_pool._spares[worktree_pool._repo_key(self.repo)] = (
            self.branch, self.wt_dir,
        )
        worktree_pool.discard_all()
        assert self.wt_dir.is_dir()
        assert GitWorktreeOps.reclaim_orphaned_worktrees(self.repo) == 0
        assert self.wt_dir.is_dir()
        assert GitWorktreeOps.branch_exists(self.repo, self.branch)

    def test_fresh_spare_is_contentless_and_consumed(self) -> None:
        assert not GitWorktreeOps.spare_has_content(
            self.repo, self.branch, self.wt_dir,
        )
        assert worktree_pool.take_spare(self.repo) == (self.branch, self.wt_dir)

    def test_fresh_spare_is_discarded_by_discard_all(self) -> None:
        worktree_pool.discard_all()
        assert not self.wt_dir.exists()
        assert not GitWorktreeOps.branch_exists(self.repo, self.branch)

    def test_untracked_file_is_content(self) -> None:
        (self.wt_dir / "stray.txt").write_text("external\n")
        assert GitWorktreeOps.spare_has_content(self.repo, self.branch, self.wt_dir)
        self._all_consumers_preserve()

    def test_ignored_file_is_content(self) -> None:
        (self.wt_dir / "build.log").write_text("external\n")
        assert not GitWorktreeOps.has_uncommitted_changes(self.wt_dir)
        assert GitWorktreeOps.spare_has_content(self.repo, self.branch, self.wt_dir)
        self._all_consumers_preserve()

    def test_unique_commit_is_content(self) -> None:
        (self.wt_dir / "work.txt").write_text("committed\n")
        _git(self.wt_dir, "add", ".")
        _git(self.wt_dir, "commit", "-m", "external commit")
        assert GitWorktreeOps.spare_has_content(self.repo, self.branch, self.wt_dir)
        self._all_consumers_preserve()

    def test_unenumerable_ignored_files_are_content(self) -> None:
        """A worktree git can no longer inspect (directory unreadable)
        must be treated as holding content, never destroyed."""
        if os.geteuid() == 0:  # pragma: no cover — root ignores mode bits
            pytest.skip("permission-based git failure needs a non-root user")
        os.chmod(self.wt_dir, 0)
        try:
            assert GitWorktreeOps.list_ignored_files(self.wt_dir) is None
            assert GitWorktreeOps.spare_has_content(
                self.repo, self.branch, self.wt_dir,
            )
            assert worktree_pool.take_spare(self.repo) is None
        finally:
            os.chmod(self.wt_dir, 0o755)
        assert self.wt_dir.is_dir()
