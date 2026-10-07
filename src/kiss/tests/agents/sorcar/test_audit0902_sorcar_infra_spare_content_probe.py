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
from kiss.tests.conftest import is_root, posix_only


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
    (path / "src").mkdir()
    (path / "src" / "module.py").write_text("x = 1\n")
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

    def _interrupt_checkout(self) -> None:
        """Leave the spare as a killed ``reset --hard`` would: files partially
        written, no index in the worktree's gitdir."""
        index = Path(_git(self.wt_dir, "rev-parse", "--git-path", "index").strip())
        assert index.is_file()
        index.unlink()
        (self.wt_dir / "README.md").unlink()
        status = _git(self.wt_dir, "status", "--porcelain")
        assert "D  README.md" in status and "?? .gitignore" in status

    def test_interrupted_checkout_is_contentless_and_reclaimed(self) -> None:
        """A spare whose populating checkout was killed mid-way has no
        index: git reports every tracked file as a staged deletion and
        the written files as untracked.  That is checkout debris, not
        external content, so the orphan reclaim pass discards it
        instead of preserving it forever."""
        self._interrupt_checkout()
        assert not GitWorktreeOps.spare_has_content(
            self.repo, self.branch, self.wt_dir,
        )
        with worktree_pool._pool_lock:
            worktree_pool._spares.clear()
        head_before = _git(self.repo, "rev-parse", "HEAD")
        assert GitWorktreeOps.reclaim_orphaned_worktrees(self.repo) == 1
        assert not self.wt_dir.exists()
        assert not GitWorktreeOps.branch_exists(self.repo, self.branch)
        assert _git(self.repo, "rev-parse", "HEAD") == head_before
        assert (self.repo / "README.md").read_text() == "# repo\n"
        assert _git(self.repo, "status", "--porcelain") == ""

    def test_interrupted_checkout_with_unique_commit_is_content(self) -> None:
        """A missing index excuses the status output only; a commit that
        exists on no other ref is still content and is preserved."""
        (self.wt_dir / "work.txt").write_text("committed\n")
        _git(self.wt_dir, "add", ".")
        _git(self.wt_dir, "commit", "-m", "external commit")
        self._interrupt_checkout()
        assert GitWorktreeOps.spare_has_content(self.repo, self.branch, self.wt_dir)
        self._all_consumers_preserve()

    def test_interrupted_checkout_with_external_untracked_file_is_content(self) -> None:
        """A file outside the HEAD tree in an index-less spare was written
        externally, and the plain status walk cannot tell it from the
        half-written checkout; the probe still preserves it."""
        self._interrupt_checkout()
        (self.wt_dir / "rescue.txt").write_text("external\n")
        assert GitWorktreeOps.spare_has_content(self.repo, self.branch, self.wt_dir)
        self._all_consumers_preserve()
        assert (self.wt_dir / "rescue.txt").exists()

    def test_interrupted_checkout_with_external_ignored_file_is_content(self) -> None:
        self._interrupt_checkout()
        (self.wt_dir / "build.log").write_text("external\n")
        assert GitWorktreeOps.spare_has_content(self.repo, self.branch, self.wt_dir)
        self._all_consumers_preserve()

    def test_interrupted_checkout_external_file_inside_tracked_dir_is_content(self) -> None:
        """With an empty index git's default status collapses ``src/``
        (every file in it untracked) into one line, hiding an external
        file next to the checked-out ``src/module.py``."""
        self._interrupt_checkout()
        assert "?? src/\n" in _git(self.wt_dir, "status", "--porcelain")
        (self.wt_dir / "src" / "extra.py").write_text("external\n")
        assert GitWorktreeOps.spare_has_content(self.repo, self.branch, self.wt_dir)
        self._all_consumers_preserve()

    def test_interrupted_checkout_with_edited_tracked_file_is_content(self) -> None:
        """An externally edited tracked file in an index-less spare is
        listed exactly like a checked-out one (``D `` plus ``??`` at the
        same path); only a content comparison against HEAD tells them
        apart, and it must preserve the edit."""
        self._interrupt_checkout()
        (self.wt_dir / "src" / "module.py").write_text("x = 2  # external edit\n")
        assert GitWorktreeOps.spare_has_content(self.repo, self.branch, self.wt_dir)
        self._all_consumers_preserve()
        assert (self.wt_dir / "src" / "module.py").read_text() == "x = 2  # external edit\n"

    def test_interrupted_checkout_dir_replacing_tracked_file_is_content(self) -> None:
        """A directory written where HEAD has a file reads as ``" D"`` for
        that file; the files inside it must still be enumerated."""
        self._interrupt_checkout()
        (self.wt_dir / "src" / "module.py").unlink()
        (self.wt_dir / "src" / "module.py").mkdir()
        (self.wt_dir / "src" / "module.py" / "extra.txt").write_text("external\n")
        assert GitWorktreeOps.spare_has_content(self.repo, self.branch, self.wt_dir)
        self._all_consumers_preserve()

    def test_interrupted_checkout_probe_leaves_spare_gitdir_index_less(self) -> None:
        """The scratch index must not be written into the spare's gitdir
        (``core.splitIndex`` on would otherwise drop a ``sharedindex.*``
        there) and the probe must leave no temporary directory behind."""
        _git(self.repo, "config", "core.splitIndex", "true")
        self._interrupt_checkout()
        gitdir = Path(_git(self.wt_dir, "rev-parse", "--git-dir").strip())
        before = sorted(p.name for p in gitdir.iterdir())
        tmp_before = {p for p in Path(tempfile.gettempdir()).glob("kiss-spare-index-*")}
        assert not GitWorktreeOps.spare_has_content(
            self.repo, self.branch, self.wt_dir,
        )
        assert sorted(p.name for p in gitdir.iterdir()) == before
        assert not (gitdir / "index").exists()
        assert {p for p in Path(tempfile.gettempdir()).glob("kiss-spare-index-*")} == tmp_before

    @posix_only("NTFS rejects control characters such as \\r in file names")
    def test_interrupted_checkout_with_newline_filename_is_content(self) -> None:
        """A byte-distinct external filename (``\\n`` where HEAD has ``\\r``)
        must not be folded onto the tracked name by text-mode decoding."""
        cr_name = self.repo / "known\rname.txt"
        cr_name.write_text("tracked\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-m", "cr filename")
        _git(self.wt_dir, "reset", "--hard", "main")
        self._interrupt_checkout()
        (self.wt_dir / "known\nname.txt").write_text("external\n")
        assert GitWorktreeOps.spare_has_content(self.repo, self.branch, self.wt_dir)
        self._all_consumers_preserve()

    def test_interrupted_checkout_with_unreadable_head_is_content(self) -> None:
        """When git cannot even list the HEAD tree of an index-less spare
        (corrupt gitdir), nothing can be classified as debris: preserve."""
        self._interrupt_checkout()
        head = Path(_git(self.wt_dir, "rev-parse", "--git-path", "HEAD").strip())
        head.write_text("0" * 40 + "\n")
        assert GitWorktreeOps.spare_has_content(self.repo, self.branch, self.wt_dir)
        with worktree_pool._pool_lock:
            worktree_pool._spares.clear()
        assert GitWorktreeOps.reclaim_orphaned_worktrees(self.repo) == 0
        assert self.wt_dir.is_dir()
        head.write_text(f"ref: refs/heads/{self.branch}\n")

    @posix_only("chmod 000 permission denial")
    def test_unenumerable_ignored_files_are_content(self) -> None:
        """A worktree git can no longer inspect (directory unreadable)
        must be treated as holding content, never destroyed."""
        if is_root():  # pragma: no cover — root ignores mode bits
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
