# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A rescued collision sibling must always stay git-ignored.

``GitWorktreeOps.rescue_ignored_files`` only ever lands files that git
IGNORES (that is the whole point of the rescue), so its collision
sibling must remain ignored too.  The old naming appended
``.kiss-rescued-<ns>`` AFTER the extension (``app.log`` became
``app.log.kiss-rescued-<ns>``), which no longer matches ``*.log`` — the
next auto-commit's ``git add -A`` then committed the rescue copy as a
tracked artifact.  That is exactly how the binary
``kiss-sorcar.vsix.kiss-rescued-1788470990563603656`` ended up committed
on this branch despite the repo's ``*.vsix`` ignore rule.

Extension-preserving naming (``app.kiss-rescued-<ns>.log``) fixes
single-suffix rules but is NOT sufficient (gpt-5.6-sol review): a
compound-suffix rule (``*.tar.gz``) stops matching
``model.tar.kiss-rescued-<ns>.gz``, and an exact-name rule (``.env``)
can never match any sibling of the file it names.  The guarantee is
therefore a repo-local ``*.kiss-rescued-*`` pattern that the rescue
appends (idempotently) to the DESTINATION repo's
``<git_common_dir>/info/exclude`` before landing any sibling; the
extension-preserving name is kept because it also helps non-git
tooling and still covers simple suffix rules when the exclusion cannot
be written.

Branch coverage of the modified rescue code:

* candidates present → exclusion installed (compound-suffix,
  exact-name, and ``*.log`` tests); no candidates → destination
  exclude file untouched
  (``test_no_candidates_leaves_exclude_untouched``).
* exclusion written twice → single pattern line
  (``test_exclusion_is_idempotent_across_rescues``).
* exclusion unwritable (``OSError``) → rescue still proceeds; landed
  siblings that are provably still ignored produce no warning
  (``app.log``), unignored ones are warned about (``.env``), and
  non-sibling landings are never checked
  (``test_rescue_survives_unwritable_exclude``).

All tests use real git repositories and real worktrees; no mocks.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import tempfile
from pathlib import Path

import pytest

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.tests.agents.sorcar.test_worktree_ignored_file_rescue import (
    _git,
    _make_repo,
)


class TestRescuedSiblingStaysIgnored:
    """Collision siblings of ignored files must remain ignored."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-rescue-ign-")
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.branch = "kiss/wt-rescue-ign"
        self.wt_dir = self.repo / ".kiss-worktrees" / "kiss_wt-rescue-ign"
        GitWorktreeOps.ensure_excluded(self.repo)
        assert GitWorktreeOps.create(self.repo, self.branch, self.wt_dir)

    def teardown_method(self) -> None:
        info_dir = self.repo / ".git" / "info"
        if info_dir.exists():  # undo the unwritable-exclude test's chmod
            info_dir.chmod(info_dir.stat().st_mode | stat.S_IWUSR)
        GitWorktreeOps.cleanup_partial(self.repo, self.branch, self.wt_dir)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _exclude_file(self) -> Path:
        return self.repo / ".git" / "info" / "exclude"

    def _assert_sibling_ignored(self, sibling: Path) -> None:
        """The sibling must be ignored and invisible to auto-commit."""
        check = _git(self.repo, "check-ignore", "-q", sibling.name)
        assert check.returncode == 0, (
            f"{sibling.name} is not git-ignored: the next "
            "auto-commit's `git add -A` would commit the rescue copy"
        )
        status = _git(self.repo, "status", "--porcelain")
        assert sibling.name not in status.stdout, (
            "the rescue sibling shows up as untracked work"
        )

    def test_sibling_of_extension_ignored_file_stays_ignored(self) -> None:
        """``*.log`` collision sibling must not become auto-committable."""
        (self.repo / "app.log").write_text("user's log\n")
        (self.wt_dir / "app.log").write_text("worktree's log\n")
        rescued, ok = GitWorktreeOps.rescue_ignored_files(
            self.wt_dir, self.repo,
        )
        assert ok is True
        assert rescued == 1
        siblings = [
            p for p in self.repo.iterdir() if ".kiss-rescued-" in p.name
        ]
        assert len(siblings) == 1
        assert siblings[0].read_text() == "worktree's log\n"
        assert siblings[0].suffix == ".log", (
            "the rescue sibling must keep the original extension so "
            "non-git tooling (and simple suffix ignore rules) still "
            "recognize it"
        )
        self._assert_sibling_ignored(siblings[0])

    def test_sibling_of_compound_suffix_file_stays_ignored(self) -> None:
        """``*.tar.gz`` collision sibling must stay ignored.

        ``model.tar.gz`` becomes ``model.tar.kiss-rescued-<ns>.gz``,
        which no longer matches ``*.tar.gz`` — only the installed
        ``*.kiss-rescued-*`` local exclusion keeps it out of the next
        ``git add -A``.
        """
        for root in (self.repo, self.wt_dir):
            gi = root / ".gitignore"
            gi.write_text(gi.read_text() + "*.tar.gz\n")
        (self.repo / "model.tar.gz").write_bytes(b"user's archive")
        (self.wt_dir / "model.tar.gz").write_bytes(b"worktree's archive")
        rescued, ok = GitWorktreeOps.rescue_ignored_files(
            self.wt_dir, self.repo,
        )
        assert ok is True
        assert rescued == 1
        siblings = list(self.repo.glob("model.tar.kiss-rescued-*.gz"))
        assert len(siblings) == 1
        assert siblings[0].read_bytes() == b"worktree's archive"
        self._assert_sibling_ignored(siblings[0])
        assert (
            "*.kiss-rescued-*"
            in self._exclude_file().read_text().splitlines()
        )

    def test_sibling_of_exact_name_ignored_file_stays_ignored(self) -> None:
        """``.env`` (exact-name rule) collision sibling must stay ignored.

        No sibling of ``.env`` can ever match the exact-name ``.env``
        ignore rule, so only the ``*.kiss-rescued-*`` local exclusion
        protects it.  The ``.env.kiss-rescued-<ns>`` name SHAPE is also
        pinned (the pre-existing hardening tests glob for it).
        """
        (self.repo / ".env").write_text("SECRET=users\n")
        (self.wt_dir / ".env").write_text("SECRET=agents\n")
        rescued, ok = GitWorktreeOps.rescue_ignored_files(
            self.wt_dir, self.repo,
        )
        assert ok is True
        assert rescued == 1
        siblings = list(self.repo.glob(".env.kiss-rescued-*"))
        assert len(siblings) == 1
        assert siblings[0].read_text() == "SECRET=agents\n"
        self._assert_sibling_ignored(siblings[0])

    def test_no_candidates_leaves_exclude_untouched(self) -> None:
        """A rescue with nothing to land must not edit info/exclude."""
        before = self._exclude_file().read_text()
        rescued, ok = GitWorktreeOps.rescue_ignored_files(
            self.wt_dir, self.repo,
        )
        assert (rescued, ok) == (0, True)
        assert self._exclude_file().read_text() == before
        assert "*.kiss-rescued-*" not in before

    def test_exclusion_is_idempotent_across_rescues(self) -> None:
        """Repeated rescues append the exclusion pattern exactly once."""
        for name in ("one.log", "two.log"):
            (self.repo / name).write_text("user\n")
            (self.wt_dir / name).write_text("agent\n")
            rescued, ok = GitWorktreeOps.rescue_ignored_files(
                self.wt_dir, self.repo,
            )
            assert ok is True
            assert rescued >= 1
        lines = self._exclude_file().read_text().splitlines()
        assert lines.count("*.kiss-rescued-*") == 1

    def test_rescue_survives_unwritable_exclude(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An unwritable exclude file must not block the rescue.

        The rescue's job is preserving the only copy of task output;
        ignore hygiene is secondary.  With ``info/`` read-only the
        exclusion cannot be installed, so the rescue proceeds and
        instead verifies each landed sibling with ``git check-ignore``:
        the ``app.log`` sibling still matches ``*.log`` (no warning),
        the ``.env`` sibling matches nothing and is warned about, and
        the collision-free ``notes.log`` landing is never checked.
        """
        if os.geteuid() == 0:  # pragma: no cover — CI runs unprivileged
            pytest.skip("root ignores directory write permissions")
        (self.repo / ".env").write_text("SECRET=users\n")
        (self.wt_dir / ".env").write_text("SECRET=agents\n")
        (self.repo / "app.log").write_text("user's log\n")
        (self.wt_dir / "app.log").write_text("worktree's log\n")
        (self.wt_dir / "notes.log").write_text("collision-free\n")
        info_dir = self._exclude_file().parent
        self._exclude_file().unlink()
        info_dir.chmod(0o555)
        try:
            with caplog.at_level(
                logging.WARNING, logger="kiss.agents.sorcar.git_worktree",
            ):
                rescued, ok = GitWorktreeOps.rescue_ignored_files(
                    self.wt_dir, self.repo,
                )
        finally:
            info_dir.chmod(0o755)
        assert ok is True
        assert rescued == 3
        assert (self.repo / "notes.log").read_text() == "collision-free\n"
        install_warnings = [
            r for r in caplog.records
            if "rescue-sibling exclusion" in r.getMessage()
        ]
        assert len(install_warnings) == 1
        unignored = [
            r.getMessage() for r in caplog.records
            if "NOT git-ignored" in r.getMessage()
        ]
        assert len(unignored) == 1
        assert ".env.kiss-rescued-" in unignored[0]
        assert "app.kiss-rescued-" not in unignored[0]


class TestLandRescuedFileReturnPaths:
    """Branch coverage for the landed-path returns of the rescue.

    ``_land_rescued_file`` now returns the landed :class:`Path` (or
    ``None`` when nothing needed landing) so the caller can verify
    siblings when the exclusion could not be written.  Every reachable
    return is pinned here through the real filesystem and real git.

    Two branches are NOT reachable deterministically and stay
    documented instead of covered (per the no-test-doubles rule):

    * the ``FileExistsError`` recovery inside ``_land_rescued_file``
      requires an external writer to create the destination in the
      window between the ``exists()`` check and ``os.link`` — a true
      cross-process race with no deterministic single-process trigger;
    * the exclusive-create copy's mid-write cleanup (``BaseException``
      → unlink → re-raise) requires the write itself to fail (disk
      full / I/O error) after ``os.open`` succeeded.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-rescue-land-")
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.branch = "kiss/wt-rescue-land"
        self.wt_dir = self.repo / ".kiss-worktrees" / "kiss_wt-rescue-land"
        GitWorktreeOps.ensure_excluded(self.repo)
        assert GitWorktreeOps.create(self.repo, self.branch, self.wt_dir)

    def teardown_method(self) -> None:
        sub = self.repo / "sub"
        if sub.exists():  # undo the unwritable-destination test's chmod
            sub.chmod(0o755)
        GitWorktreeOps.cleanup_partial(self.repo, self.branch, self.wt_dir)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_same_path_rescue_is_noop(self) -> None:
        """Rescuing a repo into itself lands nothing and succeeds."""
        assert GitWorktreeOps.rescue_ignored_files(
            self.repo, self.repo,
        ) == (0, True)

    def test_symlink_landings_new_identical_and_collision(self) -> None:
        """Symlink sources: land new, skip identical, sibling differing."""
        os.symlink("fresh-target", self.wt_dir / "fresh.log")
        os.symlink("same-target", self.repo / "same.log")
        os.symlink("same-target", self.wt_dir / "same.log")
        os.symlink("theirs", self.repo / "diff.log")
        os.symlink("ours", self.wt_dir / "diff.log")
        rescued, ok = GitWorktreeOps.rescue_ignored_files(
            self.wt_dir, self.repo,
        )
        assert ok is True
        assert rescued == 2, "new + collision sibling; identical skipped"
        assert os.readlink(self.repo / "fresh.log") == "fresh-target"
        assert os.readlink(self.repo / "diff.log") == "theirs"
        siblings = list(self.repo.glob("diff.kiss-rescued-*.log"))
        assert len(siblings) == 1
        assert os.readlink(siblings[0]) == "ours"
        assert list(self.repo.glob("same.kiss-rescued-*")) == []

    def test_unwritable_destination_dir_fails_closed(self) -> None:
        """A landing ``OSError`` marks the rescue failed (ok=False)."""
        if os.geteuid() == 0:  # pragma: no cover — CI runs unprivileged
            pytest.skip("root ignores directory write permissions")
        (self.wt_dir / "sub").mkdir()
        (self.wt_dir / "sub" / "err.log").write_text("unlandable\n")
        sub = self.repo / "sub"
        sub.mkdir()
        sub.chmod(0o555)
        try:
            rescued, ok = GitWorktreeOps.rescue_ignored_files(
                self.wt_dir, self.repo,
            )
        finally:
            sub.chmod(0o755)
        assert ok is False, "callers must preserve the worktree"
        assert rescued == 0

    def test_non_file_source_is_skipped(self) -> None:
        """A source that is neither file nor symlink lands nothing."""
        src_dir = Path(self.tmpdir) / "adir"
        src_dir.mkdir()
        assert GitWorktreeOps._land_rescued_file(
            src_dir, self.repo / "adir",
        ) is None
        assert not (self.repo / "adir").exists()

    def test_cross_device_copy_fallback_returns_landed_path(self) -> None:
        """``os.link`` EXDEV falls back to the exclusive-create copy."""
        shm = Path("/dev/shm")
        if not shm.is_dir():  # pragma: no cover — Linux always has it
            pytest.skip("/dev/shm not available")
        if shm.stat().st_dev == self.repo.stat().st_dev:
            pytest.skip(  # pragma: no cover — tmpfs vs disk in CI
                "test dirs share a device; cannot force EXDEV"
            )
        src_dir = Path(tempfile.mkdtemp(prefix="kiss-xdev-", dir=shm))
        try:
            src = src_dir / "xdev.log"
            src.write_text("copied across devices\n")
            dst = self.repo / "xdev.log"
            assert GitWorktreeOps._land_rescued_file(src, dst) == dst
            assert dst.read_text() == "copied across devices\n"
            assert dst.stat().st_nlink == 1, "copied, not hard-linked"
        finally:
            shutil.rmtree(src_dir, ignore_errors=True)
