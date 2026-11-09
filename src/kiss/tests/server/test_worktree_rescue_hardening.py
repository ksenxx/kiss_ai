# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Hardening tests for the ignored-file rescue and reclaim owner guard.

Covers the defects the gpt-5.6-sol review confirmed in the first
rescue implementation:

* a symlinked destination ancestor must not carry the rescue outside
  the repository (containment), and such a failure must PRESERVE the
  worktree (fail closed) instead of letting teardown destroy the only
  copy;
* a destination that exists with different content must not silently
  drop the worktree's version — it lands under a
  ``<name>.kiss-rescued-<ns>`` sibling;
* an identical existing destination is already safe (no duplicate);
* a deferred discard keeps the worktree pending (its directory and
  the unrescued output survive);
* ``reclaim_orphaned_worktrees`` skips a worktree whose recorded
  owner process is still alive (cross-process live-worktree adoption)
  and still reclaims one whose owner is dead.

The swap tests run against both landing backends the module constant
``_RESCUE_DIR_FD`` selects: the descriptor-pinned landing (POSIX only:
it needs ``O_DIRECTORY``, ``O_NOFOLLOW`` and ``dir_fd``) and the
pathname landing of the platforms without them (native Windows), which
checks the containment again after the landing and removes only a copy
it can prove is its own.

All tests use real git repositories, real worktrees, real processes;
no mocks.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import git_worktree
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import _RESCUE_SIBLING_MARKER, GitWorktreeOps
from kiss.tests.conftest import posix_only
from kiss.tests.server.test_worktree_ignored_file_rescue import (
    _make_repo,
    _redirect_db,
    _restore_db,
)

_BACKENDS = pytest.mark.parametrize(
    "dir_fd",
    [
        pytest.param(True, marks=posix_only("the pinned landing needs O_DIRECTORY and dir_fd")),
        False,
    ],
    ids=["dir_fd", "no_dir_fd"],
)


@contextlib.contextmanager
def _rescue_backend(dir_fd: bool) -> Iterator[None]:
    """Select the rescue's landing backend through its module constant.

    True is the descriptor-pinned landing, False the pathname landing of
    the platforms without descriptor-relative file operations; every
    filesystem call stays real, only the capability selector changes.
    """
    saved = git_worktree._RESCUE_DIR_FD
    git_worktree._RESCUE_DIR_FD = dir_fd
    try:
        yield
    finally:
        git_worktree._RESCUE_DIR_FD = saved


class TestRescueHardening:
    """Containment, atomicity, and collision behavior of the rescue."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-rescue-hard-")
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.outside = Path(self.tmpdir) / "outside"
        self.outside.mkdir()
        self.branch = "kiss/wt-hardening"
        self.wt_dir = self.repo / ".kiss-worktrees" / "kiss_wt-hardening"
        GitWorktreeOps.ensure_excluded(self.repo)
        assert GitWorktreeOps.create(self.repo, self.branch, self.wt_dir)

    def teardown_method(self) -> None:
        GitWorktreeOps.cleanup_partial(self.repo, self.branch, self.wt_dir)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @_BACKENDS
    def test_symlinked_ancestor_never_escapes_repo(self, dir_fd: bool) -> None:
        """A main-tree dir symlink pointing outside blocks the rescue."""
        os.symlink(self.outside, self.repo / "data")
        (self.wt_dir / "data").mkdir()
        (self.wt_dir / "data" / "out.csv").write_text("secret\n")
        with _rescue_backend(dir_fd):
            rescued, ok = GitWorktreeOps.rescue_ignored_files(
                self.wt_dir, self.repo,
            )
        assert not (self.outside / "out.csv").exists(), (
            "rescue escaped the repository through a symlinked ancestor"
        )
        assert rescued == 0
        assert ok is False, (
            "a rescue that could not land a file must fail closed"
        )

    @_BACKENDS
    def test_ancestor_swapped_after_containment_check_never_escapes_repo(
        self, dir_fd: bool,
    ) -> None:
        """An ancestor replaced by an outside symlink BETWEEN the
        containment check and the copy must not carry the write outside
        the repository (audit A09: the check and the write were separate
        pathname operations).  The rescuing thread is held at the exact
        return of ``_rescue_dst_contained`` by a profile hook; the main
        thread then swaps ``repo/data`` for the symlink and releases it.
        Without descriptor-relative operations the pathname landing follows
        the swapped ancestor (review 2026-11-08 Q3/F4: it once reported that
        as ``(1, True)``); it now creates and checks the ancestors one at a
        time and refuses the swapped-in link before writing anything, so
        nothing reaches the outside and the rescue fails closed."""
        (self.wt_dir / "data").mkdir()
        (self.wt_dir / "data" / "out.csv").write_text("valuable output\n")
        (self.repo / "data").mkdir()
        checked = threading.Event()
        swapped = threading.Event()
        contained_code = GitWorktreeOps._rescue_dst_contained.__code__

        def hold_after_check(frame: Any, event: str, arg: Any) -> None:
            if event == "return" and frame.f_code is contained_code and not checked.is_set():
                checked.set()
                swapped.wait(30)

        result: list[tuple[int, bool]] = []
        threading.setprofile(hold_after_check)
        try:
            with _rescue_backend(dir_fd):
                worker = threading.Thread(
                    target=lambda: result.append(
                        GitWorktreeOps.rescue_ignored_files(self.wt_dir, self.repo),
                    ),
                )
                worker.start()
                assert checked.wait(30), "the rescue never ran its containment check"
                (self.repo / "data").rmdir()
                os.symlink(self.outside, self.repo / "data")
                swapped.set()
                worker.join(60)
        finally:
            threading.setprofile(None)
            swapped.set()
        assert not worker.is_alive()
        assert not (self.outside / "out.csv").exists(), (
            "rescue escaped the repository through an ancestor swapped "
            "after the containment check"
        )
        assert list(self.outside.iterdir()) == [], "nothing is written outside the repository"
        assert result == [(0, False)], "an escaped rescue must fail closed"
        assert (self.wt_dir / "data" / "out.csv").read_text() == "valuable output\n"

    @_BACKENDS
    def test_ancestor_swapped_after_landing_never_touches_an_outside_file(
        self, dir_fd: bool,
    ) -> None:
        """An ancestor swapped for an outside symlink AFTER the copy landed
        must not make the rescue unlink the pathname it landed at (audit
        S4/F1): that name now resolves to an unrelated file outside the
        repository, while the landed copy moved with the renamed directory
        and is still inside it.  The rescuing thread is held at the return
        of ``_land_in_dir`` (the landing proper, before the pathname
        backend's check of where it landed) by a profile hook; the main
        thread renames ``repo/data`` to ``repo/saved-data``, points
        ``repo/data`` at the outside directory holding the victim, and
        releases it.  The pathname landing then finds the victim at its
        name, not the copy it created: it touches nothing and fails closed,
        so the worktree keeps the only sure copy while the landed one stays
        inside the renamed directory."""
        (self.wt_dir / "data").mkdir()
        (self.wt_dir / "data" / "out.csv").write_text("AGENT OUTPUT\n")
        (self.repo / "data").mkdir()
        victim = self.outside / "out.csv"
        victim.write_text("UNRELATED USER FILE\n")
        landed = threading.Event()
        swapped = threading.Event()
        land_code = GitWorktreeOps._land_in_dir.__code__

        def hold_after_landing(frame: Any, event: str, arg: Any) -> None:
            if event == "return" and frame.f_code is land_code and not landed.is_set():
                landed.set()
                swapped.wait(30)

        result: list[tuple[int, bool]] = []
        threading.setprofile(hold_after_landing)
        try:
            with _rescue_backend(dir_fd):
                worker = threading.Thread(
                    target=lambda: result.append(
                        GitWorktreeOps.rescue_ignored_files(self.wt_dir, self.repo),
                    ),
                )
                worker.start()
                assert landed.wait(30), "the rescue never ran its landing"
                (self.repo / "data").rename(self.repo / "saved-data")
                os.symlink(self.outside, self.repo / "data")
                swapped.set()
                worker.join(60)
        finally:
            threading.setprofile(None)
            swapped.set()
        assert not worker.is_alive()
        assert victim.read_text() == "UNRELATED USER FILE\n", (
            "the rescue unlinked a file it never created"
        )
        assert list(self.outside.iterdir()) == [victim], "nothing else reached the outside"
        saved_copy = self.repo / "saved-data" / "out.csv"
        assert saved_copy.read_text() == "AGENT OUTPUT\n", (
            "the landed copy went with the renamed directory and stays inside the repo"
        )
        if dir_fd:
            assert result == [(1, True)], "a copy landed inside the repository is safe"
        else:
            assert result == [(0, False)], (
                "a pathname landing whose name no longer names its copy must fail closed"
            )
        assert (self.wt_dir / "data" / "out.csv").read_text() == "AGENT OUTPUT\n"

    @_BACKENDS
    def test_an_alias_of_the_worktree_is_not_an_identical_copy(self, dir_fd: bool) -> None:
        """A main-tree directory symlinked into the worktree being rescued is refused.

        Through ``repo/data -> <worktree>/data`` the destination is the
        source itself, so the pathname landing once found it "identical"
        and reported ``(0, True)``: removing the worktree then destroyed the
        only copy (review 2026-11-08 finding 1)."""
        (self.wt_dir / "data").mkdir()
        (self.wt_dir / "data" / "out.csv").write_text("only copy\n")
        os.symlink(self.wt_dir / "data", self.repo / "data")
        with _rescue_backend(dir_fd):
            result = GitWorktreeOps.rescue_ignored_files(self.wt_dir, self.repo)
        assert result == (0, False), "the rescue must fail closed, keeping the worktree"
        assert (self.wt_dir / "data" / "out.csv").read_text() == "only copy\n"

    def _rescue_swapping_data_at(self, code: Any, target: Path) -> list[tuple[int, bool]]:
        """Run a pathname-backend rescue of ``data/out.csv`` with ``repo/data`` swapped.

        The rescuing thread is held at the first return of *code*; then
        ``repo/data`` is renamed to ``repo/saved-data`` and replaced by a
        symlink to *target*, and the rescue goes on.

        Returns:
            The rescue's ``(rescued, ok)`` result, in a list.
        """
        (self.wt_dir / "data").mkdir()
        (self.wt_dir / "data" / "out.csv").write_text("AGENT OUTPUT\n")
        (self.repo / "data").mkdir()
        held = threading.Event()
        swapped = threading.Event()

        def hold(frame: Any, event: str, arg: Any) -> None:
            if event == "return" and frame.f_code is code and not held.is_set():
                held.set()
                swapped.wait(30)

        result: list[tuple[int, bool]] = []
        threading.setprofile(hold)
        try:
            with _rescue_backend(False):
                worker = threading.Thread(
                    target=lambda: result.append(
                        GitWorktreeOps.rescue_ignored_files(self.wt_dir, self.repo),
                    ),
                )
                worker.start()
                assert held.wait(30), "the rescue never reached the held call"
                (self.repo / "data").rename(self.repo / "saved-data")
                os.symlink(target, self.repo / "data")
                swapped.set()
                worker.join(60)
        finally:
            threading.setprofile(None)
            swapped.set()
        assert not worker.is_alive()
        assert (self.wt_dir / "data" / "out.csv").read_text() == "AGENT OUTPUT\n"
        return result

    def test_a_swap_after_the_ownership_check_unlinks_nothing_outside(self) -> None:
        """An ancestor swapped after the landed copy was proven the rescue's own.

        The pathname backend then found the name outside the repository
        and unlinked it by pathname, deleting the unrelated outside file
        the swapped-in symlink now led to (review 2026-11-08 finding 2)."""
        victim = self.outside / "out.csv"
        victim.write_text("UNRELATED USER FILE\n")
        result = self._rescue_swapping_data_at(os.path.samestat.__code__, self.outside)
        assert victim.read_text() == "UNRELATED USER FILE\n", (
            "the rescue unlinked a file it never created"
        )
        assert (self.repo / "saved-data" / "out.csv").read_text() == "AGENT OUTPUT\n"
        assert result == [(0, False)]

    def test_a_swap_before_the_link_writes_nothing_outside(self) -> None:
        """An ancestor swapped after the temporary copy was written, before its link.

        The link then failed with ``ENOENT``, which the landing took for a
        filesystem without hard links: its fallback copy went outside the
        repository, and the failing unlink of the temporary file skipped the
        containment check (review 2026-11-08 finding 3)."""
        result = self._rescue_swapping_data_at(git_worktree._copy_to_fd.__code__, self.outside)
        assert list(self.outside.iterdir()) == [], "nothing is written outside the repository"
        assert result == [(0, False)]

    def test_partial_rescue_lands_independent_inodes(self) -> None:
        """A file landed before the rescue fails is a copy, not an alias.

        The worktree is preserved when one file cannot be landed, so a
        hard link from main to the worktree's file would make the two
        trees share one writable inode: a later edit on either side
        would silently appear on the other.
        """
        os.symlink(self.outside, self.repo / "data")
        (self.wt_dir / "data").mkdir()
        (self.wt_dir / "data" / "out.csv").write_text("blocked\n")
        (self.wt_dir / "run.log").write_text("task output\n")
        rescued, ok = GitWorktreeOps.rescue_ignored_files(
            self.wt_dir, self.repo,
        )
        assert (rescued, ok) == (1, False)
        landed = self.repo / "run.log"
        kept = self.wt_dir / "run.log"
        assert landed.read_text() == "task output\n"
        assert not os.path.samefile(landed, kept), (
            "the preserved worktree and the main tree share one inode"
        )
        with open(kept, "a") as fh:
            fh.write("resumed task edit\n")
        assert landed.read_text() == "task output\n"

    @_BACKENDS
    def test_collision_lands_kiss_rescued_sibling(self, dir_fd: bool) -> None:
        """Different existing content is preserved beside, not dropped.

        A destination that is a symlink (here ``data/out.csv`` pointing
        outside the repository) counts as "taken" too: the worktree's
        regular file lands beside it and the link is neither followed
        nor replaced."""
        (self.repo / ".env").write_text("SECRET=users\n")
        (self.wt_dir / ".env").write_text("SECRET=agents\n")
        (self.repo / "data").mkdir()
        os.symlink(self.outside / "out.csv", self.repo / "data" / "out.csv")
        (self.wt_dir / "data").mkdir()
        (self.wt_dir / "data" / "out.csv").write_text("AGENT OUTPUT\n")
        with _rescue_backend(dir_fd):
            rescued, ok = GitWorktreeOps.rescue_ignored_files(
                self.wt_dir, self.repo,
            )
        assert ok is True
        assert rescued == 2
        assert (self.repo / ".env").read_text() == "SECRET=users\n"
        siblings = list(self.repo.glob(".env.kiss-rescued-*"))
        assert len(siblings) == 1, (
            "the worktree's differing version must land beside the "
            "user's file"
        )
        assert siblings[0].read_text() == "SECRET=agents\n"
        assert (self.repo / "data" / "out.csv").is_symlink()
        assert not (self.outside / "out.csv").exists(), (
            "the rescue wrote through the symlink at the destination"
        )
        csv_siblings = list((self.repo / "data").glob("out.kiss-rescued-*.csv"))
        assert len(csv_siblings) == 1
        assert csv_siblings[0].read_text() == "AGENT OUTPUT\n"

    def test_identical_existing_destination_is_skipped(self) -> None:
        """Identical bytes are already safe: no sibling, ok=True."""
        (self.repo / ".env").write_text("SAME\n")
        (self.wt_dir / ".env").write_text("SAME\n")
        rescued, ok = GitWorktreeOps.rescue_ignored_files(
            self.wt_dir, self.repo,
        )
        assert ok is True
        assert rescued == 0
        assert list(self.repo.glob(".env.kiss-rescued-*")) == []

    def test_no_descriptor_platform_lands_by_pathname_inside_the_repository(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Where the platform has no descriptor-relative file operations (native
        Windows: no ``O_DIRECTORY``, ``O_NOFOLLOW`` or ``dir_fd``) the rescue
        lands by pathname, as it did before the pinned landing existed: a file
        in a directory the main tree lacks arrives as a private copy with the
        source's timestamp, no temporary file stays behind, a second rescue
        finds the copy identical, and nothing is logged as failed."""
        (self.wt_dir / "data").mkdir()
        src = self.wt_dir / "data" / "out.csv"
        src.write_text("AGENT OUTPUT\n")
        os.utime(src, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
        with _rescue_backend(False), caplog.at_level(logging.WARNING, logger="kiss"):
            result = GitWorktreeOps.rescue_ignored_files(self.wt_dir, self.repo)
            landed = self.repo / "data" / "out.csv"
            assert result == (1, True), "the pathname landing rescues into a new directory"
            assert landed.read_text() == "AGENT OUTPUT\n"
            assert landed.stat().st_mtime_ns == 1_600_000_000_000_000_000, "timestamp kept"
            assert not os.path.samefile(landed, src), "a private copy, not the worktree's inode"
            assert list(self.repo.glob(f"**/*{_RESCUE_SIBLING_MARKER}*")) == [], "no temp file"
            assert GitWorktreeOps.rescue_ignored_files(self.wt_dir, self.repo) == (0, True), (
                "the landed copy is identical: the second rescue lands nothing"
            )
        assert "Failed to rescue" not in caplog.text, caplog.text

    def _agent_with_worktree(
        self, name: str, files: dict[str, str],
    ) -> ChatSorcarAgent:
        """An agent whose ``create_worktree`` ran and wrote *files*."""
        agent = ChatSorcarAgent(name)
        agent.work_dir = str(self.repo)
        assert "Created worktree" in agent.create_worktree()
        wt_work_dir = agent._wt_work_dir
        assert wt_work_dir is not None
        for rel, content in files.items():
            target = wt_work_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        return agent

    def test_merge_preserves_worktree_when_rescue_fails(self) -> None:
        """Fail closed end to end: teardown keeps the only copy."""
        saved = _redirect_db(self.tmpdir)
        try:
            agent = self._agent_with_worktree("rescue-fail-test", {
                "tracked.txt": "work\n",
                "data/out.csv": "secret\n",
            })
            assert agent._wt_pending
            # The hostile ancestor appears only AFTER the worktree got
            # its real data/ directory, so the rescue at merge time is
            # what hits the symlink.
            os.symlink(self.outside, self.repo / "data")
            wt_dir = agent._wt_dir
            assert wt_dir is not None
            msg = agent.merge()
            assert "Cannot merge" in msg, msg
            assert wt_dir.exists(), (
                "the worktree holding the only copy of the unrescuable "
                "file was destroyed"
            )
            assert (wt_dir / "data" / "out.csv").is_file()
        finally:
            _restore_db(saved)

    def test_deferred_discard_keeps_worktree_pending(self) -> None:
        """A deferred discard keeps the claim, the directory and the output."""
        saved = _redirect_db(self.tmpdir)
        try:
            agent = self._agent_with_worktree(
                "discard-defer-test", {"data/out.csv": "secret\n"},
            )
            assert agent._wt_pending
            os.symlink(self.outside, self.repo / "data")
            msg = agent.discard(rescue_ignored=True)
            assert "Discard deferred" in msg, msg
            assert agent._wt_pending
            assert agent._wt is not None and agent._wt.wt_dir.exists(), (
                "deferral must keep the only copy of the output"
            )
        finally:
            _restore_db(saved)


class TestReclaimOwnerPidGuard:
    """Cross-process reclaim skips live owners, reclaims dead ones."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-owner-pid-")
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.branch = "kiss/wt-owner-guard"
        self.wt_dir = self.repo / ".kiss-worktrees" / "kiss_wt-owner-guard"
        GitWorktreeOps.ensure_excluded(self.repo)
        assert GitWorktreeOps.create(self.repo, self.branch, self.wt_dir)
        assert GitWorktreeOps.save_original_branch(
            self.repo, self.branch,
            GitWorktreeOps.current_branch(self.repo) or "",
        )
        (self.wt_dir / "work.txt").write_text("live task output\n")

    def teardown_method(self) -> None:
        GitWorktreeOps.cleanup_partial(self.repo, self.branch, self.wt_dir)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _set_owner(self, pid: int) -> None:
        subprocess.run(
            ["git", "-C", str(self.repo), "config",
             f"branch.{self.branch}.kiss-owner-pid", str(pid)],
            capture_output=True, check=True,
        )

    def test_live_foreign_owner_is_skipped(self) -> None:
        """A worktree owned by a living other process is untouchable."""
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            self._set_owner(proc.pid)
            reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)
            assert reclaimed == 0
            assert self.wt_dir.exists(), (
                "reclaim adopted a live worktree of another process"
            )
            assert (self.wt_dir / "work.txt").is_file()
        finally:
            proc.kill()
            proc.wait()

    def test_dead_owner_is_reclaimed(self) -> None:
        """A worktree whose owner process died is reclaimed normally."""
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        self._set_owner(proc.pid)
        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)
        assert reclaimed == 1
        assert not self.wt_dir.exists()
        assert (self.repo / "work.txt").is_file()
