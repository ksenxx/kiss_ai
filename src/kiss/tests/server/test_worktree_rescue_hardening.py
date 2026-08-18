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
* a deferred discard keeps ``_pending_review`` so a tab close cannot
  auto-merge work the user asked to throw away;
* ``reclaim_orphaned_worktrees`` skips a worktree whose recorded
  owner process is still alive (cross-process live-worktree adoption)
  and still reclaims one whose owner is dead;
* ``take_spare`` refuses a spare contaminated by an external writer
  instead of destroying the content on consume.

All tests use real git repositories, real worktrees, real processes;
no mocks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar import worktree_pool
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.tests.server.test_worktree_ignored_file_rescue import (
    _make_repo,
    _redirect_db,
    _restore_db,
    _stub_parent_run,
)


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

    def test_symlinked_ancestor_never_escapes_repo(self) -> None:
        """A main-tree dir symlink pointing outside blocks the rescue."""
        os.symlink(self.outside, self.repo / "data")
        (self.wt_dir / "data").mkdir()
        (self.wt_dir / "data" / "out.csv").write_text("secret\n")
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

    def test_collision_lands_kiss_rescued_sibling(self) -> None:
        """Different existing content is preserved beside, not dropped."""
        (self.repo / ".env").write_text("SECRET=users\n")
        (self.wt_dir / ".env").write_text("SECRET=agents\n")
        rescued, ok = GitWorktreeOps.rescue_ignored_files(
            self.wt_dir, self.repo,
        )
        assert ok is True
        assert rescued == 1
        assert (self.repo / ".env").read_text() == "SECRET=users\n"
        siblings = list(self.repo.glob(".env.kiss-rescued-*"))
        assert len(siblings) == 1, (
            "the worktree's differing version must land beside the "
            "user's file"
        )
        assert siblings[0].read_text() == "SECRET=agents\n"

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

    def test_merge_preserves_worktree_when_rescue_fails(self) -> None:
        """Fail closed end to end: teardown keeps the only copy."""
        saved = _redirect_db(self.tmpdir)
        original_run = _stub_parent_run({
            "tracked.txt": "work\n",
            "data/out.csv": "secret\n",
        })
        try:
            agent = WorktreeSorcarAgent("rescue-fail-test")
            agent.run("task", work_dir=str(self.repo), auto_commit=True)
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
            cast(Any, SorcarAgent.__mro__[1]).run = original_run
            _restore_db(saved)

    def test_deferred_discard_keeps_pending_review(self) -> None:
        """A deferred discard must not drop the pending-review guard."""
        saved = _redirect_db(self.tmpdir)
        original_run = _stub_parent_run({"data/out.csv": "secret\n"})
        try:
            agent = WorktreeSorcarAgent("discard-defer-test")
            agent.run("task", work_dir=str(self.repo), auto_commit=True)
            assert agent._wt_pending
            os.symlink(self.outside, self.repo / "data")
            agent._pending_review = True
            msg = agent.discard(rescue_ignored=True)
            assert "Discard deferred" in msg, msg
            assert agent._wt_pending
            assert agent._pending_review is True, (
                "deferral must keep the never-auto-merge protection"
            )
        finally:
            cast(Any, SorcarAgent.__mro__[1]).run = original_run
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
        proc = subprocess.Popen(["sleep", "60"])
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
        proc = subprocess.Popen(["true"])
        proc.wait()
        self._set_owner(proc.pid)
        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)
        assert reclaimed == 1
        assert not self.wt_dir.exists()
        assert (self.repo / "work.txt").is_file()


class TestTakeSpareContaminationGuard:
    """take_spare preserves externally contaminated spares."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-spare-guard-")
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        worktree_pool.discard_all()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_contaminated_spare_refused_and_preserved(self) -> None:
        """An untracked external file blocks consumption."""
        assert worktree_pool.prewarm(self.repo)
        with worktree_pool._pool_lock:
            branch, wt_dir = worktree_pool._spares[
                worktree_pool._repo_key(self.repo)
            ]
        (wt_dir / "external.txt").write_text("do not destroy\n")
        assert worktree_pool.take_spare(self.repo) is None
        assert (wt_dir / "external.txt").is_file(), (
            "take_spare destroyed external content on consume"
        )
        assert GitWorktreeOps.branch_exists(self.repo, branch)

    def test_clean_spare_is_consumed(self) -> None:
        """A pristine spare is still handed out (fast path intact)."""
        assert worktree_pool.prewarm(self.repo)
        spare = worktree_pool.take_spare(self.repo)
        assert spare is not None
        branch, wt_dir = spare
        assert wt_dir.is_dir()
        GitWorktreeOps.cleanup_partial(self.repo, branch, wt_dir)
