# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the spare-worktree pool.

The pool (``kiss.agents.sorcar.worktree_pool``) pre-creates the next
task's git worktree on a background thread so the submit path skips the
full-checkout ``git worktree add``.  These tests exercise the real git
operations on real temporary repositories — no mocks.

Unreachable-without-doubles branches, documented per the testing
policy instead of being mocked:

* ``worktree_pool.prewarm``'s outer ``except Exception`` (unexpected
  git failure after the lock is taken) and the inner ``ensure_excluded``
  failure require filesystem/permission faults injected mid-operation.
* ``server._prewarm_task_dependencies``'s per-module ``except`` requires
  breaking an installed package's import mid-session.
* ``GitWorktreeOps.create`` failure inside ``_acquire_task_worktree``'s
  inline path (marked ``pragma: no cover`` in the source) requires git
  itself to fail after the same call just succeeded for the pool.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from kiss.agents.sorcar import worktree_pool
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _enable_pool() -> str | None:
    """Re-enable background refills (the suite disables them globally).

    Returns:
        The previous ``KISS_DISABLE_WORKTREE_POOL`` value for restore.
    """
    prev = os.environ.get(worktree_pool._DISABLE_ENV)
    os.environ[worktree_pool._DISABLE_ENV] = "0"
    return prev


def _restore_pool_env(prev: str | None) -> None:
    """Restore the pool-disable env var to *prev*."""
    if prev is None:
        os.environ.pop(worktree_pool._DISABLE_ENV, None)
    else:
        os.environ[worktree_pool._DISABLE_ENV] = prev


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True, check=True,
    )
    _run_git(path, "config", "user.email", "test@test.com")
    _run_git(path, "config", "user.name", "Test")
    (path / "README.md").write_text("# Test\n")
    _run_git(path, "add", ".")
    _run_git(path, "commit", "-m", "initial")
    return path


class TestWorktreePool:
    """Pool refill, consumption, staleness, and reclaim protection."""

    def setup_method(self) -> None:
        self._prev_env = _enable_pool()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo = _make_repo(Path(self._tmpdir.name) / "repo")

    def teardown_method(self) -> None:
        worktree_pool.discard_all()
        self._tmpdir.cleanup()
        _restore_pool_env(self._prev_env)

    def test_prewarm_creates_spare_and_is_idempotent(self) -> None:
        assert worktree_pool.prewarm(self.repo) is True
        branches = worktree_pool.spare_branches()
        assert len(branches) >= 1
        (branch,) = {
            b for b in branches if GitWorktreeOps.branch_exists(self.repo, b)
        }
        assert branch.startswith("kiss/wt-")
        # Second prewarm keeps the same single spare.
        assert worktree_pool.prewarm(self.repo) is True
        spare = worktree_pool.take_spare(self.repo)
        assert spare is not None
        assert spare[0] == branch
        assert spare[1].is_dir()
        # Pool is now empty for this repo.
        assert worktree_pool.take_spare(self.repo) is None

    def test_prewarm_returns_false_while_refill_in_flight(self) -> None:
        key = worktree_pool._repo_key(self.repo)
        with worktree_pool._pool_lock:
            worktree_pool._prewarming.add(key)
        try:
            assert worktree_pool.prewarm(self.repo) is False
            assert worktree_pool.prewarm_async(self.repo) is None
        finally:
            with worktree_pool._pool_lock:
                worktree_pool._prewarming.discard(key)

    def test_prewarm_async_fills_pool_once(self) -> None:
        thread = worktree_pool.prewarm_async(self.repo)
        assert thread is not None
        thread.join(timeout=60)
        assert not thread.is_alive()
        assert worktree_pool.spare_branches()
        # With a spare pooled, no new refill thread is spawned.
        assert worktree_pool.prewarm_async(self.repo) is None

    def test_prewarm_async_noop_when_disabled(self) -> None:
        os.environ[worktree_pool._DISABLE_ENV] = "1"
        try:
            assert worktree_pool.pool_enabled() is False
            assert worktree_pool.prewarm_async(self.repo) is None
            assert worktree_pool.spare_branches() == set()
        finally:
            os.environ[worktree_pool._DISABLE_ENV] = "0"

    def test_take_spare_drops_stale_entry(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        spare = worktree_pool.take_spare(self.repo)
        assert spare is not None
        branch, wt_dir = spare
        # Put it back, then destroy the branch behind the pool's back
        # (as an external `git branch -D` or reclaim pass would).
        with worktree_pool._pool_lock:
            worktree_pool._spares[worktree_pool._repo_key(self.repo)] = spare
        GitWorktreeOps.remove(self.repo, wt_dir)
        _run_git(self.repo, "branch", "-D", branch)
        assert worktree_pool.take_spare(self.repo) is None

    def test_prewarm_reports_failure_when_create_fails(self) -> None:
        # A FILE at .kiss-worktrees blocks worktree creation for real.
        blocker = self.repo / ".kiss-worktrees"
        blocker.write_text("not a directory\n")
        try:
            assert worktree_pool.prewarm(self.repo) is False
            assert worktree_pool.take_spare(self.repo) is None
        finally:
            blocker.unlink()

    def test_prewarm_maintenance_reclaims_orphans(self) -> None:
        # Simulate a crashed process's leftover clean worktree.
        orphan_dir = self.repo / ".kiss-worktrees" / "orphan"
        _run_git(
            self.repo, "worktree", "add", "-b", "kiss/wt-orphan",
            str(orphan_dir),
        )
        assert worktree_pool.prewarm(
            self.repo, exclude_branches_fn=lambda: set(),
        )
        assert not GitWorktreeOps.branch_exists(self.repo, "kiss/wt-orphan")
        assert not orphan_dir.exists()

    def test_prewarm_without_exclusions_skips_maintenance(self) -> None:
        orphan_dir = self.repo / ".kiss-worktrees" / "orphan2"
        _run_git(
            self.repo, "worktree", "add", "-b", "kiss/wt-orphan2",
            str(orphan_dir),
        )
        assert worktree_pool.prewarm(self.repo)
        # No exclusion set was available, so no reclaim ran: the
        # orphan must be untouched.
        assert GitWorktreeOps.branch_exists(self.repo, "kiss/wt-orphan2")
        assert orphan_dir.exists()
        GitWorktreeOps.remove(self.repo, orphan_dir)
        _run_git(self.repo, "branch", "-D", "kiss/wt-orphan2")

    def test_prewarm_maintenance_survives_broken_exclusion_fn(self) -> None:
        def broken() -> set[str]:
            raise RuntimeError("exclusion source is gone")

        assert worktree_pool.prewarm(self.repo, exclude_branches_fn=broken)
        assert worktree_pool.spare_branches()

    def test_reset_worktree_to_moves_spare_to_new_head(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        (self.repo / "new_file.txt").write_text("fresh\n")
        _run_git(self.repo, "add", ".")
        _run_git(self.repo, "commit", "-m", "advance main")
        spare = worktree_pool.take_spare(self.repo)
        assert spare is not None
        branch, wt_dir = spare
        assert GitWorktreeOps.reset_worktree_to(wt_dir, "main") is True
        assert (wt_dir / "new_file.txt").read_text() == "fresh\n"
        assert _run_git(wt_dir, "rev-parse", branch) == _run_git(
            self.repo, "rev-parse", "main",
        )
        GitWorktreeOps.cleanup_partial(self.repo, branch, wt_dir)

    def test_reset_worktree_to_reports_bad_ref(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        spare = worktree_pool.take_spare(self.repo)
        assert spare is not None
        branch, wt_dir = spare
        assert GitWorktreeOps.reset_worktree_to(
            wt_dir, "no-such-ref-anywhere",
        ) is False
        GitWorktreeOps.cleanup_partial(self.repo, branch, wt_dir)

    def test_discard_all_removes_spares(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        (branch,) = worktree_pool.spare_branches()
        worktree_pool.discard_all()
        assert worktree_pool.spare_branches() == set()
        assert not GitWorktreeOps.branch_exists(self.repo, branch)

    def test_discard_all_waits_for_inflight_refill(self) -> None:
        thread = worktree_pool.prewarm_async(self.repo)
        assert thread is not None
        # Called while the refill is (very likely) still creating the
        # worktree: discard_all must join it so the spare it publishes
        # is swept too, never leaked after the cleanup returned.
        worktree_pool.discard_all()
        assert not thread.is_alive()
        assert worktree_pool.spare_branches() == set()
        time.sleep(0.5)
        assert worktree_pool.spare_branches() == set()
        result = subprocess.run(
            ["git", "-C", str(self.repo), "branch", "--list", "kiss/wt-*"],
            capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == ""

    def test_take_spare_rejects_tampered_checkout(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        with worktree_pool._pool_lock:
            spare = worktree_pool._spares[worktree_pool._repo_key(self.repo)]
        branch, wt_dir = spare
        # An external command switches the spare's checkout to a new
        # branch; the recorded branch and the directory both still
        # exist, but consuming it would reset (and later commit onto)
        # the WRONG branch.
        _run_git(wt_dir, "checkout", "-q", "-b", "kiss/wt-tampered")
        assert worktree_pool.take_spare(self.repo) is None
        # The tampered directory is left alone for the reclaim pass.
        assert wt_dir.is_dir()
        GitWorktreeOps.remove(self.repo, wt_dir)
        _run_git(self.repo, "branch", "-D", branch)
        _run_git(self.repo, "branch", "-D", "kiss/wt-tampered")

    def test_spare_marker_written_at_creation(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        (branch,) = worktree_pool.spare_branches()
        assert GitWorktreeOps.load_spare_marker(self.repo, branch)

    def test_reclaim_discards_crashed_spare_without_merging(self) -> None:
        # Spare created while the repo sits on `main`, which then gains
        # one more commit.
        assert worktree_pool.prewarm(self.repo)
        (spare_branch,) = worktree_pool.spare_branches()
        (self.repo / "main_only.txt").write_text("main work\n")
        _run_git(self.repo, "add", ".")
        _run_git(self.repo, "commit", "-m", "main advances")
        # Simulate a daemon crash: the pool's in-memory state is lost,
        # the worktree stays registered on disk.
        with worktree_pool._pool_lock:
            worktree_pool._spares.clear()
        # The user switches to a divergent branch that lacks main's
        # latest commit...
        _run_git(self.repo, "checkout", "-q", "-b", "feature",  "HEAD~1")
        head_before = _run_git(self.repo, "rev-parse", "HEAD")
        # ...and the next reclaim pass runs.  Without the spare marker
        # this squash-merged main's snapshot into `feature`; with it,
        # the contentless spare is simply discarded.
        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)
        assert reclaimed == 1
        assert _run_git(self.repo, "rev-parse", "HEAD") == head_before
        assert not (self.repo / "main_only.txt").exists()
        assert not GitWorktreeOps.branch_exists(self.repo, spare_branch)
        _run_git(self.repo, "checkout", "-q", "main")

    def test_reclaim_preserves_spare_with_unexpected_content(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        (spare_branch,) = worktree_pool.spare_branches()
        with worktree_pool._pool_lock:
            spare = worktree_pool._spares.pop(
                worktree_pool._repo_key(self.repo),
            )
        _, wt_dir = spare
        (wt_dir / "mystery.txt").write_text("who wrote this?\n")
        assert GitWorktreeOps.reclaim_orphaned_worktrees(self.repo) == 0
        assert GitWorktreeOps.branch_exists(self.repo, spare_branch)
        assert (wt_dir / "mystery.txt").exists()
        GitWorktreeOps.cleanup_partial(self.repo, spare_branch, wt_dir)


def _wait_for_refill(timeout: float = 60.0) -> None:
    """Wait until no pool refill thread is running."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with worktree_pool._pool_lock:
            busy = bool(worktree_pool._prewarming)
        if not busy:
            # Also wait out the thread's tail between pool-map update
            # and thread exit.
            for thread in threading.enumerate():
                if thread.name.startswith("worktree-pool-prewarm-"):
                    thread.join(timeout=timeout)
            return
        time.sleep(0.05)
    raise AssertionError("pool refill did not finish in time")


class TestAcquireTaskWorktree:
    """The agent's setup path consumes and refills the pool."""

    def setup_method(self) -> None:
        self._prev_env = _enable_pool()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo = _make_repo(Path(self._tmpdir.name) / "repo")

    def teardown_method(self) -> None:
        _wait_for_refill()
        worktree_pool.discard_all()
        self._tmpdir.cleanup()
        _restore_pool_env(self._prev_env)

    def test_setup_consumes_pooled_spare(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        (pooled_branch,) = worktree_pool.spare_branches()
        agent = WorktreeSorcarAgent("pool-test")
        wt_work = agent._try_setup_worktree(self.repo, str(self.repo))
        assert wt_work is not None
        assert agent._wt is not None
        assert agent._wt.branch == pooled_branch
        assert agent._wt.original_branch == "main"
        # The spare was consumed; a refill was scheduled for the next
        # task and mints a DIFFERENT branch.
        _wait_for_refill()
        refilled = worktree_pool.spare_branches()
        assert refilled and pooled_branch not in refilled
        agent.discard()

    def test_setup_inline_fallback_on_empty_pool(self) -> None:
        agent = WorktreeSorcarAgent("pool-test")
        wt_work = agent._try_setup_worktree(self.repo, str(self.repo))
        assert wt_work is not None
        assert agent._wt is not None
        assert GitWorktreeOps.branch_exists(self.repo, agent._wt.branch)
        _wait_for_refill()
        assert worktree_pool.spare_branches()
        agent.discard()

    def test_setup_falls_back_when_spare_reset_fails(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        spare = worktree_pool.take_spare(self.repo)
        assert spare is not None
        branch, wt_dir = spare
        # Lock the spare's index so `git reset --hard` fails while the
        # checkout-validation in take_spare still passes, then re-pool
        # it: acquisition must discard it and create a fresh worktree
        # inline.
        admin_dir = Path(_run_git(wt_dir, "rev-parse", "--absolute-git-dir"))
        (admin_dir / "index.lock").write_text("")
        with worktree_pool._pool_lock:
            worktree_pool._spares[worktree_pool._repo_key(self.repo)] = spare
        agent = WorktreeSorcarAgent("pool-test")
        wt_work = agent._try_setup_worktree(self.repo, str(self.repo))
        assert wt_work is not None
        assert agent._wt is not None
        assert agent._wt.branch != branch
        assert not GitWorktreeOps.branch_exists(self.repo, branch)
        agent.discard()

    def test_setup_falls_back_when_spare_is_corrupted(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        spare = worktree_pool.take_spare(self.repo)
        assert spare is not None
        branch, wt_dir = spare
        # A spare whose git link is destroyed fails take_spare's
        # checkout validation; it is left on disk (its content is not
        # ours to judge) and the setup creates a worktree inline.
        (wt_dir / ".git").write_text("gitdir: /nonexistent\n")
        with worktree_pool._pool_lock:
            worktree_pool._spares[worktree_pool._repo_key(self.repo)] = spare
        agent = WorktreeSorcarAgent("pool-test")
        wt_work = agent._try_setup_worktree(self.repo, str(self.repo))
        assert wt_work is not None
        assert agent._wt is not None
        assert agent._wt.branch != branch
        assert wt_dir.is_dir()
        agent.discard()
        GitWorktreeOps.remove(self.repo, wt_dir)
        _run_git(self.repo, "branch", "-D", branch)

    def test_contaminated_spare_refused_not_destroyed(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        with worktree_pool._pool_lock:
            (_, spare_dir), = worktree_pool._spares.values()
        # An external writer drops a file into the idle spare.  The
        # spare is never written to by the pool, so this content is
        # not ours to destroy: consumption refuses it (preserving the
        # file for the reclaim pass to inspect) and the task falls
        # back to a fresh inline worktree.  The old behavior —
        # consume + `git clean -fdq` — deleted the external file
        # (gpt-5.6-sol review finding).
        (spare_dir / "generated.txt").write_text("stray build output\n")
        agent = WorktreeSorcarAgent("pool-test")
        wt_work = agent._try_setup_worktree(self.repo, str(self.repo))
        assert wt_work is not None
        assert Path(wt_work) != spare_dir
        assert (spare_dir / "generated.txt").read_text() == (
            "stray build output\n"
        )
        agent.discard()

    def test_consume_clears_spare_marker(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        agent = WorktreeSorcarAgent("pool-test")
        wt_work = agent._try_setup_worktree(self.repo, str(self.repo))
        assert wt_work is not None
        assert agent._wt is not None
        # Once real work can land in the worktree, the marker must be
        # gone — a crash mid-task must route it through the normal
        # orphan reclaim (merge), not the spare-discard path.
        assert not GitWorktreeOps.load_spare_marker(
            self.repo, agent._wt.branch,
        )
        agent.discard()

    def test_dirty_state_copied_into_spare_based_worktree(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        (self.repo / "uncommitted.txt").write_text("dirty\n")
        agent = WorktreeSorcarAgent("pool-test")
        wt_work = agent._try_setup_worktree(self.repo, str(self.repo))
        assert wt_work is not None
        assert (Path(wt_work) / "uncommitted.txt").read_text() == "dirty\n"
        assert agent._wt is not None and agent._wt.baseline_commit
        agent.discard()

    def test_merge_from_spare_based_worktree(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        agent = WorktreeSorcarAgent("pool-test")
        wt_work = agent._try_setup_worktree(self.repo, str(self.repo))
        assert wt_work is not None
        (Path(wt_work) / "task_output.txt").write_text("done\n")
        message = agent.merge()
        assert "Successfully merged" in message
        assert (self.repo / "task_output.txt").read_text() == "done\n"
        assert _run_git(self.repo, "branch", "--show-current") == "main"

    def test_live_branches_include_pooled_spares(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        (pooled_branch,) = worktree_pool.spare_branches()
        agent = WorktreeSorcarAgent("pool-test")
        assert pooled_branch in agent._live_worktree_branches()

    def test_reclaim_with_live_exclusions_preserves_pooled_spare(self) -> None:
        assert worktree_pool.prewarm(self.repo)
        (pooled_branch,) = worktree_pool.spare_branches()
        agent = WorktreeSorcarAgent("pool-test")
        # A reclaim pass built with the agent's live-branch exclusion
        # set (the set every inline setup passes) must leave the
        # pooled spare untouched: without the pool union in
        # _live_worktree_branches the spare — a clean worktree with no
        # kiss-original config — would be adopted, merged and deleted.
        GitWorktreeOps.reclaim_orphaned_worktrees(
            self.repo,
            exclude_branches=agent._live_worktree_branches(),
        )
        GitWorktreeOps.sweep_orphaned_state(self.repo)
        assert GitWorktreeOps.branch_exists(self.repo, pooled_branch)
        assert worktree_pool.take_spare(self.repo) is not None
