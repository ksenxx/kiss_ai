# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regressions for area-E redundancy/race fixes in
``sorcar/git_worktree.py``.

Covers audit findings:

* **E-R4** — ``rescue_ignored_files`` re-inlined the ``git ls-files``
  invocation because ``list_ignored_files`` conflated "git failed"
  with "no ignored files".  ``list_ignored_files`` now returns
  ``None`` on git failure; the rescue keeps failing closed and the
  orphan-reclaim spare probe preserves a spare it cannot enumerate.
* **E-RC2** — ``repo_lock`` is a ``threading.RLock`` (in-process
  only), so two PROCESSES reclaiming the same dead owner's worktree
  could interleave stage → cherry-pick → commit → cleanup and lose
  the orphan's work.  The whole reclaim (and
  ``sweep_orphaned_state``) now runs under an ``flock`` on
  ``<git_common_dir>/kiss-reclaim.lock``.

Every test uses a REAL git repository and REAL child processes.  No
mocks, patches or doubles, and no LLM calls.

Unreachable branch note: the orphan-reclaim SPARE-content probe's
``ignored is None`` arm (preserve a pool spare whose ignored files
cannot be enumerated) requires ``git ls-files`` to fail inside a
worktree that ``git worktree list`` just reported as registered —
i.e. git breaking between two calls of one loop iteration.  That
cannot be produced end-to-end without a git test double, so the
``None`` contract is covered here via ``list_ignored_files`` /
``rescue_ignored_files`` directly instead.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps

_RACE_DELAY_ENV = "KISS_RACE_DELAY"
_RECLAIM_MARKER = "Auto-merged by orphan-worktree reclaim"


def _make_repo(path: Path) -> Path:
    """Create a real git repo with one initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True, check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True, check=True,
    )
    return path


def _plant_orphan_worktree(repo: Path, branch: str) -> Path:
    """Register a ``kiss/wt-*`` worktree with staged dirty work.

    Simulates a Sorcar process killed mid-task: the worktree is
    registered, ``kiss-original`` config is saved, and one staged
    file awaits reclaim.  The owner pid ``create`` stamps (this live
    test process) is removed so the worktree looks like one left
    behind by a dead process and any process may reclaim it.
    """
    wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    subprocess.run(
        ["git", "-C", str(repo), "config", "--unset",
         f"branch.{branch}.kiss-owner-pid"],
        capture_output=True, check=True,
    )
    assert GitWorktreeOps.save_original_branch(repo, branch, "main")
    (wt_dir / "orphan_output.txt").write_text("the orphan's work\n")
    GitWorktreeOps.stage_all(wt_dir)
    return wt_dir


def _reclaim_worker(repo_str: str, barrier, out_queue) -> None:
    """Child process: reclaim orphans at the barrier, report the count.

    ``KISS_RACE_DELAY`` widens the window between the reclaim guards
    and the commit/merge/cleanup mutations, so on unserialised code
    both children reliably enter the mutation phase together.
    """
    os.environ[_RACE_DELAY_ENV] = "0.1"
    try:
        barrier.wait(timeout=60)
        count = GitWorktreeOps.reclaim_orphaned_worktrees(Path(repo_str))
        out_queue.put(count)
    except BaseException as exc:  # pragma: no cover — child crash
        out_queue.put(f"error: {exc!r}")
    finally:
        os.environ.pop(_RACE_DELAY_ENV, None)


class TestListIgnoredFiles(unittest.TestCase):
    """E-R4 — git failure is distinguishable from no ignored files."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_rr_e_ign_"))
        self.repo = _make_repo(self.tmp / "repo")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_none_outside_a_repository(self) -> None:
        bare_dir = self.tmp / "not_a_repo"
        bare_dir.mkdir()
        self.assertIsNone(GitWorktreeOps.list_ignored_files(bare_dir))

    def test_returns_empty_list_for_clean_repository(self) -> None:
        self.assertEqual(GitWorktreeOps.list_ignored_files(self.repo), [])

    def test_returns_ignored_paths(self) -> None:
        (self.repo / ".gitignore").write_text("*.log\n")
        (self.repo / "debug.log").write_text("x\n")
        self.assertEqual(
            GitWorktreeOps.list_ignored_files(self.repo), ["debug.log"],
        )

    def test_rescue_fails_closed_when_enumeration_fails(self) -> None:
        bare_dir = self.tmp / "fake_worktree"
        bare_dir.mkdir()
        (bare_dir / "precious.txt").write_text("do not lose\n")
        rescued, ok = GitWorktreeOps.rescue_ignored_files(
            bare_dir, self.repo,
        )
        self.assertEqual(rescued, 0)
        self.assertFalse(ok)
        # Nothing was copied into the repo.
        self.assertFalse((self.repo / "precious.txt").exists())

    def test_rescue_still_lands_ignored_files(self) -> None:
        branch = "kiss/wt-100-rescue"
        wt_dir = _plant_orphan_worktree(self.repo, branch)
        (wt_dir / ".gitignore").write_text("*.secret\n")
        (wt_dir / "keys.secret").write_text("hunter2\n")
        rescued, ok = GitWorktreeOps.rescue_ignored_files(wt_dir, self.repo)
        self.assertTrue(ok)
        self.assertGreaterEqual(rescued, 1)
        self.assertEqual(
            (self.repo / "keys.secret").read_text(), "hunter2\n",
        )


class TestCrossProcessReclaim(unittest.TestCase):
    """E-RC2 — concurrent reclaimers cannot destroy the orphan's work."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_rr_e_reclaim_"))
        self.repo = _make_repo(self.tmp / "repo")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _reclaim_commit_count(self, repo: Path | None = None) -> int:
        result = subprocess.run(
            ["git", "-C", str(repo or self.repo), "log", "--format=%B",
             "main"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.count(_RECLAIM_MARKER)

    def test_two_processes_reclaim_exactly_once(self) -> None:
        # Several rounds: the pre-fix interleaving is timing-dependent
        # (git's transient index.lock absorbs some overlaps), so one
        # round reproduced the loss only ~40% of the time; four rounds
        # catch a regression with ~87% probability while the fixed
        # code must pass every round deterministically.
        for round_no in range(4):
            repo = _make_repo(self.tmp / f"repo_round_{round_no}")
            branch = "kiss/wt-200-race"
            wt_dir = _plant_orphan_worktree(repo, branch)

            ctx = multiprocessing.get_context()
            barrier = ctx.Barrier(2)
            out_queue = ctx.Queue()
            procs = [
                ctx.Process(
                    target=_reclaim_worker,
                    args=(str(repo), barrier, out_queue),
                )
                for _ in range(2)
            ]
            for proc in procs:
                proc.start()
            results = [out_queue.get(timeout=120) for _ in procs]
            for proc in procs:
                proc.join(timeout=60)

            for value in results:
                self.assertIsInstance(value, int, f"worker failed: {value}")
            # Exactly one process reclaimed the worktree; the other
            # found nothing left — never both.
            self.assertEqual(sum(results), 1, f"results: {results}")
            # The orphan's work survived, exactly once, on main.
            self.assertEqual(
                (repo / "orphan_output.txt").read_text(),
                "the orphan's work\n",
            )
            self.assertEqual(self._reclaim_commit_count(repo), 1)
            self.assertFalse(wt_dir.exists())
            self.assertFalse(GitWorktreeOps.branch_exists(repo, branch))
            # The main tree is left clean for the user.
            self.assertFalse(GitWorktreeOps.has_uncommitted_changes(repo))

    def test_reclaim_waits_for_peer_holding_the_lock(self) -> None:
        # Deterministic serialisation proof: while a peer process
        # holds <git_common_dir>/kiss-reclaim.lock, a reclaim must
        # BLOCK — pre-fix code (no cross-process lock) reclaims
        # immediately and this test fails.
        import fcntl

        branch = "kiss/wt-203-block"
        wt_dir = _plant_orphan_worktree(self.repo, branch)
        lock_path = self.repo / ".git" / "kiss-reclaim.lock"
        handle = open(lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            ctx = multiprocessing.get_context()
            barrier = ctx.Barrier(1)
            out_queue = ctx.Queue()
            proc = ctx.Process(
                target=_reclaim_worker,
                args=(str(self.repo), barrier, out_queue),
            )
            proc.start()
            time.sleep(1.0)
            # The child is still waiting on the flock: nothing
            # reclaimed, the worktree untouched.
            self.assertTrue(out_queue.empty())
            self.assertTrue(wt_dir.exists())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        self.assertEqual(out_queue.get(timeout=120), 1)
        proc.join(timeout=60)
        self.assertFalse(wt_dir.exists())

    def test_reclaim_lock_file_lives_in_git_common_dir(self) -> None:
        _plant_orphan_worktree(self.repo, "kiss/wt-201-lockfile")
        self.assertEqual(
            GitWorktreeOps.reclaim_orphaned_worktrees(self.repo), 1,
        )
        self.assertTrue((self.repo / ".git" / "kiss-reclaim.lock").exists())

    def test_sweep_orphaned_state_serialised_by_same_lock(self) -> None:
        # A leftover expendable branch whose worktree is gone is
        # plumbing debris; the sweep (now under the cross-process
        # lock) still removes it.
        branch = "kiss/wt-202-debris"
        wt_dir = _plant_orphan_worktree(self.repo, branch)
        subprocess.run(
            ["git", "-C", str(self.repo), "worktree", "remove",
             "--force", str(wt_dir)],
            capture_output=True, check=True,
        )
        deleted = GitWorktreeOps.sweep_orphaned_state(self.repo)
        self.assertEqual(deleted, 1)
        self.assertTrue((self.repo / ".git" / "kiss-reclaim.lock").exists())


if __name__ == "__main__":
    unittest.main()
