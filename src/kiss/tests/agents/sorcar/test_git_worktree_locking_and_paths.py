# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regressions for ``sorcar/git_worktree.py``.

Covers audit findings:

* **R4** — ``GitWorktreeOps.remove()`` / ``prune()`` mutate the shared
  worktree registry but were called without ``repo_lock`` from the
  teardown path, so they could run while another tab held the lock for
  ``worktree add`` / ``stash`` / ``checkout`` / ``cherry-pick``.
* **R5** — ``_append_info_line``'s read-then-append was guarded only by
  an in-process lock, so two Sorcar PROCESSES duplicated the entry in
  ``info/exclude`` / ``info/attributes``.
* **I2** — the failed-cherry-pick cleanup discarded
  ``cherry-pick --abort``'s return code.
* **I3** — ``.splitlines()`` + ``.strip()`` on git-emitted paths and on
  git-config values.
* **D1** — ``reclaim_orphaned_worktrees`` re-staged before ``commit_all``
  and re-inlined ``cleanup_partial``.

Every test uses REAL temporary git repositories, REAL threads and REAL
child processes.  No mocks, patches or doubles.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    MergeResult,
    _git,
    _race_delay,
    repo_lock,
)

_RACE_DELAY_ENV = "KISS_RACE_DELAY"


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a git command in *cwd*, raising on failure."""
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def _make_repo(path: Path) -> Path:
    """Create a git repo at *path* with one initial commit on ``main``."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True, text=True, check=True,
    )
    _run("config", "user.email", "t@t.com", cwd=path)
    _run("config", "user.name", "T", cwd=path)
    _run("config", "commit.gpgsign", "false", cwd=path)
    (path / "f.txt").write_text("a\n")
    (path / "g.txt").write_text("x\n")
    _run("add", "-A", cwd=path)
    _run("commit", "-m", "initial", cwd=path)
    return path


def _append_worker(repo_str: str, barrier, iterations: int) -> None:
    """Child process: hammer ``ensure_scratch_merge_driver`` on one repo."""
    from kiss.agents.sorcar.git_worktree import GitWorktreeOps as Ops

    repo = Path(repo_str)
    barrier.wait()
    for _ in range(iterations):
        Ops.ensure_scratch_merge_driver(repo)


class RemoveTakesRepoLockTest(unittest.TestCase):
    """R4: teardown must serialise against other tabs' git sequences."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_e_r4_"))
        self.repo = _make_repo(self.tmp / "repo")
        self.wt_dir = self.repo / ".kiss-worktrees" / "kiss_wt-a"
        self.assertTrue(
            GitWorktreeOps.create(self.repo, "kiss/wt-a", self.wt_dir)
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _assert_serialised(self, operation) -> None:
        """Assert *operation* cannot start while ``repo_lock`` is held."""
        order: list[str] = []
        acquired = threading.Event()

        def holder() -> None:
            with repo_lock(self.repo):
                order.append("lock-acquired")
                acquired.set()
                time.sleep(1.0)
                order.append("lock-released")

        thread = threading.Thread(target=holder, name="repo-lock-holder")
        thread.start()
        self.assertTrue(acquired.wait(timeout=10))
        operation()
        order.append("op-done")
        thread.join(timeout=10)
        self.assertEqual(
            order, ["lock-acquired", "lock-released", "op-done"],
        )

    def test_remove_waits_for_repo_lock(self) -> None:
        """``remove()`` blocks until the other tab releases the lock."""
        self._assert_serialised(
            lambda: GitWorktreeOps.remove(self.repo, self.wt_dir)
        )
        self.assertFalse(self.wt_dir.exists())

    def test_prune_waits_for_repo_lock(self) -> None:
        """``prune()`` blocks until the other tab releases the lock."""
        self._assert_serialised(lambda: GitWorktreeOps.prune(self.repo))

    def test_remove_is_reentrant_for_an_existing_lock_holder(self) -> None:
        """A caller already holding ``repo_lock`` must not deadlock."""
        with repo_lock(self.repo):
            GitWorktreeOps.remove(self.repo, self.wt_dir)
        self.assertFalse(self.wt_dir.exists())

    def test_remove_refuses_the_main_working_tree(self) -> None:
        """The main tree is never removed, even under the new lock."""
        GitWorktreeOps.remove(self.repo, self.repo)
        self.assertTrue((self.repo / "f.txt").exists())

    def test_remove_prunes_when_directory_is_already_gone(self) -> None:
        """A vanished worktree directory still gets unregistered."""
        shutil.rmtree(self.wt_dir)
        GitWorktreeOps.remove(self.repo, self.wt_dir)
        self.assertNotIn(
            "kiss/wt-a", GitWorktreeOps.checked_out_branches(self.repo),
        )

    def test_remove_falls_back_to_rmtree_for_a_corrupt_worktree(self) -> None:
        """A worktree git refuses to remove is deleted directly."""
        (self.wt_dir / ".git").unlink()
        (self.wt_dir / ".git").mkdir()
        GitWorktreeOps.remove(self.repo, self.wt_dir)
        self.assertFalse(self.wt_dir.exists())


class AppendInfoLineCrossProcessTest(unittest.TestCase):
    """R5: the info/ plumbing files must not accrue duplicate lines."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_e_r5_"))
        self.repo = _make_repo(self.tmp / "repo")
        self._saved_delay = os.environ.get(_RACE_DELAY_ENV)
        os.environ[_RACE_DELAY_ENV] = "0.05"

    def tearDown(self) -> None:
        if self._saved_delay is None:
            os.environ.pop(_RACE_DELAY_ENV, None)
        else:
            os.environ[_RACE_DELAY_ENV] = self._saved_delay
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_concurrent_processes_append_each_line_once(self) -> None:
        """Two real processes racing the read-append write one line each."""
        ctx = multiprocessing.get_context("spawn")
        barrier = ctx.Barrier(2)
        procs = [
            ctx.Process(
                target=_append_worker, args=(str(self.repo), barrier, 5),
            )
            for _ in range(2)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
            self.assertEqual(p.exitcode, 0)

        attributes = self.repo / ".git" / "info" / "attributes"
        lines = attributes.read_text().splitlines()
        self.assertEqual(
            Counter(lines),
            Counter([
                "PROGRESS.md merge=kiss-scratch",
                "src/kiss/INJECTIONS.md merge=kiss-scratch",
            ]),
        )

    def test_append_preserves_a_file_without_trailing_newline(self) -> None:
        """A pre-existing unterminated last line is not glued to ours."""
        info = self.repo / ".git" / "info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "exclude").write_text("*.log")
        GitWorktreeOps.ensure_excluded(self.repo)
        self.assertEqual(
            (info / "exclude").read_text(), "*.log\n.kiss-worktrees/\n",
        )
        GitWorktreeOps.ensure_excluded(self.repo)
        self.assertEqual(
            (info / "exclude").read_text(), "*.log\n.kiss-worktrees/\n",
        )


class GitOutputParsingTest(unittest.TestCase):
    """I3: git-emitted paths and config values must not be mangled."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_e_i3_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_worktree_paths_with_a_trailing_space_survive(self) -> None:
        """A repo whose directory name ends in a space stays resolvable."""
        repo = _make_repo(self.tmp / "proj ")
        self.assertTrue(str(repo).endswith(" "))
        wt_dir = repo / ".kiss-worktrees" / "kiss_wt-a"
        self.assertTrue(GitWorktreeOps.create(repo, "kiss/wt-a", wt_dir))

        pairs = GitWorktreeOps.registered_worktrees(repo)
        self.assertEqual(len(pairs), 2)
        for wt_path, _branch in pairs:
            self.assertTrue(
                wt_path.exists(), f"{wt_path!r} was mangled by parsing",
            )
        self.assertEqual(
            {b for _d, b in pairs}, {"main", "kiss/wt-a"},
        )
        self.assertEqual(
            GitWorktreeOps.checked_out_branches(repo), {"main", "kiss/wt-a"},
        )

    def test_multiline_config_values_do_not_forge_branch_names(self) -> None:
        """A multi-line ``branch.*`` value yields no bogus section name."""
        repo = _make_repo(self.tmp / "proj")
        _run(
            "config", "branch.kiss/wt-a.description", "line one\nbranch.evil.x",
            cwd=repo,
        )
        self.assertEqual(
            GitWorktreeOps._config_branch_sections(repo), {"kiss/wt-a"},
        )

    def test_config_branch_sections_empty_when_nothing_matches(self) -> None:
        """``git config`` exiting 1 on no match yields an empty set."""
        repo = _make_repo(self.tmp / "plain")
        self.assertEqual(GitWorktreeOps._config_branch_sections(repo), set())

    def test_detached_head_worktrees_are_skipped(self) -> None:
        """A detached worktree contributes no branch entry."""
        repo = _make_repo(self.tmp / "det")
        head = _run("rev-parse", "HEAD", cwd=repo).stdout.strip()
        wt_dir = repo / ".kiss-worktrees" / "kiss_wt-d"
        _run("worktree", "add", "--detach", str(wt_dir), head, cwd=repo)
        self.assertEqual(
            GitWorktreeOps.checked_out_branches(repo), {"main"},
        )


class FailedCherryPickCleanupTest(unittest.TestCase):
    """I2: a failed baseline merge must not leave a half-applied tree."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_e_i2_"))
        self.repo = _make_repo(self.tmp / "repo")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_branch(self) -> str:
        """Create ``kiss/wt-a`` with a baseline plus two commits.

        Returns:
            The baseline commit SHA.
        """
        _run("checkout", "-b", "kiss/wt-a", cwd=self.repo)
        (self.repo / "f.txt").write_text("a-baseline\n")
        _run("commit", "-am", "baseline", cwd=self.repo)
        baseline = _run("rev-parse", "HEAD", cwd=self.repo).stdout.strip()
        (self.repo / "g.txt").write_text("g-agent\n")
        _run("commit", "-am", "agent 1", cwd=self.repo)
        (self.repo / "f.txt").write_text("a-agent\n")
        _run("commit", "-am", "agent 2", cwd=self.repo)
        _run("checkout", "main", cwd=self.repo)
        return baseline

    def _assert_no_pick_in_progress(self) -> None:
        """Assert no sequencer / CHERRY_PICK_HEAD state survived."""
        git_dir = self.repo / ".git"
        self.assertFalse((git_dir / "sequencer").exists())
        self.assertFalse((git_dir / "CHERRY_PICK_HEAD").exists())
        self.assertFalse((git_dir / "MERGE_MSG").exists())

    def test_unstaged_conflict_leaves_the_users_work_untouched(self) -> None:
        """A pick refused up front must not touch the user's edits."""
        baseline = self._make_branch()
        (self.repo / "f.txt").write_text("a-user-unstaged\n")
        before = GitWorktreeOps.status_porcelain(self.repo)

        result = GitWorktreeOps.squash_merge_from_baseline(
            self.repo, "kiss/wt-a", baseline,
        )

        self.assertEqual(result, MergeResult.CONFLICT)
        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), before)
        self.assertEqual(
            (self.repo / "f.txt").read_text(), "a-user-unstaged\n",
        )
        self.assertEqual((self.repo / "g.txt").read_text(), "x\n")
        self._assert_no_pick_in_progress()

    def test_real_conflict_leaves_no_partial_index(self) -> None:
        """A genuine cross-branch conflict leaves nothing staged behind.

        Committing on main first makes ``HEAD != baseline^``, so the
        ``-X theirs`` tie-breaker is deliberately NOT applied and the
        cherry-pick of the agent's ``f.txt`` commit really conflicts.
        """
        baseline = self._make_branch()
        (self.repo / "f.txt").write_text("a-user-committed\n")
        _run("commit", "-am", "user work on main", cwd=self.repo)

        result = GitWorktreeOps.squash_merge_from_baseline(
            self.repo, "kiss/wt-a", baseline,
        )

        self.assertEqual(result, MergeResult.CONFLICT)
        self.assertEqual(
            _git("diff", "--cached", "--quiet", cwd=self.repo).returncode, 0,
        )
        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), "")
        self.assertNotIn("<<<<<<<", (self.repo / "f.txt").read_text())
        self._assert_no_pick_in_progress()

    def test_untracked_collision_preserves_the_untracked_file(self) -> None:
        """A pick blocked by an untracked file keeps that file intact."""
        _run("checkout", "-b", "kiss/wt-a", cwd=self.repo)
        (self.repo / "f.txt").write_text("a-baseline\n")
        _run("commit", "-am", "baseline", cwd=self.repo)
        baseline = _run("rev-parse", "HEAD", cwd=self.repo).stdout.strip()
        (self.repo / "new.txt").write_text("from-agent\n")
        _run("add", "-A", cwd=self.repo)
        _run("commit", "-m", "agent adds new.txt", cwd=self.repo)
        _run("checkout", "main", cwd=self.repo)
        (self.repo / "new.txt").write_text("user-untracked\n")
        before = GitWorktreeOps.status_porcelain(self.repo)

        result = GitWorktreeOps.squash_merge_from_baseline(
            self.repo, "kiss/wt-a", baseline,
        )

        self.assertEqual(result, MergeResult.CONFLICT)
        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), before)
        self.assertEqual(
            (self.repo / "new.txt").read_text(), "user-untracked\n",
        )
        self._assert_no_pick_in_progress()

    def test_abort_helper_restores_a_partially_applied_index(self) -> None:
        """When ``--abort`` fails and the tree changed, state is reset."""
        before = GitWorktreeOps.status_porcelain(self.repo)
        (self.repo / "g.txt").write_text("half-applied\n")
        _run("add", "g.txt", cwd=self.repo)
        self.assertNotEqual(
            GitWorktreeOps.status_porcelain(self.repo), before,
        )

        GitWorktreeOps._abort_cherry_pick(self.repo, before)

        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), before)
        self.assertEqual((self.repo / "g.txt").read_text(), "x\n")

    def test_abort_helper_keeps_unrelated_dirty_state(self) -> None:
        """Nothing changed by the pick means nothing to undo."""
        (self.repo / "f.txt").write_text("user-edit\n")
        before = GitWorktreeOps.status_porcelain(self.repo)

        GitWorktreeOps._abort_cherry_pick(self.repo, before)

        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), before)
        self.assertEqual((self.repo / "f.txt").read_text(), "user-edit\n")

    def test_successful_baseline_merge_still_works(self) -> None:
        """The happy path is unaffected by the new cleanup."""
        baseline = self._make_branch()
        result = GitWorktreeOps.squash_merge_from_baseline(
            self.repo, "kiss/wt-a", baseline,
        )
        self.assertEqual(result, MergeResult.SUCCESS)
        self.assertEqual((self.repo / "g.txt").read_text(), "g-agent\n")


class ReclaimEfficiencyTest(unittest.TestCase):
    """D1: reclaim must stage once and reuse ``cleanup_partial``."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_e_d1_"))
        self.repo = _make_repo(self.tmp / "repo")
        self.log = self.tmp / "git-calls.log"
        real_git = shutil.which("git")
        assert real_git is not None
        shim_dir = self.tmp / "bin"
        shim_dir.mkdir()
        shim = shim_dir / "git"
        shim.write_text(
            "#!/bin/sh\n"
            f'printf "%s\\n" "$*" >> "{self.log}"\n'
            f'exec "{real_git}" "$@"\n'
        )
        shim.chmod(0o755)
        self._saved_path = os.environ["PATH"]
        os.environ["PATH"] = f"{shim_dir}{os.pathsep}{self._saved_path}"

    def tearDown(self) -> None:
        os.environ["PATH"] = self._saved_path
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _count(self, needle: str) -> int:
        """Count logged git invocations containing *needle*."""
        if not self.log.exists():
            return 0
        return sum(1 for line in self.log.read_text().splitlines()
                   if needle in line)

    def test_reclaim_stages_once_and_does_not_double_prune(self) -> None:
        """One orphan reclaim runs ``git add -A`` exactly once."""
        wt_dir = self.repo / ".kiss-worktrees" / "kiss_wt-a"
        self.assertTrue(GitWorktreeOps.create(self.repo, "kiss/wt-a", wt_dir))
        GitWorktreeOps.save_original_branch(self.repo, "kiss/wt-a", "main")
        (wt_dir / "agent.txt").write_text("agent work\n")

        self.log.write_text("")
        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)

        self.assertEqual(reclaimed, 1)
        self.assertEqual(self._count(" add -A"), 1)
        self.assertLessEqual(self._count(" worktree prune"), 2)
        self.assertEqual(
            (self.repo / "agent.txt").read_text(), "agent work\n",
        )
        self.assertFalse(wt_dir.exists())
        self.assertFalse(
            GitWorktreeOps.branch_exists(self.repo, "kiss/wt-a"),
        )


class RaceDelayHookTest(unittest.TestCase):
    """The test-only race hook must never disturb production."""

    def setUp(self) -> None:
        self._saved = os.environ.get(_RACE_DELAY_ENV)

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop(_RACE_DELAY_ENV, None)
        else:
            os.environ[_RACE_DELAY_ENV] = self._saved

    def test_unset_variable_is_a_no_op(self) -> None:
        """Nothing sleeps when the variable is absent."""
        os.environ.pop(_RACE_DELAY_ENV, None)
        started = time.monotonic()
        _race_delay()
        self.assertLess(time.monotonic() - started, 0.01)

    def test_a_non_numeric_value_is_ignored(self) -> None:
        """A malformed value must not raise into a git operation."""
        os.environ[_RACE_DELAY_ENV] = "not-a-number"
        started = time.monotonic()
        _race_delay()
        self.assertLess(time.monotonic() - started, 0.01)

    def test_an_oversized_value_is_capped(self) -> None:
        """A stray large value can never stall a real run."""
        os.environ[_RACE_DELAY_ENV] = "60"
        started = time.monotonic()
        _race_delay()
        self.assertLess(time.monotonic() - started, 0.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
