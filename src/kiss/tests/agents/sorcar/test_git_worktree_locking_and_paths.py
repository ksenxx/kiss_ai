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
    _git_common_dir,
    _race_delay,
    repo_lock,
)
from kiss.tests.conftest import posix_only

_RACE_DELAY_ENV = "KISS_RACE_DELAY"


def _checked_out(repo: Path) -> set[str]:
    """The branches checked out in any worktree of *repo* (the sweep's in-use set)."""
    return {branch for _wt_dir, branch in GitWorktreeOps.registered_worktrees(repo)}


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
        self.assertNotIn("kiss/wt-a", _checked_out(self.repo))

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
        # Resolved: git reports macOS's /var/folders temp dirs as /private/var.
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_e_i3_")).resolve()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    @posix_only("Windows strips trailing spaces from directory names")
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
        self.assertEqual(_checked_out(repo), {"main", "kiss/wt-a"})

    @posix_only("Windows forbids control characters and trailing spaces in names")
    def test_repo_paths_keep_carriage_returns_and_trailing_spaces(self) -> None:
        """A root with a CR, or a separate git dir ending in a space, resolves to itself.

        ``rev-parse`` prints paths without ``-z``: translating its CR to
        LF named the neighbour ``carriage\\nrepo``, and stripping the
        separate git dir's trailing space put ``info/exclude`` in a
        fresh ``metadata`` directory next to the real ``metadata ``.
        """
        carriage = _make_repo(self.tmp / "carriage\rrepo")
        _make_repo(self.tmp / "carriage\nrepo")
        self.assertEqual(GitWorktreeOps.discover_repo(carriage), carriage)
        git_dir = self.tmp / "metadata "
        repo = self.tmp / "separate"
        repo.mkdir()
        subprocess.run(
            ["git", "init", "-b", "main", "--separate-git-dir", str(git_dir), str(repo)],
            capture_output=True, text=True, check=True,
        )
        (repo / "f.txt").write_text("a\n")
        _run("add", "-A", cwd=repo)
        _run("-c", "user.email=t@t.com", "-c", "user.name=T", "commit", "-m", "init", cwd=repo)
        wt_dir = self.tmp / "linked"
        _run("worktree", "add", "--detach", str(wt_dir), cwd=repo)
        for tree in (repo, wt_dir):
            common = _git_common_dir(tree)
            self.assertIsNotNone(common)
            assert common is not None
            self.assertEqual(common.resolve(), git_dir.resolve())
        GitWorktreeOps.ensure_excluded(repo)
        self.assertIn(".kiss-worktrees/\n", (git_dir / "info" / "exclude").read_text())
        self.assertFalse((self.tmp / "metadata").exists())

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
        self.assertEqual(_checked_out(repo), {"main"})


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

    def test_a_merge_never_overwrites_what_the_main_tree_ignores(self) -> None:
        """Ignored main-tree files at paths the branch starts tracking are kept.

        ``stash --include-untracked`` leaves ignored files where they are,
        and both ``merge --squash`` and ``cherry-pick`` silently overwrite
        one (or delete an ignored directory in the way of a file), so
        every merge mode is refused up front and the tree is left as it
        was; so is a branch switch of the main tree.
        """
        (self.repo / ".gitignore").write_text(".env\nbuild\nout/\nlocal.cfg\n*.log\nvendor/\n")
        (self.repo / "local.cfg.example").write_text("EXAMPLE=1\n" * 20)
        (self.repo / "mix").mkdir()
        (self.repo / "mix" / "tracked.txt").write_text("tracked\n")
        _run("add", "-A", cwd=self.repo)
        _run("commit", "-m", "ignore local files", cwd=self.repo)
        _run("checkout", "-b", "kiss/wt-a", cwd=self.repo)
        (self.repo / "f.txt").write_text("a-baseline\n")
        _run("commit", "-am", "baseline", cwd=self.repo)
        baseline = _run("rev-parse", "HEAD", cwd=self.repo).stdout.strip()
        (self.repo / ".gitignore").write_text("")
        (self.repo / ".env").write_text("EXAMPLE=1\n")
        (self.repo / "build").mkdir()
        (self.repo / "build" / "out.txt").write_text("built\n")
        (self.repo / "out").write_text("a file now\n")
        # A rename git detects (``R100``, not an addition) onto an ignored path,
        # and a file where a directory of tracked files held an ignored leaf.
        _run("mv", "local.cfg.example", "local.cfg", cwd=self.repo)
        _run("rm", "-q", "mix/tracked.txt", cwd=self.repo)
        (self.repo / "mix").write_text("a file now\n")
        (self.repo / "vendor").mkdir()
        (self.repo / "vendor" / "inner").write_text("ours\n")
        _run("add", "-A", cwd=self.repo)
        _run("commit", "-m", "track local files", cwd=self.repo)
        _run("checkout", "main", cwd=self.repo)
        (self.repo / ".env").write_text("SECRET=local\n")
        (self.repo / "build").write_text("ignored file\n")
        (self.repo / "out").mkdir()
        (self.repo / "out" / "cache.bin").write_text("ignored dir\n")
        (self.repo / "local.cfg").write_text("SECRET=renamed onto\n")
        (self.repo / "mix" / "debug.log").write_text("ignored leaf\n")
        _make_repo(self.repo / "vendor")  # a nested repository git does not look into
        (self.repo / "vendor" / "inner").write_text("theirs\n")
        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), "")  # nothing to stash
        self.assertEqual(
            GitWorktreeOps.ignored_collisions(self.repo, "kiss/wt-a", baseline),
            [".env", "build/out.txt", "local.cfg", "mix", "out", "vendor/inner"],
        )

        for mode in (baseline, None):
            with self.subTest(baseline=mode):
                result = GitWorktreeOps.squash_merge(self.repo, "kiss/wt-a", mode)
                self.assertEqual(result, MergeResult.CONFLICT)
                self.assertIsNone(GitWorktreeOps.begin_conflicted_merge(
                    self.repo, "kiss/wt-a", mode, "",
                ))
                self.assertEqual((self.repo / ".env").read_text(), "SECRET=local\n")
                self.assertEqual((self.repo / "build").read_text(), "ignored file\n")
                self.assertEqual(
                    (self.repo / "out" / "cache.bin").read_text(), "ignored dir\n",
                )
                self.assertEqual((self.repo / "local.cfg").read_text(), "SECRET=renamed onto\n")
                self.assertEqual((self.repo / "mix" / "debug.log").read_text(), "ignored leaf\n")
                self.assertEqual((self.repo / "vendor" / "inner").read_text(), "theirs\n")
                self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), "")
                self._assert_no_pick_in_progress()
        ok, err = GitWorktreeOps.checkout(self.repo, "kiss/wt-a")
        self.assertFalse(ok)
        self.assertIn(".env", err)
        self.assertEqual(GitWorktreeOps.current_branch(self.repo), "main")
        self.assertEqual((self.repo / ".env").read_text(), "SECRET=local\n")

    def test_a_merge_replaces_tracked_files_with_directories_and_back(self) -> None:
        """Only ignored content is in the way: a tracked shape the branch replaces reaches git.

        The directory ``shape/`` of tracked files becomes the file
        ``shape`` and the tracked file ``f.txt`` the directory ``f.txt/``;
        nothing the main tree ignores sits at either, so both merge modes
        apply the replacement instead of refusing it.
        """
        for mode in ("baseline", "legacy"):
            with self.subTest(mode=mode):
                repo = _make_repo(self.tmp / mode)
                (repo / "shape").mkdir()
                (repo / "shape" / "old.txt").write_text("old\n")
                _run("add", "-A", cwd=repo)
                _run("commit", "-m", "shapes", cwd=repo)
                _run("checkout", "-b", "kiss/wt-a", cwd=repo)
                (repo / "g.txt").write_text("g-baseline\n")
                _run("commit", "-am", "baseline", cwd=repo)
                baseline = _run("rev-parse", "HEAD", cwd=repo).stdout.strip()
                _run("rm", "-q", "shape/old.txt", "f.txt", cwd=repo)
                (repo / "shape").write_text("a file now\n")
                (repo / "f.txt").mkdir()
                (repo / "f.txt" / "new.txt").write_text("a directory now\n")
                _run("add", "-A", cwd=repo)
                _run("commit", "-m", "replace shapes", cwd=repo)
                _run("checkout", "main", cwd=repo)
                self.assertEqual(GitWorktreeOps.ignored_collisions(repo, "kiss/wt-a", baseline), [])
                # Without git's word on what is tracked (a corrupt index), every
                # shape in the way is refused, as before.
                index = (repo / ".git" / "index").read_bytes()
                (repo / ".git" / "index").write_bytes(b"garbage")
                self.assertEqual(
                    GitWorktreeOps.ignored_collisions(repo, "kiss/wt-a", baseline),
                    ["f.txt/new.txt", "shape"],
                )
                (repo / ".git" / "index").write_bytes(index)
                result = GitWorktreeOps.squash_merge(
                    repo, "kiss/wt-a", baseline if mode == "baseline" else None,
                )
                self.assertEqual(result, MergeResult.SUCCESS)
                self.assertEqual((repo / "shape").read_text(), "a file now\n")
                self.assertEqual((repo / "f.txt" / "new.txt").read_text(), "a directory now\n")
                self.assertEqual(GitWorktreeOps.status_porcelain(repo), "")

    def test_successful_baseline_merge_still_works(self) -> None:
        """The happy path is unaffected by the new cleanup."""
        baseline = self._make_branch()
        result = GitWorktreeOps.squash_merge_from_baseline(
            self.repo, "kiss/wt-a", baseline,
        )
        self.assertEqual(result, MergeResult.SUCCESS)
        self.assertEqual((self.repo / "g.txt").read_text(), "g-agent\n")


@posix_only("shebang git shim on PATH cannot shadow git.exe on Windows")
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
