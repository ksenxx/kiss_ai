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
* **I2** — the cleanup of a failed merge: what git applied is taken
  back, what the user saved meanwhile is left alone (up to the moment
  git itself writes the files), and no merge is attempted into a git
  operation of their own (a cherry-pick sequence, a merge, a rebase).
* **LOCKKEY** — ``repo_lock`` is one lock per repository, whichever of
  its checkouts names it.
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
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    MergeResult,
    _git,
    _git_common_dir,
    _race_delay,
    repo_lock,
)
from kiss.agents.sorcar.merge_conflict_resolver import resolve_merge_conflict
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.conftest import posix_only
from kiss.tests.lock_injection import inject
from kiss.tests.server.pause import Pause

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


def _launched_resolver(parent_agent: object, prompt: str, repo: Path) -> None:
    """Stand-in for the merge agent: record that it was launched on *repo*."""
    (repo / ".git" / "resolver-launched").touch()


def _settling_resolver(parent_agent: object, prompt: str, repo: Path) -> None:
    """Stand-in for the merge agent: settle the conflict in ``f.txt`` and stage it."""
    (repo / "f.txt").write_text("settled\n")
    _run("add", "f.txt", cwd=repo)


class SaveBeforeGitWrites:
    """A :class:`Pause` matcher on ``_git``: an editor saves *path* as git is asked to write files.

    Once, on the entry of the first git command that writes working
    files (``reset --merge`` / ``--hard``, ``read-tree -u``, ``restore``,
    ``rm``, ``checkout``): after every look the merge itself takes at
    the tree, so only git's own check of the file can see the save.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.saved = False

    def __call__(self, frame_locals: dict[str, Any]) -> bool:
        args = frame_locals.get("args", ())
        if not self.saved and (
            args[:2] in (("reset", "--merge"), ("reset", "--hard"))
            or (args[:1] == ("read-tree",) and "-u" in args)
            or {"restore", "rm", "checkout"} & set(args)
        ):
            self.saved = True
            with self.path.open("a") as file:
                file.write("saved since\n")
        return False


class _Merge(threading.Thread):
    """Squash-merges ``kiss/wt-a`` into *repo* and keeps whatever that raised."""

    def __init__(self, repo: Path) -> None:
        super().__init__(daemon=True)
        self.repo = repo
        self.raised: BaseException | None = None

    def run(self) -> None:
        try:
            GitWorktreeOps.squash_merge_branch(self.repo, "kiss/wt-a")
        except BaseException as exc:
            self.raised = exc


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

    def test_every_checkout_of_a_repository_takes_the_same_lock(self) -> None:
        """A linked worktree, and a checkout whose git dir is kept apart, name one lock."""
        self.assertIs(repo_lock(self.wt_dir), repo_lock(self.repo))
        # A ``.git`` file is read as git reads it. A carriage return in the git dir's name
        # is part of the name; the line ends after the path, however many, are not; the
        # path ends at a NUL.
        meta_name = "meta" if os.name == "nt" else "me\rta"
        apart, meta, linked = self.tmp / "apart", self.tmp / meta_name, self.tmp / "linked"
        unseen = self.tmp / "unseen"
        subprocess.run(
            ["git", "init", "-q", "-b", "main", f"--separate-git-dir={meta}", str(apart)],
            capture_output=True, check=True,
        )
        identity = ("-c", "user.name=T", "-c", "user.email=t@t.com")
        _run(*identity, "commit", "--allow-empty", "-m", "initial", cwd=apart)
        _run("worktree", "add", "-q", "-b", "side", str(linked), cwd=apart)
        _run("worktree", "add", "-q", "-b", "unseen", str(unseen), cwd=apart)
        with (linked / ".git").open("ab") as pointer:
            pointer.write(b"\r\n")
        pointer_path = unseen / ".git"
        named = pointer_path.read_bytes().removesuffix(b"\n")
        with pointer_path.open("r+b") as pointer:
            pointer.write(named + b"\0not read\n")
            pointer.truncate()
        lock = repo_lock(apart)
        self.assertIs(repo_lock(linked), lock)
        self.assertIs(repo_lock(meta), lock)
        self.assertIsNot(repo_lock(self.repo), lock)
        # Each path is locked as the repository git itself finds from it: a worktree's
        # own git dir, and a ``.git`` that is no git dir, which git looks past.
        stray = self.repo / ".kiss-worktrees" / "stray"
        (stray / ".git").mkdir(parents=True)
        for path in (self.repo / ".git" / "worktrees" / "kiss_wt-a", stray, linked, unseen):
            shared = subprocess.run(
                ["git", "-C", path, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                capture_output=True, check=True,
            ).stdout  # as bytes: read as text, the carriage return in the name is a line end
            self.assertIs(repo_lock(path), repo_lock(Path(os.fsdecode(shared)[:-1])), path)
        # So does a path below a checkout, as git looks upwards; one in no repository
        # (below a ``.git`` file that names no path, even) is locked by itself.
        self.assertIs(repo_lock(self.repo / "below" / "it"), repo_lock(self.repo))
        (self.tmp / ".git").write_bytes(b"gitdir: \0")
        self.assertIs(repo_lock(self.tmp / "gone"), repo_lock(self.tmp / "gone"))
        self.assertIsNot(repo_lock(self.tmp / "gone"), repo_lock(self.repo))
        # The repository is told from its files, not asked of a git that can fail (a
        # config it cannot parse): no checkout, locked before or not, takes another lock.
        (meta / "config").write_text("[broken\n")
        self.assertIs(repo_lock(linked), lock)
        self.assertIs(repo_lock(unseen), lock)

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
    """I2: a failed merge leaves neither a half-applied tree nor a reset user edit."""

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

    def _while_git_merges_f(self, script: str) -> None:
        """Make git run the ``sh`` *script*, in the repo root, when it merges ``f.txt``.

        A merge driver runs while git computes the merge, after the
        caller's look at the tree and before git's own: what the
        script writes is what an editor saves at that moment.  The
        script's exit status is the driver's verdict (non-zero: a
        conflict in ``f.txt``).
        """
        driver = self.tmp / "driver.sh"
        driver.write_text(script)
        _run("config", "merge.editor.driver", f"sh {shlex.quote(str(driver))}", cwd=self.repo)
        with (self.repo / ".git" / "info" / "attributes").open("a") as attributes:
            attributes.write("f.txt merge=editor\n")

    def _assert_only_edit(self, name: str, content: str, status: str = "M") -> None:
        """Assert the tree differs from HEAD by the *content* of *name* alone (unstaged: ``M``)."""
        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), f"{status} {name}")
        self.assertEqual((self.repo / name).read_text(), content)
        self._assert_no_pick_in_progress()
        self.assertFalse((self.repo / ".git" / "SQUASH_MSG").exists())

    @posix_only("the merge driver is a sh script")
    def test_an_edit_saved_while_git_merges_survives_the_refusal(self) -> None:
        """Git refuses to merge over a file saved after the caller's check: nothing is reset."""
        baseline = self._make_branch()
        self._while_git_merges_f("echo UNSAVED USER WORK > g.txt\n")

        result = GitWorktreeOps.squash_merge_from_baseline(self.repo, "kiss/wt-a", baseline)

        self.assertEqual(result, MergeResult.CONFLICT)
        self._assert_only_edit("g.txt", "UNSAVED USER WORK\n")
        # The conflict resolver's re-application is refused the same way.
        self.assertIsNone(GitWorktreeOps.begin_conflicted_merge(self.repo, "kiss/wt-a", baseline))
        self._assert_only_edit("g.txt", "UNSAVED USER WORK\n")

    @posix_only("the hook is a sh script")
    def test_an_edit_saved_as_git_squash_merges_survives_the_refusal(self) -> None:
        """The same for a worktree without a baseline (``git merge --squash``).

        Staged too (``M `` in the first status column): an index that
        changed is no proof that the merge applied anything.
        """
        self._make_branch()
        for stage, status in (("", "M"), ("git add g.txt", "M ")):
            with self.subTest(stage=stage):
                self._save_g_as_git_squash_merges(stage)

                result = GitWorktreeOps.squash_merge_branch(self.repo, "kiss/wt-a")

                self.assertEqual(result, MergeResult.CONFLICT)
                self._assert_only_edit("g.txt", "UNSAVED USER WORK\n", status)
                self.assertIsNone(
                    GitWorktreeOps.begin_conflicted_merge(self.repo, "kiss/wt-a", None),
                )
                self._assert_only_edit("g.txt", "UNSAVED USER WORK\n", status)
                _run("reset", "--hard", cwd=self.repo)
                (self.repo / ".git" / "saved").unlink()

    def _save_g_as_git_squash_merges(self, then: str) -> None:
        """Make an editor save ``g.txt`` and run the ``sh`` line *then*, once per merge.

        ``git merge`` records ORIG_HEAD before it looks at the tree: what
        the hook writes then (once, ``.git/saved``: ``git reset`` records
        ORIG_HEAD too) is a save right after the caller's check.
        """
        hook = self.repo / ".git" / "hooks" / "reference-transaction"
        hook.parent.mkdir(exist_ok=True)
        hook.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = prepared ] && [ ! -e .git/saved ] && grep -q " ORIG_HEAD$"; then\n'
            f"    : > .git/saved\n    echo UNSAVED USER WORK > g.txt\n    {then}\nfi\n"
        )
        hook.chmod(0o755)

    @posix_only("the hook is a sh script")
    def test_a_stop_landing_on_a_merge_git_never_applied_resets_nothing(self) -> None:
        """Unstaged, the index is as it was; staged, the file is not what a merge wrote."""
        self._make_branch()
        lingering = self.repo / ".git" / "lingering"
        for stage, status in (("", "M"), ("git add g.txt", "M ")):
            with self.subTest(stage=stage):
                self._save_g_as_git_squash_merges(f"{stage}\n    : > .git/lingering\n    sleep 0.3")
                merger = _Merge(self.repo)
                merger.start()
                while not lingering.exists():
                    self.assertTrue(merger.is_alive(), repr(merger.raised))
                    time.sleep(0.002)

                self.assertEqual(inject(merger), 1)  # the runner's Stop, while git lingers
                merger.join(timeout=30)

                self.assertIsInstance(merger.raised, KeyboardInterrupt)
                self._assert_only_edit("g.txt", "UNSAVED USER WORK\n", status)
                _run("reset", "--hard", cwd=self.repo)
                lingering.unlink()
                (self.repo / ".git" / "saved").unlink()

    @posix_only("the merge driver is a sh script")
    def test_a_conflicting_merge_is_taken_back_around_an_edit_saved_meanwhile(self) -> None:
        """The cleanup of an applied merge keeps an edit to a file the merge did not touch."""
        (self.repo / "u.txt").write_text("u\n")
        _run("add", "u.txt", cwd=self.repo)
        _run("commit", "-m", "a file no branch touches", cwd=self.repo)
        baseline = self._make_branch()
        (self.repo / "f.txt").write_text("a-main\n")  # a competing commit: the conflict is real
        _run("commit", "-am", "main moves on", cwd=self.repo)
        self._while_git_merges_f("echo saved meanwhile > u.txt\nexit 1\n")

        for mode in (baseline, None):
            with self.subTest(baseline=mode):
                result = GitWorktreeOps.squash_merge(self.repo, "kiss/wt-a", mode)

                self.assertEqual(result, MergeResult.CONFLICT)
                self._assert_only_edit("u.txt", "saved meanwhile\n")
                self.assertEqual((self.repo / "f.txt").read_text(), "a-main\n")
                self.assertEqual((self.repo / "g.txt").read_text(), "x\n")
                # No merge agent is run around the edit: its cleanup would reset it.
                wt = GitWorktree(self.repo, "kiss/wt-a", "main", self.tmp / "removed", mode)
                self.assertEqual(
                    resolve_merge_conflict(None, wt, None, None, run_agent=_launched_resolver),
                    MergeResult.CONFLICT,
                )
                self.assertFalse((self.repo / ".git" / "resolver-launched").exists())
                self._assert_only_edit("u.txt", "saved meanwhile\n")

    @posix_only("the smudge filter is a sh command")
    def test_a_file_saved_since_the_merge_wrote_it_is_never_reset(self) -> None:
        """Be it a merged file (``e.txt``) or a conflicted one (``f.txt``): the merge stays."""
        (self.repo / "e.txt").write_text("e\n")
        (self.repo / "u.txt").write_text("u\n")
        _run("add", "e.txt", "u.txt", cwd=self.repo)
        _run("commit", "-m", "a file the branch edits and one it does not", cwd=self.repo)
        baseline = self._make_branch()
        _run("checkout", "kiss/wt-a", cwd=self.repo)
        (self.repo / "e.txt").write_text("e-agent\n")
        _run("commit", "-am", "agent 3", cwd=self.repo)
        _run("checkout", "main", cwd=self.repo)
        (self.repo / "f.txt").write_text("a-main\n")
        _run("commit", "-am", "main moves on", cwd=self.repo)
        (self.repo / ".git" / "info" / "attributes").write_text("g.txt filter=late\n")

        for mode in (baseline, None):
            for saved in ("e.txt", "f.txt"):
                with self.subTest(baseline=mode, saved=saved):
                    # A smudge filter runs as git writes ``g.txt``, after it wrote
                    # ``e.txt`` and ``f.txt``: once per merge, an editor saves
                    # *saved* and a file the merge does not touch.
                    _run(
                        "config", "filter.late.smudge",
                        "sh -c 'cat; test -e .git/late || { : > .git/late; "
                        f"echo saved since >> {saved}; echo saved since > u.txt; }}'",
                        cwd=self.repo,
                    )

                    result = GitWorktreeOps.squash_merge(self.repo, "kiss/wt-a", mode)

                    self.assertEqual(result, MergeResult.CONFLICT)
                    # No merge agent is launched on top of it, nor its cleanup.
                    wt = GitWorktree(self.repo, "kiss/wt-a", "main", self.tmp / "removed", mode)
                    self.assertEqual(
                        resolve_merge_conflict(None, wt, None, None, run_agent=_launched_resolver),
                        MergeResult.CONFLICT,
                    )
                    self.assertFalse((self.repo / ".git" / "resolver-launched").exists())
                    self.assertTrue((self.repo / saved).read_text().endswith("saved since\n"))
                    self.assertEqual((self.repo / "u.txt").read_text(), "saved since\n")
                    self.assertEqual(GitWorktreeOps.unresolved_paths(self.repo), ["f.txt"])
                    _run("reset", "--hard", cwd=self.repo)
                    (self.repo / ".git" / "late").unlink()

    def test_a_file_saved_right_before_the_merge_is_taken_back_is_never_reset(self) -> None:
        """After every look the merge takes at it, be it conflicted (``f.txt``) or merged."""
        baseline = self._make_branch()
        (self.repo / "f.txt").write_text("a-main\n")
        _run("commit", "-am", "main moves on", cwd=self.repo)

        for mode in (baseline, None):
            for name in ("f.txt", "g.txt"):
                with self.subTest(baseline=mode, saved=name):
                    with Pause(_git.__code__, SaveBeforeGitWrites(self.repo / name)):
                        result = GitWorktreeOps.squash_merge(self.repo, "kiss/wt-a", mode)

                    self.assertEqual(result, MergeResult.CONFLICT)
                    self.assertTrue((self.repo / name).read_text().endswith("saved since\n"))
                    # The merge stays in place around the save, whole.
                    self.assertEqual(GitWorktreeOps.unresolved_paths(self.repo), ["f.txt"])
                    self.assertTrue((self.repo / "g.txt").read_text().startswith("g-agent\n"))
                    _run("reset", "--hard", cwd=self.repo)

    @posix_only("the filters are sh scripts that read the command line of their git")
    def test_a_file_git_cannot_compare_is_never_reset(self) -> None:
        """A look at the tree that fails says nothing about it: the merge stays.

        A smudge filter saves ``f.txt`` as git writes ``g.txt``; from then
        on the required clean filter of ``f.txt`` fails every git command
        that reads the file, but the listing of the unmerged paths.
        """
        baseline = self._make_branch()
        (self.repo / "f.txt").write_text("a-main\n")
        _run("commit", "-am", "main moves on", cwd=self.repo)
        clean = self.tmp / "clean.sh"
        clean.write_text(
            'if [ -e .git/late ] && ! ps -o args= -p "$PPID" | grep -q -e --diff-filter=U; then\n'
            "    exit 1\nfi\nexec cat\n"
        )
        save = "sh -c 'cat; test -e .git/late || { : > .git/late; echo saved since >> f.txt; }'"
        for key, value in (
            ("late.smudge", save),
            ("broken.clean", f"exec sh {shlex.quote(str(clean))}"),
            ("broken.smudge", "cat"),
            ("broken.required", "true"),
        ):
            _run("config", f"filter.{key}", value, cwd=self.repo)
        attributes = self.repo / ".git" / "info" / "attributes"

        for mode in (baseline, None):
            with self.subTest(baseline=mode):
                attributes.write_text("f.txt filter=broken\ng.txt filter=late\n")

                result = GitWorktreeOps.squash_merge(self.repo, "kiss/wt-a", mode)

                self.assertEqual(result, MergeResult.CONFLICT)
                self.assertTrue((self.repo / "f.txt").read_text().endswith("saved since\n"))
                attributes.write_text("")  # no filter runs for the cleanup
                _run("reset", "--hard", cwd=self.repo)
                (self.repo / ".git" / "late").unlink()

    @posix_only("the hook is a sh script")
    def test_a_rollback_git_cannot_finish_is_a_conflict_no_merge_agent_is_run_on(self) -> None:
        """The hook that rejects the commit breaks the checkout of ``f.txt``: git stops half-way.

        The tree it leaves is reported as a conflict, not as a merge
        that was taken back, with what git failed at and what is left;
        no merge agent applies the branch to it again; nor is the
        agent's own rejected commit reported as taken back.
        """
        baseline = self._make_branch()
        hook = self.repo / ".git" / "hooks" / "pre-commit"
        hook.parent.mkdir(exist_ok=True)
        hook.write_text(
            "#!/bin/sh\n"
            "echo 'f.txt filter=gone' > .git/info/attributes\n"
            "git config filter.gone.required true\n"
            "git config filter.gone.clean cat\n"
            "git config filter.gone.smudge false\n"
            "exit 1\n"
        )
        hook.chmod(0o755)

        for mode in (baseline, None):
            with self.subTest(baseline=mode):
                with self.assertLogs("kiss.agents.sorcar.git_worktree", "WARNING") as logs:
                    result = GitWorktreeOps.squash_merge(self.repo, "kiss/wt-a", mode)

                self.assertEqual(result, MergeResult.CONFLICT)
                left = GitWorktreeOps.status_porcelain(self.repo)
                self.assertNotEqual(left, "")
                self.assertIn("smudge filter gone failed", "\n".join(logs.output))
                self.assertIn(left, "\n".join(logs.output))
                wt = GitWorktree(self.repo, "kiss/wt-a", "main", self.tmp / "removed", mode)
                self.assertEqual(
                    resolve_merge_conflict(None, wt, None, None, run_agent=_launched_resolver),
                    MergeResult.CONFLICT,
                )
                self.assertFalse((self.repo / ".git" / "resolver-launched").exists())
                self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), left)
                (self.repo / ".git" / "info" / "attributes").write_text("")
                _run("reset", "--hard", cwd=self.repo)

        (self.repo / "f.txt").write_text("a-main\n")
        _run("commit", "--no-verify", "-am", "main moves on", cwd=self.repo)
        wt = GitWorktree(self.repo, "kiss/wt-a", "main", self.tmp / "removed", baseline)
        self.assertEqual(
            resolve_merge_conflict(None, wt, None, None, run_agent=_settling_resolver),
            MergeResult.CONFLICT,
        )

    @posix_only("the hook is a sh script")
    def test_a_rejected_commit_resets_no_file_saved_since_the_merge_wrote_it(self) -> None:
        """The hook that rejects the commit stands for the editor too: it saves a file first.

        A file the merge did not touch keeps the save while the merge is
        taken back around it; one the merge wrote keeps it as well, and
        the merge stays in place.
        """
        (self.repo / "u.txt").write_text("u\n")
        _run("add", "u.txt", cwd=self.repo)
        _run("commit", "-m", "a file the task leaves alone", cwd=self.repo)
        baseline = self._make_branch()
        hook = self.repo / ".git" / "hooks" / "pre-commit"
        hook.parent.mkdir(exist_ok=True)

        for mode in (baseline, None):
            for saved, result in (
                ("u.txt", MergeResult.MERGE_FAILED), ("g.txt", MergeResult.CONFLICT),
            ):
                hook.write_text(f"#!/bin/sh\necho saved since > {saved}\nexit 1\n")
                hook.chmod(0o755)
                with self.subTest(saved=saved, baseline=mode):
                    self.assertEqual(
                        GitWorktreeOps.squash_merge(self.repo, "kiss/wt-a", mode), result,
                    )
                    self.assertEqual((self.repo / saved).read_text(), "saved since\n")
                    self.assertEqual(
                        GitWorktreeOps.has_staged_changes(self.repo),
                        result == MergeResult.CONFLICT,
                    )
                    _run("reset", "--hard", cwd=self.repo)
            # The squash merge that follows is no fast-forward.
            (self.repo / "u.txt").write_text(f"main moved on after {mode}\n")
            _run("commit", "--no-verify", "-am", "main moves on", cwd=self.repo)

    @posix_only("the hook is a sh script")
    def test_a_file_saved_where_the_task_deleted_one_survives_a_rejected_commit(self) -> None:
        """Whether git ignores the file or not: it stays, and the merge around it.

        Taking the merge back means writing the deleted file again, and
        git writes over a file it ignores without a word.  The save is
        the rejecting hook's, or lands as git is asked to write.
        """
        (self.repo / ".gitignore").write_text("*.log\n")
        for name in ("d.txt", "d.log"):
            (self.repo / name).write_text("tracked\n")
        _run("add", "-f", "-A", cwd=self.repo)
        _run("commit", "-m", "two files, one of them ignored", cwd=self.repo)
        baseline = self._make_branch()
        _run("checkout", "kiss/wt-a", cwd=self.repo)
        _run("rm", "-q", "d.txt", "d.log", cwd=self.repo)
        _run("commit", "-m", "agent deletes both", cwd=self.repo)
        _run("checkout", "main", cwd=self.repo)
        hook = self.repo / ".git" / "hooks" / "pre-commit"
        hook.parent.mkdir(exist_ok=True)

        for mode in (baseline, None):
            for name in ("d.txt", "d.log"):
                for late in (False, True):
                    saves = "" if late else f"echo saved since > {name}\n"
                    hook.write_text(f"#!/bin/sh\n{saves}exit 1\n")
                    hook.chmod(0o755)
                    save = SaveBeforeGitWrites(self.repo / name)
                    save.saved = not late
                    with self.subTest(baseline=mode, saved=name, late=late):
                        with Pause(_git.__code__, save):
                            result = GitWorktreeOps.squash_merge(self.repo, "kiss/wt-a", mode)

                        self.assertEqual(result, MergeResult.CONFLICT)
                        self.assertEqual((self.repo / name).read_text(), "saved since\n")
                        self.assertTrue(GitWorktreeOps.has_staged_changes(self.repo))
                    _run("reset", "--hard", cwd=self.repo)

    @posix_only("the hook is a sh script")
    def test_a_rejected_commit_takes_replaced_shapes_back_but_for_a_file_saved_inside(self) -> None:
        """A directory the merge made of a file goes again, unless a file git ignores is in it.

        The task turns the file ``f.txt`` into a directory and the
        directory ``shape/`` into a file; taken back, git removes the
        directory ``f.txt/`` with whatever it ignores in there.
        """
        (self.repo / ".gitignore").write_text("*.log\n")
        (self.repo / "shape").mkdir()
        (self.repo / "shape" / "old.txt").write_text("old\n")
        _run("add", "-A", cwd=self.repo)
        _run("commit", "-m", "shapes", cwd=self.repo)
        _run("checkout", "-b", "kiss/wt-a", cwd=self.repo)
        (self.repo / "g.txt").write_text("g-baseline\n")
        _run("commit", "-am", "baseline", cwd=self.repo)
        baseline = _run("rev-parse", "HEAD", cwd=self.repo).stdout.strip()
        _run("rm", "-q", "shape/old.txt", "f.txt", cwd=self.repo)
        (self.repo / "shape").write_text("a file now\n")
        (self.repo / "f.txt").mkdir()
        (self.repo / "f.txt" / "new.txt").write_text("a directory now\n")
        _run("add", "-A", cwd=self.repo)
        _run("commit", "-m", "replace shapes", cwd=self.repo)
        _run("checkout", "main", cwd=self.repo)
        hook = self.repo / ".git" / "hooks" / "pre-commit"
        hook.parent.mkdir(exist_ok=True)

        for mode in (baseline, None):
            for saves, expected in (
                ("", MergeResult.MERGE_FAILED),
                ("echo saved since > f.txt/notes.log\n", MergeResult.CONFLICT),
            ):
                hook.write_text(f"#!/bin/sh\n{saves}exit 1\n")
                hook.chmod(0o755)
                with self.subTest(baseline=mode, saves=saves):
                    result = GitWorktreeOps.squash_merge(self.repo, "kiss/wt-a", mode)

                    self.assertEqual(result, expected)
                    if saves:
                        self.assertEqual(
                            (self.repo / "f.txt" / "notes.log").read_text(), "saved since\n",
                        )
                        self.assertTrue(GitWorktreeOps.has_staged_changes(self.repo))
                    else:
                        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), "")
                        self.assertEqual((self.repo / "f.txt").read_text(), "a\n")
                        self.assertEqual((self.repo / "shape" / "old.txt").read_text(), "old\n")
                _run("reset", "--hard", cwd=self.repo)

    @posix_only("the hook is a sh script")
    def test_a_submodule_the_task_removed_is_back_after_a_rejected_commit(self) -> None:
        """Its checkout is still there, and git's own rule for it holds: nothing is in the way."""
        sub = _make_repo(self.tmp / "sub")
        self.assertEqual(
            _git(
                "-c", "protocol.file.allow=always", "submodule", "add", str(sub), "sub",
                cwd=self.repo,
            ).returncode,
            0,
        )
        _run("commit", "-m", "a submodule", cwd=self.repo)
        _run("checkout", "-b", "kiss/wt-a", cwd=self.repo)
        _run("rm", "-q", "--cached", "sub", cwd=self.repo)
        _run("commit", "-m", "agent removes the submodule", cwd=self.repo)
        _run("checkout", "main", cwd=self.repo)
        hook = self.repo / ".git" / "hooks" / "pre-commit"
        hook.parent.mkdir(exist_ok=True)
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        result = GitWorktreeOps.squash_merge_branch(self.repo, "kiss/wt-a")

        self.assertEqual(result, MergeResult.MERGE_FAILED)
        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), "")
        self.assertEqual((self.repo / "sub" / "f.txt").read_text(), "a\n")

    @posix_only("the hook is a sh script")
    def test_a_save_staged_since_the_merge_wrote_the_index_is_never_reset(self) -> None:
        """Taking the merge back unstages what the user staged since and leaves its files alone.

        When the index cannot be reset (a git command of the user's holds
        it), the merge stays in place, whole.
        """
        (self.repo / "u.txt").write_text("u\n")
        _run("add", "u.txt", cwd=self.repo)
        _run("commit", "-m", "a file no branch touches", cwd=self.repo)
        baseline = self._make_branch()
        (self.repo / "f.txt").write_text("a-main\n")
        _run("commit", "-am", "main moves on", cwd=self.repo)
        hook = self.repo / ".git" / "hooks" / "post-index-change"
        hook.parent.mkdir(exist_ok=True)
        for users_git, status, kept in (
            (
                "echo staged since > u.txt; echo new > new.txt; git add u.txt new.txt",
                "M u.txt\n?? new.txt",
                {"u.txt": "staged since\n", "new.txt": "new\n"},
            ),
            (": > .git/index.lock", "UU f.txt\nM  g.txt", {}),
        ):
            # git runs this hook once it has written the index: after the one
            # that holds the merge (``AUTO_MERGE``), the user's git runs, once.
            hook.write_text(
                "#!/bin/sh\n"
                "if [ -e .git/AUTO_MERGE ] && [ ! -e .git/saved ]; then\n"
                f"    : > .git/saved\n    {users_git}\nfi\n"
            )
            hook.chmod(0o755)
            for mode in (baseline, None):
                with self.subTest(users_git=users_git, baseline=mode):
                    result = GitWorktreeOps.squash_merge(self.repo, "kiss/wt-a", mode)

                    self.assertEqual(result, MergeResult.CONFLICT)
                    (self.repo / ".git" / "index.lock").unlink(missing_ok=True)
                    self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), status)
                    for name, content in kept.items():
                        self.assertEqual((self.repo / name).read_text(), content)
                    _run("reset", "--hard", cwd=self.repo)
                    _run("clean", "-fdq", cwd=self.repo)
                    (self.repo / ".git" / "saved").unlink()

    def test_a_submodule_the_merge_moved_is_no_save(self) -> None:
        """Its checkout lags the index after any merge, and no reset writes it: taken back."""
        sub = _make_repo(self.tmp / "sub")
        self.assertEqual(
            _git(
                "-c", "protocol.file.allow=always", "submodule", "add", str(sub), "sub",
                cwd=self.repo,
            ).returncode,
            0,
        )
        _run("commit", "-m", "a submodule", cwd=self.repo)
        _run("checkout", "-b", "kiss/wt-a", cwd=self.repo)
        # The clone ``submodule add`` made has no identity of its own: a host
        # without a global one (a fresh CI or VM user) cannot commit in it.
        _run("config", "user.email", "t@t.com", cwd=self.repo / "sub")
        _run("config", "user.name", "T", cwd=self.repo / "sub")
        (self.repo / "sub" / "f.txt").write_text("moved\n")
        _run("commit", "-am", "the submodule moves on", cwd=self.repo / "sub")
        (self.repo / "f.txt").write_text("a-agent\n")
        _run("commit", "-am", "agent moves the submodule and edits f", cwd=self.repo)
        _run("checkout", "main", cwd=self.repo)
        _run("checkout", "HEAD~1", cwd=self.repo / "sub")
        (self.repo / "f.txt").write_text("a-main\n")
        _run("commit", "-am", "main moves on", cwd=self.repo)

        result = GitWorktreeOps.squash_merge_branch(self.repo, "kiss/wt-a")

        self.assertEqual(result, MergeResult.CONFLICT)
        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), "")
        self.assertEqual(
            GitWorktreeOps.begin_conflicted_merge(self.repo, "kiss/wt-a", None), ["f.txt"],
        )

    def test_a_conflict_is_taken_back_whatever_strategy_the_user_merges_with(self) -> None:
        """Only ``ort`` writes the ``AUTO_MERGE`` that tells a save from what the merge wrote."""
        self._make_branch()
        (self.repo / "f.txt").write_text("a-main\n")
        _run("commit", "-am", "main moves on", cwd=self.repo)
        _run("config", "pull.twohead", "resolve", cwd=self.repo)

        result = GitWorktreeOps.squash_merge_branch(self.repo, "kiss/wt-a")

        self.assertEqual(result, MergeResult.CONFLICT)
        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), "")

    def test_a_resolution_rerere_would_replay_stays_a_conflict(self) -> None:
        """Replayed and staged (``rerere.autoupdate``), git fails with a merge and no conflict."""
        baseline = self._make_branch()
        (self.repo / "f.txt").write_text("a-main\n")
        _run("commit", "-am", "main moves on", cwd=self.repo)
        _run("config", "rerere.enabled", "true", cwd=self.repo)
        _run("config", "rerere.autoupdate", "true", cwd=self.repo)
        # The user resolved this very conflict once, and rerere recorded it.
        _git("merge", "--squash", "kiss/wt-a", cwd=self.repo)
        (self.repo / "f.txt").write_text("resolved once\n")
        _run("commit", "-am", "resolved once", cwd=self.repo)
        _run("reset", "--hard", "HEAD~1", cwd=self.repo)

        for mode in (baseline, None):
            with self.subTest(baseline=mode):
                result = GitWorktreeOps.squash_merge(self.repo, "kiss/wt-a", mode)

                self.assertEqual(result, MergeResult.CONFLICT)
                self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), "")
                # The resolver gets the conflict as the merge wrote it.
                self.assertEqual(
                    GitWorktreeOps.begin_conflicted_merge(self.repo, "kiss/wt-a", mode),
                    ["f.txt"],
                )
                self.assertTrue((self.repo / "f.txt").read_text().startswith("<<<<<<<"))
                _run("reset", "--hard", cwd=self.repo)

    def test_a_conflicting_merge_leaves_the_users_cherry_pick_sequence_alone(self) -> None:
        """Neither the merge nor its resolver touches a sequence the user is in the middle of."""
        _run("checkout", "-b", "side", cwd=self.repo)
        (self.repo / "g.txt").write_text("side 1\n")
        _run("commit", "-am", "side 1", cwd=self.repo)
        (self.repo / "h.txt").write_text("side 2\n")
        _run("add", "h.txt", cwd=self.repo)
        _run("commit", "-m", "side 2", cwd=self.repo)
        _run("checkout", "main", cwd=self.repo)
        baseline = self._make_branch()
        (self.repo / "f.txt").write_text("a-main\n")
        (self.repo / "g.txt").write_text("g-main\n")
        _run("commit", "-am", "main moves on", cwd=self.repo)
        # The user's pick of ``side 1`` stops on g.txt; they settle it by
        # hand and commit, with ``side 2`` still to come.
        picked = subprocess.run(
            ["git", "-C", str(self.repo), "cherry-pick", "side~1", "side"], capture_output=True,
        )
        self.assertNotEqual(picked.returncode, 0)
        (self.repo / "g.txt").write_text("settled by hand\n")
        _run("commit", "-am", "side 1, settled by hand", cwd=self.repo)
        head = _run("rev-parse", "HEAD", cwd=self.repo).stdout
        self.assertTrue((self.repo / ".git" / "sequencer").is_dir())

        result = GitWorktreeOps.squash_merge_from_baseline(self.repo, "kiss/wt-a", baseline)

        self.assertEqual(result, MergeResult.CONFLICT)
        # Nor is the conflict handed to the merge agent: the cleanup after it
        # (``cherry-pick --quit``) would forget the pick still to come.
        wt = GitWorktree(self.repo, "kiss/wt-a", "main", self.tmp / "removed", baseline)
        resolved = resolve_merge_conflict(None, wt, None, None, run_agent=_launched_resolver)
        self.assertEqual(resolved, MergeResult.CONFLICT)
        self.assertFalse((self.repo / ".git" / "resolver-launched").exists())
        self.assertEqual(_run("rev-parse", "HEAD", cwd=self.repo).stdout, head)
        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), "")
        _run("cherry-pick", "--continue", cwd=self.repo)
        self.assertEqual((self.repo / "h.txt").read_text(), "side 2\n")

    def test_the_users_merge_in_progress_is_not_stashed_away(self) -> None:
        """``git stash`` resets the tree and ``MERGE_HEAD`` with it: no merge would be left."""
        _run("checkout", "-b", "other", cwd=self.repo)
        (self.repo / "g.txt").write_text("g-other\n")
        _run("commit", "-am", "other", cwd=self.repo)
        _run("checkout", "main", cwd=self.repo)
        (self.repo / "f.txt").write_text("a-main\n")
        _run("commit", "-am", "main moves on", cwd=self.repo)
        _run("merge", "--no-commit", "other", cwd=self.repo)
        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), "M  g.txt")

        self.assertIsNone(GitWorktreeOps.stash_if_dirty(self.repo))

        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), "M  g.txt")
        self.assertTrue((self.repo / ".git" / "MERGE_HEAD").exists())
        self.assertEqual(_run("stash", "list", cwd=self.repo).stdout, "")

    @posix_only("the sequence editor is a sed command")
    def test_a_rebase_the_user_paused_is_not_checked_out_of(self) -> None:
        """Clean and with ``HEAD`` detached: nothing to stash, and a checkout would end it."""
        baseline = self._make_branch()
        _run("branch", "upstream", cwd=self.repo)
        for branch, name in (("main", "mine.txt"), ("upstream", "theirs.txt")):
            _run("checkout", branch, cwd=self.repo)
            (self.repo / name).write_text("new\n")
            _run("add", name, cwd=self.repo)
            _run("commit", "-m", f"{branch} moves on", cwd=self.repo)
        _run("checkout", "main", cwd=self.repo)
        # ``rebase -i`` stops after the first commit for the user to edit it.
        subprocess.run(
            ["git", "-C", str(self.repo), "rebase", "-i", "upstream"],
            env={**os.environ, "GIT_SEQUENCE_EDITOR": "sed -i.bak 1s/^pick/edit/"},
            capture_output=True, check=True,
        )
        head = _run("rev-parse", "HEAD", cwd=self.repo).stdout
        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), "")
        wt = GitWorktree(self.repo, "kiss/wt-a", "main", self.tmp / "removed", baseline)

        result, *_ = SorcarAgent("merge")._do_merge(wt)

        self.assertEqual(result, MergeResult.CHECKOUT_FAILED)
        self.assertEqual(_run("rev-parse", "HEAD", cwd=self.repo).stdout, head)
        self.assertIsNone(GitWorktreeOps.current_branch(self.repo))
        subprocess.run(
            ["git", "-C", str(self.repo), "rebase", "--continue"],
            env={**os.environ, "GIT_EDITOR": "true"}, capture_output=True, check=True,
        )
        self.assertEqual(GitWorktreeOps.current_branch(self.repo), "main")
        self.assertEqual((self.repo / "theirs.txt").read_text(), "new\n")
        self.assertEqual((self.repo / "mine.txt").read_text(), "new\n")

    def test_a_merge_onto_an_index_with_unmerged_entries_is_not_attempted(self) -> None:
        """The user's own conflict in the main tree is no base for a merge and is left as is."""
        baseline = self._make_branch()
        _run("checkout", "-b", "other", cwd=self.repo)
        (self.repo / "g.txt").write_text("g-other\n")
        _run("commit", "-am", "other", cwd=self.repo)
        _run("checkout", "main", cwd=self.repo)
        (self.repo / "g.txt").write_text("g-main\n")
        _run("commit", "-am", "main moves on", cwd=self.repo)
        merged = subprocess.run(
            ["git", "-C", str(self.repo), "merge", "other"], capture_output=True,
        )
        self.assertNotEqual(merged.returncode, 0)
        before = GitWorktreeOps.status_porcelain(self.repo)
        self.assertEqual(before, "UU g.txt")
        conflict = (self.repo / "g.txt").read_text()

        for mode in (baseline, None):
            with self.subTest(baseline=mode):
                self.assertEqual(
                    GitWorktreeOps.squash_merge(self.repo, "kiss/wt-a", mode), MergeResult.CONFLICT,
                )
                self.assertIsNone(
                    GitWorktreeOps.begin_conflicted_merge(self.repo, "kiss/wt-a", mode),
                )
                self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), before)
                self.assertEqual((self.repo / "g.txt").read_text(), conflict)
                self.assertTrue((self.repo / ".git" / "MERGE_HEAD").exists())

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
                self.assertIsNone(
                    GitWorktreeOps.begin_conflicted_merge(self.repo, "kiss/wt-a", mode),
                )
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
