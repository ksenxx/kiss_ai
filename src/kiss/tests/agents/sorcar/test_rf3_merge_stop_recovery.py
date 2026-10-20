# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop anywhere in ``commit_and_merge`` leaves the user's repo recoverable.

Review RF3 demonstrated five ways the merge and the worktree cleanup
could lose or strand work.  Each class here reproduces one against a
real repository, with the runner's Stop delivered as a real
asynchronous ``KeyboardInterrupt`` (``PyThreadState_SetAsyncExc``):

* a ``git status`` that cannot run must keep the worktree, not read as
  "everything committed" (:class:`TestCleanupStatusFailure`);
* a git child (a commit behind a slow hook) must not outlive the Stop
  and land later (:class:`TestStopKillsGitChild`);
* a Stop right after ``git stash push`` must still pop the stash
  (:class:`TestStopDuringStashCreation`);
* the no-baseline squash path must undo a half-applied merge
  (:class:`TestStopDuringSquashApply`);
* retiring the worktree must keep the ``git stash pop`` instructions of
  an interrupted merge (:class:`TestRetirementKeepsRecoveryWarning`).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _git
from kiss.agents.sorcar.sorcar_agent import _WorktreeCleanupOutcome
from kiss.tests.agents.sorcar.test_audit1020_merge_stop_stash import (
    USER_NOTES,
    _make_repo,
    _MergeThread,
)
from kiss.tests.conftest import posix_only
from kiss.tests.lock_injection import inject

AGENT_EDIT = "a\nB-agent\nc\n"
_SH_HOOKS = posix_only("the lingering git hooks are sh scripts")


def _install_hook(repo: Path, hooks: Path, name: str, body: str) -> None:
    """Install hook *name* with shell *body* for *repo* via ``core.hooksPath``."""
    hooks.mkdir(exist_ok=True)
    hook = hooks / name
    hook.write_text(f"#!/bin/sh\n{body}\n")
    hook.chmod(0o755)
    subprocess.run(
        ["git", "-C", str(repo), "config", "core.hooksPath", str(hooks)],
        check=True,
    )


class _MergeStopHarness:
    """A repo, an agent whose worktree holds one committed edit, and a Stop injector."""

    def setup_method(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.repo = _make_repo(self.tmpdir / "repo")
        self.agent = ChatSorcarAgent("test")
        self.agent.work_dir = str(self.repo)
        assert self.agent.create_worktree().startswith("Created worktree ")
        wt = self.agent._wt
        assert wt is not None
        self.wt = wt
        (wt.wt_dir / "f.txt").write_text(AGENT_EDIT)
        assert GitWorktreeOps.commit_all(wt.wt_dir, "agent work")
        self.ready = self.tmpdir / "ready"
        self.hooks = self.tmpdir / "hooks"

    def teardown_method(self) -> None:
        if self.agent._wt is not None:
            try:
                self.agent.discard()
            except Exception:
                pass
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _merge_stopped_when(self, moment, *, before_stop=None) -> None:
        """Run the merge on a thread and inject the Stop once *moment()* is true.

        Args:
            moment: Polled every 0.5 ms; the Stop is delivered when it
                returns True.
            before_stop: Optional callable run just before the Stop.
        """
        merger = _MergeThread(self.agent)
        merger.start()
        deadline = time.monotonic() + 30
        while not moment():
            assert time.monotonic() < deadline, "the awaited merge step never happened"
            assert merger.is_alive(), f"merge ended early: {merger.raised!r}"
            time.sleep(0.0005)
        if before_stop is not None:
            before_stop()
        assert inject(merger) == 1
        merger.join(timeout=30)
        assert not merger.is_alive()
        assert isinstance(merger.raised, KeyboardInterrupt), repr(merger.raised)

    def _assert_main_untouched(self) -> None:
        """Main holds exactly the user's untracked notes, no stash, no squash state."""
        assert GitWorktreeOps.status_porcelain(self.repo) == "?? notes.txt"
        assert (self.repo / "notes.txt").read_text() == USER_NOTES
        assert (self.repo / "f.txt").read_text() == "a\nb\nc\n"
        assert not (self.repo / ".git" / "SQUASH_MSG").exists()
        assert _git("stash", "list", cwd=self.repo).stdout == ""
        assert _git("rev-parse", "--verify", self.wt.branch, cwd=self.repo).returncode == 0


class TestCleanupStatusFailure(_MergeStopHarness):
    """``git status`` failing during cleanup keeps the worktree and its only copy."""

    def test_unverifiable_worktree_is_preserved(self) -> None:
        only_copy = self.wt.wt_dir / "only-copy.txt"
        only_copy.write_text("exists nowhere else\n")
        index = Path(_git("rev-parse", "--git-path", "index", cwd=self.wt.wt_dir).stdout.strip())
        lock = (self.wt.wt_dir / index).with_name("index.lock")
        lock.touch()  # every commit fails, as if another git were running
        _git("config", "status.relativePaths", "not-a-bool", cwd=self.repo)
        try:
            assert "git status failed" in GitWorktreeOps.status_porcelain(self.wt.wt_dir)

            assert self.agent._retire_previous_worktree()

            assert (
                self.agent._last_preserve_outcome is _WorktreeCleanupOutcome.PRESERVED_COMMIT_FAILED
            )
            assert only_copy.read_text() == "exists nowhere else\n"
            warning = self.agent._merge_conflict_warning or ""
            assert str(self.wt.wt_dir) in warning and "git status failed" in warning
        finally:
            lock.unlink()
            _git("config", "--unset", "status.relativePaths", cwd=self.repo)


@_SH_HOOKS
class TestStopKillsGitChild(_MergeStopHarness):
    """A Stop during the squash commit's hook ends git; the commit never lands."""

    def test_commit_behind_slow_hook_does_not_land_later(self) -> None:
        (self.repo / "notes.txt").write_text(USER_NOTES)
        progress = self.tmpdir / "progress"
        release = self.tmpdir / "release"
        # The hook reports it is alive every 20 ms until released (or
        # 6 s pass), so a hook still running after the Stop is visible.
        # Each report also goes to git's stdout: that wakes the thread
        # waiting on git's pipes, which is the only place an
        # asynchronous ``KeyboardInterrupt`` can land while git runs.
        _install_hook(
            self.repo,
            self.hooks,
            "pre-commit",
            f"touch {self.ready}\ni=0\n"
            f"while [ ! -e {release} ] && [ $i -lt 300 ]; do"
            f" echo tick; echo tick >> {progress}; sleep 0.02; i=$((i+1)); done\nexit 0",
        )
        head_before = _git("rev-parse", "HEAD", cwd=self.repo).stdout

        self._merge_stopped_when(self.ready.exists)

        time.sleep(0.1)
        size = progress.stat().st_size
        time.sleep(0.3)
        assert progress.stat().st_size == size, "the commit's hook outlived the Stop"
        release.touch()
        time.sleep(0.3)
        assert _git("rev-parse", "HEAD", cwd=self.repo).stdout == head_before
        assert not (self.repo / ".git" / "index.lock").exists()
        self._assert_main_untouched()
        assert self.agent._merge_conflict_warning is None


class TestStopDuringStashCreation(_MergeStopHarness):
    """A Stop as ``git stash push`` finishes still gives the user's notes back."""

    def test_stash_taken_by_an_interrupted_push_is_popped(self) -> None:
        notes = self.repo / "notes.txt"
        notes.write_text(USER_NOTES)

        # ``git stash push -u`` removes the untracked notes last; the
        # Stop lands while that push is exiting or in the ref read
        # right after it, before ``stash_if_dirty`` could return True.
        self._merge_stopped_when(lambda: not notes.exists())

        self._assert_main_untouched()
        assert self.agent._merge_conflict_warning is None


@_SH_HOOKS
class TestStopDuringSquashApply(_MergeStopHarness):
    """A Stop between ``merge --squash`` and its commit leaves main's index clean."""

    def test_no_baseline_squash_is_rolled_back(self) -> None:
        assert self.wt.baseline_commit is None  # main was clean at create_worktree
        (self.repo / "notes.txt").write_text(USER_NOTES)
        # Fires when ``merge --squash`` has written the index with the
        # agent's edit staged, and lingers there.
        _install_hook(
            self.repo,
            self.hooks,
            "post-index-change",
            'case "$(ps -o args= -p $PPID)" in'
            f" *merge*--squash*) touch {self.ready}; sleep 0.5;; esac",
        )

        self._merge_stopped_when(self.ready.exists)

        self._assert_main_untouched()
        assert self.agent._merge_conflict_warning is None


@_SH_HOOKS
class TestRetirementKeepsRecoveryWarning(_MergeStopHarness):
    """Retiring the worktree after a failed stash pop keeps the recovery hint."""

    def test_stash_pop_instructions_survive_retirement(self) -> None:
        (self.repo / "notes.txt").write_text(USER_NOTES)
        _install_hook(
            self.repo,
            self.hooks,
            "pre-commit",
            f"touch {self.ready}\nsleep 0.5\nexit 0",
        )

        # A file appearing on main while the merge runs makes the stash
        # pop fail, so the user is owed the ``git stash pop`` warning.
        def intrude() -> None:
            (self.repo / "notes.txt").write_text("intruder\n")

        self._merge_stopped_when(self.ready.exists, before_stop=intrude)
        assert "kiss: auto-stash before merge" in _git("stash", "list", cwd=self.repo).stdout
        warning = self.agent._merge_conflict_warning or ""
        assert "git stash pop" in warning

        assert self.agent._retire_previous_worktree()

        kept = self.agent._merge_conflict_warning or ""
        assert "git stash pop" in kept
        assert f"committed to branch '{self.wt.branch}'" in kept
