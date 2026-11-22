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
  (:class:`TestStopDuringStashCreation`), and the merge's own entry,
  not one the user pushed meanwhile (:class:`TestUserStashDuringMerge`);
* the no-baseline squash path must undo a half-applied merge
  (:class:`TestStopDuringSquashApply`);
* retiring the worktree must keep the ``git stash pop`` instructions of
  an interrupted merge (:class:`TestRetirementKeepsRecoveryWarning`).

The steps that follow the squash commit (the stash pop, the branch's
deletion) are finished when a Stop cuts them
(:class:`TestStopAfterTheSquashLanded`, audit 2026-11-21 GW1 F2 and F3).
The message of a failed merge, which alone names the stash it kept,
outlives a Stop in the re-attach that follows it
(:class:`TestStopDuringTheReattach`, audit 2026-11-21 GW2 N1).
"""

from __future__ import annotations

import functools
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from types import CodeType
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _git
from kiss.tests.agents.sorcar.test_audit1020_merge_stop_stash import (
    USER_NOTES,
    _make_repo,
    _MergeThread,
)
from kiss.tests.conftest import posix_only
from kiss.tests.lock_injection import inject
from kiss.tests.server.pause import Pause, Pauses

AGENT_EDIT = "a\nB-agent\nc\n"
_SH_HOOKS = posix_only("the lingering git hooks are sh scripts")


def _git_call(prefix: tuple[str, ...], frame_locals: dict[str, Any]) -> bool:
    """A :class:`Pause` matcher: the call whose ``args`` start with *prefix* (any, when empty)."""
    return bool(frame_locals.get("args", ())[: len(prefix)] == prefix)


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
        merger = _MergeThread(self.agent.merge)
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

    def _merge_stopped_at(
        self, *git_args: str, code: CodeType = _git.__code__,
        before_stop: Callable[[], object] | None = None,
        merge: Callable[[], object] | None = None,
    ) -> None:
        """Run the merge on a thread and inject the Stop at the entry of a call it makes.

        Args:
            git_args: The first arguments of the git call to stop at.
            code: The function to stop at, at its first call, instead of ``_git``.
            before_stop: Optional callable run just before the Stop.
            merge: The merge to run instead of ``agent.merge``.
        """
        merger = _MergeThread(merge or self.agent.merge)
        with Pause(code, functools.partial(_git_call, git_args)) as held:
            merger.start()
            assert held.paused.wait(30), f"the merge never ran git {git_args}: {merger.raised!r}"
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

            assert self.wt.wt_dir.exists(), "worktree deleted with the only copy"
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


class _UserStashAt:
    """Profile hook: as *function* first returns, the user stashes a new draft.

    The user's ``git stash`` takes none of the merge's locks, so the
    top of the stash is no longer the merge's entry.  With *stop* the
    runner's Stop is raised right after, where an injected one is.
    """

    def __init__(self, function: Any, repo: Path, stop: bool) -> None:
        self.code = function.__code__
        self.repo = repo
        self.stop = stop
        self.fired = False

    def __call__(self, frame: Any, event: str, arg: Any) -> None:
        if event == "return" and frame.f_code is self.code and not self.fired:
            self.fired = True
            (self.repo / "draft-b.txt").write_text("draft B\n")
            subprocess.run(
                ["git", "-C", str(self.repo), "stash", "push", "-qum", "user B"], check=True,
            )
            if self.stop:
                raise KeyboardInterrupt


class TestUserStashDuringMerge(_MergeStopHarness):
    """The merge gives back the entry it stashed, wherever the stash holds it (C10-F6)."""

    @pytest.mark.parametrize(
        ("function", "outcome", "notes"),
        [
            (GitWorktreeOps.squash_merge, "Successfully merged", True),
            (GitWorktreeOps.checkout, "Cannot checkout", True),
            (GitWorktreeOps.stash_if_dirty, "", True),
            (GitWorktreeOps.stash_if_dirty, "", False),
        ],
        ids=["success", "checkout-failed", "stopped", "stopped-with-nothing-stashed"],
    )
    def test_a_stash_the_user_pushes_meanwhile_stays_theirs(
        self, function: Any, outcome: str, notes: bool,
    ) -> None:
        """The three restore paths, each with a user's entry on top of the merge's.

        The merge popped the top of the stash: the user's new draft came
        back and the notes it had stashed itself stayed hidden.  A merge
        stopped as its stash step returned took ``refs/stash`` having
        moved for an entry of its own, also when it had stashed nothing.
        """
        if notes:
            (self.repo / "notes.txt").write_text(USER_NOTES)
        if function is GitWorktreeOps.checkout:  # the branch the task started from is gone
            _git("checkout", "-qb", "other", cwd=self.repo)
            _git("branch", "-m", "main", "renamed", cwd=self.repo)
        hook = _UserStashAt(function, self.repo, stop=not outcome)
        sys.setprofile(hook)
        try:
            if outcome:
                message = self.agent.merge()
                assert outcome in message and "git stash pop" not in message, message
            else:
                with pytest.raises(KeyboardInterrupt):
                    self.agent.merge()
        finally:
            sys.setprofile(None)

        assert hook.fired
        assert self.agent._merge_conflict_warning is None
        assert GitWorktreeOps.status_porcelain(self.repo) == ("?? notes.txt" if notes else "")
        stashed = _git("stash", "list", cwd=self.repo).stdout.splitlines()
        assert len(stashed) == 1 and stashed[0].endswith("user B"), stashed


@_SH_HOOKS
class TestStopDuringSquashApply(_MergeStopHarness):
    """A Stop between the squash being staged and its commit leaves main's index clean."""

    def test_no_baseline_squash_is_rolled_back(self) -> None:
        assert self.wt.baseline_commit is None  # main was clean at create_worktree
        (self.repo / "notes.txt").write_text(USER_NOTES)
        # Fires when the squash has written the index with the agent's
        # edit staged, and lingers there.  The index tells the moment,
        # not the command that wrote it: the squash is prepared apart
        # and published, no ``git merge --squash`` runs in the checkout.
        _install_hook(
            self.repo,
            self.hooks,
            "post-index-change",
            'case "$(git diff --cached --diff-filter=M --name-only)" in'
            f" f.txt) touch {self.ready}; sleep 0.5;; esac",
        )

        self._merge_stopped_when(self.ready.exists)

        self._assert_main_untouched()
        assert self.agent._merge_conflict_warning is None


class TestStopAfterTheSquashLanded(_MergeStopHarness):
    """A Stop in a step that follows the squash commit has that step finished.

    The stash pop and the branch's deletion are safe to repeat.  Cut
    by a Stop, the pop left the user's notes in ``git stash`` with no
    warning (and the retry said "Successfully merged" without them);
    a branch deleted while its task stayed pending made the retry
    report a merge conflict.
    """

    @pytest.mark.parametrize(
        ("code", "git_args", "pending"),
        [
            (_git.__code__, ("stash", "apply"), True),
            (GitWorktreeOps._drop_stash.__code__, (), True),  # before the drop's thread starts
            (_git.__code__, ("stash", "drop"), True),  # on that thread: raised as the pop ends
            (_git.__code__, ("config", "--remove-section"), False),  # ``git branch -D`` has run
        ],
        ids=["stash-apply", "drop-entry", "stash-drop", "branch-config"],
    )
    def test_the_step_the_stop_cut_is_finished(
        self, code: CodeType, git_args: tuple[str, ...], pending: bool,
    ) -> None:
        notes = self.repo / "notes.txt"
        notes.write_text(USER_NOTES)

        self._merge_stopped_at(*git_args, code=code)

        assert (self.repo / "f.txt").read_text() == AGENT_EDIT
        assert GitWorktreeOps.status_porcelain(self.repo) == "?? notes.txt"
        assert notes.read_text() == USER_NOTES
        assert _git("stash", "list", cwd=self.repo).stdout == ""
        assert self.agent._merge_conflict_warning is None
        # Pending with its branch, which a retry deletes, or neither.
        assert bool(self.agent._wts) == pending
        assert GitWorktreeOps.branch_exists(self.repo, self.wt.branch) == pending
        if pending:
            assert "Successfully merged" in self.agent.merge()
            assert GitWorktreeOps.status_porcelain(self.repo) == "?? notes.txt"
        assert not self.agent._wts
        assert GitWorktreeOps.load_original_branch(self.repo, self.wt.branch) is None

    def test_a_pop_that_cannot_be_finished_is_named(self) -> None:
        notes = self.repo / "notes.txt"
        notes.write_text(USER_NOTES)

        # A file appears where the stash would put the notes: git refuses the pop.
        self._merge_stopped_at(
            "stash", "apply", before_stop=functools.partial(notes.write_text, "intruder\n"),
        )

        assert (self.repo / "f.txt").read_text() == AGENT_EDIT
        assert "kiss: auto-stash before merge" in _git("stash", "list", cwd=self.repo).stdout
        assert "git stash pop" in (self.agent._merge_conflict_warning or "")

    def test_the_warnings_of_the_merge_outlive_a_stop_in_the_cleanup(self) -> None:
        # The user's edit of the line the task changed: the pop conflicts.
        (self.repo / "f.txt").write_text("a\nB-user\nc\n")
        # Another git's lock on the branch: it cannot be deleted.
        lock = self.repo / ".git" / "refs" / "heads" / f"{self.wt.branch}.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)

        self._merge_stopped_at("branch", "-D", before_stop=lock.touch)

        warning = self.agent._merge_conflict_warning or ""
        assert "kiss: auto-stash before merge" in _git("stash", "list", cwd=self.repo).stdout
        assert "git stash pop" in warning
        assert GitWorktreeOps.branch_exists(self.repo, self.wt.branch)
        assert f"git branch -D {self.wt.branch}" in warning
        assert not self.agent._wts


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


@_SH_HOOKS
class TestStopDuringTheReattach(_MergeStopHarness):
    """A Stop while a failed merge's worktree is checked out again keeps that merge's message.

    The stash of a failed merge is kept on purpose, and only the message
    ``commit_and_merge`` returns said so: stopped in the
    ``git worktree add`` that follows the merge, the tool said nothing.
    """

    def test_the_message_of_the_failed_merge_is_queued(self) -> None:
        (self.repo / "notes.txt").write_text(USER_NOTES)
        _install_hook(self.repo, self.hooks, "pre-commit", "exit 1")  # no squash commit

        self._merge_stopped_at("worktree", "add", merge=self.agent.commit_and_merge)

        assert "kiss: auto-stash before merge" in _git("stash", "list", cwd=self.repo).stdout
        warning = self.agent._merge_conflict_warning or ""
        assert f"Merge of '{self.wt.branch}'" in warning and "git stash pop" in warning

    def test_a_merge_the_stop_cut_as_well_has_no_message_to_queue(self) -> None:
        (self.repo / "notes.txt").write_text(USER_NOTES)
        # The stop watchdog injects again while the thread lives: into
        # the merge, nothing stashed yet, then into its re-attach.
        stops = [
            Pause(_git.__code__, functools.partial(_git_call, git_args))
            for git_args in (("stash", "push"), ("worktree", "add"))
        ]
        merger = _MergeThread(self.agent.commit_and_merge)
        with Pauses(*stops):
            merger.start()
            for held in stops:
                assert held.paused.wait(30), f"a Stop's git call never ran: {merger.raised!r}"
                assert inject(merger) == 1
                held.release()
        merger.join(timeout=30)
        assert isinstance(merger.raised, KeyboardInterrupt), repr(merger.raised)

        self._assert_main_untouched()
        assert self.agent._merge_conflict_warning is None
