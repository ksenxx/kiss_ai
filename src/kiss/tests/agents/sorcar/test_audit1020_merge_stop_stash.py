# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop during the ``commit_and_merge`` tool's merge gives the user's stash back.

``_do_merge_locked`` stashes the user's uncommitted main-tree edits,
checks out the original branch, squash-merges the task branch and pops
the stash.  The runner's Stop injects ``KeyboardInterrupt`` into the
task thread, and CPython raises it at the first bytecode after
whichever git call is in flight: a user pressing Stop while the squash
commit's ``pre-commit`` hook runs (linters take seconds) lands between
the stash push and the pop.  The merge used to shield only the conflict
resolver sub-step, so the interrupt left the user's edits sitting in
``git stash`` with no warning.

These tests drive the real merge from a worker thread against a real
repository whose ``pre-commit`` hook signals the test and lingers; the
test injects the interrupt exactly as the runner does
(``PyThreadState_SetAsyncExc``) while the commit is in flight.
"""

from __future__ import annotations

import _thread
import functools
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _git
from kiss.tests.conftest import posix_only
from kiss.tests.lock_injection import inject

pytestmark = posix_only("the lingering git hooks are sh scripts")

USER_NOTES = "uncommitted user notes\n"


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    for key, value in (("user.email", "t@t.com"), ("user.name", "T")):
        subprocess.run(["git", "-C", str(path), "config", key, value], check=True)
    (path / "f.txt").write_text("a\nb\nc\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "initial"], check=True)
    return path


def _install_lingering_pre_commit(repo: Path, hooks: Path, ready: Path) -> None:
    """Make every commit in *repo* touch *ready*, then linger 0.5 s before passing."""
    hooks.mkdir()
    hook = hooks / "pre-commit"
    hook.write_text(f"#!/bin/sh\ntouch {ready}\nsleep 0.5\nexit 0\n")
    hook.chmod(0o755)
    subprocess.run(
        ["git", "-C", str(repo), "config", "core.hooksPath", str(hooks)],
        check=True,
    )


class _MergeThread(threading.Thread):
    """Runs *call* (``agent.merge``, a ``stash_pop``) and keeps whatever it raised."""

    def __init__(self, call: Callable[[], object]) -> None:
        super().__init__(daemon=True)
        self.call = call
        self.raised: BaseException | None = None

    def run(self) -> None:
        try:
            self.call()
        except BaseException as exc:
            self.raised = exc


def _install_racing_git(tmpdir: Path, repo: Path, after_drop: str) -> str:
    """Return a ``PATH`` whose ``git`` loses its first ``stash drop`` to the user's own stash.

    The user's entry is pushed just before the real drop runs, so the drop,
    which goes by position, takes theirs; the shell *after_drop* then runs
    before the script ends.  Every other command is git's own.
    """
    real, raced = shutil.which("git"), tmpdir / "raced"
    (tmpdir / "bin").mkdir()
    script = tmpdir / "bin" / "git"
    script.write_text(
        "#!/bin/sh\n"
        f'case " $* " in *" stash drop "*) if [ ! -e {raced} ]; then\n'
        f"    touch {raced}\n"
        f"    echo \"the user's work\" > {repo}/u.txt\n"
        f"    {real} -C {repo} stash push -q --include-untracked -m \"the user's own\" -- u.txt\n"
        f'    {real} "$@"; status=$?\n'
        f"    {after_drop}\n"
        "    exit $status\n"
        "fi;; esac\n"
        f'exec {real} "$@"\n'
    )
    script.chmod(0o755)
    return f"{script.parent}{os.pathsep}{os.environ['PATH']}"


def _stop_as_the_drop_thread_starts(frame: FrameType, event: str, arg: Any) -> None:
    """Profile hook: inject the Stop as ``_drop_stash``'s call that starts its thread returns."""
    if (event == "c_return" and arg is _thread.start_new_thread
            and frame.f_code is GitWorktreeOps._drop_stash.__code__):
        sys.setprofile(None)
        inject(threading.current_thread())


class TestStopDuringSquashCommit:
    """Stop lands right after the squash commit returns, before the stash pop."""

    def setup_method(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.repo = _make_repo(self.tmpdir / "repo")
        self.agent = ChatSorcarAgent("test")
        self.agent.work_dir = str(self.repo)
        assert self.agent.create_worktree().startswith("Created worktree ")
        wt = self.agent._wt
        assert wt is not None
        self.wt = wt
        (wt.wt_dir / "f.txt").write_text("a\nB-agent\nc\n")
        assert GitWorktreeOps.commit_all(wt.wt_dir, "agent work")
        self.ready = self.tmpdir / "commit-ready"
        _install_lingering_pre_commit(self.repo, self.tmpdir / "hooks", self.ready)

    def teardown_method(self) -> None:
        if self.agent._wt is not None:
            try:
                self.agent.discard()
            except Exception:
                pass
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _merge_stopped_during_commit(self) -> None:
        """Run the merge on a thread and inject the Stop while ``git commit`` runs."""
        merger = _MergeThread(self.agent.merge)
        merger.start()
        deadline = time.monotonic() + 30
        while not self.ready.exists():
            assert time.monotonic() < deadline, "the squash commit never ran its hook"
            assert merger.is_alive(), f"merge ended early: {merger.raised!r}"
            time.sleep(0.002)
        assert inject(merger) == 1
        merger.join(timeout=30)
        assert not merger.is_alive()
        assert isinstance(merger.raised, KeyboardInterrupt), repr(merger.raised)

    def test_users_edits_are_restored_from_the_stash(self) -> None:
        (self.repo / "notes.txt").write_text(USER_NOTES)

        self._merge_stopped_during_commit()

        assert _git("stash", "list", cwd=self.repo).stdout == ""
        assert (self.repo / "notes.txt").read_text() == USER_NOTES
        assert "?? notes.txt" in GitWorktreeOps.status_porcelain(self.repo)
        assert _git("rev-parse", "--verify", self.wt.branch, cwd=self.repo).returncode == 0
        assert self.agent._merge_conflict_warning is None

    def test_clean_main_tree_needs_no_pop(self) -> None:
        self._merge_stopped_during_commit()

        assert _git("stash", "list", cwd=self.repo).stdout == ""
        assert _git("rev-parse", "--verify", self.wt.branch, cwd=self.repo).returncode == 0
        assert self.agent._merge_conflict_warning is None


class TestDropTookTheUsersEntry:
    """A drop that removed the user's own stash entry puts it back, Stop or not.

    ``git stash drop`` goes by position, and the user's ``git stash`` (which
    takes none of the merge's locks) moves every position between the look
    at the list and the drop.  The entry the drop then removed is in no ref
    until it is stored back: a Stop between the two used to lose it.
    """

    def setup_method(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.repo = _make_repo(self.tmpdir / "repo")
        (self.repo / "f.txt").write_text(USER_NOTES)
        assert GitWorktreeOps.stash_if_dirty(self.repo) is True

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_a_stop_as_the_drop_ends_does_not_lose_it(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dropped = self.tmpdir / "dropped"
        racing = _install_racing_git(self.tmpdir, self.repo, f"touch {dropped}; sleep 0.5")
        monkeypatch.setenv("PATH", racing)
        popper = _MergeThread(functools.partial(GitWorktreeOps.stash_pop, self.repo))
        popper.start()
        deadline = time.monotonic() + 30
        while not dropped.exists():
            assert time.monotonic() < deadline and popper.is_alive(), repr(popper.raised)
            time.sleep(0.002)
        assert inject(popper) == 1
        popper.join(timeout=30)

        assert isinstance(popper.raised, KeyboardInterrupt), repr(popper.raised)
        listed = _git("stash", "list", cwd=self.repo).stdout.splitlines()
        assert len(listed) == 1 and listed[0].endswith("the user's own"), listed
        assert (self.repo / "f.txt").read_text() == USER_NOTES

    def test_a_stop_as_the_drop_starts_waits_for_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Stop raised as the call that starts the drop's thread returns is raised once
        that thread has ended.  (With no thread to be had the drop runs on the caller's,
        as before: that takes a process out of threads, which no test here makes.)"""
        monkeypatch.setenv("PATH", _install_racing_git(self.tmpdir, self.repo, "sleep 0.5"))
        sys.setprofile(_stop_as_the_drop_thread_starts)
        try:
            with pytest.raises(KeyboardInterrupt):
                GitWorktreeOps.stash_pop(self.repo)
        finally:
            sys.setprofile(None)

        listed = _git("stash", "list", cwd=self.repo).stdout.splitlines()
        assert len(listed) == 1 and listed[0].endswith("the user's own"), listed
        assert (self.repo / "f.txt").read_text() == USER_NOTES

    def test_one_that_cannot_be_stored_back_is_named_for_recovery(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        lock = self.repo / ".git" / "refs" / "stash.lock"  # git cannot update refs/stash
        monkeypatch.setenv("PATH", _install_racing_git(self.tmpdir, self.repo, f": > {lock}"))
        with caplog.at_level(logging.ERROR, logger="kiss.agents.sorcar.git_worktree"):
            assert GitWorktreeOps.stash_pop(self.repo)
        lock.unlink()

        recovery = re.search(r"git stash store ([0-9a-f]{40})", caplog.text)
        assert recovery, caplog.text
        assert _git("stash", "store", recovery[1], cwd=self.repo).returncode == 0
        assert _git("stash", "pop", cwd=self.repo).returncode == 0
        assert (self.repo / "u.txt").read_text() == "the user's work\n"
