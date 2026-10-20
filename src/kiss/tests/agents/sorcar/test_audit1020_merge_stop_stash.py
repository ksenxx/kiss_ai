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

import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

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
    """Runs ``agent.merge()`` and keeps whatever it raised."""

    def __init__(self, agent: ChatSorcarAgent) -> None:
        super().__init__(daemon=True)
        self.agent = agent
        self.raised: BaseException | None = None

    def run(self) -> None:
        try:
            self.agent.merge()
        except BaseException as exc:
            self.raised = exc


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
        merger = _MergeThread(self.agent)
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
