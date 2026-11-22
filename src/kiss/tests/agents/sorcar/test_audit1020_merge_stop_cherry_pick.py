# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop between the squash merge from a baseline and its commit leaves the main index clean.

``squash_merge_from_baseline`` prepares the task branch's changes apart,
publishes its checked files and index entries in separate steps, and commits
them in a second step.  The runner's Stop injects
``KeyboardInterrupt`` into the task thread at the first bytecode after
whichever git call is in flight; one landing when git returns from
writing the merge used to leave the half-squash staged in the user's
main index with no commit, no sequencer state to ``--abort``, and no
warning.  The merge is now undone on any ``BaseException`` before it
propagates.

These tests drive the real merge from a worker thread against a real
repository and hold that thread where git has just written the merge
(the entry of the call that names the written tree ``AUTO_MERGE``);
the test injects the interrupt there exactly as the runner does
(``PyThreadState_SetAsyncExc``), so it lands before the commit.  The
last test lingers in a ``post-commit`` hook instead.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _git
from kiss.tests.agents.sorcar.test_audit1020_merge_stop_stash import (
    USER_NOTES,
    _install_lingering_pre_commit,
    _make_repo,
    _MergeThread,
)
from kiss.tests.conftest import posix_only
from kiss.tests.lock_injection import inject
from kiss.tests.server.pause import Pause

pytestmark = posix_only("the lingering git hooks are sh scripts")

AGENT_CONTENT = "a\nB-agent\nc\n"


def _naming_the_written_tree(frame_locals: dict[str, Any]) -> bool:
    """A :class:`Pause` matcher on ``_git``: git has written the merge, its tree gets its name."""
    return bool(frame_locals.get("args", ())[:2] == ("update-ref", "AUTO_MERGE"))


def _writing_the_merge(frame_locals: dict[str, Any]) -> bool:
    """A :class:`Pause` matcher on ``_git``: git is about to be asked to write the merge."""
    args = frame_locals.get("args", ())
    return "read-tree" in args and "-u" in args


class _IndexLockBoundary:
    """Hold the real lock-creation return boundary for runner-style interruption."""

    def __init__(self, path: Path) -> None:
        """Observe the acquisition of this repository's index lock."""
        self.path = path
        self.paused = threading.Event()
        self.release = threading.Event()

    def __call__(self, frame: FrameType, event: str, arg: object) -> None:
        """Pause once after the native exclusive creation returns."""
        if self.paused.is_set():
            return
        created = event == "c_return" and arg is os.link
        opened = (
            event == "return" and frame.f_code is Path.open.__code__
            and frame.f_locals.get("self") == self.path
            and frame.f_locals.get("mode") == "xb"
        )
        if (created or opened) and self.path.exists():
            self.paused.set()
            self.release.wait(30)


class TestStopDuringCherryPick:
    """Stop lands right after git has written the merge, before the commit."""

    def setup_method(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.repo = _make_repo(self.tmpdir / "repo")
        self.head_before = GitWorktreeOps.head_sha(self.repo)
        # A dirty main tree at creation gives the worktree a baseline
        # commit, which the merge is made from.
        (self.repo / "notes.txt").write_text(USER_NOTES)
        self.agent = ChatSorcarAgent("test")
        self.agent.work_dir = str(self.repo)
        assert self.agent.create_worktree().startswith("Created worktree ")
        wt = self.agent._wt
        assert wt is not None
        assert wt.baseline_commit
        self.wt = wt
        (wt.wt_dir / "f.txt").write_text(AGENT_CONTENT)
        assert GitWorktreeOps.commit_all(wt.wt_dir, "agent work")
        self.ready = self.tmpdir / "pick-ready"

    def teardown_method(self) -> None:
        if self.agent._wt is not None:
            try:
                self.agent.discard()
            except Exception:
                pass
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _merge_stopped_once_written(
        self, matcher: Callable[[dict[str, Any]], bool] = _naming_the_written_tree,
    ) -> None:
        """Run the merge on a thread and inject the Stop at the git call *matcher* names.

        By default as git returns from writing the merge.
        """
        merger = _MergeThread(self.agent.merge)
        with Pause(_git.__code__, matcher) as written:
            merger.start()
            assert written.paused.wait(30), f"the merge was never written: {merger.raised!r}"
            assert inject(merger) == 1
        merger.join(timeout=30)
        assert not merger.is_alive()
        assert isinstance(merger.raised, KeyboardInterrupt), repr(merger.raised)

    def _merge_stopped_during_hook(self) -> None:
        """Run the merge on a thread and inject the Stop while the lingering hook runs."""
        merger = _MergeThread(self.agent.merge)
        merger.start()
        deadline = time.monotonic() + 30
        while not self.ready.exists():
            assert time.monotonic() < deadline, "the merge never ran its hook"
            assert merger.is_alive(), f"merge ended early: {merger.raised!r}"
            time.sleep(0.002)
        assert inject(merger) == 1
        merger.join(timeout=30)
        assert not merger.is_alive()
        assert isinstance(merger.raised, KeyboardInterrupt), repr(merger.raised)

    def _assert_main_tree_handed_back(self, expected_status: str) -> None:
        """The main repo has no merge commit, no staged squash and no pick in progress."""
        assert GitWorktreeOps.head_sha(self.repo) == self.head_before
        assert GitWorktreeOps.status_porcelain(self.repo) == expected_status
        assert not (self.repo / ".git" / "sequencer").exists()
        assert not (self.repo / ".git" / "CHERRY_PICK_HEAD").exists()
        assert (self.repo / "f.txt").read_text() == "a\nb\nc\n"
        assert _git("stash", "list", cwd=self.repo).stdout == ""
        assert _git("rev-parse", "--verify", self.wt.branch, cwd=self.repo).returncode == 0
        assert self.agent._merge_conflict_warning is None

    def _assert_merge_retry_succeeds(self) -> None:
        """A second ``merge()`` on the handed-back tree lands the agent's work."""
        self.agent.merge()
        assert GitWorktreeOps.head_sha(self.repo) != self.head_before
        assert (self.repo / "f.txt").read_text() == AGENT_CONTENT
        assert self.agent._wt is None

    def test_users_edits_are_restored_and_the_index_is_clean(self) -> None:
        self._merge_stopped_once_written()

        self._assert_main_tree_handed_back("?? notes.txt")
        assert (self.repo / "notes.txt").read_text() == USER_NOTES
        self._assert_merge_retry_succeeds()
        assert (self.repo / "notes.txt").read_text() == USER_NOTES

    def test_clean_main_tree_stays_clean(self) -> None:
        (self.repo / "notes.txt").unlink()

        self._merge_stopped_once_written()

        self._assert_main_tree_handed_back("")
        self._assert_merge_retry_succeeds()

    def test_stop_before_git_writes_the_merge_names_no_written_tree(self) -> None:
        """Nothing was written: no ``AUTO_MERGE`` is left for a later merge to take back."""
        self._merge_stopped_once_written(_writing_the_merge)

        self._assert_main_tree_handed_back("?? notes.txt")
        assert _git("rev-parse", "-q", "--verify", "AUTO_MERGE", cwd=self.repo).returncode != 0
        self._assert_merge_retry_succeeds()

    def test_stop_at_index_lock_acquisition_is_retryable(self) -> None:
        """Stop at native lock creation leaves neither a lock nor a partially applied merge."""
        boundary = _IndexLockBoundary(self.repo / ".git" / "index.lock")
        merger = _MergeThread(self.agent.merge)
        threading.setprofile(boundary)
        try:
            merger.start()
            assert boundary.paused.wait(30), f"lock creation was not reached: {merger.raised!r}"
            assert inject(merger) == 1
        finally:
            boundary.release.set()
            threading.setprofile(None)
            merger.join(timeout=30)
        assert not merger.is_alive()
        assert isinstance(merger.raised, KeyboardInterrupt)
        assert not boundary.path.exists()
        self._assert_main_tree_handed_back("?? notes.txt")
        self._assert_merge_retry_succeeds()

    def test_stop_after_the_squash_commit_keeps_the_commit(self) -> None:
        """The undo must be a no-op once the merge has been committed."""
        hooks = self.tmpdir / "commit-hooks"
        _install_lingering_pre_commit(self.repo, hooks, self.ready)
        # Lingering after the commit is made: a Stop during ``pre-commit`` ends git before it.
        (hooks / "pre-commit").rename(hooks / "post-commit")

        self._merge_stopped_during_hook()

        assert GitWorktreeOps.head_sha(self.repo) != self.head_before
        assert (self.repo / "f.txt").read_text() == AGENT_CONTENT
        assert GitWorktreeOps.status_porcelain(self.repo) == "?? notes.txt"
        assert (self.repo / "notes.txt").read_text() == USER_NOTES
        assert _git("stash", "list", cwd=self.repo).stdout == ""
