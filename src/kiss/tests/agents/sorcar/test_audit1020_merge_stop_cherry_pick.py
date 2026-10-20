# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop between the squash cherry-pick and its commit leaves the main index clean.

``squash_merge_from_baseline`` stages the task branch's changes with
``git cherry-pick --no-commit`` and commits them in a second step.
The runner's Stop injects ``KeyboardInterrupt`` into the task thread
at the first bytecode after whichever git call is in flight; one
landing when the cherry-pick returns used to leave the half-squash
staged in the user's main index with no commit, no sequencer state to
``--abort``, and no warning.  The cherry-pick is now undone on any
``BaseException`` before it propagates.

These tests drive the real merge from a worker thread against a real
repository whose ``post-index-change`` hook signals the test and
lingers only while a cherry-pick runs (the stash push before it also
writes the index); the test injects the interrupt exactly as the
runner does (``PyThreadState_SetAsyncExc``) while that pick is in
flight, so it lands when the pick returns.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

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

pytestmark = posix_only("the lingering git hooks are sh scripts")

AGENT_CONTENT = "a\nB-agent\nc\n"


def _install_lingering_cherry_pick_hook(repo: Path, hooks: Path, ready: Path) -> None:
    """Make every index write of a ``git cherry-pick`` touch *ready* and linger 0.5 s."""
    hooks.mkdir()
    hook = hooks / "post-index-change"
    hook.write_text(
        "#!/bin/sh\n"
        'case "$(ps -o args= -p $PPID)" in\n'
        f"  *cherry-pick*) touch {ready}; sleep 0.5 ;;\n"
        "esac\n"
        "exit 0\n"
    )
    hook.chmod(0o755)
    subprocess.run(
        ["git", "-C", str(repo), "config", "core.hooksPath", str(hooks)],
        check=True,
    )


class TestStopDuringCherryPick:
    """Stop lands right after ``git cherry-pick --no-commit`` returns, before the commit."""

    def setup_method(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.repo = _make_repo(self.tmpdir / "repo")
        self.head_before = GitWorktreeOps.head_sha(self.repo)
        # A dirty main tree at creation gives the worktree a baseline
        # commit, which routes the merge through the cherry-pick.
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
        _install_lingering_cherry_pick_hook(self.repo, self.tmpdir / "hooks", self.ready)

    def teardown_method(self) -> None:
        if self.agent._wt is not None:
            try:
                self.agent.discard()
            except Exception:
                pass
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _merge_stopped_during_hook(self) -> None:
        """Run the merge on a thread and inject the Stop while the lingering hook runs."""
        merger = _MergeThread(self.agent)
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
        _git("config", "--unset", "core.hooksPath", cwd=self.repo)
        self.agent.merge()
        assert GitWorktreeOps.head_sha(self.repo) != self.head_before
        assert (self.repo / "f.txt").read_text() == AGENT_CONTENT
        assert self.agent._wt is None

    def test_users_edits_are_restored_and_the_index_is_clean(self) -> None:
        self._merge_stopped_during_hook()

        self._assert_main_tree_handed_back("?? notes.txt")
        assert (self.repo / "notes.txt").read_text() == USER_NOTES
        self._assert_merge_retry_succeeds()
        assert (self.repo / "notes.txt").read_text() == USER_NOTES

    def test_clean_main_tree_stays_clean(self) -> None:
        (self.repo / "notes.txt").unlink()

        self._merge_stopped_during_hook()

        self._assert_main_tree_handed_back("")
        self._assert_merge_retry_succeeds()

    def test_stop_after_the_squash_commit_keeps_the_commit(self) -> None:
        """The undo must be a no-op once the pick has been committed."""
        _git("config", "--unset", "core.hooksPath", cwd=self.repo)
        _install_lingering_pre_commit(self.repo, self.tmpdir / "commit-hooks", self.ready)

        self._merge_stopped_during_hook()

        assert GitWorktreeOps.head_sha(self.repo) != self.head_before
        assert (self.repo / "f.txt").read_text() == AGENT_CONTENT
        assert GitWorktreeOps.status_porcelain(self.repo) == "?? notes.txt"
        assert (self.repo / "notes.txt").read_text() == USER_NOTES
        assert _git("stash", "list", cwd=self.repo).stdout == ""
