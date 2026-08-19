# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests confirming bugs found in worktree audit round 5 are FIXED.

BUG-19: discard() now acquires repo_lock — checkout serialized with
        merge/release from other tabs.
BUG-20: _release_worktree checkout failure now sets _merge_conflict_warning.
BUG-21: checkout_error() removed — checkout() returns (bool, stderr).
BUG-22: _check_merge_conflict misses staged files — KNOWN LIMITATION.
BUG-23: _try_setup_worktree now checks commit_staged return value and
        uses --no-verify for the baseline commit.
BUG-24: _get_worktree_changed_files returns [] on git diff failure —
        KNOWN LIMITATION (conservative: no false-positive changes).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    _git,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.agents.sorcar.test_worktree_audit5 import (  # noqa: F401
    _make_repo,
    _redirect_db,
    _restore_db,
)


class TestBug22ConflictMissesStaged:
    """BUG-22: This is a known limitation. The fix would require
    adding a staged-files check to _check_merge_conflict. Kept as-is
    because git merge --squash handles it at merge time.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_staged_overlap_not_detected(self) -> None:
        """BUG-22: Staged overlap is not detected (known limitation)."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test22"
        wt_work = agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt = agent._wt
        assert wt is not None

        (wt.wt_dir / "shared.txt").write_text("agent version\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "agent changes shared.txt")

        (repo / "shared.txt").write_text("user version\n")
        _git("add", "shared.txt", cwd=repo)

        server = VSCodeServer()
        server.work_dir = str(repo)
        state = agent_state.AgentState(
            "task-t22", agent=agent, tab_id="t22", server_owned=True,
        )
        state.use_worktree = True
        agent_state.register(state)
        try:
            has_conflict = server._check_merge_conflict("t22")
        finally:
            agent_state.unregister(state.task_id, state)
        assert has_conflict is True


class TestBug24SilentDiscardOnGitFailure:
    """BUG-24 + BUG-51: _get_worktree_changed_files originally returned []
    when git diff failed (conservative).  The BUG-51 fix replaced that
    behavior with a fallback to ``git status --porcelain`` so real
    agent changes are not silently discarded when ``git diff`` fails
    (e.g. due to a bad baseline SHA).
    """

    def test_get_changed_files_falls_back_to_status_on_diff_failure(
        self,
    ) -> None:
        """BUG-51 fix: returns files from ``git status`` fallback when
        ``git diff`` against the baseline fails."""
        tmpdir = tempfile.mkdtemp()
        saved = _redirect_db(tmpdir)
        try:
            repo = _make_repo(Path(tmpdir) / "repo")

            agent = WorktreeSorcarAgent("test")
            agent._chat_id = "test24"
            wt_work = agent._try_setup_worktree(repo, str(repo))
            assert wt_work is not None

            wt = agent._wt
            assert wt is not None

            (wt.wt_dir / "important.txt").write_text("agent work\n")
            GitWorktreeOps.commit_all(wt.wt_dir, "important agent changes")

            server = VSCodeServer()
            server.work_dir = str(repo)
            state = agent_state.AgentState(
                "task-t24", agent=agent, tab_id="t24", server_owned=True,
            )
            state.use_worktree = True
            agent_state.register(state)
            agent._wt = GitWorktree(
                repo_root=wt.repo_root,
                branch=wt.branch,
                original_branch=wt.original_branch,
                wt_dir=wt.wt_dir,
                baseline_commit="0000000000000000000000000000000000000000",
            )

            try:
                changed_after = server._get_worktree_changed_files("t24")
            finally:
                agent_state.unregister(state.task_id, state)
            assert "important.txt" in changed_after

        finally:
            _restore_db(saved)
            shutil.rmtree(tmpdir, ignore_errors=True)
