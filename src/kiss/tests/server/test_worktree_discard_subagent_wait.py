# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Work-loss audit: discard must not delete a worktree under a live writer.

``_commit_and_clean_worktree`` (the merge/release path) waits for
abandoned sub-agent threads before removing the worktree, and
preserves the directory when one is still running.  ``discard()`` used
to remove the directory immediately — so an abandoned sub-agent still
writing into the worktree had its next write fail (its output lost)
or, worse, half-recreated the directory as an unregistered zombie.
The automatic empty-branch discard can hit this without any user
intent: the changed-files probe runs before the child has written
anything.

These tests build the racing sub-agent from real parts: a real git
repository and worktree, a real ``ThreadPoolExecutor`` thread wrapped
in the real ``_AbandonedSubagent`` bookkeeping class.  No mocks.

Also covers the pool guard: ``worktree_pool.discard_all`` must
preserve a spare an external writer put content into (mirroring the
reclaim pass), while still removing clean spares.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar import worktree_pool
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _AbandonedSubagent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.tests.server.test_worktree_ignored_file_rescue import (
    _make_repo,
    _redirect_db,
    _restore_db,
    _stub_parent_run,
)


class TestDiscardWaitsForSubagents:
    """discard() honours the abandoned-sub-agent wait."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-discard-wait-")
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self._original_run: Any = None
        self.pool = ThreadPoolExecutor(max_workers=1)

    def teardown_method(self) -> None:
        self.pool.shutdown(wait=True)
        if self._original_run is not None:
            cast(Any, SorcarAgent.__mro__[1]).run = self._original_run
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _pending_agent(self) -> WorktreeSorcarAgent:
        """Run one stubbed worktree task; return the pending agent."""
        self._original_run = _stub_parent_run({"tracked.txt": "work\n"})
        agent = WorktreeSorcarAgent("discard-wait-test")
        agent.run("task", work_dir=str(self.repo), auto_commit=True)
        assert agent._wt_pending
        return agent

    def _abandon_child(
        self, agent: WorktreeSorcarAgent, child: Any,
    ) -> None:
        """Register *child* (a callable) as a real abandoned sub-agent."""
        future = self.pool.submit(child)
        with agent._abandoned_lock:
            agent._abandoned_subagents.append(
                _AbandonedSubagent(
                    future, SorcarAgent("child-usage"), (0.0, 0, 0),
                )
            )

    def test_discard_waits_for_finishing_subagent(self) -> None:
        """A child that finishes during the wait writes before removal."""
        agent = self._pending_agent()
        wt_dir = agent._wt_dir
        assert wt_dir is not None
        write_ok = threading.Event()

        def late_writer() -> str:
            time.sleep(0.5)
            (wt_dir / "late-child-output.txt").write_text("late\n")
            write_ok.set()
            return "done"

        self._abandon_child(agent, late_writer)
        msg = agent.discard()
        assert "Discarded" in msg, msg
        assert write_ok.is_set(), (
            "the worktree was deleted under the still-writing sub-agent"
        )
        assert not wt_dir.exists()

    def test_discard_deferred_while_subagent_running(self) -> None:
        """A child that outlives the wait defers the discard entirely."""
        agent = self._pending_agent()
        wt_dir = agent._wt_dir
        assert wt_dir is not None
        release = threading.Event()

        def wedged_child() -> str:
            release.wait(120)
            return "done"

        self._abandon_child(agent, wedged_child)
        try:
            msg = agent.discard()
            assert "Discard deferred" in msg, msg
            assert wt_dir.exists(), (
                "a deferred discard must leave the worktree in place"
            )
            assert agent._wt_pending, (
                "a deferred discard must keep the pending handle for retry"
            )
        finally:
            release.set()
        assert agent.reclaim_abandoned_subagents(timeout=120)
        msg = agent.discard()
        assert "Discarded" in msg, msg
        assert not wt_dir.exists()


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in *cwd* capturing output."""
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, check=False,
    )


class TestDiscardAllPreservesDirtySpare:
    """worktree_pool.discard_all mirrors the reclaim spare guard."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-pool-guard-")
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        worktree_pool.discard_all()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dirty_spare_preserved_clean_spare_removed(self) -> None:
        """A spare with external content survives discard_all."""
        assert worktree_pool.prewarm(self.repo)
        spare = worktree_pool.take_spare(self.repo)
        assert spare is not None
        branch, wt_dir = spare
        # Put it back so discard_all owns it again.
        worktree_pool._spares[worktree_pool._repo_key(self.repo)] = spare
        (wt_dir / "external-writer.txt").write_text("do not destroy\n")
        worktree_pool.discard_all()
        assert wt_dir.is_dir(), (
            "discard_all destroyed a spare carrying external content"
        )
        assert GitWorktreeOps.branch_exists(self.repo, branch)
        # A clean spare is still removed as before.
        (wt_dir / "external-writer.txt").unlink()
        worktree_pool._spares[worktree_pool._repo_key(self.repo)] = spare
        worktree_pool.discard_all()
        assert not wt_dir.exists()
        assert not GitWorktreeOps.branch_exists(self.repo, branch)
