# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review fix — ``_check_worktree_busy`` must cover the run-startup window.

``commands.py`` registers ``state.task_thread`` BEFORE calling
``thread.start()``, and the worker only raises ``is_task_active`` once
it is running.  ``_check_worktree_busy`` used to gate a manual
worktree action on ``state.is_task_active`` alone, so in that window a
manual merge/discard click passed the gate, claimed ``is_merging``,
and merged or destroyed the pending worktree the just-submitted task
was about to resume in — and the new worker then saw ``is_merging``
and refused the run the user just typed.

The gate must use the same created-but-unstarted-counts-as-alive
predicate the rest of the server standardized on:
``state.is_task_active or state.thread_alive()``.

Real ``VSCodeServer``, real git repo, real registered worktree agent
state, no mocks.  The submitted-but-unstarted worker is a REAL
``threading.Thread`` that has not been started (``ident is None``) —
exactly what ``commands.py`` registers in the window.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer
from kiss.tests.server.test_worktree_no_autocommit_branch import _init_repo

_TAB = "rr-startup-window-tab"


def _noop() -> None:
    """Body of the never-started worker thread."""


class TestStartupWindowBlocksWorktreeActions(unittest.TestCase):
    """A submitted-but-unstarted run must make the tab busy."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-rr-startup-window-")
        self.repo = Path(self.tmpdir) / "repo"
        self.repo.mkdir()
        _init_repo(str(self.repo))
        self.branch = "kiss/wt-rr-startup"
        subprocess.run(
            ["git", "-C", str(self.repo), "branch", self.branch],
            capture_output=True, check=True,
        )
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]

        agent = WorktreeSorcarAgent("rr startup-window agent")
        agent._wt = GitWorktree(
            repo_root=self.repo,
            branch=self.branch,
            original_branch="main",
            wt_dir=self.repo / ".kiss-worktrees" / "rr-startup",
            baseline_commit=None,
        )
        with self.server._state_lock:
            self.state = AgentState(
                "rr-startup-task",
                tab_id=_TAB,
                server_owned=True,
                agent=agent,
            )
            self.state.use_worktree = True
            agent_state.register(self.state)

    def tearDown(self) -> None:
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _register_unstarted_worker(self) -> threading.Thread:
        """Install a created-but-unstarted worker, as ``_cmd_run`` does."""
        thread = threading.Thread(target=_noop, daemon=True)
        with self.server._state_lock:
            self.state.task_thread = thread
        assert thread.ident is None
        assert not self.state.is_task_active
        return thread

    def test_check_worktree_busy_reports_busy_in_startup_window(self) -> None:
        self._register_unstarted_worker()
        with self.server._state_lock:
            busy = self.server._check_worktree_busy(
                self.state, "merging", self.repo,
            )
        assert busy is not None, (
            "_check_worktree_busy admitted a manual action while a "
            "submitted run's worker thread was registered but not "
            "yet started"
        )
        self.assertFalse(busy["success"])
        self.assertIn("still running", busy["message"])

    def test_merge_and_discard_refused_in_startup_window(self) -> None:
        self._register_unstarted_worker()
        for action in ("merge", "discard"):
            result = self.server._handle_worktree_action(action, _TAB)
            self.assertFalse(result["success"], action)
            self.assertIn("still running", result["message"])
            # The refusal must not leak a claim: the arriving worker
            # checks is_merging and would refuse the submitted run.
            self.assertFalse(self.state.is_merging)
        # Nothing was merged or destroyed.
        agent = self.state.agent
        assert agent is not None
        self.assertIsNotNone(agent._wt)
        self.assertTrue(GitWorktreeOps.branch_exists(self.repo, self.branch))

    def test_gate_reopens_when_no_worker_is_installed(self) -> None:
        # Control: with no registered worker the same action passes
        # the busy gate — proving the refusals above are attributable
        # to the startup-window predicate.  "nothing" (leave-as-is)
        # only writes the preserve marker, so it exercises the full
        # admission path without needing a physical worktree.
        result = self.server._handle_worktree_action("nothing", _TAB)
        self.assertTrue(result["success"], result)

    def test_dead_worker_thread_does_not_block(self) -> None:
        # A finished worker (started, ran, joined) must NOT read as
        # busy — thread_alive() is about the startup window, not about
        # tabs whose task already completed.
        thread = threading.Thread(target=_noop, daemon=True)
        thread.start()
        thread.join(timeout=30)
        with self.server._state_lock:
            self.state.task_thread = thread
            busy = self.server._check_worktree_busy(
                self.state, "merging", self.repo,
            )
        self.assertIsNone(busy)


if __name__ == "__main__":
    unittest.main()
