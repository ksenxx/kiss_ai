# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shutdown must wait for EVERY worktree mutation that claims ``is_merging``.

Concurrency audit (F3, reviewer R3 finding 7): ``_await_active_merges``
— the only consumer of ``AgentState.merge_thread`` — joins the threads
published there so ``stop_async`` / the SIGTERM ``finally`` do not
return while git is still rewriting the repository.  Only the direct
Merge/Discard click published its thread; the resume-time
``_finalize_pending_worktree`` (auto-commit merge or discard), the
empty-worktree auto-discard in ``_present_pending_worktree`` and the
tab-close worktree retirement all claimed ``is_merging`` WITHOUT a
thread, so the shutdown wait saw nothing and returned immediately.

The interleaving is forced for real with the per-repo ``repo_lock``:
a holder thread owns it so the mutation blocks inside its git section
with the claim already taken, and the shutdown wait is then started
and must still be blocked half a second later.  Real git, real
threads, no mocks.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    repo_lock,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.merge_flow import _PendingOutcome
from kiss.server.server import VSCodeServer
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.test_worktree_audit18 import (
    _make_repo,
    _register_wt_state,
    _server,
)


class TestAwaitActiveMergesCoversAllClaims(unittest.TestCase):
    """Every worktree-mutating ``is_merging`` claim publishes its thread."""

    def setUp(self) -> None:
        agent_state.agent_states.clear()
        self.addCleanup(agent_state.agent_states.clear)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_dir = Path(tmp.name)
        self.repo = _make_repo(self.tmp_dir / "repo")
        # Constructed first: creating a RemoteAccessServer resets the
        # process-global agent-state registry populated below.
        self.remote = RemoteAccessServer(
            use_tunnel=False,
            url_file=self.tmp_dir / "remote-url.json",
            uds_path=self.tmp_dir / "kiss.sock",
        )
        self.server: VSCodeServer = _server(self.repo)
        self.state: AgentState = _register_wt_state("a")
        # An EMPTY pending worktree: every path below picks "discard"
        # or "retire", each of which mutates git under repo_lock.
        branch = "kiss/wt-f3-claim"
        self.wt_dir = self.repo / ".kiss-worktrees" / "kiss_wt-f3-claim"
        self.assertTrue(GitWorktreeOps.create(self.repo, branch, self.wt_dir))
        GitWorktreeOps.save_original_branch(self.repo, branch, "main")
        cast(WorktreeSorcarAgent, self.state.agent)._wt = GitWorktree(
            repo_root=self.repo, branch=branch, original_branch="main",
            wt_dir=self.wt_dir,
        )

    def _assert_shutdown_waits_for(self, mutate: Callable[[], Any]) -> Any:
        """Run *mutate* behind a held ``repo_lock``; the shutdown wait must block.

        Returns *mutate*'s result once everything has been released.
        """
        lock = repo_lock(self.repo)
        release = threading.Event()
        held = threading.Event()

        def hold_lock() -> None:
            with lock:
                held.set()
                release.wait(timeout=30)

        threading.Thread(target=hold_lock, daemon=True).start()
        self.assertTrue(held.wait(5))

        results: list[Any] = []
        claimant = threading.Thread(
            target=lambda: results.append(mutate()), daemon=True,
        )
        claimant.start()
        deadline = time.monotonic() + 5
        while not self.state.is_merging and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(self.state.is_merging, "mutation never claimed the tab")
        self.assertTrue(claimant.is_alive(), "mutation must be blocked on git")

        waited: list[float] = []

        def shutdown_wait() -> None:
            self.remote._await_active_merges(timeout=30)
            waited.append(time.monotonic())

        waiter = threading.Thread(target=shutdown_wait, daemon=True)
        waiter.start()
        time.sleep(0.5)
        try:
            self.assertTrue(
                waiter.is_alive(),
                "_await_active_merges returned while the worktree "
                "mutation was still rewriting the repository",
            )
        finally:
            release.set()
        claimant.join(timeout=20)
        finished = time.monotonic()
        waiter.join(timeout=20)
        self.assertFalse(claimant.is_alive(), "mutation hung after release")
        self.assertFalse(waiter.is_alive(), "shutdown wait hung after release")
        # The wait returned only once the mutation had finished.
        self.assertGreaterEqual(waited[0], finished - 0.5)
        self.assertFalse(self.wt_dir.exists(), "empty worktree must be gone")
        self.assertFalse(self.state.is_merging)
        self.assertIsNone(self.state.merge_thread)
        return results[0]

    def test_resume_time_finalize(self) -> None:
        outcome = self._assert_shutdown_waits_for(
            lambda: self.server._finalize_pending_worktree("a"),
        )
        self.assertEqual(outcome, _PendingOutcome.FINALIZED)

    def test_present_pending_worktree_auto_discard(self) -> None:
        self._assert_shutdown_waits_for(
            lambda: self.server._present_pending_worktree("a"),
        )

    def test_present_claim_release_clears_thread(self) -> None:
        # ``_finalize_pending_worktree`` hands the presentation claim
        # (thread included) to the caller when auto-commit is off; the
        # release must drop both fields again.
        self.state.auto_commit_mode = False
        outcome = self.server._finalize_pending_worktree("a")
        self.assertEqual(outcome, _PendingOutcome.PRESENT_CLAIMED)
        self.assertIs(self.state.merge_thread, threading.current_thread())
        self.server._release_present_claim("a")
        self.assertFalse(self.state.is_merging)
        self.assertIsNone(self.state.merge_thread)

    def test_deferred_tab_close_teardown(self) -> None:
        self.state.frontend_closed = True
        self._assert_shutdown_waits_for(
            lambda: self.server._dispose_if_closed("a"),
        )
        self.assertIsNone(agent_state.find_by_tab("a"), "state must be retired")

    def test_immediate_tab_close_teardown(self) -> None:
        self._assert_shutdown_waits_for(
            lambda: self.server._drop_tab_state("a"),
        )
        self.assertIsNone(agent_state.find_by_tab("a"), "state must be retired")


if __name__ == "__main__":
    unittest.main()
