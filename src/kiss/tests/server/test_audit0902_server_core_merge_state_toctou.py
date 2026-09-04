# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 (server-core): stale-state claim in the worktree action.

``_MergeFlowMixin._handle_worktree_action`` used to resolve the tab's
:class:`AgentState` (and read its agent's ``_wt_pending`` / repo
root) OUTSIDE ``_state_lock`` and only then take the lock to run
``_check_worktree_busy`` and claim ``is_merging`` — on the object it
had resolved earlier.  A ``closeTab`` (or a new ``run``) landing in
that window unregisters the state and releases / carries over the
worktree agent, so the busy check then inspects an object nobody else
can see, the claim lands on it, and ``wt.merge()`` runs concurrently
with the close path's ``_release_worktree()`` on the SAME worktree.

The window is made deterministic here by a real
:class:`WorktreeSorcarAgent` subclass whose ``_wt_pending`` property
blocks the merge thread the first time it is read; ``_close_tab``
runs while it is blocked.  With the state resolved and claimed inside
one locked section the close either waits for the claim (and is
deferred because the tab is busy) or wins outright (and the action is
refused), so exactly one of ``merge()`` / ``_release_worktree()``
ever runs.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


class _WindowAgent(WorktreeSorcarAgent):
    """Real agent that blocks in ``_wt_pending`` once and counts disposals."""

    def __init__(self, repo_root: Path) -> None:
        super().__init__("Audit0902 window agent")
        self._wt = GitWorktree(
            repo_root=repo_root,
            branch="kiss/wt-audit0902",
            original_branch="main",
            wt_dir=repo_root / "wt",
            baseline_commit=None,
        )
        self.window_entered = threading.Event()
        self.window_release = threading.Event()
        self.blocking_thread: threading.Thread | None = None
        self.merge_calls = 0
        self.release_calls = 0
        self._counts = threading.Lock()

    @property
    def _repo_root(self) -> Path | None:
        """Block the designated thread's FIRST read, then behave normally.

        ``_repo_root`` is the LAST agent attribute the worktree action
        reads before it takes ``_state_lock``, so blocking here parks
        the merge thread exactly in the check-then-act window.  The
        value is computed before blocking so the merge thread keeps
        working with the snapshot it took, as the real race would.
        """
        value = self._wt.repo_root if self._wt else None
        if (
            self.blocking_thread is threading.current_thread()
            and not self.window_entered.is_set()
        ):
            self.window_entered.set()
            self.window_release.wait(timeout=30)
        return value

    def merge(self) -> str:
        with self._counts:
            self.merge_calls += 1
        self._wt = None
        return "Successfully merged worktree branch."

    def _release_worktree(self) -> str | None:
        with self._counts:
            self.release_calls += 1
        self._wt = None
        return None


class TestWorktreeActionStateToctou(unittest.TestCase):
    """A close racing a merge click must never dispose the worktree twice."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-audit0902-toctou-")
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

        def capture(event: dict[str, Any]) -> None:
            with self._events_lock:
                self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_close_tab_during_merge_resolution_disposes_once(self) -> None:
        tab_id = "audit0902-wt-tab"
        agent = _WindowAgent(Path(self.tmpdir))
        with self.server._state_lock:
            state = agent_state.AgentState(
                f"task-for-{tab_id}", tab_id=tab_id,
                server_owned=True, agent=agent,
            )
            state.use_worktree = True
            agent_state.register(state)
        self.server.tab_registry.open_tab(tab_id, "audit")

        results: dict[str, dict[str, Any]] = {}

        def merge_click() -> None:
            results["merge"] = self.server._handle_worktree_action(
                "merge", tab_id,
            )

        merger = threading.Thread(target=merge_click, daemon=True)
        agent.blocking_thread = merger
        merger.start()
        assert agent.window_entered.wait(timeout=30), (
            "merge thread never reached the state-resolution window"
        )

        closer = threading.Thread(
            target=self.server._close_tab, args=(tab_id,), daemon=True,
        )
        closer.start()
        # Give the close every chance to run inside the window.  With
        # the state resolved under the lock, the close blocks here
        # until the merge thread is released below.
        closer.join(timeout=0.5)
        agent.window_release.set()
        merger.join(timeout=30)
        closer.join(timeout=30)
        assert not merger.is_alive() and not closer.is_alive()

        disposals = agent.merge_calls + agent.release_calls
        assert disposals == 1, (
            f"BUG: worktree disposed {disposals} times "
            f"(merge={agent.merge_calls}, release={agent.release_calls}) "
            "— the merge claimed a state the close had already "
            "unregistered and both paths acted on the same worktree"
        )
        assert results["merge"]["success"] is (agent.merge_calls == 1), (
            f"merge outcome {results['merge']!r} disagrees with the "
            f"number of merges that ran ({agent.merge_calls})"
        )
        # Whatever the ordering, the tab ends up fully disposed: the
        # close either ran first or was deferred and completed by the
        # merge's ``_dispose_if_closed``.
        assert agent_state.find_by_tab(tab_id) is None, (
            "tab state leaked after close + merge"
        )

    def test_early_refusals_resolved_under_the_lock(self) -> None:
        """Every pre-claim refusal still answers exactly as before."""
        act = self.server._handle_worktree_action
        assert act("merge", "no-such-tab")["message"] == (
            "Worktree mode is not enabled"
        )
        tab_id = "audit0902-refusals"
        agent = _WindowAgent(Path(self.tmpdir))
        state = agent_state.AgentState(
            f"task-for-{tab_id}", tab_id=tab_id, server_owned=True,
        )
        state.use_worktree = False
        agent_state.register(state)
        assert act("merge", tab_id)["message"] == (
            "Worktree mode is not enabled"
        )
        state.use_worktree = True
        assert act("merge", tab_id)["message"] == (
            "No pending worktree changes to act on"
        ), "agent-less state"
        state.agent = agent
        agent._wt = None
        assert act("merge", tab_id)["message"] == (
            "No pending worktree changes to act on"
        ), "agent without a pending worktree"
        agent._wt = GitWorktree(
            repo_root=None,  # type: ignore[arg-type]
            branch="kiss/wt-audit0902-b",
            original_branch="main",
            wt_dir=Path(self.tmpdir) / "wt",
            baseline_commit=None,
        )
        assert act("bogus", tab_id)["message"] == (
            "Unknown action: bogus"
        ), "the mode / pending checks come first, then the action"
        assert act("discard", tab_id)["message"] == (
            "No pending worktree changes to act on"
        ), "pending worktree with an unknown repository root"
        assert agent.merge_calls == 0 and agent.release_calls == 0


if __name__ == "__main__":
    unittest.main()
