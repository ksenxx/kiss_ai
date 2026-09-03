# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review-2 #1: a tab close whose preserve marker cannot be written must
keep the worktree protected from a same-daemon reclaim.

``WorktreeSorcarAgent._keep_for_review`` fails closed — when the
``kiss-preserve`` marker cannot be written (``.git/config.lock`` held)
the agent keeps ``self._wt`` — but the tab-teardown paths in
:class:`VSCodeServer` (``_drop_tab_state`` / ``_dispose_if_closed``)
used to unregister the :class:`AgentState` BEFORE running that
preserve step.  ``JsonPrinter.live_worktree_branches`` derives the
reclaim exclusion set from registered states only, and
``reclaim_orphaned_worktrees`` exempts its own pid from the owner
protection, so the retained claim was invisible: the next task in the
same daemon squash-merged the unaccepted work into the user's branch
and deleted the worktree.

Everything here is real: a real git repository, a real
``.git/config.lock`` held during the close, the real server teardown,
the real reclaim pass with the exclusion set a second agent in the same
process would build, and a real retry once the lock is gone.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome


class _Env:
    """Isolated KISS_HOME + repo + a real server with captured broadcasts."""

    def __init__(self) -> None:
        self.isolated = IsolatedKissHome("kiss-audit0902-fix2-teardown-")
        self.repo: Path = self.isolated.repo
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()
        self.server.printer.broadcast = self._capture  # type: ignore[assignment]

    def _capture(self, event: dict[str, Any]) -> None:
        with self._events_lock:
            self.events.append(event)

    def cleanup(self) -> None:
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
        self.isolated.cleanup()

    def park_unaccepted_work(self, tab_id: str, *, active: bool = False) -> WorktreeSorcarAgent:
        """Register a tab whose agent holds a stopped task's uncommittable work."""
        agent = WorktreeSorcarAgent(f"audit0902-fix2-{tab_id}")
        # Auto-commit OFF + an uncommitted file: the preserve step must
        # keep the worktree directory instead of committing it.
        agent.auto_commit_enabled = False
        assert agent._try_setup_worktree(self.repo, str(self.repo)) is not None
        wt = agent._wt
        assert wt is not None
        (wt.wt_dir / "agent.txt").write_text("unaccepted work\n", encoding="utf-8")
        agent._pending_review = True
        agent.printer = self.server.printer  # type: ignore[attr-defined]
        with self.server._state_lock:
            state = agent_state.AgentState(
                f"task-for-{tab_id}", tab_id=tab_id, server_owned=True, agent=agent,
                is_task_active=active,
            )
            state.use_worktree = True
            state.auto_commit_mode = False
            agent_state.register(state)
        self.server.tab_registry.open_tab(tab_id, "audit")
        return agent

    def warnings(self) -> list[str]:
        with self._events_lock:
            return [
                str(e.get("message", "")) for e in self.events if e.get("type") == "warning"
            ]


@pytest.fixture
def env() -> Iterator[_Env]:
    e = _Env()
    try:
        yield e
    finally:
        e.cleanup()


def _wt_branches(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "branch", "--list", "kiss/wt-*"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [line.strip().lstrip("+* ").strip() for line in out.splitlines() if line.strip()]


def _same_daemon_reclaim(env: _Env) -> int:
    """Run the reclaim exactly as another task in this daemon would.

    ``_try_setup_worktree`` of a second agent builds its exclusion set
    through the printer's ``live_worktree_branches`` bridge (registered
    states only); this reproduces that wiring without a second LLM run.
    """
    other = WorktreeSorcarAgent("audit0902-fix2-other-tab")
    other.printer = env.server.printer  # type: ignore[attr-defined]
    return GitWorktreeOps.reclaim_orphaned_worktrees(
        env.repo, exclude_branches=other._live_worktree_branches(),
    )


def _assert_untouched(env: _Env, agent: WorktreeSorcarAgent, wt_dir: Path, branch: str) -> None:
    assert wt_dir.is_dir(), "the kept worktree directory was deleted"
    assert branch in _wt_branches(env.repo), "the kept branch was deleted"
    assert not (env.repo / "agent.txt").exists(), "unaccepted work was published to main"
    assert (wt_dir / "agent.txt").is_file()


def test_close_tab_with_marker_failure_survives_same_daemon_reclaim(env: _Env) -> None:
    """closeTab while ``.git/config.lock`` is held: the state stays registered,
    the live exclusion still names the branch, a same-process reclaim leaves
    the worktree alone, and the retry after the lock is gone writes the marker."""
    tab_id = "audit0902-fix2-close"
    agent = env.park_unaccepted_work(tab_id)
    wt = agent._wt
    assert wt is not None
    wt_dir, branch = wt.wt_dir, wt.branch

    lock = env.repo / ".git" / "config.lock"
    lock.write_text("")
    try:
        env.server._close_tab(tab_id)

        # (a) the agent kept its claim and the decision is not durable yet.
        assert agent._wt is wt, "the in-memory claim was dropped"
        assert agent._pending_review is True
        assert GitWorktreeOps._load_branch_config(env.repo, branch, "kiss-preserve") is None
        # (b) the state is still registered, so the live exclusion sees it.
        state = agent_state.find_by_tab(tab_id)
        assert state is not None and state.agent is agent, (
            "teardown unregistered the state although the keep decision is not durable"
        )
        assert state.busy() is False
        assert branch in env.server.printer.live_worktree_branches()
        # The user was told why the worktree is still pending.
        assert any("still pending" in w for w in env.warnings()), env.warnings()

        # A second task in the same daemon runs its reclaim pass now.
        assert _same_daemon_reclaim(env) == 0
        _assert_untouched(env, agent, wt_dir, branch)
        # (c) even a reclaim that sees NO live exclusion (the own-pid
        # exemption is what made this destructive) respects the claim.
        assert GitWorktreeOps.reclaim_orphaned_worktrees(env.repo, exclude_branches=set()) == 0
        _assert_untouched(env, agent, wt_dir, branch)
        # Still not on disk: only the process-local claim protected it.
        assert GitWorktreeOps._load_branch_config(env.repo, branch, "kiss-preserve") is None
    finally:
        if lock.exists():
            lock.unlink()

    # git config is writable again: the next disposal attempt for this
    # tab (the user closes it again / its chat is rebound to another
    # tab) makes the decision durable and only then drops the state.
    env.server._close_tab(tab_id)
    assert GitWorktreeOps.load_preserve_marker(env.repo, branch)
    assert agent._wt is None
    assert agent._pending_review is False
    assert agent_state.find_by_tab(tab_id) is None
    _assert_untouched(env, agent, wt_dir, branch)
    # With the marker durable the process-local claim is gone: a reclaim
    # in a fresh process would now rely on the marker alone.
    subprocess.run(
        ["git", "-C", str(env.repo), "config", "--unset", f"branch.{branch}.kiss-preserve"],
        check=True, capture_output=True,
    )
    assert not GitWorktreeOps.load_preserve_marker(env.repo, branch)
    GitWorktreeOps.save_preserve_marker(env.repo, branch)
    assert GitWorktreeOps.reclaim_orphaned_worktrees(env.repo, exclude_branches=set()) == 0
    _assert_untouched(env, agent, wt_dir, branch)


def test_deferred_disposal_with_marker_failure_keeps_state(env: _Env) -> None:
    """The deferred path (tab closed while its task ran; disposal at task
    end) must obey the same rule as the immediate path."""
    tab_id = "audit0902-fix2-deferred"
    agent = env.park_unaccepted_work(tab_id, active=True)
    wt = agent._wt
    assert wt is not None
    wt_dir, branch = wt.wt_dir, wt.branch

    # Closing a busy tab only marks it; the state stays.
    env.server._close_tab(tab_id)
    state = agent_state.find_by_tab(tab_id)
    assert state is not None and state.frontend_closed is True
    assert agent._wt is wt

    lock = env.repo / ".git" / "config.lock"
    lock.write_text("")
    try:
        # The task ends: the runner drops the lifecycle flag and disposes.
        with env.server._state_lock:
            state.is_task_active = False
        env.server._dispose_if_closed(tab_id)

        assert agent._wt is wt
        assert agent_state.find_by_tab(tab_id) is state
        assert state.busy() is False
        assert branch in env.server.printer.live_worktree_branches()
        assert _same_daemon_reclaim(env) == 0
        assert GitWorktreeOps.reclaim_orphaned_worktrees(env.repo, exclude_branches=set()) == 0
        _assert_untouched(env, agent, wt_dir, branch)
    finally:
        if lock.exists():
            lock.unlink()

    # A later lifecycle transition retries the deferred disposal.
    env.server._dispose_if_closed(tab_id)
    assert GitWorktreeOps.load_preserve_marker(env.repo, branch)
    assert agent._wt is None
    assert agent_state.find_by_tab(tab_id) is None
    _assert_untouched(env, agent, wt_dir, branch)


def test_concurrent_close_during_teardown_is_deferred_not_duplicated(env: _Env) -> None:
    """Two closes racing: the second must neither retire the worktree a
    second time nor drop the state the first one decided to keep."""
    tab_id = "audit0902-fix2-double"
    agent = env.park_unaccepted_work(tab_id)
    wt = agent._wt
    assert wt is not None
    wt_dir, branch = wt.wt_dir, wt.branch

    entered = threading.Event()
    release = threading.Event()
    calls = {"n": 0}
    calls_lock = threading.Lock()
    original = agent._preserve_pending_worktree_for_review

    def blocking_preserve() -> bool:
        with calls_lock:
            calls["n"] += 1
        entered.set()
        assert release.wait(timeout=30)
        return original()

    # Real agent, real preserve — only its entry is gated so the second
    # close provably arrives while the first teardown is in flight.
    agent._preserve_pending_worktree_for_review = blocking_preserve  # type: ignore[method-assign]

    lock = env.repo / ".git" / "config.lock"
    lock.write_text("")
    try:
        first = threading.Thread(target=env.server._close_tab, args=(tab_id,), daemon=True)
        first.start()
        assert entered.wait(timeout=30)
        second = threading.Thread(target=env.server._close_tab, args=(tab_id,), daemon=True)
        second.start()
        second.join(timeout=30)
        assert not second.is_alive(), "the second close blocked behind the first teardown"
        state = agent_state.find_by_tab(tab_id)
        assert state is not None and state.frontend_closed is True
        release.set()
        first.join(timeout=60)
        assert not first.is_alive()
    finally:
        if lock.exists():
            lock.unlink()

    assert calls["n"] == 1, "the worktree was retired more than once"
    assert agent._wt is wt
    assert agent_state.find_by_tab(tab_id) is not None
    _assert_untouched(env, agent, wt_dir, branch)

    release.set()
    env.server._close_tab(tab_id)
    assert calls["n"] == 2
    assert agent._wt is None
    assert agent_state.find_by_tab(tab_id) is None
    assert GitWorktreeOps.load_preserve_marker(env.repo, branch)


def test_close_tab_without_pending_worktree_unregisters(env: _Env) -> None:
    """Regression guard: a tab whose agent holds no worktree is still
    unregistered by the close, and a durable preserve still unregisters."""
    tab_id = "audit0902-fix2-plain"
    with env.server._state_lock:
        agent_state.register(
            agent_state.AgentState(f"task-for-{tab_id}", tab_id=tab_id, server_owned=True),
        )
    env.server.tab_registry.open_tab(tab_id, "audit")
    env.server._close_tab(tab_id)
    assert agent_state.find_by_tab(tab_id) is None

    tab_id = "audit0902-fix2-durable"
    agent = env.park_unaccepted_work(tab_id)
    wt = agent._wt
    assert wt is not None
    state = agent_state.find_by_tab(tab_id)
    assert state is not None
    # Deferred-disposal no-ops: nothing to dispose, tab still shown, tab busy.
    env.server._dispose_if_closed("")
    env.server._dispose_if_closed(tab_id)
    assert agent_state.find_by_tab(tab_id) is state and agent._wt is wt
    with env.server._state_lock:
        state.frontend_closed = True
        state.is_task_active = True
    env.server._dispose_if_closed(tab_id)
    assert agent_state.find_by_tab(tab_id) is state and agent._wt is wt
    with env.server._state_lock:
        state.frontend_closed = False
        state.is_task_active = False

    env.server._close_tab(tab_id)
    assert agent._wt is None
    assert agent_state.find_by_tab(tab_id) is None
    assert GitWorktreeOps.load_preserve_marker(env.repo, wt.branch)
    assert wt.wt_dir.is_dir()
    # Closing a tab that never had a state is still harmless.
    env.server._close_tab("audit0902-fix2-unknown")
    assert agent_state.find_by_tab("audit0902-fix2-unknown") is None
