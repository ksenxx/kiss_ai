# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""An ``is_merging`` claim whose thread died must not wedge the daemon.

Field incident (``~/.kiss/kiss-web-stderr.log``, 2026-09-22 00:44 UTC):
the thread disposing a stopped task's closed tab escaped with a
``KeyboardInterrupt`` raised inside the auto-commit's LLM call, after
claiming ``state.is_merging`` and before releasing it.  The claim then
stayed set on a registered ``use_worktree`` state for hours, so every
prompt the classifier routed to the main tree was refused with
"A worktree merge is in progress. Wait for it to finish before starting
a task." and the closed tab counted as active forever (which also made
the extension defer the daemon restart that would have cleared it).

Every claim records its claiming thread in ``merge_thread`` and is
released by that thread, so a claim whose thread is dead is a leak by
definition.  :meth:`AgentState.merge_in_progress` releases such a claim
and every gate that used to read ``is_merging`` directly goes through
it.  These tests drive the real server with real ``run`` commands and a
real worktree agent; only the agent's parent-class ``run`` is stubbed
(as in :mod:`test_deferred_disposal_stale_stop_signal`) and
``printer.broadcast`` is captured.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.server.task_runner import _wt_merge_on_repo
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome

_MERGE_IN_PROGRESS = "A worktree merge is in progress"
_TAB_MERGE_IN_PROGRESS = "Cannot run a task while a merge is in progress"


def _dead_thread() -> threading.Thread:
    """A thread that claimed nothing and has already finished."""
    thread = threading.Thread(target=lambda: None, name="dead-claimant")
    thread.start()
    thread.join()
    return thread


class _Env:
    """Isolated KISS_HOME + repo + a real server with captured broadcasts."""

    def __init__(self) -> None:
        self.isolated = IsolatedKissHome("kiss-orphaned-claim-")
        self.repo: Path = self.isolated.repo
        self.server = VSCodeServer()
        self.server.work_dir = str(self.repo)
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()
        self.server.printer.broadcast = self._capture  # type: ignore[assignment]
        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run
        self._parent_class.run = _stub_run_ok

    def _capture(self, event: dict[str, Any]) -> None:
        with self._events_lock:
            self.events.append(event)

    def event_types(self) -> list[str]:
        with self._events_lock:
            return [str(e.get("type", "")) for e in self.events]

    def error_texts(self) -> list[str]:
        with self._events_lock:
            return [str(e.get("text", "")) for e in self.events if e.get("type") == "error"]

    def cleanup(self) -> None:
        self._parent_class.run = self._original_run
        for state in agent_state.snapshot():
            if state.agent is not None and state.agent._wt_pending:
                try:
                    state.agent.discard()
                except Exception:  # pragma: no cover — cleanup best-effort
                    pass
            agent_state.unregister(state.task_id, state)
        self.isolated.cleanup()

    def run_command(self, tab_id: str, prompt: str, *, use_worktree: bool) -> None:
        """Drive a real ``run`` command and wait for its task thread."""
        self.server._handle_command({
            "type": "run",
            "prompt": prompt,
            "workDir": str(self.repo),
            "tabId": tab_id,
            "useWorktree": use_worktree,
            "autoCommit": True,
            "model": "",
        })
        state = agent_state.find_by_tab(tab_id)
        if state is None:
            return
        thread = state.task_thread
        if thread is not None:
            thread.join(timeout=120)
            assert not thread.is_alive(), "the task thread did not finish"

    def register_claimed_state(
        self, tab_id: str, claimant: threading.Thread | None, *, closed: bool,
    ) -> agent_state.AgentState:
        """Register an idle ``use_worktree`` state that holds an
        ``is_merging`` claim recorded for *claimant*, the way the disposal
        and merge paths claim it.  The agent has no worktree any more (as
        in the incident, where the retire had already dropped it), so its
        repository is unknown and ``_wt_merge_on_repo`` conservatively
        treats the claim as blocking every repository."""
        agent = WorktreeSorcarAgent(f"orphaned-{tab_id}")
        with self.server._state_lock:
            state = agent_state.AgentState(
                f"task-for-{tab_id}", tab_id=tab_id, server_owned=True, agent=agent,
            )
            state.use_worktree = True
            state.frontend_closed = closed
            state.is_merging = True
            state.merge_thread = claimant
            agent_state.register(state)
        if not closed:
            self.server.tab_registry.open_tab(tab_id, "orphaned")
        return state


def _stub_run_ok(self_agent: object, **kwargs: object) -> str:
    return "ok"


@pytest.fixture
def env() -> Iterator[_Env]:
    e = _Env()
    try:
        yield e
    finally:
        e.cleanup()


def test_dead_claimant_no_longer_blocks_main_tree_runs(
    env: _Env, caplog: pytest.LogCaptureFixture,
) -> None:
    """The incident: a claim left by a dead thread on another tab must
    not refuse a main-tree prompt; the gate releases it and says so."""
    stale = env.register_claimed_state("wedged-tab", _dead_thread(), closed=True)

    with caplog.at_level(logging.WARNING, logger="kiss.server.agent_state"):
        env.run_command("next-prompt", "cron list", use_worktree=False)

    assert [t for t in env.error_texts() if _MERGE_IN_PROGRESS in t] == []
    assert "task_done" in env.event_types(), env.event_types()
    assert stale.is_merging is False
    assert stale.merge_thread is None
    assert any(
        "Releasing orphaned merge claim on tab wedged-tab" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_live_claimant_still_blocks_main_tree_runs_and_is_logged(
    env: _Env, caplog: pytest.LogCaptureFixture,
) -> None:
    """A claim whose thread is still running is real: the main-tree run is
    refused as before, and the refusal now names the blocking tab."""
    release = threading.Event()
    claimant = threading.Thread(target=release.wait, name="live-claimant")
    claimant.start()
    try:
        live = env.register_claimed_state("merging-tab", claimant, closed=False)
        with caplog.at_level(logging.WARNING, logger="kiss.server.task_runner"):
            env.run_command("refused-prompt", "cron list", use_worktree=False)
    finally:
        release.set()
        claimant.join(timeout=10)

    refused = [t for t in env.error_texts() if _MERGE_IN_PROGRESS in t]
    assert len(refused) == 1, env.error_texts()
    assert live.is_merging is True
    assert live.merge_thread is claimant
    assert any(
        "Refusing main-tree run on tab refused-prompt" in r.getMessage()
        and "tab merging-tab" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_dead_claimant_on_the_same_tab_lets_a_new_run_start(env: _Env) -> None:
    """The run admission gate of the tab itself (``commands._cmd_run``)
    also goes through the predicate: a prompt on the wedged tab runs."""
    stale = env.register_claimed_state("same-tab", _dead_thread(), closed=False)

    env.run_command("same-tab", "hello", use_worktree=False)

    assert [t for t in env.error_texts() if _TAB_MERGE_IN_PROGRESS in t] == []
    assert "task_done" in env.event_types(), env.event_types()
    assert stale.is_merging is False


def test_dead_claimant_does_not_keep_a_closed_tab_busy(env: _Env) -> None:
    """The closed tab counted as active forever (``busy()``); with the
    claim released the deferred disposal drops it."""
    stale = env.register_claimed_state("closed-tab", _dead_thread(), closed=True)

    with env.server._state_lock:
        assert stale.busy() is False
    env.server._dispose_if_closed("closed-tab")

    assert agent_state.find_by_tab("closed-tab") is None
    assert not _wt_merge_on_repo(stale, env.repo)


def test_dead_claimant_is_not_reported_as_an_active_task(env: _Env) -> None:
    """``activeTasksQuery`` (the extension's pre-restart check) counted the
    wedged tab as active forever, which deferred the very daemon restart
    that would have cleared the claim.  The unlocked snapshot must
    neither report it nor race the release (it locks inside)."""
    from kiss.server.web_server import _snapshot_active_tabs

    stale = env.register_claimed_state("phantom-active", _dead_thread(), closed=True)

    assert _snapshot_active_tabs() == []
    assert stale.is_merging is False


def test_main_tree_claim_refusal_is_logged(
    env: _Env, caplog: pytest.LogCaptureFixture,
) -> None:
    """The other silent main-tree refusal (a Discard / manual Git Commit
    holding the repository) now leaves a trace in the log too."""
    from kiss.server.server import MainTreeClaim

    holder: list[MainTreeClaim] = []
    with env.server._state_lock:
        assert env.server._claim_main_tree(env.repo, "manual commit", holder=holder)
    try:
        with caplog.at_level(logging.WARNING, logger="kiss.server.task_runner"):
            env.run_command("claimed-prompt", "cron list", use_worktree=False)
    finally:
        env.server._release_main_tree_claim(holder[0])

    refused = [t for t in env.error_texts() if "manual commit is in progress" in t]
    assert len(refused) == 1, env.error_texts()
    assert any(
        "Refusing main-tree run on tab claimed-prompt: manual commit in progress"
        in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_claims_without_a_dead_thread_are_kept(env: _Env) -> None:
    """Only a claim whose recorded thread has died is released: a claim
    with no recorded thread or with a not-yet-started thread stands."""
    no_thread = env.register_claimed_state("no-thread", None, closed=False)
    unstarted = env.register_claimed_state(
        "unstarted", threading.Thread(target=lambda: None), closed=False,
    )

    with env.server._state_lock:
        assert no_thread.merge_in_progress() is True
        assert no_thread.busy() is True
        assert unstarted.merge_in_progress() is True
        assert unstarted.busy() is True
    assert _wt_merge_on_repo(no_thread, env.repo)
    assert _wt_merge_on_repo(unstarted, env.repo)

    idle = env.register_claimed_state("idle", None, closed=False)
    idle.is_merging = False
    with env.server._state_lock:
        assert idle.merge_in_progress() is False
        assert idle.busy() is False


def test_dead_claimant_does_not_refuse_a_manual_worktree_action(env: _Env) -> None:
    """The merge/discard busy guard (``_check_worktree_busy``) releases the
    orphaned claim instead of answering "A merge or discard is already in
    progress on this tab"."""
    stale = env.register_claimed_state("discard-tab", _dead_thread(), closed=False)
    agent = stale.agent
    assert isinstance(agent, WorktreeSorcarAgent)
    assert agent._try_setup_worktree(env.repo, str(env.repo)) is not None
    wt = agent._wt
    assert wt is not None
    (wt.wt_dir / "scratch.txt").write_text("scratch\n", encoding="utf-8")

    result = env.server._handle_worktree_action("discard", "discard-tab")

    assert result.get("success") is True, result
    assert agent._wt is None
    assert not wt.wt_dir.exists()
    assert stale.is_merging is False
