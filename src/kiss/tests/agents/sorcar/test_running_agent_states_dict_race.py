# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for races on ``_RunningAgentState.running_agent_states``.

The process-global registry mapping frontend tab id →
:class:`_RunningAgentState` is mutated by code in both
``kiss.agents.sorcar`` (parallel sub-agent spawning in
``ChatSorcarAgent._run_tasks_parallel``, worktree register /
unregister in ``WorktreeSorcarAgent``) and ``kiss.agents.vscode``
(server tab lifecycle).  When those mutations were not serialised
against the VS Code server's iteration loops, the iteration could
observe the dict mid-resize and raise
``RuntimeError: dictionary changed size during iteration`` — and
even when the bare-name iteration did not crash, the scan could
miss / double-count tabs.

These integration tests interleave the sorcar-side mutators with a
typical vscode-side iteration loop (held under
``VSCodeServer._state_lock``) and assert no exception is raised.

A small ``random.uniform(0, 0.005)`` sleep is inserted before each
suspected racing statement to widen the interleaving window and
make the race deterministic across runs.
"""

from __future__ import annotations

import random
import threading
import time

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.server import VSCodeServer


@pytest.fixture(autouse=True)
def _clean_registry():
    """Restore the process-global registry between tests."""
    saved = dict(_RunningAgentState.running_agent_states)
    _RunningAgentState.running_agent_states.clear()
    saved_agents = dict(ChatSorcarAgent.running_agents)
    ChatSorcarAgent.running_agents.clear()
    yield
    _RunningAgentState.running_agent_states.clear()
    _RunningAgentState.running_agent_states.update(saved)
    ChatSorcarAgent.running_agents.clear()
    ChatSorcarAgent.running_agents.update(saved_agents)


def _iterate_under_state_lock(
    server: VSCodeServer,
    stop: threading.Event,
    errors: list[BaseException],
) -> None:
    """Mimic VSCodeServer iteration loops.

    Iterates ``running_agent_states.values()`` under
    ``self._state_lock`` exactly as production code does in e.g.
    ``_get_running_tasks`` and ``_resolve_parent_tab_id_for_sub``.
    Yields between every entry so that a concurrent producer thread
    has many opportunities to mutate the dict mid-iteration.
    """
    try:
        while not stop.is_set():
            with server._state_lock:
                count = 0
                for st in _RunningAgentState.running_agent_states.values():
                    if st.is_subagent:
                        count += 1
                    # ``time.sleep(0)`` is a hint to the scheduler;
                    # combined with a tiny random sleep it makes the
                    # iteration straddle multiple GIL preemption
                    # boundaries.
                    time.sleep(random.uniform(0, 0.001))
                assert count >= 0
    except BaseException as exc:  # pragma: no cover — race surface
        errors.append(exc)


class TestParallelSubagentMutationRace:
    """``ChatSorcarAgent._run_tasks_parallel`` registry mutations are race-safe.

    Reproduces the race that existed before
    :attr:`_RunningAgentState._registry_lock` was bound to
    ``VSCodeServer._state_lock``: a parallel sub-agent thread would
    mutate ``running_agent_states`` while a server-side iteration
    loop was scanning, raising
    ``RuntimeError: dictionary changed size during iteration``.

    The producer call site lives inside a nested closure
    (``_run_single``) so we drive it indirectly by replaying the
    exact two-statement sequence the production code executes — but
    *without* acquiring ``_registry_lock`` from the test side, so
    the assertion only passes when the production code itself
    acquires the lock.
    """

    @staticmethod
    def _producer_via_helpers(tab_id: str) -> None:
        """Insert + pop a sub-state via the production-code helpers.

        ``ChatSorcarAgent._run_tasks_parallel`` now routes its dict
        mutations through :meth:`_RunningAgentState.register` /
        :meth:`_RunningAgentState.unregister`; calling the same
        helpers from the test exercises the exact code path the
        production hot-path uses.  When those helpers do not hold
        ``_registry_lock``, the reader thread observes a resize;
        with the lock in place, the operations serialise.
        """
        sub_state = _RunningAgentState(tab_id, "")
        sub_state.is_subagent = True
        time.sleep(random.uniform(0, 0.005))
        _RunningAgentState.register(tab_id, sub_state)
        time.sleep(random.uniform(0, 0.005))
        _RunningAgentState.unregister(tab_id)

    def test_worktree_register_unregister_does_not_break_iteration(
        self,
    ) -> None:
        """``WorktreeSorcarAgent._register_running_state`` is race-safe.

        Drives the real production methods (no replica): the helper
        scans the dict for an existing entry and then inserts;
        without the registry lock, a concurrent reader iterating
        under ``_state_lock`` races the insert.  With the fix, the
        helper acquires ``_registry_lock`` (the same object the
        reader holds via ``_state_lock``), serialising the two.
        """
        server = VSCodeServer()
        errors: list[BaseException] = []
        stop = threading.Event()
        reader = threading.Thread(
            target=_iterate_under_state_lock,
            args=(server, stop, errors),
            daemon=True,
        )
        reader.start()

        def _register_unregister(idx: int) -> None:
            try:
                agent = WorktreeSorcarAgent(f"WT-{idx}")
                agent._chat_id = f"chat-{idx}"
                time.sleep(random.uniform(0, 0.005))
                agent._register_running_state()
                time.sleep(random.uniform(0, 0.005))
                agent._unregister_running_state()
            except BaseException as exc:  # pragma: no cover — race surface
                errors.append(exc)

        try:
            for batch in range(10):
                workers = [
                    threading.Thread(
                        target=_register_unregister,
                        args=(batch * 10 + i,),
                        daemon=True,
                    )
                    for i in range(10)
                ]
                for w in workers:
                    w.start()
                for w in workers:
                    w.join(timeout=5)
        finally:
            stop.set()
            reader.join(timeout=5)
        assert not errors, (
            f"Race detected — got {len(errors)} exception(s); "
            f"first: {type(errors[0]).__name__}: {errors[0]}"
        )

    def test_parallel_subagent_register_unregister_does_not_break_iteration(
        self,
    ) -> None:
        """Real-thread emulation of ``_run_tasks_parallel`` insert/pop.

        We monkey-patch the ``ThreadPoolExecutor.map`` step out by
        calling a tiny replica that performs the same dict mutations
        the production code performs — but in a controlled, fast
        loop so the test runs in milliseconds.  The replica is kept
        in lockstep with the production code: when the production
        ``__setitem__`` / ``pop`` calls are wrapped under
        ``_registry_lock``, dict atomicity is restored; when they
        are not, the reader thread observes a resize.

        This test must FAIL on the unfixed code (before the
        ``with _RunningAgentState._registry_lock:`` wrappers were
        added to ``_run_tasks_parallel``) and PASS on the fixed
        code.  See the ``_producer_replica`` docstring above for
        why no test-side lock is taken.
        """
        server = VSCodeServer()
        errors: list[BaseException] = []
        stop = threading.Event()
        reader = threading.Thread(
            target=_iterate_under_state_lock,
            args=(server, stop, errors),
            daemon=True,
        )
        reader.start()

        def _worker(idx: int) -> None:
            try:
                self._producer_via_helpers(f"sub_{idx}")
            except BaseException as exc:  # pragma: no cover — race surface
                errors.append(exc)

        try:
            for batch in range(20):
                workers = [
                    threading.Thread(
                        target=_worker, args=(batch * 10 + i,), daemon=True,
                    )
                    for i in range(10)
                ]
                for w in workers:
                    w.start()
                for w in workers:
                    w.join(timeout=5)
        finally:
            stop.set()
            reader.join(timeout=5)
        assert not errors, (
            f"Race detected — got {len(errors)} exception(s); "
            f"first: {type(errors[0]).__name__}: {errors[0]}"
        )


class TestStateLockIsRegistryLock:
    """The server's ``_state_lock`` MUST be the same object as ``_registry_lock``.

    The fix relies on this identity so that producers in
    ``kiss.agents.sorcar`` (which can only reach
    ``_RunningAgentState._registry_lock``) and consumers in
    ``kiss.agents.vscode`` (which acquire ``self._state_lock``)
    serialise against each other.  If a future refactor splits the
    two locks again, the registry-resize races come back.
    """

    def test_state_lock_is_registry_lock(self) -> None:
        server = VSCodeServer()
        assert server._state_lock is _RunningAgentState._registry_lock
