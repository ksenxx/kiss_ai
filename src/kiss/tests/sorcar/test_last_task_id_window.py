# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``_last_task_id`` must never be ``None`` mid-run.

``ChatSorcarAgent.run`` used to clear ``self._last_task_id = None`` on
entry and only re-publish it after ``_add_task``.  Between those two
statements sit a SQLite read (``build_chat_prompt``'s chat/task-chain
context load) and a SQLite write — a window of tens of milliseconds,
far longer under DB contention.  Three server threads read the
attribute during that window (``commands._owner_task_id``,
``merge_flow._state_task_key`` and the printer's broadcast fan-out),
and all three resolve ``None``: the user's queued message, generated
commit message or merge action is then stamped with no task id and
mis-routed or dropped.  ``_task_id_lock`` did not help, because only
the writer ever took it.

The tests below use real threads and the real server-side readers.
The window is opened deterministically by a subclass that parks
inside the REAL ``build_chat_prompt`` — no sleeps, no patching of
production code.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any, cast

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state
from kiss.server.commands import _owner_task_id
from kiss.server.merge_flow import _state_task_key
from kiss.tests.sorcar.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    wait_for,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-f4-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


class _ParkingAgent(ChatSorcarAgent):
    """Real agent that parks inside the real prompt-building step.

    ``build_chat_prompt`` is the first of the two SQLite round-trips
    that make up the window under test, so stopping there reproduces
    the exact interleaving without altering what the code does.
    """

    def __init__(self, name: str) -> None:
        """Create the agent with its two synchronisation events."""
        super().__init__(name)
        self.parked = threading.Event()
        self.release = threading.Event()

    def build_chat_prompt(self, prompt: str) -> str:
        """Signal that the window is open, wait, then build for real."""
        self.parked.set()
        self.release.wait(30)
        return super().build_chat_prompt(prompt)


def _run_task(
    env: IsolatedKissHome, agent: ChatSorcarAgent, server: StandInModelServer,
    prompt: str,
) -> str:
    """Run one real task through the stand-in model."""
    return agent.run(
        prompt_template=prompt,
        model_name=STANDIN_MODEL,
        model_config=server.model_config,
        work_dir=str(env.repo),
    )


def test_followup_run_never_exposes_a_none_task_id(
    env: IsolatedKissHome,
) -> None:
    """A second run must not blank the id its readers depend on.

    The server threads that resolve "which task is this tab running?"
    read the attribute directly.  While run #2 is still allocating its
    row, the previous run's id is stale but valid; ``None`` is not an
    answer any of those callers can use.
    """
    server = StandInModelServer(lambda request: finish_response("f4 done"))
    agent = _ParkingAgent("f4-window")
    try:
        _run_task(env, agent, server, "FIRST TASK")
        first_task_id = agent._last_task_id
        assert isinstance(first_task_id, str) and first_task_id

        state = agent_state.AgentState("f4-state", agent=cast(Any, agent))
        second: dict[str, Any] = {}

        def run_second() -> None:
            try:
                second["result"] = _run_task(env, agent, server, "SECOND TASK")
            except BaseException as exc:  # noqa: BLE001 — recorded below
                second["error"] = exc

        thread = threading.Thread(target=run_second, daemon=True)
        thread.start()
        try:
            assert agent.parked.wait(30), "run #2 never reached the window"

            assert agent._last_task_id is not None, (
                "run #2 blanked _last_task_id while its history row was "
                "still being written; every server-thread reader now "
                "resolves None"
            )
            assert _owner_task_id(state) == first_task_id, (
                "a user message queued during run #2's startup would be "
                "stamped with no task id"
            )
            assert _state_task_key(state) == first_task_id, (
                "a merge/commit-message action during run #2's startup "
                "would resolve no task id"
            )
        finally:
            agent.release.set()

        thread.join(timeout=60)
        assert not thread.is_alive()
        assert second.get("error") is None, second.get("error")
        assert isinstance(agent._last_task_id, str) and agent._last_task_id
        assert agent._last_task_id != first_task_id, (
            "run #2 must publish its own task id once allocated"
        )
        assert _owner_task_id(state) == agent._last_task_id
    finally:
        agent.release.set()
        server.stop()


def test_concurrent_readers_always_see_a_usable_task_id(
    env: IsolatedKissHome,
) -> None:
    """Hammer the readers from another thread across three real runs.

    Once the agent has run once, no reader may ever observe ``None``
    or an id that is not a real ``task_history`` row id.
    """
    server = StandInModelServer(lambda request: finish_response("f4 done"))
    agent = ChatSorcarAgent("f4-hammer")
    state = agent_state.AgentState("f4-hammer-state", agent=cast(Any, agent))
    observed: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            observed.append(_owner_task_id(state))

    try:
        _run_task(env, agent, server, "WARMUP TASK")
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            for i in range(3):
                _run_task(env, agent, server, f"HAMMER TASK {i}")
            assert wait_for(lambda: len(observed) > 1000, timeout=10.0)
        finally:
            stop.set()
            thread.join(timeout=30)
    finally:
        server.stop()

    assert "" not in observed, (
        "a concurrent reader observed an unresolvable task id while a "
        "follow-up run was starting up"
    )
