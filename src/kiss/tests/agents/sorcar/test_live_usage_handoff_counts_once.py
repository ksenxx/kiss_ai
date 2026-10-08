# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A session being banked is counted exactly once by live usage readers.

``RelentlessAgent.perform_task`` banks a finished session into the
ledger BEFORE dropping it from ``_current_executor``.  A server-thread
reader that sums "banked + live" between those two statements used to
see the session twice; ``live_usage_snapshot`` (reached through
``_live_agent_usage``) decides from the same ledger prefix it summed
whether the live session is already in it.

Real classes, real threads, no doubles: the reader thread polls while
the agent thread performs the handoff thousands of times.
"""

from __future__ import annotations

import threading

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.agents.sorcar.sorcar_agent import _live_agent_usage
from kiss.core.kiss_agent import KISSAgent


def _spent_executor(name: str) -> KISSAgent:
    """Return a real KISSAgent that has spent $1 / 100 tokens / 3 steps."""
    executor = KISSAgent(name)
    executor.budget_used = 1.0
    executor.total_tokens_used = 100
    executor.step_count = 3
    return executor


def test_reader_never_sees_a_session_twice_during_the_handoff() -> None:
    """Every read during bank-then-clear is the banked total, or that plus one session."""
    agent = RelentlessAgent("handoff")
    sessions = 400
    stop = threading.Event()
    bad: list[tuple[float, int, int]] = []

    def read_forever() -> None:
        while not stop.is_set():
            budget, tokens, steps = _live_agent_usage(agent)
            # The spend is a whole number of $1/100/3 sessions, and the
            # live session can add at most one more; a double count
            # breaks the budget/tokens/steps proportion or exceeds the
            # banked count by more than one session.
            whole = round(budget)
            if (
                abs(budget - whole) > 1e-9
                or tokens != whole * 100
                or steps != whole * 3
                or whole > sessions
            ):
                bad.append((budget, tokens, steps))

    readers = [threading.Thread(target=read_forever, daemon=True) for _ in range(4)]
    for reader in readers:
        reader.start()
    for i in range(sessions):
        executor = _spent_executor(f"s{i}")
        agent._current_executor = executor
        # The handoff exactly as perform_task does it: bank, then clear.
        agent._accumulate_usage(executor)
        agent._current_executor = None
    stop.set()
    for reader in readers:
        reader.join(timeout=30)
    assert not bad, f"{len(bad)} reads double-counted a session, e.g. {bad[:3]}"
    assert _live_agent_usage(agent) == (float(sessions), sessions * 100, sessions * 3)


def test_live_session_is_added_until_banked() -> None:
    """Before the bank the live session counts; after it, the ledger alone does."""
    agent = RelentlessAgent("live")
    executor = _spent_executor("s0")
    agent._current_executor = executor
    assert _live_agent_usage(agent) == (1.0, 100, 3)
    agent._accumulate_usage(executor)
    assert _live_agent_usage(agent) == (1.0, 100, 3), "banked but still current: once"
    agent._current_executor = None
    assert _live_agent_usage(agent) == (1.0, 100, 3)
