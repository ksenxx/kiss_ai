# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 0903: attribution of sub-agent spend must not lose updates.

``_attribute_sub_usage`` performs a read-modify-write of the parent
agent's cumulative usage counters (``budget_used``,
``total_tokens_used``, ``total_steps``).  It is called from DIFFERENT
threads of the same agent: the agent thread (``_attribute_tts_usage``
banks a ``talk`` synthesis call's spend) and server threads (a
finished ``run_agent`` sub-task's attribution from the dispatch job
thread, the merge-conflict resolver).  Two concurrent
read-modify-writes could interleave and one side's increment silently
vanish from the task's accounting (a lost update).  The fix commits
every increment as one atomic record through
``RelentlessAgent._attribute_usage``.

No mocks/patches of the code under test: real ``ChatSorcarAgent``
objects, real threads.  ``sys.setswitchinterval`` only shortens the
GIL preemption slice so the pre-fix interleaving is hit reliably; the
post-fix assertion is exact and timing-independent.
"""

from __future__ import annotations

import sys
import threading

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage

# Enough concurrent read-modify-writes that a single missed
# serialization loses updates with near-certainty at a 1 µs GIL slice
# (the pre-fix run reproducibly lost >25% of them).
_DIRECT_CALLS = 20_000


@pytest.fixture
def fast_gil_switch() -> object:
    """Shrink the GIL switch interval for the duration of one test."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield None
    finally:
        sys.setswitchinterval(previous)


def test_concurrent_direct_attributions_never_lose_an_update(
    fast_gil_switch: object,
) -> None:
    """Concurrent attributors (e.g. a sub-task bank racing a TTS bank
    on a reused agent) serialize."""
    parent = ChatSorcarAgent("audit0903-usage-lock-direct")
    parent.budget_used = 0.0
    parent.total_tokens_used = 0
    parent.total_steps = 0

    def attributor() -> None:
        for _ in range(_DIRECT_CALLS):
            _attribute_sub_usage(parent, 0.0, 1, 1)

    threads = [threading.Thread(target=attributor) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    expected = 4 * _DIRECT_CALLS
    assert parent.total_tokens_used == expected, (
        f"lost updates: banked {parent.total_tokens_used} of {expected} tokens"
    )
    assert parent.total_steps == expected
