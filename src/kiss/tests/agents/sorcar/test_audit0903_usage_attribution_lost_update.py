# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 0903: attribution of sub-agent spend must not lose updates.

``_attribute_sub_usage`` performs a read-modify-write of the parent
agent's cumulative usage counters (``budget_used``,
``total_tokens_used``, ``total_steps``).  It has two callers that run
on DIFFERENT threads of the same agent:

* the agent thread — ``SorcarAgent._run_tasks_parallel``'s ``finally``
  banks a finished fan-out's totals, and ``_attribute_tts_usage`` banks
  a ``talk`` synthesis call's spend; and
* server threads — ``reclaim_abandoned_subagents`` (called by
  worktree cleanup / teardown / discard, per its own docstring
  "the agent thread and server-side worktree cleanup call this
  concurrently") banks abandoned children's spend while holding only
  ``_abandoned_lock``.

``_abandoned_lock`` serializes reclaimers against each other, but the
agent thread's direct ``_attribute_sub_usage`` calls never take it, so
two concurrent read-modify-writes could interleave and one side's
increment silently vanished from the task's accounting (a lost
update).  The fix serializes every writer of the cumulative counters
under the agent's ``_usage_lock`` (``_attribute_sub_usage`` and
``RelentlessAgent._accumulate_usage``).

No mocks/patches of the code under test: real ``ChatSorcarAgent``
objects, real ``_AbandonedSubagent`` items with real completed
futures, real threads.  ``sys.setswitchinterval`` only shortens the
GIL preemption slice so the pre-fix interleaving is hit reliably; the
post-fix assertion is exact and timing-independent.
"""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import (
    _AbandonedSubagent,
    _attribute_sub_usage,
)

# Enough concurrent read-modify-writes that a single missed
# serialization loses updates with near-certainty at a 1 µs GIL slice
# (the pre-fix run reproducibly lost >25% of them).
_DIRECT_CALLS = 20_000
_RECLAIM_CALLS = 4_000


def _make_done_future(pool: ThreadPoolExecutor) -> object:
    """Return a real, already-completed future."""
    future = pool.submit(lambda: "done")
    assert future.result(timeout=10) == "done"
    return future


@pytest.fixture
def fast_gil_switch() -> object:
    """Shrink the GIL switch interval for the duration of one test."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield None
    finally:
        sys.setswitchinterval(previous)


def test_reclaim_and_direct_attribution_never_lose_an_update(
    fast_gil_switch: object,
) -> None:
    """Concurrent reclaim (server thread) + direct attribution (agent
    thread) must bank every single increment exactly once."""
    parent = ChatSorcarAgent("audit0903-usage-lock-parent")
    parent.budget_used = 0.0
    parent.total_tokens_used = 0
    parent.total_steps = 0

    with ThreadPoolExecutor(max_workers=1) as pool:
        items: list[_AbandonedSubagent] = []
        for i in range(_RECLAIM_CALLS):
            child = ChatSorcarAgent(f"audit0903-usage-lock-child-{i}")
            child.budget_used = 0.0
            child.total_tokens_used = 1
            child.total_steps = 1
            items.append(
                _AbandonedSubagent(
                    _make_done_future(pool),  # type: ignore[arg-type]
                    child,
                    (0.0, 0, 0),
                )
            )

        def reclaimer() -> None:
            # The real server-side path: feed one abandoned child at a
            # time and bank it through reclaim_abandoned_subagents, so
            # every reclaim call performs one read-modify-write of the
            # parent's counters (under _abandoned_lock, exactly like
            # production).
            for item in items:
                with parent._abandoned_lock:
                    parent._abandoned_subagents.append(item)
                assert parent.reclaim_abandoned_subagents()

        def direct_attributor() -> None:
            # The real agent-thread path: what _run_tasks_parallel's
            # finally and _attribute_tts_usage do.
            for _ in range(_DIRECT_CALLS):
                _attribute_sub_usage(parent, 0.0, 1, 1)

        threads = [
            threading.Thread(target=reclaimer),
            threading.Thread(target=direct_attributor),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    expected = _DIRECT_CALLS + _RECLAIM_CALLS
    assert parent.total_tokens_used == expected, (
        f"lost updates: banked {parent.total_tokens_used} of {expected} tokens"
    )
    assert parent.total_steps == expected, (
        f"lost updates: banked {parent.total_steps} of {expected} steps"
    )
    with parent._abandoned_lock:
        assert parent._abandoned_subagents == []


def test_concurrent_direct_attributions_never_lose_an_update(
    fast_gil_switch: object,
) -> None:
    """Two direct attributors (e.g. a fan-out bank racing a TTS bank on
    a reused agent) also serialize."""
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


def test_attribution_tolerates_agents_without_a_usage_lock() -> None:
    """A bare agent-shaped object (no ``_usage_lock``) is still banked.

    ``_attribute_sub_usage`` is typed ``agent: Any``; the lock is a
    Sorcar-agent attribute, so the function must fall back gracefully
    for minimal agent objects.
    """
    bare = SimpleNamespace(
        budget_used=0.0, total_tokens_used=0, total_steps=0, printer=None,
    )
    _attribute_sub_usage(bare, 1.5, 10, 2)
    assert (bare.budget_used, bare.total_tokens_used, bare.total_steps) == (
        1.5, 10, 2,
    )
