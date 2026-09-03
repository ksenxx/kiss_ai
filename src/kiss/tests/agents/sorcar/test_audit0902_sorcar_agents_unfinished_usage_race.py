# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 (sorcar-agents): abandoned-child usage must never regress.

``run_tasks_parallel`` fills each child's ``sub_usage`` slot from the
child's own ``finally`` (its FINAL figure, written just before its
future completes).  On the abandon path
``_collect_unfinished_usage`` refreshes the slots of children whose
futures are still pending from a LIVE read of the child's counters.

Both writers now share one lock (review fix #3; the deterministic
interleaving test lives in
``test_audit0902_fix_sorcar_unfinished_usage_lock.py``).  These tests
pin the collector's remaining behaviour: a pending child's zero slot is
filled from the live read, finished and never-started children are left
alone, and a live read that lags an already-published slot never lowers
it.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import _collect_unfinished_usage

_INITIAL = (1.0, 100, 1)
_FINAL = (2.0, 200, 2)


def _set_usage(agent: ChatSorcarAgent, usage: tuple[float, int, int]) -> None:
    """Publish *usage* on *agent*'s counters like ``_accumulate_usage`` does."""
    agent.budget_used, agent.total_tokens_used, agent.total_steps = usage


def test_pending_child_slot_takes_live_figure() -> None:
    """A child still running has its zero slot filled from the live read."""
    agent = ChatSorcarAgent("audit0902-pending")
    _set_usage(agent, _INITIAL)
    sub_usage: list[tuple[float, int, int]] = [(0.0, 0, 0)]
    release = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future: Future[Any] = pool.submit(release.wait, 10)
        _collect_unfinished_usage([future], [agent], sub_usage, threading.Lock())
        assert not future.done()
        release.set()
    assert sub_usage[0] == _INITIAL


def test_finished_and_unstarted_children_are_left_alone() -> None:
    """Done futures and never-started children (agent None) are skipped."""
    agent = ChatSorcarAgent("audit0902-done")
    _set_usage(agent, _FINAL)
    with ThreadPoolExecutor(max_workers=1) as pool:
        done = pool.submit(lambda: "x")
        done.result(timeout=10)
    never_started: Future[str] = Future()
    sub_usage: list[tuple[float, int, int]] = [_INITIAL, (0.0, 0, 0)]
    _collect_unfinished_usage(
        [done, never_started], [agent, None], sub_usage, threading.Lock(),
    )
    assert sub_usage == [_INITIAL, (0.0, 0, 0)]


def test_torn_live_read_never_lowers_a_published_slot() -> None:
    """A live read that lags the slot (session handoff) cannot regress it."""
    agent = ChatSorcarAgent("audit0902-torn")
    _set_usage(agent, _INITIAL)
    # The child already published a higher figure (its final one) but
    # its future has not completed yet: the live read must not win.
    sub_usage: list[tuple[float, int, int]] = [_FINAL]
    release = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future: Future[Any] = pool.submit(release.wait, 10)
        _collect_unfinished_usage([future], [agent], sub_usage, threading.Lock())
        release.set()
    assert sub_usage[0] == _FINAL
