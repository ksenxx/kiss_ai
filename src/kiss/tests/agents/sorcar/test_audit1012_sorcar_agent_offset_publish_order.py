# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Printer usage offsets must never decrease (audit 2026-10-12, sorcar-agent).

``RelentlessAgent.perform_task`` re-bases the printer's per-task usage
offsets at every session start, and ``_attribute_sub_usage`` re-bases
them from a child's own thread when a ``run_agent`` job or a ``talk``
synthesis folds its spend into the parent's ledger.  Both writers
snapshot the append-only ledger and publish the snapshot as the three
offsets.  Before the fix the agent thread took its snapshot OUTSIDE
``_OFFSET_PUBLISH_LOCK`` (``usage_snapshot()`` at the top of the session
prologue, ``_set_printer_offsets(budget, tokens, steps)`` a few
statements later), so this interleaving regressed the tab's live cost
for the whole session:

1. agent thread: ``budget_banked = usage_snapshot()`` -> B
2. dispatch thread: ``_attribute_usage(child)`` appends ``d``, then
   publishes offsets ``B + d`` under the lock
3. agent thread: ``_set_printer_offsets(B, ...)`` publishes ``B`` —
   the child's spend vanishes from every event of the session

A scratch run of the old sequence against the real ``JsonPrinter``
(four folding threads, one re-basing thread, two seconds) showed 25
such regressions.  ``_set_printer_offsets`` now snapshots inside the
same lock, so publishes land in snapshot order and the offsets are
monotone.  The tests drive the real agent, ledger and ``JsonPrinter``
on real threads; no code under test is replaced.
"""

from __future__ import annotations

import threading

from kiss.agents.sorcar.relentless_agent import _OFFSET_PUBLISH_LOCK
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _attribute_sub_usage
from kiss.server.json_printer import JsonPrinter


def _check_monotone(agent: SorcarAgent, seen: list[float], regressions: list[float]) -> None:
    """Record a regression when the published budget offset dropped.

    The read happens under the publish lock, so it is ordered with
    every publish and with every other check: a drop below the largest
    value any earlier check saw means a publish carried an older
    snapshot than its predecessor.
    """
    with _OFFSET_PUBLISH_LOCK:
        published = float(agent.printer.budget_offset)  # type: ignore[union-attr]
        if published < seen[0] - 1e-12:
            regressions.append(published)
        seen[0] = max(seen[0], published)


def _fold_until(
    agent: SorcarAgent,
    stop: threading.Event,
    seen: list[float],
    regressions: list[float],
) -> None:
    """Fold child spend into *agent* until *stop*, checking after each fold."""
    while not stop.is_set():
        _attribute_sub_usage(agent, 0.001, 1, 1)
        _check_monotone(agent, seen, regressions)


def test_session_rebase_never_regresses_a_concurrent_fold() -> None:
    """The agent thread's rebase cannot overwrite a newer fold's offsets."""
    agent = SorcarAgent("offset-order")
    agent.printer = JsonPrinter()
    agent.reset_usage()
    stop = threading.Event()
    seen = [0.0]
    regressions: list[float] = []
    folders = [
        threading.Thread(target=_fold_until, args=(agent, stop, seen, regressions))
        for _ in range(4)
    ]
    for thread in folders:
        thread.start()
    # The session-start rebase, as perform_task does it, as fast as it can.
    deadline = threading.Event()
    timer = threading.Timer(2.0, deadline.set)
    timer.start()
    try:
        while not deadline.is_set():
            agent._set_printer_offsets()
            _check_monotone(agent, seen, regressions)
    finally:
        stop.set()
        timer.cancel()
        for thread in folders:
            thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in folders)
    assert regressions == []
    # The last publish, whichever writer made it, equals the ledger.
    budget, tokens, steps = agent.usage_snapshot()
    assert agent.printer.budget_offset == budget
    assert agent.printer.tokens_offset == tokens
    assert agent.printer.steps_offset == steps


def test_set_printer_offsets_publishes_the_banked_totals() -> None:
    """A rebase with no printer is a no-op; with one it publishes the ledger."""
    agent = SorcarAgent("offset-plain")
    agent.reset_usage()
    agent._attribute_usage(0.5, 7, 3)
    agent._set_printer_offsets()  # no printer: nothing to publish, no error
    agent.printer = JsonPrinter()
    agent._set_printer_offsets()
    assert agent.printer.budget_offset == 0.5
    assert agent.printer.tokens_offset == 7
    assert agent.printer.steps_offset == 3
