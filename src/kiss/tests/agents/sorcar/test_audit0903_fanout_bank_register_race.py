# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 0903: an abandoned child's spend is banked exactly once.

``run_tasks_parallel``'s ``finally`` runs three steps on the abandon
path: refresh the not-yet-published slots (``_collect_unfinished_usage``),
register every still-running child as abandoned with its slot value as
the ``counted`` baseline (``_register_abandoned``), and sum the slots
into ``totals_out`` — which the parent immediately banks into its own
cumulative counters.

A worker's ``finally`` publishes its FINAL slot value the moment the
child unwinds.  Before the fix, only the collect step took
``sub_usage_lock``; a worker publishing between the registration and
the summation made the parent bank the FINAL figure while the
registered ``counted`` baseline stayed at the older value, so the next
``reclaim_abandoned_subagents`` banked the (final − older) delta a
SECOND time — the abandoned child's spend was double-counted for good.
The fix holds ``sub_usage_lock`` across registration and summation, so
the figure banked for a registered child always equals its ``counted``
baseline.

End-to-end, no mocks of the code under test: a real ``SorcarAgent``
parent, a real ``ChatSorcarAgent`` child spawned by the real fan-out
engine, a real server printer whose child ignores Stop until the test
releases it, and the real reclaim path.  The ``KISS_RACE_DELAY`` hook
(the codebase's sanctioned window-widener, a production no-op) sits
between registration and summation; the test releases the child the
instant it is registered so its final publish targets exactly that
window.  With the fix the publish blocks on ``sub_usage_lock`` until
the summation is done, so the conservation assertion holds regardless
of timing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar import sorcar_agent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.server.parallel_agent_harness import (
    CapturePrinter,
    IsolatedKissHome,
)

_INITIAL = (1.0, 100, 1)
_FINAL = (2.0, 200, 2)


def _set_usage(agent: Any, usage: tuple[float, int, int]) -> None:
    """Publish *usage* on *agent*'s counters like ``_accumulate_usage`` does."""
    agent.budget_used, agent.total_tokens_used, agent.total_steps = usage


class _ParkingPrinter(CapturePrinter):
    """A real server printer whose child ignores Stop until released.

    The child records an initial spend as soon as its history row
    exists, asks the parent to stop, parks until the test releases it
    (so the parent's grace period elapses and abandons it), then
    records its final spend and unwinds — making its worker publish
    the final figure into its ``sub_usage`` slot.
    """

    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__()
        self.release = threading.Event()
        self._stop_event = stop_event
        self.children: list[Any] = []

    def agent_task_allocated(self, agent: Any, task_id: Any, chat_id: str = "") -> None:
        super().agent_task_allocated(agent, task_id, chat_id)
        _set_usage(agent, _INITIAL)
        self.children.append(agent)
        self._stop_event.set()
        assert self.release.wait(60), "test released no child"
        _set_usage(agent, _FINAL)
        raise RuntimeError("child unwinding after being abandoned")


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-audit0903-bank-register-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


def test_abandoned_child_final_publish_is_never_double_banked(
    env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Banked-at-abandonment plus reclaimed-later equals the child's
    spend exactly once, even when the child's final publish lands while
    the parent is banking."""
    monkeypatch.setattr(sorcar_agent, "_SUBAGENT_STOP_GRACE_SECONDS", 0.3)
    monkeypatch.setenv("KISS_RACE_DELAY", "0.1")
    stop_event = threading.Event()
    printer = _ParkingPrinter(stop_event)
    printer._thread_local.stop_event = stop_event

    parent = SorcarAgent("audit0903-bank-register-parent")
    parent.set_printer(printer)
    parent.model_name = "claude-fable-5-1"
    parent.work_dir = str(env.repo)
    _set_usage(parent, (0.0, 0, 0))

    def _release_on_registration() -> None:
        # Release the parked child the moment the parent registers it
        # as abandoned, so the child's final publish races the parent's
        # totals summation as closely as the scheduler allows.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with parent._abandoned_lock:
                if parent._abandoned_subagents:
                    break
            time.sleep(0.0005)
        printer.release.set()

    watcher = threading.Thread(target=_release_on_registration, daemon=True)
    watcher.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            parent._run_tasks_parallel(["park until released"])
    finally:
        printer.release.set()
    watcher.join(timeout=30)

    assert len(printer.children) == 1
    # The child was registered as abandoned before its worker finished.
    banked_at_abandonment = (
        parent.budget_used, parent.total_tokens_used, parent.total_steps,
    )
    # Let the abandoned worker finish and bank whatever it spent after
    # the figure counted at abandonment.
    assert parent.reclaim_abandoned_subagents(timeout=30)
    with parent._abandoned_lock:
        assert parent._abandoned_subagents == []

    total = (parent.budget_used, parent.total_tokens_used, parent.total_steps)
    assert total == _FINAL, (
        f"abandoned child's spend was not banked exactly once: banked "
        f"{banked_at_abandonment} at abandonment, {total} after reclaim, "
        f"child spent {_FINAL}"
    )
