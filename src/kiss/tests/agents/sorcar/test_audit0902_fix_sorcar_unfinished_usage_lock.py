# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review fix #3: the abandoned-child usage refresh is serialised with the
child's own final write.

``run_tasks_parallel`` keeps one ``(cost, tokens, steps)`` slot per child.
A child writes its FINAL figure into its slot in its worker's ``finally``;
when the parent abandons children that ignore Stop,
``_collect_unfinished_usage`` refreshes the still-pending slots from a
LIVE read.  Before the fix that read-modify-write was unsynchronised
with the worker's write: a child publishing between the parent's read
of the slot and its write had its final figure replaced by the older
live one, and — its future now being done — it was not registered as
abandoned either, so the parent's totals undercounted it for good.

The interleaving test below synchronises through the test's OWN list
(its ``__getitem__`` signals the worker and waits, bounded, for the
worker's publish); nothing in the implementation is relied on for
timing.  With the shared lock the worker cannot publish while the
collector is inside its read-modify-write, so that wait necessarily
times out and the worker's final figure lands last.

The end-to-end test drives the real abandon path: real
``ChatSorcarAgent`` children, a real ``JsonPrinter`` whose
task-allocation hook parks each child until the test releases it
(the child "ignores Stop"), and the parent's real totals banking
(``SorcarAgent._run_tasks_parallel`` + ``reclaim_abandoned_subagents``).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from kiss.agents.sorcar import sorcar_agent
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _collect_unfinished_usage
from kiss.tests.server.parallel_agent_harness import (
    CapturePrinter,
    IsolatedKissHome,
)

_INITIAL = (1.0, 100, 1)
_FINAL = (2.0, 200, 2)


def _set_usage(agent: Any, usage: tuple[float, int, int]) -> None:
    """Publish *usage* on *agent*'s counters like ``_accumulate_usage`` does."""
    agent.budget_used, agent.total_tokens_used, agent.total_steps = usage


class _SlotsSignallingReads(list[tuple[float, int, int]]):
    """The parent's slot list, instrumented by the test.

    Reading a slot (what the collector does right before its write)
    wakes the worker and then waits — bounded — for the worker to
    publish.  Without the fix the publish lands inside that window and
    the collector's subsequent write destroys it; with the fix the
    worker blocks on the shared lock, the wait times out, and the
    collector's write is followed by the worker's.
    """

    def __init__(self, read: threading.Event, published: threading.Event) -> None:
        super().__init__([(0.0, 0, 0)])
        self.read = read
        self.published = published

    def __getitem__(self, idx: Any) -> Any:  # type: ignore[override]
        value = super().__getitem__(idx)
        self.read.set()
        self.published.wait(1.0)
        return value


def test_child_publishing_inside_the_refresh_window_keeps_its_final_figure() -> None:
    """The worker's final write can never be overwritten by the live refresh."""
    agent = ChatSorcarAgent("audit0902-fix-lock-child")
    _set_usage(agent, _INITIAL)
    read = threading.Event()
    published = threading.Event()
    slots = _SlotsSignallingReads(read, published)
    lock = threading.Lock()

    def worker() -> str:
        assert read.wait(10)
        _set_usage(agent, _FINAL)
        # Exactly what run_tasks_parallel's worker does in its finally.
        with lock:
            slots[0] = _FINAL
        published.set()
        return "done"

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker)
        _collect_unfinished_usage([future], [agent], slots, lock)
        future.result(timeout=10)

    assert published.is_set()
    assert list.__getitem__(slots, 0) == _FINAL, (
        f"the live refresh overwrote the child's final figure: {list(slots)}"
    )


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-audit0902-fix-lock-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


class _ParkingPrinter(CapturePrinter):
    """A real server printer whose children ignore Stop until released.

    ``ChatSorcarAgent.run`` calls ``agent_task_allocated`` as soon as
    the child's history row exists.  Here each child records an initial
    spend, parks until *release* is set (so the parent's grace period
    elapses and it abandons the child), then records its final spend
    and unwinds — which makes the worker publish that final figure.
    """

    def __init__(self, expected_children: int, stop_event: threading.Event) -> None:
        super().__init__()
        self.release = threading.Event()
        self._stop_event = stop_event
        self._expected = expected_children
        self._parked = 0
        self._parked_lock = threading.Lock()
        self.children: list[Any] = []

    def agent_task_allocated(self, agent: Any, task_id: Any, chat_id: str = "") -> None:
        super().agent_task_allocated(agent, task_id, chat_id)
        _set_usage(agent, _INITIAL)
        with self._parked_lock:
            self.children.append(agent)
            self._parked += 1
            if self._parked == self._expected:
                # Every child is now spending: ask the parent to stop.
                self._stop_event.set()
        assert self.release.wait(60), "test released no child"
        _set_usage(agent, _FINAL)
        raise RuntimeError("child unwinding after being abandoned")


def test_abandon_path_banks_live_then_final_usage_end_to_end(
    env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parent totals hold the live figures at abandonment and the final
    figures once the abandoned children finish."""
    monkeypatch.setattr(sorcar_agent, "_SUBAGENT_STOP_GRACE_SECONDS", 0.3)
    stop_event = threading.Event()
    printer = _ParkingPrinter(expected_children=2, stop_event=stop_event)
    printer._thread_local.stop_event = stop_event

    parent = SorcarAgent("audit0902-fix-lock-parent")
    parent.set_printer(printer)
    parent.model_name = "claude-fable-5-1"
    parent.work_dir = str(env.repo)
    _set_usage(parent, (0.0, 0, 0))

    try:
        with pytest.raises(KeyboardInterrupt):
            parent._run_tasks_parallel(["first child", "second child"])
        # Abandoned: the parent banked what the children had spent so far.
        assert len(printer.children) == 2
        assert (parent.budget_used, parent.total_tokens_used, parent.total_steps) == (
            2 * _INITIAL[0], 2 * _INITIAL[1], 2 * _INITIAL[2],
        )
        with parent._abandoned_lock:
            assert len(parent._abandoned_subagents) == 2
        assert not parent.reclaim_abandoned_subagents()
    finally:
        printer.release.set()

    # The children unwind and publish their final figures; the parent
    # banks the remainder exactly once.
    assert parent.reclaim_abandoned_subagents(timeout=30)
    assert (parent.budget_used, parent.total_tokens_used, parent.total_steps) == (
        2 * _FINAL[0], 2 * _FINAL[1], 2 * _FINAL[2],
    )
    with parent._abandoned_lock:
        assert parent._abandoned_subagents == []
    for child in printer.children:
        assert child.last_task_id
