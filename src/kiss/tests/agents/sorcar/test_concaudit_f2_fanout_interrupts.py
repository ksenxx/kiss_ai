# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Fan-out engine versus an injected stop (F2 findings 2, 3 and 5).

The server's Stop button sets the task's stop event and then injects a
``KeyboardInterrupt`` into the task thread with
``PyThreadState_SetAsyncExc``; CPython delivers it at an arbitrary
bytecode boundary.  Three boundaries in ``sorcar_agent.py`` were
unprotected:

* **Finding 2** — ``run_tasks_parallel`` submitted every child with a
  list comprehension OUTSIDE the ``except BaseException`` region.  An
  interrupt during submission left ``futures`` empty and ``abandoned``
  False, so the ``finally`` joined the already-running children with
  ``shutdown(wait=True)`` — for as long as a wedged child took — and
  registered none of them as abandoned.
* **Finding 5** — ``_run_tasks_parallel`` started the live-usage
  monitor thread BEFORE the ``try`` whose ``finally`` stops it.  An
  interrupt landing inside ``Thread.start()`` after the OS thread was
  created (``self._started.wait()`` is Python code) leaked a daemon
  emitting ``usage_info`` every second for the rest of the process.
* **Finding 3** — ``reclaim_abandoned_subagents`` waited on a
  pre-wait snapshot and computed its return value from that snapshot
  alone, so a child abandoned during the wait was kept in the list but
  ignored: the method returned True while that child was still
  writing into the shared working directory.

The interrupts are delivered for real, in the fan-out thread, by a
``sys.settrace`` tracer raising ``KeyboardInterrupt`` at an exact line
boundary — the same arbitrary-boundary delivery the server performs,
made deterministic.  Children are real ``ChatSorcarAgent`` sub-agents
spawned by the real engine; they park in the printer's
``agent_task_allocated`` hook (before any model call) and ignore Stop
until the test releases them, exactly like a wedged sub-agent.
"""

from __future__ import annotations

import inspect
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from kiss.agents.sorcar import sorcar_agent
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _AbandonedSubagent
from kiss.tests.server.parallel_agent_harness import (
    CapturePrinter,
    IsolatedKissHome,
)

_MONITOR_THREAD_NAME = "live-usage-monitor"


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-concaudit-f2-fanout-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


class _ParkingPrinter(CapturePrinter):
    """A real server printer whose children ignore Stop until released."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.children: list[Any] = []
        self._children_lock = threading.Lock()

    def agent_task_allocated(self, agent: Any, task_id: Any, chat_id: str = "") -> None:
        super().agent_task_allocated(agent, task_id, chat_id)
        with self._children_lock:
            self.children.append(agent)
        assert self.release.wait(120), "test released no child"
        raise RuntimeError("child unwinding after being released")

    def child_count(self) -> int:
        """Return how many children have parked so far."""
        with self._children_lock:
            return len(self.children)


def _monitor_threads() -> list[threading.Thread]:
    """Return every live thread named like the live-usage monitor."""
    return [t for t in threading.enumerate() if t.name == _MONITOR_THREAD_NAME]


def _make_parent(env: IsolatedKissHome, printer: CapturePrinter) -> SorcarAgent:
    parent = SorcarAgent("concaudit-f2-parent")
    parent.set_printer(printer)
    parent.model_name = "claude-fable-5-1"
    parent.work_dir = str(env.repo)
    parent.budget_used, parent.total_tokens_used, parent.total_steps = 0.0, 0, 0
    return parent


def _run_fanout_with_tracer(
    parent: SorcarAgent,
    printer: CapturePrinter,
    stop_event: threading.Event,
    tasks: list[str],
    tracer: Any,
    outcome: dict[str, Any],
) -> None:
    """Thread body: run the fan-out under *tracer*, record how it ended."""
    printer._thread_local.stop_event = stop_event
    sys.settrace(tracer)
    try:
        outcome["result"] = parent._run_tasks_parallel(tasks)
    except BaseException as exc:  # noqa: BLE001 — the injected stop is expected
        outcome["exc"] = exc
    finally:
        sys.settrace(None)
        outcome["elapsed"] = time.monotonic() - outcome["started"]


def test_stop_during_submission_abandons_children_and_returns_promptly(
    env: IsolatedKissHome,
) -> None:
    """Finding 2: a stop injected mid-submission takes the abandon path.

    The tracer holds the parent at the first line boundary after the
    first child was submitted (as if descheduled) until that child has
    parked, then sets the stop event and raises ``KeyboardInterrupt``
    there — before the second task is submitted.  The call must return
    promptly (no ``shutdown(wait=True)`` on the parked child), register
    the parked child as abandoned, and never submit the second task.
    """
    printer = _ParkingPrinter()
    parent = _make_parent(env, printer)
    stop_event = threading.Event()
    state = {"injected": False}
    engine_file = sorcar_agent.__file__

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if (
            event != "line"
            or state["injected"]
            or frame.f_code.co_name != "run_tasks_parallel"
            or frame.f_code.co_filename != engine_file
        ):
            return tracer
        futures = frame.f_locals.get("futures")
        if not futures or len(futures) != 1:
            return tracer
        deadline = time.monotonic() + 60
        while printer.child_count() == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        state["injected"] = True
        stop_event.set()
        raise KeyboardInterrupt("injected stop during submission")

    outcome: dict[str, Any] = {"started": time.monotonic()}
    runner = threading.Thread(
        target=_run_fanout_with_tracer,
        args=(parent, printer, stop_event, ["park A", "park B"], tracer, outcome),
    )
    runner.start()
    try:
        runner.join(timeout=60)
        hung = runner.is_alive()
    finally:
        # Never leave a parked child behind, even when the assertion
        # below is about to fail because the parent hung on it.
        printer.release.set()
        runner.join(timeout=60)
    assert not hung, "the fan-out waited for the parked child instead of abandoning it"
    assert state["injected"], "the tracer never found the submission window"
    assert isinstance(outcome.get("exc"), KeyboardInterrupt), outcome
    assert outcome["elapsed"] < 30, outcome
    assert printer.child_count() == 1, "the second task was submitted after the stop"
    with parent._abandoned_lock:
        registered = list(parent._abandoned_subagents)
    assert len(registered) == 1, "the running child was not registered as abandoned"
    assert registered[0].agent is printer.children[0]
    assert parent.reclaim_abandoned_subagents(timeout=60)
    assert _monitor_threads() == []


def test_stop_inside_monitor_thread_start_does_not_leak_the_monitor(
    env: IsolatedKissHome,
) -> None:
    """Finding 5: an interrupt inside ``Thread.start()`` still stops the monitor.

    ``Thread.start`` creates the OS thread and then waits for it in
    Python (``self._started.wait()``); an injected stop can land on
    that line.  The monitor thread is then already running, so unless
    the start happened under the ``try`` whose ``finally`` stops it,
    nothing ever sets ``_done`` and the daemon emits forever.
    """
    printer = CapturePrinter()
    parent = _make_parent(env, printer)
    stop_event = threading.Event()
    start_code = threading.Thread.start.__code__
    lines, first = inspect.getsourcelines(threading.Thread.start)
    wait_lines = [first + i for i, ln in enumerate(lines) if "self._started.wait()" in ln]
    assert len(wait_lines) == 1, "threading.Thread.start changed shape"
    state = {"injected": False}

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if (
            event == "line"
            and not state["injected"]
            and frame.f_code is start_code
            and frame.f_lineno == wait_lines[0]
            and frame.f_locals["self"].name == _MONITOR_THREAD_NAME
        ):
            state["injected"] = True
            stop_event.set()
            raise KeyboardInterrupt("injected stop inside Thread.start")
        return tracer

    outcome: dict[str, Any] = {"started": time.monotonic()}
    runner = threading.Thread(
        target=_run_fanout_with_tracer,
        args=(parent, printer, stop_event, ["never submitted"], tracer, outcome),
    )
    runner.start()
    runner.join(timeout=60)
    assert not runner.is_alive()
    assert state["injected"], "the tracer never saw the monitor thread start"
    assert isinstance(outcome.get("exc"), KeyboardInterrupt), outcome
    # The monitor polls once a second; a stopped one is gone well before
    # this deadline, a leaked one is still here (and emitting) after it.
    deadline = time.monotonic() + 5
    while _monitor_threads() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _monitor_threads() == [], "the live-usage monitor thread leaked"


def test_monitor_stop_tolerates_a_start_that_never_happened() -> None:
    """``stop()`` is safe when ``start()`` never ran or ran without a printer."""
    never_started = sorcar_agent._LiveUsageMonitor(None, CapturePrinter())
    never_started.stop()
    assert never_started._done.is_set()
    no_printer = sorcar_agent._LiveUsageMonitor(None, None)
    no_printer.start()
    no_printer.stop()
    assert no_printer._thread is None


class _LiveChild:
    """A real pool thread standing in for an abandoned sub-agent."""

    def __init__(self, parent: SorcarAgent, name: str) -> None:
        self.release = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = self.pool.submit(self._park)
        self.agent = ChatSorcarAgent(name)
        self.item = _AbandonedSubagent(self.future, self.agent, (0.0, 0, 0))
        with parent._abandoned_lock:
            parent._abandoned_subagents.append(self.item)

    def _park(self) -> str:
        assert self.release.wait(120)
        return ""

    def finish(self) -> None:
        """Let the thread exit and reclaim the pool."""
        self.release.set()
        self.pool.shutdown(wait=True)


def test_reclaim_reports_a_child_abandoned_during_its_wait(
    env: IsolatedKissHome,
) -> None:
    """Finding 3: the return value covers children registered mid-wait."""
    parent = _make_parent(env, CapturePrinter())
    child_a = _LiveChild(parent, "concaudit-f2-child-a")
    child_b: list[_LiveChild] = []
    result: list[bool] = []

    def reclaim() -> None:
        result.append(parent.reclaim_abandoned_subagents(timeout=60))

    reclaimer = threading.Thread(target=reclaim)
    reclaimer.start()
    try:
        # Let the reclaimer snapshot [A] and enter its wait, then
        # abandon B (a later fan-out ending) and finish A so the wait
        # returns with B still running.
        time.sleep(0.3)
        assert reclaimer.is_alive()
        child_b.append(_LiveChild(parent, "concaudit-f2-child-b"))
        child_a.finish()
        reclaimer.join(timeout=60)
        assert not reclaimer.is_alive()
        assert result == [False], "reclaim said no child was live while B was"
        with parent._abandoned_lock:
            assert [i.agent for i in parent._abandoned_subagents] == [child_b[0].agent]
    finally:
        child_a.finish()
        for child in child_b:
            child.finish()
    assert parent.reclaim_abandoned_subagents(timeout=60)
    with parent._abandoned_lock:
        assert parent._abandoned_subagents == []
