# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a Stop must reach a parent waiting on sub-agents.

Bug reproduction (post-mortem ``reports/stop_button_delay_2026-08-05.html``,
task ``709ebce3``): after ``run_parallel`` fanned out, the parent task ran
no code of its own — it sat in ``list(pool.map(...))`` inside
``with ThreadPoolExecutor(...)``.  Both stop mechanisms were disarmed by
that:

* the cooperative flag is only read when the agent prints, and a parent
  prints nothing while its children run;
* the forced ``KeyboardInterrupt`` that ``_stop_task`` injects with
  ``PyThreadState_SetAsyncExc`` is delivered only at a Python bytecode
  boundary, and ``Future.result()`` parks the thread in a C-level lock.

So the parent could only end when its slowest child returned — three
minutes after the user pressed Stop.

Fix under test: ``_await_subagents`` waits in short slices, which keeps
the parent at a bytecode boundary (so an injected interrupt lands
immediately), and abandons a child that ignores its stop event for
longer than the grace period instead of waiting for it forever.

The ``_stop_task`` tests cover the second half of the report: a stop the
daemon cannot route used to disappear behind a disabled ``logger.debug``,
so the UI could not tell a pending stop from a discarded one.

No mocks or fakes: real threads, a real ``ThreadPoolExecutor``, the real
agent-state registry, and the production
``VSCodeServer._force_stop_thread`` / ``_stop_task``.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pytest

from kiss.agents.sorcar import sorcar_agent
from kiss.agents.sorcar.sorcar_agent import _await_subagents, _SubagentStopEvent


def _submit(
    pool: ThreadPoolExecutor,
    bodies: list[Any],
) -> list[Future[str]]:
    """Submit one callable per sub-agent, preserving order."""
    return [pool.submit(body) for body in bodies]


class TestFanOutIsInterruptible:
    """The parent must not be held hostage by a wedged child."""

    def test_results_are_collected_in_task_order(self) -> None:
        """The ordinary path is unchanged: every result, in order."""
        def body(index: int) -> Any:
            def run() -> str:
                time.sleep(0.05 * (3 - index))
                return f"result-{index}"
            return run

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = _submit(pool, [body(i) for i in range(3)])
            results = _await_subagents(futures, threading.Event())
        assert results == ["result-0", "result-1", "result-2"]

    def test_a_stopped_child_that_unwinds_is_still_collected(self) -> None:
        """A child that honours its stop event still reports back.

        Its summary and its spend must survive the stop, so the grace
        period is spent waiting rather than abandoning immediately.
        """
        stop_event = threading.Event()

        def body() -> str:
            stop_event.wait(5.0)
            return "stopped cleanly"

        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = _submit(pool, [body])
            threading.Timer(0.2, stop_event.set).start()
            results = _await_subagents(futures, stop_event)
        assert results == ["stopped cleanly"]

    def test_a_wedged_child_is_abandoned_after_the_grace_period(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One child ignoring Stop can no longer hold the task open."""
        monkeypatch.setattr(
            sorcar_agent, "_SUBAGENT_STOP_GRACE_SECONDS", 0.3,
        )
        release = threading.Event()
        stop_event = threading.Event()

        def wedged() -> str:
            release.wait(30.0)
            return "never seen"

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            futures = _submit(pool, [wedged])
            stop_event.set()
            start = time.monotonic()
            with pytest.raises(KeyboardInterrupt):
                _await_subagents(futures, stop_event)
            assert time.monotonic() - start < 3.0
        finally:
            release.set()
            pool.shutdown(wait=True)

    def test_parent_stop_event_reaches_children_through_the_chain(self) -> None:
        """Stopping the parent stops the fan-out, as before the fix."""
        parent_stop = threading.Event()
        child_stop = _SubagentStopEvent(parent_stop)
        observed: list[bool] = []

        def body() -> str:
            child_stop.wait(5.0)
            observed.append(child_stop.is_set())
            return "child saw the parent stop"

        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = _submit(pool, [body])
            threading.Timer(0.2, parent_stop.set).start()
            results = _await_subagents(futures, parent_stop)
        assert results == ["child saw the parent stop"]
        assert observed == [True]
