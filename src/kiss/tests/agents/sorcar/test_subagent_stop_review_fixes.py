# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the review-round fixes to the "interact with a
RUNNING sub-agent" feature:

1. ``VSCodeServer._open_persisted_subagent_tabs`` must SUBSCRIBE the
   reopened deterministic frontend tab (``{parent_tab_id}__sub_{id}``)
   to a STILL-RUNNING sub-agent's live stream — otherwise the input
   textbox shown on that tab is a dead surface (Stop / prompt
   injection cannot resolve the sub-agent, live events never arrive).

2. ``VSCodeServer._stop_task`` must FORCE-STOP a sub-agent wedged in an
   uninterruptible call (never polling its cooperative stop event) by
   injecting ``KeyboardInterrupt`` into the pool worker thread
   published on the sub-agent's registry state.

3. The force-stop watchdog's ownership guard must NEVER interrupt a
   SIBLING task that a reused ``ThreadPoolExecutor`` worker thread
   picked up after the stopped sub-agent finished cooperatively.

4. ``_SubagentStopEvent.wait`` semantics: own set, parent set mid-wait,
   and timeout expiry.

All tests drive the real production code (``_run_tasks_parallel``,
``VSCodeServer._stop_task`` / ``_open_persisted_subagent_tabs``, the
real registry and printer) — no mocks of the code under test.
"""

from __future__ import annotations

import threading
import time

from kiss.agents.sorcar.sorcar_agent import _SubagentStopEvent


class TestSubagentStopEventWaitSemantics:
    """``_SubagentStopEvent.wait`` must observe its own flag, the
    parent chain, and timeouts."""

    def test_wait_returns_true_when_own_flag_set(self) -> None:
        ev = _SubagentStopEvent(threading.Event())
        ev.set()
        assert ev.wait(0.0) is True
        assert ev.wait(1.0) is True

    def test_wait_times_out_false_when_nothing_set(self) -> None:
        ev = _SubagentStopEvent(threading.Event())
        start = time.monotonic()
        assert ev.wait(0.15) is False
        assert time.monotonic() - start < 5.0

    def test_wait_wakes_on_parent_set_mid_wait(self) -> None:
        parent = threading.Event()
        ev = _SubagentStopEvent(parent)
        threading.Timer(0.1, parent.set).start()
        assert ev.wait(5.0) is True

    def test_wait_without_parent_and_none_chain(self) -> None:
        ev = _SubagentStopEvent(None)
        assert ev.wait(0.05) is False
        ev.set()
        assert ev.wait(None) is True
        assert ev.is_set() is True
