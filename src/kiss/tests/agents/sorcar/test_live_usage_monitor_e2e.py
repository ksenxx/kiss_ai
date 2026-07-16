# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the parent task's cost/tokens header must reflect the
agent AND all of its parallel sub-agents at every turn.

Reproduces the issue where, while ``run_parallel`` blocks the parent's
turn, nothing emits ``usage_info`` on the PARENT task — so the header
(chat webview top bar, sorcar CLI interactive) shows a stale figure that
excludes all live sub-agent spend until every sub-agent finishes.

The fix is ``_LiveUsageMonitor`` (``sorcar_agent.py``): while parallel
sub-agents run it polls their live spend and broadcasts parent-task
``usage_info`` events whose printer-applied offsets yield the aggregate
(parent cumulative + parent live session + all sub-agents).

No mocks, patches, fakes, or test doubles: real agents, a real
``JsonPrinter`` subclass that records its own broadcasts, and (for the
slow test) real LLM calls.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _attribute_sub_usage,
    _live_agent_usage,
    _LiveUsageMonitor,
)
from kiss.core.kiss_agent import KISSAgent
from kiss.server.json_printer import JsonPrinter

FAST_MODEL = "claude-haiku-4-5"

skip_no_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


class _RecordingPrinter(JsonPrinter):
    """Captures every broadcast event AFTER ``_inject_task_id`` runs."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self._capture_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        event = self._inject_task_id(event)
        with self._capture_lock:
            self.events.append(dict(event))
        super().broadcast(event)

    def usage_events(self, task_id: str) -> list[dict[str, Any]]:
        """Return recorded ``usage_info`` events for *task_id*."""
        with self._capture_lock:
            return [
                e
                for e in self.events
                if e.get("type") == "usage_info" and e.get("taskId") == task_id
            ]


def _wait_for(condition, timeout: float = 5.0) -> bool:
    """Poll *condition* until true or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def _parent_with_task(printer: _RecordingPrinter, task_id: str) -> SorcarAgent:
    """Build a parent agent bound to *printer* with *task_id* thread-local."""
    printer._thread_local.task_id = task_id
    parent = SorcarAgent("live-usage-parent")
    parent.printer = printer
    return parent


class TestLiveUsageMonitor:
    """Deterministic e2e tests of the live-usage monitor itself."""

    def test_emits_parent_usage_info_as_subagents_spend(self) -> None:
        """While sub-agents spend, the PARENT task must receive usage_info
        events whose cost includes the sub-agents' live spend plus the
        parent's cumulative offset — the exact aggregate the header shows."""
        printer = _RecordingPrinter()
        parent = _parent_with_task(printer, "parent-task")
        # Parent spent $0.05 / 500 tokens / 3 steps in prior sessions —
        # snapshotted into the printer offsets at session start
        # (relentless_agent.perform_task does exactly this).
        parent.budget_used = 0.05
        parent.total_tokens_used = 500
        parent.total_steps = 3
        printer.budget_offset = 0.05
        printer.tokens_offset = 500
        printer.steps_offset = 3

        monitor = _LiveUsageMonitor(parent, printer, interval=0.02)
        sub: Any = KISSAgent("sub-0")
        monitor.track(sub)
        monitor.start()
        try:
            # The sub-agent's first LLM call lands: $0.25 / 1000 tokens.
            sub.budget_used = 0.25
            sub.total_tokens_used = 1000
            sub.total_steps = 4
            assert _wait_for(
                lambda: any(
                    e.get("cost") == "$0.3000"
                    for e in printer.usage_events("parent-task")
                )
            ), "no parent usage_info with aggregated cost was broadcast"
            # A second call: totals must keep tracking upward.
            sub.budget_used = 0.40
            sub.total_tokens_used = 2000
            sub.total_steps = 5
            assert _wait_for(
                lambda: any(
                    e.get("cost") == "$0.4500"
                    for e in printer.usage_events("parent-task")
                )
            ), "parent usage_info did not track the sub-agent's later spend"
        finally:
            monitor.stop()
        events = printer.usage_events("parent-task")
        first = next(e for e in events if e.get("cost") == "$0.3000")
        assert first["total_tokens"] == 1500
        assert first["total_steps"] == 7
        latest = next(e for e in events if e.get("cost") == "$0.4500")
        assert latest["total_tokens"] == 2500
        assert latest["total_steps"] == 8
        # Change-only emission: no two consecutive identical snapshots.
        snapshots = [
            (e.get("cost"), e.get("total_tokens"), e.get("total_steps"))
            for e in events
        ]
        for a, b in zip(snapshots, snapshots[1:]):
            assert a != b, f"duplicate consecutive usage_info emitted: {a}"

    def test_includes_parent_live_executor_session(self) -> None:
        """The parent's own in-flight session spend (its live executor)
        must be part of the emitted raw values."""
        printer = _RecordingPrinter()
        parent = _parent_with_task(printer, "parent-exec-task")
        executor = KISSAgent("parent-session-0")
        executor.budget_used = 0.10
        executor.total_tokens_used = 300
        executor.step_count = 2
        parent._current_executor = executor

        monitor = _LiveUsageMonitor(parent, printer, interval=0.02)
        sub: Any = KISSAgent("sub-0")
        sub.budget_used = 0.20
        sub.total_tokens_used = 700
        monitor.track(sub)
        monitor.start()
        try:
            assert _wait_for(
                lambda: any(
                    e.get("cost") == "$0.3000"
                    and e.get("total_tokens") == 1000
                    and e.get("total_steps") == 2
                    for e in printer.usage_events("parent-exec-task")
                )
            ), "parent live executor spend missing from the aggregate"
        finally:
            monitor.stop()

    def test_sub_agent_live_session_spend_included(self) -> None:
        """A sub-agent's own in-flight executor session (not yet folded
        into its totals by relentless) must be counted."""
        sub: Any = KISSAgent("sub")
        sub.budget_used = 0.10
        sub.total_tokens_used = 100
        sub.total_steps = 1
        sub_executor = KISSAgent("sub-session")
        sub_executor.budget_used = 0.05
        sub_executor.total_tokens_used = 50
        sub_executor.step_count = 2
        sub._current_executor = sub_executor
        assert _live_agent_usage(sub) == (
            pytest.approx(0.15),
            150,
            3,
        )

    def test_live_agent_usage_without_executor(self) -> None:
        """Agents without a live executor report just their totals."""
        sub: Any = KISSAgent("sub")
        sub.budget_used = 0.10
        sub.total_tokens_used = 100
        assert _live_agent_usage(sub) == (pytest.approx(0.10), 100, 0)

    def test_no_printer_is_a_noop(self) -> None:
        """Without a printer the monitor never starts a thread."""
        parent = SorcarAgent("no-printer-parent")
        monitor = _LiveUsageMonitor(parent, None, interval=0.01)
        monitor.start()
        assert monitor._thread is None
        monitor.stop()  # must not raise

    def test_stop_joins_before_offsets_bump_no_double_count(self) -> None:
        """After ``stop()`` returns no further emission may occur, so the
        subsequent ``_attribute_sub_usage`` offset bump cannot be combined
        with a stale raw emission into a double-counted display."""
        printer = _RecordingPrinter()
        parent = _parent_with_task(printer, "stop-task")
        monitor = _LiveUsageMonitor(parent, printer, interval=0.01)
        sub: Any = KISSAgent("sub-0")
        monitor.track(sub)
        monitor.start()
        sub.budget_used = 0.25
        sub.total_tokens_used = 1000
        sub.total_steps = 4
        assert _wait_for(lambda: len(printer.usage_events("stop-task")) >= 1)
        monitor.stop()
        emitted_before = len(printer.usage_events("stop-task"))
        _attribute_sub_usage(parent, 0.25, 1000, 4)
        assert printer.budget_offset == pytest.approx(0.25)
        assert printer.tokens_offset == 1000
        assert printer.steps_offset == 4
        assert parent.budget_used == pytest.approx(0.25)
        time.sleep(0.1)
        assert len(printer.usage_events("stop-task")) == emitted_before, (
            "monitor emitted after stop(): a stale raw emission would be "
            "offset by the just-bumped totals and double-count the display"
        )
        # Every pre-stop emission carried offset 0 — never double counted.
        for e in printer.usage_events("stop-task"):
            assert e["cost"] == "$0.2500"

    def test_monitor_survives_a_misbehaving_agent(self) -> None:
        """An exception while polling one agent must neither kill the
        monitor nor blind the header to the OTHER sub-agents' spend."""

        class _ExplodingUsage:
            @property
            def budget_used(self) -> float:
                raise ValueError("usage backend unavailable")

        printer = _RecordingPrinter()
        parent = _parent_with_task(printer, "resilient-task")
        monitor = _LiveUsageMonitor(parent, printer, interval=0.01)
        monitor.track(_ExplodingUsage())
        good = KISSAgent("good-sub")
        good.budget_used = 0.30
        good.total_tokens_used = 10
        monitor.track(good)
        monitor.start()
        try:
            assert _wait_for(
                lambda: any(
                    e.get("cost") == "$0.3000"
                    for e in printer.usage_events("resilient-task")
                )
            ), "a misbehaving sibling hid the healthy sub-agent's spend"
            assert monitor._thread is not None and monitor._thread.is_alive()
        finally:
            monitor.stop()


@skip_no_key
class TestRunTasksParallelLiveUsage:
    """Real-LLM e2e: the fixed fan-out path streams live parent usage."""

    @pytest.mark.slow
    def test_parent_usage_streams_during_fanout(self, tmp_path: Path) -> None:
        """``SorcarAgent._run_tasks_parallel`` must broadcast parent-task
        ``usage_info`` while sub-agents run, and the final attribution
        must equal the sub-agents' summed spend (no double counting)."""
        printer = _RecordingPrinter()
        parent = _parent_with_task(printer, "fanout-parent-task")
        parent.model_name = FAST_MODEL
        parent.work_dir = str(tmp_path)

        results = parent._run_tasks_parallel(
            [
                "What is 2 + 2? Reply with just the number.",
                "What is 3 + 3? Reply with just the number.",
            ],
            max_workers=2,
        )
        assert len(results) == 2

        live = printer.usage_events("fanout-parent-task")
        assert live, (
            "no live parent-task usage_info during the fan-out: the "
            "header would stay stale until every sub-agent finished"
        )
        costs = [float(e["cost"].lstrip("$")) for e in live]
        assert max(costs) > 0.0
        # Monotonic non-decreasing live cost stream.
        assert costs == sorted(costs)
        # Final attribution: parent totals now include the sub-agents'
        # spend, and the printer offsets equal those totals exactly
        # (displayed final == attributed final; nothing double counted).
        assert parent.budget_used > 0.0
        assert printer.budget_offset == pytest.approx(parent.budget_used)
        assert printer.tokens_offset == parent.total_tokens_used
        assert printer.steps_offset == parent.total_steps
        # Live stream never exceeded the final attributed total.
        assert max(costs) <= parent.budget_used + 1e-9
