# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: result panel must include subagent cost offsets.

When the parent agent invokes ``run_parallel``, sub-agent budget/tokens/
steps are accumulated into the parent's printer as ``budget_offset``,
``tokens_offset``, and ``steps_offset``.  These offsets MUST be added
on top of the executor's own usage when the final ``result`` panel is
broadcast — otherwise the reported total cost is lower than the sum of
sub-agent costs, which is what the user reported.

This test exercises ``JsonPrinter.print`` with ``type="result"``
directly (no LLM calls, no mocks): it sets the offsets, fires a result
print, and inspects the broadcast event.
"""

from __future__ import annotations

import threading

from kiss.agents.vscode.json_printer import JsonPrinter


class _CapturePrinter(JsonPrinter):
    """Browser printer that records all broadcast events."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []
        self._ev_lock = threading.Lock()

    def broadcast(self, event: dict) -> None:
        event = self._inject_task_id(event)
        with self._ev_lock:
            self.events.append(event)


def _result_events(printer: _CapturePrinter) -> list[dict]:
    return [e for e in printer.events if e.get("type") == "result"]


def test_result_panel_adds_budget_offset_from_subagents() -> None:
    """Final ``result`` panel cost = executor cost + sub-agent offset.

    Reproduces the issue where the parent agent's displayed cost was
    smaller than the sum of sub-agent costs.  Three sub-agents costing
    0.0615, 0.0721, 0.0636 (sum = 0.1972) plus an executor cost of
    0.1058 should yield a displayed total of $0.3030.
    """
    printer = _CapturePrinter()
    printer._thread_local.task_id = "parent-tab"
    # Simulate run_parallel having accumulated three sub-agents'
    # cost / tokens / steps into the printer offsets.
    printer.budget_offset = 0.0615 + 0.0721 + 0.0636  # = 0.1972
    printer.tokens_offset = 12_345
    printer.steps_offset = 30

    # Executor finishes and prints its own "result" with the executor's
    # OWN budget — sub-agent cost lives in budget_offset, not here.
    printer.print(
        "summary: All three sub-tasks done\nsuccess: true\n",
        type="result",
        cost="$0.1058",
        total_tokens=4_321,
        step_count=7,
    )

    results = _result_events(printer)
    assert len(results) == 1, results
    event = results[0]

    # Total cost must be executor cost + sub-agent offset.
    expected_total_cost = 0.1058 + 0.1972
    assert event["cost"].startswith("$"), event
    got = float(event["cost"][1:])
    assert abs(got - expected_total_cost) < 1e-6, (
        f"expected ${expected_total_cost:.4f}, got {event['cost']!r}"
    )
    assert event["total_tokens"] == 4_321 + 12_345
    assert event["step_count"] == 7 + 30


def test_result_panel_with_no_offsets_unchanged() -> None:
    """Without offsets the result cost equals the executor's own cost.

    Guards the fix against double-counting when no sub-agents ran.
    """
    printer = _CapturePrinter()
    printer._thread_local.task_id = "solo-tab"

    printer.print(
        "summary: ok\nsuccess: true\n",
        type="result",
        cost="$0.0500",
        total_tokens=1_000,
        step_count=3,
    )

    results = _result_events(printer)
    assert len(results) == 1
    event = results[0]
    assert event["cost"] == "$0.0500"
    assert event["total_tokens"] == 1_000
    assert event["step_count"] == 3


def test_result_panel_handles_na_cost() -> None:
    """``cost="N/A"`` must be left alone even when offsets are set."""
    printer = _CapturePrinter()
    printer._thread_local.task_id = "na-tab"
    printer.budget_offset = 0.20
    printer.tokens_offset = 100
    printer.steps_offset = 4

    printer.print(
        "summary: ok\nsuccess: true\n",
        type="result",
        cost="N/A",
        total_tokens=10,
        step_count=1,
    )

    results = _result_events(printer)
    event = results[0]
    assert event["cost"] == "N/A"
    # tokens / steps offsets still applied
    assert event["total_tokens"] == 110
    assert event["step_count"] == 5
