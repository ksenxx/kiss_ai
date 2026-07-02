# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8 (group A): malformed ``$`` cost on the ``usage_info`` path.

``JsonPrinter.print`` has two branches that apply the per-task
``budget_offset`` to a ``"$..."`` cost string:

* the ``result`` branch (via ``_broadcast_result``) wraps the
  ``float(cost[1:])`` arithmetic in ``try/except ValueError`` and
  passes a malformed cost through verbatim (behaviour pinned by
  ``test_simplification_lockdown_printer_config.py``); but
* the ``usage_info`` branch performed the SAME arithmetic with no
  guard, so a malformed dollar cost (e.g. ``"$abc"``) raised
  ``ValueError`` out of ``print()`` — killing the agent worker thread
  that emitted the usage line.

Both paths must tolerate the same inputs identically.
"""

from __future__ import annotations

import unittest
from typing import Any

from kiss.tests.agents.vscode._memory_printer import MemoryPrinter


def _make_task_printer(task_id: str, tab_id: str = "tab1") -> MemoryPrinter:
    printer = MemoryPrinter()
    printer.subscribe_tab(task_id, tab_id)
    printer._thread_local.task_id = task_id
    return printer


def _events_of(printer: MemoryPrinter, ev_type: str) -> list[dict[str, Any]]:
    return [e for e in printer.emitted if e.get("type") == ev_type]


class TestUsageInfoMalformedCost(unittest.TestCase):
    """``usage_info`` must tolerate malformed '$' costs like ``result``."""

    def test_malformed_dollar_cost_does_not_raise(self) -> None:
        """A '$'-prefixed non-numeric cost must not raise ValueError."""
        printer = _make_task_printer("BH8A-1")
        printer.budget_offset = 1.0
        printer.print(
            "usage", type="usage_info",
            total_tokens=5, cost="$abc", total_steps=2,
        )
        events = _events_of(printer, "usage_info")
        assert len(events) == 1, printer.emitted
        # Parity with the result path: malformed cost passes through.
        assert events[0]["cost"] == "$abc", events[0]

    def test_result_and_usage_info_parity_on_malformed_cost(self) -> None:
        """Both branches emit the identical passed-through cost."""
        printer = _make_task_printer("BH8A-2")
        printer.budget_offset = 2.5
        printer.print(
            "usage", type="usage_info",
            total_tokens=1, cost="$1,25", total_steps=1,
        )
        printer.print(
            "done", type="result",
            total_tokens=1, cost="$1,25", step_count=1,
        )
        usage = _events_of(printer, "usage_info")[0]
        result = _events_of(printer, "result")[0]
        assert usage["cost"] == result["cost"] == "$1,25", (usage, result)

    def test_wellformed_dollar_cost_still_offset(self) -> None:
        """The guard must not break the normal offset arithmetic."""
        printer = _make_task_printer("BH8A-3")
        printer.budget_offset = 0.5
        printer.print(
            "usage", type="usage_info",
            total_tokens=0, cost="$1.2500", total_steps=0,
        )
        assert _events_of(printer, "usage_info")[0]["cost"] == "$1.7500"


if __name__ == "__main__":
    unittest.main()
