"""The merged ``result`` event is never over-stated by a concurrent offset publish.

``RelentlessAgent._emit_merged_result_event`` subtracts the printer's
usage offsets from the agent's cumulative snapshot and the printer adds
its CURRENT offsets back inside ``print``.  A finished child's fold
(``sorcar_agent._attribute_sub_usage``) publishes new, larger offsets
from another thread; when that publish landed between the emit's read
and the printer's add, the result carried the old offsets subtracted
and the new ones added, over-stating the task's spend.  The emit now
reads and prints under ``_OFFSET_PUBLISH_LOCK``, the lock every
publisher holds, so the printed total can never exceed the ledger.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage
from kiss.core.printer import Printer

_WAIT = 30.0
_UNIT = (1.0, 100, 3)


class _OffsetAddingPrinter(Printer):
    """Printer that, like the server printer, adds its offsets inside ``print``.

    Each call records the total it would broadcast together with the
    agent's cumulative snapshot taken right after the add: the ledger
    only grows, so a correct total is never above that snapshot.
    """

    def __init__(self, agent: RelentlessAgent) -> None:
        self.agent = agent
        self.tokens_offset = 0
        self.budget_offset = 0.0
        self.steps_offset = 0
        self.records: list[tuple[tuple[float, int, int], tuple[float, int, int]]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:
        """Add the current offsets to the raw totals and record the outcome."""
        total = (
            float(kwargs["cost_usd"]) + self.budget_offset,
            int(kwargs["total_tokens"]) + self.tokens_offset,
            int(kwargs["step_count"]) + self.steps_offset,
        )
        self.records.append((total, self.agent.usage_snapshot()))

    def token_callback(self, token: str) -> None:
        """Unused by the merged-result emit path."""

    def reset(self) -> None:
        """Unused by the merged-result emit path."""


def test_merged_result_total_never_exceeds_ledger_under_concurrent_folds() -> None:
    """Concurrent child folds cannot over-state the merged result event."""
    agent = RelentlessAgent("merged-result-offsets")
    printer = _OffsetAddingPrinter(agent)
    agent.printer = printer  # type: ignore[assignment]
    stop = threading.Event()
    # The race window is a few microseconds wide; a short switch
    # interval makes the interpreter hand the GIL over inside it.
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)

    def fold_storm() -> None:
        while not stop.is_set():
            # Several banks, then ONE publish: the offsets lag the
            # ledger by a few units, so a publish landing inside the
            # emit shows up as a jump of several units.
            for _ in range(4):
                agent._attribute_usage(*_UNIT)
            _attribute_sub_usage(agent, *_UNIT)

    folders = [threading.Thread(target=fold_storm) for _ in range(3)]
    for folder in folders:
        folder.start()
    try:
        for _ in range(2000):
            agent._emit_merged_result_event(
                {"success": True, "is_continue": False, "summary": "done"}
            )
    finally:
        stop.set()
        for folder in folders:
            folder.join(timeout=_WAIT)
            assert not folder.is_alive()
        sys.setswitchinterval(old_interval)

    assert len(printer.records) == 2000
    for total, ceiling in printer.records:
        assert total[0] <= ceiling[0] + 1e-9, (total, ceiling)
        assert total[1] <= ceiling[1], (total, ceiling)
        assert total[2] <= ceiling[2], (total, ceiling)
    # The final offsets are the last published snapshot, untouched by
    # the emits themselves.
    assert (printer.budget_offset, printer.tokens_offset, printer.steps_offset) <= (
        agent.usage_snapshot()
    )
