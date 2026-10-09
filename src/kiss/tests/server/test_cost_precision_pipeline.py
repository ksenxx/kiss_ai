# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests: a task's exact spend survives the event protocol.

Reproduces the October 2026 cost audit finding: the ``usage_info`` and
``result`` events carried a task's cost only as the ``"$x.xxxx"``
display string, and ``daemon_client`` parsed that rounded string back
into the :class:`TaskResult` a ``run_agent`` / ``run_parallel`` parent
charges to its own ledger.  A child that spent $0.000049 was charged
as $0, and every dispatched child lost up to $0.00005 of real spend
in the parent's persisted total.  The events now also carry
``cost_usd``, the unrounded USD figure (printer offset applied), and
the client prefers it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar.daemon_client import _net_totals, _to_task_result
from kiss.agents.sorcar.persistence import _add_task
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _agent_usage, _attribute_sub_usage
from kiss.core.kiss_agent import KISSAgent
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.server.task_update import charge_side_channel_usage

TINY = 0.000049  # rounds to "$0.0000"


class _CapturingPrinter(JsonPrinter):
    """A real :class:`JsonPrinter` whose broadcasts land in ``events``."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event* instead of sending it anywhere."""
        with self._events_lock:
            self.events.append(event)

    def broadcast_transient(
        self, event: dict[str, Any], task_id: Any = None, tab_id: str = "",
    ) -> None:
        """Record the tab-stamped *event* (a late charge's ``usage_info``)."""
        self.broadcast(dict(event, taskId=task_id))


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Leave no agent states behind."""
    yield
    for state in agent_state.snapshot():
        agent_state.unregister(state.task_id, state)


def _events_of(printer: _CapturingPrinter, kind: str) -> list[dict[str, Any]]:
    with printer._events_lock:
        return [e for e in printer.events if e.get("type") == kind]


def test_usage_info_event_carries_exact_cost_with_offset() -> None:
    """The per-step ``usage_info`` keeps the unrounded cost, offset included."""
    printer = _CapturingPrinter()
    printer.set_usage_offsets("", tokens=10, budget=0.1, steps=1)
    printer.print(
        "Steps: 1", type="usage_info", total_tokens=5, cost=f"${TINY:.4f}",
        cost_usd=TINY, total_steps=1,
    )
    [event] = _events_of(printer, "usage_info")
    assert event["cost"] == "$0.1000"
    assert event["cost_usd"] == pytest.approx(0.1 + TINY)
    assert event["total_tokens"] == 15
    assert event["total_steps"] == 2


def test_usage_info_without_exact_cost_falls_back_to_the_string() -> None:
    """An emitter that only knows the display string still yields a number."""
    printer = _CapturingPrinter()
    printer.set_usage_offsets("", tokens=0, budget=0.25, steps=0)
    printer.print("", type="usage_info", total_tokens=1, cost="$0.5000", total_steps=1)
    [event] = _events_of(printer, "usage_info")
    assert event["cost"] == "$0.7500"
    assert event["cost_usd"] == pytest.approx(0.75)
    printer.print("", type="usage_info", total_tokens=1, cost="N/A", total_steps=1)
    assert "cost_usd" not in _events_of(printer, "usage_info")[1]


def test_kiss_agent_result_event_carries_exact_cost() -> None:
    """``KISSAgent``'s terminal ``result`` event keeps the unrounded spend."""
    printer = _CapturingPrinter()
    agent = KISSAgent("precise-result")
    agent.printer = printer
    agent.budget_used = TINY
    agent.total_tokens_used = 100
    agent.step_count = 1
    agent._print_result("done")
    [event] = _events_of(printer, "result")
    assert event["cost"] == "$0.0000"
    assert event["cost_usd"] == pytest.approx(TINY)


def test_final_usage_totals_carry_exact_cost() -> None:
    """The run's last ``usage_info`` (absolute ledger totals) is unrounded."""
    printer = _CapturingPrinter()
    agent = SorcarAgent("precise-final")
    agent.printer = printer
    _attribute_sub_usage(agent, TINY, 100, 1)
    agent._emit_usage_totals()
    [event] = _events_of(printer, "usage_info")
    assert event["cost"] == "$0.0000"
    assert event["cost_usd"] == pytest.approx(TINY)
    assert event["total_tokens"] == 100


def test_late_charge_usage_info_carries_exact_cost() -> None:
    """A side channel's late charge republishes the row's exact total."""
    task_id, _ = _add_task(
        "precise late charge", "",
        {"cost": 1.0, "tokens": 100, "steps": 2, "endTs": int(time.time() * 1000)},
    )
    printer = _CapturingPrinter()
    charge_side_channel_usage(printer, SorcarAgent("late"), task_id, TINY, 1, 0)
    [event] = [e for e in _events_of(printer, "usage_info") if e.get("taskId") == task_id]
    assert event["cost"] == "$1.0000"
    assert event["cost_usd"] == pytest.approx(1.0 + TINY)


def test_daemon_client_prefers_the_exact_cost() -> None:
    """``run_agent``'s client folds ``cost_usd``, not the rounded string."""
    charged = {"cost": 0.0, "tokens": 0, "steps": 0}
    event = {
        "type": "result", "text": "ok", "success": True,
        "cost": f"${TINY:.4f}", "cost_usd": TINY, "total_tokens": 100, "step_count": 1,
    }
    totals = _net_totals(event, charged)
    assert totals["cost"] == pytest.approx(TINY)
    assert _to_task_result(event, totals=totals).cost == pytest.approx(TINY)
    assert _to_task_result(event).cost == pytest.approx(TINY)
    # Older daemons without the field: the string is still honoured.
    legacy = {"type": "result", "cost": "$0.1235", "total_tokens": 1, "step_count": 1}
    assert _net_totals(legacy, charged)["cost"] == pytest.approx(0.1235)
    # A banked ancestor delta is subtracted from the exact figure.
    banked = dict(event, ancestor_charged={"cost": 0.000009, "tokens": 10, "steps": 0})
    assert _net_totals(banked, charged)["cost"] == pytest.approx(TINY - 0.000009)
    assert charged["cost"] == pytest.approx(0.000009)


def test_child_spend_reaches_the_parent_ledger_exactly() -> None:
    """End to end: child agent -> result event -> client -> parent ledger."""
    from kiss.agents.sorcar.agent_dispatch import _attribute_dispatch_usage

    printer = _CapturingPrinter()
    child = KISSAgent("precise-child")
    child.printer = printer
    child.budget_used = TINY
    child.total_tokens_used = 100
    child.step_count = 1
    child._print_result("child done")
    [event] = _events_of(printer, "result")
    charged = {"cost": 0.0, "tokens": 0, "steps": 0}
    result = _to_task_result(event, totals=_net_totals(event, charged))

    parent = SorcarAgent("precise-parent")
    _attribute_dispatch_usage(parent, result, None, "")
    budget, tokens, _steps = _agent_usage(parent)
    assert budget == pytest.approx(TINY)
    assert tokens == 100


def test_terminal_result_builders_carry_exact_cost() -> None:
    """The daemon's failure / Stop ``result`` and the agent's own carry ``cost_usd``."""
    from kiss.server.task_runner import _result_event

    agent = SorcarAgent("precise-terminal")
    _attribute_sub_usage(agent, TINY, 100, 1)
    event = _result_event("stopped", success=False, agent=agent)
    assert event["cost"] == "$0.0000"
    assert event["cost_usd"] == pytest.approx(TINY)
    assert _result_event("never ran", success=False)["cost_usd"] == 0.0

    printer = _CapturingPrinter()
    agent.printer = printer
    agent._emit_merged_result_event({"success": True, "summary": "ok"})
    [event] = _events_of(printer, "result")
    assert event["cost_usd"] == pytest.approx(TINY)


def test_recovery_and_report_readers_prefer_exact_cost() -> None:
    """Killed-task recovery and the cost report read ``cost_usd`` first."""
    from kiss.agents.sorcar.persistence import _structured_usage_totals
    from kiss.scripts.cost_report import _event_cost

    exact = {"type": "usage_info", "cost": "$0.0000", "cost_usd": TINY,
             "total_tokens": 5, "total_steps": 1}
    legacy = {"type": "usage_info", "cost": "$0.1235", "total_tokens": 5, "total_steps": 1}
    assert _structured_usage_totals(exact) == {"steps": 1, "tokens": 5, "cost": TINY}
    assert _structured_usage_totals(legacy)["cost"] == 0.1235
    assert _event_cost(exact) == TINY
    assert _event_cost(legacy) == 0.1235
    assert _event_cost({"cost": "N/A"}) == 0.0
