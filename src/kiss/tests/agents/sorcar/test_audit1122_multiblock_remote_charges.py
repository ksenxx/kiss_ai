# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A remote run of several ``<task>`` blocks: each block's spend under its own row id.

``run_agent(machine=...)`` on a prompt of several ``<task>`` blocks runs
one task per block on the machine, each with its own row and counters.
The local parent is charged their sum once when the dispatch returns;
the record it keeps of that fold (``remote_subtasks``) is one per task,
so that

* the Spend panel files each block under the model it ran on;
* spend the machine reports later for ANY block (``getTaskUsage``,
  ``charge_remote_subtask_usage``) charges the parent that block's
  growth, once.  Audit 2026-11-22: the whole sum was recorded under the
  last block's id, so an earlier block's report named an unknown task
  and the last block's growth hid under the inflated record;
* a block the run was cut in before it reported is recorded at zero;
* a run of one task, and one whose task the daemon never announced by
  its row, keep the one record they always had.

The events are the daemon's, folded by the client's own ``_Observed``;
the parent is a real finished row, charged through a real agent's
late-charge policy in a history database of the test's own.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.agent_dispatch import _attribute_dispatch_usage
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.daemon_client import TaskResult, _Observed
from kiss.server.task_update import charge_remote_subtask_usage
from kiss.tests.agents.sorcar.test_audit1103_remote_costs import (
    ASTRA,
    FABLE,
    OPUS,
    _costs_by_model,
    _row,
    _task,
)
from kiss.tests.agents.sorcar.test_history_date_range import _redirect, _restore

A, B = "a" * 32, "b" * 32


@pytest.fixture(autouse=True)
def history(tmp_path: Path) -> Iterator[None]:
    """A history database of the test's own."""
    saved = _redirect(str(tmp_path))
    try:
        yield
    finally:
        th._close_thread_db()
        _restore(saved)


def _announce(task: str, model: str) -> dict[str, Any]:
    """The ``task_settings`` event of *task*, run on *model* (no row named when *task* is empty)."""
    return {"type": "task_settings", "taskId": task, "settings": {"model": model}}


def _spend(task: str, cost: float, kind: str = "result") -> dict[str, Any]:
    """A *kind* event of *task* with totals of *cost* USD, 100 tokens and one step per dollar."""
    return {
        "type": kind, "taskId": task, "cost": f"${cost:.4f}", "cost_usd": cost,
        "total_tokens": int(cost * 100), "step_count": int(cost), "success": True,
    }


def _fold(events: list[dict[str, Any]]) -> tuple[str, TaskResult]:
    """Fold a run of *events* on the machine ``box`` into a finished $10 parent; return both."""
    seen = _Observed(run_token="run")
    charged: dict[str, dict[str, Any]] = {}
    for event in events:
        seen.absorb(event, charged)
    result = seen.task_result()
    parent = _task(OPUS, 10.0, 1000, 10)
    _attribute_dispatch_usage(
        ChatSorcarAgent("multiblock-parent"), result, parent_task_id=parent, machine="box",
    )
    return parent, result


def _records() -> dict[str, tuple[str, float, int, int]]:
    """The ``remote_subtasks`` records: ``{task_id: (model, cost, tokens, steps)}``."""
    rows = th._get_db().execute("SELECT task_id, model, cost, tokens, steps FROM remote_subtasks")
    return {row[0]: (row[1], row[2], row[3], row[4]) for row in rows}


def test_each_block_is_recorded_and_charged_late_under_its_own_id() -> None:
    parent, result = _fold([
        _announce(A, ASTRA), _spend(A, 1.0, "usage_info"), _spend(A, 1.0),
        _announce(B, FABLE), _spend(B, 2.0),
    ])
    # The result and the parent's charge are the sum, once, as they were.
    assert (result.task_id, result.cost, result.tokens, result.steps) == (B, 3.0, 300, 3)
    assert result.settings == {"model": FABLE}
    assert _row(parent) == (13.0, 1300, 13)
    assert _records() == {A: (ASTRA, 1.0, 100, 1), B: (FABLE, 2.0, 200, 2)}
    assert _costs_by_model() == {OPUS: 10.0, ASTRA: 1.0, FABLE: 2.0}
    # The machine's later report of either block charges that block's growth, once.
    for task, cost in ((A, 1.5), (B, 2.5), (A, 1.5), (B, 2.5)):
        charge_remote_subtask_usage(
            None, task, {"cost": cost, "tokens": int(cost * 100), "steps": int(cost) + 1},
        )
    assert _row(parent) == (14.0, 1400, 15)
    assert _costs_by_model() == {OPUS: 10.0, ASTRA: 1.5, FABLE: 2.5}


def test_a_block_cut_before_it_reported_is_recorded_at_zero() -> None:
    parent, result = _fold([_announce(A, ASTRA), _spend(A, 1.0), _announce(B, FABLE)])
    assert (result.task_id, result.cost, result.tokens, result.steps) == (B, 1.0, 100, 1)
    assert _records() == {A: (ASTRA, 1.0, 100, 1), B: (FABLE, 0.0, 0, 0)}
    # What the block went on to spend on the machine is all growth.
    charge_remote_subtask_usage(None, B, {"cost": 2.0, "tokens": 200, "steps": 2})
    assert _row(parent) == (13.0, 1300, 13)


@pytest.mark.parametrize(("events", "records"), [
    # One task, announced by its row.
    ([_announce(A, ASTRA), _spend(A, 1.0, "usage_info"), _spend(A, 2.0)],
     {A: (ASTRA, 2.0, 200, 2)}),
    # Announced, but its one report names the run, not the row.
    ([_announce(A, ASTRA), _spend("run", 2.0)], {A: (ASTRA, 2.0, 200, 2)}),
    # Never announced by a row: one task, whatever its reports name.
    ([_announce("", ASTRA), _spend(B, 1.0, "usage_info"), _spend(A, 2.0)],
     {A: (ASTRA, 2.0, 200, 2)}),
    # No row named at all: nothing to ask the machine about later.
    ([_announce("", ASTRA), _spend("", 2.0)], {}),
])
def test_a_run_of_one_task_keeps_its_one_record(
    events: list[dict[str, Any]], records: dict[str, tuple[str, float, int, int]],
) -> None:
    parent, result = _fold(events)
    assert (result.cost, result.tokens, result.steps) == (2.0, 200, 2)
    assert _row(parent) == (12.0, 1200, 12)
    assert _records() == records
