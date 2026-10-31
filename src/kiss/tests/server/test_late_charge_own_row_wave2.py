# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A late charge for a task's own unfinished row logs no false warning.

``SorcarAgent._charge_last_task`` charges spend made after its task's
row was saved through ``charge_task_usage(None, task_id, ...)``: it
passes no live agent because the spend is already on that agent's
counters.  When the row is still unfinished, ``_charge_locked`` warned
"side-channel spend ... not charged to running task ...: no live agent"
although nothing was lost: the row's final save reads those counters.
A running ancestor with no live agent still warns, since that spend is
lost.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest

from kiss.agents.sorcar.persistence import _add_task, _get_db
from kiss.server.task_update import charge_side_channel_usage

pytestmark = pytest.mark.usefixtures("clean_registry")

_LOGGER = "kiss.server.task_update"


def _task(finished: bool, parent: str = "") -> str:
    """Add a task row costing $1.00 / 100 tokens / 2 steps; return its id."""
    extra: dict[str, Any] = {"cost": 1.0, "tokens": 100, "steps": 2}
    if finished:
        extra["endTs"] = int(time.time() * 1000)
    if parent:
        extra["parent_task_id"] = parent
    task_id, _ = _add_task("late charge task", "", extra)
    return task_id


def _row(task_id: str) -> tuple[int, float, int]:
    """The persisted ``(tokens, cost, steps)`` of *task_id*."""
    row = _get_db().execute(
        "SELECT tokens, cost, steps FROM task_history WHERE id = ?", (task_id,),
    ).fetchone()
    return int(row[0]), float(row[1]), int(row[2])


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _LOGGER]


def test_own_unfinished_row_without_agent_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    task_id = _task(finished=False)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        charge_side_channel_usage(None, None, task_id, 0.25, 40, 1)
    assert _warnings(caplog) == []
    # The unfinished row is left to its final save.
    assert _row(task_id) == (100, 1.0, 2)


def test_running_ancestor_without_agent_still_warns(caplog: pytest.LogCaptureFixture) -> None:
    root = _task(finished=False)
    child = _task(finished=True, parent=root)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        charge_side_channel_usage(None, None, child, 0.25, 40, 1)
    assert any(
        f"not charged to running task {root}: no live agent" in message
        for message in _warnings(caplog)
    )
    assert _row(child) == (140, 1.25, 3)
