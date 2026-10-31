# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The merge agent's spend follows the daemon's one late-spend policy.

``_MergeFlowMixin._persist_merge_agent_usage`` charged the merge agent's
spend to the merged task's own row only.  It now goes through
``charge_side_channel_usage`` like every other late spend: a finished
row and its finished ancestors get it, a running ancestor banks it on
its live agent, and the merge tab gets the new totals stamped with the
task's id.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _add_task, _get_db
from kiss.agents.sorcar.sorcar_agent import _agent_usage, _attribute_sub_usage
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.merge_flow import _MergeFlowMixin
from kiss.tests.server.parallel_agent_harness import CapturePrinter

pytestmark = pytest.mark.usefixtures("clean_registry")


class _Flow(_MergeFlowMixin):
    """The merge-flow mixin with the printer its usage method uses."""

    def __init__(self) -> None:
        self.capture = CapturePrinter()
        self.printer = self.capture


def _task(finished: bool, parent: str = "") -> str:
    """Add a task row costing $0.50 / 100 tokens / 3 steps; return its id."""
    extra: dict[str, Any] = {"tokens": 100, "cost": 0.5, "steps": 3}
    if finished:
        extra["endTs"] = int(time.time() * 1000)
    if parent:
        extra["parent_task_id"] = parent
    task_id, _ = _add_task("merged task", "", extra)
    return task_id


def _row(task_id: str) -> tuple[int, float, int]:
    """The persisted ``(tokens, cost, steps)`` of *task_id*."""
    row = _get_db().execute(
        "SELECT tokens, cost, steps FROM task_history WHERE id = ?", (task_id,),
    ).fetchone()
    return int(row[0]), float(row[1]), int(row[2])


def _merged_agent(task_id: str) -> tuple[ChatSorcarAgent, tuple[float, int, int]]:
    """An agent of *task_id* whose merge agent spent $0.25 / 40 tokens / 2 steps."""
    agent = ChatSorcarAgent("merged")
    agent._last_task_id = task_id
    before = _agent_usage(agent)
    _attribute_sub_usage(agent, 0.25, 40, 2)
    return agent, before


def test_finished_ancestors_get_the_merge_spend() -> None:
    parent = _task(finished=True)
    child = _task(finished=True, parent=parent)
    agent, before = _merged_agent(child)
    flow = _Flow()

    flow._persist_merge_agent_usage(agent, before, "merge-tab")

    assert _row(child) == (140, 0.75, 5)
    assert _row(parent) == (140, 0.75, 5)
    by_task = {e["taskId"]: e for e in flow.capture.events_of_type("usage_info")}
    assert by_task[child]["tabId"] == "merge-tab"
    assert by_task[child]["total_tokens"] == 140
    # The ancestor's totals go to the tabs watching it, not the merge tab.
    assert by_task[parent]["tabId"] != "merge-tab"
    assert flow.capture.events_of_type("tasks_updated")


def test_a_running_ancestor_banks_the_merge_spend() -> None:
    parent = _task(finished=False)
    child = _task(finished=True, parent=parent)
    parent_agent = ChatSorcarAgent("parent")
    agent_state.register(AgentState(parent, agent=parent_agent, is_task_active=True))
    agent, before = _merged_agent(child)

    _Flow()._persist_merge_agent_usage(agent, before, "merge-tab")

    assert _row(child) == (140, 0.75, 5)
    assert _row(parent) == (100, 0.5, 3)
    assert _agent_usage(parent_agent) == (0.25, 40, 2)
