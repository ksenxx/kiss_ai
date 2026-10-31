# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A retired sub-task's commit-message spend reaches its still-running parent.

When a sub-agent's tab closes after its task ended with an unmerged
worktree, ``retire_for_disposal`` commits the work with a fast-model
commit message billed to the sub-agent after its row was saved.  The
spend is added to the sub-task's row; its parent is still running, so the
parent's final save reads only its live counters, and the share must be
banked on the parent's live agent (the daemon's late-usage policy,
reached through the printer's ``charge_task_usage``).  The commit message
is a real fast-model call, so the test needs an API-billed fast model.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _add_task, _get_db
from kiss.agents.sorcar.sorcar_agent import _agent_usage
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.tests.agents.sorcar.test_autocommit_notifications import _setup_worktree_agent
from kiss.tests.agents.sorcar.test_cost_audit_20261020 import (
    _skip_unless_fast_model_is_api_billed,
)


def _row_cost(task_id: str) -> float:
    """The persisted cost of *task_id*."""
    row = _get_db().execute(
        "SELECT cost FROM task_history WHERE id = ?", (task_id,),
    ).fetchone()
    return float(row[0])


def _task(cost: float, finished: bool, parent: str = "") -> str:
    """Add a task row; return its id."""
    extra: dict[str, Any] = {"cost": cost, "tokens": 100, "steps": 2}
    if finished:
        extra["endTs"] = int(time.time() * 1000)
    if parent:
        extra["parent_task_id"] = parent
    task_id, _ = _add_task("retire sweep task", "", extra)
    return task_id


def test_retired_sub_task_charges_its_running_parent(tmp_path: Path) -> None:
    _skip_unless_fast_model_is_api_billed()
    root = _task(3.0, finished=False)
    child = _task(0.5, finished=True, parent=root)
    parent_agent = ChatSorcarAgent("retire-parent")
    state = AgentState(root, agent=parent_agent, tab_id="tab-retire-parent", server_owned=True)
    agent_state.register(state)
    try:
        agent, _repo, wt_dir = _setup_worktree_agent(tmp_path, "retire-sub")
        agent.printer = JsonPrinter()
        agent._last_task_id = child
        (wt_dir / "work.txt").write_text("unmerged sub-task work\n")
        before = _agent_usage(agent)[0]

        assert agent.retire_for_disposal()

        spent = _agent_usage(agent)[0] - before
        assert spent > 0
        assert _row_cost(child) == pytest.approx(0.5 + spent)
        assert _row_cost(root) == pytest.approx(3.0)
        assert _agent_usage(parent_agent)[0] == pytest.approx(spent)
    finally:
        agent_state.unregister(root, state)
