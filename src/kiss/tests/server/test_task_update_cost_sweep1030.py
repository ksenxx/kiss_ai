# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: a stopped task update reports what it spent.

``TaskUpdate.cost`` is the spend of the last run, shown on the
task-info panel's status line.  ``TaskUpdateRunner._run`` set it only
from a run that returned: an update child that was stopped (or failed)
reported $0 although its spend had been charged to the task.

Real pieces throughout: a ``TaskUpdateRunner`` running the production
child against a scripted local model endpoint, the real agent-state
registry and the task's real stop event.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _add_task
from kiss.agents.sorcar.sorcar_agent import _agent_usage
from kiss.server import agent_state
from kiss.server.task_update import TaskUpdateRunner
from kiss.tests.agents.sorcar.local_model_server import MODEL, serve, tool_call_body

DEADLINE_S = 30.0


def _wait_until(predicate: Any, what: str) -> None:
    """Poll *predicate* until true; fail naming *what* after :data:`DEADLINE_S`."""
    deadline = time.monotonic() + DEADLINE_S
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.02)


def test_a_stopped_update_reports_the_spend_charged_to_the_task(tmp_path: Path) -> None:
    """The child calls the model a few times, the task is stopped: its cost is shown."""
    parent = ChatSorcarAgent("task-update-cost-parent")
    parent.work_dir = str(tmp_path)
    task_id, chat_id = _add_task(
        "Parent task prompt", chat_id="",
        extra={"model": MODEL, "work_dir": str(tmp_path), "startTs": int(time.time() * 1000)},
    )
    parent.resume_chat_by_id(chat_id)
    parent._last_task_id = task_id
    stop_event = threading.Event()
    state = agent_state.AgentState(
        task_id, agent=parent, tab_id="tab-update-cost", server_owned=True,
        stop_event=stop_event, is_task_active=True,
    )
    agent_state.register(state)
    # The scripted reply repeats: the child calls ``task_context`` on
    # every step until the stop ends it.
    script = [tool_call_body("task_context", {"task_id": task_id}, prompt_tokens=20_000)]
    runner = TaskUpdateRunner()
    try:
        with serve(script) as (url, requests):
            parent.model_name = MODEL
            parent.model_config = {"base_url": url, "api_key": "local"}
            assert runner.poll(task_id, parent, force=True).running
            _wait_until(lambda: len(requests) >= 2, "the child's model calls")
            stop_event.set()
            _wait_until(lambda: not runner.poll(task_id, parent).running, "the update to end")
    finally:
        agent_state.unregister(task_id, state)

    upd = runner.poll(task_id, parent)
    assert upd.error == "The task was stopped before this update finished."
    charged = _agent_usage(parent)[0]
    assert charged > 0.0, "the stopped child's spend was not charged to the task"
    assert upd.cost == pytest.approx(charged)
