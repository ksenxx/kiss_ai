# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The terminal ``task_done`` event carries the agent's own ``success`` verdict.

A run that ends with ``finish(success=False)`` — a sub-agent's partial
result on budget exhaustion, or any task the agent gave up on — used to
end in a bare ``{"type": "task_done"}``; the frontend's ``markTabDone``
reads ``ev.success === false`` from that event, so nothing on the
terminal event said the task had failed.  Now the parsed verdict rides
on the event; an unparsable result adds nothing.

Runs the real ``VSCodeServer._run_task`` with a scripted agent and a
real ``JsonPrinter`` subclass; no mocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.persistence import _add_task
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.core.utils import finish
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer


class _CapturePrinter(JsonPrinter):
    """Real printer subclass that records every broadcast event."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*, then run the real record/persist path."""
        self.events.append(dict(event))
        super().broadcast(event)


class _ScriptedAgent(WorktreeSorcarAgent):
    """Agent whose ``run`` returns the ``finish`` result named in the prompt.

    ``[fail]`` → ``finish(success=False)``; ``[unparsable]`` → non-YAML
    text; anything else → ``finish(success=True)``.
    """

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Allocate a task row and return the scripted result."""
        prompt_template = kwargs.get("prompt_template", "")
        printer = kwargs.get("printer")
        task_id, self._chat_id = _add_task(prompt_template, chat_id=self._chat_id or "")
        with self._task_id_lock:
            self._last_task_id = task_id
        if printer is not None:
            printer._thread_local.task_id = str(task_id)
            printer._thread_local.task_id = ""
        if "[unparsable]" in prompt_template:
            return "not yaml at all: [unbalanced"
        return finish(
            "[fail]" not in prompt_template,
            summary_in_html=f"<p>partial or done: {prompt_template}</p>",
        )


def _task_done_event(tmp_path: Path, tab_id: str, prompt: str) -> dict[str, Any]:
    models = get_available_models()
    if not models:
        pytest.skip("no models configured in this environment")
    printer = _CapturePrinter()
    server = VSCodeServer(printer=printer)
    agent = _ScriptedAgent("Sorcar VS Code")
    state = AgentState(f"pre-{tab_id}", agent=agent, tab_id=tab_id, server_owned=True)
    agent_state.register(state)
    try:
        server._run_task({
            "tabId": tab_id,
            "prompt": prompt,
            "workDir": str(tmp_path),
            "model": models[0],
            "_state_key": state.task_id,
        })
    finally:
        with agent_state.STATE_LOCK:
            stale = [st.task_id for st in agent_state.snapshot() if st.tab_id == tab_id]
        for task_id in stale:
            agent_state.unregister(task_id)
    done = [e for e in printer.events if e.get("type") == "task_done"]
    assert len(done) == 1, [e.get("type") for e in printer.events]
    return done[0]


def test_failed_finish_marks_task_done_unsuccessful(tmp_path: Path) -> None:
    ev = _task_done_event(tmp_path, "tdv-fail-tab", "tdv gave up [fail]")
    assert ev["success"] is False
    assert ev["tabId"] == "tdv-fail-tab"


def test_successful_finish_marks_task_done_successful(tmp_path: Path) -> None:
    ev = _task_done_event(tmp_path, "tdv-ok-tab", "tdv all good")
    assert ev["success"] is True


def test_unparsable_result_carries_no_verdict(tmp_path: Path) -> None:
    ev = _task_done_event(tmp_path, "tdv-raw-tab", "tdv raw [unparsable]")
    assert "success" not in ev
