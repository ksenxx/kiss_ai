# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: an agent that raises while a Stop is pending ends with ONE result.

When the user pressed Stop and the agent then raised a plain
``Exception`` (a tool aborted by the stop, say), the per-subtask
``except Exception`` branch broadcast ``Task failed: …`` without
acknowledging the stop.  The watchdog armed by ``_stop_task`` therefore
still owned the thread and, one second later, injected
``KeyboardInterrupt`` into whatever the runner was doing — when that
was the failure broadcast itself, the outer handler ran
``_cancel_outcome`` and broadcast a SECOND terminal result ("Task
stopped by user"), both recorded under the same row.

The fix reports a stop that was pending when the agent raised as the
stop (``_cancel_outcome``), which also acknowledges it so the watchdog
has nothing left to inject.

The test drives the REAL ``_run_task`` worker, the REAL stop watchdog
and the REAL SQLite persistence.  The agent's LLM loop presses Stop
and raises at once; the printer dwells past the watchdog's grace
period on the terminal ``result`` broadcast, where an injection would
land before the fix.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from typing import Any

import pytest

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer


class _SlowResultPrinter(JsonPrinter):
    """Real printer that records events and dwells on the first ``result``."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*; the first ``result`` takes longer than the watchdog's grace."""
        self.events.append(dict(event))
        if event.get("type") == "result" and self.events.count(event) == 1:
            time.sleep(1.6)
        super().broadcast(event)


class _StopThenRaiseAgent(WorktreeSorcarAgent):
    """Agent whose loop presses Stop and then raises a plain exception."""

    server: VSCodeServer
    tab_id: str
    task_id: str = ""

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Allocate the task row, press Stop, raise."""
        self.task_id, self._chat_id = _persistence._add_task(
            kwargs.get("prompt_template", ""),
            chat_id=self._chat_id or "",
        )
        with self._task_id_lock:
            self._last_task_id = self.task_id
        self.server._stop_task(self.tab_id)
        raise RuntimeError("tool aborted")


def test_exception_with_pending_stop_yields_one_result() -> None:
    models = get_available_models()
    if not models:
        pytest.skip("no models configured in this environment")
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    tab_id = "stop-then-raise-tab"
    printer = _SlowResultPrinter()
    server = VSCodeServer(printer=printer)
    agent = _StopThenRaiseAgent("Sorcar VS Code")
    agent.server = server
    agent.tab_id = tab_id
    # ``_cmd_run`` installs the stop event and the thread before start.
    state = AgentState(
        f"pre-{tab_id}",
        agent=agent,
        tab_id=tab_id,
        server_owned=True,
        stop_event=threading.Event(),
    )
    agent_state.register(state)
    worker = threading.Thread(
        target=server._run_task,
        args=(
            {
                "type": "run",
                "tabId": tab_id,
                "prompt": "stop then raise",
                "workDir": tempfile.mkdtemp(prefix="stop-then-raise-"),
                "model": models[0],
                "autoCommit": False,
                "_state_key": state.task_id,
            },
        ),
        daemon=True,
    )
    state.task_thread = worker
    try:
        worker.start()
        worker.join(timeout=30)
        assert not worker.is_alive(), "worker never finished"
    finally:
        with agent_state.STATE_LOCK:
            stale = [st.task_id for st in agent_state.snapshot() if st.tab_id == tab_id]
        for key in stale:
            agent_state.unregister(key)

    results = [e for e in printer.events if e.get("type") == "result"]
    assert len(results) == 1, [e.get("text") for e in results]
    assert results[0]["text"] == "Task stopped by user"
    assert results[0]["success"] is False
    ends = [e for e in printer.events if e.get("type") == "task_stopped"]
    assert len(ends) == 1, [e.get("type") for e in printer.events]
    _persistence._flush_chat_events()
    row = (
        _persistence._get_db()
        .execute(
            "SELECT result FROM task_history WHERE id = ?",
            (agent.task_id,),
        )
        .fetchone()
    )
    assert row is not None and row["result"] == "Task stopped by user"
