# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a Stop clicked after the agent loop ended is not force-injected.

After the last ``agent.run`` returns, ``_run_task_inner``'s post-run
``finally`` still persists the row, auto-commits and presents the
worktree.  During that cleanup ``state.task_thread`` and
``state.stop_event`` are still set, so ``_stop_task`` accepts a Stop
click and arms the watchdog; and because the run completed normally
``_cancel_outcome`` never raised ``stop_acknowledged``, so one second
later ``_force_stop_thread`` injected ``KeyboardInterrupt`` into the
cleanup: the row stayed at the abrupt-failure sentinel and the runner
logged "Cleanup interrupted".

The fix acknowledges the stop at the top of the post-run ``finally``
(there is nothing left to interrupt), exactly as ``_cancel_outcome``
does for a stop caught inside the loop.

The test drives the REAL ``_run_task`` worker, the REAL stop watchdog
and the REAL SQLite persistence.  The only substitution is the agent's
LLM loop (``run`` returns a finished result at once) and its
``_flush_warnings`` hook, the first call of the post-run cleanup, which
presses Stop and then dwells longer than the watchdog's grace period
so an injection, if any, lands inside the cleanup.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from typing import Any

import pytest

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.core.utils import finish
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer

pytestmark = pytest.mark.usefixtures("stubbed_agent_model")

_RESULT_TEXT = "stop-after-loop finished"


class _CapturePrinter(JsonPrinter):
    """Real printer subclass that records every broadcast event."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*, then run the real record/persist path."""
        self.events.append(dict(event))
        super().broadcast(event)


class _StopDuringCleanupAgent(WorktreeSorcarAgent):
    """Agent that finishes at once and presses Stop from inside the cleanup."""

    server: VSCodeServer
    tab_id: str
    task_id: str = ""

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Allocate the task row and return a successful result."""
        self.task_id, self._chat_id = _persistence._add_task(
            kwargs.get("prompt_template", ""),
            chat_id=self._chat_id or "",
        )
        with self._task_id_lock:
            self._last_task_id = self.task_id
        return finish(True, summary_in_html=f"<p>{_RESULT_TEXT}</p>")

    def _flush_warnings(self, printer: Any) -> None:
        """Press Stop, then dwell past the watchdog's 1 s grace period."""
        super()._flush_warnings(printer)
        self.server._stop_task(self.tab_id)
        time.sleep(1.6)


def test_stop_after_agent_loop_does_not_interrupt_cleanup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    models = get_available_models()
    if not models:
        pytest.skip("no models configured in this environment")
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    tab_id = "stop-after-loop-tab"
    printer = _CapturePrinter()
    server = VSCodeServer(printer=printer)
    agent = _StopDuringCleanupAgent("Sorcar VS Code")
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
    work_dir = tempfile.mkdtemp(prefix="stop-after-loop-")
    worker = threading.Thread(
        target=server._run_task,
        args=(
            {
                "type": "run",
                "tabId": tab_id,
                "prompt": "finish immediately",
                "workDir": work_dir,
                "model": models[0],
                "autoCommit": False,
                "_state_key": state.task_id,
            },
        ),
        daemon=True,
    )
    state.task_thread = worker
    try:
        with caplog.at_level(logging.DEBUG, logger="kiss.server.task_runner"):
            worker.start()
            worker.join(timeout=30)
            assert not worker.is_alive(), "worker never finished"
            # Give a late (+6 s) injection no chance to be the thing
            # that ended the worker: the watchdog's second attempt is
            # only reached when the first one was refused.
            time.sleep(0.2)
    finally:
        with agent_state.STATE_LOCK:
            stale = [st.task_id for st in agent_state.snapshot() if st.tab_id == tab_id]
        for key in stale:
            agent_state.unregister(key)

    assert agent.task_id, "agent.run never allocated a row"
    assert "Cleanup interrupted" not in caplog.text
    _persistence._flush_chat_events()
    row = (
        _persistence._get_db()
        .execute(
            "SELECT result FROM task_history WHERE id = ?",
            (agent.task_id,),
        )
        .fetchone()
    )
    assert row is not None
    assert _RESULT_TEXT in str(row["result"]), row["result"]
    types = [e.get("type") for e in printer.events]
    assert types.count("task_done") == 1, types
    assert {"type": "status", "running": False, "tabId": tab_id} in printer.events
