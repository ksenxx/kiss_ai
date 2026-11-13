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

The acknowledgement takes ``STATE_LOCK``, the lock the watchdog injects
under, so an interrupt injected just before it lands on the worker as
it takes the lock: before the stop is acknowledged.  Raised out of the
post-run ``finally`` it skipped the whole cleanup (the row kept no
result and a "stopped" result followed the failure's); raised out of
``_run_task``'s setup-failure handler it skipped the failure ``result``.
The acknowledgement is therefore repeated until it has run.

The tests drive the REAL ``_run_task`` worker, the REAL stop watchdog
(or its injection step, ``inject_if_owned``) and the REAL SQLite
persistence.  The only substitution is the agent's LLM loop (``run``
returns or fails at once) and one hook per test that places the Stop:
``_flush_warnings``, the first call of the post-run cleanup, presses
Stop and dwells longer than the watchdog's grace period; a gate parks
the worker until the test holds ``STATE_LOCK``, so the worker's next
take of it, the acknowledgement, is where the interrupt lands.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
import time
from functools import partial
from typing import Any

import pytest

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.core.locked import locked_call
from kiss.core.models.model_info import get_available_models
from kiss.core.utils import finish
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer
from kiss.server.task_runner import _state_owns_thread, inject_if_owned

_RESULT_TEXT = "stop-after-loop finished"
_LOOP_FAILURE = "boom at the end of the loop"
_SETUP_FAILURE = "boom during setup"


class _CapturePrinter(JsonPrinter):
    """Real printer subclass that records every broadcast event."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*, then run the real record/persist path."""
        self.events.append(dict(event))
        super().broadcast(event)

    def result_texts(self) -> list[str]:
        """Return the text of every ``result`` event broadcast so far."""
        return [e["text"] for e in self.events if e.get("type") == "result"]


class _Gate:
    """Parks the worker once, until the test holds ``STATE_LOCK``."""

    def __init__(self) -> None:
        self.parked = threading.Event()
        self.resume = threading.Event()

    def park(self) -> None:
        """Block the calling worker until the test resumes it (first call only)."""
        if not self.parked.is_set():
            self.parked.set()
            self.resume.wait(30)


class _GatedPrinter(_CapturePrinter):
    """Capture printer that parks the worker right after the loop's failure ``result``."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = _Gate()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Broadcast *event*; park once it was the failed subtask's ``result``."""
        super().broadcast(event)
        if event.get("type") == "result" and _LOOP_FAILURE in str(event.get("text")):
            self.gate.park()


def _allocate_row(agent: ChatSorcarAgent, prompt: str) -> str:
    """Allocate *agent*'s ``task_history`` row as ``ChatSorcarAgent.run`` does."""
    task_id, agent._chat_id = _persistence._add_task(prompt, chat_id=agent._chat_id or "")
    with agent._task_id_lock:
        agent._last_task_id = task_id
    return task_id


class _StopDuringCleanupAgent(ChatSorcarAgent):
    """Agent that finishes at once and presses Stop from inside the cleanup."""

    server: AgentServer
    tab_id: str
    task_id: str = ""

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Allocate the task row and return a successful result."""
        self.task_id = _allocate_row(self, kwargs.get("prompt_template", ""))
        return finish(True, summary_in_html=f"<p>{_RESULT_TEXT}</p>")

    def _flush_warnings(self, printer: Any) -> None:
        """Press Stop, then dwell past the watchdog's 1 s grace period."""
        super()._flush_warnings(printer)
        self.server._stop_task(self.tab_id)
        time.sleep(1.6)


class _LoopFailureAgent(ChatSorcarAgent):
    """Agent whose run allocates its row and fails."""

    task_id: str = ""

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Allocate the task row, then raise."""
        self.task_id = _allocate_row(self, kwargs.get("prompt_template", ""))
        raise RuntimeError(_LOOP_FAILURE)


class _SetupFailureAgent(ChatSorcarAgent):
    """Agent whose pre-run classification parks at its gate, then fails the setup."""

    gate: _Gate

    def classify_task_for_run(self, *args: Any, **kwargs: Any) -> None:
        """Park, then raise before the run's agent loop starts."""
        self.gate.park()
        raise RuntimeError(_SETUP_FAILURE)


def _launch(
    tab_id: str, agent: ChatSorcarAgent, printer: _CapturePrinter,
) -> tuple[AgentServer, AgentState, threading.Thread]:
    """Register a run of *agent* on *tab_id* as ``_cmd_run`` does; the worker is not started."""
    models = get_available_models()
    if not models:
        pytest.skip("no models configured in this environment")
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    server = AgentServer(printer=printer)
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
                "prompt": "finish immediately",
                "workDir": tempfile.mkdtemp(prefix="stop-after-loop-"),
                "model": models[0],
                "autoCommit": False,
                "classifyTasks": False,
                "_state_key": state.task_id,
            },
        ),
        daemon=True,
    )
    state.task_thread = worker
    return server, state, worker


def _unregister_tab(tab_id: str) -> None:
    """Drop every state the test left registered for *tab_id*."""
    with agent_state.STATE_LOCK:
        stale = [st.task_id for st in agent_state.snapshot() if st.tab_id == tab_id]
    for key in stale:
        agent_state.unregister(key)


def _row_result(task_id: str) -> str:
    """Return the persisted ``result`` of the ``task_history`` row *task_id*."""
    _persistence._flush_chat_events()
    row = (
        _persistence._get_db()
        .execute("SELECT result FROM task_history WHERE id = ?", (task_id,))
        .fetchone()
    )
    assert row is not None
    return str(row["result"])


def _in_locked_call(thread: threading.Thread) -> bool:
    """True when *thread*'s innermost Python frame is ``locked_call``'s."""
    frame = sys._current_frames().get(thread.ident or 0)
    return frame is not None and frame.f_code is locked_call.__code__


def _wait_until_blocked_on_a_lock(thread: threading.Thread) -> None:
    """Return once *thread* waits in ``locked_call``: there on two samples 50 ms apart.

    A take of a free lock is over in microseconds, so only a thread
    waiting for a held lock is seen there twice.
    """
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if _in_locked_call(thread):
            time.sleep(0.05)
            if _in_locked_call(thread):
                return
        time.sleep(0.01)
    raise AssertionError("worker never waited for STATE_LOCK")


def _stop_at_acknowledgement(state: AgentState, worker: threading.Thread, gate: _Gate) -> None:
    """Run *worker* and land a Stop's interrupt on its next take of ``STATE_LOCK``.

    The worker parks at *gate*; holding ``STATE_LOCK``, the test sets the
    stop event (``_stop_task``'s first step), lets the worker run on
    until it waits for the lock, and injects through the watchdog's own
    guarded step.  Releasing the lock hands it to the worker together
    with the pending interrupt.
    """
    worker.start()
    assert gate.parked.wait(30), "worker never reached the gate"
    with agent_state.STATE_LOCK:
        assert state.stop_event is not None
        state.stop_event.set()
        gate.resume.set()
        _wait_until_blocked_on_a_lock(worker)
        assert inject_if_owned(worker, partial(_state_owns_thread, state, worker)) == 1
    worker.join(timeout=30)
    assert not worker.is_alive(), "worker never finished"


def test_stop_after_agent_loop_does_not_interrupt_cleanup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tab_id = "stop-after-loop-tab"
    printer = _CapturePrinter()
    agent = _StopDuringCleanupAgent("Sorcar")
    agent.server, _state, worker = _launch(tab_id, agent, printer)
    agent.tab_id = tab_id
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
        _unregister_tab(tab_id)

    assert agent.task_id, "agent.run never allocated a row"
    assert "Cleanup interrupted" not in caplog.text
    assert _RESULT_TEXT in _row_result(agent.task_id)
    types = [e.get("type") for e in printer.events]
    assert types.count("task_done") == 1, types
    assert {"type": "status", "running": False, "tabId": tab_id} in printer.events


def test_interrupt_in_the_loop_end_acknowledgement_keeps_the_cleanup() -> None:
    """The failed subtask's row and terminal event survive an interrupt landing in the ack."""
    tab_id = "stop-at-loop-ack-tab"
    printer = _GatedPrinter()
    agent = _LoopFailureAgent("Sorcar")
    _server, state, worker = _launch(tab_id, agent, printer)
    try:
        _stop_at_acknowledgement(state, worker, printer.gate)
    finally:
        _unregister_tab(tab_id)

    assert printer.result_texts() == [f"Task failed: {_LOOP_FAILURE}"]
    assert _LOOP_FAILURE in _row_result(agent.task_id)
    types = [e.get("type") for e in printer.events]
    assert types.count("task_error") == 1, types
    assert {"type": "status", "running": False, "tabId": tab_id} in printer.events


def test_interrupt_in_the_setup_failure_acknowledgement_keeps_the_result() -> None:
    """A setup failure is still reported when an interrupt lands in its acknowledgement."""
    tab_id = "stop-at-setup-ack-tab"
    printer = _CapturePrinter()
    agent = _SetupFailureAgent("Sorcar")
    agent.gate = _Gate()
    _server, state, worker = _launch(tab_id, agent, printer)
    try:
        _stop_at_acknowledgement(state, worker, agent.gate)
    finally:
        _unregister_tab(tab_id)

    assert printer.result_texts() == [f"Task failed: RuntimeError: {_SETUP_FAILURE}"]
    assert {"type": "status", "running": False, "tabId": tab_id} in printer.events
