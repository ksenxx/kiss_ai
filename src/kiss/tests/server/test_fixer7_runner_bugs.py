# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for fixer-7 findings (round 6).

Covers:

- F1: ``_cmd_run`` wedged a tab forever when the ``clear`` broadcast or
  ``thread.start()`` raised after ``tab.task_thread`` was assigned.
- F3: ``_resolve_user_answer_queue`` lacked the task-ownership filter
  its two task_runner siblings have, letting a stale viewer answer be
  hijacked by an unrelated running task.
- F16: a ``KeyboardInterrupt`` (Stop during pre-task setup) escaped
  ``_run_task``'s ``except Exception`` so the task vanished silently
  with no ``result`` event.

No mocks/patches/fakes: real :class:`JsonPrinter` subclasses (the same
technique the existing server regression tests use) capture or steer
broadcasts, real :class:`AgentServer` / :class:`AgentState` /
:class:`ChatSorcarAgent` objects are exercised.
"""

from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer


class _CapturePrinter(JsonPrinter):
    """Real printer subclass that records every broadcast event."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event* instead of writing it to stdout."""
        self.events.append(event)


def _clear_states() -> None:
    """Remove every entry from the global task-state registry."""
    with agent_state.STATE_LOCK:
        agent_state.agent_states.clear()



class _FailingClearPrinter(_CapturePrinter):
    """Transport whose ``clear`` broadcast always fails (subclass error)."""

    def broadcast(self, event: dict[str, Any]) -> None:
        """Raise for ``clear`` events; record everything else."""
        if event.get("type") == "clear":
            raise RuntimeError("transport send failed")
        super().broadcast(event)


def test_f1_failed_clear_broadcast_does_not_wedge_tab() -> None:
    printer = _FailingClearPrinter()
    server = AgentServer(printer=printer)
    tab_id = "f1-tab"
    cmd = {"type": "run", "tabId": tab_id, "prompt": "do something"}
    try:
        with pytest.raises(RuntimeError):
            server._cmd_run(dict(cmd))
        tab = agent_state.find_by_tab(tab_id)
        assert tab is not None
        assert tab.task_thread is None
        assert tab.stop_event is None
        assert tab.user_answer_queue is None
        with pytest.raises(RuntimeError):
            server._cmd_run(dict(cmd))
        tab = agent_state.find_by_tab(tab_id)
        assert tab is not None
        assert tab.task_thread is None
        assert tab.pending_user_messages == []
        assert all(e.get("type") != "prompt" for e in printer.events)
    finally:
        _clear_states()



def test_f3_stale_viewer_answer_not_hijacked_by_unrelated_task() -> None:
    printer = _CapturePrinter()
    server = AgentServer(printer=printer)
    viewer, owner = "f3-viewer", "f3-owner"
    try:
        printer.subscribe_tab(100, viewer)
        printer.subscribe_tab(100, owner)
        owner_state = AgentState("200", tab_id=owner, server_owned=True)
        owner_state.user_answer_queue = queue.Queue(maxsize=1)
        owner_state.is_task_active = True
        owner_state.agent = ChatSorcarAgent("Sorcar")
        owner_state.agent._last_task_id = "200"
        agent_state.register(owner_state)
        printer.subscribe_tab(200, owner)

        server._cmd_user_answer({"tabId": viewer, "answer": "stale"})
        assert owner_state.user_answer_queue.empty()
        assert all(e.get("type") != "askUserDone" for e in printer.events)

        owner_state.agent._last_task_id = "100"
        agent_state.rekey(owner_state, "100")
        server._cmd_user_answer({"tabId": viewer, "answer": "real answer"})
        assert owner_state.user_answer_queue.get_nowait() == "real answer"
    finally:
        _clear_states()



class _StopDuringSetupPrinter(_CapturePrinter):
    """Injects a real ``KeyboardInterrupt`` into the task setup path.

    Raises on the first ``status running:true`` broadcast — i.e. inside
    ``_run_task``'s ``try`` before ``_run_task_inner`` runs, the same
    place a Stop-injected ``KeyboardInterrupt`` lands during pre-task
    setup on a large repo.
    """

    def __init__(self) -> None:
        super().__init__()
        self._raised = False

    def broadcast(self, event: dict[str, Any]) -> None:
        """Raise KeyboardInterrupt once on the first status broadcast."""
        if (
            not self._raised
            and event.get("type") == "status"
            and event.get("running") is True
        ):
            self._raised = True
            raise KeyboardInterrupt("stop clicked during setup")
        super().broadcast(event)


def test_f16_keyboard_interrupt_during_setup_broadcasts_result(
    tmp_path: Path,
) -> None:
    printer = _StopDuringSetupPrinter()
    server = AgentServer(printer=printer)
    tab_id = "f16-tab"
    try:
        server._run_task({
            "tabId": tab_id,
            "prompt": "x",
            "workDir": str(tmp_path),
        })
        results = [e for e in printer.events if e.get("type") == "result"]
        assert len(results) == 1
        assert results[0]["success"] is False
        assert results[0]["tabId"] == tab_id
        assert "stopped" in results[0]["text"].lower()
        assert any(
            e.get("type") == "status" and e.get("running") is False
            for e in printer.events
        )
    finally:
        _clear_states()
