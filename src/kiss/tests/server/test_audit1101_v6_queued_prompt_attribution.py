# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: a prompt queued while a new run is admitted is not filed under the previous task.

A finished task keeps its ``ChatSorcarAgent`` on the tab's state (it
may hold a pending worktree), and ``_cmd_run`` carries that agent
into the next run's fresh state.  Until the worker allocates the new
``task_history`` row, ``agent.last_task_id`` still names the PREVIOUS
task.  ``_queue_prompt_on_owner`` stamped the echo of a prompt queued
in that window with that id, so the prompt — queued on the NEW run —
was recorded and persisted as a turn of the OLD task, and because the
id was nonempty the deferred-attribution path
(``unattributed_prompt_echoes``) never repaired it.

The queuing code now trusts the agent's id only when it is the
registered owner's own (the state is re-keyed to its row on
allocation); otherwise the echo is deferred to the run that consumes
the prompt.

Real pieces throughout: the production ``_cmd_run`` admission (a
nonexistent agent script makes the worker fail at setup without a
model call), the real registry, ``_cmd_append_user_message`` and the
real sqlite event store.  A profile hook holds the worker thread at
``_run_task``'s entry so the admission window is open for the duration
of the test; it replaces no code.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _add_task
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import AgentServer

DEADLINE_S = 15.0


class _EntryGate:
    """A profile hook that parks the thread entering *code* until released."""

    def __init__(self, code: Any) -> None:
        self.code = code
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, frame: Any, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is self.code:
            self.entered.set()
            assert self.release.wait(DEADLINE_S), "gate timed out"
            sys.setprofile(None)


def _admit_gated(server: AgentServer, cmd: dict[str, Any]) -> tuple[_EntryGate, AgentState]:
    """Admit *cmd* through ``_cmd_run`` with its worker parked at ``_run_task``'s entry."""
    gate = _EntryGate(server._run_task.__func__.__code__)  # type: ignore[attr-defined]
    threading.setprofile(gate)
    try:
        server._cmd_run(cmd)
        assert gate.entered.wait(DEADLINE_S), "the worker never reached _run_task"
    finally:
        threading.setprofile(None)
    state = agent_state.agent_states[cmd["_state_key"]]
    assert state.task_thread is not None and state.task_thread.is_alive()
    return gate, state


def _finish_gated(gate: _EntryGate, state: AgentState) -> None:
    """Stop the parked run and let its worker finish (no redispatch of queued prompts)."""
    if state.stop_event is not None:
        state.stop_event.set()
    thread = state.task_thread
    gate.release.set()
    if thread is not None:
        thread.join(DEADLINE_S)
        assert not thread.is_alive(), "the worker did not finish"


def _persisted_prompt_texts(task_id: str) -> list[str]:
    """The ``prompt`` events persisted under *task_id*."""
    persistence._flush_chat_events(task_id)
    rows = persistence._get_db().execute(
        "SELECT event_json FROM events WHERE task_id = ? ORDER BY seq", (task_id,),
    ).fetchall()
    events = [json.loads(row[0]) for row in rows]
    return [e["text"] for e in events if e.get("type") == "prompt"]


def test_prompt_queued_during_admission_is_not_filed_under_the_previous_task(
    tmp_path: Path,
) -> None:
    """The echo carries no stale task id and nothing lands in the old task's events."""
    server = AgentServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    old_id, chat_id = _add_task("previous task", chat_id="")
    agent = ChatSorcarAgent("carried-agent")
    with agent._task_id_lock:
        agent._last_task_id = old_id
    agent.work_dir = str(tmp_path)
    # The finished task's state, retained on the tab with its agent.
    old = AgentState(old_id, chat_id=chat_id, tab_id="carry-tab", agent=agent, server_owned=True)
    agent_state.register(old)
    cmd = {
        "type": "run", "tabId": "carry-tab", "chatId": chat_id, "prompt": "new task",
        "workDir": str(tmp_path), "seaPath": str(tmp_path / "does-not-exist.py"),
    }
    gate, state = _admit_gated(server, cmd)
    try:
        assert state.agent is agent, "the previous task's agent is carried into the new run"
        assert state.task_id != old_id

        server._cmd_append_user_message({"tabId": "carry-tab", "prompt": "steer NEW task"})

        assert state.pending_user_messages == ["steer NEW task"]
        with lock:
            echoes = [e for e in events if e.get("type") == "prompt"]
        assert len(echoes) == 1
        assert echoes[0]["text"] == "steer NEW task"
        assert echoes[0].get("taskId", "") != old_id, "the echo was filed under the previous task"
        # Deferred to the run that consumes the prompt, not stamped stale.
        assert state.unattributed_prompt_echoes == ["steer NEW task"]
        assert _persisted_prompt_texts(old_id) == []
    finally:
        _finish_gated(gate, state)
        for s in agent_state.snapshot():
            agent_state.unregister(s.task_id, s)
