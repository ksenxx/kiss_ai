# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: a dispatched child whose prompt is ``/xxx help`` keeps its parentage.

``/xxx help`` never runs the SEA: the task runner answers from the
script's ``description()`` on the model-free help path and persists
the row itself (``_finish_sea_help_task``), bypassing
``ChatSorcarAgent.run``'s persistence.  That path built the row's
``extra`` without the agent's ``subagent`` marker, so a child
dispatched with ``parentTaskId`` into its parent's chat was stored
with an empty ``parent_task_id``: it passed the root-chat filter of
the chat readers, replayed as a turn of the parent's chat, and
``resumeSession``'s latest-row reader selected its help transcript
instead of the parent's task.

Real pieces throughout: the production ``AgentServer._cmd_run`` with a
registered parent state, the real sqlite history and the production
chat readers (``_load_all_chat_events_by_chat_id``,
``_load_latest_chat_events_by_chat_id``).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _load_all_chat_events_by_chat_id,
    _load_latest_chat_events_by_chat_id,
)
from kiss.server import agent_state
from kiss.server.server import AgentServer

DEADLINE_S = 20.0


def _wait_until(predicate: Any, timeout: float) -> None:
    """Block until *predicate* is true (polling), or fail after *timeout*."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


def test_help_only_child_is_persisted_under_its_parent(tmp_path: Path) -> None:
    """The help child's row carries the parent's id; root readers leave it out."""
    server = AgentServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    parent_id, chat_id = _add_task(
        "Parent for help child", chat_id="",
        extra={"work_dir": str(tmp_path), "startTs": int(time.time() * 1000)},
    )
    parent = ChatSorcarAgent("help-child-parent")
    parent.resume_chat_by_id(chat_id)
    with parent._task_id_lock:
        parent._last_task_id = parent_id
    parent_state = agent_state.AgentState(
        parent_id, agent=parent, tab_id="help-child-parent-tab", server_owned=True,
        chat_id=chat_id, stop_event=threading.Event(), is_task_active=True,
    )
    agent_state.register(parent_state)
    child_tab = "help-child-tab"
    try:
        server._cmd_run({
            "type": "run", "tabId": child_tab, "taskId": "help-client-token",
            "prompt": "/sh help", "chatId": chat_id, "workDir": str(tmp_path),
            "parentTaskId": parent_id, "parentTabId": parent_state.tab_id,
            "useWorktree": False, "autoCommit": False, "useMemory": False,
        })

        def child_finished() -> bool:
            with lock:
                return any(
                    e.get("type") == "status" and e.get("running") is False
                    and e.get("tabId") == child_tab
                    for e in events
                )

        _wait_until(child_finished, DEADLINE_S)
        with sqlite3.connect(persistence._DB_PATH) as db:
            rows = db.execute(
                "SELECT id, parent_task_id, task, result FROM task_history "
                "WHERE chat_id = ? AND task = ?", (chat_id, "/sh help"),
            ).fetchall()
            assert len(rows) == 1, rows
            help_id, recorded_parent, _prompt, result = rows[0]
            roots = [
                r[0] for r in db.execute(
                    "SELECT task FROM task_history WHERE chat_id = ? "
                    "AND parent_task_id = '' ORDER BY rowid", (chat_id,),
                ).fetchall()
            ]
        assert result, "the help answer was not persisted"
        with lock:
            results = [
                e for e in events
                if e.get("type") == "result" and e.get("tabId") == child_tab
            ]
        assert results and results[0]["success"] is True
        # The child's row names its parent, like a child run by the model does.
        assert recorded_parent == parent_id
        # The parent's chat has one root turn: the parent's.
        assert roots == ["Parent for help child"]
        tasks, _truncated = _load_all_chat_events_by_chat_id(chat_id)
        assert [t["task_id"] for t in tasks] == [parent_id]
        latest = _load_latest_chat_events_by_chat_id(chat_id)
        assert latest is not None
        assert latest["task_id"] == parent_id
        assert latest["task"] == "Parent for help child"
        assert help_id != parent_id
    finally:
        agent_state.unregister(parent_id, parent_state)
        sweep = server._orphan_sweep_thread
        if sweep is not None:
            sweep.join(timeout=10)
