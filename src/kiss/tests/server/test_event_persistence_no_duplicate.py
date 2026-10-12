# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: each broadcast event is persisted EXACTLY ONCE.

Before this fix, ``JsonPrinter._persist_event`` called BOTH
``_append_chat_event`` (synchronous) AND ``_queue_chat_event`` (async)
for every display event, so every event was stored twice in the
``events`` table.  In the UI this manifested as every panel — most
visibly the sub-agents' ``result`` panels in their own sub-agent tabs
— rendering twice when a task or chat session was reopened from
history.
"""

from __future__ import annotations

import shutil
import tempfile
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _flush_chat_events,
    _load_chat_events_by_task_id,
)
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.tests.server.conftest import redirect_persistence, restore_persistence


class TestEventPersistenceNoDuplicate:
    """``_persist_event`` writes each event exactly once to the DB."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_broadcast_persists_each_event_exactly_once(self) -> None:
        printer = JsonPrinter()

        agent = ChatSorcarAgent("agent")
        task_id, _ = _add_task("hello task")
        agent._last_task_id = task_id
        printer._thread_local.task_id = str(task_id)
        printer.agent_task_allocated(agent, task_id)

        broadcast_events: list[dict[str, Any]] = [
            {"type": "text_delta", "text": "alpha"},
            {"type": "tool_call", "name": "Bash"},
            {"type": "tool_result", "content": "ok", "is_error": False},
            {"type": "result", "text": "done"},
        ]
        for ev in broadcast_events:
            printer.broadcast(dict(ev))

        _flush_chat_events()

        loaded = _load_chat_events_by_task_id(task_id)
        assert loaded is not None
        events = loaded.get("events")
        assert isinstance(events, list)

        type_counts: dict[str, int] = {}
        for ev in events:
            type_counts[str(ev.get("type"))] = (
                type_counts.get(str(ev.get("type")), 0) + 1
            )
        assert type_counts.get("text_delta") == 1, type_counts
        assert type_counts.get("tool_call") == 1, type_counts
        assert type_counts.get("tool_result") == 1, type_counts
        assert type_counts.get("result") == 1, type_counts
        assert len(events) == len(broadcast_events), events
