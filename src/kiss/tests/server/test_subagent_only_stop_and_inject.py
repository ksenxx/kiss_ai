# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end backend wiring for the "interact with a RUNNING sub-agent"
feature: the input textbox shown on a running sub-agent's chat tab must
be able to

* STOP ONLY that sub-agent's task (the parent keeps running), and
* INJECT follow-up prompts into that sub-agent's live conversation.

Wiring under test (all production code, no mocks of it): the server
resolves a frontend viewer tab (subscribed to the sub-agent's task
stream) to the sub-agent's task-keyed ``agent_state`` entry, which
makes both ``_stop_task(viewer_tab)`` and ``_cmd_append_user_message``
reach ONLY that sub-agent.  A sub-agent is a daemon sub-task with its
own ``stop_event``, so stopping it leaves the parent's event untouched.
"""

from __future__ import annotations

import threading
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _clear_registry() -> None:
    with agent_state.STATE_LOCK:
        agent_state.agent_states.clear()


class TestViewerTabResolvesToSubagent:
    """A frontend viewer tab subscribed to a sub-agent's task stream
    must resolve to the sub-agent's task-keyed agent state for both
    Stop and prompt injection."""

    def setup_method(self) -> None:
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()

    def _make_server(self) -> tuple[VSCodeServer, list[dict[str, Any]]]:
        server = VSCodeServer()
        events: list[dict[str, Any]] = []
        lock = threading.Lock()

        def capture(event: dict[str, Any]) -> None:
            with lock:
                events.append(event)

        server.printer.broadcast = capture  # type: ignore[assignment]
        return server, events

    def _register_subagent(
        self, server: VSCodeServer, *, task_id: str, viewer_tab: str,
    ) -> tuple[agent_state.AgentState, agent_state.AgentState]:
        """Register a live parent task and a live sub-agent state under
        its task id, and subscribe *viewer_tab* to the sub-agent's task
        stream (as the server does when a frontend tab opens the
        sub-agent's transcript).  Returns ``(parent, sub)``."""
        parent = agent_state.AgentState(
            "parent",
            agent=ChatSorcarAgent("parent"),  # type: ignore[arg-type]
            chat_id="chat-1",
            is_task_active=True,
            stop_event=threading.Event(),
        )
        agent_state.register(parent)
        agent = ChatSorcarAgent("sub")
        agent._last_task_id = task_id
        state = agent_state.AgentState(
            task_id,
            agent=agent,  # type: ignore[arg-type]
            chat_id="chat-1",
            parent_task_id="parent",
            is_task_active=True,
            stop_event=threading.Event(),
        )
        agent_state.register(state)
        server.printer.subscribe_tab(task_id, viewer_tab)
        return parent, state

    def test_stop_on_viewer_tab_sets_only_subagent_event(self) -> None:
        server, _events = self._make_server()
        parent, state = self._register_subagent(
            server, task_id="77", viewer_tab="viewer-tab",
        )
        assert state.is_subagent, (
            "a state with parent_task_id set must report is_subagent"
        )
        assert state.stop_event is not None
        server._stop_task("viewer-tab")
        assert state.stop_event.is_set(), (
            "Stop on the sub-agent's chat tab must set the "
            "sub-agent's own stop event"
        )
        assert parent.stop_event is not None and not parent.stop_event.is_set(), (
            "stopping the sub-agent must not stop the parent task"
        )

    def test_append_user_message_routes_to_subagent_queue(self) -> None:
        server, events = self._make_server()
        parent, state = self._register_subagent(
            server, task_id="77", viewer_tab="viewer-tab",
        )
        server._cmd_append_user_message(
            {"tabId": "viewer-tab", "prompt": "add more tests"},
        )
        assert state.pending_user_messages == ["add more tests"], (
            "a prompt typed on the sub-agent's chat tab must land in "
            "the SUB-AGENT's pending_user_messages queue; got "
            f"{state.pending_user_messages!r}"
        )
        assert parent.pending_user_messages == [], (
            "the prompt must not be queued into the parent task"
        )
        echo = [
            e for e in events
            if e.get("type") == "prompt" and e.get("tabId") == "viewer-tab"
        ]
        assert echo and echo[0].get("text") == "add more tests", (
            "the queued prompt must be echoed back on the viewer tab"
        )
