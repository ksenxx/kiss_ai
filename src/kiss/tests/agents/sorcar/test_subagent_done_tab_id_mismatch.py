# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: ``subagentDone`` must broadcast the frontend tab id.

Bug
---
When ``run_parallel`` spawns sub-agents, the backend creates internal
state keyed by ``sub_tab_id = f"task-{parent_task_id}__sub_{idx}"``.
However the frontend tab is created by ``new_tab`` → ``createNewTab()``
with a randomly-generated tab id.  The ``openSubagentTab`` event
(broadcast from ``_replay_session``) correctly uses this
frontend-generated tab id.  But ``subagentDone`` (broadcast from
``_run_single``'s ``finally`` block) was using the backend's
``sub_tab_id``, which doesn't match the frontend tab id — so the
frontend handler ``tabs.find(t => t.id === ev.tab_id)`` returned
``undefined`` and ``isDone`` was never set, leaving the purple ◉
indicator permanently pulsing.

Fix
---
``_run_single``'s ``finally`` block now resolves the actual frontend
viewer tab ids from the printer's subscriber map (``_fanout_targets``)
and broadcasts ``subagentDone`` for each one, falling back to
``sub_tab_id`` for the ``_open_persisted_subagent_tabs`` path.
"""

from __future__ import annotations

import unittest
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.json_printer import JsonPrinter


class _CapturePrinter(JsonPrinter):
    """Printer that captures broadcast events."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record every broadcast event, with taskId injection."""
        event = self._inject_task_id(event)
        self.events.append(dict(event))
        with self._lock:
            self._record_event(event)


# A fixed fake task id the mock sub-agent will "allocate" during run().
_FAKE_SUB_TASK_ID = "99999"

# The frontend-generated tab id (simulates what createNewTab() produces).
_FRONTEND_TAB_ID = "frontend-random-abc123"


def _patched_run(self: ChatSorcarAgent, **kwargs: Any) -> str:
    """Mock for ``ChatSorcarAgent.run`` that simulates the sub-agent
    lifecycle: setting ``_last_task_id`` and subscribing the frontend
    viewer tab to the printer (as ``_reattach_running_chat`` would).
    """
    self._last_task_id = _FAKE_SUB_TASK_ID
    printer: Any = kwargs.get("printer") or self.printer
    if printer is not None and hasattr(printer, "subscribe_tab"):
        printer.subscribe_tab(_FAKE_SUB_TASK_ID, _FRONTEND_TAB_ID)
    return "success: true\nsummary: done"


class TestSubagentDoneTabIdMatchesViewerTab(unittest.TestCase):
    """The ``subagentDone`` event must carry the frontend tab ids
    that are subscribed to the sub-agent's event stream, not just
    the backend's internal ``sub_tab_id``."""

    def setUp(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def test_subagent_done_broadcasts_viewer_tab_ids(self) -> None:
        """When a frontend tab subscribes to a sub-agent's event stream
        via the printer's subscriber map, the ``subagentDone`` event
        must include that viewer tab id (not just the backend's
        internal ``sub_tab_id``).

        Simulates the full ``_run_tasks_parallel`` lifecycle:

        1. Parent agent creates a sub-agent and calls ``run()``.
        2. During ``run()``, ``_last_task_id`` is set and the
           frontend's randomly-generated tab id is subscribed to
           the sub-agent's event stream (via ``subscribe_tab``).
        3. After ``run()`` returns, the ``finally`` block broadcasts
           ``subagentDone``.
        4. The ``subagentDone`` event must carry the frontend tab id
           so the frontend handler can find the tab and stop the
           pulsing indicator.
        """
        printer = _CapturePrinter()
        agent = ChatSorcarAgent("test-parent")
        agent._chat_id = "test-chat-123"
        agent._last_task_id = "10000000000000000000000000000000"
        agent.printer = printer  # type: ignore[assignment]

        # Register the parent agent state so _run_tasks_parallel can
        # resolve parent_tab_id.
        parent_tab_id = "parent-tab-xyz"
        parent_state = _RunningAgentState(parent_tab_id, "test-model")
        parent_state.agent = agent  # type: ignore[assignment]
        _RunningAgentState.register(parent_tab_id, parent_state)

        # Patch run() on the ChatSorcarAgent CLASS so the sub-agent
        # (a fresh ChatSorcarAgent instance created by _run_single)
        # uses our mock that sets _last_task_id and subscribes the
        # frontend viewer tab.
        original_run = ChatSorcarAgent.run
        ChatSorcarAgent.run = _patched_run  # type: ignore[assignment]
        try:
            agent._run_tasks_parallel(["compute 1+1"], max_workers=1)
        finally:
            ChatSorcarAgent.run = original_run  # type: ignore[assignment]

        # Collect all subagentDone events
        done_events = [
            e for e in printer.events if e.get("type") == "subagentDone"
        ]

        assert len(done_events) >= 1, (
            f"Expected at least one subagentDone event, got {len(done_events)}. "
            f"All events: {[e.get('type') for e in printer.events]}"
        )

        # The subagentDone event(s) must include the frontend viewer
        # tab id.  Before the fix, only the backend's internal
        # sub_tab_id ("task-100__sub_0") was broadcast — this id does
        # NOT match the frontend tab's randomly-generated id, so the
        # frontend handler ``tabs.find(t => t.id === ev.tab_id)``
        # returned undefined and isDone was never set.
        done_tab_ids = {e.get("tab_id") for e in done_events}
        self.assertIn(
            _FRONTEND_TAB_ID,
            done_tab_ids,
            f"subagentDone must broadcast with the frontend viewer tab id "
            f"'{_FRONTEND_TAB_ID}', but only broadcast for tab_ids: "
            f"{done_tab_ids}. The frontend handler uses ev.tab_id to find "
            f"the tab and set isDone=true — if the id doesn't match, the "
            f"purple ◉ indicator keeps pulsing forever.",
        )

    def test_subagent_done_includes_backend_tab_id_as_fallback(self) -> None:
        """The ``subagentDone`` event must also include the backend's
        internal ``sub_tab_id`` as a fallback for the
        ``_open_persisted_subagent_tabs`` path where tab ids are
        deterministic (not randomly generated).
        """
        printer = _CapturePrinter()
        agent = ChatSorcarAgent("test-parent")
        agent._chat_id = "test-chat-456"
        agent._last_task_id = "20000000000000000000000000000000"
        agent.printer = printer  # type: ignore[assignment]

        parent_tab_id = "parent-tab-def"
        parent_state = _RunningAgentState(parent_tab_id, "test-model")
        parent_state.agent = agent  # type: ignore[assignment]
        _RunningAgentState.register(parent_tab_id, parent_state)

        original_run = ChatSorcarAgent.run
        ChatSorcarAgent.run = _patched_run  # type: ignore[assignment]
        try:
            agent._run_tasks_parallel(["compute 2+2"], max_workers=1)
        finally:
            ChatSorcarAgent.run = original_run  # type: ignore[assignment]

        done_events = [
            e for e in printer.events if e.get("type") == "subagentDone"
        ]

        done_tab_ids = {e.get("tab_id") for e in done_events}
        # The backend's sub_tab_id is "task-<parent_task_id>__sub_<idx>"
        backend_sub_tab_id = f"task-{agent._last_task_id}__sub_0"
        self.assertIn(
            backend_sub_tab_id,
            done_tab_ids,
            f"subagentDone must also broadcast with the backend's internal "
            f"sub_tab_id '{backend_sub_tab_id}' as a fallback for the "
            f"persisted sub-agent tab path. Got: {done_tab_ids}",
        )

    def test_subagent_done_without_subscribers_uses_backend_id(self) -> None:
        """When no frontend tab has subscribed to the sub-agent's event
        stream (e.g. the ``new_tab`` → ``resumeSession`` round-trip
        hasn't completed yet), the ``subagentDone`` event must still
        be broadcast with the backend's ``sub_tab_id``.
        """
        printer = _CapturePrinter()
        agent = ChatSorcarAgent("test-parent")
        agent._chat_id = "test-chat-789"
        agent._last_task_id = "30000000000000000000000000000000"
        agent.printer = printer  # type: ignore[assignment]

        parent_tab_id = "parent-tab-ghi"
        parent_state = _RunningAgentState(parent_tab_id, "test-model")
        parent_state.agent = agent  # type: ignore[assignment]
        _RunningAgentState.register(parent_tab_id, parent_state)

        def _run_no_subscribe(self_agent: ChatSorcarAgent, **kw: Any) -> str:
            """Mock run() that sets _last_task_id but does NOT subscribe
            any frontend tab.
            """
            self_agent._last_task_id = "77777"
            return "success: true\nsummary: done"

        original_run = ChatSorcarAgent.run
        ChatSorcarAgent.run = _run_no_subscribe  # type: ignore[assignment]
        try:
            agent._run_tasks_parallel(["compute 3+3"], max_workers=1)
        finally:
            ChatSorcarAgent.run = original_run  # type: ignore[assignment]

        done_events = [
            e for e in printer.events if e.get("type") == "subagentDone"
        ]

        assert len(done_events) >= 1, (
            f"Expected at least one subagentDone event. "
            f"All events: {[e.get('type') for e in printer.events]}"
        )

        done_tab_ids = {e.get("tab_id") for e in done_events}
        backend_sub_tab_id = f"task-{agent._last_task_id}__sub_0"
        self.assertIn(
            backend_sub_tab_id,
            done_tab_ids,
            f"When no frontend tab is subscribed, subagentDone must use "
            f"the backend's sub_tab_id '{backend_sub_tab_id}'. "
            f"Got: {done_tab_ids}",
        )


if __name__ == "__main__":
    unittest.main()
