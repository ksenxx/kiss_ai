# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a direct ``_run_task`` on a tab with an idle state gets its own key.

Audit finding C3-F2.  ``_cmd_run`` registers every run under a fresh
key, but ``_resolve_run_state_locked`` — the path of callers that
invoke ``_run_task`` directly — used to reuse the tab's idle state
*and its key*, the previous run's ``task_history`` row id.  A run that
failed during setup then recorded and persisted its failure ``result``
under that previous row (the finished task's replay gains an error it
never had) and armed its 300 s ``cleanup_task`` timer against it.

Driven over a real daemon: a wire ``run`` leaves the tab's state idle,
keyed by the finished run's row id; a direct ``_run_task`` with an
unknown ``seaCommand`` fails during setup.  Only the LLM boundary is
stubbed.
"""

from __future__ import annotations

import threading

from kiss.agents.sorcar import persistence as _persistence
from kiss.server import agent_state
from kiss.tests.server.test_run_agent_subagent_tab import DaemonLocalHarness


class RunStateReuseKeyTest(DaemonLocalHarness):
    """A reused idle state is re-keyed for the new run."""

    def test_setup_failure_is_not_filed_under_the_previous_run(self) -> None:
        """The failed run's ``result`` names its own key, not the finished run's row."""
        released = threading.Event()
        released.set()
        self._install_stub({}, released, "no child in this test")
        received = self._open_viewer()
        tab_id = "webtab-reuse-1"

        self._send_from_viewer({"type": "run", "prompt": "first task", "tabId": tab_id})
        self._wait_for(
            lambda: any(
                e.get("type") == "result" and e.get("tabId") == tab_id for e in list(received)
            ),
            what="the first run's result",
        )
        # The finished run leaves the tab's state idle, keyed by its
        # persisted ``task_history`` row id.
        idle = self._wait_for(
            lambda: next(
                (
                    s
                    for s in [agent_state.find_by_tab(tab_id)]
                    if s is not None and s.task_thread is None
                ),
                None,
            ),
            what="the first run's state to go idle",
        )
        first_row_id = idle.task_id
        _persistence._flush_chat_events(first_row_id)
        assert _persistence._load_chat_events_by_task_id(first_row_id) is not None, first_row_id

        before = len(received)
        self.server._agent_server._run_task(
            {
                "tabId": tab_id,
                "prompt": "second task",
                "seaCommand": "no_such_command_audit1020",
            }
        )
        failure = self._wait_for(
            lambda: next(
                (
                    e
                    for e in list(received)[before:]
                    if e.get("type") == "result" and e.get("tabId") == tab_id
                ),
                None,
            ),
            timeout=5.0,
            what="the failed run's result to reach the tab",
        )
        assert "not a registered command" in str(failure.get("text", "")), failure
        assert failure.get("taskId") != first_row_id, (
            "the failed run recorded its result under the previous run's history row"
        )
        state = agent_state.find_by_tab(tab_id)
        assert state is not None
        assert state.task_id == failure.get("taskId")
        assert agent_state.get(first_row_id) is None
        _persistence._flush_chat_events(first_row_id)
        first_row = _persistence._load_chat_events_by_task_id(first_row_id)
        assert first_row is not None
        first_events = first_row["events"]
        assert isinstance(first_events, list)
        assert all(
            "not a registered command" not in str(e.get("text", "")) for e in first_events
        ), first_events
