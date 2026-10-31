# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a ``daemon_client.run`` that stops waiting leaves a running task's tab open.

``run()`` used to send ``closeTab`` on every exit, so a plain timeout
removed the still-running task's ``api-…`` tab from the shared registry,
and so from every surface.  Over a real daemon (only the LLM boundary
stubbed, see :class:`DaemonLocalHarness`):

* a timed-out top-level run keeps its registry tab while the task runs
  and after it ends, until a user closes it;
* a finished top-level run still closes its tab;
* a timed-out sub-agent run (whose ``api-…`` tab no surface shows)
  still closes it, so the daemon disposes of its state at task end.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from kiss.agents.sorcar import daemon_client
from kiss.server import agent_state
from kiss.tests.server.test_run_agent_subagent_tab import PARENT_TAB_ID, DaemonLocalHarness


class RunTabCloseTest(DaemonLocalHarness):
    """When the client's ``closeTab`` reaches the daemon, seen from the registry."""

    def _api_tabs(self) -> list[str]:
        """The ``api-…`` tab ids the shared registry lists."""
        return [
            str(entry.get("tabId", ""))
            for entry in self.server._agent_server.tab_registry.snapshot()
            if str(entry.get("tabId", "")).startswith("api-")
        ]

    def _api_states(self) -> list[agent_state.AgentState]:
        """The daemon's states of ``api-…`` tabs."""
        return [s for s in agent_state.snapshot() if s.tab_id.startswith("api-")]

    def _run_until_timeout(self, marker: str, release: threading.Event, **kwargs: Any) -> None:
        """Run *marker* with a short timeout while its task blocks on *release*."""
        reached: dict[str, Any] = {}
        self._install_stub(reached, release, marker)
        with self.assertRaises(TimeoutError):
            daemon_client.run(
                marker, work_dir=self.repo, endpoint_file=self.endpoint_file,
                timeout=3.0, **kwargs,
            )
        self._wait_for(lambda: "tabs" in reached, what="the task to reach the stub")
        # The client has left; a closeTab it sent is handled by now.
        self._wait_for(
            lambda: len(self.server._printer._local_clients) == 1, what="the client to leave",
        )
        time.sleep(0.3)

    def test_timed_out_top_level_run_keeps_its_tab(self) -> None:
        """The task outlives the client's wait, and so does its tab."""
        self._open_viewer()
        release = threading.Event()
        try:
            self._run_until_timeout("slow top-level task w3a", release)
            [tab_id] = self._api_tabs()
            state = agent_state.find_by_tab(tab_id)
            assert state is not None and state.busy() and not state.frontend_closed
        finally:
            release.set()
        self._wait_for(
            lambda: not (state.busy() if (state := agent_state.find_by_tab(tab_id)) else True),
            what="the task to end",
        )
        assert self._api_tabs() == [tab_id]
        self._send_from_viewer({"type": "closeTab", "tabId": tab_id})
        self._wait_for(lambda: not self._api_tabs(), what="the user's close")

    def test_finished_top_level_run_closes_its_tab(self) -> None:
        """A run that returned closes its tab on every surface."""
        release = threading.Event()
        release.set()
        self._install_stub({}, release, "quick top-level task w3b")
        self._open_viewer()
        result = daemon_client.run(
            "quick top-level task w3b", work_dir=self.repo,
            endpoint_file=self.endpoint_file, timeout=60,
        )
        assert result.success is True
        self._wait_for(lambda: not self._api_tabs(), what="the tab to close")

    def test_timed_out_subagent_run_is_disposed_at_task_end(self) -> None:
        """A sub-agent run's hidden tab is closed, so its state does not outlive the task."""
        release = threading.Event()
        release.set()
        self._install_stub({}, release, "unused marker")
        self._open_viewer()
        parent = daemon_client.run(
            "parent seed task w3c", work_dir=self.repo,
            endpoint_file=self.endpoint_file, timeout=60,
        )
        assert parent.task_id
        release = threading.Event()
        try:
            self._run_until_timeout(
                "slow sub-agent task w3c", release,
                parent_task_id=parent.task_id, parent_tab_id=PARENT_TAB_ID,
            )
            assert self._api_tabs() == []
            [state] = self._api_states()
            assert state.busy() and state.frontend_closed
        finally:
            release.set()
        self._wait_for(lambda: not self._api_states(), what="the state's disposal")
