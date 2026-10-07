# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for subagent tabs created by run_parallel.

``run_parallel`` is N ``run_agent`` calls: every sub-task is submitted
to the kiss-web daemon (:mod:`kiss.agents.sorcar.agent_dispatch`), so
these tests run a real daemon on a private loopback endpoint
(:class:`DaemonLocalHarness`) and watch it through a webview-like
viewer connection.  They verify that, for a task calling
``run_parallel``:

- a ``new_tab`` event names each sub-task and the calling tab;
- the viewer tab opened for a sub-agent receives its streaming events;
- a ``subagentDone`` closes each sub-agent tab when its run ends.

Uses real LLM calls with claude-haiku-4-5 and tight budgets.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

from kiss.agents.sorcar import cron_agent
from kiss.tests.server.test_run_agent_subagent_tab import DaemonLocalHarness

FAST_MODEL = "claude-haiku-4-5"
PARENT_TAB_ID = "webtab-subagent-tabs-1"

skip_no_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@skip_no_key
class TestSubagentTabEvents(DaemonLocalHarness):
    """Verify that run_parallel broadcasts subagent tab events."""

    def setUp(self) -> None:
        super().setUp()
        # Inside the daemon, run_parallel submits its sub-tasks back
        # through the daemon's own local endpoint, which the scheduler
        # records at boot.
        self._saved_daemon_endpoint = cron_agent._daemon_endpoint_file
        cron_agent._daemon_endpoint_file = str(self.endpoint_file)

    def tearDown(self) -> None:
        cron_agent._daemon_endpoint_file = self._saved_daemon_endpoint
        super().tearDown()

    def _run_until_done(
        self, prompt: str, received: list[dict[str, Any]], timeout: float,
    ) -> list[dict[str, Any]]:
        """Run *prompt* on the viewer's tab and return the events seen.

        While the run is in flight, the viewer subscribes to every
        sub-agent tab the daemon announces (what media/main.js does on
        ``new_tab``), so the sub-agents' streams reach it.
        """
        self._send_from_viewer({
            "type": "run",
            "tabId": PARENT_TAB_ID,
            "prompt": prompt,
            "workDir": self.repo,
            "model": FAST_MODEL,
            "useWorktree": False,
            "autoCommit": False,
            "isParallel": True,
            "useWebTools": False,
            "maxBudget": 2.0,
        })
        subscribed: set[str] = set()
        deadline = time.monotonic() + timeout
        started = False
        while time.monotonic() < deadline:
            events = list(received)
            for ev in events:
                if ev.get("type") != "new_tab":
                    continue
                sub_task_id = str(ev.get("task_id") or "")
                if sub_task_id and sub_task_id not in subscribed:
                    subscribed.add(sub_task_id)
                    self._send_from_viewer({
                        "type": "resumeSession",
                        "taskId": sub_task_id,
                        "tabId": f"{PARENT_TAB_ID}__sub_{sub_task_id}",
                    })
            for ev in events:
                if ev.get("type") != "status" or ev.get("tabId") != PARENT_TAB_ID:
                    continue
                if ev.get("running"):
                    started = True
                elif started:
                    return list(received)
            time.sleep(0.05)
        raise AssertionError(
            f"the run did not finish within {timeout}s; "
            f"event types: {[e.get('type') for e in received]}"
        )

    @pytest.mark.slow
    def test_parallel_creates_subagent_tab_events(self) -> None:
        """Running a task with run_parallel creates new_tab events."""
        received = self._open_viewer()
        events = self._run_until_done(
            "Call run_parallel with these two tasks and nothing else: "
            "['Reply with the word ALPHA', 'Reply with the word BETA']. "
            "Then finish with the combined results.",
            received,
            timeout=300,
        )

        open_events = [e for e in events if e.get("type") == "new_tab"]
        assert len(open_events) >= 2, (
            f"Expected at least 2 new_tab events, got {len(open_events)}. "
            f"Event types: {[e.get('type') for e in events]}"
        )
        sub_task_ids: set[str] = set()
        for ev in open_events:
            assert ev.get("task_id"), f"Missing task_id in new_tab: {ev}"
            assert ev.get("parent_tab_id") == PARENT_TAB_ID, (
                f"new_tab must name the calling tab: {ev}"
            )
            sub_task_ids.add(ev["task_id"])
        assert len(sub_task_ids) == len(open_events), (
            "Sub-agent task IDs must be unique"
        )

        # One subagentDone per viewer sub-agent tab: the signal the
        # webview closes the tab on.
        viewer_sub_tabs = {f"{PARENT_TAB_ID}__sub_{t}" for t in sub_task_ids}
        self._wait_for(
            lambda: viewer_sub_tabs <= {
                str(e.get("tab_id"))
                for e in list(received)
                if e.get("type") == "subagentDone"
            },
            timeout=30,
            what=f"subagentDone for every sub-agent tab {sorted(viewer_sub_tabs)}",
        )

        # Streaming events of the sub-agents are stamped with their own
        # task id, never the parent's.
        sub_events = [
            e for e in list(received)
            if e.get("taskId") in sub_task_ids
            and e.get("type") not in ("new_tab", "subagentDone")
        ]
        assert sub_events, (
            "Expected streaming events stamped with a sub-agent task id"
        )

    @pytest.mark.slow
    def test_parallel_subagent_events_have_correct_types(self) -> None:
        """Sub-agent events include standard streaming types."""
        received = self._open_viewer()
        events = self._run_until_done(
            "Use the run_parallel tool to run one task: "
            "'Read this message and reply with DONE'. "
            "Then finish.",
            received,
            timeout=300,
        )

        open_events = [e for e in events if e.get("type") == "new_tab"]
        assert len(open_events) >= 1, (
            f"Expected a new_tab event, got types: "
            f"{[e.get('type') for e in events]}"
        )
        sub_task_id = open_events[0]["task_id"]
        sub_types = {
            e.get("type")
            for e in events
            if e.get("taskId") == sub_task_id
            and e.get("type") not in ("new_tab", "subagentDone")
        }
        assert "result" in sub_types or "text_delta" in sub_types, (
            f"Expected result or text_delta in sub-agent events, got: {sub_types}"
        )
