"""A browser closing an API run's tab must not cut the API client off its run.

A ``daemon_client.run`` shows its task in the web app under a synthetic
``api-…`` tab.  Closing that tab in a browser hides the task from every
surface but does not stop it (``_drop_tab_state``): the task's events
are fanned out per subscriber tab, and the close used to drop the tab's
subscription, so the API client (which addressed that tab) received
nothing more — not the second step, not the ``result`` — and returned
an empty failed :class:`TaskResult` when the task's terminal status
arrived, although the task had finished successfully (audit1101 S4 F2,
the mechanism behind it).  Now the close keeps the running task's
subscription of a tab a local client addressed, headless: its copies
reach that client only, and the tab is no longer a viewer for later
sub-agent spawns.

A reconnect of that client after the close (audit1101 S4 F2 proper)
sends a recovery ``resumeSession``: the subscription and transcript
come back, the tab the user closed does not.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import daemon_client, local_endpoint
from kiss.tests.agents.sorcar.test_daemon_client_reconnect_2026_10_31 import (
    PARENT_PROMPT,
    SUMMARY,
    ReconnectHarness,
)
from kiss.tests.server.parallel_agent_harness import STANDIN_MODEL, wait_for

pytestmark = pytest.mark.timeout(300)

#: Task events fanned out per subscriber tab (never the lifecycle
#: broadcasts ``status`` / ``task_done`` / ``tabs_state``, which every
#: client receives whatever it shows).
FANNED_OUT = {"llm_call", "usage_info", "tool_call", "tool_result", "text_end", "result"}


def _frames_for(frames: list[Any], tab_id: str, types: set[str]) -> list[str]:
    """The types among *types* of the frames stamped *tab_id* in *frames* (JSON strings)."""
    found: list[str] = []
    for raw in frames:
        if not isinstance(raw, str):
            continue
        frame = json.loads(raw)
        if frame.get("tabId") == tab_id and frame.get("type") in types:
            found.append(str(frame["type"]))
    return found


def _announced(frames: list[Any], tab_id: str) -> list[str]:
    """The sub-agent tab announcements of *tab_id* among *frames* (what opens it in a browser)."""
    return _frames_for(frames, tab_id, {"openSubagentTab"})


class _Surface:
    """Another client of the daemon: a local viewer that closes the run's tab.

    The harness daemon is private (local clients only), so a web app
    cannot connect to it; what a browser would receive is observed on
    the daemon itself instead: every payload handed to its real
    ``_send_to_remote_clients`` (the browsers' send) is recorded while
    that send still runs.
    """

    def __init__(self, case: ReconnectHarness) -> None:
        self.ws = local_endpoint.connect(Path(case.endpoint_file))
        self.remote_payloads: list[Any] = []
        printer: Any = case.server._agent_server.printer  # a WebPrinter
        original = printer._send_to_remote_clients

        def recording_send(data: Any) -> None:
            self.remote_payloads.append(data)
            original(data)

        printer._send_to_remote_clients = recording_send
        case.addCleanup(setattr, printer, "_send_to_remote_clients", original)

    def send(self, frame: dict[str, Any]) -> None:
        local_endpoint.send(self.ws, json.dumps(frame))

    def close(self) -> None:
        self.ws.close()

    def fanned_out_for(self, tab_id: str, start: int) -> list[str]:
        """The task-event copies of *tab_id* a browser would have received since *start*."""
        return _frames_for(self.remote_payloads[start:], tab_id, FANNED_OUT)


class ClosedTabKeepsApiClient(ReconnectHarness):
    """Another surface closes the run's tab while the API client waits."""

    def _close_on_surface(self, surface: _Surface, tab: str, task_id: str) -> None:
        """Close *tab* from *surface* and wait for the close's whole tail.

        The registry drops the tab first and the printer's viewer
        subscription after it, in a later lock section
        (``_close_tab`` → ``_drop_tab_state``), so both are awaited.
        """
        registry = self.server._agent_server.tab_registry
        printer = self.server._agent_server.printer
        surface.send({"type": "closeTab", "tabId": tab})
        self.assertTrue(
            wait_for(
                lambda: not registry.has_tab(tab) and tab not in printer.viewer_tab_ids(task_id),
                timeout=10,
            )
        )

    def test_closing_the_run_tab_in_a_browser_keeps_the_client_its_result(self) -> None:
        """The client still gets its run's events and real result; the browser gets none."""
        worker, out = self._start_run(reconnect_seconds=10.0)
        self._await_blocked_task()
        state = self._probe_state()
        tab, task_id = state.tab_id, str(state.task_id)
        printer = self.server._agent_server.printer
        surface = _Surface(self)
        try:
            self.assertTrue(wait_for(lambda: tab in printer.viewer_tab_ids(task_id), timeout=10))
            self._close_on_surface(surface, tab, task_id)
            self.assertTrue(state.frontend_closed)
            # The tab shows on no surface, so it parents no later
            # spawn, but the client that addressed it keeps its stream.
            self.assertEqual(printer.viewer_tab_ids(task_id), [])
            self.assertEqual(printer._fanout_targets(task_id), [tab])
            time.sleep(0.5)  # the close's own broadcasts land
            mark = len(surface.remote_payloads)
            self.release.set()
            result = self._collect(worker, out)
            self.assertIsInstance(result, daemon_client.TaskResult, result)
            self.assertTrue(result.success, result)
            self.assertIn(SUMMARY, result.text)
            self.assertEqual(result.steps, 2)
            self.assertEqual(result.task_id, task_id)
            self.assertFalse(self.server._agent_server.tab_registry.has_tab(tab))
            self.assertTrue(state.frontend_closed)
            self.assertEqual(surface.fanned_out_for(tab, mark), [])
        finally:
            surface.close()
            self.release.set()

    def test_a_reconnect_after_the_close_does_not_reopen_the_tab(self) -> None:
        """A dropped connection recovers the subscription, not the tab the user closed."""
        worker, out = self._start_run(reconnect_seconds=60.0)
        self._await_blocked_task()
        state = self._probe_state()
        tab, task_id = state.tab_id, str(state.task_id)
        registry = self.server._agent_server.tab_registry
        printer = self.server._agent_server.printer
        surface = _Surface(self)
        try:
            self._close_on_surface(surface, tab, task_id)
            mark = len(surface.remote_payloads)
            self.relay.cut()
            self.assertTrue(
                wait_for(lambda: self.relay.accepted >= 2, timeout=60),
                "the client never reconnected through the relay",
            )
            # The recovery re-attached the client (its tab is a fan-out
            # target again, headless) without publishing the tab.
            self.assertTrue(wait_for(lambda: tab in printer._fanout_targets(task_id), timeout=30))
            self.assertFalse(wait_for(lambda: registry.has_tab(tab), timeout=2))
            self.assertTrue(state.frontend_closed)
            self.assertEqual(printer.viewer_tab_ids(task_id), [])
            self.release.set()
            result = self._collect(worker, out)
            self.assertIsInstance(result, daemon_client.TaskResult, result)
            self.assertTrue(result.success, result)
            self.assertIn(SUMMARY, result.text)
            self.assertEqual(result.reconnects, 1)
            self.assertFalse(registry.has_tab(tab))
            self.assertEqual(surface.fanned_out_for(tab, mark), [])
        finally:
            surface.close()
            self.release.set()

    def test_a_reconnect_into_a_tab_still_open_keeps_it_a_viewer(self) -> None:
        """A recovery into a tab nobody closed changes nothing: the browsers keep their copies."""
        worker, out = self._start_run(reconnect_seconds=60.0)
        self._await_blocked_task()
        state = self._probe_state()
        tab, task_id = state.tab_id, str(state.task_id)
        registry = self.server._agent_server.tab_registry
        printer = self.server._agent_server.printer
        surface = _Surface(self)
        try:
            self.assertTrue(wait_for(lambda: registry.has_tab(tab), timeout=10))
            self.relay.cut()
            self.assertTrue(
                wait_for(lambda: self.relay.accepted >= 2, timeout=60),
                "the client never reconnected through the relay",
            )
            self.assertTrue(wait_for(lambda: tab in printer._fanout_targets(task_id), timeout=30))
            self.assertTrue(registry.has_tab(tab))
            self.assertFalse(state.frontend_closed)
            self.assertEqual(printer.viewer_tab_ids(task_id), [tab])
            mark = len(surface.remote_payloads)
            self.release.set()
            result = self._collect(worker, out)
            self.assertIsInstance(result, daemon_client.TaskResult, result)
            self.assertTrue(result.success, result)
            self.assertEqual(result.reconnects, 1)
            self.assertIn("result", surface.fanned_out_for(tab, mark))
        finally:
            surface.close()
            self.release.set()

    def test_a_reconnect_does_not_reopen_a_closed_sub_agent_tab(self) -> None:
        """A sub-agent dispatch's recovery leaves the ``__sub_`` tab the user closed closed.

        Every chat page shows a sub-agent under ``{parent}__sub_{task}``;
        the dispatch's own ``api-…`` tab was never closed, so its
        recovered subscription is what it was.
        """
        parent = daemon_client.run(
            PARENT_PROMPT,
            work_dir=self.repo,
            endpoint_file=self.relay_endpoint,
            model=STANDIN_MODEL,
            model_config=self.standin.model_config,
            max_budget=5.0,
            use_web_tools=False,
            auto_classify=False,
            use_memory=False,
            is_parallel=False,
            timeout=120.0,
        )
        self.assertTrue(parent.success, parent)
        worker, out = self._start_run(
            reconnect_seconds=60.0,
            parent_task_id=parent.task_id,
            parent_tab_id="webtab-parent",
        )
        self._await_blocked_task()
        state = self._probe_state()
        self.assertTrue(state.is_subagent)
        tab, task_id = state.tab_id, str(state.task_id)
        sub_tab = f"webtab-parent__sub_{task_id}"
        server = self.server._agent_server
        printer = server.printer
        surface = _Surface(self)
        try:
            surface.send({"type": "closeTab", "tabId": sub_tab})
            self.assertTrue(wait_for(lambda: sub_tab in server._closed_subagent_tabs, timeout=10))
            time.sleep(0.5)  # the close's own broadcasts land
            mark = len(surface.remote_payloads)
            self.relay.cut()
            self.assertTrue(
                wait_for(lambda: self.relay.accepted >= 3, timeout=60),
                "the client never reconnected through the relay",
            )
            self.assertTrue(wait_for(lambda: tab in printer.viewer_tab_ids(task_id), timeout=30))
            self.assertIn(sub_tab, server._closed_subagent_tabs)
            self.release.set()
            result = self._collect(worker, out)
            self.assertIsInstance(result, daemon_client.TaskResult, result)
            self.assertTrue(result.success, result)
            self.assertIn(SUMMARY, result.text)
            self.assertEqual(result.reconnects, 1)
            self.assertEqual(_announced(surface.remote_payloads[mark:], sub_tab), [])
            # The dispatch's own tab is the viewer it was at dispatch:
            # its copies still take the browsers' send.
            self.assertIn("result", surface.fanned_out_for(tab, mark))
        finally:
            surface.close()
            self.release.set()

    def test_the_row_allocation_keeps_a_closed_tab_headless(self) -> None:
        """A close and recovery in the run's setup window: the row allocation changes nothing.

        The daemon subscribes the launching tab under the run's row id
        the moment the row exists (``_on_run_task_id_allocated``, and
        again for every later subtask of a sequential run).  That
        subscription used to be a viewer's whatever the close decided,
        so an absent tab fanned the run out to the browsers again.
        """
        server = self.server._agent_server
        printer = server.printer
        parked, release = threading.Event(), threading.Event()
        original = server._on_run_task_id_allocated

        def parking(*args: Any, **kwargs: Any) -> None:
            parked.set()
            if not release.wait(timeout=60):
                raise TimeoutError("parked too long in the allocation hook")
            original(*args, **kwargs)

        server._on_run_task_id_allocated = parking  # type: ignore[method-assign]
        self.addCleanup(delattr, server, "_on_run_task_id_allocated")
        worker, out = self._start_run(reconnect_seconds=60.0)
        try:
            self.assertTrue(parked.wait(timeout=60), "the run never allocated its row")
            state = self._probe_state()
            tab, task_id = state.tab_id, str(state.task_id)
            surface = _Surface(self)
            try:
                self._close_on_surface(surface, tab, task_id)
                self.relay.cut()
                self.assertTrue(
                    wait_for(lambda: self.relay.accepted >= 2, timeout=60),
                    "the client never reconnected through the relay",
                )
                self.assertTrue(
                    wait_for(lambda: tab in printer._fanout_targets(task_id), timeout=30),
                )
                self.assertEqual(printer.viewer_tab_ids(task_id), [])
                mark = len(surface.remote_payloads)
                release.set()
                self._await_blocked_task()
                self.assertIn(tab, printer._fanout_targets(task_id))
                self.assertEqual(printer.viewer_tab_ids(task_id), [])
                self.assertTrue(state.frontend_closed)
                self.release.set()
                result = self._collect(worker, out)
                self.assertIsInstance(result, daemon_client.TaskResult, result)
                self.assertTrue(result.success, result)
                self.assertIn(SUMMARY, result.text)
                self.assertEqual(result.reconnects, 1)
                self.assertFalse(server.tab_registry.has_tab(tab))
                self.assertEqual(surface.fanned_out_for(tab, mark), [])
            finally:
                surface.close()
        finally:
            release.set()
            self.release.set()

    def test_a_tab_reopened_from_history_is_a_viewer_again(self) -> None:
        """A surface resuming the chat into the tab makes it a viewer: the headless mark goes."""
        worker, out = self._start_run(reconnect_seconds=10.0)
        self._await_blocked_task()
        state = self._probe_state()
        tab, task_id = state.tab_id, str(state.task_id)
        registry = self.server._agent_server.tab_registry
        printer = self.server._agent_server.printer
        surface = _Surface(self)
        try:
            self._close_on_surface(surface, tab, task_id)
            self.assertEqual(printer.viewer_tab_ids(task_id), [])
            surface.send({"type": "resumeSession", "chatId": state.chat_id, "tabId": tab})
            self.assertTrue(wait_for(lambda: registry.has_tab(tab), timeout=10))
            self.assertTrue(wait_for(lambda: printer.viewer_tab_ids(task_id) == [tab], timeout=10))
            # The replay publishes the tab and subscribes the viewer before it
            # clears the headless mark (``_commit_replay_publication``), and
            # the reads above are not under the daemon's state lock.
            self.assertTrue(wait_for(lambda: not state.frontend_closed, timeout=10))
            mark = len(surface.remote_payloads)
            self.release.set()
            result = self._collect(worker, out)
            self.assertTrue(result.success, result)
            # A shown tab's copies reach the browsers again.
            self.assertIn("result", surface.fanned_out_for(tab, mark))
        finally:
            surface.close()
            self.release.set()
