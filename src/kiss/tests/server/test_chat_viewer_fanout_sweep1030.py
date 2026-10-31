# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""What the tabs viewing a chat receive from its runs (sweep 2026-10-30, V4).

Three defects in ``task_runner``'s viewer fan-out, each reproduced end
to end:

* ``_subscribe_chat_viewers`` scans the chat's viewers under
  ``_state_lock``, releases it, and re-takes it per viewer.  The
  per-viewer step re-checked only "is the viewer busy", so a tab
  closed (or moved to another chat) inside that window was subscribed
  to the run again after its teardown had dropped its subscriptions.
* A viewer subscribed to two live runs of one chat got
  ``running=False`` when the first ended, so its spinner stopped while
  the second run kept streaming into it.
* ``_subscribe_chat_viewers`` ran for every ``task_history`` row of a
  run, so a viewer of a multi-``<task>`` run got a ``clear`` at every
  subtask and lost the earlier subtasks' transcript, while the
  launcher keeps one transcript for the whole run.

Everything is real: an ``AgentServer``, runs through ``_cmd_run``
against a local stand-in model endpoint, the history-click attach
(``_replay_session``) and the tab-close path (``_close_tab``).  The
race test widens the scan-to-subscribe window with the production
``KISS_RACE_DELAY`` hook and lands the close inside it.  No mocks,
patches or test doubles.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from typing import Any

import kiss.agents.sorcar.persistence as persistence
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
    wait_for,
)
from kiss.tests.server.test_autocommit_wire_toggle import _FakeCredentials


def _in_scan_window(thread: threading.Thread) -> bool:
    """True while *thread* sleeps between ``_subscribe_chat_viewers``' scan and per-viewer step."""
    frame = sys._current_frames().get(thread.ident or 0)
    return (
        frame is not None
        and frame.f_code.co_name == "_race_delay"
        and frame.f_back is not None
        and frame.f_back.f_code.co_name == "_subscribe_chat_viewers"
    )


def _wait_in_scan_window(thread: threading.Thread, timeout: float = 10.0) -> bool:
    """Poll every millisecond until *thread* is inside the window (it lasts 100 ms)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _in_scan_window(thread):
            return True
        time.sleep(0.001)
    return False


def _captured(printer: CapturePrinter) -> list[dict[str, Any]]:
    """A snapshot of every event *printer* has broadcast so far."""
    with printer._capture_lock:
        return list(printer.captured)


def _rows_of(chat_id: str) -> list[str]:
    """The ``task_history`` row ids of *chat_id* in creation order (ids are random hex)."""
    rows = (
        persistence._get_db()
        .execute("SELECT id FROM task_history WHERE chat_id = ? ORDER BY rowid", (chat_id,))
        .fetchall()
    )
    return [str(row[0]) for row in rows]


class TestClosedTabIsNotSubscribed(unittest.TestCase):
    """A tab closed inside the scan-to-subscribe window stays unsubscribed."""

    def test_tab_closed_inside_the_window_gets_no_subscription_and_no_events(self) -> None:
        chat_id = "chat-sweep1030-closed"
        launcher, viewer = "sweep1030-closed-launcher", "sweep1030-closed-viewer"
        task_id = "sweep1030-closed-row"
        printer = CapturePrinter()
        server = AgentServer(printer=printer)
        with server._state_lock:
            server._tab_chat_views[launcher] = chat_id
            server._tab_chat_views[viewer] = chat_id
        os.environ["KISS_RACE_DELAY"] = "0.1"
        try:
            worker = threading.Thread(
                target=server._subscribe_chat_viewers,
                args=(task_id, chat_id),
                kwargs={"source_tab_id": launcher, "start_ms": 1},
                daemon=True,
            )
            worker.start()
            self.assertTrue(
                _wait_in_scan_window(worker),
                "the worker never reached its scan-to-subscribe window",
            )
            # Holding the lock the per-viewer step needs pins the close
            # inside the window, whatever the scheduler does.
            with server._state_lock:
                server._close_tab(viewer)
                closed_at = len(_captured(printer))
            worker.join(timeout=10)
            self.assertFalse(worker.is_alive())
        finally:
            os.environ.pop("KISS_RACE_DELAY", None)
        self.assertNotIn(
            viewer,
            printer._fanout_targets(task_id),
            "BUG: the closed tab was subscribed to the run after its teardown",
        )
        late = [e for e in _captured(printer)[closed_at:] if e.get("tabId") == viewer]
        self.assertEqual(late, [], "BUG: the closed tab was sent the run's clear/status")


class _RunHarness(unittest.TestCase):
    """A real server, an isolated KISS home and a stand-in model endpoint."""

    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-sweep1030-viewer-")
        self.addCleanup(self.home.cleanup)
        self.home.write_config(max_budget=5.0, use_web_browser=False)
        self.credentials = _FakeCredentials()
        self.addCleanup(self.credentials.restore)
        self.standin = StandInModelServer(self._respond)
        self.addCleanup(self.standin.stop)
        self.addCleanup(self._join_workers)
        self.release = threading.Event()
        self.addCleanup(self.release.set)
        self.printer = CapturePrinter()
        self.server = AgentServer(printer=self.printer)
        self.server.work_dir = str(self.home.repo)

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        return finish_response("sweep1030 done")

    def _join_workers(self) -> None:
        with agent_state.STATE_LOCK:
            threads = [
                state.task_thread
                for state in agent_state.agent_states.values()
                if state.task_thread is not None
            ]
        for thread in threads:
            thread.join(timeout=30)

    def _start(self, tab_id: str, chat_id: str, prompt: str, **extra: Any) -> threading.Thread:
        """Submit a run on *tab_id* in *chat_id* (plus *extra* fields); return its worker."""
        self.server._cmd_run(
            {
                "type": "run",
                "tabId": tab_id,
                "chatId": chat_id,
                "prompt": prompt,
                "model": STANDIN_MODEL,
                "workDir": str(self.home.repo),
                "isParallel": False,
                "useWebTools": False,
                "maxBudget": 5.0,
                "modelConfig": self.standin.model_config,
                **extra,
            }
        )
        state = agent_state.find_by_tab(tab_id)
        assert state is not None and state.task_thread is not None, tab_id
        return state.task_thread

    def _finish(self, worker: threading.Thread) -> None:
        worker.join(timeout=60)
        self.assertFalse(worker.is_alive(), "the run did not finish")

    def _viewer_status(self, viewer: str) -> list[bool]:
        return [
            bool(e.get("running"))
            for e in _captured(self.printer)
            if e.get("type") == "status" and e.get("tabId") == viewer
        ]


class TestViewerOfTwoRuns(_RunHarness):
    """The end of one run does not stop a viewer that still watches another."""

    def setUp(self) -> None:
        super().setUp()
        self.release_second = threading.Event()
        self.addCleanup(self.release_second.set)

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        # Only agent-loop calls (they offer tools) park: they come after
        # the run's row exists, unlike a task classifier's call.  The
        # second run's context names the first run's prompt too, so its
        # marker is tested first.
        text = request_text(request) if request.get("tools") else ""
        if "SWEEP1030-SECOND" in text:
            self.release_second.wait(60)
        elif "SWEEP1030-FIRST" in text:
            self.release.wait(60)
        return finish_response("sweep1030 done")

    def test_viewer_keeps_running_until_the_last_run_it_watches_ends(self) -> None:
        chat_id = "chat-sweep1030-two-runs"
        viewer = "sweep1030-two-viewer"
        with self.server._state_lock:
            self.server._tab_chat_views[viewer] = chat_id
        first = self._start("sweep1030-api-first", chat_id, "SWEEP1030-FIRST: finish")
        self.assertTrue(wait_for(lambda: len(self.printer.tasks_for_tab(viewer)) == 1, 60))
        second = self._start("sweep1030-api-second", chat_id, "SWEEP1030-SECOND: finish")
        self.assertTrue(wait_for(lambda: len(self.printer.tasks_for_tab(viewer)) == 2, 60))

        self.release.set()
        self._finish(first)
        self.assertNotIn(
            False,
            self._viewer_status(viewer),
            "BUG: the first run's end stopped the viewer that still streams the second run",
        )

        self.release_second.set()
        self._finish(second)
        self.assertEqual(self._viewer_status(viewer)[-1], False)


class TestViewerKeepsTranscriptAcrossSubtasks(_RunHarness):
    """A viewer of a multi-``<task>`` run is cleared once per run, like the launcher."""

    def setUp(self) -> None:
        super().setUp()
        self.calls = 0
        self.calls_lock = threading.Lock()
        self.first_call = threading.Event()

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        # Each subtask answers with one ``finish`` call, so the first
        # agent-loop call (it offers tools; a task classifier's call,
        # made before the row exists, does not) is the first subtask's.
        with self.calls_lock:
            self.calls += bool(request.get("tools"))
            first = self.calls == 1 and bool(request.get("tools"))
        if first:
            self.first_call.set()
            self.release.wait(60)
        return finish_response("sweep1030 subtask done")

    def test_history_viewer_keeps_the_first_subtask_when_the_second_starts(self) -> None:
        chat_id = "chat-sweep1030-subtasks"
        launcher, viewer = "sweep1030-sub-launcher", "sweep1030-sub-viewer"
        worker = self._start(
            launcher,
            chat_id,
            "<task>first sweep1030 subtask</task><task>second sweep1030 subtask</task>",
        )
        self.assertTrue(self.first_call.wait(60))
        self.server._replay_session(chat_id, viewer)
        attached_at = len(_captured(self.printer))

        self.release.set()
        self._finish(worker)
        cleared = [
            e
            for e in _captured(self.printer)[attached_at:]
            if e.get("type") == "clear" and e.get("tabId") == viewer
        ]
        self.assertEqual(cleared, [], "BUG: the second subtask wiped the viewer's transcript")
        rows = _rows_of(chat_id)
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(self.printer.tasks_for_tab(viewer)), set(rows))

    def test_idle_viewer_is_cleared_and_started_once_per_run(self) -> None:
        chat_id = "chat-sweep1030-subtasks-idle"
        launcher, viewer = "sweep1030-idle-launcher", "sweep1030-idle-viewer"
        with self.server._state_lock:
            self.server._tab_chat_views[viewer] = chat_id
        self.release.set()
        worker = self._start(
            launcher,
            chat_id,
            "<task>first sweep1030 idle subtask</task><task>second sweep1030 idle subtask</task>",
        )
        self._finish(worker)
        starts = [
            (e["type"], e.get("running"))
            for e in _captured(self.printer)
            if e.get("tabId") == viewer and e.get("type") in ("clear", "status")
        ]
        self.assertEqual(starts, [("clear", None), ("status", True), ("status", False)])
        self.assertEqual(set(self.printer.tasks_for_tab(viewer)), set(_rows_of(chat_id)))


if __name__ == "__main__":
    unittest.main()
