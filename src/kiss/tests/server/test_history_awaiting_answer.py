# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: the History panel's "?" for a task blocked on a question.

While a running task sits in ``ask_user_question`` its ``getHistory``
row carries ``awaiting_answer=True`` (the panel draws a "?" instead of
the running spinner), and the daemon nudges every surface to repaint
with a global ``tasks_updated`` both when the question opens and when
the wait ends.  Once the user has answered, the row goes back to
``awaiting_answer=False`` while the task keeps running.

The test runs the production ``WebPrinter`` and captures the frames at
its single wire chokepoint (``_send_to_ws_clients``) with NO tab
subscribed to the task: the nudges are sent from the agent thread,
where an event without an explicit ``tabId`` is routed to the task's
subscribers only, so a History panel on a surface that closed the
task's tabs would never hear of the question.
"""

from __future__ import annotations

import json
import queue
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence as th
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.server.web_server import Payload, WebPrinter

OWNER_TAB = "owner-tab"


class TestHistoryAwaitingAnswer(unittest.TestCase):
    """``awaiting_answer`` on history rows follows the pending question."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="kiss-history-ask-test-")
        self.orig_db_path = th._DB_PATH  # type: ignore[attr-defined]
        th._close_db()
        th._DB_PATH = Path(self.tmp) / "history.db"  # type: ignore[attr-defined]
        self.printer = WebPrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.server.work_dir = self.tmp
        # Every frame that would leave the daemon for a connected client.
        self.events: list[dict[str, Any]] = []
        self.events_lock = threading.Lock()
        original_send = self.printer._send_to_ws_clients

        def capture(data: Payload, tab_id: str = "") -> None:
            if isinstance(data, str):
                with self.events_lock:
                    self.events.append(json.loads(data))
            original_send(data, tab_id)

        self.printer._send_to_ws_clients = capture  # type: ignore[method-assign]
        self.stop = threading.Event()
        self.done = threading.Event()
        self.done.set()
        self.answer: dict[str, str] = {}

    def tearDown(self) -> None:
        self.stop.set()
        self.done.wait(timeout=2.0)
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
        th._close_db()
        th._DB_PATH = self.orig_db_path  # type: ignore[attr-defined]
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _wait_for(self, cond, what: str) -> None:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if cond():
                return
            time.sleep(0.01)
        self.fail(f"timed out waiting for {what}; events={self.events[-10:]}")

    def _start_task_asking(self, question: str) -> str:
        """Persist a task, register it as running and block it on *question*.

        The asking thread doubles as the task's worker thread, so
        ``_get_running_task_ids`` sees the task alive for as long as
        the question (and the test's ``stop`` event) keep it blocked.
        """
        task_id, _ = th._add_task("task that asks")
        state = agent_state.AgentState(
            str(task_id),
            chat_id="chat-ask",
            tab_id=OWNER_TAB,
            server_owned=True,
            is_task_active=True,
        )
        state.user_answer_queue = queue.Queue(maxsize=1)
        # Deliberately no subscribe_tab: the user closed the task's tabs.
        self.done.clear()

        def ask_from_agent_thread() -> None:
            self.printer._thread_local.task_id = str(task_id)
            self.printer._thread_local.stop_event = self.stop
            try:
                self.answer["value"] = self.server._ask_user_question(question)
            except KeyboardInterrupt:
                self.answer["value"] = "<interrupted>"
            # Keep the "task" alive after the answer so the row is still
            # running when the spinner is expected back.
            self.stop.wait(timeout=5.0)
            self.done.set()

        worker = threading.Thread(target=ask_from_agent_thread, daemon=True)
        state.task_thread = worker
        agent_state.register(state)
        worker.start()
        self._wait_for(
            lambda: state.pending_ask_question == question,
            "the question to become pending",
        )
        return str(task_id)

    def _history_row(self, task_id: str) -> dict[str, Any]:
        with self.events_lock:
            before = len(self.events)
        self.server._handle_command({"type": "getHistory"})
        with self.events_lock:
            new_events = list(self.events[before:])
        for ev in new_events:
            if ev.get("type") == "history":
                for row in ev.get("sessions", []):
                    if str(row.get("task_id")) == task_id:
                        return dict(row)
        self.fail(f"no history row for task {task_id}")

    def _tasks_updated_count(self) -> int:
        with self.events_lock:
            return sum(1 for ev in self.events if ev.get("type") == "tasks_updated")

    def test_row_shows_question_mark_until_answered(self) -> None:
        """Pending: awaiting_answer + a tasks_updated nudge; answered: the spinner is back."""
        task_id = self._start_task_asking("Which branch?")

        row = self._history_row(task_id)
        self.assertTrue(row["is_running"], row)
        self.assertTrue(row["awaiting_answer"], row)
        self._wait_for(
            lambda: self._tasks_updated_count() >= 1,
            "the tasks_updated nudge after the question opened",
        )
        self.assertEqual(
            self._tasks_updated_count(), 1,
            "opening the question nudges every surface to repaint its rows",
        )

        self.server._handle_command(
            {"type": "userAnswer", "tabId": OWNER_TAB, "answer": "main"},
        )
        self._wait_for(lambda: self.answer.get("value") == "main", "the answer to reach the agent")
        self._wait_for(
            lambda: self._tasks_updated_count() >= 2,
            "the tasks_updated nudge after the answer",
        )

        row = self._history_row(task_id)
        self.assertTrue(row["is_running"], "the task keeps running after the answer")
        self.assertFalse(row["awaiting_answer"], row)

    def test_finished_task_row_is_not_awaiting(self) -> None:
        """A persisted row with no live state carries awaiting_answer=False."""
        task_id, _ = th._add_task("finished task")
        th._save_task_result(result="done", task_id=task_id)
        row = self._history_row(str(task_id))
        self.assertFalse(row["is_running"])
        self.assertFalse(row["awaiting_answer"])
        self.assertEqual(self._tasks_updated_count(), 0)


if __name__ == "__main__":
    unittest.main()
