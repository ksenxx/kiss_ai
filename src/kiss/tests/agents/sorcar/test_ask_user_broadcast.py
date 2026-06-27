# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for multi-client ask-user prompt broadcast.

When one chat is open in multiple frontend tabs/clients, an agent
``ask_user_question`` must appear in every subscribed view of that chat.
After any one view answers, every sibling view must clear the modal so
no stale answer window remains.
"""

from __future__ import annotations

import queue
import threading
import time
import unittest

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer
from kiss.tests.agents.vscode._memory_printer import MemoryPrinter


class TestAskUserBroadcastLifecycle(unittest.TestCase):
    """Ask-user prompts open and close across all task subscribers."""

    def setUp(self) -> None:
        self.printer = MemoryPrinter()
        self.server = VSCodeServer(printer=self.printer)

    def tearDown(self) -> None:
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.clear()

    def test_question_fans_out_and_answer_clears_every_subscriber(self) -> None:
        """Answering in one subscribed tab clears every subscribed modal.

        This drives the backend through the same code paths used by the
        web transports: the agent thread calls ``_ask_user_question``;
        ``MemoryPrinter`` mirrors production task-event fanout to all
        subscribers; and the user's answer enters through
        ``_handle_command({type: "userAnswer"})``.  Before the fix,
        both tabs receive ``askUser`` but only the answering frontend
        clears itself locally — no backend event tells the other tab to
        close its stale modal.
        """
        owner_q: queue.Queue[str] = queue.Queue(maxsize=1)
        self.server._get_tab("owner-tab").user_answer_queue = owner_q
        self.server._get_tab("viewer-tab")
        self.server.printer.subscribe_tab("task-42", "owner-tab")
        self.server.printer.subscribe_tab("task-42", "viewer-tab")
        stop = threading.Event()
        result: dict[str, str] = {}
        started = threading.Event()
        done = threading.Event()

        def ask_from_agent_thread() -> None:
            self.server.printer._thread_local.task_id = "task-42"
            self.server.printer._thread_local.stop_event = stop
            started.set()
            result["answer"] = self.server._ask_user_question("Proceed?")
            done.set()

        waiter = threading.Thread(target=ask_from_agent_thread, daemon=True)
        waiter.start()
        self.assertTrue(started.wait(timeout=1.0))

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            ask_events = [
                ev for ev in self.printer.emitted if ev.get("type") == "askUser"
            ]
            if len(ask_events) >= 2:
                break
            time.sleep(0.01)

        ask_events = [
            ev for ev in self.printer.emitted if ev.get("type") == "askUser"
        ]
        self.assertEqual(
            {ev.get("tabId") for ev in ask_events},
            {"owner-tab", "viewer-tab"},
        )
        self.assertTrue(all(ev.get("question") == "Proceed?" for ev in ask_events))

        self.server._handle_command({
            "type": "userAnswer",
            "tabId": "viewer-tab",
            "answer": "yes",
        })
        self.assertTrue(done.wait(timeout=1.0))
        waiter.join(timeout=1.0)
        stop.set()
        self.assertEqual(result.get("answer"), "yes")

        clear_events = [
            ev
            for ev in self.printer.emitted
            if ev.get("type") == "askUserDone"
        ]
        self.assertEqual(
            {ev.get("tabId") for ev in clear_events},
            {"owner-tab", "viewer-tab"},
        )


if __name__ == "__main__":
    unittest.main()
