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
from dataclasses import dataclass

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.server import VSCodeServer
from kiss.tests.agents.vscode._memory_printer import MemoryPrinter


@dataclass(frozen=True)
class _AskResult:
    """Observed state from one ask-user lifecycle."""

    answer: str
    ask_tabs: set[object]
    clear_tabs: set[object]


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
        result = self._ask_and_answer(
            task_id="task-42",
            owner_tab="owner-tab",
            answer_tab="viewer-tab",
            question="Proceed?",
            answer="yes",
        )
        self.assertEqual(result.answer, "yes")
        self.assertEqual(result.ask_tabs, {"owner-tab", "viewer-tab"})
        self.assertEqual(result.clear_tabs, {"owner-tab", "viewer-tab"})

    def test_answer_only_clears_the_task_that_consumed_it(self) -> None:
        """Stale subscriber sets must not receive unrelated close events.

        Completed task subscriber sets are intentionally retained so
        post-task broadcasts can still reach viewers.  A browser tab can
        therefore appear in both an old completed task and the new task
        it is currently answering.  The close event must target the task
        whose live answer queue consumed the answer, not every historic
        subscriber set containing the answering tab.
        """
        self.server.printer.subscribe_tab("old-completed-task", "viewer-tab")
        self.server.printer.subscribe_tab("old-completed-task", "unrelated-tab")

        result = self._ask_and_answer(
            task_id="new-running-task",
            owner_tab="owner-tab",
            answer_tab="viewer-tab",
            question="Continue new task?",
            answer="continue",
        )
        self.assertEqual(result.answer, "continue")
        self.assertEqual(result.ask_tabs, {"owner-tab", "viewer-tab"})
        self.assertEqual(result.clear_tabs, {"owner-tab", "viewer-tab"})
        self.assertNotIn("unrelated-tab", result.clear_tabs)

    def _ask_and_answer(
        self,
        *,
        task_id: str,
        owner_tab: str,
        answer_tab: str,
        question: str,
        answer: str,
    ) -> _AskResult:
        """Run one real ask-user lifecycle through the backend."""
        owner_q: queue.Queue[str] = queue.Queue(maxsize=1)
        self.server._get_tab(owner_tab).user_answer_queue = owner_q
        self.server._get_tab(answer_tab)
        self.server.printer.subscribe_tab(task_id, owner_tab)
        self.server.printer.subscribe_tab(task_id, answer_tab)
        stop = threading.Event()
        result: dict[str, str] = {}
        started = threading.Event()
        done = threading.Event()

        def ask_from_agent_thread() -> None:
            self.server.printer._thread_local.task_id = task_id
            self.server.printer._thread_local.stop_event = stop
            started.set()
            result["answer"] = self.server._ask_user_question(question)
            done.set()

        waiter = threading.Thread(target=ask_from_agent_thread, daemon=True)
        waiter.start()
        self.assertTrue(started.wait(timeout=1.0))
        self._wait_for_event_count("askUser", 2)

        ask_events = [
            ev for ev in self.printer.emitted if ev.get("type") == "askUser"
        ]
        self.assertTrue(all(ev.get("question") == question for ev in ask_events))

        self.server._handle_command({
            "type": "userAnswer",
            "tabId": answer_tab,
            "answer": answer,
        })
        self.assertTrue(done.wait(timeout=1.0))
        waiter.join(timeout=1.0)
        stop.set()

        clear_events = [
            ev
            for ev in self.printer.emitted
            if ev.get("type") == "askUserDone"
        ]
        return _AskResult(
            answer=result.get("answer", ""),
            ask_tabs={ev.get("tabId") for ev in ask_events},
            clear_tabs={ev.get("tabId") for ev in clear_events},
        )

    def _wait_for_event_count(self, event_type: str, count: int) -> None:
        """Wait until MemoryPrinter captured at least *count* events."""
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            events = [
                ev for ev in self.printer.emitted if ev.get("type") == event_type
            ]
            if len(events) >= count:
                return
            time.sleep(0.01)
        self.fail(f"Timed out waiting for {count} {event_type!r} events")


if __name__ == "__main__":
    unittest.main()
