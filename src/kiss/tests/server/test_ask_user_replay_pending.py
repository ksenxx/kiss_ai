# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for pending ask-user replay to (re)joining clients.

A running task's ``ask_user_question`` modal must appear on EVERY
client viewing the task's tab — including a client that connects or
reloads AFTER the question was broadcast.  Such clients synchronize
through ``resumeSession`` (the ready pipeline replays every registry
tab), so the replay path must re-emit a still-pending question, and
must NOT re-emit a question that was already answered or abandoned.
"""

from __future__ import annotations

import queue
import threading
import time
import unittest

from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.server._memory_printer import MemoryPrinter


class _SlowAskPrinter(MemoryPrinter):
    """MemoryPrinter whose initial ``askUser`` broadcast is slow.

    Emulates a slow transport for the agent thread's initial (fanout)
    ``askUser`` broadcast: entry is signalled via
    ``ask_broadcast_entered`` and the emit blocks until ``release_ask``
    is set, so a test can deterministically interleave a concurrent
    ``userAnswer`` with the in-flight broadcast.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ask_broadcast_entered = threading.Event()
        self.release_ask = threading.Event()

    def broadcast(self, event: dict[str, object]) -> None:
        """Delay the fanout ``askUser`` emit until released."""
        if event.get("type") == "askUser" and "tabId" not in event:
            self.ask_broadcast_entered.set()
            self.release_ask.wait(timeout=2.0)
        super().broadcast(event)


class TestAskUserAnswerOrdering(unittest.TestCase):
    """The initial ``askUser`` can never be ordered after its answer.

    Regression for a review finding: ``_ask_user_question`` used to
    publish ``pending_ask_question`` under ``_state_lock`` but
    broadcast the initial ``askUser`` OUTSIDE it.  A concurrent
    ``userAnswer`` (e.g. answering the copy a session replay had
    already re-emitted) could then emit ``askUserDone`` before the
    initial ``askUser`` hit the wire — reopening the modal on every
    client after its answer had closed it, forever.
    """

    def tearDown(self) -> None:
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()

    def test_slow_initial_broadcast_still_precedes_ask_user_done(self) -> None:
        """A concurrent answer must serialize behind the slow broadcast."""
        printer = _SlowAskPrinter()
        server = VSCodeServer(printer=printer)
        state = agent_state.AgentState(
            "task-race",
            chat_id="chat-race",
            tab_id="owner-tab",
            server_owned=True,
            is_task_active=True,
        )
        state.user_answer_queue = queue.Queue(maxsize=1)
        agent_state.register(state)
        printer.subscribe_tab("task-race", "owner-tab")
        stop = threading.Event()
        done = threading.Event()
        answer: dict[str, str] = {}

        def ask_from_agent_thread() -> None:
            printer._thread_local.task_id = "task-race"
            printer._thread_local.stop_event = stop
            try:
                answer["value"] = server._ask_user_question("Race?")
            finally:
                done.set()

        threading.Thread(target=ask_from_agent_thread, daemon=True).start()
        self.assertTrue(
            printer.ask_broadcast_entered.wait(timeout=2.0),
            "agent thread never reached the askUser broadcast",
        )

        def answer_from_client_thread() -> None:
            server._handle_command({
                "type": "userAnswer",
                "tabId": "owner-tab",
                "answer": "yes",
            })

        answerer = threading.Thread(
            target=answer_from_client_thread, daemon=True,
        )
        answerer.start()
        # Give the answer thread a real chance to overtake the
        # in-flight broadcast, then let the broadcast finish.
        time.sleep(0.1)
        printer.release_ask.set()
        answerer.join(timeout=2.0)
        self.assertTrue(done.wait(timeout=2.0))
        self.assertEqual(answer.get("value"), "yes")
        stop.set()

        types = [
            ev.get("type")
            for ev in printer.emitted
            if ev.get("type") in ("askUser", "askUserDone")
        ]
        self.assertIn("askUser", types)
        self.assertIn("askUserDone", types)
        last_ask = len(types) - 1 - types[::-1].index("askUser")
        first_done = types.index("askUserDone")
        self.assertLess(
            last_ask,
            first_done,
            f"askUser emitted after askUserDone (stale modal): {types}",
        )


class TestAskUserReplayPending(unittest.TestCase):
    """Pending ask-user questions follow the session-replay path."""

    def setUp(self) -> None:
        self.printer = MemoryPrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.stop = threading.Event()
        self.done = threading.Event()
        self.answer: dict[str, str] = {}

    def tearDown(self) -> None:
        self.stop.set()
        self.done.wait(timeout=2.0)
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()

    def _start_pending_question(
        self, *, task_id: str, chat_id: str, owner_tab: str, question: str,
    ) -> None:
        """Register a live task and block its thread on *question*."""
        state = agent_state.AgentState(
            task_id,
            chat_id=chat_id,
            tab_id=owner_tab,
            server_owned=True,
            is_task_active=True,
        )
        state.user_answer_queue = queue.Queue(maxsize=1)
        agent_state.register(state)
        self.server.printer.subscribe_tab(task_id, owner_tab)
        started = threading.Event()

        def ask_from_agent_thread() -> None:
            self.server.printer._thread_local.task_id = task_id
            self.server.printer._thread_local.stop_event = self.stop
            started.set()
            try:
                self.answer["value"] = self.server._ask_user_question(question)
            except KeyboardInterrupt:
                self.answer["value"] = "<interrupted>"
            finally:
                self.done.set()

        threading.Thread(target=ask_from_agent_thread, daemon=True).start()
        self.assertTrue(started.wait(timeout=1.0))
        self._wait_for(
            lambda: any(
                ev.get("type") == "askUser" for ev in self.printer.emitted
            ),
            "initial askUser broadcast",
        )

    def _ask_events_for_tab(self, tab_id: str) -> list[dict[str, object]]:
        """Return every ``askUser`` event stamped for *tab_id*."""
        return [
            ev
            for ev in self.printer.emitted
            if ev.get("type") == "askUser" and ev.get("tabId") == tab_id
        ]

    def _wait_for(self, cond, what: str) -> None:
        """Poll *cond* until true or fail after one second."""
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if cond():
                return
            time.sleep(0.01)
        self.fail(f"Timed out waiting for {what}")

    def test_resume_replays_pending_question_to_new_viewer_tab(self) -> None:
        """A client joining mid-question receives the askUser modal.

        The joining client's ready pipeline runs ``resumeSession`` for
        the shared tab; with no recorded chat rows the replay reattaches
        the running chat and must re-emit the pending question stamped
        with the resuming tab id.
        """
        self._start_pending_question(
            task_id="task-pending",
            chat_id="chat-pending",
            owner_tab="owner-tab",
            question="Deploy to production?",
        )
        self.server._handle_command({
            "type": "resumeSession",
            "chatId": "chat-pending",
            "tabId": "viewer-tab",
        })
        replayed = self._ask_events_for_tab("viewer-tab")
        self.assertEqual(len(replayed), 1)
        self.assertEqual(replayed[0].get("question"), "Deploy to production?")

        self.server._handle_command({
            "type": "userAnswer",
            "tabId": "viewer-tab",
            "answer": "yes",
        })
        self.assertTrue(self.done.wait(timeout=1.0))
        self.assertEqual(self.answer.get("value"), "yes")
        clear_tabs = {
            ev.get("tabId")
            for ev in self.printer.emitted
            if ev.get("type") == "askUserDone"
        }
        self.assertIn("owner-tab", clear_tabs)
        self.assertIn("viewer-tab", clear_tabs)

    def test_resume_after_answer_does_not_replay_stale_question(self) -> None:
        """A client joining after the answer sees no stale modal."""
        self._start_pending_question(
            task_id="task-answered",
            chat_id="chat-answered",
            owner_tab="owner-tab",
            question="Continue?",
        )
        self.server._handle_command({
            "type": "userAnswer",
            "tabId": "owner-tab",
            "answer": "go",
        })
        self.assertTrue(self.done.wait(timeout=1.0))
        self.server._handle_command({
            "type": "resumeSession",
            "chatId": "chat-answered",
            "tabId": "late-tab",
        })
        self.assertEqual(self._ask_events_for_tab("late-tab"), [])

    def test_resume_after_stop_does_not_replay_stale_question(self) -> None:
        """A question abandoned by a stopped task is never replayed."""
        self._start_pending_question(
            task_id="task-stopped",
            chat_id="chat-stopped",
            owner_tab="owner-tab",
            question="Still there?",
        )
        self.stop.set()
        self.assertTrue(self.done.wait(timeout=2.0))
        self.assertEqual(self.answer.get("value"), "<interrupted>")
        self.server._handle_command({
            "type": "resumeSession",
            "chatId": "chat-stopped",
            "tabId": "late-tab",
        })
        self.assertEqual(self._ask_events_for_tab("late-tab"), [])


if __name__ == "__main__":
    unittest.main()
