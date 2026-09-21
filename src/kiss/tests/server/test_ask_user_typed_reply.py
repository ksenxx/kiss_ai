# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a chat message typed while a question is pending answers it.

Reproduces sorcar.db task ``e8a8407967d645c28c87750eda7a6cc0``: the agent
called ``ask_user_question`` and the user replied by typing into the chat
box instead of the answer box.  The reply reached the daemon as a
``run``/``appendUserMessage`` command, was queued on
``pending_user_messages`` (a steering message drained only before the
agent's NEXT model step) and the agent — blocked inside the tool on
``user_answer_queue.get()`` — never woke up.

Post-fix, a plain message typed while ``pending_ask_question`` is set is
delivered as the answer, every viewer tab receives ``askUserDone``, and
nothing is left on the steering queue.
"""

from __future__ import annotations

import queue
import threading
import time
import unittest

from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.server._memory_printer import MemoryPrinter

_QUESTION = "Paste the Bot User OAuth Token (xoxb-...)"
_REPLY = "Can you do it autonomously?"


class TestTypedReplyAnswersPendingQuestion(unittest.TestCase):
    """Typed chat messages resolve a blocked ``ask_user_question``."""

    def setUp(self) -> None:
        self.printer = MemoryPrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.stop = threading.Event()
        self.result: dict[str, str] = {}
        self.done = threading.Event()
        self.waiter: threading.Thread | None = None

    def tearDown(self) -> None:
        self.stop.set()
        if self.waiter is not None:
            self.waiter.join(timeout=2.0)
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()

    def _start_blocked_agent(
        self, task_id: str, tab_id: str, *, question: str = _QUESTION,
    ) -> agent_state.AgentState:
        """Register a running task and block its agent thread in the ask tool."""
        state = agent_state.AgentState(
            task_id,
            tab_id=tab_id,
            server_owned=True,
            is_task_active=True,
        )
        state.user_answer_queue = queue.Queue(maxsize=1)
        agent_state.register(state)
        self.server.printer.subscribe_tab(task_id, tab_id)

        def ask_from_agent_thread() -> None:
            self.server.printer._thread_local.task_id = task_id
            self.server.printer._thread_local.stop_event = self.stop
            try:
                self.result["answer"] = self.server._ask_user_question(question)
            except KeyboardInterrupt:
                self.result["answer"] = "<interrupted>"
            self.done.set()

        self.waiter = threading.Thread(target=ask_from_agent_thread, daemon=True)
        # The ``run``-on-busy-tab branch keys on an installed worker thread.
        state.task_thread = self.waiter
        self.waiter.start()
        self._wait_for_event("askUser")
        with agent_state.STATE_LOCK:
            self.assertEqual(state.pending_ask_question, question)
        return state

    def _wait_for_event(self, event_type: str) -> None:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if any(ev.get("type") == event_type for ev in self.printer.emitted):
                return
            time.sleep(0.01)
        self.fail(f"Timed out waiting for a {event_type!r} event")

    def _events(self, event_type: str) -> list[dict[str, object]]:
        return [ev for ev in self.printer.emitted if ev.get("type") == event_type]

    def _assert_answered(self, state: agent_state.AgentState, tab_id: str) -> None:
        self.assertTrue(
            self.done.wait(timeout=2.0),
            "agent stayed blocked in ask_user_question after the typed reply",
        )
        self.assertEqual(self.result.get("answer"), _REPLY)
        with agent_state.STATE_LOCK:
            self.assertEqual(state.pending_ask_question, "")
            self.assertEqual(state.pending_user_messages, [])
            self.assertEqual(state.queued_followup_tasks, [])
        self.assertEqual(
            {ev.get("tabId") for ev in self._events("askUserDone")}, {tab_id},
        )
        # The reply still shows up in the transcript as the user's message.
        echoes = [ev for ev in self._events("prompt") if ev.get("text") == _REPLY]
        self.assertEqual(len(echoes), 1)
        self.assertEqual(echoes[0].get("tabId"), tab_id)

    def test_append_user_message_answers_pending_question(self) -> None:
        """A follow-up typed into the running tab (``appendUserMessage``) answers."""
        state = self._start_blocked_agent("task-typed-A", "tab-A")
        self.server._handle_command({
            "type": "appendUserMessage",
            "tabId": "tab-A",
            "prompt": _REPLY,
        })
        self._assert_answered(state, "tab-A")

    def test_run_on_busy_tab_answers_pending_question(self) -> None:
        """A ``run`` submitted on the busy tab answers instead of queueing steering."""
        state = self._start_blocked_agent("task-typed-B", "tab-B")
        self.server._handle_command({
            "type": "run",
            "tabId": "tab-B",
            "prompt": _REPLY,
        })
        self._assert_answered(state, "tab-B")

    def test_viewer_tab_message_answers_pending_question(self) -> None:
        """A message from a viewer tab subscribed to the task answers it."""
        state = self._start_blocked_agent("task-typed-C", "owner-C")
        self.server.printer.subscribe_tab("task-typed-C", "viewer-C")
        self.server._handle_command({
            "type": "appendUserMessage",
            "tabId": "viewer-C",
            "prompt": _REPLY,
        })
        self._assert_viewer_answered(state, {"owner-C", "viewer-C"})

    def test_run_from_subagent_viewer_tab_answers_pending_question(self) -> None:
        """A ``run`` from a client-local sub-agent tab answers, not starts a task.

        A ``run_agent`` sub-agent keeps its ``api-…`` source tab on its
        state while clients render it in a ``<parent>__sub_<task>`` tab
        that is only SUBSCRIBED to the task, so ``find_by_tab`` never
        resolves it.  Pre-fix a ``run`` typed there started a brand-new
        task in that tab while the sub-agent stayed blocked.
        """
        state = self._start_blocked_agent("child-task", "api-source")
        viewer = "parent-tab__sub_child-task"
        self.server.printer.subscribe_tab("child-task", viewer)
        self.server._handle_command({
            "type": "run",
            "tabId": viewer,
            "prompt": _REPLY,
        })
        self._assert_viewer_answered(state, {"api-source", viewer})
        with agent_state.STATE_LOCK:
            self.assertIsNone(agent_state.find_by_tab(viewer))
        self.assertEqual(self._events("clear"), [])

    def _assert_viewer_answered(
        self, state: agent_state.AgentState, expected_done_tabs: set[str],
    ) -> None:
        self.assertTrue(self.done.wait(timeout=2.0))
        self.assertEqual(self.result.get("answer"), _REPLY)
        with agent_state.STATE_LOCK:
            self.assertEqual(state.pending_ask_question, "")
            self.assertEqual(state.pending_user_messages, [])
        self.assertEqual(
            {ev.get("tabId") for ev in self._events("askUserDone")},
            expected_done_tabs,
        )

    def test_ask_user_done_precedes_the_next_question(self) -> None:
        """The close for question 1 is on the wire before question 2 opens.

        The answer wakes the agent, which may call ``ask_user_question``
        again at once.  Clients clear whichever question is showing on
        ``askUserDone`` (it carries no question identity), so a close
        emitted after question 2's ``askUser`` would dismiss it.  The
        fix emits the close inside the ``STATE_LOCK`` critical section
        that delivers the answer, and question 2's ``askUser`` needs
        that lock, so the order below is guaranteed by construction.

        Scope: this pins the post-fix invariant.  It does not force the
        pre-fix interleaving (handler thread descheduled between the
        lock release and the close) — that would need a hook into the
        handler thread, which this suite does not use — so against the
        old code it would only fail when the scheduler happened to
        interleave that way.
        """
        state = agent_state.AgentState(
            "task-typed-F",
            tab_id="tab-F",
            server_owned=True,
            is_task_active=True,
        )
        state.user_answer_queue = queue.Queue(maxsize=1)
        agent_state.register(state)
        self.server.printer.subscribe_tab("task-typed-F", "tab-F")
        second_question = "And the signing secret?"

        def ask_twice() -> None:
            self.server.printer._thread_local.task_id = "task-typed-F"
            self.server.printer._thread_local.stop_event = self.stop
            try:
                first = self.server._ask_user_question(_QUESTION)
                second = self.server._ask_user_question(second_question)
            except KeyboardInterrupt:
                first = second = "<interrupted>"
            self.result["answer"] = first
            self.result["second"] = second
            self.done.set()

        self.waiter = threading.Thread(target=ask_twice, daemon=True)
        state.task_thread = self.waiter
        self.waiter.start()
        self._wait_for_event("askUser")
        self.server._handle_command({
            "type": "appendUserMessage", "tabId": "tab-F", "prompt": _REPLY,
        })
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and len(self._events("askUser")) < 2:
            time.sleep(0.01)
        kinds = [
            ev.get("type") for ev in self.printer.emitted
            if ev.get("type") in ("askUser", "askUserDone")
        ]
        self.assertEqual(kinds, ["askUser", "askUserDone", "askUser"])
        with agent_state.STATE_LOCK:
            self.assertEqual(state.pending_ask_question, second_question)
        self.server._handle_command({
            "type": "userAnswer", "tabId": "tab-F", "answer": "secret",
        })
        self.assertTrue(self.done.wait(timeout=2.0))
        self.assertEqual(self.result.get("answer"), _REPLY)
        self.assertEqual(self.result.get("second"), "secret")

    def test_task_tagged_message_keeps_followup_semantics(self) -> None:
        """A ``<task>`` block typed while a question is pending is queued, not an answer."""
        state = self._start_blocked_agent("task-typed-D", "tab-D")
        self.server._handle_command({
            "type": "appendUserMessage",
            "tabId": "tab-D",
            "prompt": "<task>run the linter afterwards</task>",
        })
        self.assertFalse(self.done.wait(timeout=0.2))
        with agent_state.STATE_LOCK:
            self.assertEqual(state.pending_ask_question, _QUESTION)
            self.assertEqual(state.queued_followup_tasks, ["run the linter afterwards"])
            self.assertEqual(state.pending_user_messages, [])
        self.assertEqual(self._events("askUserDone"), [])
        # A real answer afterwards still resolves the question.
        self.server._handle_command({
            "type": "userAnswer", "tabId": "tab-D", "answer": _REPLY,
        })
        self.assertTrue(self.done.wait(timeout=2.0))
        self.assertEqual(self.result.get("answer"), _REPLY)

    def test_message_without_pending_question_is_steering(self) -> None:
        """With no question pending a typed message stays a steering message."""
        state = agent_state.AgentState(
            "task-typed-E",
            tab_id="tab-E",
            server_owned=True,
            is_task_active=True,
        )
        state.user_answer_queue = queue.Queue(maxsize=1)
        agent_state.register(state)
        self.server.printer.subscribe_tab("task-typed-E", "tab-E")
        self.server._handle_command({
            "type": "appendUserMessage",
            "tabId": "tab-E",
            "prompt": _REPLY,
        })
        with agent_state.STATE_LOCK:
            self.assertEqual(state.pending_user_messages, [_REPLY])
        self.assertTrue(state.user_answer_queue.empty())
        self.assertEqual(self._events("askUserDone"), [])


if __name__ == "__main__":
    unittest.main()
