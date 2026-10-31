# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: an ask with no text is pending on every path that looks for a pending ask.

``ask_user_question`` may be called with an empty question.  The ask
still blocks the task until an answer arrives, and its ``askUser``
frames name it by a fresh ``questionId`` (kept as the state's
``pending_ask_id``).  The paths that look for a pending ask keyed on
the question TEXT instead: a page that opened the chat got no replayed
``askUser``, a reply typed into the chat box was queued as steering
the blocked task never drains, a reply typed into a viewer tab started
a new task, the History panel showed the spinner instead of the "?",
and interrupting the ask left it open on every page (no
``askUserDone``).

The task's worker thread is played by the test, as in
``test_ask_question_identity_wave3.py``: a thread blocked in the real
``_ask_user_question``, inside a registered tool call as the agent's
own tool calls are.  Everything between it and the pages (routing, the
answer queue, the frames, the replay, the history rows, the interrupt)
is the real daemon on loopback connections; no mocks, patches or test
doubles.
"""

from __future__ import annotations

import queue
import threading

from kiss.agents.sorcar import persistence as th
from kiss.core.tool_interrupt import (
    ToolCallInterrupted,
    end_tool_call,
    new_tool_call,
    register_tool_call,
    unregister_tool_call,
)
from kiss.server import agent_state
from kiss.tests.server.two_client_harness import Client, TwoClientHarness

CHAT_ID = "chat-empty-ask"
TASK_TAB = "task-tab-empty-ask"
VIEWER_TAB = "viewer-tab-empty-ask"


class EmptyAskIsPendingTest(TwoClientHarness):
    """A task blocked on ``ask_user_question("")``."""

    def setUp(self) -> None:
        super().setUp()
        self.answers: list[str] = []
        self.stop = threading.Event()
        self.task_id, _ = th._add_task("a task that asks nothing in words", chat_id=CHAT_ID)
        self.state = agent_state.AgentState(
            self.task_id, chat_id=CHAT_ID, tab_id=TASK_TAB, server_owned=True,
            is_task_active=True,
        )
        self.state.user_answer_queue = queue.Queue(maxsize=1)
        self.worker = threading.Thread(target=self._ask_in_a_tool_call, daemon=True)
        self.state.task_thread = self.worker
        agent_state.register(self.state)
        self.server._printer.subscribe_tab(self.task_id, TASK_TAB)

    def tearDown(self) -> None:
        self.stop.set()
        self.worker.join(timeout=5)
        super().tearDown()

    def _ask_in_a_tool_call(self) -> None:
        """Play the task's worker thread: ask an empty question in an ``ask_user_question`` call."""
        agent = self.server._agent_server
        agent.printer.adopt_thread_binding(self.task_id)
        agent.printer._thread_local.stop_event = self.stop
        token = new_tool_call("ask_user_question")
        register_tool_call(token)
        try:
            self.answers.append(agent._ask_user_question(""))
            end_tool_call(token)
        except ToolCallInterrupted:
            self.answers.append("<interrupted>")
        except KeyboardInterrupt:
            self.answers.append("<stopped>")
        finally:
            unregister_tool_call(token)

    def _start_asking(self, page: Client) -> str:
        """Start the worker; return the ``questionId`` of the ask *page* is shown."""
        self.worker.start()
        ask = self.wait_event(page, "askUser", tabId=TASK_TAB)
        self.assertEqual(ask["question"], "")
        self.assertTrue(ask["questionId"], "the empty ask is named like any other")
        return str(ask["questionId"])

    def _answered(self, answer: str) -> None:
        """Assert the worker got *answer*, delivered as the answer and not as steering."""
        self.worker.join(timeout=30)
        self.assertFalse(self.worker.is_alive(), "the reply never reached the waiting ask")
        self.assertEqual(self.answers, [answer])
        self.assertEqual(self.state.pending_user_messages, [], "the reply is no steering")

    def test_a_page_opening_the_chat_is_shown_the_ask_and_answers_from_its_tab(self) -> None:
        owner = self.open_client("owner")
        joiner = self.open_client("joiner")
        ask = self._start_asking(owner)
        joiner.send({"type": "resumeSession", "chatId": CHAT_ID, "tabId": VIEWER_TAB})
        replayed = self.wait_event(joiner, "askUser", timeout=10, tabId=VIEWER_TAB)
        self.assertEqual((replayed["question"], replayed["questionId"]), ("", ask))
        # Typed into the viewer tab's chat box: the answer, not a new task.
        joiner.send({"type": "run", "tabId": VIEWER_TAB, "prompt": "from the viewer tab"})
        self._answered("from the viewer tab")
        self.assertIsNone(agent_state.find_by_tab(VIEWER_TAB), "no task started in the viewer")

    def test_a_reply_typed_into_the_chat_box_answers_the_ask(self) -> None:
        owner = self.open_client("owner")
        self._start_asking(owner)
        owner.send({"type": "appendUserMessage", "tabId": TASK_TAB, "prompt": "typed reply"})
        self._answered("typed reply")
        self.wait_event(owner, "askUserDone", tabId=TASK_TAB)

    def test_the_history_row_shows_the_ask(self) -> None:
        owner = self.open_client("owner")
        self._start_asking(owner)
        owner.send({"type": "getHistory"})
        history = self.wait_event(owner, "history")
        row = next(s for s in history["sessions"] if str(s.get("task_id")) == self.task_id)
        self.assertTrue(row["awaiting_answer"], row)

    def test_interrupting_the_ask_closes_it_on_every_page(self) -> None:
        owner = self.open_client("owner")
        other = self.open_client("other")
        self._start_asking(owner)
        owner.send({"type": "interruptTool", "tabId": TASK_TAB, "toolName": "ask_user_question"})
        self.wait_event(owner, "tool_interrupt_ack", accepted=True)
        self.wait_event(other, "askUserDone", timeout=10, tabId=TASK_TAB)
        self.worker.join(timeout=30)
        self.assertEqual(self.answers, ["<interrupted>"])
