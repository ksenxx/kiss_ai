# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a late answer to one question never answers the agent's next question.

Every surface shows the same ``askUser`` question.  When one surface
answers it and the agent asks its next question at once, another
surface's answer to the first question can reach the daemon after the
second question is already pending.  A ``userAnswer`` names the
question it answers (``question``, the text of the ``askUser`` it
closes) and the daemon delivers it only while that question is the
pending one.  Before the fix the late answer was taken as the answer
to the second question, which it also dismissed on every surface.

The asking side is the daemon's real ``_ask_user_question`` (the
callback the agent's ``ask_user_question`` tool calls) on a thread that
plays the task's worker thread, as in ``test_history_awaiting_answer.py``:
having a real model ask two questions back to back would need LLM
credentials and is not deterministic.  Everything between that thread
and the two pages (routing, the answer queue, the ``askUser`` and
``askUserDone`` frames) is the real daemon on a loopback connection.
"""

from __future__ import annotations

import queue
import threading
from functools import partial

from kiss.server import agent_state
from kiss.tests.server.two_client_harness import Client, TwoClientHarness

TASK_ID = "910301"
TASK_TAB = "task-tab-late-answer"
FIRST_QUESTION = "Deploy to staging first?"
SECOND_QUESTION = "Which region should it run in?"


def _closed_at_least(client: Client, count: int) -> bool:
    """Whether *client* got at least *count* ``askUserDone`` frames."""
    return len(client.of_type("askUserDone")) >= count


class LateUserAnswerTest(TwoClientHarness):
    """A ``userAnswer`` reaches the agent only for the question it answers."""

    def setUp(self) -> None:
        super().setUp()
        self.stop = threading.Event()
        self.answers: list[str] = []
        self.questions: tuple[str, ...] = ()
        state = agent_state.AgentState(
            TASK_ID, tab_id=TASK_TAB, server_owned=True, is_task_active=True,
        )
        state.user_answer_queue = queue.Queue(maxsize=1)
        self.worker = threading.Thread(target=self._ask_questions, daemon=True)
        state.task_thread = self.worker
        agent_state.register(state)
        self.server._printer.subscribe_tab(TASK_ID, TASK_TAB)

    def tearDown(self) -> None:
        self.stop.set()
        if self.worker.is_alive():
            self.worker.join(timeout=5)
        super().tearDown()

    def _ask_questions(self) -> None:
        """Play the task's worker thread: ask each question as soon as the last is answered."""
        agent = self.server._agent_server
        agent.printer._thread_local.task_id = TASK_ID
        agent.printer._thread_local.stop_event = self.stop
        try:
            for question in self.questions:
                self.answers.append(agent._ask_user_question(question))
        except KeyboardInterrupt:
            self.answers.append("<stopped>")

    def _answer(self, client: Client, answer: str, question: str | None) -> None:
        """Send *client*'s ``userAnswer``, naming *question* unless it is ``None``."""
        cmd = {"type": "userAnswer", "tabId": TASK_TAB, "answer": answer}
        if question is not None:
            cmd["question"] = question
        client.send(cmd)

    def _ask_events(self, client: Client) -> list[tuple[str, str]]:
        """The ``askUser`` / ``askUserDone`` frames *client* got, in order."""
        return [
            (str(e["type"]), str(e.get("question", "")))
            for e in list(client.events)
            if e.get("type") in ("askUser", "askUserDone")
        ]

    def test_a_late_answer_to_the_first_question_does_not_answer_the_second(self) -> None:
        first = self.open_client("first")
        second = self.open_client("second")
        self.questions = (FIRST_QUESTION, SECOND_QUESTION)
        self.worker.start()
        for client in (first, second):
            self.wait_event(client, "askUser", question=FIRST_QUESTION)
        self._answer(first, "yes", FIRST_QUESTION)
        for client in (first, second):
            self.wait_event(client, "askUser", question=SECOND_QUESTION)

        # The second page still showed the first question when its user
        # answered it; then its user answers the second one.  Frames of
        # one connection are handled in order, so the late answer is
        # processed before the real one.
        self._answer(second, "no", FIRST_QUESTION)
        self._answer(second, "eu-west", SECOND_QUESTION)

        self.worker.join(timeout=30)
        self.assertFalse(self.worker.is_alive(), "the agent never got its answers")
        self.assertEqual(self.answers, ["yes", "eu-west"])
        expected = [
            ("askUser", FIRST_QUESTION),
            ("askUserDone", ""),
            ("askUser", SECOND_QUESTION),
            ("askUserDone", ""),
        ]
        for client in (first, second):
            self.wait_for(
                partial(_closed_at_least, client, 2),
                what=f"{client.name}: both questions closed",
            )
            self.assertEqual(self._ask_events(client), expected, client.name)

    def test_an_answer_that_names_no_question_is_delivered(self) -> None:
        """An older client's ``userAnswer`` (no ``question``) still answers."""
        page = self.open_client("page")
        self.questions = (FIRST_QUESTION,)
        self.worker.start()
        self.wait_event(page, "askUser", question=FIRST_QUESTION)
        self._answer(page, "yes", None)
        self.worker.join(timeout=30)
        self.assertEqual(self.answers, ["yes"])
        self.wait_event(page, "askUserDone")
