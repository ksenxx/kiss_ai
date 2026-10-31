# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: an answer naming a ``questionId`` answers that one ask, whatever its text.

Every ``askUser`` frame carries a ``questionId`` minted for that one
``ask_user_question`` call (a replay of the pending question re-sends
the same id), and a ``userAnswer`` naming a ``questionId`` reaches the
agent only while that ask is the pending one.  The question text alone
cannot tell two asks apart: an agent asking "Proceed?" twice in a row,
or a later run of the tab asking what an earlier run asked, took a
delayed answer to the first ask as the answer to the second.

The task's worker thread is played by the test, as in
``test_late_user_answer_sweep1030.py``: a thread blocked in the real
``_ask_user_question``.  Everything between it and the pages (routing,
the answer queue, the ``askUser`` / ``askUserDone`` frames, the
``resumeSession`` replay) is the real daemon on loopback connections.
"""

from __future__ import annotations

import queue
import threading
from functools import partial

from kiss.server import agent_state
from kiss.tests.server.two_client_harness import Client, TwoClientHarness

TASK_ID = "930301"
CHAT_ID = "chat-question-identity"
TASK_TAB = "task-tab-question-identity"
VIEWER_TAB = "viewer-tab-question-identity"
QUESTION = "Proceed?"


def _asks_at_least(client: Client, count: int) -> bool:
    """Whether *client* got at least *count* ``askUser`` frames."""
    return len(client.of_type("askUser")) >= count


class QuestionIdentityTest(TwoClientHarness):
    """A ``userAnswer`` naming a ``questionId`` reaches only that ask."""

    def setUp(self) -> None:
        super().setUp()
        self.answers: list[str] = []
        self.workers: list[tuple[threading.Thread, threading.Event]] = []
        state = agent_state.AgentState(
            TASK_ID, chat_id=CHAT_ID, tab_id=TASK_TAB, server_owned=True, is_task_active=True,
        )
        state.user_answer_queue = queue.Queue(maxsize=1)
        agent_state.register(state)
        self.server._printer.subscribe_tab(TASK_ID, TASK_TAB)

    def tearDown(self) -> None:
        for worker, stop in self.workers:
            stop.set()
            worker.join(timeout=5)
        super().tearDown()

    def _run_asking(self, *questions: str) -> tuple[threading.Thread, threading.Event]:
        """Start a worker thread of the task that asks *questions* one after another."""
        stop = threading.Event()
        worker = threading.Thread(target=self._ask, args=(questions, stop), daemon=True)
        self.workers.append((worker, stop))
        worker.start()
        return worker, stop

    def _ask(self, questions: tuple[str, ...], stop: threading.Event) -> None:
        """Play the task's worker thread: ask each question once the last is answered."""
        agent = self.server._agent_server
        agent.printer._thread_local.task_id = TASK_ID
        agent.printer._thread_local.stop_event = stop
        try:
            for question in questions:
                self.answers.append(agent._ask_user_question(question))
        except KeyboardInterrupt:
            self.answers.append("<stopped>")

    @staticmethod
    def _answer(
        client: Client, answer: str, question_id: str,
        tab_id: str = TASK_TAB, question: str = QUESTION,
    ) -> None:
        """Send *client*'s ``userAnswer`` to the ask *question_id* (whose text is *question*)."""
        client.send({
            "type": "userAnswer", "tabId": tab_id, "answer": answer,
            "question": question, "questionId": question_id,
        })

    def _ask_ids(self, client: Client, count: int) -> list[str]:
        """The ``questionId`` of each of the first *count* ``askUser`` frames *client* got."""
        self.wait_for(partial(_asks_at_least, client, count), what=f"{client.name}: {count} asks")
        ids = [str(e.get("questionId", "")) for e in client.of_type("askUser")[:count]]
        for question_id in ids:
            self.assertTrue(question_id, "every askUser names its ask")
        return ids

    def test_a_late_answer_to_an_identical_earlier_ask_is_dropped(self) -> None:
        first = self.open_client("first")
        second = self.open_client("second")
        worker, _ = self._run_asking(QUESTION, QUESTION)
        (ask_one,) = self._ask_ids(first, 1)
        self.assertEqual(self._ask_ids(second, 1), [ask_one], "one ask, one id on every page")
        self._answer(first, "yes", ask_one)
        ask_two = self._ask_ids(first, 2)[1]
        self.assertNotEqual(ask_two, ask_one, "the same text asked again is another ask")
        self.assertEqual(self._ask_ids(second, 2)[1], ask_two)

        # The second page still showed the first ask when its user
        # answered it; frames of one connection are handled in order,
        # so the late answer is processed before the real one.
        self._answer(second, "no", ask_one)
        self._answer(second, "eu-west", ask_two)
        worker.join(timeout=30)
        self.assertFalse(worker.is_alive(), "the agent never got its answers")
        self.assertEqual(self.answers, ["yes", "eu-west"])

    def test_a_late_answer_from_an_earlier_run_is_dropped(self) -> None:
        page = self.open_client("page")
        worker, stop = self._run_asking(QUESTION)
        (earlier,) = self._ask_ids(page, 1)
        stop.set()  # the run is stopped while its question is open
        worker.join(timeout=30)
        self.assertEqual(self.answers, ["<stopped>"])

        # Nothing is pending: an answer to the stopped run's ask is
        # dropped.  The page's frames are handled in order, so its
        # ``pong`` comes once the answer was.
        self._answer(page, "too late", earlier)
        page.send({"type": "ping"})
        self.wait_event(page, "pong")
        worker, _ = self._run_asking(QUESTION)  # a later run asks the same
        later = self._ask_ids(page, 2)[1]
        self.assertNotEqual(later, earlier)
        self._answer(page, "still too late", earlier)
        self._answer(page, "go", later)
        worker.join(timeout=30)
        self.assertEqual(self.answers, ["<stopped>", "go"])
        self.wait_event(page, "askUserDone")
        self.assertEqual(len(page.of_type("askUserDone")), 1, "only the delivered answer closed")

    def test_a_replayed_question_keeps_its_id(self) -> None:
        owner = self.open_client("owner")
        joiner = self.open_client("joiner")
        worker, _ = self._run_asking(QUESTION)
        (ask,) = self._ask_ids(owner, 1)
        # The joining page opens the chat in a tab of its own: the
        # replay shows it the pending question under the same id.
        joiner.send({"type": "resumeSession", "chatId": CHAT_ID, "tabId": VIEWER_TAB})
        replayed = self.wait_event(joiner, "askUser", tabId=VIEWER_TAB)
        self.assertEqual(replayed["questionId"], ask)
        self._answer(joiner, "from the viewer", ask, tab_id=VIEWER_TAB)
        worker.join(timeout=30)
        self.assertEqual(self.answers, ["from the viewer"])

    def test_an_ask_with_no_text_is_answered_by_its_id(self) -> None:
        page = self.open_client("page")
        worker, _ = self._run_asking("")
        (ask,) = self._ask_ids(page, 1)
        self._answer(page, "answered", ask, question="")
        worker.join(timeout=30)
        self.assertFalse(worker.is_alive(), "the agent never got its answer")
        self.assertEqual(self.answers, ["answered"])
