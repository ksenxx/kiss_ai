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
OTHER_TASK_ID = "930302"
OTHER_CHAT_ID = "chat-question-identity-other"
OTHER_TAB = "task-tab-question-identity-other"


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

    def _run_asking(
        self, *questions: str, task_id: str = TASK_ID, answers: list[str] | None = None,
    ) -> tuple[threading.Thread, threading.Event]:
        """Start a worker thread of task *task_id* that asks *questions* one after another.

        Its answers go to *answers* (``self.answers`` when not given).
        """
        stop = threading.Event()
        heard = self.answers if answers is None else answers
        worker = threading.Thread(
            target=self._ask, args=(questions, stop, task_id, heard), daemon=True,
        )
        self.workers.append((worker, stop))
        worker.start()
        return worker, stop

    def _ask(
        self, questions: tuple[str, ...], stop: threading.Event, task_id: str, answers: list[str],
    ) -> None:
        """Play the worker thread of *task_id*: ask each question once the last is answered."""
        agent = self.server._agent_server
        agent.printer._thread_local.task_id = task_id
        agent.printer._thread_local.stop_event = stop
        try:
            for question in questions:
                answers.append(agent._ask_user_question(question))
        except KeyboardInterrupt:
            answers.append("<stopped>")

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

    def test_a_tab_bound_to_another_chat_answers_the_question_it_shows(self) -> None:
        """What is answered or typed in a tab is for the task the tab shows.

        ``TASK_TAB`` launched the task of ``CHAT_ID``, which is asking,
        and was then resumed onto another chat, whose task is asking
        too.  The tab went on showing and answering the question of the
        task it launched and no longer shows: the answer to the shown
        question was dropped as late, and a line typed as a message or
        as a run answered the hidden one.
        """
        page = self.open_client("page")
        other = agent_state.AgentState(
            OTHER_TASK_ID, chat_id=OTHER_CHAT_ID, tab_id=OTHER_TAB,
            server_owned=True, is_task_active=True,
        )
        other.user_answer_queue = queue.Queue(maxsize=1)
        agent_state.register(other)
        self.server._printer.subscribe_tab(OTHER_TASK_ID, OTHER_TAB)
        hidden, _ = self._run_asking("Hidden?")
        launched = agent_state.get(TASK_ID)
        assert launched is not None
        # A ``run`` typed into a tab is a line for its task while that task's thread is installed.
        launched.task_thread = hidden
        shown_answers: list[str] = []
        shown, _ = self._run_asking(
            "One?", "Two?", "Three?", task_id=OTHER_TASK_ID, answers=shown_answers,
        )
        self.wait_event(page, "askUser", tabId=TASK_TAB, question="Hidden?")
        self.wait_event(page, "askUser", tabId=OTHER_TAB, question="One?")

        page.send({"type": "resumeSession", "chatId": OTHER_CHAT_ID, "tabId": TASK_TAB})
        one = self.wait_event(page, "askUser", tabId=TASK_TAB, question="One?")
        self._answer(page, "the modal's", one["questionId"], question="One?")
        self.wait_event(page, "askUser", tabId=TASK_TAB, question="Two?")
        page.send({"type": "appendUserMessage", "tabId": TASK_TAB, "prompt": "a message"})
        self.wait_event(page, "askUser", tabId=TASK_TAB, question="Three?")
        page.send({"type": "run", "tabId": TASK_TAB, "prompt": "a run"})
        shown.join(timeout=30)
        self.assertEqual(shown_answers, ["the modal's", "a message", "a run"])
        # With no question open, a message is one for the running task shown; a
        # run cannot start while the task the tab launched has the tab's state.
        page.send({"type": "appendUserMessage", "tabId": TASK_TAB, "prompt": "a note"})
        self.wait_for(lambda: other.pending_user_messages == ["a note"], what="the note queued")
        page.send({"type": "run", "tabId": TASK_TAB, "prompt": "a new task"})
        refusal = self.wait_event(page, "error", tabId=TASK_TAB, code="prompt_refused")
        self.assertIn("still running in another chat", refusal["text"])
        self.assertIs(agent_state.find_by_tab(TASK_TAB), launched)
        self.assertEqual(launched.pending_user_messages, [])
        self.assertEqual(self.answers, [], "the task the tab no longer shows was answered")
        self.assertTrue(hidden.is_alive(), "the hidden task's question is still open")

    def test_an_ask_with_no_text_is_answered_by_its_id(self) -> None:
        page = self.open_client("page")
        worker, _ = self._run_asking("")
        (ask,) = self._ask_ids(page, 1)
        self._answer(page, "answered", ask, question="")
        worker.join(timeout=30)
        self.assertFalse(worker.is_alive(), "the agent never got its answer")
        self.assertEqual(self.answers, ["answered"])
