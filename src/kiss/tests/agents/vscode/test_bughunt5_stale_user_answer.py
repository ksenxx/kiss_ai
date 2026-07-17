# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 5: stale duplicate answer poisons the next question (BUG-5E-3).

``_cmd_user_answer`` supports the multi-viewer case: every tab viewing
a running task renders the ``askUser`` modal, and ANY of them may
answer (``_resolve_user_answer_queue`` routes a viewer's answer to the
task-owner tab's queue).  But nothing closes the OTHER viewers' modals
once one viewer has answered, and ``_cmd_user_answer`` enqueues an
answer regardless of whether a question is currently pending.  So when
viewer A answers question 1 (consumed by the agent) and viewer B then
submits its still-open modal, B's answer sits in the ``maxsize=1``
queue — and the agent's NEXT ``ask_user_question`` returns B's stale
answer for question 1 instantly, without the user ever seeing
question 2.

Fix: ``_ask_user_question`` drains the answer queue immediately before
broadcasting the new question, so only answers submitted after the
question was asked can answer it.
"""

from __future__ import annotations

import queue
import threading
import unittest
from typing import Any

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.server import VSCodeServer


class TestStaleUserAnswer(unittest.TestCase):
    """A late duplicate answer must not answer the next question."""

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

        def capture(event: dict[str, Any]) -> None:
            with self._events_lock:
                self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        # A running task owned by ``owner-tab`` with a second viewer
        # (``viewer-tab``) subscribed to the same task — the exact
        # multi-viewer setup ``_resolve_user_answer_queue`` documents.
        self.task_key = "4242"
        self.owner_tab = "owner-tab"
        self.viewer_tab = "viewer-tab"
        tab = self.server._get_tab(self.owner_tab)
        tab.stop_event = threading.Event()
        tab.user_answer_queue = queue.Queue(maxsize=1)
        tab.is_task_active = True
        self.server.printer.subscribe_tab(self.task_key, self.owner_tab)
        self.server.printer.subscribe_tab(self.task_key, self.viewer_tab)
        self.tab = tab

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def _ask_on_worker(self, question: str) -> tuple[
        threading.Thread, list[str], threading.Event,
    ]:
        """Run ``_ask_user_question`` on a fake agent thread."""
        answers: list[str] = []
        done = threading.Event()
        stop_event = self.tab.stop_event

        def worker() -> None:
            tl = self.server.printer._thread_local
            tl.task_id = self.task_key
            tl.stop_event = stop_event
            try:
                answers.append(self.server._ask_user_question(question))
            except KeyboardInterrupt:
                answers.append("<interrupted>")
            finally:
                done.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        return t, answers, done

    def test_late_duplicate_answer_does_not_answer_next_question(
        self,
    ) -> None:
        # Question 1: viewer A answers; the agent consumes it.
        t1, answers1, done1 = self._ask_on_worker("Q1: proceed?")
        self._wait_for_ask_user(count=1)
        self.server._cmd_user_answer({
            "type": "userAnswer", "tabId": self.owner_tab, "answer": "yes",
        })
        assert done1.wait(timeout=10)
        t1.join(timeout=10)
        assert answers1 == ["yes"]

        # Viewer B's modal for Q1 is still open (nothing closed it).
        # B submits its answer late — AFTER the agent already consumed
        # A's answer.  It lands in the owner tab's queue.
        self.server._cmd_user_answer({
            "type": "userAnswer", "tabId": self.viewer_tab,
            "answer": "stale-answer-for-Q1",
        })

        # Question 2 must NOT be answered by B's stale Q1 answer.
        t2, answers2, done2 = self._ask_on_worker("Q2: which file?")
        self._wait_for_ask_user(count=2)
        finished_instantly = done2.wait(timeout=1.0)
        if finished_instantly:
            assert answers2 and answers2[0] != "stale-answer-for-Q1", (
                "BUG: ask_user_question returned the stale multi-viewer "
                "answer submitted for the PREVIOUS question — the user "
                f"never saw Q2 (got {answers2!r})"
            )
        # The agent is (correctly) still waiting: deliver a real answer.
        self.server._cmd_user_answer({
            "type": "userAnswer", "tabId": self.owner_tab,
            "answer": "fresh-answer-for-Q2",
        })
        assert done2.wait(timeout=10)
        t2.join(timeout=10)
        assert answers2 == ["fresh-answer-for-Q2"]

    def _wait_for_ask_user(self, count: int) -> None:
        """Block until *count* askUser events have been broadcast."""
        deadline = threading.Event()
        for _ in range(200):
            with self._events_lock:
                seen = len(
                    [e for e in self.events if e.get("type") == "askUser"],
                )
            if seen >= count:
                return
            deadline.wait(0.02)
        raise AssertionError(f"askUser #{count} never broadcast")


if __name__ == "__main__":
    unittest.main()
