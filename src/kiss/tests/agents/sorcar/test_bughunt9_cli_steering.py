# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 9: three real issues found in ``cli_steering.py``.

1. ``SteeringSession.ask_user_question`` stale-answer race.  The pump
   thread's ``_on_submit`` checks ``_question_pending`` and enqueues
   into ``_answer_q``; the worker's ``finally`` clears the flag only
   AFTER ``get()`` returns.  A line submitted in that window is parked
   in the (maxsize=1) queue with no waiter, and the NEXT
   ``ask_user_question`` call returned it instantly — before the user
   even saw the new question.  Fix: drain leftover answers before
   arming ``_question_pending``.

2. ``_InputBox._menu_h`` auto-dismiss (terminal too small to fit even
   one menu row) reset ``_menu_open`` / ``_menu_items`` / ``_menu_sel``
   / ``_menu_scroll`` but left ``_menu_repls`` populated — inconsistent
   with ``_reset_completion_state``, the documented single source of
   truth for tearing down menu state.  Fix: clear ``_menu_repls`` too.

3. ``_normalize_candidates`` stripped only TRAILING newlines while its
   contract promises a candidate "can never break the single-row menu
   rendering".  An EMBEDDED ``\\n`` in a display string survived and,
   when the row was written at an absolute position, pushed the rest of
   the row onto the next screen line — corrupting the menu/box layout.
   Fix: flatten embedded line breaks in display text to spaces.
"""

from __future__ import annotations

import io
import threading

from kiss.agents.sorcar.cli_steering import (
    SteeringSession,
    _InputBox,
    _normalize_candidates,
)
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent


def _make_session(chat_id: str) -> tuple[SteeringSession, _RunningAgentState]:
    """Build a real SteeringSession the way the CLI does (no PTY needed)."""
    agent = SorcarAgent("bh9-steering")
    state = _RunningAgentState(chat_id, "", agent=None)
    state.chat_id = chat_id
    state.is_task_active = True
    session = SteeringSession(agent, state, chat_id)
    return session, state


class TestAskUserStaleAnswer:
    def test_next_question_ignores_answer_left_by_previous_race(self) -> None:
        """A leftover answer from the submit/clear race must not be
        served as the next question's answer.

        The leftover state is produced through the REAL code path: the
        pump thread's ``_on_submit`` runs while ``_question_pending``
        is still set (the window between the previous question's
        ``get()`` returning and its ``finally`` clearing the flag), so
        the line lands in ``_answer_q`` with no waiter to consume it.
        """
        session, _state = _make_session("bh9-ask-chat")
        # --- reproduce the race window deterministically ---
        session._question_pending.set()
        session._on_submit("stale answer from previous question")
        session._question_pending.clear()
        # --- next question must NOT return the stale leftover ---
        answers: list[str] = []

        def _worker() -> None:
            answers.append(session.ask_user_question("What now?"))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        # The question must still be pending (i.e. NOT answered by the
        # stale leftover) so the user's real submit is what resolves it.
        assert session._question_pending.wait(timeout=5.0), (
            "ask_user_question resolved before any user submit — it"
            " consumed the stale leftover answer"
        )
        session._on_submit("real answer")
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert answers == ["real answer"]

    def test_fresh_question_still_gets_normal_answer(self) -> None:
        """Regression guard: the drain must not break the normal flow."""
        session, _state = _make_session("bh9-ask-chat2")
        answers: list[str] = []

        def _worker() -> None:
            answers.append(session.ask_user_question("Pick one"))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        assert session._question_pending.wait(timeout=5.0)
        session._on_submit("blue")
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert answers == ["blue"]
        # Submits after the question completes go to the steering queue.
        session._on_submit("follow-up instruction")
        assert _state.pending_user_messages == ["follow-up instruction"]


class TestMenuAutoDismissClearsReplacements:
    def test_auto_dismiss_resets_all_menu_state(self) -> None:
        """When the terminal has no room for the menu, the auto-dismiss
        in ``_menu_h`` must tear down ALL menu state — including the
        replacement list — exactly like ``_reset_completion_state``.
        """
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)

        def completer(buf: str) -> list[str]:
            del buf
            return ["/alpha", "/beta"]

        box.completer_fn = completer
        box.buf = "/a"
        assert box._open_completion_menu()
        assert box._menu_open
        assert box._menu_repls == ["/alpha", "/beta"]
        # Grow the buffer so tall (500+ body rows) that no terminal can
        # fit a single menu row above the box: room collapses to zero
        # and the menu must auto-dismiss completely.
        box.buf = "x" + "\n" * 500
        with box.lock:
            assert box._menu_h() == 0
        assert box._menu_open is False
        assert box._menu_items == []
        assert box._menu_sel == 0
        assert box._menu_scroll == 0
        assert box._menu_repls == []


class TestNormalizeCandidatesEmbeddedNewlines:
    def test_embedded_newlines_flattened_in_display(self) -> None:
        repls, displays = _normalize_candidates(
            # Mixed str/tuple candidates are supported at runtime even
            # though the annotation advertises homogeneous lists.
            [("task-1", "task-1: fix the\nlogin bug\n"), "plain\ncand\n"],  # type: ignore[arg-type]
        )
        assert repls == ["task-1", "plain\ncand"]
        assert displays == ["task-1: fix the login bug", "plain cand"]
        for disp in displays:
            assert "\n" not in disp
            assert "\r" not in disp

    def test_menu_rows_never_contain_line_breaks(self) -> None:
        """End-to-end through the real Tab-completion path: an opened
        menu must never hold a display row with an embedded line break
        (which would corrupt the absolutely-positioned row rendering).
        """
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)

        def completer(buf: str) -> list[tuple[str, str]]:
            del buf
            return [
                ("one", "first line\nsecond line"),
                ("two", "third\r\nfourth"),
            ]

        box.completer_fn = completer
        box.buf = "x"
        assert box._open_completion_menu()
        assert box._menu_open
        assert box._menu_items == [
            "first line second line", "third fourth",
        ]
        for item in box._menu_items:
            assert "\n" not in item and "\r" not in item
        # Accept still inserts the (unflattened) replacement text.
        box._menu_accept()
        assert box.buf == "one"
