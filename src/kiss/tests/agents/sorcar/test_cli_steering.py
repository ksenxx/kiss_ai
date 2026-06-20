# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the sorcar CLI steering input box.

These exercise the real key-parsing, message-queueing and
agent-injection paths used when a user types follow-up instructions
into the bottom input box while a task runs on the command line.  No
real terminal is required: the box's edit/parse logic and the queue ->
drain pipeline run independently of the scroll-region rendering.
"""

from __future__ import annotations

import io
import threading
from typing import Any

from kiss.agents.sorcar.cli_panel import (
    _term_size,
    body_cursor_col,
    panel_cols,
)
from kiss.agents.sorcar.cli_steering import (
    _BOX_H,
    AnchoredRepl,
    SteeringSession,
    _box_top_row,
    _InputBox,
    _StdoutProxy,
    run_with_steering,
    supports_steering,
)
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent


class _RecordingModel:
    """Minimal model recording injected messages (mirrors Model.add_*)."""

    def __init__(self) -> None:
        self.conversation: list[dict[str, str]] = []

    def add_message_to_conversation(self, role: str, content: str) -> None:
        self.conversation.append({"role": role, "content": content})


def _make_box() -> _InputBox:
    return _InputBox(threading.RLock(), io.StringIO())


class TestInputBoxFeed:
    def test_typing_builds_buffer(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"hello", submitted.append, lambda: None)
        assert box.buf == "hello"
        assert submitted == []

    def test_enter_submits_and_clears(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"fix the bug\n", submitted.append, lambda: None)
        assert submitted == ["fix the bug"]
        assert box.buf == ""

    def test_carriage_return_submits(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"do it\r", submitted.append, lambda: None)
        assert submitted == ["do it"]

    def test_backspace_deletes_last_char(self) -> None:
        box = _make_box()
        box.feed(b"abc", lambda _s: None, lambda: None)
        box.feed(b"\x7f", lambda _s: None, lambda: None)
        assert box.buf == "ab"

    def test_ctrl_u_clears_line(self) -> None:
        box = _make_box()
        box.feed(b"some text", lambda _s: None, lambda: None)
        box.feed(b"\x15", lambda _s: None, lambda: None)
        assert box.buf == ""

    def test_ctrl_c_invokes_abort(self) -> None:
        box = _make_box()
        aborted: list[bool] = []
        box.feed(b"\x03", lambda _s: None, lambda: aborted.append(True))
        assert aborted == [True]

    def test_escape_sequences_ignored(self) -> None:
        box = _make_box()
        # Up arrow then "x" — the CSI sequence must be swallowed.
        box.feed(b"\x1b[Ax", lambda _s: None, lambda: None)
        assert box.buf == "x"

    def test_multiple_lines_in_one_chunk(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"one\ntwo\n", submitted.append, lambda: None)
        assert submitted == ["one", "two"]
        assert box.buf == ""


class TestInputBoxRender:
    def test_draw_renders_box_and_scroll_safe_markers(self) -> None:
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box._active = True
        box.buf = "do something"
        box.status = " queued: 1 "
        box.redraw()
        text = out.getvalue()
        # Rounded border glyphs and the input marker are present.
        assert "╭" in text and "╮" in text
        assert "╰" in text and "╯" in text
        # The cyan chevron precedes the typed text (separated by ANSI
        # codes), so assert both the marker and the text appear.
        assert "› " in text
        assert "do something" in text
        assert "queued: 1" in text
        # No static caret glyph is drawn — the real blinking cursor is
        # parked right after the typed text instead.
        assert "▏" not in text
        col = body_cursor_col("do something", panel_cols())
        assert f";{col}H" in text

    def test_draw_shows_placeholder_when_empty(self) -> None:
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box._active = True
        box.redraw()
        text = out.getvalue()
        assert "Add an instruction" in text

    def test_draw_parks_caret_after_chevron_when_empty(self) -> None:
        # With an empty buffer the caret sits immediately after the ``› ``
        # chevron, exactly like the idle sorcar prompt's blinking cursor.
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box._active = True
        box.redraw()
        text = out.getvalue()
        col = body_cursor_col("", panel_cols())
        assert f";{col}H" in text


class TestStdoutProxyCaret:
    def test_output_restores_then_reparks_caret(self) -> None:
        # While the box is active, agent output is wrapped so it lands in
        # the scroll region (ESC8 restore / ESC7 save) and the blinking
        # caret is returned to the box body afterwards.
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box._active = True
        box.buf = "hi"
        proxy = _StdoutProxy(out, box.lock, box)
        proxy.write("agent says hello")
        text = out.getvalue()
        assert "\x1b8" in text  # restore output position
        assert "agent says hello" in text
        assert "\x1b7" in text  # re-save advanced output position
        col = body_cursor_col("hi", panel_cols())
        assert f";{col}H" in text  # caret re-parked in body

    def test_output_is_plain_when_box_inactive(self) -> None:
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        proxy = _StdoutProxy(out, box.lock, box)
        proxy.write("plain text")
        text = out.getvalue()
        assert text == "plain text"

    def test_redraw_noop_when_inactive(self) -> None:
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box.redraw()
        assert out.getvalue() == ""

    def test_stop_noop_when_inactive(self) -> None:
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box.stop()  # must not raise or touch termios
        assert out.getvalue() == ""


class TestSteeringQueue:
    def _session(self) -> tuple[SteeringSession, _RunningAgentState, SorcarAgent]:
        agent = SorcarAgent("steer-test")
        chat_id = "steer-chat-1"
        state = _RunningAgentState(chat_id, "", agent=None)
        state.chat_id = chat_id
        state.is_task_active = True
        session = SteeringSession(agent, state, chat_id)
        return session, state, agent

    def test_submit_queues_instruction(self) -> None:
        session, state, _ = self._session()
        session._on_submit("please add tests")
        assert state.pending_user_messages == ["please add tests"]

    def test_submit_strips_and_skips_blank(self) -> None:
        session, state, _ = self._session()
        session._on_submit("   ")
        session._on_submit("  real one  ")
        assert state.pending_user_messages == ["real one"]

    def test_queued_count_reflected_in_status(self) -> None:
        session, _state, _ = self._session()
        session._on_submit("a")
        session._on_submit("b")
        assert "2" in session.box.status

    def test_queue_then_agent_drain_injects_messages(self) -> None:
        """End-to-end: a queued line is injected before the next step."""
        session, state, agent = self._session()
        agent._tab_id = state.chat_id  # type: ignore[attr-defined]
        _RunningAgentState.register(state.chat_id, state)
        try:
            session._on_submit("steer left")
            session._on_submit("then steer right")
            model = _RecordingModel()
            agent._drain_pending_user_messages(model)
        finally:
            _RunningAgentState.unregister(state.chat_id)
        assert [m["content"] for m in model.conversation] == [
            "steer left",
            "then steer right",
        ]
        # Drained messages are not re-injected on a second drain.
        assert state.pending_user_messages == []

    def test_ask_user_question_receives_submitted_line(self) -> None:
        session, _state, _ = self._session()
        answers: list[str] = []
        done = threading.Event()

        def asker() -> None:
            answers.append(session.ask_user_question("Which file?"))
            done.set()

        t = threading.Thread(target=asker)
        t.start()
        # Wait until the worker is parked waiting for an answer.
        assert session._question_pending.wait(timeout=5)
        session._on_submit("src/main.py")
        assert done.wait(timeout=5)
        assert answers == ["src/main.py"]
        # The answered line must NOT be queued as a steering instruction.
        assert session.state.pending_user_messages == []


class _FakeRunAgent:
    """Real stand-in agent whose only behaviour is returning a result."""

    def __init__(self) -> None:
        self.called_with: dict[str, Any] | None = None

    def run(self, **kwargs: Any) -> str:
        self.called_with = kwargs
        return "summary: done\n"


class TestRunWithSteeringFallback:
    def test_supports_steering_false_without_tty(self) -> None:
        # The pytest process has no controlling TTY on stdin/stdout.
        assert supports_steering() is False

    def test_falls_back_to_plain_run(self) -> None:
        agent = _FakeRunAgent()
        result = run_with_steering(agent, {"prompt_template": "hi"})  # type: ignore[arg-type]
        assert result == "summary: done\n"
        assert agent.called_with == {"prompt_template": "hi"}


class TestInputBoxHistory:
    """Up/Down arrow recall through ``_InputBox.history``."""

    def test_up_arrow_recalls_last(self) -> None:
        box = _make_box()
        box.history = ["first", "second"]
        box.feed(b"\x1b[A", lambda _s: None, lambda: None)
        assert box.buf == "second"

    def test_up_twice_goes_further_back(self) -> None:
        box = _make_box()
        box.history = ["first", "second"]
        box.feed(b"\x1b[A\x1b[A", lambda _s: None, lambda: None)
        assert box.buf == "first"

    def test_down_returns_to_draft(self) -> None:
        box = _make_box()
        box.history = ["first", "second"]
        box.feed(b"draft", lambda _s: None, lambda: None)
        box.feed(b"\x1b[A", lambda _s: None, lambda: None)
        assert box.buf == "second"
        box.feed(b"\x1b[B", lambda _s: None, lambda: None)
        assert box.buf == "draft"

    def test_typing_resets_history_pointer(self) -> None:
        box = _make_box()
        box.history = ["first"]
        box.feed(b"\x1b[A", lambda _s: None, lambda: None)
        assert box.buf == "first"
        box.feed(b"x", lambda _s: None, lambda: None)
        # After editing, Up should restart at the latest history entry,
        # not advance further back from the old pointer.
        assert box.buf == "firstx"
        box.feed(b"\x1b[A", lambda _s: None, lambda: None)
        assert box.buf == "first"


class TestInputBoxCompletionMenu:
    """In-place completion menu opened by Tab.

    With two or more candidates Tab opens a menu drawn above the input
    box; ``buf`` stays at what the user typed until they select with
    Enter.  With exactly one candidate Tab accepts it directly (no
    menu).  Without a completer Tab is silently dropped.
    """

    def test_tab_opens_menu_with_first_selected(self) -> None:
        box = _make_box()
        box.completer_fn = lambda _buf: ["/help ", "/clear "]
        box.feed(b"/he", lambda _s: None, lambda: None)
        box.feed(b"\t", lambda _s: None, lambda: None)
        assert box._menu_open is True
        assert box._menu_items == ["/help ", "/clear "]
        assert box._menu_sel == 0
        # ``buf`` is left untouched until the user picks a candidate.
        assert box.buf == "/he"

    def test_enter_on_open_menu_accepts_does_not_submit(self) -> None:
        box = _make_box()
        box.completer_fn = lambda _buf: ["/help ", "/clear "]
        box.feed(b"/he\t", lambda _s: None, lambda: None)
        submitted: list[str] = []
        box.feed(b"\n", submitted.append, lambda: None)
        # First Enter only accepts the highlighted candidate.
        assert submitted == []
        assert box.buf == "/help "
        assert box._menu_open is False
        # Next Enter submits the now-completed line.
        box.feed(b"\n", submitted.append, lambda: None)
        assert submitted == ["/help "]

    def test_tab_advances_selection_when_menu_open(self) -> None:
        box = _make_box()
        box.completer_fn = lambda _buf: ["/help ", "/clear "]
        box.feed(b"/", lambda _s: None, lambda: None)
        box.feed(b"\t", lambda _s: None, lambda: None)
        box.feed(b"\t", lambda _s: None, lambda: None)
        assert box._menu_open is True
        assert box._menu_sel == 1
        # ``buf`` only changes once Enter accepts the new selection.
        assert box.buf == "/"
        submitted: list[str] = []
        box.feed(b"\n", submitted.append, lambda: None)
        assert box.buf == "/clear "
        assert submitted == []

    def test_down_arrow_navigates_menu(self) -> None:
        box = _make_box()
        box.completer_fn = lambda _buf: ["a", "b", "c"]
        box.feed(b"\t", lambda _s: None, lambda: None)
        assert box._menu_sel == 0
        box.feed(b"\x1b[B", lambda _s: None, lambda: None)
        assert box._menu_sel == 1
        box.feed(b"\x1b[B", lambda _s: None, lambda: None)
        assert box._menu_sel == 2
        # Past the last item wraps back to the first.
        box.feed(b"\x1b[B", lambda _s: None, lambda: None)
        assert box._menu_sel == 0

    def test_up_arrow_navigates_menu_backwards(self) -> None:
        box = _make_box()
        box.completer_fn = lambda _buf: ["a", "b", "c"]
        box.feed(b"\t", lambda _s: None, lambda: None)
        # Up from idx 0 wraps to the last candidate.
        box.feed(b"\x1b[A", lambda _s: None, lambda: None)
        assert box._menu_sel == 2

    def test_shift_tab_moves_selection_up(self) -> None:
        box = _make_box()
        box.completer_fn = lambda _buf: ["a", "b", "c"]
        box.feed(b"\t\t", lambda _s: None, lambda: None)
        assert box._menu_sel == 1
        # CSI Z (Shift+Tab) goes back.
        box.feed(b"\x1b[Z", lambda _s: None, lambda: None)
        assert box._menu_sel == 0

    def test_typing_closes_menu_and_appends(self) -> None:
        box = _make_box()
        box.completer_fn = lambda _buf: ["alpha", "beta"]
        box.feed(b"\t", lambda _s: None, lambda: None)
        assert box._menu_open is True
        box.feed(b"x", lambda _s: None, lambda: None)
        assert box._menu_open is False
        assert box.buf == "x"

    def test_backspace_with_buffer_closes_menu(self) -> None:
        box = _make_box()
        box.completer_fn = lambda _buf: ["alpha", "beta"]
        box.feed(b"ab\t", lambda _s: None, lambda: None)
        assert box._menu_open is True
        box.feed(b"\x7f", lambda _s: None, lambda: None)
        assert box._menu_open is False
        assert box.buf == "a"

    def test_backspace_with_empty_buffer_dismisses_menu(self) -> None:
        box = _make_box()
        box.completer_fn = lambda _buf: ["alpha", "beta"]
        box.feed(b"\t", lambda _s: None, lambda: None)
        assert box._menu_open is True
        box.feed(b"\x7f", lambda _s: None, lambda: None)
        assert box._menu_open is False
        assert box.buf == ""

    def test_ctrl_g_cancels_menu(self) -> None:
        box = _make_box()
        box.completer_fn = lambda _buf: ["alpha", "beta"]
        box.feed(b"\t", lambda _s: None, lambda: None)
        box.feed(b"\x07", lambda _s: None, lambda: None)
        assert box._menu_open is False
        assert box.buf == ""

    def test_ctrl_c_dismisses_menu_without_aborting(self) -> None:
        box = _make_box()
        box.completer_fn = lambda _buf: ["alpha", "beta"]
        box.feed(b"\t", lambda _s: None, lambda: None)
        aborted: list[bool] = []
        box.feed(b"\x03", lambda _s: None, lambda: aborted.append(True))
        assert box._menu_open is False
        assert aborted == []
        # A second Ctrl+C with the menu now closed propagates as abort.
        box.feed(b"\x03", lambda _s: None, lambda: aborted.append(True))
        assert aborted == [True]

    def test_single_candidate_replaces_buf_no_menu(self) -> None:
        box = _make_box()
        box.completer_fn = lambda _buf: ["/help "]
        box.feed(b"/h\t", lambda _s: None, lambda: None)
        assert box.buf == "/help "
        assert box._menu_open is False

    def test_no_completer_drops_tab(self) -> None:
        box = _make_box()
        box.feed(b"a\tb", lambda _s: None, lambda: None)
        # Tab without a completer is silently dropped — not typed.
        assert box.buf == "ab"
        assert box._menu_open is False

    def test_empty_candidate_list_keeps_menu_closed(self) -> None:
        box = _make_box()
        box.completer_fn = lambda _buf: []
        box.feed(b"\t", lambda _s: None, lambda: None)
        assert box._menu_open is False
        assert box.buf == ""

    def test_menu_renders_candidates_above_box(self) -> None:
        """Integration test reproducing the original bug.

        Before the fix, pressing Tab with multiple candidates silently
        cycled ``buf`` through them — *no menu was painted* and the
        input box stayed pinned to the bottom with the user unable to
        see the alternatives.  After the fix, the candidates must
        appear stacked above the input box (which has moved up to
        make room for them), with the first one highlighted with the
        ``❯`` marker.
        """
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box.completer_fn = lambda _buf: ["/help ", "/history "]
        box._active = True
        box.feed(b"/", lambda _s: None, lambda: None)
        # Snapshot the rendered output after the menu opens.
        out.seek(0)
        out.truncate(0)
        box.feed(b"\t", lambda _s: None, lambda: None)
        text = out.getvalue()
        # Both candidates must appear in the painted output.
        assert "/help" in text, "first candidate missing from menu render"
        assert "/history" in text, "second candidate missing from menu render"
        # The highlighted candidate carries the ❯ marker.
        assert "❯" in text, "no selection marker drawn on highlighted row"
        # The input box itself was moved up by the menu height (2 rows).
        rows, _ = _term_size()
        menu_top = _box_top_row(rows, _BOX_H + 2)
        # Menu row 1, menu row 2, then the box's top border at menu_top+2.
        assert f"\x1b[{menu_top};1H" in text
        assert f"\x1b[{menu_top + 1};1H" in text
        assert f"\x1b[{menu_top + 2};1H" in text
        # Scroll region was shrunk to leave room for the menu.
        assert f"\x1b[1;{max(rows - (_BOX_H + 2), 1)}r" in text
        # buf stays as what the user typed until they accept.
        assert box.buf == "/"
        assert box._menu_open is True

    def test_menu_close_clears_menu_rows_in_scroll_region(self) -> None:
        """Closing the menu must clear rows that re-join the scroll region.

        Otherwise leftover candidate glyphs would linger at the bottom
        of the agent output area after the user dismisses the menu.
        """
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box.completer_fn = lambda _buf: ["one", "two", "three"]
        box._active = True
        box.feed(b"\t", lambda _s: None, lambda: None)
        # Drop the open-menu render; we only care about what is emitted
        # by the close that follows.
        out.seek(0)
        out.truncate(0)
        box.feed(b"\x07", lambda _s: None, lambda: None)
        text = out.getvalue()
        rows, _ = _term_size()
        # The rows that *were* menu rows must have been cleared (each
        # gets an ESC[<r>;1H followed by ESC[2K) before the scroll
        # region is restored to the unreserved height.
        old_top = _box_top_row(rows, _BOX_H + 3)
        for r in range(old_top, old_top + 3):
            assert f"\x1b[{r};1H\x1b[2K" in text, f"row {r} not cleared"
        # And the DECSTBM region is widened back to the no-menu height.
        assert f"\x1b[1;{max(rows - _BOX_H, 1)}r" in text


class TestInputBoxCompletionMenuEdgeCases:
    """Edge cases for the in-place completion menu (gpt-5.1 review)."""

    def test_candidate_ansi_escapes_sanitised_in_render(self) -> None:
        """A candidate containing an ANSI SGR sequence must not emit
        its raw ESC bytes into the painted menu row.  Otherwise a
        malicious / corrupted candidate could repaint the terminal.
        """
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box.completer_fn = lambda _buf: [
            "\x1b[31mred\x1b[0m candidate",
            "plain candidate",
        ]
        box._active = True
        box.feed(b"\t", lambda _s: None, lambda: None)
        text = out.getvalue()
        # The candidate payload contained an ESC byte; menu_row must
        # have stripped it so the terminal cannot interpret the SGR
        # change as a real colour escape.
        assert "\x1b[31m" not in text, (
            "raw candidate ANSI escape leaked into terminal output"
        )
        assert "\x1b[0m red" not in text, (
            "raw candidate reset escape leaked into terminal output"
        )
        # The "second candidate" must still render normally.
        assert "plain candidate" in text

    def test_stop_while_menu_open_resets_menu_state(self) -> None:
        """stop() must leave the box with no lingering menu state, so a
        later start() does not paint phantom menu rows nor mistakenly
        trip the "menu shrank" clear path in _draw_locked.
        """
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box.completer_fn = lambda _buf: ["alpha", "beta"]
        # Forge an active start state without a real tty.
        box._active = True
        box._rows, _ = _term_size()
        box.feed(b"\t", lambda _s: None, lambda: None)
        assert box._menu_open is True
        # stop() needs termios for tcsetattr; short-circuit by faking
        # the saved tty state to None (the call early-returns the
        # tcsetattr block).
        box._old_term = None
        box.stop()
        assert box._menu_open is False
        assert box._menu_items == []
        assert box._drawn_menu_h == 0

    def test_bracketed_paste_dismisses_open_menu(self) -> None:
        box = _make_box()
        box.completer_fn = lambda _buf: ["one", "two"]
        box.feed(b"\t", lambda _s: None, lambda: None)
        assert box._menu_open is True
        # Bracketed paste sequence with content.
        box.feed(
            b"\x1b[200~hello\x1b[201~",
            lambda _s: None,
            lambda: None,
        )
        assert box._menu_open is False
        assert box.buf.endswith("hello")

    def test_menu_h_auto_dismisses_when_no_room(
        self, monkeypatch: Any,
    ) -> None:
        """On a terminal too small to fit even one menu row above the
        box, ``_menu_h`` must auto-dismiss the menu so the next arrow
        / Enter is not silently consumed by an invisible widget.
        """
        box = _make_box()
        box.completer_fn = lambda _buf: ["a", "b"]
        box.feed(b"\t", lambda _s: None, lambda: None)
        assert box._menu_open is True
        # Force a tiny terminal: rows - _BOX_H - 1 == 0 → no room.
        monkeypatch.setattr(
            "kiss.agents.sorcar.cli_steering._term_size",
            lambda: (_BOX_H + 1, 80),
        )
        assert box._menu_h() == 0
        assert box._menu_open is False
        assert box._menu_items == []

    def test_ask_user_question_dismisses_open_menu(self) -> None:
        """Switching the box into ask-user-question mode must close
        any open completion menu so the first Enter submits the
        answer rather than picking a stale candidate.
        """
        agent = SorcarAgent("steer-test")
        state = _RunningAgentState("chat-id", "", agent=None)
        box = _make_box()
        session = SteeringSession(
            agent, state, "chat-id", box=box,
        )
        box.completer_fn = lambda _buf: ["one", "two"]
        box.feed(b"\t", lambda _s: None, lambda: None)
        assert box._menu_open is True

        answers: list[str] = []

        def ask() -> None:
            answers.append(session.ask_user_question("pick?"))

        worker = threading.Thread(target=ask, daemon=True)
        worker.start()
        # Wait until the worker has parked on the queue.
        assert session._question_pending.wait(timeout=2.0)
        # By the time the title is flipped, the menu must be closed.
        assert box._menu_open is False
        # Unblock the worker and finish.
        session._answer_q.put("done")
        worker.join(timeout=2.0)
        assert answers == ["done"]


class TestInputBoxEOF:
    """Ctrl+D semantics: empty buffer → on_eof; non-empty → no-op."""

    def test_ctrl_d_empty_invokes_eof(self) -> None:
        box = _make_box()
        eofs: list[bool] = []
        box.feed(
            b"\x04",
            lambda _s: None,
            lambda: None,
            lambda: eofs.append(True),
        )
        assert eofs == [True]

    def test_ctrl_d_with_buffer_does_nothing(self) -> None:
        box = _make_box()
        eofs: list[bool] = []
        box.feed(
            b"abc\x04",
            lambda _s: None,
            lambda: None,
            lambda: eofs.append(True),
        )
        assert box.buf == "abc"
        assert eofs == []


class TestSteeringSessionSharedBox:
    """When a box is passed in, the session must not own its lifecycle."""

    def test_shared_box_marks_session_as_non_owner(self) -> None:
        box = _make_box()
        agent = SorcarAgent("steer-test")
        state = _RunningAgentState("c", "", agent=None)
        state.chat_id = "c"
        state.is_task_active = True
        session = SteeringSession(
            agent, state, "c",
            box=box,
            lock=threading.RLock(),
            real_stdout=io.StringIO(),
            real_stderr=io.StringIO(),
        )
        assert session._owns_box is False
        assert session.box is box

    def test_default_session_owns_box(self) -> None:
        agent = SorcarAgent("steer-test")
        state = _RunningAgentState("c", "", agent=None)
        state.chat_id = "c"
        state.is_task_active = True
        session = SteeringSession(agent, state, "c")
        assert session._owns_box is True


class TestAnchoredReplWiring:
    """Bare wiring tests for :class:`AnchoredRepl` without a real TTY.

    A full end-to-end test would require driving termios and a fake
    PTY; here we only exercise the wiring (completer/history plumbing)
    that does not need an active box.
    """

    def test_constructor_seeds_history_and_completer(self) -> None:
        def completer(_buf: str) -> list[str]:
            return []

        repl = AnchoredRepl(
            completer_fn=completer, history=["one", "two"],
        )
        assert repl.box.history == ["one", "two"]
        assert repl.box.completer_fn is completer

    def test_constructor_handles_no_history(self) -> None:
        repl = AnchoredRepl()
        assert repl.box.history == []
        assert repl.box.completer_fn is None


class TestRunSteeringLoop:
    """Wiring tests for :meth:`AnchoredRepl.run_steering_loop`.

    A full TTY-driven test would require a PTY pair + termios; here
    we only exercise the title/status flip + restore around the loop
    using ``is_done`` that returns True immediately so the select
    loop never blocks on stdin.  ``sys.stdin`` is replaced with the
    read end of a real OS pipe so :func:`sys.stdin.fileno` works
    under pytest's stdin capture.
    """

    def _build_repl(self) -> AnchoredRepl:
        repl = AnchoredRepl()
        # Replace the real stdout-bound box with one writing to a
        # StringIO so redraw() does not touch the terminal.
        repl.box = _InputBox(repl.lock, io.StringIO())
        return repl

    def _swap_stdin_pipe(self) -> tuple[int, int, Any]:
        """Replace ``sys.stdin`` with a real-pipe-backed file object.

        Returns ``(read_fd, write_fd, saved_stdin)`` so the caller can
        restore stdin and close both ends in ``finally``.
        """
        import os as _os
        import sys as _sys

        r, w = _os.pipe()
        saved = _sys.stdin
        _sys.stdin = _os.fdopen(r, "r", closefd=False)
        return r, w, saved

    def _restore_stdin(self, r: int, w: int, saved: Any) -> None:
        import os as _os
        import sys as _sys

        try:
            _sys.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        _sys.stdin = saved
        for fd in (r, w):
            try:
                _os.close(fd)
            except OSError:
                pass

    def test_loop_flips_title_to_steer_and_restores(self) -> None:
        from kiss.agents.sorcar.cli_panel import IDLE_TITLE, STEER_TITLE

        repl = self._build_repl()
        repl.box.title = IDLE_TITLE
        repl.box.status = "prev-status"
        seen_title: list[str] = []

        def is_done() -> bool:
            if not seen_title:
                seen_title.append(repl.box.title)
            return True

        r, w, saved = self._swap_stdin_pipe()
        try:
            repl.run_steering_loop(
                on_submit=lambda _s: None,
                on_abort=lambda: None,
                is_done=is_done,
            )
        finally:
            self._restore_stdin(r, w, saved)
        assert seen_title == [STEER_TITLE]
        assert repl.box.title == IDLE_TITLE
        assert repl.box.status == "prev-status"

    def test_on_idle_fired_when_no_stdin_activity(self) -> None:
        repl = self._build_repl()
        idle_calls: list[int] = []
        steps: list[int] = [0]

        def is_done() -> bool:
            steps[0] += 1
            return steps[0] > 2

        def on_idle() -> None:
            idle_calls.append(1)

        r, w, saved = self._swap_stdin_pipe()
        try:
            repl.run_steering_loop(
                on_submit=lambda _s: None,
                on_abort=lambda: None,
                is_done=is_done,
                on_idle=on_idle,
            )
        finally:
            self._restore_stdin(r, w, saved)
        assert idle_calls, "on_idle was never invoked"

    def test_on_idle_exception_does_not_crash_loop(self) -> None:
        repl = self._build_repl()
        steps: list[int] = [0]

        def is_done() -> bool:
            steps[0] += 1
            return steps[0] > 2

        def on_idle() -> None:
            raise RuntimeError("boom")

        r, w, saved = self._swap_stdin_pipe()
        try:
            repl.run_steering_loop(
                on_submit=lambda _s: None,
                on_abort=lambda: None,
                is_done=is_done,
                on_idle=on_idle,
            )
        finally:
            self._restore_stdin(r, w, saved)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
