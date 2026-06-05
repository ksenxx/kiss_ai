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

from kiss.agents.sorcar.cli_panel import body_cursor_col, panel_cols
from kiss.agents.sorcar.cli_steering import (
    SteeringSession,
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


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
