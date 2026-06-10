# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shift+Enter / backslash-continuation newline behavior in the CLI.

Covers the steering box (raw CSI-u and modifyOtherKeys Shift+Enter
sequences insert a newline instead of submitting), the panel renderer
(newlines shown as a visible ``⏎``), and the idle REPL prompt
(trailing-backslash continuation joins lines with real newlines).
"""

from __future__ import annotations

import io
import sys
import threading

from kiss.agents.sorcar.cli_panel import clip_buf
from kiss.agents.sorcar.cli_repl import _read_line
from kiss.agents.sorcar.cli_steering import _InputBox


def _make_box() -> _InputBox:
    return _InputBox(threading.RLock(), io.StringIO())


def _feed_collect(box: _InputBox, data: bytes) -> list[str]:
    submitted: list[str] = []
    box.feed(data, submitted.append, lambda: None)
    return submitted


def test_csi_u_shift_enter_inserts_newline() -> None:
    """Kitty-protocol Shift+Enter (ESC[13;2u) adds a newline to the buffer."""
    box = _make_box()
    submitted = _feed_collect(box, b"hi\x1b[13;2uthere\r")
    assert submitted == ["hi\nthere"]


def test_modify_other_keys_shift_enter_inserts_newline() -> None:
    """xterm modifyOtherKeys Shift+Enter (ESC[27;2;13~) adds a newline."""
    box = _make_box()
    submitted = _feed_collect(box, b"a\x1b[27;2;13~b\r")
    assert submitted == ["a\nb"]


def test_other_csi_sequences_still_swallowed() -> None:
    """Arrow keys and other CSI sequences are ignored, not inserted."""
    box = _make_box()
    submitted = _feed_collect(box, b"\x1b[Ax\x1b[1;5Cy\r")
    assert submitted == ["xy"]


def test_plain_enter_still_submits() -> None:
    """Plain Enter submits the buffer unchanged."""
    box = _make_box()
    submitted = _feed_collect(box, b"plain\r")
    assert submitted == ["plain"]
    assert box.buf == ""


def test_clip_buf_renders_newline_as_symbol() -> None:
    """Newlines in the edit buffer display as a visible return symbol."""
    assert clip_buf("a\nb", 40) == "a⏎b"


def test_read_line_backslash_continuation(monkeypatch) -> None:
    """A trailing backslash continues input; parts join with newlines."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("first \\\nsecond\n"))
    line = _read_line("> ")
    assert line == "first \nsecond"


def test_read_line_double_continuation(monkeypatch) -> None:
    """Multiple trailing backslashes chain multiple continuation lines."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("a\\\nb\\\nc\n"))
    line = _read_line("> ")
    assert line == "a\nb\nc"


def test_read_line_eof_during_continuation(monkeypatch) -> None:
    """EOF while continuing returns what was typed, sans the backslash."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("dangling\\\n"))
    line = _read_line("> ")
    assert line == "dangling"


def test_read_line_without_backslash_unchanged(monkeypatch) -> None:
    """A normal line without a trailing backslash is returned as-is."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("just one line\n"))
    line = _read_line("> ")
    assert line == "just one line"
