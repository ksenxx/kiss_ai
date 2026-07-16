# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: Up/Down arrows navigate lines and task history.

Reproduces the user-reported gap where the Up and Down arrow keys did
not move the cursor between lines of a multi-line buffer in the sorcar
CLI REPL input box (the anchored
:class:`~kiss.ui.cli.cli_steering._InputBox` used for both the
idle prompt and mid-task steering).  The required behaviour matches the
chat webview's input textbox (``media/main.js``):

* Up cycles to the older history entry only when the textbox is empty
  or a history browse is already in progress; otherwise it moves the
  caret up one line (to the start of the text on the first line).
* Down cycles to the newer history entry only while browsing history;
  otherwise it moves the caret down one line (to the end of the text
  on the last line).
* Typing stops the history browse.

Keystrokes are fed as the raw bytes a real terminal emits and the
resulting edit buffer / cursor is asserted — no mocks.
"""

from __future__ import annotations

import io
import threading

import pytest

from kiss.ui.cli.cli_steering import _InputBox

UP = b"\x1b[A"
DOWN = b"\x1b[B"
LEFT = b"\x1b[D"
SS3_UP = b"\x1bOA"
SS3_DOWN = b"\x1bOB"
ALT_ENTER = b"\x1b\r"  # inserts a real newline into the buffer


@pytest.fixture
def box() -> _InputBox:
    return _InputBox(threading.RLock(), io.StringIO())


def _feed(box: _InputBox, data: bytes) -> list[str]:
    """Feed *data* to *box*, returning the lines submitted via Enter."""
    submitted: list[str] = []
    box.feed(data, submitted.append, lambda: pytest.fail("unexpected abort"))
    return submitted


# --- In-buffer line navigation (webview textarea caret behaviour) ---


def test_up_moves_cursor_to_previous_line(box: _InputBox) -> None:
    """Up in a multi-line buffer moves the caret up, not into history."""
    box.history = ["old task"]
    _feed(box, b"ab" + ALT_ENTER + b"cd" + UP + b"X")
    assert box.buf == "abX\ncd"
    assert box.cursor == 3


def test_down_moves_cursor_to_next_line(box: _InputBox) -> None:
    box.history = ["old task"]
    _feed(box, b"ab" + ALT_ENTER + b"cd" + UP + UP + DOWN + b"X")
    # Up, Up puts the caret at the start of line 1; Down returns to
    # column 0 of line 2.
    assert box.buf == "ab\nXcd"
    assert box.cursor == 4


def test_up_on_first_line_moves_caret_to_start(box: _InputBox) -> None:
    """Up on the first line of non-empty text never recalls history."""
    box.history = ["old task"]
    _feed(box, b"abc" + UP)
    assert box.buf == "abc"
    assert box.cursor == 0
    _feed(box, b"X")
    assert box.buf == "Xabc"


def test_down_on_last_line_moves_caret_to_end(box: _InputBox) -> None:
    box.history = ["old task"]
    _feed(box, b"abc" + LEFT + LEFT + DOWN)
    assert box.buf == "abc"
    assert box.cursor == 3
    _feed(box, b"!")
    assert box.buf == "abc!"


def test_up_down_clamp_column_to_shorter_line(box: _InputBox) -> None:
    """Moving onto a shorter line clamps the caret to that line's end."""
    _feed(box, b"ab" + ALT_ENTER + b"cdef" + UP)
    # From column 4 of "cdef" up onto "ab": clamp to column 2.
    assert box.buf == "ab\ncdef"
    assert box.cursor == 2
    _feed(box, DOWN)
    # Back down keeps column 2 within "cdef".
    assert box.cursor == 5


def test_up_down_preserve_column_across_lines(box: _InputBox) -> None:
    _feed(box, b"abcd" + ALT_ENTER + b"wxyz" + LEFT + LEFT + UP)
    # Caret at column 2 of "wxyz" moves to column 2 of "abcd".
    assert box.cursor == 2
    _feed(box, DOWN)
    assert box.cursor == 7


def test_up_down_navigate_three_line_buffer(box: _InputBox) -> None:
    _feed(box, b"one" + ALT_ENTER + b"two" + ALT_ENTER + b"three")
    # From the end of "three" (column 5) Up lands at the end of "two"
    # (column clamped to 3); a second Up reaches the end of "one".
    _feed(box, UP + b"X")
    assert box.buf == "one\ntwoX\nthree"
    _feed(box, UP + b"W")
    assert box.buf == "oneW\ntwoX\nthree"
    _feed(box, DOWN + DOWN + b"Y")
    assert box.buf == "oneW\ntwoX\nthreYe"


def test_ss3_application_mode_up_down_navigate_lines(box: _InputBox) -> None:
    """DECCKM application-cursor-mode arrows (``ESC O A/B``) also work."""
    box.history = ["old task"]
    _feed(box, b"ab" + ALT_ENTER + b"cd" + SS3_UP + b"X")
    assert box.buf == "abX\ncd"
    _feed(box, SS3_DOWN + b"Y")
    assert box.buf == "abX\ncdY"


def test_split_up_arrow_across_feed_chunks(box: _InputBox) -> None:
    """An Up sequence split across reads still navigates lines."""
    _feed(box, b"ab" + ALT_ENTER + b"cd")
    _feed(box, b"\x1b")
    _feed(box, b"[A")
    _feed(box, b"X")
    assert box.buf == "abX\ncd"


# --- History cycling (webview cycleHistoryUp/Down behaviour) ---


def test_up_on_empty_buffer_recalls_newest_history(box: _InputBox) -> None:
    box.history = ["one", "two"]  # oldest -> newest
    _feed(box, UP)
    assert box.buf == "two"
    assert box.cursor == len("two")


def test_up_keeps_browsing_older_entries(box: _InputBox) -> None:
    """Up while browsing cycles older even though the buffer is full."""
    box.history = ["one", "two"]
    _feed(box, UP + UP)
    assert box.buf == "one"
    # At the oldest entry Up stays put (webview clamps at the end).
    _feed(box, UP)
    assert box.buf == "one"


def test_down_returns_to_newer_then_empty_draft(box: _InputBox) -> None:
    box.history = ["one", "two"]
    _feed(box, UP + UP + DOWN)
    assert box.buf == "two"
    _feed(box, DOWN)
    assert box.buf == ""
    assert box.cursor == 0
    # A further Down is a no-op (not browsing, empty single line).
    _feed(box, DOWN)
    assert box.buf == ""


def test_ss3_up_recalls_history_on_empty_buffer(box: _InputBox) -> None:
    box.history = ["one", "two"]
    _feed(box, SS3_UP)
    assert box.buf == "two"
    _feed(box, SS3_UP + SS3_DOWN)
    assert box.buf == "two"


def test_up_browses_history_across_multiline_entry(box: _InputBox) -> None:
    """A recalled multi-line entry keeps cycling on Up (webview parity)."""
    box.history = ["first", "a\nb\nc"]
    _feed(box, UP)
    assert box.buf == "a\nb\nc"
    # Still browsing: Up cycles to the older entry instead of moving
    # the caret up inside the recalled multi-line text.
    _feed(box, UP)
    assert box.buf == "first"


def test_typing_stops_history_browse(box: _InputBox) -> None:
    """After an edit, Up navigates the text instead of cycling history."""
    box.history = ["one", "two"]
    _feed(box, UP + b"x")
    assert box.buf == "twox"
    _feed(box, UP)
    # Not browsing any more and single-line text: caret jumps to start.
    assert box.buf == "twox"
    assert box.cursor == 0


def test_up_with_no_history_and_empty_buffer_is_noop(box: _InputBox) -> None:
    _feed(box, UP + DOWN)
    assert box.buf == ""
    assert box.cursor == 0


def test_submit_after_history_recall(box: _InputBox) -> None:
    box.history = ["one", "two"]
    submitted = _feed(box, UP + b"\r")
    assert submitted == ["two"]
    assert box.buf == ""


def test_draft_restored_after_browse_started_from_empty(
    box: _InputBox,
) -> None:
    """Down past the newest entry restores the (empty) pre-browse draft."""
    box.history = ["one"]
    _feed(box, UP)
    assert box.buf == "one"
    _feed(box, DOWN)
    assert box.buf == ""
    # And Up starts a fresh browse again.
    _feed(box, UP)
    assert box.buf == "one"
