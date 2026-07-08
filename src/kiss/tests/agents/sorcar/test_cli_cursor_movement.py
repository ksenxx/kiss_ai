# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: Left/Right arrow keys move the input-box cursor.

Reproduces the user-reported bug where the Left and Right arrow keys
did not move the cursor in the sorcar CLI REPL input box (the anchored
:class:`~kiss.agents.sorcar.cli_steering._InputBox` used for both the
idle prompt and mid-task steering), so text typed earlier could not be
edited.  Keystrokes are fed as the raw bytes a real terminal emits and
the resulting edit buffer / submitted line is asserted — no mocks.
"""

from __future__ import annotations

import io
import threading

import pytest

from kiss.agents.sorcar.cli_panel import (
    PROMPT_MARKER,
    body_cursor_col,
    display_width,
    panel_body,
)
from kiss.agents.sorcar.cli_steering import _InputBox

LEFT = b"\x1b[D"
RIGHT = b"\x1b[C"
HOME = b"\x1b[H"
END = b"\x1b[F"
BACKSPACE = b"\x7f"
ENTER = b"\r"


@pytest.fixture
def box() -> _InputBox:
    return _InputBox(threading.RLock(), io.StringIO())


def _feed(box: _InputBox, data: bytes) -> list[str]:
    """Feed *data* to *box*, returning the lines submitted via Enter."""
    submitted: list[str] = []
    box.feed(data, submitted.append, lambda: pytest.fail("unexpected abort"))
    return submitted


def test_left_arrow_moves_cursor_for_mid_line_insert(box: _InputBox) -> None:
    """Typing after Left inserts before the last char, not at the end."""
    _feed(box, b"helo" + LEFT + b"l")
    assert box.buf == "hello"
    assert box.cursor == 4


def test_left_then_right_restores_insert_at_end(box: _InputBox) -> None:
    _feed(box, b"ab" + LEFT + RIGHT + b"c")
    assert box.buf == "abc"
    assert box.cursor == 3


def test_left_at_start_and_right_at_end_are_noops(box: _InputBox) -> None:
    _feed(box, LEFT + LEFT + b"x" + RIGHT + RIGHT + b"y")
    assert box.buf == "xy"
    assert box.cursor == 2


def test_backspace_deletes_before_cursor_mid_line(box: _InputBox) -> None:
    """Backspace after Left removes the char before the cursor."""
    _feed(box, b"abcd" + LEFT + BACKSPACE)
    assert box.buf == "abd"
    assert box.cursor == 2


def test_submit_uses_edited_buffer_and_resets_cursor(box: _InputBox) -> None:
    submitted = _feed(box, b"helo" + LEFT + b"l" + ENTER)
    assert submitted == ["hello"]
    assert box.buf == ""
    assert box.cursor == 0


def test_ss3_application_mode_arrows_move_cursor(box: _InputBox) -> None:
    """DECCKM application-cursor-mode arrows (``ESC O D/C``) also move."""
    _feed(box, b"helo" + b"\x1bOD" + b"l" + b"\x1bOC" + b"!")
    assert box.buf == "hello!"


def test_home_and_end_keys(box: _InputBox) -> None:
    _feed(box, b"world" + HOME + b"hello ")
    assert box.buf == "hello world"
    _feed(box, END + b"!")
    assert box.buf == "hello world!"
    assert box.cursor == len("hello world!")


def test_home_end_tilde_variants(box: _InputBox) -> None:
    """``ESC[1~``/``ESC[7~`` = Home and ``ESC[4~``/``ESC[8~`` = End."""
    _feed(box, b"bc" + b"\x1b[1~" + b"a" + b"\x1b[4~" + b"d")
    assert box.buf == "abcd"
    _feed(box, b"\x1b[7~" + b"0" + b"\x1b[8~" + b"e")
    assert box.buf == "0abcde"


def test_split_left_arrow_across_feed_chunks(box: _InputBox) -> None:
    """An arrow sequence split across reads still moves the cursor."""
    _feed(box, b"ad")
    _feed(box, b"\x1b")
    _feed(box, b"[D")
    _feed(box, b"bc")
    assert box.buf == "abcd"


def test_newline_insert_at_cursor(box: _InputBox) -> None:
    """Alt+Enter (``ESC \\r``) inserts the newline at the cursor."""
    _feed(box, b"ab" + LEFT + b"\x1b\r")
    assert box.buf == "a\nb"
    assert box.cursor == 2


def test_paste_inserts_at_cursor(box: _InputBox) -> None:
    _feed(box, b"ad" + LEFT + b"\x1b[200~bc\x1b[201~")
    assert box.buf == "abcd"
    assert box.cursor == 3
    _feed(box, b"X")
    assert box.buf == "abcXd"


def test_history_recall_places_cursor_at_end(box: _InputBox) -> None:
    box.history = ["first task"]
    _feed(box, b"\x1b[A")  # Up arrow recalls history
    assert box.buf == "first task"
    assert box.cursor == len("first task")
    _feed(box, b"!")
    assert box.buf == "first task!"


def test_ctrl_u_clears_line_and_cursor(box: _InputBox) -> None:
    _feed(box, b"abc" + LEFT + b"\x15")
    assert box.buf == ""
    assert box.cursor == 0
    _feed(box, b"x")
    assert box.buf == "x"


def test_line_continuation_enter_keeps_cursor_after_newline(
    box: _InputBox,
) -> None:
    submitted = _feed(box, b"abc\\" + ENTER)
    assert submitted == []
    assert box.buf == "abc\n"
    assert box.cursor == 4


def test_arrows_cross_embedded_newlines(box: _InputBox) -> None:
    """The flat cursor walks across ``\\n`` from Shift+Enter lines."""
    _feed(box, b"ab\x1b\rcd" + LEFT + LEFT + LEFT + b"X")
    assert box.buf == "abX\ncd"


def test_body_cursor_col_tracks_mid_buffer_cursor() -> None:
    """The caret column reflects the cursor, not the end of the buffer."""
    base = 3 + display_width(PROMPT_MARKER)
    assert body_cursor_col("hello", 80, 3) == (0, base + 3)
    assert body_cursor_col("hello", 80, 0) == (0, base)
    # Multi-line: cursor on the first line of "ab\ncd".
    assert body_cursor_col("ab\ncd", 80, 1) == (0, base + 1)
    # Cursor on the second line.
    assert body_cursor_col("ab\ncd", 80, 4) == (1, base + 1)
    # Default (no cursor) keeps the legacy end-of-buffer behaviour.
    assert body_cursor_col("ab\ncd", 80) == (1, base + 2)


def test_body_cursor_col_wide_chars() -> None:
    """East-Asian wide characters count two columns before the caret."""
    base = 3 + display_width(PROMPT_MARKER)
    assert body_cursor_col("你好x", 80, 2) == (0, base + 4)


def test_caret_visible_when_cursor_scrolled_off_left_edge() -> None:
    """A long line horizontally scrolls so the caret stays in view."""
    cols = 20
    buf = "x" * 100
    row, col = body_cursor_col(buf, cols, 5)
    assert row == 0
    # The caret column must lie inside the panel body (between the
    # borders), never past the right edge.
    assert 3 <= col <= cols - 1
    # The rendered body row must contain the char right after the
    # cursor so the user can see where the insertion point is.
    rows, _ = panel_body(buf, cols, cursor=5)
    assert rows[0].strip("| ").startswith(PROMPT_MARKER.strip())


def test_long_line_render_shows_window_around_cursor() -> None:
    cols = 24
    buf = "abcdefghijklmnopqrstuvwxyz0123456789"
    rows, is_placeholder = panel_body(buf, cols, cursor=0)
    assert not is_placeholder
    # Cursor at start: the visible window must include the head of the
    # buffer (chars "abc"), not only the tail.
    assert "abc" in rows[0]
    # Legacy behaviour (no cursor): tail is shown.
    rows_tail, _ = panel_body(buf, cols)
    assert "789" in rows_tail[0]
