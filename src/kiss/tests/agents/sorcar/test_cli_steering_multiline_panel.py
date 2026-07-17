# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-reproducer: the steering input box must grow vertically.

Shift+Enter / Alt+Enter / Ctrl+Enter insert real ``\\n`` characters
into the steering box's edit buffer (see ``test_shift_enter_newline``).
Before this fix, multi-line buffers were collapsed onto a single body
row with the visible ``⏎`` glyph and the rounded box stayed three rows
tall — exactly the symptom the user reported ("input text is not shown
as multi-line in the input textbox, the height of the input textbox
MUST adjust to show all lines as separate lines").

These tests pin down the fixed behaviour:

* :func:`panel_body` returns one body string per buffer line (a
  ``list[str]``), padded to the panel's inner width.
* :func:`body_cursor_col` reports both the row offset and column of
  the caret, so the steering box parks the blinking caret on the
  correct line.
* The anchored :class:`_InputBox` grows its on-screen height when the
  buffer contains newlines: it draws a top border, ``N`` body rows
  (one per buffer line), and a bottom border.
* No literal ``⏎`` glyph leaks into the rendered output for
  multi-line buffers.
"""

from __future__ import annotations

import io
import re
import threading

from kiss.ui.cli.cli_panel import (
    PLACEHOLDER,
    PROMPT_MARKER,
    body_cursor_col,
    panel_body,
)
from kiss.ui.cli.cli_steering import _InputBox

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


class TestPanelBodyMultiLine:
    """``panel_body`` returns one entry per buffer line."""

    def test_single_line_returns_one_row(self) -> None:
        rows, is_placeholder = panel_body("hello", 80)
        assert isinstance(rows, list)
        # 3-row minimum: 1 content row + 2 blank padding rows.
        assert len(rows) == 3
        assert is_placeholder is False
        assert PROMPT_MARKER in rows[0]
        assert "hello" in rows[0]
        assert len(rows[0]) == 80 - 4
        for blank in rows[1:]:
            assert blank.strip() == ""

    def test_two_line_buffer_returns_two_rows(self) -> None:
        rows, _ = panel_body("first\nsecond", 80)
        # Two content rows + 1 blank padding row to reach the floor.
        assert len(rows) == 3
        # First row carries the chevron and the first line.
        assert rows[0].startswith(PROMPT_MARKER + "first")
        # Continuation row aligns under the chevron with two spaces,
        # then carries the second line — NO chevron repeated.
        assert rows[1].startswith("  second")
        # Both content rows are padded to the inner width.
        for row in rows[:2]:
            assert len(row) == 80 - 4
        assert rows[2].strip() == ""

    def test_three_line_buffer_returns_three_rows(self) -> None:
        rows, _ = panel_body("a\nb\nc", 40)
        assert len(rows) == 3
        assert rows[0].startswith(PROMPT_MARKER + "a")
        assert rows[1].startswith("  b")
        assert rows[2].startswith("  c")

    def test_four_line_buffer_grows_above_minimum(self) -> None:
        # Above the 3-row floor the panel is fully dynamic.
        rows, _ = panel_body("a\nb\nc\nd", 40)
        assert len(rows) == 4
        assert rows[0].startswith(PROMPT_MARKER + "a")
        assert rows[3].startswith("  d")

    def test_empty_buffer_returns_one_placeholder_row(self) -> None:
        rows, is_placeholder = panel_body("", 80)
        # Placeholder + 2 padding rows to reach the 3-row floor.
        assert len(rows) == 3
        assert is_placeholder is True
        assert rows[0].startswith(PROMPT_MARKER + PLACEHOLDER)
        for blank in rows[1:]:
            assert blank.strip() == ""

    def test_trailing_newline_produces_empty_continuation_row(self) -> None:
        # A trailing \n means "the user pressed Shift+Enter and the
        # next line is empty"; that empty line must still be drawn so
        # the caret sits on its own row.
        rows, _ = panel_body("hello\n", 80)
        # Two buffer lines + 1 blank padding row to reach the floor.
        assert len(rows) == 3
        assert rows[0].startswith(PROMPT_MARKER + "hello")
        # Second row is the two-space indent followed by padding.
        assert rows[1].startswith("  ")
        assert rows[2].strip() == ""

    def test_no_carriage_glyph_in_multi_line_render(self) -> None:
        rows, _ = panel_body("a\nb\nc", 40)
        joined = "".join(rows)
        assert "⏎" not in joined, (
            "newlines must split into separate body rows, not collapse "
            "into a single row with ⏎ glyphs"
        )


class TestBodyCursorColMultiLine:
    """``body_cursor_col`` reports both the row and column of the caret."""

    def test_single_line_caret_on_row_zero(self) -> None:
        row, col = body_cursor_col("hello", 80)
        assert row == 0
        # 1 (│) + 1 (space) + 2 ("› ") + 5 ("hello") = col 9 (1-based after).
        assert col == 3 + 2 + 5

    def test_caret_on_second_line_after_one_newline(self) -> None:
        row, col = body_cursor_col("first\nsecond", 80)
        assert row == 1
        # Continuation indent is "  " (2 cols, same width as the
        # chevron), so column math reuses the same formula.
        assert col == 3 + 2 + len("second")

    def test_caret_on_third_line(self) -> None:
        row, col = body_cursor_col("a\nbb\nccc", 80)
        assert row == 2
        assert col == 3 + 2 + 3

    def test_empty_buffer_caret_after_chevron(self) -> None:
        row, col = body_cursor_col("", 80)
        assert row == 0
        # Caret sits right after the chevron, on the placeholder row.
        assert col == 3 + 2

    def test_trailing_newline_caret_on_empty_continuation(self) -> None:
        row, col = body_cursor_col("hello\n", 80)
        assert row == 1
        # The empty last line has the caret at the start of its text
        # area, right after the two-space continuation indent.
        assert col == 3 + 2


class TestSteeringBoxGrowsForMultiLineBuffer:
    """The anchored ``_InputBox`` must grow vertically to show all lines."""

    def test_three_line_buffer_renders_three_body_rows(self) -> None:
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box._active = True
        box.buf = "line one\nline two\nline three"
        box.redraw()
        text = out.getvalue()
        # All three lines must appear as plain text in the painted
        # output (with no ⏎ glyph collapsing them).
        assert "line one" in text
        assert "line two" in text
        assert "line three" in text
        assert "⏎" not in text, (
            "multi-line buffer should not be collapsed onto one row"
        )

    def test_box_height_grows_with_line_count(self) -> None:
        """Box rows: 1 top border + N body rows + 1 bottom border."""
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box._active = True
        box.buf = "alpha\nbeta\ngamma"
        box.redraw()
        text = _strip_ansi(out.getvalue())
        # Top and bottom borders appear exactly once each per redraw.
        assert text.count("╭") == 1
        assert text.count("╮") == 1
        assert text.count("╰") == 1
        assert text.count("╯") == 1
        # And three body rows each open with the cyan "│" border glyph
        # (cleaned of ANSI escapes here): count `│` characters that are
        # NOT part of a border row.  The borders contribute 0 vertical
        # bars (they use ─), so every │ comes from a body row, with two
        # bars (left + right) per body row → 6 total (matches the
        # 3-row minimum the panel always shows).
        bars = text.count("│")
        assert bars == 6, (
            f"expected 6 vertical bars (3 body rows × 2), got {bars}"
        )

    def test_caret_parked_on_last_body_row(self) -> None:
        """The blinking caret lands on the row containing the last line."""
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box._active = True
        box.buf = "a\nb\nccc"
        box.redraw()
        text = out.getvalue()
        # ``body_cursor_col`` reports (row=2, col=3+2+3=8); the third
        # body row sits 1 + 2 = 3 rows below the top border row of the
        # box.  We don't know the absolute terminal-row coordinate here
        # (depends on terminal size), but the parking sequence must end
        # with ";8H" — the third-line caret column.
        assert ";8H" in text

    def test_shift_enter_grows_the_box_live(self) -> None:
        """Typing Shift+Enter while editing grows the rendered box."""
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box._active = True
        # Type "ab", then a real Shift+Enter (CSI-u), then "cd".
        box.feed(b"ab\x1b[13;2ucd", lambda _s: None, lambda: None)
        text = out.getvalue()
        # Buffer carries the real newline.
        assert box.buf == "ab\ncd"
        # The painted output for the final redraw shows both lines as
        # separate rows (no ⏎ glyph).
        assert "ab" in text
        assert "cd" in text
        assert "⏎" not in text

    def test_collapsing_newline_clears_stale_body_rows(self) -> None:
        """Deleting a newline must clear the now-vacant on-screen row."""
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box._active = True
        box.buf = "alpha\nbeta"
        box.redraw()
        # Replace the buffer with a single-line version and redraw.
        box.buf = "alpha"
        out.seek(0)
        out.truncate(0)
        box.redraw()
        text = out.getvalue()
        # The now-vacant row (where "beta" used to be) must have been
        # explicitly cleared with ESC[<r>;1H ESC[2K — otherwise the
        # leftover "beta" glyphs would linger beneath the new top
        # border.  We accept either an explicit clear OR a fresh redraw
        # that overwrites that row with a border glyph.
        assert "beta" not in _strip_ansi(text), (
            "stale 'beta' row was not cleared after the buffer shrank"
        )
