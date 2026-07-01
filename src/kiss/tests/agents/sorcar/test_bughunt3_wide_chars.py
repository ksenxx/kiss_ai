# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt test: panel width math must use display columns, not len().

CJK characters and emoji occupy two terminal columns each but count as a
single code point.  :mod:`kiss.agents.sorcar.cli_panel` used plain
``len()`` for clipping, padding and caret placement, so a buffer of wide
characters rendered a body row wider than the panel — pushing the right
border off the line — and parked the blinking caret in the wrong column.
"""

from __future__ import annotations

import unicodedata

import pytest

from kiss.agents.sorcar.cli_panel import (
    body_cursor_col,
    clip_buf,
    panel_body,
)


def _width(text: str) -> int:
    """Return the terminal display width of *text* (independent oracle)."""
    total = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total


def test_cjk_body_fits_panel_inner_width() -> None:
    """A CJK buffer must render a body of exactly ``cols - 4`` columns."""
    cols = 40
    rows, is_placeholder = panel_body("汉" * 40, cols)
    assert not is_placeholder
    # Single-line buffer pads to the 3-row minimum; only the content
    # row (index 0) carries the wide-char text.
    assert len(rows) == 3
    body = rows[0]
    assert _width(body) == cols - 4, (
        f"wide-char body is {_width(body)} columns, expected {cols - 4}; "
        "the right border is pushed off the row"
    )


def test_emoji_body_fits_panel_inner_width() -> None:
    """An emoji buffer must render a body of exactly ``cols - 4`` columns."""
    cols = 80
    rows, _ = panel_body("😀" * 50, cols)
    assert len(rows) == 3
    assert _width(rows[0]) == cols - 4


def test_short_emoji_body_padded_to_inner_width() -> None:
    """A short emoji buffer is padded to exactly the inner width."""
    cols = 80
    rows, _ = panel_body("😀" * 5, cols)
    assert len(rows) == 3
    assert _width(rows[0]) == cols - 4


def test_clip_buf_tail_respects_display_width() -> None:
    """The clipped tail must fit the columns available after the chevron."""
    cols = 40
    avail = cols - 4 - _width("› ")
    tail = clip_buf("汉" * 100, cols)
    assert _width(tail) <= avail, (
        f"clipped tail is {_width(tail)} columns wide, only {avail} fit"
    )


def test_cursor_col_counts_wide_chars_as_two_columns() -> None:
    """The caret must land after the rendered text, not after len() chars."""
    # body row: "│ " (cols 1-2) + "› " (2 cols) + "汉汉" (4 cols) → col 9.
    assert body_cursor_col("汉汉", 80) == (0, 9)


def test_cursor_never_overruns_right_border() -> None:
    """With a long wide-char buffer the caret stays inside the panel."""
    for cols in (20, 40, 80):
        _row, col = body_cursor_col("汉" * 200, cols)
        assert col <= cols - 1, (
            f"caret column {col} overruns the panel at width {cols}"
        )


def test_ascii_behavior_unchanged() -> None:
    """Sanity: ASCII buffers keep the historical geometry."""
    cols = 80
    rows, _ = panel_body("hello", cols)
    # 3-row minimum: 1 content row + 2 padding rows.
    assert len(rows) == 3
    assert rows[0] == ("› hello").ljust(cols - 4)
    assert body_cursor_col("hello", cols) == (0, 3 + 2 + 5)
    assert clip_buf("x" * 100, cols) == "x" * (cols - 4 - 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
