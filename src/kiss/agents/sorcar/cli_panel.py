# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared rounded-border input panel for the ``sorcar`` command line.

The idle REPL prompt (:mod:`kiss.agents.sorcar.cli_repl`) and the
anchored steering box (:mod:`kiss.agents.sorcar.cli_steering`) used to
draw two visually different input dialogs — the idle prompt was framed
by plain horizontal rules while the steering box used a rounded border.
Both now render the *same* panel through the helpers in this module, so
the input dialog looks identical whether the agent is idle (waiting for
a task) or running (queuing follow-up instructions).
"""

from __future__ import annotations

import shutil

_ESC = "\x1b"
# ANSI styling shared by every panel render.
CYAN = f"{_ESC}[36m"
DIM = f"{_ESC}[2m"
RESET = f"{_ESC}[0m"
# Cursor marker shown before the editable text inside the panel body.
PROMPT_MARKER = "› "
# Titles shown in the panel's top border for each input mode.
IDLE_TITLE = " sorcar · type a task, then Enter · Ctrl+D to exit "
STEER_TITLE = " steer · type, then Enter to queue · Ctrl+C to abort "
# Dim placeholder shown in the panel body when the edit buffer is empty.
PLACEHOLDER = "Add an instruction for the agent while it works…"


def _term_size() -> tuple[int, int]:
    """Return ``(rows, cols)`` for the controlling terminal.

    Falls back to ``(24, 80)`` when the size cannot be determined.

    Returns:
        A ``(rows, cols)`` tuple, both guaranteed ``>= 1``.
    """
    size = shutil.get_terminal_size(fallback=(80, 24))
    return max(size.lines, 1), max(size.columns, 1)


def panel_cols() -> int:
    """Return the current terminal width (>= 10), falling back to 80.

    Returns:
        The number of columns to draw the panel at.
    """
    return max(_term_size()[1], 10)


def panel_top(title: str, cols: int) -> str:
    """Return the rounded top border line carrying *title*.

    Args:
        title: Text embedded at the left of the top border.
        cols: Total panel width in columns.

    Returns:
        The ``╭─…╮`` border line, clipped to *cols*.
    """
    title = title[: cols - 4]
    top = "╭─" + title + "─" * max(cols - 3 - len(title), 0) + "╮"
    return top[:cols]


def panel_bottom(status: str, cols: int) -> str:
    """Return the rounded bottom border line carrying *status*.

    Args:
        status: Right-aligned text embedded in the bottom border.
        cols: Total panel width in columns.

    Returns:
        The ``╰…╯`` border line, clipped to *cols*.
    """
    status = status[: cols - 4]
    bfill = "─" * max(cols - 3 - len(status), 0)
    bottom = "╰" + bfill + status + "─╯"
    return bottom[:cols]


def clip_buf(buf: str, cols: int) -> str:
    """Return the visible (tail-clipped) slice of *buf* for the body row.

    When the edit buffer is wider than the room left after the chevron
    the trailing end is shown (so the caret stays in view), matching the
    way a real input line scrolls horizontally.

    Args:
        buf: The current edit buffer.
        cols: Total panel width in columns.

    Returns:
        The portion of *buf* that fits on the body row (possibly the
        whole buffer, or its tail when it would overflow).
    """
    inner_w = cols - 4  # room between "│ " and " │"
    avail = inner_w - len(PROMPT_MARKER)
    if len(buf) > avail:
        return buf[len(buf) - avail :]
    return buf


def panel_body(buf: str, cols: int) -> tuple[str, bool]:
    """Return the inner body text for *buf* padded to the panel width.

    The body always opens with the :data:`PROMPT_MARKER` chevron — the
    same ``› `` the idle ``sorcar`` prompt shows — so the steering box
    carries the chevron on the left whether or not anything is typed.

    No caret glyph is appended: callers park the real (blinking) terminal
    cursor right after the typed text (see :func:`body_cursor_col`), so
    the steering box shows the same blinking caret as the idle prompt
    instead of a static block.

    Args:
        buf: The current edit buffer (empty shows the placeholder).
        cols: Total panel width in columns.

    Returns:
        A ``(body, is_placeholder)`` tuple where *body* is padded to the
        panel's inner width and starts with :data:`PROMPT_MARKER`, and
        *is_placeholder* is ``True`` when the empty-buffer placeholder is
        shown (so callers can dim the text after the chevron).
    """
    inner_w = cols - 4  # room between "│ " and " │"
    if buf:
        body = PROMPT_MARKER + clip_buf(buf, cols)
        return body[:inner_w].ljust(inner_w), False
    body = PROMPT_MARKER + PLACEHOLDER
    return body[:inner_w].ljust(inner_w), True


def body_cursor_col(buf: str, cols: int) -> int:
    """Return the 1-based terminal column for the body row's blinking caret.

    The caret sits right after the chevron and any visible typed text,
    exactly where the idle ``sorcar`` prompt leaves the real cursor. The
    body row is drawn as ``│`` (col 1), a space (col 2), then the body
    text (starting at col 3), so the caret lands at
    ``3 + len(PROMPT_MARKER) + len(visible_text)``.

    Args:
        buf: The current edit buffer.
        cols: Total panel width in columns.

    Returns:
        The column (1-based) at which to park the blinking cursor.
    """
    shown = clip_buf(buf, cols) if buf else ""
    return 3 + len(PROMPT_MARKER) + len(shown)
