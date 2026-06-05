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


def panel_cols() -> int:
    """Return the current terminal width (>= 10), falling back to 80.

    Returns:
        The number of columns to draw the panel at.
    """
    cols = shutil.get_terminal_size((80, 24)).columns
    return max(cols, 10)


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


def panel_body(buf: str, cols: int) -> tuple[str, bool]:
    """Return the inner body text for *buf* padded to the panel width.

    The body always opens with the :data:`PROMPT_MARKER` chevron — the
    same ``› `` the idle ``sorcar`` prompt shows — so the steering box
    carries the chevron on the left whether or not anything is typed.

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
        shown = buf
        avail = inner_w - len(PROMPT_MARKER)
        if len(shown) > avail:
            shown = shown[len(shown) - avail :]
        body = PROMPT_MARKER + shown + "▏"
        return body[:inner_w].ljust(inner_w), False
    body = PROMPT_MARKER + PLACEHOLDER
    return body[:inner_w].ljust(inner_w), True
