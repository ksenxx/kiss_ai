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
import unicodedata

_ESC = "\x1b"
# ANSI styling shared by every panel render.
CYAN = f"{_ESC}[36m"
DIM = f"{_ESC}[2m"
BOLD = f"{_ESC}[1m"
# 256-color "DarkOrange" (xterm-256 index 208 ≈ #FF8700) — the closest
# universal-terminal match to Claude Code's "coral" brand orange used
# for its completion / prompt-bar highlight.
ORANGE = f"{_ESC}[38;5;208m"
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


def char_width(ch: str) -> int:
    """Return the terminal display width (columns) of a single character.

    Args:
        ch: A single character.

    Returns:
        ``0`` for combining marks, ``2`` for East-Asian wide/fullwidth
        characters (CJK, emoji), and ``1`` otherwise.
    """
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def display_width(text: str) -> int:
    """Return the terminal display width (columns) of *text*.

    Args:
        text: The string to measure.

    Returns:
        The number of terminal columns *text* occupies when printed.
    """
    return sum(char_width(ch) for ch in text)


def _clip_pad(text: str, width: int) -> str:
    """Clip *text* to at most *width* display columns and pad with spaces.

    Args:
        text: The text to fit.
        width: The exact display width of the returned string.

    Returns:
        A string occupying exactly *width* terminal columns.
    """
    w = 0
    kept: list[str] = []
    for ch in text:
        cw = char_width(ch)
        if w + cw > width:
            break
        kept.append(ch)
        w += cw
    return "".join(kept) + " " * (width - w)


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
    # Newlines (from Shift+Enter or a bracketed paste) render as ⏎ and
    # tabs as a space so the one-row body never emits control chars.
    shown = buf.replace("\n", "⏎").replace("\t", " ")
    inner_w = cols - 4  # room between "│ " and " │"
    avail = inner_w - display_width(PROMPT_MARKER)
    if display_width(shown) <= avail:
        return shown
    w = 0
    idx = len(shown)
    for k in range(len(shown) - 1, -1, -1):
        cw = char_width(shown[k])
        if w + cw > avail:
            break
        w += cw
        idx = k
    return shown[idx:]


# Continuation indent shown on every body row past the first when the
# edit buffer contains embedded newlines (Shift+Enter inserts a real
# ``\n``).  Two spaces aligns continuation text under the ``› `` chevron
# on the opening row so a multi-line input visually reads as one
# block.
_CONT_INDENT = "  "

# Minimum number of body rows the input panel always shows.  The framed
# dialog reserves three lines of vertical space (chevron / placeholder
# row + two padding rows) even when the buffer is empty or a single
# line, so the user has an obvious 3-row "text area" to type into.
# Above three lines the panel dynamically grows with the buffer (see
# :func:`panel_body`).
MIN_BODY_ROWS = 3


def panel_body(
    buf: str, cols: int, *, min_rows: int = MIN_BODY_ROWS,
) -> tuple[list[str], bool]:
    """Return the body row(s) for *buf*, one entry per buffer line.

    The first row opens with the :data:`PROMPT_MARKER` chevron — the
    same ``› `` the idle ``sorcar`` prompt shows — and every subsequent
    row (when *buf* contains embedded newlines) opens with a two-space
    continuation indent (:data:`_CONT_INDENT`) so the lines align under
    the chevron.  Each row is padded to the panel's inner width so
    the steering box can paint a contiguous rectangle whose height
    grows with the number of lines in *buf*.

    The returned list always contains at least *min_rows* entries
    (default :data:`MIN_BODY_ROWS` = 3).  Buffers with fewer lines are
    padded with all-blank rows below the content so the framed dialog
    keeps a stable three-row "text area" look matching the user-visible
    contract; buffers with more lines grow the list one row per
    embedded ``\\n`` (dynamic adjustment upward).

    No caret glyph is appended: callers park the real (blinking) terminal
    cursor right after the typed text on the appropriate row (see
    :func:`body_cursor_col`), so the steering box shows the same blinking
    caret as the idle prompt instead of a static block.

    Args:
        buf: The current edit buffer (empty shows the placeholder).
        cols: Total panel width in columns.
        min_rows: Minimum number of body rows to return.  Defaults to
            :data:`MIN_BODY_ROWS`; callers that want the legacy
            single-row behaviour can pass ``min_rows=1``.

    Returns:
        A ``(rows, is_placeholder)`` tuple where *rows* is a list of
        body strings (one per buffer line, each padded to ``cols - 4``,
        with trailing blank rows up to *min_rows*) and *is_placeholder*
        is ``True`` when the empty-buffer placeholder is shown on the
        first row (so callers can dim the text after the chevron).
    """
    inner_w = cols - 4  # room between "│ " and " │"
    rows: list[str]
    is_placeholder = False
    if not buf:
        rows = [_clip_pad(PROMPT_MARKER + PLACEHOLDER, inner_w)]
        is_placeholder = True
    else:
        lines = buf.split("\n")
        rows = []
        for i, line in enumerate(lines):
            prefix = PROMPT_MARKER if i == 0 else _CONT_INDENT
            # ``clip_buf`` tail-clips a single buffer line so a long
            # line always shows the caret end.  After splitting on
            # ``\n`` the ``\n`` → ``⏎`` substitution in ``clip_buf``
            # never fires.
            rows.append(_clip_pad(prefix + clip_buf(line, cols), inner_w))
    blank = " " * inner_w
    while len(rows) < max(min_rows, 1):
        rows.append(blank)
    return rows, is_placeholder


def menu_row(text: str, selected: bool, cols: int) -> str:
    """Return one fully-styled in-place completion menu row.

    Drawn above the input panel's top border as
    ``│ ❯ candidate    │`` (bold orange — the same coral-orange Claude
    Code uses for its highlighted prompt entry) or
    ``│   candidate    │`` (dim grey otherwise) so the highlighted row
    pops against the surrounding dim entries. Border glyphs stay cyan
    to keep the menu visually anchored to the input box below, using
    the same rounded-border column layout as :func:`panel_body`.

    Args:
        text: Candidate text shown in the row.
        selected: ``True`` when this row is the currently highlighted
            candidate (rendered bold-orange with an ``❯`` marker).
        cols: Total panel width in columns.

    Returns:
        The full row string with ANSI styling and the leading/trailing
        ``│`` border glyphs, ready to be written after an absolute
        cursor positioning escape.
    """
    inner_w = cols - 4  # room between "│ " and " │"
    # Sanitise the candidate text: newlines/tabs render as ⏎/space, all
    # other control characters (C0 \\x00-\\x1f including ESC, DEL \\x7f
    # and C1 \\x80-\\x9f — notably U+009B which is the one-character
    # CSI introducer) are stripped.  Without this an attacker- or
    # bug-generated candidate could inject ANSI styling that leaks past
    # the right border and corrupts the rest of the screen.
    raw = text.replace("\n", "⏎").replace("\t", " ")
    shown = "".join(
        ch for ch in raw if ch >= " " and not ("\x7f" <= ch <= "\x9f")
    )
    marker = "❯ " if selected else "  "
    body = _clip_pad(marker + shown, inner_w)
    if selected:
        inner = f"{ORANGE}{BOLD}{body}{RESET}"
    else:
        inner = f"{DIM}{body}{RESET}"
    return f"{CYAN}│{RESET} {inner} {CYAN}│{RESET}"


def body_cursor_col(buf: str, cols: int) -> tuple[int, int]:
    """Return the ``(row, col)`` body-grid position for the blinking caret.

    The caret sits right after the chevron (or the continuation indent
    on continuation rows) and any visible typed text on the *last*
    buffer line, exactly where the idle ``sorcar`` prompt leaves the
    real cursor.  Every body row is drawn as ``│`` (col 1), a space
    (col 2), then the body text (starting at col 3); the first row's
    text opens with the two-column :data:`PROMPT_MARKER` chevron and
    every continuation row opens with the same-width
    :data:`_CONT_INDENT`, so the column math is identical regardless of
    which row the caret lands on.

    Args:
        buf: The current edit buffer.
        cols: Total panel width in columns.

    Returns:
        A ``(row, col)`` tuple where *row* is the 0-based body-row
        index of the caret (``0`` for a single-line buffer or an
        empty buffer showing the placeholder, ``buf.count('\\n')`` for
        a multi-line buffer) and *col* is the 1-based terminal column
        at which to park the blinking cursor.
    """
    if not buf:
        return 0, 3 + display_width(PROMPT_MARKER)
    lines = buf.split("\n")
    last = lines[-1]
    # The continuation indent is exactly as wide as the chevron, so
    # ``clip_buf`` (which subtracts the chevron's width from the
    # available room) reports the right visible-tail width for both
    # the first row and any continuation row.
    shown = clip_buf(last, cols)
    return len(lines) - 1, 3 + display_width(PROMPT_MARKER) + display_width(shown)
