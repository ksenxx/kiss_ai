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
# Red (optionally blinking) — used by the ``/voice`` ``Listening ...``
# / ``Transcribing ...`` indicator at the beginning of the panel's top
# border; it blinks only after the sorcar wake word is detected.
BLINK = f"{_ESC}[5m"
RED = f"{_ESC}[31m"
# Yellow — used for CLI notification-style messages (all notifications
# render yellow).
YELLOW = f"{_ESC}[33m"
# Cursor marker shown before the editable text inside the panel body.
PROMPT_MARKER = "› "
# Extended-keyboard-protocol enable / disable pairs shared by the idle
# prompt (:mod:`kiss.agents.sorcar.cli_prompt`) and the mid-task
# steering box (:mod:`kiss.agents.sorcar.cli_steering`) so both input
# paths behave identically on every terminal:
#
# * ``ESC[>4;2m`` — xterm ``modifyOtherKeys`` level 2.  Makes
#   Shift/Ctrl/Alt+Enter emit ``ESC[27;<m>;13~``.
# * ``ESC[>1u``   — Kitty keyboard protocol, push flag 1 (disambiguate
#   escape codes).  Makes Shift+Enter emit ``ESC[13;<m>u``.
#
# The disable pair restores ``modifyOtherKeys`` level 0 (``ESC[>4;0m``)
# and pops the pushed Kitty flag entry (``ESC[<u``).
KEYBOARD_PROTO_ENABLE = f"{_ESC}[>4;2m{_ESC}[>1u"
KEYBOARD_PROTO_DISABLE = f"{_ESC}[>4;0m{_ESC}[<u"
# Modifier+Enter escape sequences emitted by terminals for every
# modifier value ``<m> = 2..16`` (per kitty / xterm conventions the
# modifier code is ``1 + bits`` with bit 0 = Shift, bit 1 = Alt,
# bit 2 = Ctrl, bit 3 = Cmd/Meta — so 2 = Shift+Enter, 3 = Alt+Enter,
# 5 = Ctrl+Enter, 9 = Cmd/Meta+Enter, … 16 = Cmd+Ctrl+Alt+Shift+Enter).
# Both input paths (cli_prompt key bindings and cli_steering's
# ``_NEWLINE_AFTER_ESC``) derive their tables from these canonical
# tuples so the two paths can never drift apart.
#
# ``MODIFY_OTHER_KEYS_ENTER`` is the xterm modifyOtherKeys=2 form
# (``ESC[27;<m>;13~``); ``CSI_U_ENTER`` is the kitty / CSI-u form
# (``ESC[13;<m>u``).
MODIFY_OTHER_KEYS_ENTER = tuple(f"{_ESC}[27;{_m};13~" for _m in range(2, 17))
CSI_U_ENTER = tuple(f"{_ESC}[13;{_m}u" for _m in range(2, 17))
# Titles shown in the panel's top border for each input mode.  A
# short hint advertises TAB autocompletion and the multi-line entry
# chords (Alt+Enter / Shift+Enter insert a newline instead of
# submitting).
IDLE_TITLE = (
    "KISS Sorcar · type a task, Enter to submit · "
    "Tab to autocomplete · Alt+Enter/Shift+Enter for newline · "
    "Ctrl+D to exit "
)
STEER_TITLE = (
    "Dynamic steer · type a task, Enter to steer · "
    "Tab to autocomplete · Alt+Enter/Shift+Enter for newline · "
    "Ctrl+C to abort "
)
# Shared user-visible literals for the ``ask_user_question`` flow and
# the queued-instruction feedback, used by both the in-process steering
# session (:mod:`kiss.agents.sorcar.cli_steering`) and the daemon
# client (:mod:`kiss.agents.sorcar.cli_client`) so the two modes can
# never diverge in wording or styling.
ASK_TITLE = " answer the question above, then Enter "
QUESTION_FMT = f"\n{YELLOW}? {{question}}{RESET}\n"
QUEUED_FMT = f"{DIM}▸ queued: {{text}}{RESET}\n"
QUEUED_STATUS_FMT = " queued: {n} "
# Dim placeholder shown in the panel body when the edit buffer is empty.
PLACEHOLDER = "Add an instruction for the agent while it works to steer…"


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
    clipped = _clip_to_width(text, width)
    return clipped + " " * (width - display_width(clipped))


def panel_cols() -> int:
    """Return the current terminal width (>= 10), falling back to 80.

    Returns:
        The number of columns to draw the panel at.
    """
    return max(_term_size()[1], 10)


def _clip_to_width(text: str, width: int) -> str:
    """Clip *text* to at most *width* display columns (no padding).

    Args:
        text: The text to clip.
        width: Maximum display width of the returned string.

    Returns:
        The longest prefix of *text* whose display width is at most
        *width*.
    """
    w = 0
    kept: list[str] = []
    for ch in text:
        cw = char_width(ch)
        if w + cw > width:
            break
        kept.append(ch)
        w += cw
    return "".join(kept)


def panel_top(title: str, cols: int) -> str:
    """Return the rounded top border line carrying *title*.

    Args:
        title: Text embedded at the left of the top border.
        cols: Total panel width in columns.

    Returns:
        The ``╭─…╮`` border line, occupying exactly *cols* terminal
        columns.  The title is clipped by display width (not
        codepoints) so wide (CJK / emoji) characters cannot push the
        border past the terminal edge.
    """
    title = _clip_to_width(title, max(cols - 4, 0))
    fill = "─" * max(cols - 3 - display_width(title), 0)
    return _clip_to_width("╭─" + title + fill + "╮", cols)


def voice_panel_top(
    indicator: str, title: str, cols: int, *, blink: bool = False,
) -> str:
    """Return the fully styled top border opened by a red voice indicator.

    Used by ``/voice``: while the wake-word listener runs, the panel's
    header (top border) starts with the transient *indicator* text
    (``Listening ...`` / ``Transcribing ...``) rendered red, followed by
    the normal *title* and border fill in cyan.  The indicator blinks
    only when *blink* is true — i.e. once the sorcar wake word has been
    detected — and is steady red before that.

    Args:
        indicator: Plain (ANSI-free) indicator text placed at the
            beginning of the header, right after the ``╭─`` corner.
        title: The panel title that follows the indicator.
        cols: Total panel width in columns.
        blink: ``True`` to render the indicator blinking (wake word
            detected / transcription in progress).

    Returns:
        The complete top border line with ANSI styling, occupying
        exactly *cols* terminal columns when printed.
    """
    top = panel_top(indicator + title, cols)
    ind = _clip_to_width(indicator, max(cols - 4, 0))
    # ``top`` is "╭─" + clipped(indicator + title) + fill + "╮"; the
    # first ``len(ind)`` characters after the corner are exactly the
    # (possibly clipped) indicator, so slicing splits it from the rest.
    rest = top[2 + len(ind):]
    style = (BLINK + RED) if blink else RED
    return f"{CYAN}╭─{RESET}{style}{ind}{RESET}{CYAN}{rest}{RESET}"


def panel_bottom(status: str, cols: int) -> str:
    """Return the rounded bottom border line carrying *status*.

    Args:
        status: Right-aligned text embedded in the bottom border.
        cols: Total panel width in columns.

    Returns:
        The ``╰…╯`` border line, occupying exactly *cols* terminal
        columns.  The status is clipped by display width (not
        codepoints) so wide (CJK / emoji) characters cannot push the
        border past the terminal edge.
    """
    status = _clip_to_width(status, max(cols - 4, 0))
    bfill = "─" * max(cols - 3 - display_width(status), 0)
    return _clip_to_width("╰" + bfill + status + "─╯", cols)


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
    # With the cursor parked at the end of the buffer,
    # :func:`visible_line_window` reduces exactly to the tail-clip
    # behaviour this function guarantees (see its docstring), so the
    # windowing logic lives in one place.
    shown, _ = visible_line_window(buf, cols, len(buf))
    return shown


def visible_line_window(
    line: str, cols: int, cursor: int,
) -> tuple[str, int]:
    """Return the visible slice of one buffer line keeping *cursor* shown.

    Like :func:`clip_buf` this fits a single buffer line into the room
    left after the chevron, but instead of always showing the tail it
    horizontally scrolls the window so the character position *cursor*
    (an index into *line*, ``0..len(line)``) stays visible: the text
    before the cursor is tail-clipped so the caret can never fall off
    the left edge, then the leftover width is filled with the
    characters after the cursor.  With the cursor at the end of the
    line this reduces exactly to :func:`clip_buf`'s tail behaviour.

    Args:
        line: One buffer line (no embedded newlines).
        cols: Total panel width in columns.
        cursor: Character index of the caret within *line*; clamped to
            ``0..len(line)``.

    Returns:
        A ``(shown, caret_width)`` tuple where *shown* is the visible
        slice (control characters already sanitised the same way
        :func:`clip_buf` does) and *caret_width* is the display width
        of the visible text before the caret — i.e. the caret's column
        offset from the start of the body text.
    """
    # Tabs render as a space; the mapping is 1:1 per character so
    # cursor offsets into *line* stay aligned with *shown*.
    shown = line.replace("\n", "⏎").replace("\t", " ")
    inner_w = cols - 4  # room between "│ " and " │"
    avail = inner_w - display_width(PROMPT_MARKER)
    cursor = max(0, min(cursor, len(shown)))
    if display_width(shown) <= avail:
        return shown, display_width(shown[:cursor])
    # Tail-clip the text before the cursor so the caret is always in
    # view, then fill the remaining width with the text after it.
    w = 0
    start = cursor
    for k in range(cursor - 1, -1, -1):
        cw = char_width(shown[k])
        if w + cw > avail:
            break
        w += cw
        start = k
    end = cursor
    while end < len(shown):
        cw = char_width(shown[end])
        if w + cw > avail:
            break
        w += cw
        end += 1
    return shown[start:end], display_width(shown[start:cursor])


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
    buf: str,
    cols: int,
    *,
    min_rows: int = MIN_BODY_ROWS,
    cursor: int | None = None,
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
        cursor: Optional caret index into *buf* (``0..len(buf)``).
            The row containing the cursor is clipped with
            :func:`visible_line_window` so the caret stays visible
            even when the cursor sits in the scrolled-off head of a
            long line; ``None`` keeps the legacy tail-clip on every
            row (caret at the end of the buffer).

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
        cur_line = -1
        cur_off = 0
        if cursor is not None:
            cursor = max(0, min(cursor, len(buf)))
            cur_line = buf.count("\n", 0, cursor)
            cur_off = cursor - (buf.rfind("\n", 0, cursor) + 1)
        rows = []
        for i, line in enumerate(lines):
            prefix = PROMPT_MARKER if i == 0 else _CONT_INDENT
            # The cursor's row scrolls horizontally to keep the caret
            # in view; every other row tail-clips via ``clip_buf`` so
            # a long line always shows its end.  After splitting on
            # ``\n`` the ``\n`` → ``⏎`` substitution in ``clip_buf``
            # never fires.
            if i == cur_line:
                shown, _ = visible_line_window(line, cols, cur_off)
            else:
                shown = clip_buf(line, cols)
            rows.append(_clip_pad(prefix + shown, inner_w))
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


def body_cursor_col(
    buf: str, cols: int, cursor: int | None = None,
) -> tuple[int, int]:
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
        cursor: Optional caret index into *buf* (``0..len(buf)``,
            clamped).  ``None`` — the legacy behaviour — places the
            caret after the end of the buffer's last line.

    Returns:
        A ``(row, col)`` tuple where *row* is the 0-based body-row
        index of the caret (``0`` for a single-line buffer or an
        empty buffer showing the placeholder) and *col* is the
        1-based terminal column at which to park the blinking cursor.
    """
    if not buf:
        return 0, 3 + display_width(PROMPT_MARKER)
    cursor = len(buf) if cursor is None else max(0, min(cursor, len(buf)))
    row = buf.count("\n", 0, cursor)
    line_start = buf.rfind("\n", 0, cursor) + 1
    line_end = buf.find("\n", cursor)
    if line_end < 0:
        line_end = len(buf)
    # The continuation indent is exactly as wide as the chevron, so
    # ``visible_line_window`` (which subtracts the chevron's width
    # from the available room) reports the right caret offset for both
    # the first row and any continuation row.
    _, caret_w = visible_line_window(
        buf[line_start:line_end], cols, cursor - line_start,
    )
    return row, 3 + display_width(PROMPT_MARKER) + caret_w
