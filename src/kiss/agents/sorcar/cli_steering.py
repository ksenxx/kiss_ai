# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Claude-CLI-style anchored input box for the ``sorcar`` command line.

While a task runs in the terminal the user can keep typing follow-up
instructions into a bordered input box pinned to the bottom of the
screen.  Submitted lines are queued exactly the way the VS Code
frontend's ``appendUserMessage`` command queues them — appended to the
owning :class:`~kiss.agents.sorcar.running_agent_state._RunningAgentState`'s
``pending_user_messages`` list under ``_registry_lock`` so the live
agent's pre-step hook
(:meth:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent._drain_pending_user_messages`)
injects them into the model conversation before the next model step.

Implementation notes (researched against Claude Code / opencode TUIs):

* The bottom box is anchored using the DEC "Set Top and Bottom Margins"
  (DECSTBM) scroll region — ``ESC[1;{rows-BOX_H}r``.  All agent output
  scrolls inside the region while the box rows below it stay put.
* The terminal is put in a semi-raw mode (``ICANON``/``ECHO`` off,
  ``ISIG`` kept so ``Ctrl+C`` still interrupts) and keys are read in the
  main thread via :func:`select.select` so the loop can also notice the
  worker finishing.
* Agent output is intercepted by swapping :data:`sys.stdout` for a
  lock-guarded wrapper *before* the agent builds its
  :class:`~kiss.core.print_to_console.ConsolePrinter`, so console writes
  and box redraws never interleave on the wire.

Everything degrades gracefully: on Windows, when stdin/stdout is not a
TTY, or when the terminal is too small, :func:`run_with_steering` simply
runs the agent normally with no box.
"""

from __future__ import annotations

import codecs
import ctypes
import logging
import os
import queue
import re
import select
import signal
import sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from kiss.agents.sorcar.cli_panel import (
    _ESC,
    CYAN,
    DIM,
    IDLE_TITLE,
    PROMPT_MARKER,
    RESET,
    STEER_TITLE,
    _term_size,
    body_cursor_col,
    menu_row,
    panel_body,
    panel_bottom,
    panel_cols,
    panel_top,
)
from kiss.agents.sorcar.cli_panel import (
    MIN_BODY_ROWS as _MIN_BODY_ROWS,
)
from kiss.agents.sorcar.persistence import _allocate_chat_id
from kiss.agents.sorcar.running_agent_state import _RunningAgentState

if TYPE_CHECKING:
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

logger = logging.getLogger(__name__)

try:  # POSIX-only terminal control; absent on Windows.
    import termios

    _HAVE_TERMIOS = True
except ImportError:  # pragma: no cover - exercised only on Windows
    termios = None  # type: ignore[assignment]
    _HAVE_TERMIOS = False

# Minimum height (rows) reserved at the bottom of the screen for the
# input box: 1 top-border row + :data:`~kiss.agents.sorcar.cli_panel
# .MIN_BODY_ROWS` body rows + 1 bottom-border row.  The *effective*
# height grows when the edit buffer contains embedded newlines
# (Shift+Enter / multi-line paste) — see :func:`_box_h_for`.  The
# minimum body-row count is shared with :func:`panel_body` so the
# rendered frame and the reserved scroll-region area always agree.
_BOX_H = 2 + _MIN_BODY_ROWS
# Minimum terminal height for which the anchored box is worthwhile.
_MIN_ROWS = _BOX_H + 3


def _box_body_h(buf: str) -> int:
    """Return the number of body rows required to show *buf*.

    Always at least :data:`_MIN_BODY_ROWS` (3) so the framed input
    dialog keeps a stable "text area" look — the user sees three lines
    of vertical space whether the buffer is empty, a single line, or
    holds embedded ``\\n`` characters.  Above the floor the count
    grows with the buffer (one body row per ``\\n``) so multi-line
    input (Shift+Enter / bracketed paste / programmatic assignment) is
    fully visible.

    Args:
        buf: The current edit buffer.

    Returns:
        The number of body rows (``>= _MIN_BODY_ROWS``).
    """
    lines = 1 if not buf else buf.count("\n") + 1
    return max(_MIN_BODY_ROWS, lines)


def _box_h_for(buf: str) -> int:
    """Return the full reserved height (rows) for the box showing *buf*.

    Equals ``2 + _box_body_h(buf)`` — one row each for the top and
    bottom borders plus one body row per buffer line.

    Args:
        buf: The current edit buffer.

    Returns:
        The full box height in rows (``>= _BOX_H``).
    """
    return 2 + _box_body_h(buf)

# Bracketed-paste markers (the terminal wraps pasted text in these once
# mode 2004 is enabled), and a pattern stripping ANSI escape sequences
# that may be embedded in pasted content.
_PASTE_START = f"{_ESC}[200~"
_PASTE_END = f"{_ESC}[201~"
_PASTE_SEQ_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b.")


def _box_top_row(rows: int, box_h: int = _BOX_H) -> int:
    """Return the box's first screen row (1-based) for *rows* total rows.

    Clamped so a terminal shrunk below the box height never produces
    zero or negative row coordinates (which would render as invalid
    control sequences).

    Args:
        rows: Total terminal rows.
        box_h: Effective height of the box (including any menu rows
            stacked above it when the in-place completion menu is
            open).  Defaults to :data:`_BOX_H` (no menu).

    Returns:
        The 1-based row of the box's top border, always ``>= 1``.
    """
    return max(rows - box_h, 1) + 1


# Maximum number of in-place completion menu rows rendered above the
# box.  When the candidate list exceeds this height the menu scrolls so
# the selected candidate stays visible.  Capped further at runtime so
# the menu + box never consume the whole terminal.
_MENU_MAX_H = 8


def _partial_suffix_len(text: str, seq: str) -> int:
    """Return the length of the longest proper prefix of *seq* ending *text*.

    Used to detect a paste terminator split across ``os.read`` chunks.

    Args:
        text: The processed input chunk.
        seq: The full sequence whose prefix may dangle at the chunk end.

    Returns:
        The number of trailing characters of *text* that form a proper
        prefix of *seq* (``0`` when none do).
    """
    for k in range(len(seq) - 1, 0, -1):
        if text.endswith(seq[:k]):
            return k
    return 0


def supports_steering() -> bool:
    """Return whether an anchored input box can be rendered.

    Requires a POSIX terminal (``termios`` available) with both stdin
    and stdout attached to a TTY.

    Returns:
        ``True`` when the interactive box should be used.
    """
    if not _HAVE_TERMIOS:
        return False
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


class _StdoutProxy:
    """A ``sys.stdout``/``sys.stderr`` replacement serialising writes with box redraws.

    All attribute access other than :meth:`write`/:meth:`flush` is
    delegated to the real stream so Rich still detects the TTY, colour
    support and terminal width correctly.

    While the box is active the real terminal cursor lives in the box
    body (blinking right after the chevron, like the idle ``sorcar``
    prompt).  Agent output, however, must land in the scroll region
    above the box.  So each write restores the saved *output* cursor
    position, emits the text, re-saves the advanced output position, and
    finally parks the visible cursor back in the box body.

    Attributes:
        _stream: The original stdout stream.
        _lock: Shared re-entrant lock guarding terminal writes.
        _box: The active input box (to re-park the caret after output).
    """

    def __init__(
        self, stream: Any, lock: threading.RLock, box: _InputBox
    ) -> None:
        self._stream = stream
        self._lock = lock
        self._box = box

    def write(self, text: str) -> int:
        """Write *text* to the underlying stream under the shared lock.

        Args:
            text: The string to write.

        Returns:
            The number of characters written.
        """
        with self._lock:
            if self._box._active:
                # Restore the output cursor, emit, re-save it, then return
                # the blinking caret to the box body.
                self._stream.write(f"{_ESC}8")
                n = self._stream.write(text)
                self._stream.write(f"{_ESC}7")
                self._box._park_cursor_locked()
            else:
                n = self._stream.write(text)
            self._stream.flush()
            return int(n)

    def flush(self) -> None:
        """Flush the underlying stream under the shared lock."""
        with self._lock:
            self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        # Delegated for isatty(), fileno(), encoding, etc.
        return getattr(self._stream, name)


class _InputBox:
    """Renders and edits the anchored bottom input line.

    The box owns the raw-terminal lifecycle (scroll region, cursor
    visibility, ``termios`` mode) and the in-progress edit buffer.  It
    does *not* know anything about agents — submitted lines are handed
    back to the owner via the callback passed to :meth:`feed`.

    Attributes:
        lock: Shared lock (also used by :class:`_StdoutProxy`).
        buf: The current edit buffer.
        title: Text shown in the top border.
        status: Right-aligned text shown in the bottom border.
    """

    def __init__(self, lock: threading.RLock, out: Any) -> None:
        self.lock = lock
        self.buf = ""
        self.title = STEER_TITLE
        self.status = ""
        self._out = out
        self._fd = -1
        self._old_term: Any = None
        self._active = False
        # Rows for which the DECSTBM scroll region was last emitted; a
        # mismatch on redraw means the terminal was resized and the
        # region must be re-anchored (see :meth:`_draw_locked`).
        self._rows = 0
        # Multi-byte UTF-8 characters and escape sequences can be split
        # across ``os.read`` chunks; the decoder buffers partial bytes
        # and ``_pending_esc`` carries an incomplete escape sequence
        # tail into the next :meth:`feed` call.
        self._decoder = codecs.getincrementaldecoder("utf-8")(
            errors="ignore"
        )
        self._pending_esc = ""
        # Inside a bracketed paste (between ``ESC[200~`` and
        # ``ESC[201~``): newlines are inserted into the buffer instead
        # of submitting, and control chars lose their key meaning.
        self._pasting = False
        # The raw termios settings applied by :meth:`start`, re-applied
        # by the SIGCONT handler after a suspend/resume cycle.
        self._raw_term: Any = None
        self._prev_sigcont: Any = None
        # Persistent input history (oldest -> newest); Up/Down browse it.
        # ``_hist_idx`` is ``None`` when not browsing; otherwise an index
        # into ``history`` (or ``len(history)`` when at the saved draft).
        self.history: list[str] = []
        self._hist_idx: int | None = None
        self._hist_saved: str = ""
        # Tab-completion callback returning ranked replacement candidates
        # for the current buffer.  Tab opens an in-place menu using
        # the returned candidates; the menu state below tracks it.
        self.completer_fn: Callable[[str], list[str]] | None = None
        # In-place completion menu: when open, candidate rows are drawn
        # above the box's top border, the scroll region shrinks by the
        # menu's height so agent output cannot scroll over the menu,
        # and Tab/arrows navigate the candidates instead of editing.
        # ``_menu_items`` is the full ranked candidate list,
        # ``_menu_sel`` is the highlighted index, and ``_menu_scroll``
        # is the first visible item (so a long candidate list scrolls
        # within the menu height cap).
        self._menu_open = False
        self._menu_items: list[str] = []
        self._menu_sel = 0
        self._menu_scroll = 0
        # Effective menu height as last rendered.  A mismatch on the
        # next draw means the scroll region must be re-anchored and
        # the rows that newly became scroll-region rows must be
        # cleared (otherwise stale menu glyphs linger inside the
        # scroll region).
        self._drawn_menu_h = 0
        # Effective box height (top border + body rows + bottom border)
        # as last rendered.  A change between redraws — caused by the
        # edit buffer gaining / losing an embedded newline (Shift+Enter,
        # paste, history recall) — also triggers a re-anchor of the
        # DECSTBM scroll region and a clear of any rows that are no
        # longer covered by the smaller reserved area.
        self._drawn_box_h = 0
        # Tiny ``(buf, candidates)`` cache used by
        # :meth:`_refresh_typing_menu` to short-circuit repeat
        # completer calls (auto-repeat, idempotent edits, etc.) so a
        # slow ``completer_fn`` does not stall keystroke processing
        # or block agent stdout writes contending for ``self.lock``.
        self._last_completed_buf: str | None = None
        self._last_completed_cands: list[str] = []

    def start(self) -> None:
        """Enter raw mode, reserve the scroll region and draw the box."""
        assert termios is not None
        rows, _ = _term_size()
        self._fd = sys.stdin.fileno()
        self._old_term = termios.tcgetattr(self._fd)
        new = termios.tcgetattr(self._fd)
        # Disable canonical mode + echo; keep ISIG so Ctrl+C interrupts.
        new[3] = new[3] & ~(termios.ICANON | termios.ECHO)
        new[6][termios.VMIN] = 1
        new[6][termios.VTIME] = 0
        termios.tcsetattr(self._fd, termios.TCSANOW, new)
        self._raw_term = new
        # Re-assert the raw mode, paste mode, scroll region and box
        # after a Ctrl+Z suspend is resumed (``fg``); without this the
        # screen stays corrupted until the next keypress or resize.
        # ``signal.signal`` only works in the main thread — skip the
        # handler elsewhere rather than failing.
        try:
            self._prev_sigcont = signal.signal(
                signal.SIGCONT, self._on_sigcont
            )
        except ValueError:  # pragma: no cover - non-main-thread start
            self._prev_sigcont = None
        with self.lock:
            out = self._out
            # Push existing content up so the box does not overwrite it.
            out.write("\n" * _BOX_H)
            # Scroll region = everything above the box (no menu open at
            # start, so the effective box height is simply ``_BOX_H``).
            out.write(f"{_ESC}[1;{max(rows - _BOX_H, 1)}r")
            # Park the output cursor on the last region row and save that
            # position; agent output is later written by restoring to it
            # (see :class:`_StdoutProxy`).
            out.write(f"{_ESC}[{max(rows - _BOX_H, 1)};1H")
            out.write(f"{_ESC}7")
            # Bracketed paste: pasted text arrives wrapped in
            # ``ESC[200~ … ESC[201~`` so embedded newlines insert line
            # breaks instead of submitting partial instructions.
            out.write(f"{_ESC}[?2004h")
            # Keep the real cursor *visible*: it rests (blinking) in the
            # box body, mirroring the idle ``sorcar`` prompt's caret.
            out.write(f"{_ESC}[?25h")
            out.flush()
            self._active = True
            self._rows = rows
            # The initial buffer is empty, so the starting reserved
            # height is the minimum :data:`_BOX_H`.  ``_draw_locked``
            # tracks subsequent changes via ``self._drawn_box_h``.
            self._drawn_box_h = _BOX_H
            self._draw_locked()

    def stop(self) -> None:
        """Reset the scroll region, restore the cursor and terminal mode."""
        if not self._active:
            return
        assert termios is not None
        rows, _ = _term_size()
        # Clear *all* rows the box (possibly grown to multiple body
        # rows by Shift+Enter and possibly stacked under an in-place
        # completion menu) currently occupies, not just the three
        # minimum input-panel rows; otherwise a grown box or open menu
        # would leak its rows under the returning idle prompt.
        drawn_box_h = self._drawn_box_h or _BOX_H
        top_row = _box_top_row(rows, drawn_box_h + self._menu_h())
        if self._prev_sigcont is not None:
            try:
                signal.signal(signal.SIGCONT, self._prev_sigcont)
            except ValueError:  # pragma: no cover - non-main-thread stop
                pass
            self._prev_sigcont = None
        with self.lock:
            out = self._out
            out.write(f"{_ESC}[?2004l")  # leave bracketed-paste mode
            out.write(f"{_ESC}[r")  # reset scroll region to full screen
            # Erase the box's rows so the steering panel does not linger
            # once the task ends.  Otherwise the idle REPL prompt would
            # be drawn *below* the stale steering box, leaving two input
            # panels stacked on screen at once.
            for row in range(top_row, rows + 1):
                out.write(f"{_ESC}[{row};1H{_ESC}[2K")
            out.write(f"{_ESC}[?25h")  # show cursor
            # Park the cursor on the box's old first row so following
            # output (the returning idle prompt) flows from there.
            out.write(f"{_ESC}[{top_row};1H")
            out.flush()
            self._active = False
        if self._old_term is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
            self._old_term = None
        # Drop completion-menu state so a later ``start()`` does not
        # paint phantom menu rows or wrongly trip the "menu shrank"
        # clear path in :meth:`_draw_locked` (which compares the next
        # ``_menu_h()`` against ``_drawn_menu_h`` from this run).
        self._reset_completion_state()
        self._drawn_menu_h = 0
        # Reset the drawn box height so a later ``start()`` does not
        # wrongly trip the "box shrank" clear path against the last
        # run's tall buffer.
        self._drawn_box_h = 0

    def _on_sigcont(self, signum: int, frame: Any) -> None:
        """Restore the box after the process is resumed (``fg``).

        A Ctrl+Z suspend hands the terminal back to the shell, which
        prints over the box and may reset the tty modes; the scroll
        region and bracketed-paste mode are also not guaranteed to
        survive.  Re-apply the raw mode and force a full re-anchor +
        redraw so the session continues seamlessly.

        Args:
            signum: The delivered signal number (``SIGCONT``).
            frame: The interrupted stack frame (unused).
        """
        del signum, frame
        if not self._active:  # pragma: no cover - resumed after stop()
            return
        if termios is not None and self._raw_term is not None and self._fd >= 0:
            try:
                termios.tcsetattr(self._fd, termios.TCSANOW, self._raw_term)
            except termios.error:  # pragma: no cover - tty went away
                logger.debug("could not re-apply raw mode", exc_info=True)
        with self.lock:
            self._out.write(f"{_ESC}[?2004h")
            # Force _draw_locked to re-emit the scroll region.
            self._rows = 0
            self._draw_locked()

    def redraw(self) -> None:
        """Redraw the box, preserving the output cursor position."""
        with self.lock:
            if self._active:
                self._draw_locked()

    def _menu_h(self) -> int:
        """Return the current rendered height of the completion menu.

        ``0`` when the menu is closed; otherwise the lesser of the
        candidate count, :data:`_MENU_MAX_H`, and however many rows
        can fit above the box without consuming the whole scroll
        region (so at least one row of agent output is always
        visible).

        If the terminal is too small to fit even one menu row above
        the box the menu is auto-dismissed (state reset to closed) —
        otherwise the user would be interacting with an invisible
        widget where arrows silently navigate and Enter silently
        overwrites the edit buffer.
        """
        if not self._menu_open or not self._menu_items:
            return 0
        rows, _ = _term_size()
        # Leave at least one scroll-region row above the menu so the
        # last line of agent output stays visible.  The box is taller
        # than :data:`_BOX_H` when the edit buffer contains embedded
        # newlines, so use the effective height for the *current*
        # buffer (not the minimum) when computing the leftover room.
        room = max(rows - _box_h_for(self.buf) - 1, 0)
        h = min(len(self._menu_items), _MENU_MAX_H, room)
        if h == 0:
            # No room to render the menu — collapse to closed so the
            # next keypress (arrows / Enter) is not consumed by an
            # invisible menu.
            self._menu_open = False
            self._menu_items = []
            self._menu_sel = 0
            self._menu_scroll = 0
        return h

    def _draw_locked(self) -> None:
        rows, _ = _term_size()
        cols = panel_cols()
        menu_h = self._menu_h()
        body_rows, is_placeholder = panel_body(self.buf, cols)
        box_body_h = len(body_rows)
        box_h = 2 + box_body_h
        eff_box_h = box_h + menu_h
        top_row = _box_top_row(rows, eff_box_h)
        out = self._out
        # Either the terminal was resized OR the menu opened/closed/
        # changed height OR the edit buffer gained/lost an embedded
        # newline that changes the body-row count: in all three cases
        # the DECSTBM scroll region must be re-anchored to
        # ``rows - eff_box_h`` so agent output cannot scroll over the
        # menu or the box, and the saved output cursor must be re-parked
        # inside the new region.  Row values are clamped to >= 1 so a
        # terminal shrunk below the box height never receives invalid
        # control sequences.
        if (
            rows != self._rows
            or menu_h != self._drawn_menu_h
            or box_h != self._drawn_box_h
        ):
            region_bottom = max(rows - eff_box_h, 1)
            # When the reserved area shrinks (the menu closed and/or
            # the buffer lost a newline) the rows that were reserved
            # are now back in the scroll region; clear them so
            # leftover menu/body glyphs don't linger as ghost text
            # at the bottom of the agent output area.  Use the
            # previous terminal row count so a resize that also
            # shrinks the reserved area clears the freed rows at their
            # OLD on-screen positions.
            prev_eff_h = (self._drawn_box_h or 0) + self._drawn_menu_h
            if prev_eff_h:
                prev_rows = self._rows or rows
                prev_top = _box_top_row(prev_rows, prev_eff_h)
                prev_bottom = prev_top + prev_eff_h - 1
                # Rows that used to be reserved but now sit in the new
                # scroll region (i.e., above the new top row).  When
                # the area grew, this range is empty and nothing is
                # cleared.
                clear_to = min(prev_bottom, top_row - 1)
                for r in range(prev_top, clear_to + 1):
                    out.write(f"{_ESC}[{r};1H{_ESC}[2K")
            out.write(f"{_ESC}[1;{region_bottom}r")
            out.write(f"{_ESC}[{region_bottom};1H")
            out.write(f"{_ESC}7")
            self._rows = rows
            self._drawn_menu_h = menu_h
            self._drawn_box_h = box_h

        # Draw the in-place completion menu rows (if open) right above
        # the box's top border, sharing the same column layout so the
        # menu visually extends the box upward.
        if menu_h:
            self._draw_menu_locked(top_row, cols, menu_h)

        box_top = top_row + menu_h
        top = panel_top(self.title, cols)
        bottom = panel_bottom(self.status, cols)

        out.write(f"{_ESC}[{box_top};1H{_ESC}[2K{CYAN}{top}{RESET}")
        # Each body row is drawn at ``box_top + 1 + i`` (the first body
        # row sits immediately under the top border).  Every body row
        # opens with a two-column prefix — the cyan ``PROMPT_MARKER``
        # chevron on row 0 and the two-space ``_CONT_INDENT`` on every
        # continuation row — followed by the typed text.  The dim
        # styling only applies to the empty-buffer placeholder shown on
        # the single body row produced for an empty buffer.
        for i, body in enumerate(body_rows):
            prefix = body[: len(PROMPT_MARKER)]
            rest = body[len(PROMPT_MARKER) :]
            rest_color = DIM if (is_placeholder and i == 0) else ""
            mid_inner = (
                f" {CYAN}{prefix}{RESET}{rest_color}{rest}{RESET} "
            )
            out.write(
                f"{_ESC}[{box_top + 1 + i};1H{_ESC}[2K"
                f"{CYAN}│{RESET}{mid_inner}{CYAN}│{RESET}"
            )
        out.write(
            f"{_ESC}[{box_top + 1 + box_body_h};1H{_ESC}[2K"
            f"{CYAN}{bottom}{RESET}"
        )
        self._park_cursor_locked(rows, cols)
        out.flush()

    def _draw_menu_locked(self, top_row: int, cols: int, menu_h: int) -> None:
        """Render the in-place completion menu rows above the box.

        Scrolls :attr:`_menu_items` so the highlighted :attr:`_menu_sel`
        row stays inside the visible window of *menu_h* rows.  Each row
        is written with an absolute cursor positioning sequence + a full
        line-clear so a previously open longer menu cannot leak glyphs
        onto the right edge.

        Args:
            top_row: 1-based screen row of the first menu line (which
                doubles as the menu's top border row in the absolute
                layout).
            cols: Total panel width in columns.
            menu_h: Number of menu rows to render (already capped by
                :meth:`_menu_h`).
        """
        n = len(self._menu_items)
        if self._menu_sel < self._menu_scroll:
            self._menu_scroll = self._menu_sel
        elif self._menu_sel >= self._menu_scroll + menu_h:
            self._menu_scroll = self._menu_sel - menu_h + 1
        if self._menu_scroll < 0:
            self._menu_scroll = 0
        if self._menu_scroll > max(n - menu_h, 0):
            self._menu_scroll = max(n - menu_h, 0)
        out = self._out
        for vi in range(menu_h):
            idx = self._menu_scroll + vi
            row = top_row + vi
            if idx >= n:
                out.write(f"{_ESC}[{row};1H{_ESC}[2K")
                continue
            line = menu_row(
                self._menu_items[idx], idx == self._menu_sel, cols,
            )
            out.write(f"{_ESC}[{row};1H{_ESC}[2K{line}")

    def _park_cursor_locked(
        self, rows: int | None = None, cols: int | None = None
    ) -> None:
        """Move the real (blinking) cursor onto the body row after the text.

        Places the caret right after the chevron and any visible typed
        text, exactly where the idle ``sorcar`` prompt leaves it, so the
        steering box shows the same blinking cursor.  The caller is
        responsible for flushing.

        Args:
            rows: Terminal row count (recomputed when ``None``).
            cols: Terminal column count (recomputed when ``None``).
        """
        if rows is None:
            rows, _ = _term_size()
        if cols is None:
            cols = panel_cols()
        # ``_StdoutProxy.write`` calls this between redraws, so the
        # parking math must match the LAST drawn box geometry (held by
        # ``_drawn_box_h``) — not what a fresh ``_draw_locked`` would
        # produce — otherwise the caret would jump off the rendered box
        # in the agent-output frame between buffer mutations.
        drawn_box_h = self._drawn_box_h or _BOX_H
        menu_h = self._menu_h()
        top_row = _box_top_row(rows, drawn_box_h + menu_h)
        # The first body row sits ``menu_h + 1`` rows below the panel's
        # top row (1 row for the top border, ``menu_h`` rows for the
        # menu above it).  ``body_cursor_col`` returns the body-row
        # index of the caret (0 for the first body row, growing with
        # each embedded newline).
        caret_row, col = body_cursor_col(self.buf, cols)
        self._out.write(
            f"{_ESC}[{top_row + menu_h + 1 + caret_row};{col}H"
        )

    def _append_paste(self, chunk: str) -> bool:
        """Append bracketed-paste content to the edit buffer.

        Newlines are kept (normalised to ``\\n``) so a multi-line paste
        stays one instruction, ANSI escape sequences embedded in the
        pasted text are stripped, and other control characters are
        dropped so they cannot act as editing keys.

        Args:
            chunk: Decoded pasted text (without the paste markers).

        Returns:
            ``True`` when the buffer changed.
        """
        chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
        chunk = _PASTE_SEQ_RE.sub("", chunk)
        cleaned = "".join(
            ch
            for ch in chunk
            if (ch >= " " and not ("\x7f" <= ch <= "\x9f"))
            or ch in ("\n", "\t")
        )
        if not cleaned:
            return False
        self.buf += cleaned
        return True

    def _history_back(self) -> None:
        """Step one entry backwards through :attr:`history`."""
        if not self.history:
            return
        if self._hist_idx is None:
            self._hist_saved = self.buf
            self._hist_idx = len(self.history)
        if self._hist_idx > 0:
            self._hist_idx -= 1
            self.buf = self.history[self._hist_idx]

    def _history_forward(self) -> None:
        """Step one entry forwards through :attr:`history` (or back to draft)."""
        if self._hist_idx is None:
            return
        if self._hist_idx < len(self.history) - 1:
            self._hist_idx += 1
            self.buf = self.history[self._hist_idx]
        else:
            self._hist_idx = None
            self.buf = self._hist_saved

    def _reset_completion_state(self) -> None:
        """Close the in-place completion menu and drop its state."""
        with self.lock:
            self._menu_open = False
            self._menu_items = []
            self._menu_sel = 0
            self._menu_scroll = 0

    def _open_completion_menu(self) -> bool:
        """Pop the in-place completion menu using :attr:`completer_fn`.

        Queries the completer with the current edit buffer.  When the
        completer returns zero candidates the menu stays closed and
        the call is a no-op.  With exactly one candidate the menu is
        not opened — instead :attr:`buf` is replaced with the single
        candidate (so trivial single-match completions behave like a
        normal Tab).  With two or more candidates the menu is opened
        with the first candidate highlighted; :attr:`buf` is *not*
        modified until the user actually picks one with Enter.

        Returns:
            ``True`` when the menu changed visible state (opened, or
            the buffer was replaced by a single-candidate completion),
            so the caller knows a redraw is needed.
        """
        if self.completer_fn is None:
            return False
        cands = [c.rstrip("\n") for c in self.completer_fn(self.buf)]
        if not cands:
            return False
        with self.lock:
            if len(cands) == 1:
                self.buf = cands[0]
                self._hist_idx = None
                return True
            self._menu_items = cands
            self._menu_sel = 0
            self._menu_scroll = 0
            self._menu_open = True
        return True

    def _refresh_typing_menu(self) -> bool:
        """Refresh the in-place menu as the user edits the buffer.

        Implements "complete while typing": every buffer edit
        (printable keystroke, backspace, Shift+Enter newline, paste,
        history recall) re-queries :attr:`completer_fn` and pops the
        menu with the returned candidates, *without* ever
        auto-replacing :attr:`buf` — unlike Tab on a single candidate,
        typing only previews (see Tab handler in :meth:`feed` for the
        single-item shortcut).  When the completer returns zero
        candidates any open menu is closed so a stale candidate list
        never lingers under the new buffer.

        Implementation notes:

        * Buffer + completer are snapshotted under :attr:`lock` and
          the (possibly slow) completer call runs *outside* the lock
          so concurrent agent stdout writes are not stalled by it.
        * A small ``(buf, candidates)`` cache short-circuits repeats
          (e.g. auto-repeat or backspace through a shared prefix) so
          the completer is not re-invoked when nothing has changed.
        * If the user typed more characters while the completer was
          running, the snapshot's results are dropped (stale-buf
          guard) so a slow late result cannot overwrite the new
          menu — the next refresh will reflect the current buffer.
        * Selection is preserved across refreshes whenever the
          previously-highlighted candidate is still present in the
          new list (matches the prompt_toolkit behaviour).

        Returns:
            ``True`` when the menu changed visible state (opened,
            refreshed with new items, or closed), so the caller
            knows a redraw is needed.
        """
        with self.lock:
            if self.completer_fn is None:
                if self._menu_open:
                    self._reset_completion_state()
                    return True
                return False
            buf = self.buf
            completer = self.completer_fn
            prev_sel_text = (
                self._menu_items[self._menu_sel]
                if (
                    self._menu_open
                    and self._menu_items
                    and 0 <= self._menu_sel < len(self._menu_items)
                )
                else None
            )
        if not buf:
            with self.lock:
                if self._menu_open:
                    self._reset_completion_state()
                    return True
                return False
        if buf == self._last_completed_buf:
            cands = self._last_completed_cands
        else:
            try:
                cands = [c.rstrip("\n") for c in completer(buf)]
            except Exception:  # noqa: BLE001 - completer must not break editing
                logger.debug(
                    "completer raised during typing refresh", exc_info=True,
                )
                cands = []
            self._last_completed_buf = buf
            self._last_completed_cands = cands
        with self.lock:
            if buf != self.buf:
                # User kept typing while the completer ran; drop these
                # stale results — the next refresh will reflect the
                # now-current buffer.
                return False
            if not cands:
                if not self._menu_open:
                    return False
                self._reset_completion_state()
                return True
            self._menu_items = cands
            # Preserve the highlighted candidate across the refresh if
            # it is still in the list; otherwise fall back to the top.
            if prev_sel_text is not None and prev_sel_text in cands:
                self._menu_sel = cands.index(prev_sel_text)
            else:
                self._menu_sel = 0
                self._menu_scroll = 0
            self._menu_open = True
        return True

    def _menu_move(self, delta: int) -> None:
        """Advance the highlighted menu candidate by *delta* (wraps).

        Args:
            delta: ``+1`` for next item, ``-1`` for previous item.  The
                index wraps modulo :attr:`_menu_items` so a single
                key press past either end returns to the opposite end.
        """
        with self.lock:
            n = len(self._menu_items)
            if n == 0:
                return
            self._menu_sel = (self._menu_sel + delta) % n

    def _menu_accept(self) -> None:
        """Replace :attr:`buf` with the highlighted candidate and close the menu."""
        with self.lock:
            if self._menu_open and self._menu_items:
                self.buf = self._menu_items[self._menu_sel]
                self._hist_idx = None
        # ``_reset_completion_state`` is the single source of truth for
        # tearing down menu state; reuse it instead of duplicating the
        # field resets here.
        self._reset_completion_state()

    def feed(
        self,
        data: bytes,
        on_submit: Any,
        on_abort: Any,
        on_eof: Any = None,
    ) -> None:
        """Process a chunk of raw keyboard input.

        Args:
            data: Raw bytes read from stdin.
            on_submit: Callable invoked with each completed line (string).
            on_abort: Callable invoked when Ctrl+C is pressed.
            on_eof: Optional callable invoked when Ctrl+D is pressed on an
                empty buffer (signalling EOF / exit).  When ``None`` the
                key is ignored.
        """
        text = self._pending_esc + self._decoder.decode(data)
        self._pending_esc = ""
        i = 0
        changed = False
        while i < len(text):
            if self._pasting:
                # Everything up to the paste terminator is content; the
                # terminator itself may be split across reads, so a
                # dangling prefix of it is deferred to the next chunk.
                end = text.find(_PASTE_END, i)
                if end < 0:
                    keep = min(
                        _partial_suffix_len(text, _PASTE_END), len(text) - i
                    )
                    if self._append_paste(text[i : len(text) - keep]):
                        # Bracketed paste is a buffer edit; refresh the
                        # in-place completion menu against the new
                        # buffer ("complete while typing" applies to
                        # paste too — a pasted prefix should immediately
                        # show its completions).  The helper closes the
                        # menu when no candidates match.
                        self._refresh_typing_menu()
                        changed = True
                    self._pending_esc = text[len(text) - keep :] if keep else ""
                    i = len(text)
                    break
                if self._append_paste(text[i:end]):
                    self._refresh_typing_menu()
                    changed = True
                self._pasting = False
                i = end + len(_PASTE_END)
                continue
            ch = text[i]
            if ch == "\x1b":
                # Bracketed paste start: buffer the pasted block (with
                # its newlines) instead of treating it as keystrokes.
                if text.startswith(_PASTE_START[1:], i + 1):
                    self._pasting = True
                    i += len(_PASTE_START)
                    continue
                # Shift+Enter (kitty CSI-u ``ESC[13;2u`` or xterm
                # modifyOtherKeys ``ESC[27;2;13~``) inserts a newline
                # into the buffer instead of submitting the line.
                shift_enter = False
                for seq in ("[13;2u", "[27;2;13~"):
                    if text.startswith(seq, i + 1):
                        self.buf += "\n"
                        # Shift+Enter is a buffer edit; refresh the
                        # menu so suggestions track multi-line input
                        # ("complete while typing").
                        self._refresh_typing_menu()
                        changed = True
                        i += 1 + len(seq)
                        shift_enter = True
                        break
                if shift_enter:
                    continue
                # A chunk ending right at the ESC may be the first half
                # of a sequence split across reads; defer it so the next
                # chunk can complete (and swallow) the sequence.
                if i + 1 >= len(text):
                    self._pending_esc = text[i:]
                    break
                # Parse other CSI escape sequences: Up/Down arrows
                # navigate the in-place completion menu when it's
                # open, or browse the input history otherwise; the
                # rest (Left/Right/F-keys, etc.) are swallowed so
                # their printable bytes do not type into the buffer.
                if text[i + 1] == "[":
                    j = i + 2
                    while j < len(text) and not ("@" <= text[j] <= "~"):
                        j += 1
                    if j >= len(text):  # split mid-sequence
                        self._pending_esc = text[i:]
                        break
                    final = text[j]
                    seq = text[i + 2 : j]
                    if final == "A" and seq == "":  # Up arrow
                        if self._menu_open:
                            self._menu_move(-1)
                        else:
                            self._history_back()
                            # Refresh the menu against the recalled
                            # buffer so "fast complete" tracks history
                            # navigation too.
                            self._refresh_typing_menu()
                        changed = True
                    elif final == "B" and seq == "":  # Down arrow
                        if self._menu_open:
                            self._menu_move(1)
                        else:
                            self._history_forward()
                            self._refresh_typing_menu()
                        changed = True
                    elif final == "Z" and seq == "":  # Shift+Tab
                        if self._menu_open:
                            self._menu_move(-1)
                            changed = True
                    i = j + 1
                    continue
                # Swallow SS3 sequences (``ESC O <final>``): arrow keys
                # in application cursor mode (DECCKM) and F1–F4, whose
                # printable bytes must not be typed into the buffer.
                if text[i + 1] == "O":
                    if i + 2 >= len(text):  # split mid-sequence
                        self._pending_esc = text[i:]
                        break
                    i += 3
                    continue
                i += 1
                continue
            if ch in ("\r", "\n"):
                if self._menu_open:
                    # Enter on an open completion menu accepts the
                    # highlighted candidate (replacing the buffer) and
                    # closes the menu *without* submitting the line —
                    # the next Enter actually submits.  This matches
                    # the prompt_toolkit dropdown behavior and the
                    # ``_accept_completion_enter`` binding in
                    # ``cli_prompt.py``.
                    self._menu_accept()
                    changed = True
                else:
                    line = self.buf
                    self.buf = ""
                    self._hist_idx = None
                    self._hist_saved = ""
                    self._reset_completion_state()
                    changed = True
                    on_submit(line)
            elif ch in ("\x7f", "\x08"):
                if self.buf:
                    self.buf = self.buf[:-1]
                    self._hist_idx = None
                    # Backspace is an edit; refresh the in-place menu
                    # so suggestions track the shrinking buffer ("fast
                    # complete" while typing).  When the buffer becomes
                    # empty or no candidates match, the helper closes
                    # the menu instead.
                    self._refresh_typing_menu()
                    changed = True
                elif self._menu_open:
                    # Backspace on an empty buffer with the menu open
                    # is a natural "dismiss" gesture — closes the menu
                    # without editing the buffer further.
                    self._reset_completion_state()
                    changed = True
            elif ch == "\x15":  # Ctrl+U clears the line
                if self.buf:
                    self.buf = ""
                    self._hist_idx = None
                    self._reset_completion_state()
                    changed = True
                elif self._menu_open:
                    self._reset_completion_state()
                    changed = True
            elif ch == "\x07":  # Ctrl+G cancels the completion menu
                if self._menu_open:
                    self._reset_completion_state()
                    changed = True
            elif ch == "\x03":  # Ctrl+C
                if self._menu_open:
                    # Ctrl+C while the menu is open just dismisses it;
                    # only a bare Ctrl+C with no menu propagates as an
                    # abort to the caller.
                    self._reset_completion_state()
                    changed = True
                    i += 1
                    continue
                on_abort()
                return
            elif ch == "\x04":  # Ctrl+D on empty buffer = EOF
                if not self.buf and on_eof is not None:
                    on_eof()
                    return
            elif ch == "\t":
                # Tab pops the in-place completion menu using
                # ``completer_fn``.  With exactly one candidate the
                # buffer is replaced directly (no menu); with multiple
                # candidates the menu opens with the first highlighted
                # and the buffer is left untouched until the user picks
                # one with Enter.  Tab while the menu is already open
                # advances the highlighted candidate so the user can
                # navigate the list from the home row too — except
                # when only one candidate is showing (which can happen
                # when "complete while typing" pre-opened the menu with
                # a single-match preview), in which case Tab accepts
                # it directly, matching the closed-menu single-match
                # shortcut.
                if self._menu_open:
                    if len(self._menu_items) == 1:
                        # Re-query before accepting so a stale
                        # single-item preview cannot overwrite the
                        # buffer with an out-of-date candidate.
                        self._reset_completion_state()
                        if not self._open_completion_menu():
                            changed = True
                        elif len(self._menu_items) == 1:
                            self._menu_accept()
                            changed = True
                        else:
                            changed = True
                    else:
                        self._menu_move(1)
                        changed = True
                elif self.completer_fn is not None:
                    if self._open_completion_menu():
                        changed = True
            elif ch >= " " and not ("\x80" <= ch <= "\x9f"):
                # The C1 control range (U+0080–U+009F) must never reach
                # the buffer: U+009B is a one-character CSI introducer
                # and would corrupt the terminal when redrawn.  DEL
                # (\x7f) is already consumed by the backspace branch.
                self.buf += ch
                self._hist_idx = None
                # "Fast complete" while typing: every printable
                # keystroke refreshes the in-place menu with current
                # candidates, mirroring the old prompt_toolkit
                # ``complete_while_typing=True`` dropdown.  Unlike Tab,
                # this never auto-replaces ``buf`` — it only previews.
                self._refresh_typing_menu()
                changed = True
            i += 1
        if len(self._pending_esc) > 64:
            # Not a real escape sequence (no terminal sends one this
            # long); drop it rather than buffering forever.
            self._pending_esc = ""
        if changed:
            self.redraw()


class SteeringSession:
    """Runs an agent task while accepting queued follow-up instructions.

    Attributes:
        agent: The live agent instance.
        state: The agent's registry entry (holds ``pending_user_messages``).
        lock: Shared terminal lock.
    """

    def __init__(
        self,
        agent: SorcarAgent,
        state: _RunningAgentState,
        chat_id: str,
        *,
        box: _InputBox | None = None,
        lock: threading.RLock | None = None,
        real_stdout: Any = None,
        real_stderr: Any = None,
    ) -> None:
        # ``chat_id`` is accepted for call-site symmetry with the
        # registry entry but the session itself never needs it.
        del chat_id
        self.agent = agent
        self.state = state
        # When ``box`` is provided the caller (typically
        # :class:`AnchoredRepl`) already owns the terminal scroll region
        # and stdout/stderr proxies; the session shares them instead of
        # tearing them down between tasks so the input bar stays pinned
        # to the bottom of the screen during idle reads too.
        self.lock = lock if lock is not None else threading.RLock()
        self._real_stdout = (
            real_stdout if real_stdout is not None else sys.stdout
        )
        self._real_stderr = (
            real_stderr if real_stderr is not None else sys.stderr
        )
        self._owns_box = box is None
        self.box = (
            box if box is not None
            else _InputBox(self.lock, self._real_stdout)
        )
        self._done = threading.Event()
        self._aborted = threading.Event()
        self._result = ""
        self._error: BaseException | None = None
        self._queued_count = 0
        # ask_user_question coordination.
        self._answer_q: queue.Queue[str] = queue.Queue(maxsize=1)
        self._question_pending = threading.Event()

    def ask_user_question(self, question: str) -> str:
        """Collect an answer to *question* through the bottom input box.

        Runs in the worker thread; blocks until the user submits a line.

        Args:
            question: The question text to display above the box.

        Returns:
            The user's typed answer (possibly empty).
        """
        with self.lock:
            sys.stdout.write(f"\n\x1b[33m? {question}\x1b[0m\n")
            sys.stdout.flush()
        prev_title = self.box.title
        # Dismiss any open in-place completion menu so the next Enter
        # in the question loop submits the user's answer instead of
        # silently picking a stale candidate from the previous edit.
        self.box._reset_completion_state()
        self.box.title = " answer the question above, then Enter "
        self.box.redraw()
        self._question_pending.set()
        try:
            return self._answer_q.get()
        finally:
            self._question_pending.clear()
            self.box.title = prev_title
            self.box.redraw()

    def _on_submit(self, line: str) -> None:
        if self._question_pending.is_set():
            try:
                self._answer_q.put_nowait(line)
            except queue.Full:  # pragma: no cover - drained by waiter
                pass
            return
        text = line.strip()
        if not text:
            return
        with _RunningAgentState._registry_lock:
            self.state.pending_user_messages.append(text)
        self._queued_count += 1
        self.box.status = f" queued: {self._queued_count} "
        with self.lock:
            sys.stdout.write(f"\x1b[2m▸ queued: {text}\x1b[0m\n")
            sys.stdout.flush()

    def _on_abort(self) -> None:
        self._aborted.set()
        self._done.set()

    def _interrupt_worker(self, worker: threading.Thread) -> None:
        """Stop the abandoned worker thread after a Ctrl+C abort.

        Aborting only stops the *waiting* loop; without this the worker
        thread would keep running ``agent.run`` in the background —
        printing over the next idle prompt and spending budget after the
        user had already been told the task was interrupted.  Any
        pending ``ask_user_question`` is first unblocked with an empty
        answer (a thread parked in ``Queue.get`` blocks at C level, where
        an async exception cannot be delivered), then a
        ``KeyboardInterrupt`` is injected into the worker — the same
        mechanism the VS Code server uses to stop a running task — and
        the worker is given a short grace period to unwind.

        Args:
            worker: The thread running ``agent.run``.
        """
        if self._question_pending.is_set():
            try:
                self._answer_q.put_nowait("")
            except queue.Full:  # pragma: no cover - waiter already fed
                pass
        if worker.is_alive() and worker.ident is not None:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(worker.ident),
                ctypes.py_object(KeyboardInterrupt),
            )
            worker.join(timeout=5.0)

    def _worker(self, run_kwargs: dict[str, Any]) -> None:
        try:
            self._result = self.agent.run(**run_kwargs)
        except BaseException as exc:  # noqa: BLE001 - surfaced to caller
            self._error = exc
        finally:
            self._done.set()

    def run(self, run_kwargs: dict[str, Any]) -> str:
        """Run the task with the box active and return the YAML result.

        Args:
            run_kwargs: Keyword arguments forwarded to ``agent.run``.

        Returns:
            The agent's YAML result string.

        Raises:
            KeyboardInterrupt: If the user aborts with Ctrl+C.
        """
        prev_stdout = sys.stdout
        prev_stderr = sys.stderr
        prev_title = self.box.title
        prev_status = self.box.status
        if self._owns_box:
            proxy = _StdoutProxy(self._real_stdout, self.lock, self.box)
            sys.stdout = cast(Any, proxy)
            sys.stderr = cast(
                Any,
                _StdoutProxy(self._real_stderr, self.lock, self.box),
            )
            self.box.start()
        # Whether the box is owned or shared, the in-task title/status
        # come from the steering preset so the user sees the "queue
        # follow-ups" hint while the agent works.
        with self.lock:
            self.box.title = STEER_TITLE
            self.box.status = ""
            if self.box._active:
                self.box.redraw()
        worker = threading.Thread(
            target=self._worker, args=(run_kwargs,), daemon=True
        )
        worker.start()
        try:
            self._loop()
        except KeyboardInterrupt:
            # SIGINT can interrupt the main thread *outside* the
            # select call guarded inside ``_loop`` — e.g. while it is
            # blocked on the shared terminal lock in ``box.feed`` →
            # ``redraw`` waiting for a worker write to finish.  Treat
            # it exactly like an in-loop Ctrl+C so the worker is still
            # interrupted below instead of leaking in the background.
            self._on_abort()
        finally:
            if self._owns_box:
                self.box.stop()
                sys.stdout = prev_stdout
                sys.stderr = prev_stderr
            else:
                # Shared box stays drawn for the next idle read; just
                # restore the title/status so the next idle read shows
                # the idle preset instead of the steering one.
                with self.lock:
                    self.box.title = prev_title
                    self.box.status = prev_status
                    if self.box._active:
                        self.box.redraw()
        if self._aborted.is_set():
            self._interrupt_worker(worker)
            raise KeyboardInterrupt
        if self._error is not None:
            raise self._error
        return self._result

    def _loop(self) -> None:
        fd = sys.stdin.fileno()
        last_size = _term_size()
        while not self._done.is_set():
            try:
                ready, _, _ = select.select([fd], [], [], 0.1)
            except (InterruptedError, OSError):
                continue
            except KeyboardInterrupt:
                self._on_abort()
                return
            if not ready:
                # Poll for terminal resizes so the box re-anchors within
                # one select timeout even when no key is pressed.
                size = _term_size()
                if size != last_size:
                    last_size = size
                    self.box.redraw()
                continue
            try:
                data = os.read(fd, 4096)
            except (InterruptedError, OSError):
                continue
            if not data:
                continue
            self.box.feed(data, self._on_submit, self._on_abort)


class AnchoredRepl:
    """Owns the bottom-anchored input box for the whole sorcar REPL.

    The box stays pinned at the bottom of the screen for both idle
    reads (the next instruction the user wants to dispatch) and task
    execution (queueing follow-up instructions into
    ``state.pending_user_messages`` exactly the way the VS Code
    extension's ``appendUserMessage`` command queues them).  The
    scroll region above the box scrolls agent output as usual, so the
    input bar behaves like Claude Code's fullscreen TUI mode — visible
    all the time, regardless of whether a task is running.

    Used as a context manager so the box is torn down on exit even
    when the REPL raises::

        with AnchoredRepl(completer_fn=fn, history=h) as repl:
            line = repl.read_idle_line()
            repl.run_task(agent, state, chat_id, run_kwargs)

    Attributes:
        lock: Shared terminal lock guarding stdout/box writes.
        box: The persistent :class:`_InputBox` rendered at the bottom.
    """

    def __init__(
        self,
        completer_fn: Callable[[str], list[str]] | None = None,
        history: list[str] | None = None,
    ) -> None:
        self.lock = threading.RLock()
        # Capture the real stdout/stderr now (before ``__enter__`` swaps
        # in the proxies) so box rendering writes straight to the
        # terminal regardless of how often ``sys.stdout`` is reassigned
        # later.
        self._real_stdout = sys.stdout
        self._real_stderr = sys.stderr
        self.box = _InputBox(self.lock, self._real_stdout)
        self.box.completer_fn = completer_fn
        self.box.history = list(history or [])
        self._prev_stdout: Any = None
        self._prev_stderr: Any = None

    def __enter__(self) -> AnchoredRepl:
        """Install stdout/stderr proxies and start the bottom box."""
        self._prev_stdout = sys.stdout
        self._prev_stderr = sys.stderr
        sys.stdout = cast(
            Any, _StdoutProxy(self._real_stdout, self.lock, self.box),
        )
        sys.stderr = cast(
            Any, _StdoutProxy(self._real_stderr, self.lock, self.box),
        )
        self.box.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Tear down the bottom box and restore the original streams."""
        del exc_type, exc, tb
        self.box.stop()
        sys.stdout = self._prev_stdout
        sys.stderr = self._prev_stderr

    def read_idle_line(self) -> str | None:
        """Read one line in idle mode through the anchored box.

        Up/Down arrows browse :attr:`_InputBox.history`, Tab cycles
        through the registered ``completer_fn`` candidates, and the
        line is echoed above the box (in the scroll region) after
        Enter so the user can see what they typed.

        Returns:
            The typed line (possibly empty), or ``None`` on Ctrl+D
            with an empty buffer.

        Raises:
            KeyboardInterrupt: When the user presses Ctrl+C.
        """
        with self.lock:
            self.box.title = IDLE_TITLE
            self.box.status = ""
            self.box.redraw()
        result: list[str] = []
        eof_flag: list[bool] = []
        abort_flag: list[bool] = []

        def on_submit(line: str) -> None:
            result.append(line)

        def on_abort() -> None:
            abort_flag.append(True)

        def on_eof() -> None:
            eof_flag.append(True)

        fd = sys.stdin.fileno()
        last_size = _term_size()
        while not result and not eof_flag and not abort_flag:
            try:
                ready, _, _ = select.select([fd], [], [], 0.1)
            except (InterruptedError, OSError):
                continue
            except KeyboardInterrupt:
                abort_flag.append(True)
                break
            if not ready:
                size = _term_size()
                if size != last_size:
                    last_size = size
                    self.box.redraw()
                continue
            try:
                data = os.read(fd, 4096)
            except (InterruptedError, OSError):
                continue
            if not data:
                eof_flag.append(True)
                break
            self.box.feed(data, on_submit, on_abort, on_eof)
        if abort_flag:
            raise KeyboardInterrupt
        if eof_flag:
            return None
        line = result[0]
        if line.strip() and (
            not self.box.history or self.box.history[-1] != line
        ):
            self.box.history.append(line)
        with self.lock:
            sys.stdout.write(f"\x1b[36m> {line}\x1b[0m\n")
            sys.stdout.flush()
        return line

    def run_steering_loop(
        self,
        on_submit: Callable[[str], None],
        on_abort: Callable[[], None],
        is_done: Callable[[], bool],
        on_idle: Callable[[], None] | None = None,
    ) -> None:
        """Process stdin through the box, flipped to steering mode.

        Used by the daemon-client REPL while a task runs on the
        ``sorcar web`` daemon: lines submitted into the bottom box
        are routed through ``on_submit`` (which the caller forwards
        as an ``appendUserMessage`` command over the UDS), Ctrl+C
        triggers ``on_abort`` (which the caller forwards as ``stop``),
        and the loop exits once ``is_done`` returns ``True`` —
        typically when the daemon's ``status:false`` event clears
        ``client.dispatcher.task_active``.

        Args:
            on_submit: Callable invoked with each completed line
                (string).  Empty lines are NOT pre-filtered — the
                caller is responsible for any input policy.
            on_abort: Callable invoked when Ctrl+C is pressed.  The
                loop continues running afterwards (so the user can
                queue more input or wait for the daemon's task to
                wind down); use ``is_done`` to actually terminate.
            is_done: Predicate polled in the select timeout window.
                The loop exits once this returns ``True``.
            on_idle: Optional callable invoked once per select
                timeout while waiting.  Callers use it to drain
                ``askUser`` questions from the dispatcher queue
                without blocking the loop.
        """
        with self.lock:
            prev_title = self.box.title
            prev_status = self.box.status
            self.box.title = STEER_TITLE
            self.box.status = ""
            self.box.redraw()
        fd = sys.stdin.fileno()
        last_size = _term_size()
        try:
            while not is_done():
                try:
                    ready, _, _ = select.select([fd], [], [], 0.1)
                except (InterruptedError, OSError):
                    continue
                except KeyboardInterrupt:
                    on_abort()
                    continue
                if not ready:
                    if on_idle is not None:
                        try:
                            on_idle()
                        except Exception:  # noqa: BLE001 - defensive
                            logger.debug("on_idle raised", exc_info=True)
                    size = _term_size()
                    if size != last_size:
                        last_size = size
                        self.box.redraw()
                    continue
                try:
                    data = os.read(fd, 4096)
                except (InterruptedError, OSError):
                    continue
                if not data:
                    continue
                self.box.feed(data, on_submit, on_abort)
        finally:
            with self.lock:
                self.box.title = prev_title
                self.box.status = prev_status
                if self.box._active:
                    self.box.redraw()

    def run_task(
        self,
        agent: SorcarAgent,
        state: _RunningAgentState,
        chat_id: str,
        run_kwargs: dict[str, Any],
    ) -> str:
        """Run an agent task while keeping the anchored box pinned.

        Mirrors :func:`run_with_steering` but threads the shared box,
        lock and proxied streams into :class:`SteeringSession` so the
        box is not torn down between tasks — instead the title and
        status flip from idle to "queue follow-ups" for the duration
        of the task, then back to idle once it ends.

        Args:
            agent: The live agent to run.
            state: The registry entry whose ``pending_user_messages``
                receives lines queued during the task.
            chat_id: The chat identifier (forwarded for symmetry).
            run_kwargs: Keyword arguments forwarded to ``agent.run``.

        Returns:
            The agent's YAML result string.
        """
        session = SteeringSession(
            agent, state, chat_id,
            box=self.box, lock=self.lock,
            real_stdout=self._real_stdout, real_stderr=self._real_stderr,
        )
        kwargs = dict(run_kwargs)
        kwargs["ask_user_question_callback"] = session.ask_user_question
        return session.run(kwargs)


def run_with_steering(
    agent: SorcarAgent, run_kwargs: dict[str, Any]
) -> str:
    """Run *agent* with a Claude-CLI-style steering input box when possible.

    When the terminal supports it, registers a transient
    :class:`_RunningAgentState` so the agent's pre-step hook can drain
    instructions the user queues in the box, then runs the task with the
    box pinned to the bottom of the screen.  Falls back to a plain
    ``agent.run`` otherwise.

    Args:
        agent: The agent to run.
        run_kwargs: Keyword arguments for ``agent.run``.

    Returns:
        The agent's YAML result string.
    """
    rows, _ = _term_size()
    if not supports_steering() or rows < _MIN_ROWS:
        return str(agent.run(**run_kwargs))

    chat_id = getattr(agent, "_chat_id", "") or _allocate_chat_id()
    agent._chat_id = chat_id  # type: ignore[attr-defined]
    # The pre-step drain hook keys off ``_tab_id``; align it with the
    # chat id so a single registry entry serves both the worktree
    # agent's own registration and the drain lookup.
    agent._tab_id = chat_id  # type: ignore[attr-defined]

    state = _RunningAgentState(
        chat_id,
        getattr(agent, "model_name", "") or "",
        agent=cast(Any, agent),
    )
    state.chat_id = chat_id
    state.is_task_active = True
    _RunningAgentState.register(chat_id, state)

    session = SteeringSession(agent, state, chat_id)
    kwargs = dict(run_kwargs)
    kwargs["ask_user_question_callback"] = session.ask_user_question
    try:
        return session.run(kwargs)
    finally:
        # The check-then-remove must be atomic against peer producers
        # (the registry's documented locking discipline), so another
        # component re-registering this chat id between the check and
        # the removal can never have its fresh entry popped.
        with _RunningAgentState._registry_lock:
            if _RunningAgentState.running_agent_states.get(chat_id) is state:
                state.is_task_active = False
                _RunningAgentState.unregister(chat_id)
