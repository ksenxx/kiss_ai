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
import contextlib
import ctypes
import logging
import os
import queue
import re
import select
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, cast

from kiss.agents.sorcar.persistence import _allocate_chat_id
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.ui.cli.cli_line_continuation import (
    ends_with_line_continuation,
)
from kiss.ui.cli.cli_panel import (
    _ESC,
    ASK_TITLE,
    CSI_U_ENTER,
    CYAN,
    DIM,
    IDLE_TITLE,
    KEYBOARD_PROTO_DISABLE,
    KEYBOARD_PROTO_ENABLE,
    MODIFY_OTHER_KEYS_ENTER,
    PROMPT_MARKER,
    QUESTION_FMT,
    QUEUED_FMT,
    QUEUED_STATUS_FMT,
    RESET,
    STEER_TITLE,
    _term_size,
    body_cursor_col,
    menu_row,
    panel_body,
    panel_bottom,
    panel_cols,
    panel_top,
    voice_panel_top,
)
from kiss.ui.cli.cli_panel import (
    MIN_BODY_ROWS as _MIN_BODY_ROWS,
)
from kiss.ui.cli.cli_prompt import _AT_RE

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

# Sub-sequences that appear immediately AFTER an ESC byte and must
# insert a newline into the buffer (not submit).  This covers every
# modifier+Enter encoding terminals emit:
#
# * ``\r``                — portable Alt+Enter (xterm-style: ESC + CR)
# * ``\n``                — tmux M-Enter / some terminals' Alt+Enter
#                           (ESC + LF)
# * ``[13;<m>u``          — kitty / CSI-u modifyOtherKeys=1 form
# * ``[27;<m>;13~``       — xterm modifyOtherKeys=2 form
#
# Modifier code ``<m>`` (per kitty / xterm conventions): bit 0 = Shift,
# bit 1 = Alt, bit 2 = Ctrl, bit 3 = Cmd/Meta — encoded as ``<m> =
# 1 + bits``.  Values 2..16 cover every Shift / Alt / Ctrl / Cmd
# combination (and Cmd+Ctrl+Alt+Shift = 16).
#
# Order matters: longest multi-char sequences are listed before single
# bytes so the prefix-startswith match in
# :meth:`_InputBox.feed` cannot mistake a CSI prefix for a bare CR / LF.
#
# The CSI tables are derived from the canonical full-sequence tuples in
# :mod:`kiss.agents.sorcar.cli_panel` (shared with the cli_prompt key
# bindings) by dropping the leading ESC byte, so the two input paths
# can never drift apart.
_NEWLINE_AFTER_ESC: tuple[str, ...] = (
    *(seq[1:] for seq in MODIFY_OTHER_KEYS_ENTER),
    *(seq[1:] for seq in CSI_U_ENTER),
    "\r",
    "\n",
)


# Navigation finals shared by the CSI (``ESC[``) and SS3 (``ESC O``)
# escape-sequence branches of :meth:`_InputBox.feed`; both dispatch
# through :meth:`_InputBox._nav_action` so the two encodings can never
# drift apart.
_NAV_FINALS: dict[str, str] = {
    "A": "up", "B": "down", "C": "right", "D": "left",
    "H": "home", "F": "end",
}


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


def _normalize_candidates(
    raw: list[str] | list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """Split completer candidates into replacement and display lists.

    The :attr:`_InputBox.completer_fn` contract accepts either plain
    replacement strings (replacement doubles as the menu row text) or
    ``(replacement, display)`` pairs — e.g. a ``--task`` value whose
    row displays ``<id>: <one-line task description>`` while accepting
    inserts only the bare id.  Trailing newlines are stripped from
    both parts, and any line break EMBEDDED in the display text is
    flattened to a single space, so a candidate can never break the
    single-row menu rendering (a raw ``\\n`` written mid-row would
    push the rest of the row onto the next screen line, corrupting
    the absolutely-positioned menu/box layout).  Replacements keep
    embedded newlines: accept inserts them into the edit buffer,
    which renders multi-line buffers correctly.

    Args:
        raw: Candidates as returned by ``completer_fn``.

    Returns:
        ``(replacements, displays)`` — two parallel lists of equal
        length.
    """
    repls: list[str] = []
    displays: list[str] = []
    for cand in raw:
        if isinstance(cand, tuple):
            repl, disp = cand
        else:
            repl = disp = cand
        repls.append(repl.rstrip("\n"))
        displays.append(" ".join(disp.rstrip("\n").splitlines()))
    return repls, displays


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
        cursor: Caret index into :attr:`buf` (``0..len(buf)``).
            Left/Right (and Home/End) move it; every buffer edit
            inserts/deletes at this position so previously-typed text
            can be corrected.  Assigning :attr:`buf` moves the cursor
            to the end of the new text; editing code that inserts or
            deletes mid-buffer re-places the cursor right after the
            assignment.
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
        # Transient status overlay: when non-empty, this plain text is
        # shown in red at the *beginning of the panel's top border*
        # (the header) — used by ``/voice`` for its ``Listening ...``
        # / ``Transcribing ...`` indicator.  ``overlay_blink`` makes it
        # blink; ``/voice`` sets it only once the sorcar wake word has
        # been detected (the listener's ``WAKE``/``TRANSCRIBING``
        # protocol lines).
        self.overlay = ""
        self.overlay_blink = False
        # Lines injected programmatically (by the ``/voice`` wake-word
        # pump) that the stdin pump treats exactly like Enter-submitted
        # typed lines.  Guarded by :attr:`lock`; drained at the top of
        # every :func:`_pump_stdin` iteration.
        self._injected: list[str] = []
        # Persistent input history (oldest -> newest); Up/Down browse
        # it when the buffer is empty or a browse is in progress
        # (otherwise they move the caret between buffer lines).
        # ``_hist_idx`` is ``None`` when not browsing; otherwise an index
        # into ``history`` (or ``len(history)`` when at the saved draft).
        self.history: list[str] = []
        self._hist_idx: int | None = None
        self._hist_saved: str = ""
        # Tab-completion callback returning ranked candidates for the
        # current buffer.  Each candidate is either a plain replacement
        # string or a ``(replacement, display)`` pair — *display* is
        # the row text shown in the in-place menu (e.g. a task id with
        # its one-line description) while *replacement* is what accept
        # actually puts into the buffer.  Tab opens an in-place menu
        # using the returned candidates; the menu state below tracks it.
        self.completer_fn: (
            Callable[[str], list[str] | list[tuple[str, str]]] | None
        ) = None
        # In-place completion menu: when open, candidate rows are drawn
        # above the box's top border, the scroll region shrinks by the
        # menu's height so agent output cannot scroll over the menu,
        # and Tab/arrows navigate the candidates instead of editing.
        # ``_menu_items`` is the full ranked candidate DISPLAY list
        # (what the rows render), ``_menu_repls`` the parallel
        # replacement list (what accept inserts into the buffer),
        # ``_menu_sel`` is the highlighted index, and ``_menu_scroll``
        # is the first visible item (so a long candidate list scrolls
        # within the menu height cap).
        self._menu_open = False
        self._menu_items: list[str] = []
        self._menu_repls: list[str] = []
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
        # The cached value holds the normalised ``(replacements,
        # displays)`` lists.
        self._last_completed_buf: str | None = None
        self._last_completed_cands: tuple[list[str], list[str]] = ([], [])

    @property
    def buf(self) -> str:
        """The current edit buffer."""
        return self._buf

    @buf.setter
    def buf(self, value: str) -> None:
        """Replace the edit buffer and move the caret to its end.

        Whole-buffer replacement (history recall, completion accept,
        Ctrl+U clear, submit reset, direct assignment by owners) puts
        the insertion point after the new text — the only sensible
        spot for a fresh buffer.  Mid-buffer edits in :meth:`feed`
        re-assign :attr:`cursor` right after setting :attr:`buf`.
        """
        self._buf = value
        self.cursor = len(value)

    def start(self) -> None:
        """Enter raw mode, reserve the scroll region and draw the box."""
        assert termios is not None
        rows, _ = _term_size()
        self._fd = sys.stdin.fileno()
        self._old_term = termios.tcgetattr(self._fd)
        new = termios.tcgetattr(self._fd)
        # Disable canonical mode + echo; keep ISIG so Ctrl+C interrupts.
        new[3] = new[3] & ~(termios.ICANON | termios.ECHO)
        # Also disable kernel CR<->LF translation on input so the box
        # can distinguish Enter (``\r``) from Ctrl+J / Alt+Enter
        # (``\n``).  With ICRNL the kernel quietly rewrites incoming
        # ``\r`` bytes to ``\n``, so the bound terminal sequences for
        # newline-insert (``ESC\r``, modifyOtherKeys 13, etc.) would
        # arrive as ``ESC\n`` and the steering box would have no way
        # to tell a real Enter from an Alt+Enter / Ctrl+J newline
        # insert.  INLCR is the symmetric LF->CR mapping; clearing it
        # makes an LF stay an LF.
        new[0] = new[0] & ~(termios.ICRNL | termios.INLCR)
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
            # Opt into the terminal's extended-keyboard protocols so
            # modifier+Enter chords (Shift+Enter, Ctrl+Enter, Alt+Enter,
            # Cmd+Enter and combinations) emit *distinct* byte
            # sequences the steering box can recognise as
            # newline-insert instead of a plain Enter (submit).  Without
            # these enable sequences most terminals collapse
            # Shift+Enter to a bare ``\r`` — indistinguishable from
            # Enter — and the whole multi-line UX breaks:
            #
            # * ``ESC[>4;2m``  — xterm modifyOtherKeys level 2.  Makes
            #   Shift/Ctrl/Alt+Enter emit ``ESC[27;<m>;13~``.  Supported
            #   by xterm, iTerm2, WezTerm, Alacritty, tmux (with
            #   ``extended-keys always`` + ``extkeys`` feature).
            # * ``ESC[>1u``    — Kitty keyboard protocol, push flag 1
            #   (disambiguate escape codes).  Makes Shift+Enter emit
            #   ``ESC[13;<m>u``.  Supported by kitty, WezTerm, foot,
            #   ghostty and (increasingly) other terminals.
            #
            # Terminals that don't support one/both of these silently
            # ignore the CSI (they may respond with a DA1-style reply
            # which the CSI parser in :meth:`feed` swallows as an
            # unknown sequence).  See :meth:`stop` for the matching
            # disable sequences.
            out.write(KEYBOARD_PROTO_ENABLE)
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
        if self._prev_sigcont is not None:
            # Suppress ValueError for non-main-thread stop.
            with contextlib.suppress(ValueError):
                signal.signal(signal.SIGCONT, self._prev_sigcont)
            self._prev_sigcont = None
        with self.lock:
            # Compute the rows to clear UNDER the lock: ``_menu_h()``
            # reads (and, when the terminal is too small, mutates) the
            # shared menu state, so a lock-free call would race the
            # pump/draw threads (w2 F4).
            drawn_box_h = self._drawn_box_h or _BOX_H
            top_row = _box_top_row(rows, drawn_box_h + self._menu_h())
            out = self._out
            out.write(f"{_ESC}[?2004l")  # leave bracketed-paste mode
            # Restore the terminal's default keyboard reporting modes
            # so downstream (or re-entered) processes see the vanilla
            # behaviour they expect.  These are the disable / pop
            # counterparts of the enable sequences written in
            # :meth:`start`:
            #
            # * ``ESC[>4;0m`` — restore modifyOtherKeys to level 0.
            # * ``ESC[<u``    — pop one Kitty keyboard flag entry off
            #   the stack (matching the ``ESC[>1u`` push).
            out.write(KEYBOARD_PROTO_DISABLE)
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
        # Drop transient overlay/completion state so a later ``start()``
        # does not paint a phantom voice header/menu rows or wrongly
        # trip the "menu shrank" clear path in :meth:`_draw_locked`
        # (which compares the next ``_menu_h()`` against
        # ``_drawn_menu_h`` from this run).
        self.overlay = ""
        self.overlay_blink = False
        with self.lock:
            self._injected.clear()
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
            # Re-opt into extended-keyboard protocols after resume —
            # the shell may have popped them off while we were
            # suspended.  Mirrors the enable sequences in :meth:`start`.
            self._out.write(KEYBOARD_PROTO_ENABLE)
            # Force _draw_locked to re-emit the scroll region.
            self._rows = 0
            self._draw_locked()

    def redraw(self) -> None:
        """Redraw the box, preserving the output cursor position."""
        with self.lock:
            if self._active:
                self._draw_locked()

    def inject_line(self, text: str) -> None:
        """Queue *text* to be submitted exactly like a typed Enter line.

        Called from background threads (the ``/voice`` wake-word pump)
        to hand a programmatically produced line to whichever
        :func:`_pump_stdin` loop currently owns the box; the pump
        drains the queue and forwards each line to its ``on_submit``
        callback, so injected lines flow through the same idle-read /
        steering paths as keyboard input while the in-progress typed
        draft (:attr:`buf`) stays untouched.

        Args:
            text: The line to submit.
        """
        with self.lock:
            self._injected.append(text)

    def drain_injected(self) -> list[str]:
        """Return and clear all pending injected lines.

        Returns:
            The injected lines in FIFO order (possibly empty).
        """
        with self.lock:
            lines = self._injected
            self._injected = []
            return lines

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

        Callers must hold :attr:`lock`: the auto-dismiss path mutates
        the shared menu state, so a lock-free call would race the
        pump/draw threads (w2 F4).
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
            self._menu_repls = []
            self._menu_sel = 0
            self._menu_scroll = 0
        return h

    def _draw_locked(self) -> None:
        rows, _ = _term_size()
        cols = panel_cols()
        menu_h = self._menu_h()
        body_rows, is_placeholder = panel_body(
            self.buf, cols, cursor=self.cursor,
        )
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
        bottom = panel_bottom(self.status, cols)

        # In voice mode the transient indicator (``Listening ...`` /
        # ``Transcribing ...``) opens the header: it is rendered red
        # at the beginning of the top border, blinking only once the
        # sorcar wake word has been detected (``overlay_blink``).
        if self.overlay:
            top_line = voice_panel_top(
                self.overlay, self.title, cols, blink=self.overlay_blink,
            )
        else:
            top_line = f"{CYAN}{panel_top(self.title, cols)}{RESET}"
        out.write(f"{_ESC}[{box_top};1H{_ESC}[2K{top_line}")
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
            if is_placeholder and i == 0:
                rest_color = DIM
            else:
                rest_color = ""
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
        # menu above it).  The voice indicator lives in the header, so
        # the caret parks on the (still visible) edit buffer as usual.
        # ``body_cursor_col`` returns the body-row index of the caret
        # (0 for the first body row, growing with each embedded
        # newline).
        caret_row, col = body_cursor_col(self.buf, cols, self.cursor)
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
        with self.lock:
            cur = self.cursor
            self.buf = self.buf[:cur] + cleaned + self.buf[cur:]
            self.cursor = cur + len(cleaned)
        return True

    def _history_back(self) -> None:
        """Step one entry backwards through :attr:`history`."""
        with self.lock:
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
        with self.lock:
            if self._hist_idx is None:
                return
            if self._hist_idx < len(self.history) - 1:
                self._hist_idx += 1
                self.buf = self.history[self._hist_idx]
            else:
                self._hist_idx = None
                self.buf = self._hist_saved

    def _cursor_up_line(self) -> None:
        """Move the caret up one buffer line (textarea-style).

        Mirrors the chat webview textbox: the caret keeps its column
        (clamped to the target line's length) when the buffer has a
        line above, and jumps to the very start of the text when it is
        already on the first line.
        """
        with self.lock:
            line_start = self.buf.rfind("\n", 0, self.cursor) + 1
            if line_start == 0:
                self.cursor = 0
                return
            col = self.cursor - line_start
            prev_start = self.buf.rfind("\n", 0, line_start - 1) + 1
            prev_len = line_start - 1 - prev_start
            self.cursor = prev_start + min(col, prev_len)

    def _cursor_down_line(self) -> None:
        """Move the caret down one buffer line (textarea-style).

        Mirrors the chat webview textbox: the caret keeps its column
        (clamped to the target line's length) when the buffer has a
        line below, and jumps to the very end of the text when it is
        already on the last line.
        """
        with self.lock:
            line_end = self.buf.find("\n", self.cursor)
            if line_end < 0:
                self.cursor = len(self.buf)
                return
            line_start = self.buf.rfind("\n", 0, self.cursor) + 1
            col = self.cursor - line_start
            next_start = line_end + 1
            next_end = self.buf.find("\n", next_start)
            if next_end < 0:
                next_end = len(self.buf)
            self.cursor = next_start + min(col, next_end - next_start)

    def _arrow_up(self) -> None:
        """Handle the Up arrow (menu closed) with webview semantics.

        History cycling happens only when a browse is already in
        progress or the buffer is empty — exactly the condition the
        chat webview's ``cycleHistoryUp`` uses (``histIdx >= 0 ||
        !inp.value``).  Otherwise the caret moves up one line inside
        the buffer like a plain textarea.
        """
        if self._hist_idx is not None or not self.buf:
            self._history_back()
            # Refresh the menu against the recalled buffer so "fast
            # complete" tracks history navigation too.
            self._refresh_typing_menu()
        else:
            self._cursor_up_line()

    def _arrow_down(self) -> None:
        """Handle the Down arrow (menu closed) with webview semantics.

        History cycling happens only while a browse is in progress —
        the chat webview's ``cycleHistoryDown`` guard (``histIdx >=
        0``).  Otherwise the caret moves down one line inside the
        buffer like a plain textarea.
        """
        if self._hist_idx is not None:
            self._history_forward()
            self._refresh_typing_menu()
        else:
            self._cursor_down_line()

    def _nav_action(self, key: str) -> None:
        """Apply one navigation key to the box (arrows / Home / End).

        Shared by the CSI (``ESC[``) and SS3 (``ESC O``) branches of
        :meth:`feed` so both encodings behave identically: Up/Down
        navigate the in-place completion menu when it is open (and
        otherwise browse the input history / move the caret between
        buffer lines — see :meth:`_arrow_up` / :meth:`_arrow_down`),
        Left/Right move the edit cursor, and Home/End jump to the
        buffer's ends.

        Args:
            key: One of ``"up"``, ``"down"``, ``"left"``, ``"right"``,
                ``"home"``, ``"end"`` (see :data:`_NAV_FINALS`).
        """
        if key == "up":
            if self._menu_open:
                self._menu_move(-1)
            else:
                self._arrow_up()
        elif key == "down":
            if self._menu_open:
                self._menu_move(1)
            else:
                self._arrow_down()
        elif key == "left":
            with self.lock:
                if self.cursor > 0:
                    self.cursor -= 1
        elif key == "right":
            with self.lock:
                if self.cursor < len(self.buf):
                    self.cursor += 1
        elif key == "home":
            with self.lock:
                self.cursor = 0
        else:  # "end"
            with self.lock:
                self.cursor = len(self.buf)

    def _reset_completion_state(self) -> None:
        """Close the in-place completion menu and drop its state."""
        with self.lock:
            self._menu_open = False
            self._menu_items = []
            self._menu_repls = []
            self._menu_sel = 0
            self._menu_scroll = 0

    def flush_pending_escape(self) -> bool:
        """Resolve a deferred lone ESC byte as a bare ESC key press.

        :meth:`feed` defers a chunk-final ``\\x1b`` byte because it may
        be the first byte of an escape sequence split across reads.
        When no follow-up bytes arrive by the select pump's next idle
        tick (0.1 s — the remainder of a real terminal-generated
        sequence arrives immediately) the byte was a bare ESC key
        press, which dismisses the open completion menu — the same
        ttimeout-style disambiguation vim uses.  Longer pending
        prefixes (e.g. ``ESC [`` of a split arrow key) are genuine
        split sequences and stay deferred; a lone ESC with no menu
        open is left pending too (it is harmlessly swallowed by the
        next chunk, preserving the old behaviour).

        Returns:
            ``True`` when a pending lone ESC was consumed and the open
            menu was closed (the box was redrawn), ``False`` otherwise.
        """
        if self._pending_esc != "\x1b" or not self._menu_open:
            return False
        self._pending_esc = ""
        self._reset_completion_state()
        self.redraw()
        return True

    def _open_completion_menu(self) -> bool:
        """Pop the in-place completion menu using :attr:`completer_fn`.

        Queries the completer with the current edit buffer.  When the
        completer returns zero candidates the menu stays closed and
        the call is a no-op.  With exactly one candidate the menu is
        not opened — instead :attr:`buf` is replaced with the single
        candidate's replacement text (so trivial single-match
        completions behave like a normal Tab) and, on slash-command
        lines, the next-level menu is chained open immediately (see
        :meth:`_chain_slash_menu`).  With two or more candidates the
        menu is opened with the first candidate highlighted;
        :attr:`buf` is *not* modified until the user actually picks
        one with Tab.

        Returns:
            ``True`` when the menu changed visible state (opened, or
            the buffer was replaced by a single-candidate completion),
            so the caller knows a redraw is needed.
        """
        if self.completer_fn is None:
            return False
        repls, displays = _normalize_candidates(self.completer_fn(self.buf))
        if not repls:
            return False
        replaced = False
        with self.lock:
            if len(repls) == 1:
                self.buf = repls[0]
                self._hist_idx = None
                replaced = True
            else:
                self._menu_items = displays
                self._menu_repls = repls
                self._menu_sel = 0
                self._menu_scroll = 0
                self._menu_open = True
        if replaced:
            self._chain_slash_menu()
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
            repls, displays = self._last_completed_cands
        else:
            try:
                repls, displays = _normalize_candidates(completer(buf))
            except Exception:  # noqa: BLE001 - completer must not break editing
                logger.debug(
                    "completer raised during typing refresh", exc_info=True,
                )
                repls, displays = [], []
            # Drop no-op candidates whose replacement is exactly the
            # current buffer: previewing them is pure noise (accepting
            # would change nothing).  This also keeps the chained menu
            # after an accept (see :meth:`_chain_slash_menu`) from
            # re-offering the just-accepted line.
            keep = [i for i, r in enumerate(repls) if r != buf]
            if len(keep) != len(repls):
                repls = [repls[i] for i in keep]
                displays = [displays[i] for i in keep]
            self._last_completed_buf = buf
            self._last_completed_cands = (repls, displays)
        with self.lock:
            if buf != self.buf:
                # User kept typing while the completer ran; drop these
                # stale results — the next refresh will reflect the
                # now-current buffer.
                return False
            if not repls:
                if not self._menu_open:
                    return False
                self._reset_completion_state()
                return True
            self._menu_items = displays
            self._menu_repls = repls
            # Preserve the highlighted candidate across the refresh if
            # it is still in the list; otherwise fall back to the top.
            if prev_sel_text is not None and prev_sel_text in displays:
                self._menu_sel = displays.index(prev_sel_text)
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
        """Accept the highlighted candidate and chain the next menu.

        Replaces :attr:`buf` with the highlighted candidate's
        *replacement* text (which may differ from the displayed row —
        task-id rows show ``<id>: <description>`` but insert only the
        bare id), closes the menu, and on slash-command lines
        immediately re-opens the next-level menu (command → argument
        options → flag values) via :meth:`_chain_slash_menu` so the
        user never needs an extra keystroke to see what comes next.
        """
        with self.lock:
            if self._menu_open and self._menu_repls:
                self.buf = self._menu_repls[self._menu_sel]
                self._hist_idx = None
        # ``_reset_completion_state`` is the single source of truth for
        # tearing down menu state; reuse it instead of duplicating the
        # field resets here.
        self._reset_completion_state()
        self._chain_slash_menu()

    def _chain_slash_menu(self) -> None:
        """Pop the next-level completion menu after a slash-line accept.

        Chained completion: accepting a candidate on a ``/`` line
        immediately re-queries the completer against the new buffer so
        the next-level menu pops without any further keystroke —
        Tab-completing ``/resume`` instantly shows its argument options
        (``--task`` / ``--limit``), Tab-completing ``--task`` instantly
        shows the recent task ids (each displayed as ``<id>: <one-line
        task description>``), and accepting a task id shows the
        remaining options.  :meth:`_refresh_typing_menu` never
        auto-replaces the buffer (preview only), so chaining cannot
        recurse.  Non-slash accepts (``@``-mention file picker,
        predictive history) keep the menu closed, matching the
        prompt_toolkit fallback in cli_prompt.py.
        """
        if self.buf.lstrip().startswith("/"):
            self._refresh_typing_menu()

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
                # Modifier+Enter inserts a real newline into the buffer
                # instead of submitting the line.  This covers every
                # encoding terminals emit for Shift+Enter, Alt+Enter,
                # Ctrl+Enter, Cmd+Enter and arbitrary combinations:
                #
                # * ``ESC\r`` — portable xterm-style Alt+Enter
                # * ``ESC\n`` — tmux M-Enter / Alt+Enter on some
                #               terminals
                # * ``ESC[13;<m>u`` — kitty / CSI-u modifyOtherKeys=1
                # * ``ESC[27;<m>;13~`` — xterm modifyOtherKeys=2
                #
                # See :data:`_NEWLINE_AFTER_ESC` for the full table.
                # The tuple is iterated longest-first so a single-byte
                # ``\r`` cannot eat the leading byte of a CSI sequence
                # whose payload happens to start with another byte
                # (``[``) — but the explicit order also protects
                # against any future addition that *would* prefix-
                # collide.
                newline_insert = False
                for seq in _NEWLINE_AFTER_ESC:
                    if text.startswith(seq, i + 1):
                        with self.lock:
                            cur = self.cursor
                            self.buf = (
                                self.buf[:cur] + "\n" + self.buf[cur:]
                            )
                            self.cursor = cur + 1
                        # The modifier+Enter is a buffer edit; refresh
                        # the in-place completion menu so suggestions
                        # track multi-line input.
                        self._refresh_typing_menu()
                        changed = True
                        i += 1 + len(seq)
                        newline_insert = True
                        break
                if newline_insert:
                    continue
                # A chunk ending right at the ESC may be the first half
                # of a sequence split across reads; defer it so the next
                # chunk can complete (and swallow) the sequence.
                if i + 1 >= len(text):
                    self._pending_esc = text[i:]
                    break
                # Parse other CSI escape sequences: Up/Down arrows
                # navigate the in-place completion menu when it's
                # open; otherwise they browse the input history when
                # the buffer is empty (or a browse is in progress)
                # and move the caret between buffer lines otherwise
                # — the same split the chat webview textbox makes;
                # Left/Right move the edit cursor and Home/End jump
                # to the buffer's ends; the rest (F-keys, etc.) are
                # swallowed so their printable bytes do not type into
                # the buffer.
                if text[i + 1] == "[":
                    j = i + 2
                    while j < len(text) and not ("@" <= text[j] <= "~"):
                        j += 1
                    if j >= len(text):  # split mid-sequence
                        self._pending_esc = text[i:]
                        break
                    final = text[j]
                    seq = text[i + 2 : j]
                    # Arrow / Home / End dispatch is shared with the
                    # SS3 branch below via :meth:`_nav_action`; the
                    # tilde-terminated Home (``ESC[1~`` / ``ESC[7~``)
                    # and End (``ESC[4~`` / ``ESC[8~``) variants map
                    # onto the same actions.
                    nav_key: str | None = None
                    if seq == "":
                        nav_key = _NAV_FINALS.get(final)
                    elif final == "~" and seq in ("1", "7"):
                        nav_key = "home"
                    elif final == "~" and seq in ("4", "8"):
                        nav_key = "end"
                    if nav_key is not None:
                        self._nav_action(nav_key)
                        changed = True
                    elif final == "Z" and seq == "":  # Shift+Tab
                        if self._menu_open:
                            self._menu_move(-1)
                            changed = True
                    i = j + 1
                    continue
                # SS3 sequences (``ESC O <final>``): arrow keys in
                # application cursor mode (DECCKM) and F1–F4.
                # Up/Down behave exactly like their CSI twins above
                # (menu navigation / history browse / line movement),
                # Left/Right/Home/End move the edit cursor; the rest
                # are swallowed so their printable bytes must not be
                # typed into the buffer.
                if text[i + 1] == "O":
                    if i + 2 >= len(text):  # split mid-sequence
                        self._pending_esc = text[i:]
                        break
                    ss3 = text[i + 2]
                    # Same shared navigation dispatch as the CSI
                    # branch above (see :meth:`_nav_action`).
                    ss3_key = _NAV_FINALS.get(ss3)
                    if ss3_key is not None:
                        self._nav_action(ss3_key)
                        changed = True
                    i += 3
                    continue
                # A bare ESC followed by an unrecognized byte (Alt+key,
                # or ESC then a fast keystroke landing in the same
                # read): ESC is the universal "close the list" gesture,
                # so it dismisses the open completion menu.  The ESC
                # byte itself is swallowed either way.  (A chunk-FINAL
                # lone ESC is deferred above and resolved by
                # :meth:`flush_pending_escape` on the pump's next idle
                # tick.)
                if self._menu_open:
                    self._reset_completion_state()
                    changed = True
                i += 1
                continue
            if ch == "\r":
                # Bare CR is the Enter key (after raw mode disables
                # ICRNL the kernel no longer rewrites it to LF).  Enter
                # NEVER accepts a highlighted completion candidate —
                # except in the ``@``-mention file picker, where Enter
                # still inserts the highlighted path without
                # submitting.  For every other menu (predictive
                # history, slash commands, /model) the open menu is
                # simply dismissed (menu navigation never modifies
                # :attr:`buf`, so the typed text is intact) and the
                # buffer the user typed is submitted as-is.  Tab
                # accepts in every menu — matching the prompt_toolkit
                # dropdown in cli_prompt.py and the webview/webapp
                # picker in main.js.
                if self._menu_open:
                    if _AT_RE.search(self.buf):
                        # File picker: accept the mention, keep typing.
                        self._menu_accept()
                        changed = True
                        i += 1
                        continue
                    self._reset_completion_state()
                    changed = True
                # Universal backslash line-continuation.  On
                # terminals that cannot disambiguate Shift+Enter
                # from Enter (notably macOS Terminal.app), ending
                # the buffer with an unescaped ``\`` makes Enter
                # insert a newline into the buffer instead of
                # submitting — the same convention every POSIX
                # shell uses.  See
                # :func:`~kiss.agents.sorcar.cli_line_continuation.ends_with_line_continuation`.
                cont, keep = ends_with_line_continuation(self.buf)
                if cont:
                    with self.lock:
                        self.buf = self.buf[:keep] + "\n"
                        self._hist_idx = None
                    # A buffer edit — refresh the in-place
                    # completion menu so suggestions track the
                    # newly-added newline the same way the
                    # Shift+Enter newline-insert paths do.
                    self._refresh_typing_menu()
                    changed = True
                else:
                    with self.lock:
                        line = self.buf
                        self.buf = ""
                        self._hist_idx = None
                        self._hist_saved = ""
                    self._reset_completion_state()
                    changed = True
                    on_submit(line)
            elif ch == "\n":
                # Bare LF is Ctrl+J / Alt+Enter on terminals that
                # send a lone LF for that key (e.g. tmux M-Enter
                # decoded down to its newline byte) — it must INSERT
                # a newline into the buffer, never submit.  With
                # ICRNL disabled in start() a real Enter arrives as
                # ``\r`` and follows the branch above, so the two
                # are reliably distinguishable.
                with self.lock:
                    cur = self.cursor
                    self.buf = self.buf[:cur] + "\n" + self.buf[cur:]
                    self.cursor = cur + 1
                    self._hist_idx = None
                # Like the Shift+Enter newline-insert paths above,
                # refresh the in-place completion menu so suggestions
                # track multi-line input ("complete while typing").
                self._refresh_typing_menu()
                changed = True
            elif ch in ("\x7f", "\x08"):
                if self.cursor > 0:
                    with self.lock:
                        cur = self.cursor
                        self.buf = self.buf[: cur - 1] + self.buf[cur:]
                        self.cursor = cur - 1
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
                    with self.lock:
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
                # one.  Tab while the menu is already open ACCEPTS the
                # highlighted candidate (Enter never does — it submits
                # the typed buffer), matching the prompt_toolkit
                # dropdown in cli_prompt.py and the webview picker;
                # Up/Down/Shift+Tab navigate the list.  A single-item
                # preview is re-queried before accepting so a stale
                # candidate cannot overwrite the buffer.
                if self._menu_open:
                    if len(self._menu_items) == 1:
                        # Re-query before accepting so a stale
                        # single-item preview cannot overwrite the
                        # buffer with an out-of-date candidate.
                        # ``_open_completion_menu`` accepts a fresh
                        # single candidate directly — replacing the
                        # buffer and chaining the next-level menu on
                        # slash-command lines — and re-opens the menu
                        # when two or more candidates match now.
                        self._reset_completion_state()
                        self._open_completion_menu()
                        changed = True
                    else:
                        self._menu_accept()
                        changed = True
                elif self._open_completion_menu():
                    # ``_open_completion_menu`` is a no-op returning
                    # False when no ``completer_fn`` is registered.
                    changed = True
            elif ch >= " " and not ("\x80" <= ch <= "\x9f"):
                # The C1 control range (U+0080–U+009F) must never reach
                # the buffer: U+009B is a one-character CSI introducer
                # and would corrupt the terminal when redrawn.  DEL
                # (\x7f) is already consumed by the backspace branch.
                with self.lock:
                    cur = self.cursor
                    self.buf = self.buf[:cur] + ch + self.buf[cur:]
                    self.cursor = cur + 1
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


@contextlib.contextmanager
def _locked_interruptible(lock: threading.RLock) -> Iterator[None]:
    """Acquire *lock*, staying responsive to SIGINT while waiting.

    On CPython 3.14+ the ``_thread`` lock implementation is
    PyMutex-based and a blocking ``lock.acquire()`` parked in C never
    returns to the bytecode loop, so a pending ``SIGINT`` on the main
    thread cannot raise ``KeyboardInterrupt`` until the lock is won —
    a worker thread flooding stdout through :class:`_StdoutProxy`
    (which reacquires the unfair lock for every chunk) can therefore
    starve the main thread indefinitely with Ctrl+C never delivered.
    Acquiring in short timed slices returns control to bytecode
    between attempts, letting the pending signal handler run and
    ``KeyboardInterrupt`` propagate to the caller.

    Args:
        lock: The shared terminal lock to acquire.

    Yields:
        ``None`` once the lock is held; released on exit.
    """
    while not lock.acquire(timeout=0.2):
        pass
    try:
        yield
    finally:
        lock.release()


def _pump_stdin(
    box: _InputBox,
    is_done: Callable[[], bool],
    on_submit: Callable[[str], None],
    on_abort: Callable[[], None],
    on_eof: Callable[[], None] | None = None,
    on_stdin_close: Callable[[], None] | None = None,
    on_idle: Callable[[], None] | None = None,
) -> None:
    """Feed stdin through *box* until *is_done* returns ``True``.

    Shared select/read pump behind :meth:`SteeringSession._loop`,
    :meth:`AnchoredRepl.read_idle_line` and
    :meth:`AnchoredRepl.run_steering_loop`.  Polls stdin with a 0.1 s
    ``select`` timeout; on each idle tick it invokes *on_idle* (when
    given) and redraws the box after a terminal resize.  A
    ``KeyboardInterrupt`` raised out of ``select`` invokes *on_abort*
    and keeps looping — callers whose abort should end the pump make
    *is_done* observe the abort.

    Args:
        box: The anchored input box receiving raw keyboard bytes.
        is_done: Predicate polled each iteration; ``True`` ends the pump.
        on_submit: Forwarded to :meth:`_InputBox.feed` for completed lines.
        on_abort: Invoked on Ctrl+C (both in-band and out-of-band).
        on_eof: Forwarded to :meth:`_InputBox.feed` (Ctrl+D on empty
            buffer) when given.
        on_stdin_close: Invoked when ``os.read`` returns EOF (stdin
            closed).  EOF always removes the fd from the select set
            (it is permanent) so the pump idles instead of spinning;
            when ``None`` no callback is invoked.
        on_idle: Invoked once per select timeout while waiting;
            exceptions are logged and swallowed.
    """
    fd = sys.stdin.fileno()
    # ``read_fds`` is emptied once stdin hits EOF: a closed stdin stays
    # readable forever with ``os.read`` returning ``b""``, so keeping
    # the fd in the select set would degrade the pump into a 100 %-CPU
    # select→read→continue spin until ``is_done`` flips.  With the
    # empty set, ``select`` simply sleeps its 0.1 s timeout and the
    # idle branch (on_idle / resize polling) keeps running as before.
    read_fds = [fd]
    last_size = _term_size()
    while not is_done():
        # Every branch that touches the box (and therefore acquires
        # the shared terminal lock) runs inside
        # :func:`_locked_interruptible`: on CPython 3.14+ a plain
        # blocking acquire parked in PyMutex C code never returns to
        # bytecode, so a SIGINT arriving while a flooding worker holds
        # the unfair lock would otherwise never surface as
        # ``KeyboardInterrupt`` and the pump (main thread) would starve
        # with Ctrl+C undelivered.  The RLock is re-entrant, so the box
        # methods' internal ``with self.lock`` acquisitions are instant
        # while the pump already holds it.  ``KeyboardInterrupt``
        # raised anywhere in the iteration invokes *on_abort* exactly
        # like the in-``select`` Ctrl+C.
        try:
            # Programmatically injected lines (the ``/voice`` wake-word
            # pump) submit exactly like Enter-terminated typed lines,
            # then the loop condition is re-polled so a completed read
            # returns promptly instead of waiting out one more select
            # timeout.
            with _locked_interruptible(box.lock):
                injected = box.drain_injected()
            if injected:
                for line in injected:
                    on_submit(line)
                continue
            try:
                ready, _, _ = select.select(read_fds, [], [], 0.1)
            except (InterruptedError, OSError):
                continue
            if not ready:
                # A chunk-final lone ESC deferred by feed() with no
                # follow-up bytes by now was a bare ESC key press:
                # dismiss the open completion menu.
                with _locked_interruptible(box.lock):
                    box.flush_pending_escape()
                if on_idle is not None:
                    try:
                        on_idle()
                    except Exception:  # noqa: BLE001 - defensive
                        logger.debug("on_idle raised", exc_info=True)
                # Poll for terminal resizes so the box re-anchors
                # within one select timeout even when no key is
                # pressed.
                size = _term_size()
                if size != last_size:
                    last_size = size
                    with _locked_interruptible(box.lock):
                        box.redraw()
                continue
            try:
                data = os.read(fd, 4096)
            except (InterruptedError, OSError):
                continue
            if not data:
                # EOF on stdin is permanent — stop selecting on the fd
                # so the pump idles at the select timeout instead of
                # busy-spinning at 100 % CPU until ``is_done`` flips.
                read_fds = []
                if on_stdin_close is not None:
                    on_stdin_close()
                continue
            with _locked_interruptible(box.lock):
                box.feed(data, on_submit, on_abort, on_eof)
        except KeyboardInterrupt:
            on_abort()
            continue


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
            sys.stdout.write(QUESTION_FMT.format(question=question))
            sys.stdout.flush()
        # Dismiss any open in-place completion menu so the next Enter
        # in the question loop submits the user's answer instead of
        # silently picking a stale candidate from the previous edit.
        self.box._reset_completion_state()
        # The pump thread reads ``title`` under the shared lock inside
        # ``_draw_locked``; mutate (and later restore) it under the
        # same lock so a redraw can never render a torn title (w2 F5).
        with self.lock:
            prev_title = self.box.title
            self.box.title = ASK_TITLE
        self.box.redraw()
        # Drop any stale answer left in the queue by the previous
        # question's submit/clear race: ``_on_submit`` can observe
        # ``_question_pending`` still set in the window after the
        # previous ``get()`` returned but before its ``finally``
        # cleared the flag, parking a line in ``_answer_q`` that no
        # waiter ever consumes.  Without this drain the leftover
        # would be returned as THIS question's answer immediately,
        # before the user even sees the question.
        while True:
            try:
                self._answer_q.get_nowait()
            except queue.Empty:
                break
        self._question_pending.set()
        try:
            return self._answer_q.get()
        finally:
            self._question_pending.clear()
            with self.lock:
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
        with self.lock:
            # ``status`` is read by locked redraws; mutate it under
            # the same lock (w2 F5).
            self.box.status = QUEUED_STATUS_FMT.format(n=self._queued_count)
            sys.stdout.write(QUEUED_FMT.format(text=text))
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
            modified = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(worker.ident),
                ctypes.py_object(KeyboardInterrupt),
            )
            if modified > 1:  # pragma: no cover - CPython invariant
                # Per the CPython docs, a return value greater than
                # one is catastrophic: more than one thread state was
                # modified and the injection must be revoked
                # immediately with ``exc=NULL`` to undo the damage.
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(worker.ident), None
                )
                logger.warning(
                    "PyThreadState_SetAsyncExc modified %d thread"
                    " states for ident %s; injection revoked",
                    modified,
                    worker.ident,
                )
                return
            if modified == 0:
                # Stale ident: the worker exited between the
                # ``is_alive`` check and the injection — nothing to
                # interrupt any more.
                logger.debug(
                    "worker thread %s already gone; no interrupt sent",
                    worker.ident,
                )
                return
            worker.join(timeout=5.0)
            if worker.is_alive():
                # The async exception is only delivered when the
                # thread next executes Python bytecode; a worker
                # parked in C code (e.g. a blocking socket read) can
                # outlive the grace period.  Surface it instead of
                # silently leaking a still-running task the user was
                # told had been interrupted.
                logger.warning(
                    "worker thread did not stop within 5s after"
                    " KeyboardInterrupt injection; it may still be"
                    " running in the background"
                )

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
            try:
                self.box.start()
            except BaseException:
                # The restoring ``finally`` below is not reached when
                # ``start()`` raises here, so the streams must be
                # restored in place — otherwise the process keeps
                # writing through proxies of a dead box.
                sys.stdout = prev_stdout
                sys.stderr = prev_stderr
                raise
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
            if self._aborted.is_set():
                # Interrupt the worker BEFORE any teardown that needs
                # the shared terminal lock (``box.stop`` / ``redraw``).
                # A flooding worker reacquires the unfair lock for
                # every ``_StdoutProxy.write`` chunk, so a main thread
                # queued behind it inside ``box.stop`` can starve
                # indefinitely while the "interrupted" task keeps
                # running and spending budget.  ``_interrupt_worker``
                # itself never touches the terminal lock, so running
                # it first stops the flood and lets teardown proceed.
                self._interrupt_worker(worker)
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
            # The worker was already interrupted in the ``finally``
            # above (before box teardown, to avoid starving on the
            # terminal lock behind a flooding worker).
            raise KeyboardInterrupt
        if self._error is not None:
            raise self._error
        return self._result

    def _loop(self) -> None:
        # A Ctrl+C inside the pump calls ``_on_abort`` which sets
        # ``_done``, so the pump exits on its next ``is_done`` poll.
        _pump_stdin(
            self.box, self._done.is_set, self._on_submit, self._on_abort,
        )


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
        completer_fn: (
            Callable[[str], list[str] | list[tuple[str, str]]] | None
        ) = None,
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
        # Lines submitted but not yet returned by
        # :meth:`read_idle_line`.  A single ``os.read`` chunk can
        # contain several Enter-terminated lines (fast typing,
        # non-bracketed paste, piped stdin); ``feed`` invokes
        # ``on_submit`` once per line but the pump's ``is_done`` is
        # only polled between chunks, so all lines of the chunk land
        # in one pump run.  The surplus is buffered here and served by
        # subsequent :meth:`read_idle_line` calls instead of being
        # silently discarded.
        self._pending_lines: list[str] = []

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
        try:
            self.box.start()
        except BaseException:
            # ``__exit__`` never runs when ``__enter__`` raises, so the
            # streams must be restored here — otherwise the process is
            # left with its global stdout/stderr permanently proxied to
            # a dead box.
            sys.stdout = self._prev_stdout
            sys.stderr = self._prev_stderr
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Tear down the bottom box and restore the original streams."""
        del exc_type, exc, tb
        self.box.stop()
        sys.stdout = self._prev_stdout
        sys.stderr = self._prev_stderr

    def read_idle_line(self) -> str | None:
        """Read one line in idle mode through the anchored box.

        Up/Down arrows browse :attr:`_InputBox.history` when the
        buffer is empty (and move the caret between buffer lines
        otherwise, like the chat webview textbox), Tab cycles
        through the registered ``completer_fn`` candidates, and the
        line is echoed above the box (in the scroll region) after
        Enter so the user can see what they typed.

        Returns:
            The typed line (possibly empty), or ``None`` on Ctrl+D
            with an empty buffer.

        Raises:
            KeyboardInterrupt: When the user presses Ctrl+C.
        """
        # Serve a line buffered from an earlier multi-line input chunk
        # first — those lines were already submitted by the user and
        # must not be dropped.
        if self._pending_lines:
            return self._finish_idle_line(self._pending_lines.pop(0))
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

        def is_done() -> bool:
            return bool(result or eof_flag or abort_flag)

        # A closed stdin (EOF read) is treated exactly like Ctrl+D.
        _pump_stdin(
            self.box, is_done, on_submit, on_abort,
            on_eof=on_eof, on_stdin_close=on_eof,
        )
        if abort_flag:
            # Lines submitted in the SAME input chunk as the Ctrl+C
            # were already Enter-terminated by the user and must not
            # be dropped: buffer them so the next read serves them,
            # exactly like buffered lines from an EARLIER chunk (which
            # always survived a Ctrl+C via ``_pending_lines``).  The
            # old code raised before looking at ``result``, silently
            # discarding e.g. the "deploy" of a "deploy\n" + Ctrl+C
            # chunk while the two-chunk variant kept it (w3 F6).
            self._pending_lines.extend(result)
            raise KeyboardInterrupt
        if eof_flag and not result:
            return None
        # A single input chunk can submit several lines (fast typing,
        # non-bracketed paste, piped stdin).  Return the first and
        # buffer the rest for subsequent calls — the old code silently
        # discarded everything after ``result[0]``.  When a line and
        # Ctrl+D/EOF arrive in the same chunk the line wins; the EOF
        # is permanent and resurfaces on the next call via
        # ``on_stdin_close``.
        line = result[0]
        self._pending_lines.extend(result[1:])
        return self._finish_idle_line(line)

    def _finish_idle_line(self, line: str) -> str:
        """Record *line* in history, echo it above the box, return it.

        Common tail of :meth:`read_idle_line` shared by the fresh-pump
        path and the buffered-surplus path so both record history and
        echo identically.

        Args:
            line: The submitted line to hand back to the REPL.

        Returns:
            *line*, unchanged.
        """
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
        try:
            _pump_stdin(
                self.box, is_done, on_submit, on_abort, on_idle=on_idle,
            )
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
