# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""prompt_toolkit input line for the sorcar REPL.

Provides the interactive dropdown that :mod:`readline` cannot render:
as soon as ``@`` is typed the file/folder picker pops up under the
input line, Up/Down (and Tab) move through the candidates, and Tab or
Enter inserts the highlighted ``./<path>`` mention without submitting
the line.  The same live menu serves ``/`` slash commands,
``/model <partial>`` model names, and whole-line predictive
completion: whenever the typed prefix matches one or more prior tasks
the menu pops with all of them so Up/Down + Tab/Enter can pick one
without re-typing.

The candidate lists come from the very same
:class:`~kiss.agents.sorcar.cli_repl.CliCompleter` backend used by the
readline fallback, so both input paths complete identically.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import completion_is_selected
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.dimension import Dimension

from kiss.agents.sorcar.cli_panel import MIN_BODY_ROWS as _MIN_BODY_ROWS
from kiss.agents.sorcar.persistence import _load_file_usage
from kiss.agents.vscode.helpers import rank_file_suggestions

# ANSI styling that matches the framed input panel (kept local so this
# module stays self-contained for the prompt_continuation callable).
_ESC = "\x1b"
_CYAN = f"{_ESC}[36m"
_RESET = f"{_ESC}[0m"

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document
    from prompt_toolkit.key_binding import KeyPressEvent

    from kiss.agents.sorcar.cli_repl import CliCompleter

logger = logging.getLogger(__name__)

# Trailing ``@<partial-path>`` token that triggers the file/folder picker.
_AT_RE = re.compile(r"@([^\s]*)$")
# ``/model <partial>`` line whose partial model name is fast-completed.
_MODEL_CMD_RE = re.compile(r"^/model\s+(.*)$")

# Modifier+Enter escape sequences emitted by terminals via either
# xterm's ``modifyOtherKeys`` mode (``ESC[27;<mod>;13~``) or the
# kitty/CSI-u keyboard protocol (``ESC[13;<mod>u``).  prompt_toolkit
# pre-maps the modifyOtherKeys variants for modifiers 2/5/6 to
# :data:`Keys.ControlM` in
# :data:`prompt_toolkit.input.ansi_escape_sequences.ANSI_SEQUENCES`
# *before* any key bindings run, so without :func:`_unmap_enter_aliases`
# below Shift/Ctrl/Ctrl-Shift+Enter on iTerm2 / macOS Terminal.app /
# the VS Code integrated terminal would all be indistinguishable from
# plain Enter and submit the buffer.  We delete those entries from the
# table so the raw escape sequence reaches the tuple key-bindings
# registered further down, which insert a real newline.
_MODIFY_OTHER_KEYS_ENTER = (
    "\x1b[27;2;13~",   # Shift+Enter
    "\x1b[27;3;13~",   # Alt+Enter
    "\x1b[27;4;13~",   # Alt+Shift+Enter
    "\x1b[27;5;13~",   # Ctrl+Enter
    "\x1b[27;6;13~",   # Ctrl+Shift+Enter
    "\x1b[27;7;13~",   # Ctrl+Alt+Enter
    "\x1b[27;8;13~",   # Ctrl+Alt+Shift+Enter
    # Meta-bit-set modifiers (``meta`` == Cmd on macOS in terminals
    # that report it — iTerm2 with "Report modifiers using CSI u" or
    # the modifyOtherKeys=2 equivalent, kitty/foot/WezTerm under the
    # CSI-u keyboard protocol).  Cmd+Enter / Cmd+Shift+Enter / … must
    # insert a newline (not autocomplete / not submit) the same way
    # Shift+Enter does, so we cover every mod value 9..16.
    "\x1b[27;9;13~",   # Cmd/Meta+Enter
    "\x1b[27;10;13~",  # Cmd+Shift+Enter
    "\x1b[27;11;13~",  # Cmd+Alt+Enter
    "\x1b[27;12;13~",  # Cmd+Alt+Shift+Enter
    "\x1b[27;13;13~",  # Cmd+Ctrl+Enter
    "\x1b[27;14;13~",  # Cmd+Ctrl+Shift+Enter
    "\x1b[27;15;13~",  # Cmd+Ctrl+Alt+Enter
    "\x1b[27;16;13~",  # Cmd+Ctrl+Alt+Shift+Enter
)


def _unmap_enter_aliases() -> None:
    """Drop pre-mapped modifier+Enter aliases from prompt_toolkit's table.

    Idempotent: deletes only the entries that map a modifier+Enter
    modifyOtherKeys sequence to :data:`Keys.ControlM` so that our tuple
    key-bindings below see the raw escape sequence and can insert a
    newline.  Other ANSI sequences are left untouched.
    """
    for seq in _MODIFY_OTHER_KEYS_ENTER:
        ANSI_SEQUENCES.pop(seq, None)


_unmap_enter_aliases()


_KEY_BINDINGS = KeyBindings()


@_KEY_BINDINGS.add("enter", filter=completion_is_selected)
def _accept_completion_enter(event: KeyPressEvent) -> None:
    """Enter on a highlighted completion inserts it instead of submitting.

    Navigating the menu (Up/Down/Tab) already placed the highlighted
    candidate's text in the buffer; closing the completion state keeps
    that text and lets the user continue editing.  Without a selection
    this binding submits the multi-line buffer instead (see
    :func:`_submit_enter`).
    """
    event.current_buffer.complete_state = None


@_KEY_BINDINGS.add("tab", filter=completion_is_selected)
def _accept_completion_tab(event: KeyPressEvent) -> None:
    """Tab on a highlighted completion confirms it (same as Enter)."""
    event.current_buffer.complete_state = None


@_KEY_BINDINGS.add("enter", filter=~completion_is_selected)
def _submit_enter(event: KeyPressEvent) -> None:
    """Enter submits the (possibly multi-line) buffer.

    The session runs with ``multiline=True`` so prompt_toolkit's
    default Enter binding inserts a newline; we override that here so
    Enter still means *submit* — matching the "type a task, then
    Enter" hint in the framed input panel.  Newlines are entered via
    the alternative bindings below (Alt+Enter, Ctrl+J, Shift+Enter).
    """
    event.current_buffer.validate_and_handle()


def _insert_newline(event: KeyPressEvent) -> None:
    """Cancel any open completion menu and insert a literal newline.

    The cancel step is what distinguishes the modifier+Enter family
    from plain Enter: when the user is navigating the live completion
    dropdown (Up/Down highlights a candidate so
    :attr:`~prompt_toolkit.buffer.Buffer.complete_state` holds a
    selected completion and the buffer text shows the highlighted
    candidate), pressing Shift / Alt / Option / Ctrl / Command+Enter
    must **restore the originally-typed text** and add a newline — not
    silently accept the autocomplete the user did not ask for.

    :meth:`~prompt_toolkit.buffer.Buffer.cancel_completion` is the
    documented prompt_toolkit entry point that calls
    :meth:`~prompt_toolkit.buffer.Buffer.go_to_completion(None)`,
    restoring ``complete_state.original_document``, and then clears
    ``complete_state`` so the menu disappears.  Calling it on a buffer
    without an open menu is a no-op (the ``if self.complete_state``
    guard inside :meth:`cancel_completion`), so this helper is safe
    for both the "menu open" and "no menu" code paths.
    """
    event.current_buffer.cancel_completion()
    event.current_buffer.insert_text("\n")


@_KEY_BINDINGS.add("escape", "enter")
def _newline_alt_enter(event: KeyPressEvent) -> None:
    """Alt+Enter (a.k.a. Meta+Enter / Esc+Enter) inserts a real newline.

    This is the portable multi-line key — it works in every terminal
    that delivers ``ESC <key>`` for Meta-modified keypresses (macOS
    Terminal.app, iTerm2, gnome-terminal, xterm, …) without any
    special keyboard-protocol opt-in.  Any open completion menu is
    dismissed (restoring the originally-typed text) before the newline
    is inserted, so Option+Enter never accepts a highlighted
    autocomplete behind the user's back.
    """
    _insert_newline(event)


@_KEY_BINDINGS.add("c-j")
def _newline_ctrl_j(event: KeyPressEvent) -> None:
    """Ctrl+J inserts a newline.

    Ctrl+J transmits a literal Linefeed (``\\n``).  Many terminals
    deliver this when the user presses Shift+Enter without a CSI-u /
    modifyOtherKeys binding active, so Ctrl+J is a reliable
    secondary multi-line key.  Like the other modifier+Enter
    bindings, any open completion menu is dismissed first so the
    highlighted completion is not accepted.
    """
    _insert_newline(event)


# Modifier+Enter escape sequences delivered by modern terminals as
# either the xterm ``modifyOtherKeys`` form ``ESC[27;<mod>;13~`` or the
# kitty/CSI-u form ``ESC[13;<mod>u`` — with ``<mod>`` = 2 (Shift),
# 3 (Alt), 4 (Alt+Shift), 5 (Ctrl), 6 (Ctrl+Shift), 7 (Ctrl+Alt),
# 8 (Ctrl+Alt+Shift).  prompt_toolkit pre-maps the modifyOtherKeys
# forms for 2/5/6 to :data:`Keys.ControlM` (plain Enter); the
# :func:`_unmap_enter_aliases` call above removes them so the parser
# falls back to per-character delivery and the tuple bindings below
# match the raw escape sequence, inserting a real ``\n`` into the
# buffer.  Every modifier+Enter combination would otherwise be
# ambiguous with plain Enter on iTerm2 / macOS Terminal.app / the
# VS Code integrated terminal — see the regression tests in
# ``tests/agents/sorcar/test_cli_multiline_input.py``.


def _bind_newline_sequence(*keys: str) -> None:
    """Register *keys* as a multi-key binding that inserts ``\\n``.

    Routes through :func:`_insert_newline` so the modifier+Enter
    escape sequences also dismiss any open completion menu (restoring
    the originally-typed text) before adding the newline — preventing
    the user-reported bug where Shift/Alt/Ctrl+Enter silently accepted
    the highlighted autocomplete instead of inserting a newline.
    """

    @_KEY_BINDINGS.add(*keys)
    def _newline(event: KeyPressEvent) -> None:
        _insert_newline(event)


def _sequence_keys(seq: str) -> tuple[str, ...]:
    """Split an escape sequence into the per-character tuple keys.

    prompt_toolkit's :class:`KeyBindings` matches multi-character
    escape sequences as a tuple of single-character / named keys; the
    leading ``ESC`` is the symbolic ``"escape"`` key and every other
    byte becomes its own one-character entry.
    """
    assert seq.startswith("\x1b"), seq
    return ("escape", *tuple(seq[1:]))


# xterm modifyOtherKeys: ``ESC[27;<mod>;13~`` for every supported modifier.
for _seq in _MODIFY_OTHER_KEYS_ENTER:
    _bind_newline_sequence(*_sequence_keys(_seq))

# kitty/CSI-u: ``ESC[13;<mod>u`` for Shift / Alt / Ctrl / Ctrl-Shift / …
# and the Meta-bit-set combinations 9..16 (Cmd+Enter and friends on
# macOS terminals that report the Meta modifier).  Two-digit modifier
# values are split across two tuple keys because the prompt_toolkit
# parser matches one byte at a time.
for _mod in ("2", "3", "4", "5", "6", "7", "8"):
    _bind_newline_sequence("escape", "[", "1", "3", ";", _mod, "u")
for _two_digit in ("9", "10", "11", "12", "13", "14", "15", "16"):
    _bind_newline_sequence(
        "escape", "[", "1", "3", ";", *tuple(_two_digit), "u",
    )


def _prompt_continuation(
    width: int, line_number: int, wrap_count: int,
) -> ANSI:
    """Render the left margin for wrapped / multi-line input rows.

    The framed input panel paints a cyan ``│`` on its first row; for
    every subsequent visual row (whether the user pressed
    Alt+Enter/Shift+Enter or the line wrapped because of
    ``wrap_lines=True``) prompt_toolkit calls this function, which
    returns ``│`` + one space so the panel border stays continuous.

    Args:
        width: The first-line prompt's display width; unused — the
            continuation line is the same fixed two columns.
        line_number: Zero-based index of the current visual row;
            unused.
        wrap_count: Number of times the row was wrapped because of
            ``wrap_lines=True``; unused.

    Returns:
        ANSI-styled ``│ `` so wrapped rows keep the panel left border.
    """
    del width, line_number, wrap_count
    return ANSI(f"{_CYAN}│{_RESET} ")


class PtkCompleter(Completer):
    """prompt_toolkit adapter over the REPL's :class:`CliCompleter`.

    Yields :class:`Completion` objects for the live dropdown:
    ``@``-mention files/folders (shown as the bare path, inserted as
    ``./<path> ``), slash commands with their help text, and
    ``/model`` model names.  Whole-line predictive matches are offered
    only when completion is explicitly requested with Tab.

    Attributes:
        cli: The shared backend that scans files and ranks candidates.
    """

    def __init__(self, cli_completer: CliCompleter) -> None:
        self.cli = cli_completer

    def get_completions(
        self, document: Document, complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        """Yield the dropdown candidates for the current input state.

        Args:
            document: The input buffer; only text before the cursor is
                considered, matching the readline completer.
            complete_event: Distinguishes typing-triggered completion
                from an explicit Tab press (unused: every category — the
                ``@``-mention picker, slash commands, ``/model`` names,
                and the whole-line predictive list — pops while typing
                so Up/Down + Tab/Enter can pick a match).

        Returns:
            The completions to display, best match first.
        """
        del complete_event  # All categories pop while typing.
        line = document.text_before_cursor
        at = _AT_RE.search(line)
        if at:
            return self._at_mention_completions(at.group(1))
        model_cmd = _MODEL_CMD_RE.match(line)
        if model_cmd:
            return self._model_completions(model_cmd.group(1))
        stripped = line.strip()
        if stripped.startswith("/") and " " not in stripped:
            return self._slash_completions(line)
        return self._predictive_completions(line)

    def _at_mention_completions(self, query: str) -> Iterable[Completion]:
        """Build the ``@``-mention picker entries for *query*.

        Each entry displays the bare relative path (folders end in
        ``/``) and inserts ``./<path> `` in place of the
        ``@<query>`` token, exactly like the readline completer and the
        VS Code extension.
        """
        usage = _load_file_usage()
        ranked = rank_file_suggestions(self.cli._files(), query, usage)
        for item in ranked:
            path = item["text"]
            if path.endswith("/"):
                meta = "folder"
            elif item["type"] == "frequent":
                meta = "recent"
            else:
                meta = "file"
            yield Completion(
                f"./{path} ",
                start_position=-(len(query) + 1),
                display=path,
                display_meta=meta,
            )

    def _model_completions(self, query: str) -> Iterable[Completion]:
        """Build ``/model <name>`` entries for the partial *query*."""
        from kiss.core.models.model_info import rank_model_suggestions

        for name in rank_model_suggestions(query):
            yield Completion(name, start_position=-len(query))

    def _slash_completions(self, line: str) -> Iterable[Completion]:
        """Build slash-command entries (built-in first, then custom)."""
        from kiss.agents.sorcar.cli_repl import SLASH_COMMANDS

        for match in self.cli._slash_matches(line):
            cmd = match.strip()
            yield Completion(
                match,
                start_position=-len(line),
                display=cmd,
                display_meta=SLASH_COMMANDS.get(cmd, "custom command"),
            )

    def _predictive_completions(self, line: str) -> Iterable[Completion]:
        """Build the whole-line predictive dropdown entries.

        Each match replaces the entire typed line.  The suffix added by
        the completion is shown as the display text (so the menu is not
        cluttered with the prefix the user already typed), and the
        ``history`` meta marks where the suggestion came from.
        """
        for cand in self.cli._predictive_matches(line):
            suffix = cand[len(line):] if cand.startswith(line) else cand
            yield Completion(
                cand,
                start_position=-len(line),
                display=suffix or cand,
                display_meta="history",
            )


def _migrate_readline_history(history_path: Path, ptk_path: Path) -> None:
    """Seed the prompt_toolkit history from the old readline history.

    Runs once: when *ptk_path* does not exist yet but the readline
    history at *history_path* does, its lines are rewritten in
    :class:`FileHistory` format so Up-arrow history survives the
    switch to prompt_toolkit.
    """
    if ptk_path.exists() or not history_path.exists():
        return
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
        with ptk_path.open("w", encoding="utf-8") as fh:
            for line in lines:
                if line.strip():
                    fh.write(f"\n# migrated-from-readline\n+{line}\n")
    except OSError:  # pragma: no cover - disk/permission error
        logger.debug("readline history migration failed", exc_info=True)


def _enforce_min_input_height(
    session: PromptSession[str], min_rows: int,
) -> None:
    """Pin the prompt_toolkit input window to at least *min_rows* rows.

    :class:`PromptSession` builds its layout lazily and the input
    :class:`~prompt_toolkit.layout.containers.Window` (wrapping the
    default :class:`~prompt_toolkit.buffer.Buffer`'s control) defaults
    to ``Dimension()`` — auto-sizing based on the buffer's line count.
    A one-line buffer therefore paints a one-row input area, which
    leaves the framed input dialog drawn around the prompt visually
    cramped (the user only sees one line of typing space even though
    the box's purpose is multi-line task input).

    Walk the layout, find the Window whose control's buffer is the
    session's default buffer, and replace its ``height`` with a
    :class:`Dimension` whose ``min`` is *min_rows* so prompt_toolkit
    reserves at least *min_rows* terminal rows for the input area on
    every render — matching the steering box's
    :data:`~kiss.agents.sorcar.cli_panel.MIN_BODY_ROWS` floor.
    Buffers with more lines still grow the area dynamically because
    ``Dimension(min=...)`` leaves ``preferred`` / ``max`` unconstrained.

    Args:
        session: The freshly constructed :class:`PromptSession`.
        min_rows: Floor for the input area's row count.
    """
    target = session.default_buffer
    for container in session.layout.walk():
        if not isinstance(container, Window):
            continue
        control = container.content
        if getattr(control, "buffer", None) is target:
            container.height = Dimension(min=min_rows)
            return


class PtkLineReader:
    """Reads REPL input lines with a live completion dropdown.

    Wraps a :class:`PromptSession` configured so the menu pops while
    typing (``@`` immediately shows the file/folder list), Up/Down
    navigate it, and Tab/Enter confirm the highlighted entry via the
    bindings in ``_KEY_BINDINGS``.  History is persisted per working
    directory next to the readline history file.
    """

    def __init__(self, completer: CliCompleter, history_path: Path) -> None:
        ptk_path = history_path.with_name(history_path.name + ".ptk")
        _migrate_readline_history(history_path, ptk_path)
        self.session: PromptSession[str] = PromptSession(
            completer=PtkCompleter(completer),
            complete_while_typing=True,
            key_bindings=_KEY_BINDINGS,
            history=FileHistory(str(ptk_path)),
            # Multi-line input: Alt+Enter / Ctrl+J / Shift+Enter /
            # Ctrl+Enter / Ctrl+Shift+Enter all insert a real ``\n``
            # into the buffer (see the bindings at module level — both
            # the xterm ``modifyOtherKeys`` and the kitty/CSI-u
            # encodings are covered); plain Enter still submits via
            # the ``~completion_is_selected`` Enter binding above.
            # Without ``multiline=True`` prompt_toolkit would short-
            # circuit Enter to "accept-line" before our key bindings
            # ever ran, so the user could never type a newline.
            multiline=True,
            # Word-wrap long lines onto the next visual row instead of
            # scrolling the input horizontally off-screen.  The
            # framed panel paints a ``│`` on each visual row via
            # :func:`_prompt_continuation`, so the box stays closed
            # even when the typed task wraps.
            wrap_lines=True,
            prompt_continuation=_prompt_continuation,
            # Reserve enough screen rows for the longest typing-triggered
            # menu — currently the ``/`` slash-command list with every
            # built-in command from :data:`SLASH_COMMANDS` (and a few
            # custom commands).  prompt_toolkit's :class:`CompletionsMenu`
            # caps its own height at 16 rows, so reserving the same number
            # makes every slash command visible at once instead of being
            # clipped to the previous default of 8 (which forced the user
            # to scroll Up/Down to discover the rest).
            reserve_space_for_menu=16,
        )
        _enforce_min_input_height(self.session, _MIN_BODY_ROWS)

    def read(self, prompt: str) -> str:
        """Read one line, rendering *prompt* (which may contain ANSI).

        Opts into the terminal's extended-keyboard protocols for the
        duration of the prompt so modifier+Enter chords (Shift+Enter,
        Ctrl+Enter, Alt+Enter, Cmd+Enter and combinations) emit
        *distinct* byte sequences the tuple key-bindings registered at
        module load can match as newline-insert instead of a plain
        Enter (submit).  Without these enable sequences most
        terminals — iTerm2, macOS Terminal.app, the VS Code integrated
        terminal, kitty / WezTerm / foot / ghostty (which ignore
        modifyOtherKeys entirely in favour of the Kitty keyboard
        protocol) — deliver Shift+Enter as a bare ``\\r``,
        indistinguishable from plain Enter, and the whole multi-line
        UX breaks:

        * ``ESC[>4;2m``  — xterm ``modifyOtherKeys`` level 2.  Makes
          Shift/Ctrl/Alt+Enter emit ``ESC[27;<m>;13~``.  Supported by
          xterm, iTerm2, WezTerm, Alacritty and tmux (with
          ``extended-keys always`` + ``extkeys`` feature).
        * ``ESC[>1u``    — Kitty keyboard protocol, push flag 1
          (disambiguate escape codes).  Makes Shift+Enter emit
          ``ESC[13;<m>u``.  Supported by kitty, WezTerm, foot,
          ghostty and (increasingly) other terminals.

        These are the same two enable / disable pairs written by
        :meth:`~kiss.agents.sorcar.cli_steering._InputBox.start` /
        :meth:`~kiss.agents.sorcar.cli_steering._InputBox.stop` for
        the mid-task steering box, so both the initial task prompt
        and the running-task steering box behave identically on every
        terminal.  Terminals that don't support one/both of these
        silently ignore the CSI.  The matching disable sequences —
        ``ESC[>4;0m`` (restore modifyOtherKeys level 0) and
        ``ESC[<u`` (pop the Kitty keyboard flag entry we pushed) —
        are written in the ``finally`` block on exit so a
        subsequently-spawned child process does not inherit our mode
        and we do not leak a stack entry into the shell.  All four
        writes are no-ops on :class:`prompt_toolkit.output.DummyOutput`
        (used by the pipe-based regression tests).

        Args:
            prompt: Prompt text, possibly containing SGR colour codes.

        Returns:
            The line typed by the user.

        Raises:
            EOFError: On Ctrl+D.
            KeyboardInterrupt: On Ctrl+C.
        """
        output = self.session.output
        output.write_raw(f"{_ESC}[>4;2m{_ESC}[>1u")
        output.flush()
        try:
            return self.session.prompt(ANSI(prompt))
        finally:
            try:
                output.write_raw(f"{_ESC}[>4;0m{_ESC}[<u")
                output.flush()
            except Exception:  # pragma: no cover - best-effort restore
                logger.debug(
                    "failed to disable extended keyboard protocols",
                    exc_info=True,
                )
