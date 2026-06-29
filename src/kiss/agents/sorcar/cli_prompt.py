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
from prompt_toolkit.key_binding import KeyBindings

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


@_KEY_BINDINGS.add("escape", "enter")
def _newline_alt_enter(event: KeyPressEvent) -> None:
    """Alt+Enter (a.k.a. Meta+Enter / Esc+Enter) inserts a real newline.

    This is the portable multi-line key — it works in every terminal
    that delivers ``ESC <key>`` for Meta-modified keypresses (macOS
    Terminal.app, iTerm2, gnome-terminal, xterm, …) without any
    special keyboard-protocol opt-in.
    """
    event.current_buffer.insert_text("\n")


@_KEY_BINDINGS.add("c-j")
def _newline_ctrl_j(event: KeyPressEvent) -> None:
    """Ctrl+J inserts a newline.

    Ctrl+J transmits a literal Linefeed (``\\n``).  Many terminals
    deliver this when the user presses Shift+Enter without a CSI-u /
    modifyOtherKeys binding active, so Ctrl+J is a reliable
    secondary multi-line key.
    """
    event.current_buffer.insert_text("\n")


# Shift+Enter as delivered by terminals that opt in to the CSI-u
# keyboard protocol (kitty, foot, WezTerm, …): ``ESC[13;2u``.  The
# sequence is *not* part of prompt_toolkit's pre-mapped ANSI table,
# so the parser falls back to per-character delivery and our tuple
# binding matches it.  (The other common Shift+Enter encoding —
# xterm's modifyOtherKeys ``ESC[27;2;13~`` — is pre-mapped by
# prompt_toolkit to :data:`Keys.ControlM`, i.e. plain Enter, so on
# those terminals the user must use Alt+Enter or Ctrl+J for newlines.)
@_KEY_BINDINGS.add("escape", "[", "1", "3", ";", "2", "u")
def _newline_shift_enter_csi_u(event: KeyPressEvent) -> None:
    """kitty/foot/WezTerm CSI-u Shift+Enter (``ESC[13;2u``) inserts newline."""
    event.current_buffer.insert_text("\n")


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
            # Multi-line input: Alt+Enter / Ctrl+J / Shift+Enter
            # insert a real ``\n`` into the buffer (see the bindings
            # at module level); plain Enter still submits via the
            # ``~completion_is_selected`` Enter binding above.  Without
            # ``multiline=True`` prompt_toolkit would short-circuit
            # Enter to "accept-line" before our key bindings ever
            # ran, so the user could never type a newline.
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

    def read(self, prompt: str) -> str:
        """Read one line, rendering *prompt* (which may contain ANSI).

        Args:
            prompt: Prompt text, possibly containing SGR colour codes.

        Returns:
            The line typed by the user.

        Raises:
            EOFError: On Ctrl+D.
            KeyboardInterrupt: On Ctrl+C.
        """
        return self.session.prompt(ANSI(prompt))
