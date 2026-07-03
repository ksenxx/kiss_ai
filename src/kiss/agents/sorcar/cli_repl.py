# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Claude-Code-style interactive REPL for the ``sorcar`` command line.

Activated when neither ``-t/--task`` nor ``-f/--file`` is passed: the
agent runs in a stateful loop so that, exactly like the bare ``claude``
command, the prompt returns and waits for the next instruction after a
task finishes instead of exiting.

The input line reproduces the VS Code extension's input-textbox "fast
completes" using the very same backend helpers, so behaviour matches the
extension precisely:

* ``@``-mention file picker — :func:`rank_file_suggestions` over
  :func:`_scan_files`, inserting ``./<path>`` (mirrors the webview's
  ``insertAtMention``).
* predictive whole-line completion — :func:`_prefix_match_tasks` lists
  every recent task starting with the typed prefix, falling back to an
  active-file identifier clipped with
  :func:`clip_autocomplete_suggestion` (mirrors the webview ``ghost``
  event produced by ``_AutocompleteMixin._complete``).  In the
  prompt_toolkit input these matches pop as a live dropdown menu so
  Up/Down + Tab/Enter pick the desired completion without re-typing.
* ``/`` slash commands matching Claude Code's quick commands
  (``/help``, ``/clear``, ``/resume``, ``/model``, ``/model list``,
  ``/cost``, ``/context``, ``/exit`` …).
* custom slash commands defined as Markdown files in
  ``~/.kiss/commands`` (user) and ``<project>/.kiss/commands``
  (project), plus Claude Code's ``~/.claude/commands`` and
  ``<project>/.claude/commands`` — see
  :mod:`kiss.agents.sorcar.custom_commands`.  The file
  body is a prompt template (``$ARGUMENTS``/``$1``…/``@{file}``/shell
  injection) that is expanded and run as a task; ``/commands`` lists
  them and Tab completes their names.
* agent skills (``<name>/SKILL.md`` directories) following the
  `Agent Skills <https://agentskills.io>`_ standard — discovered from
  ``~/.kiss/skills``, ``<project>/.kiss/skills``, Claude Code's
  ``~/.claude/skills`` and ``<project>/.claude/skills``, the
  cross-client ``.agents/skills`` convention, and the skills bundled
  with Sorcar; ``/skills`` lists them and ``/skills <name>`` shows one
  — see :mod:`kiss.agents.sorcar.skills`.
* MCP servers configured in ``~/.kiss/mcp.json``,
  ``<project>/.kiss/mcp.json``, and Claude Code's ``<project>/.mcp.json``
  — ``/mcp`` lists them with live connection status; management
  (``add``/``list``/``get``/``remove``/``auth``/``logout``/``debug``)
  lives in the ``sorcar mcp`` subcommand — see
  :mod:`kiss.agents.sorcar.mcp_servers` and
  :mod:`kiss.agents.sorcar.mcp_cli`.
* ``/model``-name fast completion — :func:`rank_model_suggestions` over
  the generation-capable models in
  :mod:`kiss.core.models.model_info` (preferring providers whose API key
  is configured), so ``/model <partial>`` completes to a real model name.

On an interactive TTY the input line is read through
:mod:`prompt_toolkit` (see :mod:`kiss.agents.sorcar.cli_prompt`): typing
``@`` immediately pops the file/folder picker under the line, Up/Down
navigate it, and Tab/Enter insert the highlighted ``./<path>``
mention; the same live menu serves ``/`` commands and ``/model`` names,
with history (Up/Down), reverse search (Ctrl+R), and emacs-style line
editing built in.  Off-TTY (or if prompt_toolkit fails to initialise)
the loop falls back to :mod:`readline`, where TAB triggers the same
completions, and finally to a plain :func:`input` loop when
``readline`` is unavailable too (e.g. stock Windows).
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
from pathlib import Path

from kiss.agents.sorcar.cli_line_continuation import (
    ends_with_line_continuation,
)
from kiss.agents.sorcar.cli_panel import (
    _ESC,
    CYAN,
    IDLE_TITLE,
    PROMPT_MARKER,
    RESET,
    panel_bottom,
    panel_cols,
    panel_top,
)
from kiss.agents.sorcar.cli_prompt import _AT_RE, _MODEL_CMD_RE, PtkLineReader
from kiss.agents.sorcar.custom_commands import (
    discover_commands,
    format_command_listing,
)
from kiss.agents.sorcar.persistence import (
    _default_kiss_dir,
    _ensure_kiss_dir,
    _load_file_usage,
    _prefix_match_tasks,
)
from kiss.agents.vscode.helpers import (
    clip_autocomplete_suggestion,
    rank_file_suggestions,
)
from kiss.agents.vscode.tricks import (
    current_sentence_partial,
    prefix_match_tricks,
)

logger = logging.getLogger(__name__)

try:  # POSIX line editing; ``readline`` is absent on stock Windows.
    # Prefer GNU readline (the ``gnureadline`` wheel) when it is
    # installed: stock macOS Python links ``readline`` against libedit,
    # which has no ``menu-complete`` and therefore cannot *cycle* through
    # completion candidates one at a time.  ``gnureadline`` ships real
    # GNU readline, enabling the Tab/Shift-Tab menu cycling configured in
    # :func:`_setup_readline`.  Fall back to the stdlib ``readline`` (and
    # its degraded list-on-double-Tab behaviour) when it is unavailable.
    try:
        import gnureadline as readline  # type: ignore[import-not-found]
    except ImportError:
        import readline

    _HAVE_READLINE = True
except ImportError:  # pragma: no cover - exercised only on Windows
    readline = None  # type: ignore[assignment]
    _HAVE_READLINE = False

# Slash commands handled locally by the REPL (never sent to the model),
# mirroring Claude Code's built-in quick commands.  Maps command -> help.
SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show available commands",
    "/clear": "Start a new chat (clear conversation context)",
    "/new": "Alias for /clear",
    "/resume": (
        "Resume a chat: /resume <chat-id>, open a specific task with "
        "/resume --task <task-id>, or list recent chats "
        "(/resume [--limit N], default 20)"
    ),
    "/model": (
        "Switch model: /model <name>, list all models with /model list, "
        "or show the current model"
    ),
    "/cost": "Show cost and token usage for this session",
    "/usage": "Alias for /cost",
    "/context": "Show token usage for this session",
    "/commands": "List custom commands (.md files in ~/.kiss/commands "
                 "and <project>/.kiss/commands)",
    "/skills": "List agent skills (SKILL.md dirs in ~/.kiss/skills, "
               "<project>/.kiss/skills, .claude/skills, .agents/skills); "
               "/skills <name> shows one",
    "/mcp": "List MCP servers (~/.kiss/mcp.json, <project>/.kiss/mcp.json, "
            "<project>/.mcp.json) with live status; manage with "
            "`sorcar mcp add/list/auth/debug`",
    "/autocommit": "Stage all changes, auto-generate a commit message, "
                   "and commit (same as the extension's Auto-commit)",
    "/exit": "Exit the sorcar CLI",
    "/quit": "Alias for /exit",
}

# Bare words (no leading slash) that also exit, matching Claude Code.
_EXIT_WORDS = {"exit", "quit"}

_PROMPT = f"{CYAN}{PROMPT_MARKER}{RESET}"
# ANSI SGR (colour) sequences embedded in the input prompt.
_ANSI_SGR_RE = re.compile(r"(\x1b\[[0-9;]*m)")


def _readline_prompt(prompt: str) -> str:
    """Bracket ANSI sequences in *prompt* with readline ignore markers.

    GNU readline counts every prompt byte as a printed column unless
    non-printing runs are wrapped in ``\\x01``/``\\x02``
    (``RL_PROMPT_START_IGNORE``/``RL_PROMPT_END_IGNORE``).  Without the
    markers readline believes the colour codes occupy screen columns,
    so it redraws/wraps the input line ~20 columns early, garbling the
    input panel on narrow terminals.

    Args:
        prompt: The prompt string, possibly containing SGR sequences.

    Returns:
        The prompt with every SGR sequence wrapped in the markers.
    """
    return _ANSI_SGR_RE.sub("\x01\\1\x02", prompt)


class CliCompleter:
    """readline completer reproducing the extension's input fast-completes.

    A single instance is installed as the readline completion function.
    The whole input line is treated as one completion "word" (the
    completer word-break set is cleared) so the completer has full
    control over ``@``-mentions, ``/`` slash commands, and whole-line
    predictive ghost text — none of which align with readline's default
    token-based completion.

    Attributes:
        work_dir: Project directory whose files seed ``@``-mention
            completion.
        active_file: Optional path whose identifiers seed predictive
            completion (mirrors the extension's active-editor file).
    """

    def __init__(self, work_dir: str, active_file: str = "") -> None:
        self.work_dir = work_dir
        self.active_file = active_file
        self._file_cache: list[str] | None = None
        self._matches: list[str] = []

    def _files(self) -> list[str]:
        """Return the (lazily scanned, cached) project file list."""
        if self._file_cache is None:
            from kiss.agents.vscode.diff_merge import _scan_files

            try:
                self._file_cache = _scan_files(self.work_dir)
            except Exception:  # pragma: no cover - defensive scan guard
                logger.debug("file scan failed", exc_info=True)
                self._file_cache = []
        return self._file_cache

    def _at_mention_matches(self, line: str, at_start: int, query: str) -> list[str]:
        """Build ``./<path>`` completions for an ``@``-mention token.

        Args:
            line: The full input line being completed.
            at_start: Index of the ``@`` that starts the mention token.
            query: Text typed after the ``@``.

        Returns:
            Candidate full-line replacements, each ending in a space.
        """
        usage = _load_file_usage()
        ranked = rank_file_suggestions(self._files(), query, usage)
        prefix = line[:at_start]
        return [f"{prefix}./{item['text']} " for item in ranked]

    def _model_matches(self, query: str) -> list[str]:
        """Return ``/model <name>`` completions for the ``/model`` command.

        Mirrors the extension's model picker: candidate model names come
        from :func:`rank_model_suggestions` over the generation-capable
        models in ``MODEL_INFO`` (preferring those with a configured API
        key), best match first.

        Args:
            query: Text typed after ``/model `` (the partial model name).

        Returns:
            Whole-line ``/model <name>`` replacements, best match first.
        """
        from kiss.core.models.model_info import rank_model_suggestions

        return [f"/model {name}" for name in rank_model_suggestions(query)]

    def _slash_matches(self, line: str) -> list[str]:
        """Return slash-command completions for *line* (e.g. ``/he``).

        Built-in commands come first, then custom commands discovered
        from ``~/.kiss/commands`` and ``<work_dir>/.kiss/commands``.
        """
        token = line.strip()
        matches = [
            f"{cmd} " for cmd in SLASH_COMMANDS if cmd.startswith(token)
        ]
        try:
            custom = discover_commands(self.work_dir)
        except Exception:  # pragma: no cover - defensive discovery guard
            logger.debug("custom command discovery failed", exc_info=True)
            custom = {}
        matches.extend(
            f"/{name} " for name in sorted(custom)
            if f"/{name}".startswith(token) and f"/{name} " not in matches
        )
        return matches

    def _predictive_matches(self, line: str) -> list[str]:
        """Return whole-line predictive completions, best match first.

        Mirrors the extension's ghost text but returns the full list so
        the prompt_toolkit dropdown can show every prefix-matched prior
        task; an identifier suffix from the active file is appended as
        a final candidate when no history match is found.

        Every candidate is a WHOLE-LINE replacement that starts with
        *line*: both frontends substitute the candidate for the entire
        input (readline runs with the word-break delimiters cleared;
        the prompt_toolkit dropdown uses ``start_position=-len(line)``),
        so trick and identifier suggestions — which natively begin only
        at the current sentence / trailing token — are spliced onto the
        untouched head of *line* here.  This mirrors the overlap-splice
        accept in the VS Code webview's ``acceptCompletion`` and keeps
        the text the user already typed from being erased on accept.
        """
        # ``_prefix_match_tasks`` already guarantees (via SQL) that
        # every match starts with *line* and is strictly longer than it.
        tasks = _prefix_match_tasks(line)
        if tasks:
            return tasks
        # INJECTIONS.md "Inject instruction" tricks are also surfaced
        # as fast-complete suggestions, but ONLY at the beginning of a
        # sentence (start of *line* or after ``[.!?]`` + whitespace).
        # ``prefix_match_tricks`` returns EVERY trick whose body
        # begins with the current sentence's leading partial — so when
        # multiple tricks share a prefix (e.g. the bundled
        # INJECTIONS.md ships two ``Reproduce the issue by writing …``
        # tricks) the dropdown menu shows them all, mirroring
        # ``_prefix_match_tasks``' multi-alternative contract.
        tricks = prefix_match_tricks(line)
        if tricks:
            # ``current_sentence_partial`` is always a suffix of *line*
            # (leading whitespace only when no interior boundary), so
            # the head before it is everything the trick must keep.
            head = line[: len(line) - len(current_sentence_partial(line))]
            return [head + trick for trick in tricks]
        suffix = self._active_file_suffix(line)
        if suffix:
            return [line + suffix]
        return []

    def _active_file_suffix(self, line: str) -> str:
        """Complete the trailing identifier of *line* from the active file."""
        if not self.active_file:
            return ""
        m = re.search(r"([\w][\w.]*)$", line)
        if not m or len(m.group(1)) < 2:
            return ""
        partial = m.group(1)
        try:
            content = Path(self.active_file).read_text()[:50000]
        except OSError:
            return ""
        words = set(re.findall(r"\b[A-Za-z_]\w{2,}\b", content))
        chains = set(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b", content))
        best = ""
        for cand in words | chains:
            if cand.startswith(partial) and len(cand) > len(partial):
                suffix = cand[len(partial):]
                if len(suffix) > len(best):
                    best = suffix
        return clip_autocomplete_suggestion(line, best)

    def complete(self, text: str, state: int) -> str | None:
        """readline completion entry point.

        Args:
            text: The completion word.  With word-break characters
                cleared this is the entire line up to the cursor.
            state: 0 on the first call for a given word, then 1, 2, …
                until ``None`` is returned.

        Returns:
            The ``state``-th candidate, or ``None`` when exhausted.
        """
        if state == 0:
            self._matches = self._build_matches(text)
        if state < len(self._matches):
            return self._matches[state]
        return None

    def _build_matches(self, line: str) -> list[str]:
        """Compute the ordered candidate list for the current *line*."""
        at = _AT_RE.search(line)
        if at:
            return self._at_mention_matches(
                line, at.start(), at.group(1),
            )
        model_cmd = _MODEL_CMD_RE.match(line)
        if model_cmd:
            return self._model_matches(model_cmd.group(1))
        stripped = line.strip()
        if stripped.startswith("/") and " " not in stripped:
            return self._slash_matches(line)
        return self._predictive_matches(line)


def _setup_readline(completer: CliCompleter, history_path: Path) -> None:
    """Install the completer and load per-directory history.

    Binds Tab to ``menu-complete`` so pressing Tab repeatedly *cycles*
    through the completion candidates one at a time (best match first),
    and Shift-Tab (``\\e[Z``) to ``menu-complete-backward`` to cycle the
    other way; ``show-all-if-ambiguous`` makes the first Tab jump
    straight into the menu instead of just inserting the common prefix.
    These ``menu-complete`` actions exist only in GNU readline, so on the
    libedit/editline backend (stock macOS Python without the
    ``gnureadline`` wheel) we fall back to ``rl_complete``, which lists
    candidates on a double-Tab but cannot cycle.

    Args:
        completer: The completion function holder to install.
        history_path: File used to persist and restore input history.
    """
    if not _HAVE_READLINE:
        return
    assert readline is not None
    readline.set_completer(completer.complete)
    readline.set_completer_delims("")
    backend = getattr(readline, "backend", "") or ""
    doc = getattr(readline, "__doc__", "") or ""
    if backend == "editline" or "libedit" in doc:
        # libedit has no menu-complete; cycling is unavailable here.
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        # GNU readline: Tab cycles forward, Shift-Tab cycles backward.
        readline.parse_and_bind("set show-all-if-ambiguous on")
        readline.parse_and_bind("tab: menu-complete")
        readline.parse_and_bind('"\\e[Z": menu-complete-backward')
        # Shift+Enter (kitty CSI-u "\e[13;2u" / xterm modifyOtherKeys
        # "\e[27;2;13~") types a backslash and accepts the line, which
        # triggers _read_line's trailing-backslash continuation so the
        # final message contains a real newline at that point.
        readline.parse_and_bind('"\\e[13;2u": "\\\\\\n"')
        readline.parse_and_bind('"\\e[27;2;13~": "\\\\\\n"')
    try:
        readline.read_history_file(str(history_path))
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(1000)


def _save_history(history_path: Path) -> None:
    """Persist the readline history to *history_path* (best effort)."""
    if not _HAVE_READLINE:
        return
    assert readline is not None
    try:
        readline.write_history_file(str(history_path))
    except OSError:  # pragma: no cover - disk/permission error
        logger.debug("could not write history file", exc_info=True)


def _history_path(work_dir: str) -> Path:
    """Return the per-working-directory history file path.

    Claude Code stores input history per working directory; we do the
    same, keyed by a hash of the absolute work dir under the kiss dir.
    The hash is a stable content digest (NOT the builtin ``hash()``,
    which is randomized per process and would make every new process
    look for a different history file).
    """
    _ensure_kiss_dir()
    resolved = str(Path(work_dir).resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    hist_dir = _default_kiss_dir() / "cli_history"
    try:
        hist_dir.mkdir(parents=True, exist_ok=True)
    except OSError:  # pragma: no cover - permission error
        return _default_kiss_dir() / f"cli_history_{digest}"
    return hist_dir / f"{digest}"


def _print_welcome(work_dir: str, model_name: str) -> None:
    """Print the Claude-Code-style welcome banner for the session."""
    print("\x1b[1m✦ KISS Sorcar\x1b[0m — interactive mode")
    print(f"  model: {model_name}")
    print(f"  cwd:   {work_dir}")
    print("  Type your task and press Enter. "
          "@ to mention files, / for commands, Tab to complete.")
    print("  Alt+Enter (or Shift+Enter / Ctrl+J) inserts a newline for "
          "multi-line input; long lines word-wrap inside the box.")
    print("  /help for commands, /exit (or Ctrl+D) to quit.\n")


def _print_help(work_dir: str = "") -> None:
    """Print the list of available slash commands.

    Args:
        work_dir: Project directory whose custom commands to list; when
            empty, only user-level custom commands are shown.
    """
    print("\nCommands:")
    for cmd, desc in SLASH_COMMANDS.items():
        print(f"  {cmd:<10} {desc}")
    custom = discover_commands(work_dir or ".")
    if custom:
        print("\nCustom commands:")
        print(format_command_listing(custom))
    print(
        "\nInput fast-completes (Tab): @path mentions files, "
        "/ completes commands, /model <partial> completes model names, "
        "and typing a prefix of a previous task suggests its completion.\n"
    )


def _print_model_list(current: str = "") -> None:
    """Print every generation model with provider and API-key status.

    Lists the models from
    :func:`kiss.core.models.model_info.get_generation_model_listing`,
    aligned in columns, with a ``✓``/``✗`` marker showing whether the
    credential needed to run each model is configured, a ``← current``
    marker on the active model, and a configured/total summary header.

    Args:
        current: The currently selected model name, marked in the list.
    """
    from kiss.core.models.model_info import get_generation_model_listing

    listing = get_generation_model_listing()
    if not listing:  # pragma: no cover - MODEL_INFO always has generation models
        print("No generation models are available.\n")
        return
    name_width = max(len(name) for name, _, _ in listing)
    prov_width = max(len(provider) for _, provider, _ in listing)
    configured = sum(1 for _, _, ok in listing if ok)
    print(
        f"\nGeneration models ({configured}/{len(listing)} with credentials "
        f"configured):"
    )
    for name, provider, ok in listing:
        mark = "✓" if ok else "✗"
        status = "configured" if ok else "no API key"
        here = "  ← current" if name == current else ""
        print(
            f"  {mark} {name:<{name_width}}  {provider:<{prov_width}}  "
            f"{status}{here}"
        )
    print()


def _make_ptk_reader(
    completer: CliCompleter, history_path: Path,
) -> PtkLineReader | None:
    """Build the prompt_toolkit line reader, or ``None`` off-TTY.

    The live dropdown (``@`` file picker with Up/Down navigation and
    Tab/Enter selection) needs full-screen control of the terminal, so
    it is only used when both stdin and stdout are TTYs; every other
    case falls back to the readline/plain-:func:`input` path.

    Args:
        completer: The shared completion backend.
        history_path: Per-working-directory readline history file; the
            prompt_toolkit history lives next to it.

    Returns:
        A ready :class:`PtkLineReader`, or ``None`` when unavailable.
    """
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return None
    except Exception:  # pragma: no cover - defensive isatty guard
        return None
    try:
        return PtkLineReader(completer, history_path)
    except Exception:  # pragma: no cover - terminal init failure
        logger.debug("prompt_toolkit init failed", exc_info=True)
        return None


def _read_line_ptk(reader: PtkLineReader, prompt: str) -> str | None:
    """Read one input line via prompt_toolkit inside the input panel.

    The panel's top border is printed above the prompt and the bottom
    border after the line is accepted; while the user types, the rows
    below the input stay free so the live completion dropdown (the
    ``@`` file/folder picker, slash-command and ``/model`` menus) can
    render there.

    A line ending in an *unescaped* backslash continues on the next
    row, joined with real newlines, following the shared POSIX-shell
    rule in :func:`~kiss.agents.sorcar.cli_line_continuation
    .ends_with_line_continuation`.  The prompt_toolkit Enter binding
    (:func:`~kiss.agents.sorcar.cli_prompt._submit_enter`) already
    applies the same rule in-buffer before ever submitting, so a
    submitted line can only end in an even (escaped-literal) number of
    backslashes — which must be returned verbatim, NOT treated as a
    continuation.  A naive ``endswith("\\")`` check here used to eat
    one of the user's escaped literal backslashes and swallow the next
    input line as a bogus continuation.

    Args:
        reader: The prompt_toolkit session wrapper.
        prompt: The (coloured) prompt marker to show inside the panel.

    Returns:
        The (possibly multi-line) input, or ``None`` on EOF (Ctrl+D).

    Raises:
        KeyboardInterrupt: When the user presses Ctrl+C at the prompt.
    """
    cols = panel_cols()
    top = f"{CYAN}{panel_top(IDLE_TITLE, cols)}{RESET}"
    bottom = f"{CYAN}{panel_bottom('', cols)}{RESET}"
    framed_prompt = f"{CYAN}│{RESET} {prompt}"
    print(top)
    try:
        line = reader.read(framed_prompt)
    except EOFError:
        print(bottom)
        return None
    except KeyboardInterrupt:
        print(bottom)
        raise
    while True:
        cont, keep = ends_with_line_continuation(line)
        if not cont:
            break
        try:
            more = reader.read(framed_prompt)
        except EOFError:
            line = line[:keep]
            break
        except KeyboardInterrupt:
            print(bottom)
            raise
        line = line[:keep] + "\n" + more
    print(bottom)
    return line


def _read_line(prompt: str, reader: PtkLineReader | None = None) -> str | None:
    """Read one input line inside the shared rounded input panel.

    When *reader* is given (interactive TTY), the line is read through
    prompt_toolkit via :func:`_read_line_ptk`, which pops the live
    ``@``-mention file/folder dropdown while typing.  Otherwise the
    readline/plain-:func:`input` path below is used.

    The idle prompt is drawn inside the very same rounded-border panel
    that the steering box (:mod:`kiss.agents.sorcar.cli_steering`) uses
    while a task runs, so the input dialog looks like one consistent
    panel whether the agent is idle or steering.  The panel's top border
    (with the idle title) is printed above the prompt, the ``│ ›`` body
    carries the readline input, and the bottom border closes it below.

    On an interactive terminal the bottom border is drawn *before* the
    input is read and the cursor is moved back up onto the body line, so
    the box stays fully closed — top *and* bottom rules visible — while
    the user is still typing the task (matching the always-framed
    steering box).  When stdout is not a TTY the bottom border is simply
    printed after the line is read.

    A line ending in a backslash (typed directly, or injected by the
    Shift+Enter readline macro bound in :func:`_setup_readline`)
    continues on the next body row; the parts are joined with real
    newlines so Shift+Enter inserts a line break into the message.

    Returns:
        The (possibly multi-line) input, or ``None`` to signal EOF
        (Ctrl+D) — the caller exits.

    Raises:
        KeyboardInterrupt: When the user presses Ctrl+C at the prompt.
    """
    if reader is not None:
        return _read_line_ptk(reader, prompt)
    cols = panel_cols()
    top = f"{CYAN}{panel_top(IDLE_TITLE, cols)}{RESET}"
    bottom = f"{CYAN}{panel_bottom('', cols)}{RESET}"
    framed_prompt = f"{CYAN}│{RESET} {prompt}"

    try:
        isatty = bool(sys.stdout.isatty())
    except Exception:  # pragma: no cover - defensive isatty guard
        isatty = False
    if not isatty:
        print(top)
        try:
            line = input(framed_prompt)
        except EOFError:
            print(bottom)
            return None
        while True:
            cont, keep = ends_with_line_continuation(line)
            if not cont:
                break
            try:
                more = input(framed_prompt)
            except EOFError:
                line = line[:keep]
                break
            line = line[:keep] + "\n" + more
        print(bottom)
        return line

    # Interactive: pre-draw the closed box, then edit on the body line.
    # ``input`` hands the prompt to readline only when stdin is also a
    # TTY; readline needs its colour codes bracketed with the
    # \x01/\x02 ignore markers to compute the prompt width correctly
    # (raw markers would be echoed verbatim in the non-readline path).
    try:
        stdin_tty = bool(sys.stdin.isatty())
    except Exception:  # pragma: no cover - defensive isatty guard
        stdin_tty = False
    if _HAVE_READLINE and stdin_tty:
        framed_prompt = _readline_prompt(framed_prompt)
    print(top)
    # Draw the body line's right border at the far column so the box is
    # fully framed (left ``│`` comes from ``framed_prompt`` below), then
    # draw the bottom rule one line down, then move the cursor back up
    # onto the (now framed) body line at column 1 for readline.
    sys.stdout.write(
        f"{_ESC}[{cols}G{CYAN}│{RESET}\r\n{bottom}{_ESC}[1A\r"
    )
    sys.stdout.flush()
    try:
        line = input(framed_prompt)
    except EOFError:
        # Cursor is still on the body line; drop below the bottom rule.
        sys.stdout.write("\n\n")
        sys.stdout.flush()
        return None
    except KeyboardInterrupt:
        # Ctrl+C leaves the cursor on the body line; step onto the
        # bottom rule and erase it so the caller's interrupt message is
        # not printed over the border (which would leave a garbled
        # "…quit)────╯" row on screen).
        sys.stdout.write(f"\n{_ESC}[2K")
        sys.stdout.flush()
        raise
    # An unescaped trailing backslash (typed directly, or injected by
    # the Shift+Enter readline macro bound in :func:`_setup_readline`)
    # continues the message on the next line, following the shared
    # POSIX-shell rule in :func:`~kiss.agents.sorcar
    # .cli_line_continuation.ends_with_line_continuation` (an even,
    # escaped-literal number of backslashes submits); the joined parts
    # are separated by real newlines.  The cursor currently sits on the
    # old bottom-rule row: clear it, frame it as the next body row,
    # redraw the bottom rule one line down, and read the continuation
    # there.
    while True:
        cont, keep = ends_with_line_continuation(line)
        if not cont:
            break
        sys.stdout.write(
            f"\r{_ESC}[2K{_ESC}[{cols}G{CYAN}│{RESET}\r\n{bottom}{_ESC}[1A\r"
        )
        sys.stdout.flush()
        try:
            more = input(framed_prompt)
        except EOFError:
            sys.stdout.write("\n")
            sys.stdout.flush()
            line = line[:keep]
            break
        except KeyboardInterrupt:
            # Same bottom-rule cleanup as the first input() above.
            sys.stdout.write(f"\n{_ESC}[2K")
            sys.stdout.flush()
            raise
        line = line[:keep] + "\n" + more
    # Enter already moved the cursor onto the bottom rule line; step past
    # it so following output never overwrites the closed box.
    sys.stdout.write("\n")
    sys.stdout.flush()
    return line


def _load_history_lines(path: Path) -> list[str]:
    """Return the saved input lines for the anchored REPL, oldest first.

    Args:
        path: Persistence path; non-existent / unreadable files yield
            an empty history (best effort).

    Returns:
        The history lines (without trailing newlines).
    """
    if not path.exists():
        return []
    try:
        return [
            ln for ln in path.read_text(encoding="utf-8").splitlines() if ln
        ]
    except OSError:
        return []


def _save_history_lines(path: Path, history: list[str]) -> None:
    """Persist the anchored REPL's input history (last 1000 lines).

    Args:
        path: Destination file; parent directories are created as needed.
        history: Lines to persist (oldest first).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(history[-1000:])
        if history:
            text += "\n"
        path.write_text(text, encoding="utf-8")
    except OSError:  # pragma: no cover - disk/permission error
        logger.debug("could not save anchored history", exc_info=True)
