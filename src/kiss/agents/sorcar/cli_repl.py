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
  :func:`_scan_files`, inserting ``PWD/<path>`` and recording file usage
  through :func:`_record_file_usage` (mirrors the webview's
  ``insertAtMention`` + ``recordFileUsage``).
* predictive ghost completion — :func:`_prefix_match_task` then
  active-file identifiers, clipped with
  :func:`clip_autocomplete_suggestion` (mirrors the webview ``ghost``
  event produced by ``_AutocompleteMixin._complete``).
* ``/`` slash commands matching Claude Code's quick commands
  (``/help``, ``/clear``, ``/resume``, ``/model``, ``/cost``,
  ``/context``, ``/exit`` …).

TAB triggers completion via :mod:`readline`, which also provides
command history (Up/Down), reverse search (Ctrl+R), and emacs-style line
editing for free — matching Claude Code's interactive-mode shortcuts.
Everything degrades gracefully to a plain :func:`input` loop when
``readline`` is unavailable (e.g. Windows) or stdin is not a TTY.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kiss.agents.sorcar.cli_helpers import (
    _print_recent_chats,
    _print_result,
    _print_run_stats,
)
from kiss.agents.sorcar.cli_panel import (
    CYAN,
    IDLE_TITLE,
    PROMPT_MARKER,
    RESET,
    panel_bottom,
    panel_cols,
    panel_top,
)
from kiss.agents.sorcar.cli_steering import run_with_steering
from kiss.agents.sorcar.persistence import (
    _default_kiss_dir,
    _ensure_kiss_dir,
    _load_file_usage,
    _prefix_match_task,
    _record_file_usage,
)
from kiss.agents.vscode.helpers import (
    clip_autocomplete_suggestion,
    rank_file_suggestions,
)

if TYPE_CHECKING:
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

logger = logging.getLogger(__name__)

try:  # POSIX line editing; ``readline`` is absent on stock Windows.
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
    "/resume": "Resume a chat: /resume <chat-id>, or list recent chats",
    "/model": "Switch model: /model <name>, or show the current model",
    "/cost": "Show cost and token usage for this session",
    "/usage": "Alias for /cost",
    "/context": "Show token usage for this session",
    "/exit": "Exit the sorcar CLI",
    "/quit": "Alias for /exit",
}

# Bare words (no leading slash) that also exit, matching Claude Code.
_EXIT_WORDS = {"exit", "quit"}

_PROMPT = f"{CYAN}{PROMPT_MARKER}{RESET}"
_AT_RE = re.compile(r"@([^\s]*)$")
_MENTION_RE = re.compile(r"PWD/(\S+)")
_ESC = "\x1b"


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
        """Build ``PWD/<path>`` completions for an ``@``-mention token.

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
        return [f"{prefix}PWD/{item['text']} " for item in ranked]

    def _slash_matches(self, line: str) -> list[str]:
        """Return slash-command completions for *line* (e.g. ``/he``)."""
        token = line.strip()
        return [
            f"{cmd} " for cmd in SLASH_COMMANDS if cmd.startswith(token)
        ]

    def _predictive_matches(self, line: str) -> list[str]:
        """Return a single whole-line predictive completion, if any.

        Mirrors the extension's ghost text: prefer a prefix-matched prior
        task, falling back to an identifier from the active file.
        """
        task = _prefix_match_task(line)
        if task and task != line and task.startswith(line):
            return [task]
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
        stripped = line.strip()
        if stripped.startswith("/") and " " not in stripped:
            return self._slash_matches(line)
        return self._predictive_matches(line)


def _setup_readline(completer: CliCompleter, history_path: Path) -> None:
    """Install the completer and load per-directory history.

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
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
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
    """
    _ensure_kiss_dir()
    digest = abs(hash(str(Path(work_dir).resolve()))) % (10**12)
    hist_dir = _default_kiss_dir() / "cli_history"
    try:
        hist_dir.mkdir(parents=True, exist_ok=True)
    except OSError:  # pragma: no cover - permission error
        return _default_kiss_dir() / f"cli_history_{digest}"
    return hist_dir / f"{digest}"


def _print_welcome(agent: SorcarAgent, work_dir: str, model_name: str) -> None:
    """Print the Claude-Code-style welcome banner for the session."""
    print("\x1b[1m✦ KISS Sorcar\x1b[0m — interactive mode")
    print(f"  model: {model_name}")
    print(f"  cwd:   {work_dir}")
    print("  Type your task and press Enter. "
          "@ to mention files, / for commands, Tab to complete.")
    print("  /help for commands, /exit (or Ctrl+D) to quit.\n")


def _print_help() -> None:
    """Print the list of available slash commands."""
    print("\nCommands:")
    for cmd, desc in SLASH_COMMANDS.items():
        print(f"  {cmd:<10} {desc}")
    print(
        "\nInput fast-completes (Tab): @path mentions files, "
        "/ completes commands, and typing a prefix of a previous task "
        "suggests its completion.\n"
    )


def _record_mentions(line: str) -> None:
    """Record file usage for every ``PWD/<path>`` mention in *line*.

    Mirrors the extension's ``recordFileUsage`` so that the most-used
    files float to the top of future ``@``-mention suggestions.
    """
    for match in _MENTION_RE.finditer(line):
        path = match.group(1)
        if path:
            try:
                _record_file_usage(path)
            except Exception:  # pragma: no cover - persistence guard
                logger.debug("record_file_usage failed", exc_info=True)


def _handle_slash(
    agent: SorcarAgent, line: str, run_kwargs: dict[str, Any],
) -> bool:
    """Handle a ``/`` slash command.

    Args:
        agent: The live agent.
        line: The raw input line beginning with ``/``.
        run_kwargs: The mutable run kwargs (e.g. for ``/model``).

    Returns:
        ``True`` if the caller should exit the REPL, ``False`` otherwise.
    """
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit"):
        return True
    if cmd == "/help":
        _print_help()
        return False
    if cmd in ("/clear", "/new"):
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

        if isinstance(agent, ChatSorcarAgent):
            agent.new_chat()
        print("Started a new chat — context cleared.\n")
        return False
    if cmd == "/resume":
        _handle_resume(agent, arg)
        return False
    if cmd == "/model":
        _handle_model(agent, arg, run_kwargs)
        return False
    if cmd in ("/cost", "/usage", "/context"):
        _print_usage(agent)
        return False
    print(f"Unknown command: {cmd}. Type /help for the list of commands.\n")
    return False


def _handle_resume(agent: SorcarAgent, arg: str) -> None:
    """Resume a chat by id, or list recent chats when no id is given."""
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    if not isinstance(agent, ChatSorcarAgent):
        print("Resume is only available in chat mode.\n")
        return
    if arg:
        agent.resume_chat_by_id(arg)
        print(f"Resumed chat {arg}.\n")
    else:
        _print_recent_chats()
        print("\nResume one with: /resume <chat-id>\n")


def _handle_model(
    agent: SorcarAgent, arg: str, run_kwargs: dict[str, Any],
) -> None:
    """Switch the model for subsequent tasks, or show the current one."""
    current = getattr(agent, "model_name", "") or run_kwargs.get("model_name", "")
    if not arg:
        print(f"Current model: {current}\n")
        return
    agent.model_name = arg  # type: ignore[attr-defined]
    run_kwargs["model_name"] = arg
    print(f"Model switched to {arg} for subsequent tasks.\n")


def _print_usage(agent: SorcarAgent) -> None:
    """Print cost and token usage for the session."""
    budget = float(getattr(agent, "budget_used", 0.0) or 0.0)
    tokens = int(getattr(agent, "total_tokens_used", 0) or 0)
    chat_id = getattr(agent, "chat_id", "") or "(new)"
    print(f"\nChat ID: {chat_id}")
    print(f"Cost: ${budget:.4f}")
    print(f"Total tokens: {tokens}\n")


def _stdout_isatty() -> bool:
    """Return whether stdout is attached to an interactive terminal."""
    try:
        return bool(sys.stdout.isatty())
    except Exception:  # pragma: no cover - defensive isatty guard
        return False


def _read_line(prompt: str) -> str | None:
    """Read one input line inside the shared rounded input panel.

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

    Returns:
        The line, or ``None`` to signal EOF (Ctrl+D) — the caller exits.

    Raises:
        KeyboardInterrupt: When the user presses Ctrl+C at the prompt.
    """
    cols = panel_cols()
    top = f"{CYAN}{panel_top(IDLE_TITLE, cols)}{RESET}"
    bottom = f"{CYAN}{panel_bottom('', cols)}{RESET}"
    framed_prompt = f"{CYAN}│{RESET} {prompt}"

    if not _stdout_isatty():
        print(top)
        try:
            line = input(framed_prompt)
        except EOFError:
            print(bottom)
            return None
        print(bottom)
        return line

    # Interactive: pre-draw the closed box, then edit on the body line.
    print(top)
    # Reserve the body line, draw the bottom rule below it, then move the
    # cursor back up onto the (now framed) body line for readline.
    sys.stdout.write(f"\n{bottom}{_ESC}[1A\r")
    sys.stdout.flush()
    try:
        line = input(framed_prompt)
    except EOFError:
        # Cursor is still on the body line; drop below the bottom rule.
        sys.stdout.write("\n\n")
        sys.stdout.flush()
        return None
    # Enter already moved the cursor onto the bottom rule line; step past
    # it so following output never overwrites the closed box.
    sys.stdout.write("\n")
    sys.stdout.flush()
    return line


def run_repl(
    agent: SorcarAgent, run_kwargs: dict[str, Any],
) -> None:
    """Run the interactive Claude-Code-style REPL loop.

    Reads instructions, runs each through :func:`run_with_steering`
    (so follow-ups can be queued while a task runs), prints the result,
    then waits for the next instruction.  The same agent instance is
    reused so the conversation is stateful across tasks.

    Args:
        agent: The agent to drive (chat, worktree, or base Sorcar).
        run_kwargs: Base keyword arguments for ``agent.run``; the
            ``prompt_template`` is replaced for each submitted line and
            any default task placeholder is ignored.
    """
    work_dir = run_kwargs.get("work_dir") or str(Path(".").resolve())
    model_name = run_kwargs.get("model_name", "") or getattr(agent, "model_name", "")
    active_file = run_kwargs.get("current_editor_file") or ""

    completer = CliCompleter(work_dir, active_file)
    history_path = _history_path(work_dir)
    _setup_readline(completer, history_path)

    _print_welcome(agent, work_dir, model_name)

    interrupt_armed = False
    try:
        while True:
            try:
                line = _read_line(_PROMPT)
            except KeyboardInterrupt:
                if interrupt_armed:
                    print("\nGoodbye.")
                    break
                interrupt_armed = True
                print("\n(Press Ctrl+C again or type /exit to quit)")
                continue
            if line is None:  # EOF / Ctrl+D
                print("\nGoodbye.")
                break
            interrupt_armed = False
            text = line.strip()
            if not text:
                continue
            if text in _EXIT_WORDS:
                break
            if text.startswith("/"):
                if _handle_slash(agent, line, run_kwargs):
                    break
                continue
            _record_mentions(line)
            _run_one(agent, line, run_kwargs)
    finally:
        _save_history(history_path)


def _run_one(
    agent: SorcarAgent, prompt: str, run_kwargs: dict[str, Any],
) -> None:
    """Run a single task line and print its result and statistics.

    Args:
        agent: The live agent.
        prompt: The user's instruction for this turn.
        run_kwargs: Base run kwargs; a copy is made with the prompt set.
    """
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    kwargs = dict(run_kwargs)
    kwargs["prompt_template"] = prompt
    start = time.time()
    try:
        result = run_with_steering(agent, kwargs)
    except KeyboardInterrupt:
        print("\n⏹  Task interrupted.\n")
        return
    elapsed = time.time() - start
    # When the agent runs verbosely it already renders the green "Result"
    # panel (whose subtitle carries tokens / cost / steps) to the console
    # as the task ends.  Re-printing the summary and the run stats here
    # would duplicate that panel, so stay completely silent and let the
    # Result panel be the last thing on screen before the prompt returns.
    if kwargs.get("verbose", True):
        return
    _print_result(result)
    if isinstance(agent, ChatSorcarAgent):
        _print_run_stats(agent, elapsed)
    else:
        print(f"\nTime: {elapsed:.1f}s")
        print(f"Cost: ${getattr(agent, 'budget_used', 0.0):.4f}")
        print(f"Total tokens: {getattr(agent, 'total_tokens_used', 0)}")
    print()
