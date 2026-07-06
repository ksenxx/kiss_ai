# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared CLI helpers for Sorcar agent entry points.

Provides argument parsing, chat-session handling, run-kwarg construction,
and post-run statistics.
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from kiss.agents.sorcar.persistence import _list_recent_chats
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.models.model_info import get_default_model

if TYPE_CHECKING:
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

_DEFAULT_TASK = """
can you find what the current weather is in San Francisco and summarize it?
"""


def _resolve_task(args: argparse.Namespace) -> str:
    """Determine the task description from parsed arguments.

    Priority: -f file > --task string > default task.

    Args:
        args: Parsed argparse namespace with 'f' and 'task' attributes.

    Returns:
        The task description string.

    Raises:
        FileNotFoundError: If -f path does not exist.
    """
    if args.file is not None:
        return Path(args.file).read_text()
    if args.task is not None:
        task: str = args.task
        return task
    return _DEFAULT_TASK


def cli_ask_user_question(question: str) -> str:
    """CLI callback for agent questions (prints and reads from stdin).

    Args:
        question: The question to display to the user.

    Returns:
        The user's typed response text.
    """
    print(f"\n>>> Agent asks: {question}")
    return input("Your answer: ")


DEFAULT_RECENT_CHATS_LIMIT = 20


def _parse_resume_arg(arg: str) -> tuple[str, str, int]:
    """Parse the argument string of ``/resume`` into ids and a limit.

    Supports the following forms (in any order, whitespace separated):

    * ``""`` — list the default number of recent chats.
    * ``"<chat-id>"`` — resume the chat with the given id.
    * ``"--task <task-id>"`` or ``"--task=<task-id>"`` — resume the
      chat containing the given task, opened at that specific task
      (rather than the latest task of the chat).
    * ``"--limit N"`` or ``"--limit=N"`` — list the most recent ``N``
      chats (only valid when no chat id is also supplied).
    * ``"<chat-id> --limit N"`` — the ``--limit`` flag is ignored when
      a chat id is present (resuming a chat does not use a limit).

    Args:
        arg: The raw argument string after ``/resume``.

    Returns:
        A ``(chat_id, task_id, limit)`` triple.  ``chat_id`` and
        ``task_id`` are empty strings when not supplied (both empty
        means the user is requesting the recent-chats listing).
        ``limit`` is the resolved listing limit (defaulting to
        :data:`DEFAULT_RECENT_CHATS_LIMIT`).

    Raises:
        ValueError: If ``--limit`` is given without a value, with a
            non-integer value, or with a non-positive value; if
            ``--task`` is given without a value; or if both a chat id
            and ``--task`` are supplied (the task id alone identifies
            the chat to open, so combining them is ambiguous).
    """
    tokens = arg.split()
    chat_id = ""
    task_id = ""
    limit = DEFAULT_RECENT_CHATS_LIMIT
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "--limit":
            if i + 1 >= len(tokens):
                raise ValueError("--limit requires a positive integer value")
            limit = _coerce_positive_int(tokens[i + 1])
            i += 2
            continue
        if token.startswith("--limit="):
            limit = _coerce_positive_int(token[len("--limit="):])
            i += 1
            continue
        if token == "--task":
            if i + 1 >= len(tokens):
                raise ValueError("--task requires a task id value")
            task_id = tokens[i + 1]
            i += 2
            continue
        if token.startswith("--task="):
            task_id = token[len("--task="):]
            if not task_id:
                raise ValueError("--task requires a task id value")
            i += 1
            continue
        if chat_id:
            raise ValueError(f"unexpected extra argument: {token!r}")
        chat_id = token
        i += 1
    if chat_id and task_id:
        raise ValueError(
            "give either a chat id or --task <task-id>, not both "
            "(the task id alone identifies the chat to open)"
        )
    return chat_id, task_id, limit


def _coerce_positive_int(value: str) -> int:
    """Coerce ``value`` to a strictly positive integer.

    Args:
        value: The string to parse.

    Returns:
        The parsed positive integer.

    Raises:
        ValueError: If ``value`` is not a positive integer.
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"--limit must be a positive integer, got {value!r}",
        ) from exc
    if parsed <= 0:
        raise ValueError(
            f"--limit must be a positive integer, got {value!r}",
        )
    return parsed


def _print_recent_chats(limit: int = DEFAULT_RECENT_CHATS_LIMIT) -> None:
    """Print the most recent chat sessions with their tasks and results.

    Each task line also surfaces the row's own ``Task ID`` and the
    ``Parent Task ID`` so the ``/resume`` picker can show per-task
    identity and the sub-agent parent relationship at a glance.
    Top-level (non-sub-agent) tasks have no parent, so the parent
    column is rendered as ``(none)``.

    Args:
        limit: Maximum number of chat sessions to display, defaulting
            to :data:`DEFAULT_RECENT_CHATS_LIMIT` (20).  Users can
            override this with ``/resume --limit N``.
    """
    chats = _list_recent_chats(limit=limit)
    if not chats:
        print("No chat sessions found.")
        return
    for entry in reversed(chats):
        print(f"\n{'=' * 72}")
        print(f"Chat ID: {entry['chat_id']}")
        print(f"{'=' * 72}")
        tasks: list[dict[str, object]] = entry["tasks"]  # type: ignore[assignment]
        for i, t in enumerate(tasks, 1):
            ts = float(t.get("timestamp", 0))  # type: ignore[arg-type]
            dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            task_text = str(t.get("task", ""))[:200]
            result_text = str(t.get("result", ""))[:200]
            task_id = str(t.get("task_id", ""))
            parent_task_id = str(t.get("parent_task_id", ""))
            parent_display = parent_task_id if parent_task_id else "(none)"
            print(f"\n  Task {i} [{dt}]:")
            print(f"    Task ID: {task_id}")
            print(f"    Parent Task ID: {parent_display}")
            print(f"    {task_text}")
            if result_text:
                print(f"  Result {i}:")
                print(f"    {result_text}")


def _launch_work_dir() -> str:
    """Return the directory the sorcar CLI was launched from.

    The installed ``sorcar`` wrapper runs
    ``uv run --directory <kiss_project> sorcar ...``, and ``uv``'s
    ``--directory`` flag changes the process working directory to the
    bundled ``kiss_project`` before the CLI starts.  As a result
    :func:`Path.cwd` reports the project directory rather than the
    user's shell directory.  The wrapper therefore records the
    original ``$PWD`` in the ``KISS_WORKDIR`` environment variable, so
    we prefer that when it is set and points at an existing directory,
    falling back to :func:`Path.cwd` for direct (non-wrapper)
    invocations where the cwd is already correct.

    Returns:
        Absolute path of the launch directory as a string.
    """
    env_dir = os.environ.get("KISS_WORKDIR", "").strip()
    if env_dir and Path(env_dir).is_dir():
        return str(Path(env_dir).resolve())
    return str(Path.cwd())


def _parse_bool_value(value: str) -> bool:
    """Parse an explicit boolean value for ``--verbose``.

    Accepts the usual spellings case-insensitively.  Anything else is
    rejected with an argparse error instead of being silently treated
    as ``False`` (w2 F23 — ``--verbose yes`` used to disable verbose
    output without any hint).

    Args:
        value: The raw command-line token following the flag.

    Returns:
        The parsed boolean.

    Raises:
        argparse.ArgumentTypeError: If *value* is not a recognised
            boolean spelling.
    """
    v = value.strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean (true/false), got {value!r}"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for all Sorcar agent entry points.

    ``allow_abbrev`` is set to ``False`` so users must spell long
    options out fully.  Otherwise argparse would expand abbreviations
    like ``--auto`` into ``--auto-commit`` BEFORE the non-interactive
    guard (:func:`_reject_interactive_only_flags`) gets to inspect
    the raw argv — silently bypassing the guard for every flag in
    the interactive-only set.
    """
    parser = argparse.ArgumentParser(
        description="Run SorcarAgent demo",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-m", "--model_name", type=str, default=get_default_model(),
        help="LLM model name (defaults to the best model for the configured API keys)",
    )
    parser.add_argument(
        "-e", "--endpoint", type=str, default=None, help="Custom endpoint for local model"
    )
    parser.add_argument(
        "--header", action="append", type=str, default=None,
        help="Custom HTTP header (format: 'Key:Value'). Can be used multiple times.",
    )
    parser.add_argument(
        "-b", "--max_budget", type=float, default=DEFAULT_CONFIG.max_budget,
        help="Maximum budget in USD",
    )
    parser.add_argument(
        "-w", "--work_dir", type=str, default=_launch_work_dir(),
        help="Working directory (defaults to the directory where sorcar is launched)",
    )
    parser.add_argument(
        "-v", "--verbose",
        type=_parse_bool_value,
        nargs="?",
        const=True,
        default=True,
        help=(
            "Print output to console (default: enabled). Works as a "
            "bare flag (-v) or with an explicit value "
            "(--verbose false)."
        ),
    )
    parser.add_argument(
        "--no-web", action="store_true", default=False,
        help="Disable browser/web tools (terminal-only mode)",
    )
    parser.add_argument(
        "-p", "--parallel", action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable parallel subagents (default: enabled). "
            "Use --no-parallel to disable."
        ),
    )
    parser.add_argument(
        "--worktree", action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Isolate every task in a git worktree branch "
            "(default: enabled). Use --no-worktree to run directly "
            "in the working tree (chat mode)."
        ),
    )
    parser.add_argument(
        "--auto-commit", dest="auto_commit",
        action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Auto-commit worktree changes when a task finishes "
            "(default: enabled). Use --no-auto-commit to skip the "
            "automatic commit and preserve the worktree for manual "
            "review."
        ),
    )
    parser.add_argument(
        "-t", "--task", type=str, default=None, help="Task description"
    )
    parser.add_argument(
        "-f", "--file", type=str, default=None,
        help="Path to a file whose contents to use as the task",
    )
    return parser


def _build_run_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build ``agent.run()`` keyword arguments from parsed CLI args."""
    task_description = _resolve_task(args)
    work_dir = args.work_dir or _launch_work_dir()
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    model_config: dict[str, Any] = {}
    if args.endpoint:
        model_config["base_url"] = args.endpoint
    if args.header:
        headers = {}
        for h in args.header:
            # Reject malformed headers loudly instead of silently
            # dropping them (w3 C-3): a ``--header`` without a colon
            # (or with an empty key) used to vanish without a message,
            # surfacing later only as an opaque downstream auth/HTTP
            # error.  Matches the strict policy of the sibling
            # ``sorcar mcp`` CLI (``mcp_cli._parse_kv`` raises
            # SystemExit for a separator-less ``--header``).
            key, found, value = h.partition(":")
            if not found or not key.strip():
                raise SystemExit(
                    f"Invalid --header {h!r}: expected 'Key:Value'"
                )
            headers[key.strip()] = value.strip()
        if headers:
            model_config["extra_headers"] = headers

    run_kwargs: dict[str, Any] = {
        "prompt_template": task_description,
        "model_name": args.model_name,
        "max_budget": args.max_budget,
        "model_config": model_config,
        "work_dir": work_dir,
        "web_tools": not args.no_web,
        "is_parallel": args.parallel,
        "verbose": args.verbose,
        "ask_user_question_callback": cli_ask_user_question,
    }
    # When the CLI runs verbosely (the default), install a recording
    # console printer.  The plain ``ConsolePrinter`` renders Rich
    # panels but does NOT persist events, so the chat webview shows a
    # blank session for CLI-launched tasks.  ``RecordingConsolePrinter``
    # both records every display event to the chat DB AND renders the
    # Rich panels to the terminal, so the same run is visible live in
    # the terminal and replayable later in the chat webview.
    if args.verbose:
        from kiss.agents.sorcar.cli_printer import RecordingConsolePrinter

        run_kwargs["printer"] = RecordingConsolePrinter()
    return run_kwargs


def _print_result(result: str) -> None:
    """Print the agent's run result without the raw YAML envelope.

    The agent returns a YAML document with ``success`` and ``summary``
    keys.  Printing that document verbatim exposes the raw YAML to the
    user; instead show only the human-readable ``summary`` text.  When
    the result is not the expected YAML mapping, fall back to printing
    it as-is so no output is ever silently dropped.

    Args:
        result: The YAML string returned by the running agent.
    """
    try:
        parsed = yaml.safe_load(result)
    except Exception:
        parsed = None
    if isinstance(parsed, dict) and "summary" in parsed:
        print(str(parsed.get("summary", "")))
    else:
        print(result)


def _print_run_stats(agent: SorcarAgent, elapsed: float) -> None:
    """Print post-run statistics (time, cost, tokens)."""
    print(f"\nTime: {elapsed:.1f}s")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")


def print_outcome(
    agent: SorcarAgent, result: str, elapsed: float, verbose: bool,
) -> None:
    """Print a finished run's result summary and statistics.

    No-op when *verbose*: a verbose agent already renders the green
    "Result" panel (whose subtitle carries tokens / cost / steps) to
    the console as the task ends, so re-printing the summary and the
    run stats here would duplicate it.  Only prints when running
    quietly so the Result panel stays the last thing shown.

    Args:
        agent: The agent that produced *result*.
        result: The YAML result string returned by ``agent.run``.
        elapsed: Wall-clock task duration in seconds.
        verbose: Whether the agent ran verbosely (skip printing).
    """
    if verbose:
        return
    _print_result(result)
    _print_run_stats(agent, elapsed)
