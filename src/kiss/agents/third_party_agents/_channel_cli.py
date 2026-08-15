# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""CLI helpers for the channel-agent entry points (kiss-slack et al.).

The channel agents (Slack, Discord, iMessage, ...) are the only
remaining command-line programs in the framework; the interactive
``sorcar`` terminal interface no longer exists.  This module provides
their shared argument parsing, run-kwarg construction, and post-run
statistics, consumed by
:func:`kiss.agents.third_party_agents._channel_agent_utils.channel_main`.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

from kiss.core import config as config_module
from kiss.core._version import __version__
from kiss.core.models.model_info import get_default_model

_DEFAULT_TASK = """
can you find what the current weather is in San Francisco and summarize it?
"""


def _parse_kv(
    pairs: list[str],
    sep: str,
    error_fmt: str = "Invalid option {pair!r}: expected KEY{sep}VALUE",
) -> tuple[tuple[str, str], ...]:
    """Parse repeated ``KEY<sep>VALUE`` CLI options into tuples.

    An entry without the separator (or with an empty key) is rejected
    loudly with :class:`SystemExit` instead of being silently dropped.

    Args:
        pairs: The raw option values (e.g. ``["FOO=bar"]``).
        sep: The key/value separator (``"="`` for env, ``":"`` for
            headers).
        error_fmt: ``str.format`` template for the rejection message,
            given ``pair`` (the offending raw value) and ``sep``.

    Returns:
        The parsed ``(key, value)`` tuples.

    Raises:
        SystemExit: When an entry has no separator or an empty key.
    """
    out: list[tuple[str, str]] = []
    for pair in pairs:
        key, found, value = pair.partition(sep)
        if not found or not key.strip():
            raise SystemExit(error_fmt.format(pair=pair, sep=sep))
        out.append((key.strip(), value.strip()))
    return tuple(out)


def _resolve_task(args: argparse.Namespace) -> str:
    """Determine the task description from parsed arguments.

    Priority: -f file > --task string > default task.

    Args:
        args: Parsed argparse namespace with 'file' and 'task' attributes.

    Returns:
        The task description string.

    Raises:
        FileNotFoundError: If -f path does not exist.
    """
    if args.file is not None:
        return Path(args.file).read_text(encoding="utf-8")
    if args.task is not None:
        task: str = args.task
        return task
    return _DEFAULT_TASK


def _parse_budget_value(value: str) -> float:
    """Parse a ``--max_budget`` value as a positive finite float.

    Rejects ``nan`` (which would disable the ``budget_used >= max_budget``
    guard entirely because every comparison with NaN is false), ``inf``,
    zero, and negative values.

    Args:
        value: Raw command-line string.

    Returns:
        The parsed budget as a float.

    Raises:
        argparse.ArgumentTypeError: If *value* is not a positive finite
            number.
    """
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--max_budget must be a positive finite number, got {value!r}"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"--max_budget must be a positive finite number, got {value!r}"
        )
    return parsed


def _launch_work_dir() -> str:
    """Return the directory the channel CLI was launched from.

    The installed wrappers run ``uv run --directory <kiss_project> ...``,
    and ``uv``'s ``--directory`` flag changes the process working
    directory to the bundled ``kiss_project`` before the CLI starts.  As
    a result :func:`Path.cwd` reports the project directory rather than
    the user's shell directory.  The wrappers therefore record the
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


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser shared by the channel-agent entry points.

    ``allow_abbrev`` is set to ``False`` so users must spell long
    options out fully; abbreviations like ``--para`` for ``--parallel``
    are rejected instead of silently expanded.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run a KISS channel agent",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-V", "--version", action="version",
        version=f"%(prog)s {__version__}",
        help="Show the version and exit",
    )
    parser.add_argument(
        "-m", "--model_name", type=str, default=None,
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
        "-b", "--max_budget", type=_parse_budget_value,
        default=None,
        help="Maximum budget in USD (defaults to the configured default budget)",
    )
    parser.add_argument(
        "-w", "--work_dir", type=str, default=_launch_work_dir(),
        help="Working directory (defaults to the directory where the CLI is launched)",
    )
    parser.add_argument(
        "--no-web", action="store_true", default=False,
        help="Disable browser/web tools",
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
        "-t", "--task", type=str, default=None, help="Task description"
    )
    parser.add_argument(
        "-f", "--file", type=str, default=None,
        help="Path to a file whose contents to use as the task",
    )
    return parser


def _build_run_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build ``agent.run()`` keyword arguments from parsed CLI args.

    ``-m`` / ``-b`` parse to ``None`` when omitted (so poll mode can
    distinguish a genuine omission from an explicit value equal to the
    default); here — the interactive path — ``None`` resolves to the
    real defaults.

    Args:
        args: Parsed CLI arguments from :func:`_build_arg_parser`.

    Returns:
        Keyword arguments for ``agent.run()``.
    """
    task_description = _resolve_task(args)
    model_name = args.model_name if args.model_name is not None else get_default_model()
    max_budget = (
        args.max_budget
        if args.max_budget is not None
        else config_module.DEFAULT_CONFIG.max_budget
    )
    work_dir = args.work_dir or _launch_work_dir()
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    model_config: dict[str, Any] = {}
    if args.endpoint:
        model_config["base_url"] = args.endpoint
    if args.header:
        headers = dict(_parse_kv(
            args.header, ":",
            error_fmt="Invalid --header {pair!r}: expected 'Key:Value'",
        ))
        if headers:
            model_config["extra_headers"] = headers

    run_kwargs: dict[str, Any] = {
        "prompt_template": task_description,
        "model_name": model_name,
        "max_budget": max_budget,
        "model_config": model_config,
        "work_dir": work_dir,
        "web_tools": not args.no_web,
        "is_parallel": args.parallel,
    }
    return run_kwargs


def _print_run_stats(agent: Any, elapsed: float) -> None:
    """Print post-run statistics (time, cost, tokens)."""
    print(f"\nTime: {elapsed:.1f}s")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")
