"""Shared CLI helpers for stateful Sorcar agent entry points.

Provides argument parsing, chat-session arg handling, run-kwarg
construction, and post-run statistics printing — shared by
``stateful_sorcar_agent.main()`` and ``worktree_sorcar_agent.main()``.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kiss.agents.sorcar.persistence import _list_recent_chats
from kiss.agents.sorcar.sorcar_agent import (
    _resolve_task,
    cli_ask_user_question,
    cli_wait_for_user,
)

if TYPE_CHECKING:
    from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent


def _print_recent_chats() -> None:
    """Print the last 10 chat sessions with their tasks and results."""
    chats = _list_recent_chats(limit=10)
    if not chats:
        print("No chat sessions found.")
        return
    for entry in reversed(chats):
        chat_id = entry["chat_id"]
        tasks = entry["tasks"]
        assert isinstance(tasks, list)
        print(f"\n{'=' * 72}")
        print(f"Chat ID: {chat_id}")
        print(f"{'=' * 72}")
        for i, t in enumerate(tasks, 1):
            assert isinstance(t, dict)
            ts = t.get("timestamp", 0)
            assert isinstance(ts, (int, float))
            dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            task_text = str(t.get("task", ""))
            result_text = str(t.get("result", ""))
            # Truncate long texts for display
            if len(task_text) > 200:
                task_text = task_text[:200] + "..."
            if len(result_text) > 200:
                result_text = result_text[:200] + "..."
            print(f"\n  Task {i} [{dt}]:")
            print(f"    {task_text}")
            if result_text:
                print(f"  Result {i}:")
                print(f"    {result_text}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the base argument parser for agent CLI entry points.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Run SorcarAgent demo")
    parser.add_argument(
        "-m", "--model_name", type=str, default="claude-opus-4-6", help="LLM model name"
    )
    parser.add_argument(
        "-e", "--endpoint", type=str, default=None, help="Custom endpoint for local model"
    )
    parser.add_argument(
        "-b", "--max_budget", type=float, default=100.0, help="Maximum budget in USD"
    )
    parser.add_argument("-w", "--work_dir", type=str, default=None, help="Working directory")
    parser.add_argument(
        "-v", "--verbose",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Print output to console",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        default=False,
        help="Disable browser/web tools (terminal-only mode)",
    )
    parser.add_argument(
        "-t", "--task", type=str, default=None, help="Prompt template/task description"
    )
    parser.add_argument(
        "-f", "--file", type=str, default=None,
        help="Path to a file whose contents to use as the task",
    )
    return parser


def _build_chat_arg_parser() -> argparse.ArgumentParser:
    """Build arg parser with chat-session CLI options.

    Extends :func:`_build_arg_parser` with ``-n``, ``--chat-id``, and
    ``-l`` flags shared by all stateful agent entry points.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = _build_arg_parser()
    parser.add_argument(
        "-n", "--new", action="store_true",
        help="Start a new chat session",
    )
    parser.add_argument(
        "-c", "--chat-id", type=str, default=None,
        help="Resume a chat session by ID",
    )
    parser.add_argument(
        "-l", "--list-chat-id", action="store_true",
        help="List the last 10 chat sessions with tasks and results",
    )
    return parser


def _apply_chat_args(
    agent: StatefulSorcarAgent,
    args: argparse.Namespace,
    task: str = "",
) -> None:
    """Apply ``-n`` and ``--chat-id`` args to an agent.

    When neither ``-n`` nor ``--chat-id`` is given and *task* is provided,
    attempts to resume a previous chat session for the same task description.

    Args:
        agent: The stateful agent to configure.
        args: Parsed argparse namespace with ``new`` and ``chat_id``.
        task: Task description used to look up a previous session.
    """
    if args.new:
        agent.new_chat()
    elif args.chat_id:
        agent.resume_chat_by_id(args.chat_id)
    elif task:
        agent.resume_chat(task)


def _build_fallback_run_kwargs() -> dict[str, Any]:
    """Build run kwargs when argparse fails (treats all argv as task text).

    Used as fallback when CLI arguments don't match the expected format.

    Returns:
        Dictionary ready to pass to ``agent.run(**kwargs)``.
    """
    import sys

    return {
        "prompt_template": " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "",
        "work_dir": str(Path(".").resolve()),
        "wait_for_user_callback": cli_wait_for_user,
        "ask_user_question_callback": cli_ask_user_question,
    }


def _build_run_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build ``run()`` keyword arguments from parsed CLI args.

    Args:
        args: Parsed argparse namespace.

    Returns:
        Dictionary ready to pass to ``agent.run(**kwargs)``.
    """
    task_description = _resolve_task(args)
    work_dir = args.work_dir or str(Path(".").resolve())
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    model_config: dict[str, Any] = {}
    if args.endpoint:
        model_config["base_url"] = args.endpoint

    return {
        "prompt_template": task_description,
        "model_name": args.model_name,
        "max_budget": args.max_budget,
        "model_config": model_config,
        "work_dir": work_dir,
        "web_tools": not args.no_web,
        "verbose": args.verbose,
        "wait_for_user_callback": cli_wait_for_user,
        "ask_user_question_callback": cli_ask_user_question,
    }


def _print_run_stats(agent: StatefulSorcarAgent, elapsed: float) -> None:
    """Print post-run statistics.

    Args:
        agent: The agent that just finished running.
        elapsed: Wall-clock seconds the run took.
    """
    print(f"\nChat ID: {agent.chat_id}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")
