"""Shared CLI helpers for Sorcar agent entry points.

Provides argument parsing, chat-session handling, run-kwarg construction,
and post-run statistics.
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
        print(f"\n{'=' * 72}")
        print(f"Chat ID: {entry['chat_id']}")
        print(f"{'=' * 72}")
        tasks: list[dict[str, object]] = entry["tasks"]  # type: ignore[assignment]
        for i, t in enumerate(tasks, 1):
            ts = float(t.get("timestamp", 0))  # type: ignore[arg-type]
            dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            task_text = str(t.get("task", ""))[:200]
            result_text = str(t.get("result", ""))[:200]
            print(f"\n  Task {i} [{dt}]:")
            print(f"    {task_text}")
            if result_text:
                print(f"  Result {i}:")
                print(f"    {result_text}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for all Sorcar agent entry points."""
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
        "--no-web", action="store_true", default=False,
        help="Disable browser/web tools (terminal-only mode)",
    )
    parser.add_argument(
        "-p", "--parallel", action="store_true", default=False,
        help="Enable parallel subagents",
    )
    parser.add_argument(
        "-t", "--task", type=str, default=None, help="Task description"
    )
    parser.add_argument(
        "-f", "--file", type=str, default=None,
        help="Path to a file whose contents to use as the task",
    )
    parser.add_argument(
        "-n", "--new", action="store_true", help="Start a new chat session",
    )
    parser.add_argument(
        "-c", "--chat-id", type=int, default=None, help="Resume a chat session by ID",
    )
    parser.add_argument(
        "-l", "--list-chat-id", action="store_true",
        help="List the last 10 chat sessions with tasks and results",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Scan for and clean up orphaned worktree branches",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--use-chat", action="store_true",
        help="Use chat mode",
    )
    group.add_argument(
        "--use-worktree", action="store_true",
        help="Use both chat mode and git worktree for isolation (for advanced users)",
    )
    return parser


def _apply_chat_args(
    agent: StatefulSorcarAgent,
    args: argparse.Namespace,
    task: str = "",
) -> None:
    """Apply ``-n`` / ``--chat-id`` args to *agent*, or resume by *task*."""
    if args.new:
        agent.new_chat()
    elif args.chat_id:
        agent.resume_chat_by_id(args.chat_id)
    elif task:
        agent.resume_chat(task)


def _build_run_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build ``agent.run()`` keyword arguments from parsed CLI args."""
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
        "is_parallel": args.parallel,
        "verbose": args.verbose,
        "ask_user_question_callback": cli_ask_user_question,
    }


def _print_run_stats(agent: StatefulSorcarAgent, elapsed: float) -> None:
    """Print post-run statistics (chat ID, time, cost, tokens)."""
    print(f"\nChat ID: {agent.chat_id}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")
