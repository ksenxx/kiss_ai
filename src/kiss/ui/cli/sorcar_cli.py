# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``sorcar`` console-script entry point.

The entry point lives in the UI layer (not in
``kiss.agents.sorcar.worktree_sorcar_agent``, its historical home)
because it dispatches into the interactive terminal client
(:mod:`kiss.ui.cli.cli_client`), the steering REPL
(:mod:`kiss.ui.cli.cli_steering`) and the MCP subcommand
(:mod:`kiss.ui.cli.mcp_cli`) — and sorcar code must only depend on
itself and ``kiss.core``.  The UI layer may freely import sorcar, so
all argument parsing and run-kwarg construction still comes from
:mod:`kiss.agents.sorcar.cli_helpers`.
"""

import sys
import time
from pathlib import Path

from kiss.agents.sorcar.cli_helpers import (
    _build_arg_parser,
    _build_run_kwargs,
    _launch_work_dir,
    print_outcome,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import (
    _reject_interactive_only_flags,
)


def main() -> None:
    """Run the ``sorcar`` CLI.

    Two modes:

    * **Interactive** (no ``-t/--task`` / ``-f/--file``): a thin
      terminal client of the local ``sorcar web`` daemon — see
      :mod:`kiss.ui.cli.cli_client`.  The ``--worktree`` /
      ``--no-worktree`` / ``--parallel`` / ``--no-parallel`` /
      ``--auto-commit`` / ``--no-auto-commit`` flags are forwarded
      to the daemon so each task can still run on an isolated git
      worktree, with parallel sub-agents and auto-commit.
      Chat-session control (new chat, resume) is driven from the
      interactive client's slash commands rather than CLI flags.
    * **Non-interactive** (``-t`` or ``-f`` supplied): runs a plain
      :class:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent` once on
      the supplied task and exits.  No git worktree isolation, no
      chat-session control — those features were always tied to the
      removed ``-c/--chat-id`` / ``-l/--list-chat-id`` /
      ``--cleanup`` / ``--use-chat`` / ``--use-worktree`` flag set.
      ``--worktree`` / ``--no-worktree`` / ``--auto-commit`` /
      ``--no-auto-commit`` are interactive-only and are rejected
      when combined with ``-t`` / ``-f`` (see
      :func:`~kiss.agents.sorcar.worktree_sorcar_agent._reject_interactive_only_flags`).
      Display events from the run are still streamed into the local
      chat DB via :class:`~kiss.ui.cli.cli_printer.RecordingConsolePrinter`
      so the run is replayable in the chat webview; only the *chat
      session* surface (resume by id) is unavailable.

    ``sorcar mcp ...`` is dispatched to the MCP management subcommand
    (:mod:`kiss.ui.cli.mcp_cli`) before normal argument parsing.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        from kiss.ui.cli.mcp_cli import run_mcp_cli

        sys.exit(run_mcp_cli(sys.argv[2:], str(Path.cwd())))

    parser = _build_arg_parser()
    args = parser.parse_args()
    work_dir = args.work_dir or _launch_work_dir()

    interactive = args.task is None and args.file is None
    if not interactive:
        # Validate AFTER argparse (so ``-t``/``-f`` are decoded
        # correctly) but BEFORE the agent is built / the LLM is
        # contacted (so the user sees the error immediately and no
        # budget is spent).
        _reject_interactive_only_flags(sys.argv)
    # ``RecordingConsolePrinter`` records every display event to the
    # chat DB AND renders the Rich panels to the terminal, so the same
    # run is visible live in the terminal and replayable later in the
    # chat webview.  Passed as a factory because the sorcar layer
    # (cli_helpers) must not import the UI layer itself.
    from kiss.ui.cli.cli_printer import RecordingConsolePrinter

    run_kwargs = _build_run_kwargs(args, printer_factory=RecordingConsolePrinter)

    if interactive:
        from kiss.ui.cli.cli_client import run_client

        sys.exit(
            run_client(
                work_dir=run_kwargs.get("work_dir") or work_dir,
                model_name=run_kwargs.get("model_name", ""),
                active_file=run_kwargs.get("current_editor_file") or "",
                use_worktree=bool(getattr(args, "worktree", True)),
                use_parallel=bool(getattr(args, "parallel", True)),
                # Fallback default matches the argparse default
                # (``--auto-commit`` default=True in cli_helpers) and
                # the sibling ``worktree``/``parallel`` fallbacks.
                auto_commit=bool(getattr(args, "auto_commit", True)),
            ),
        )

    # Non-interactive: plain SorcarAgent, no chat / no worktree.
    from kiss.ui.cli.cli_steering import run_with_steering

    agent: SorcarAgent = SorcarAgent("Sorcar Agent")
    start_time = time.time()
    result = run_with_steering(agent, run_kwargs)
    elapsed = time.time() - start_time
    print_outcome(agent, result, elapsed, run_kwargs.get("verbose", True))


if __name__ == "__main__":
    main()
