# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shell agent — runs the command in the prompt and returns its output.

Typed into an idle tab as ``/sh <command>`` (for example
``/sh git status --short``), the slash command dispatches this file as a
Sorcar Extension Agent through ``run_agent`` on the tab's working
directory, with the command as the task.  The agent runs with the
``bash`` tool profile — ``Bash`` (plus the always-present ``finish``)
and no other built-in tool — directly on the checkout (no worktree, no
auto-commit, no classification, no browser, no memory, no fan-out), so
the command's output is the result of the run.

Module-level getters (``system_prompt()``, ``tool_profile()``, ...)
follow the SEA contract in :mod:`kiss.server.agent_file`.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a shell-command execution assistant. The user's message contains a shell "
    "command they want executed in their own sandbox. Your job: (1) call the Bash tool "
    "with the user's command exactly as written (one call, unmodified), (2) then call the "
    "`finish` tool, passing the captured stdout and stderr text as the result argument. "
    "Keep any commentary brief; do not describe the command instead of running it.\n"
    "The result passed to `finish` must never be empty: copy the command's full output "
    "verbatim, preserving line order and quoting, and include stderr and any error "
    "messages. If the command produced no output, state that it produced no output and "
    "exited successfully (exit code 0). If the command exits non-zero or errors, still "
    "report all output it produced and note the exit code.\n"
)
"""The whole base system prompt of the shell agent (replaces ``SYSTEM.md``)."""


def system_prompt() -> str:
    """Return the shell agent's base system prompt."""
    return SYSTEM_PROMPT


def tool_profile() -> str:
    """Give the agent the ``Bash`` tool and no other built-in tool."""
    return "bash"


def use_worktree() -> bool:
    """Run the command directly in the tab's working directory."""
    return False


def auto_commit() -> bool:
    """Never auto-commit: the run only reports a command's output."""
    return False


def classify_tasks() -> bool:
    """Skip the task classifier: the run needs no lite/full prompt choice."""
    return False


def is_parallel() -> bool:
    """Never fan out: one command, one Bash call."""
    return False


def use_web_tools() -> bool:
    """Never enable browser tools: the command runs in the shell."""
    return False


def use_memory() -> bool:
    """Never load persistent memory tools: a command's output is not knowledge."""
    return False
