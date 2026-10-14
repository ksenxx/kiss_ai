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
auto-commit, no classification, no browser, no memory), so
the command's output is the result of the run.

``settings()`` (a ``WorkerSea`` with the ``bash`` tool profile) and
the methods ``system_prompt()`` / ``prompt(task)`` of :class:`ShSea` follow the
SEA contract in :mod:`kiss.agents.seas.base.base_sea`.
"""

from __future__ import annotations

from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea

SYSTEM_PROMPT = (
    "You run one shell command for the user in their sandbox. The user's message is the "
    "command. Your job: (1) call the Bash tool with the user's command exactly as written, "
    "in one call; (2) call the `finish` tool with the captured stdout and stderr as the "
    "result. Keep commentary brief; do not describe the command instead of running "
    "it.\n"
    "The result passed to `finish` must never be empty. Copy the full output verbatim, in "
    "order, with stderr and error messages. If there was no output, say so and that the "
    "command exited 0. If it exited non-zero, report all output and the exit code.\n"
    "Every reply is a tool call. If you will not run the command, call `finish` at once "
    "with `success=false` and a one-line reason; a prose-only reply is recorded as the "
    "command's output.\n"
)
"""The whole base system prompt of the shell agent (replaces ``SYSTEM.md``)."""


class ShSea(WorkerSea):
    """The ``/sh`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/sh help``."""
        return (
            "Runs the shell command given in the prompt directly on the tab's working directory "
            "with only the Bash tool and returns its verbatim output; use it as "
            '`/sh <command>` in the chat (e.g. `/sh git status --short`) or run_agent(agent="sh", '
            'task="<command>").'
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Return the agent's base system prompt (:data:`SYSTEM_PROMPT`)."""
        return SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A worker with Bash only, on the real checkout, running :data:`SYSTEM_PROMPT`.

        ``tool_profile`` is ``locked``: ``/sh`` is the Bash-only worker, so a
        ``run_agent(agent="sh", tool_profile="review")`` is refused rather
        than run as something else.
        """
        return settings | {
            "tool_profile": "bash",
            "locked": ["tool_profile"],
        }


