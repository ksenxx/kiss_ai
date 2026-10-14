# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Remember agent — stores the prompt as a standing instruction.

Typed into a tab as ``/remember <instruction>`` (for example
``/remember Always answer in British English``), the slash command
dispatches this file as a Sorcar Extension Agent through ``run_agent``
with the instruction as the task.  The agent appends the instruction as
a bullet line to ``$KISS_HOME/AGENTS.md`` (``$KISS_HOME/AGENTS.md``), the
user-authored file that :meth:`RelentlessAgent.perform_task` appends
to the system prompt of every task, so every later task follows it.
``/forget <instruction>`` (:mod:`kiss.agents.seas.forget.forget_sea`) removes
it again.

The file is edited through :func:`kiss.agents.seas.agents_md.add_instruction`
in the daemon process; the agent runs with the ``bash`` tool profile
directly on the checkout (no worktree, no auto-commit, no
classification, no browser, no memory).
"""

from __future__ import annotations

from typing import Any

from kiss.agents.seas.agents_md import add_instruction, list_instructions
from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.core.brand import HOME_DIR

SYSTEM_PROMPT = (
    "You store a standing instruction for the user. The user's message is the "
    "instruction every future Sorcar task must follow. (1) Call `remember_instruction` "
    "once with the message verbatim as `instruction`; do not rephrase, shorten or add "
    "to it. (2) Call `finish` with `success=true` and, as `summary_in_html`, one short "
    "HTML paragraph quoting the tool's reply. Do not edit files with Bash. Store one "
    "instruction unless the message clearly lists several. If the message is empty, "
    "finish with `success=false` and say nothing was given to remember.\n"
)
"""The whole base system prompt of the remember agent (replaces ``SYSTEM.md``)."""


class RememberSea(WorkerSea):
    """The ``/remember`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/remember help``."""
        return (
            f"Stores the prompt as a standing instruction in ~/{HOME_DIR}/AGENTS.md so every "
            "future Sorcar task follows it; use `/remember <instruction>` in the chat or "
            'run_agent(agent="remember", task="<instruction>"), and `/forget` to remove it.'
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Return the agent's base system prompt (:data:`SYSTEM_PROMPT`)."""
        return SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A $1 worker with Bash only, running :data:`SYSTEM_PROMPT`."""
        return settings | {
            "tool_profile": "bash",
            "max_budget": 1.0,
        }

    def tools(self, tools: list[Any]) -> list[Any]:
        """Return the agent's tools: :func:`remember_instruction` and ``list_instructions``."""
        return tools + [remember_instruction, list_instructions]


def remember_instruction(instruction: str) -> str:
    """Add a standing instruction to $KISS_HOME/AGENTS.md so every future task follows it.

    The instruction is stored as one bullet line (newlines collapsed to
    spaces).  Adding an instruction that is already stored (compared
    ignoring case and spacing) changes nothing.

    Args:
        instruction: The instruction text, exactly as the user wrote it.

    Returns:
        A one-line report: the path written and the stored text, or why
        nothing was written.
    """
    return add_instruction(instruction)


