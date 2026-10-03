# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Remember agent — stores the prompt as a standing instruction.

Typed into a tab as ``/remember <instruction>`` (for example
``/remember Always answer in British English``), the slash command
dispatches this file as a Sorcar Extension Agent through ``run_agent``
with the instruction as the task.  The agent appends the instruction as
a bullet line to ``~/.kiss/AGENTS.md`` (``$KISS_HOME/AGENTS.md``), the
user-authored file that :meth:`RelentlessAgent.perform_task` appends
to the system prompt of every task, so every later task follows it.
``/forget <instruction>`` (:mod:`kiss.agents.seas.forget.forget_sea`) removes
it again.

The file is edited through :func:`kiss.agents.seas.agents_md.add_instruction`
in the daemon process; the agent runs with the ``bash`` tool profile
directly on the checkout (no worktree, no auto-commit, no
classification, no browser, no memory, no fan-out).
"""

from __future__ import annotations

from typing import Any

from kiss.agents.seas.agents_md import add_instruction, list_instructions
from kiss.core.brand import HOME_DIR

SYSTEM_PROMPT = (
    "You store a standing instruction for the user. The user's message is the "
    "instruction they want every future Sorcar task to follow. Procedure: (1) call "
    "`remember_instruction` once with the user's message verbatim as `instruction` "
    "(do not rephrase, shorten, or add to it; keep its exact wording); (2) call "
    "`finish` with `success=true` and, as `summary_in_html`, one short HTML "
    "paragraph quoting the tool's reply. Do not use Bash to edit files, do not "
    "store more than one instruction unless the message clearly lists several "
    "separate instructions, and if the message is empty finish with `success=false` "
    "saying that nothing was given to remember.\n"
)
"""The whole base system prompt of the remember agent (replaces ``SYSTEM.md``)."""


def description() -> str:
    """Return the one-sentence help text shown by ``/remember help``."""
    return (
        f"Stores the prompt as a standing instruction in ~/{HOME_DIR}/AGENTS.md so every "
        "future Sorcar task follows it; use `/remember <instruction>` in the chat or "
        'run_agent(agent="remember", task="<instruction>"), and `/forget` to remove it.'
    )


def remember_instruction(instruction: str) -> str:
    """Add a standing instruction to ~/.kiss/AGENTS.md so every future task follows it.

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


def system_prompt() -> str:
    """Return the remember agent's base system prompt (:data:`SYSTEM_PROMPT`)."""
    return SYSTEM_PROMPT


def tools() -> list[Any]:
    """Return the agent's tools: :func:`remember_instruction` and ``list_instructions``."""
    return [remember_instruction, list_instructions]


def tool_profile() -> str:
    """Give the agent the ``Bash`` tool and no other built-in tool."""
    return "bash"


def max_budget() -> float:
    """Return the per-run budget cap in USD: one tool call and a finish."""
    return 1.0


def use_worktree() -> bool:
    """Never use a worktree: the agent edits ``~/.kiss/AGENTS.md``, not the checkout."""
    return False


def auto_commit() -> bool:
    """Never auto-commit: nothing in the checkout changes."""
    return False


def classify_tasks() -> bool:
    """Skip the task classifier: the run needs no lite/full prompt choice."""
    return False


def is_parallel() -> bool:
    """Never fan out: one instruction, one tool call."""
    return False


def use_web_tools() -> bool:
    """Never enable browser tools."""
    return False


def use_memory() -> bool:
    """Never load persistent memory tools: AGENTS.md is the memory here."""
    return False
