# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Forget agent — removes a standing instruction stored by ``/remember``.

Typed into a tab as ``/forget <instruction>`` (for example
``/forget Always answer in British English``), the slash command
dispatches this file as a Sorcar Extension Agent through ``run_agent``
with the text as the task.  The agent deletes the matching bullet line
from ``~/.kiss/SORCAR.md`` (``$KISS_HOME/SORCAR.md``), the file that
:meth:`RelentlessAgent.perform_task` appends to every task's system
prompt, so later tasks no longer follow the instruction.  When the text
does not match a stored line exactly, the agent lists the stored
instructions, picks the one the user means, and removes that one.

The file is edited through
:func:`kiss.agents.seas.sorcar_md.remove_instruction` in the daemon
process; the agent runs with the ``bash`` tool profile directly on the
checkout (no worktree, no auto-commit, no classification, no browser,
no memory, no fan-out).
"""

from __future__ import annotations

from typing import Any

from kiss.agents.seas.sorcar_md import list_instructions, remove_instruction

SYSTEM_PROMPT = (
    "You remove a standing instruction the user stored earlier with /remember. The "
    "user's message names the instruction to forget. Procedure: (1) call "
    "`forget_instruction` with the user's message verbatim as `instruction`; (2) if "
    "it reports that nothing matches, read the stored instructions it lists (or call "
    "`list_instructions`), pick the single stored instruction the user's message "
    "refers to, and call `forget_instruction` again with that stored text exactly as "
    "listed; when no stored instruction plausibly matches, or several do and the "
    "message does not single one out, remove nothing; (3) call `finish` with "
    "`success=true` when an instruction was removed, otherwise `success=false`, and "
    "as `summary_in_html` one short HTML paragraph saying what was removed, or why "
    "nothing was removed, listing the stored instructions in a `<ul>` in that case. "
    "Do not use Bash to edit files, and never remove more than one instruction "
    "unless the message clearly asks for several.\n"
)
"""The whole base system prompt of the forget agent (replaces ``SYSTEM.md``)."""


def description() -> str:
    """Return the one-sentence help text shown by ``/forget help``."""
    return (
        "Removes a standing instruction that /remember stored in ~/.kiss/SORCAR.md so "
        "later tasks stop following it; use it as `/forget <instruction text>` in the "
        'chat or `run_agent(agent="forget", task="<instruction text>")`.'
    )


def forget_instruction(instruction: str) -> str:
    """Remove a standing instruction from ~/.kiss/SORCAR.md.

    The instruction is matched against the stored bullet lines ignoring
    case, the bullet marker and the amount of whitespace; the text must
    otherwise be the stored line.  When nothing matches, the reply lists
    the stored instructions so the right one can be chosen.

    Args:
        instruction: The instruction text to remove.

    Returns:
        A one-line report of the removed instruction and the path
        written, or an error listing the stored instructions.
    """
    return remove_instruction(instruction)


def system_prompt() -> str:
    """Return the forget agent's base system prompt (:data:`SYSTEM_PROMPT`)."""
    return SYSTEM_PROMPT


def tools() -> list[Any]:
    """Return the agent's tools: :func:`forget_instruction` and ``list_instructions``."""
    return [forget_instruction, list_instructions]


def tool_profile() -> str:
    """Give the agent the ``Bash`` tool and no other built-in tool."""
    return "bash"


def max_budget() -> float:
    """Return the per-run budget cap in USD: a few tool calls and a finish."""
    return 1.0


def use_worktree() -> bool:
    """Never use a worktree: the agent edits ``~/.kiss/SORCAR.md``, not the checkout."""
    return False


def auto_commit() -> bool:
    """Never auto-commit: nothing in the checkout changes."""
    return False


def classify_tasks() -> bool:
    """Skip the task classifier: the run needs no lite/full prompt choice."""
    return False


def is_parallel() -> bool:
    """Never fan out: one instruction, a few tool calls."""
    return False


def use_web_tools() -> bool:
    """Never enable browser tools."""
    return False


def use_memory() -> bool:
    """Never load persistent memory tools: SORCAR.md is the memory here."""
    return False
