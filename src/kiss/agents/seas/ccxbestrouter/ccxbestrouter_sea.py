# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Claude Code + Codex bestrouter — the ``/bestrouter`` protocol on ``cc/`` and ``codex/`` models.

Same protocol as :mod:`kiss.agents.seas.bestrouter.bestrouter_sea` (one
frontier model does the work, a second one reviews it read-only through
a non-blocking ``run_agent`` on at most 75% of the budget), with the Claude
models taken through Claude Code and the reviewer through Codex:
:data:`SIMPLE_MODEL` is Claude Opus 5.5 (medium effort) for simple questions,
:data:`PRIMARY_MODEL` is Claude Fable 5.2 (max effort) and :data:`REVIEW_MODEL`
is GPT-6 Astra (xhigh effort).

Three ways to run it::

    pick ``ccxbestrouter`` in the model picker: every task of the tab runs
    through this SEA (see ``register_as_model()`` below)

    /ccxbestrouter add a --json flag to the export command and cover it with tests

    run_agent(agent="src/kiss/agents/seas/ccxbestrouter/ccxbestrouter_sea.py", task="...")
"""

from __future__ import annotations

from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.seas.bestrouter.bestrouter_sea import routing_prompt

SIMPLE_MODEL = "cc/claude-opus-5-5-medium"
"""The model a tab starts on: Claude Opus 5.5 at medium effort via Claude Code."""

PRIMARY_MODEL = "cc/claude-fable-5-2-max"
"""The model for every other task: Claude Fable 5.2 at max effort via Claude Code."""

REVIEW_MODEL = "codex/gpt-6-astra-xhigh"
"""The model that reviews and debugs the primary model's work: GPT-6 Astra (xhigh) via Codex.

An effort alias the catalog does not list: it runs, and is sized and priced, as
``codex/gpt-6-astra``, and the Codex adapter passes the effort to the CLI.
"""

SYSTEM_PROMPT = routing_prompt(
    "ccxbestrouter", SIMPLE_MODEL, PRIMARY_MODEL, REVIEW_MODEL
).replace(" (not codex)", "", 1)
"""The routing protocol added to the system prompt of every run.

The bestrouter's "(not codex)" warning is dropped: the reviewer here is the
Codex model on purpose.
"""


class CcxbestrouterSea(BaseSea):
    """The ``/ccxbestrouter`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/ccxbestrouter help``."""
        return (
            f"Answers simple questions on {SIMPLE_MODEL}, runs every other task on "
            f"{PRIMARY_MODEL} and has {REVIEW_MODEL} review and debug the "
            "result read-only through a non-blocking run_agent on at most 75% of the task "
            "budget (the bestrouter on Claude Code and Codex models); pick `ccxbestrouter` "
            "in the model picker, use `/ccxbestrouter <task>` in the chat, or "
            'run_agent(agent="ccxbestrouter", task="...").'
        )

    def register_as_model(self) -> bool:
        """List ``ccxbestrouter`` in the model picker.

        A picked ``ccxbestrouter`` makes the daemon run every task of the tab
        through this SEA on :data:`SIMPLE_MODEL` (``settings()`` below), with
        :data:`SYSTEM_PROMPT` added to the system prompt (``system_prompt()``).
        """
        return True

    def system_prompt(self, system_prompt: str) -> str:
        """Add the routing protocol to the default Sorcar system prompt."""
        return system_prompt + "\n\n" + SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Start every task of the tab on :data:`SIMPLE_MODEL`."""
        return settings | {"model": SIMPLE_MODEL}
