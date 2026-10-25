# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""OpenRouter bestrouter — the ``/bestrouter`` protocol on OpenRouter's Fable 5.1 and Astra.

Same protocol as :mod:`kiss.agents.seas.bestrouter.bestrouter_sea` (one
frontier model does the work, a second one reviews it read-only through
a non-blocking ``run_agent`` on at most 75% of the budget), with both models taken from
the OpenRouter vendor: :data:`PRIMARY_MODEL` is Claude Fable 5.1 and
:data:`REVIEW_MODEL` is GPT-6 Astra, both as ``openrouter/...`` catalog
entries, so one OpenRouter key runs the whole session.

Three ways to run it::

    pick ``orbestrouter`` in the model picker: every task of the tab runs
    through this SEA (see ``register_as_model()`` below)

    /orbestrouter add a --json flag to the export command and cover it with tests

    run_agent(agent="src/kiss/agents/seas/orbestrouter/orbestrouter_sea.py", task="...")
"""

from __future__ import annotations

from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.seas.bestrouter.bestrouter_sea import routing_prompt

PRIMARY_MODEL = "openrouter/anthropic/claude-fable-5.1"
"""The model that does every task: Claude Fable 5.1 via OpenRouter."""

REVIEW_MODEL = "openrouter/openai/gpt-6-astra"
"""The model that reviews and debugs the primary model's work: GPT-6 Astra via OpenRouter."""

SYSTEM_PROMPT = routing_prompt("orbestrouter", PRIMARY_MODEL, REVIEW_MODEL)
"""The routing protocol added to the system prompt of every run."""


class OrbestrouterSea(BaseSea):
    """The ``/orbestrouter`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/orbestrouter help``."""
        return (
            f"Runs every task on {PRIMARY_MODEL} and has {REVIEW_MODEL} review and debug the "
            "result read-only through a non-blocking run_agent on at most 75% of the task "
            "budget (the "
            "bestrouter on OpenRouter models only); pick `orbestrouter` in the model picker, "
            "use `/orbestrouter <task>` in the chat, or "
            'run_agent(agent="orbestrouter", task="...").'
        )

    def register_as_model(self) -> bool:
        """List ``orbestrouter`` in the model picker.

        A picked ``orbestrouter`` makes the daemon run every task of the tab
        through this SEA on :data:`PRIMARY_MODEL` (``settings()`` below), with
        :data:`SYSTEM_PROMPT` added to the system prompt (``system_prompt()``).
        """
        return True

    def system_prompt(self, system_prompt: str) -> str:
        """Add the routing protocol to the default Sorcar system prompt."""
        return system_prompt + "\n\n" + SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Run every task of the tab on :data:`PRIMARY_MODEL`."""
        return settings | {"model": PRIMARY_MODEL}
