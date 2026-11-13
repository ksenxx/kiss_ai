# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bedrock bestrouter — the ``/bestrouter`` protocol on Amazon Bedrock's Claude models.

Same protocol as :mod:`kiss.agents.seas.bestrouter.bestrouter_sea` (one
frontier model does the work, a second one reviews it read-only through
a non-blocking ``run_agent`` on at most 75% of the budget), with every model taken from
Amazon Bedrock's global cross-region inference profiles: :data:`SIMPLE_MODEL` is
Claude Opus 5.5 (medium effort) for simple questions, :data:`PRIMARY_MODEL` is
Claude Opus 5.5 (max effort) and :data:`REVIEW_MODEL` is Claude Fable 5.1 (max
effort), all as ``bedrock/global.anthropic...`` catalog entries, so one set of
AWS credentials runs the whole session.

Three ways to run it::

    pick ``brbestrouter`` in the model picker: every task of the tab runs
    through this SEA (see ``register_as_model()`` below)

    /brbestrouter add a --json flag to the export command and cover it with tests

    run_agent(agent="src/kiss/agents/seas/brbestrouter/brbestrouter_sea.py", task="...")
"""

from __future__ import annotations

from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.seas.bestrouter.bestrouter_sea import routing_prompt

SIMPLE_MODEL = "bedrock/global.anthropic.claude-opus-5-5-medium"
"""The model a tab starts on: Claude Opus 5.5 at medium effort via Bedrock."""

PRIMARY_MODEL = "bedrock/global.anthropic.claude-opus-5-5-max"
"""The model for every other task: Claude Opus 5.5 at max effort via Bedrock."""

REVIEW_MODEL = "bedrock/global.anthropic.claude-fable-5-1-max"
"""The model that reviews and debugs the primary model's work: Claude Fable 5.1 at max
effort via Bedrock."""

SYSTEM_PROMPT = routing_prompt(
    "brbestrouter", SIMPLE_MODEL, PRIMARY_MODEL, REVIEW_MODEL
).replace(" (not codex)", "", 1)
"""The routing protocol added to the system prompt of every run.

The bestrouter's "(not codex)" warning is dropped: it tells GPT-6 Astra from its
Codex variant and means nothing for a Claude reviewer.
"""


class BrbestrouterSea(BaseSea):
    """The ``/brbestrouter`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/brbestrouter help``."""
        return (
            f"Answers simple questions on {SIMPLE_MODEL}, runs every other task on "
            f"{PRIMARY_MODEL} and has {REVIEW_MODEL} review and debug the "
            "result read-only through a non-blocking run_agent on at most 75% of the task "
            "budget (the bestrouter on Amazon Bedrock models only); pick `brbestrouter` "
            "in the model picker, use `/brbestrouter <task>` in the chat, or "
            'run_agent(agent="brbestrouter", task="...").'
        )

    def register_as_model(self) -> bool:
        """List ``brbestrouter`` in the model picker.

        A picked ``brbestrouter`` makes the daemon run every task of the tab
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
