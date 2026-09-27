# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bestrouter agent — one frontier model does the work, a second one reviews it.

The routing protocol (:data:`SYSTEM_PROMPT`) fixes both models by name:
:data:`PRIMARY_MODEL` runs every task, including software development, and
:data:`REVIEW_MODEL` is dispatched through ``run_parallel`` for a read-only
review and debugging pass over the primary model's work, on at most 75% of
the task budget.

Three ways to run it::

    pick ``bestrouter`` in the model picker: every task of the tab runs
    through this SEA (see ``register_as_model()`` below)

    /bestrouter add a --json flag to the export command and cover it with tests

    run_agent(agent="src/kiss/agents/seas/bestrouter/bestrouter_sea.py", task="...")

Module-level getters (``add_to_system_prompt()``, ``register_as_model()``,
``model()``, ...) follow the SEA contract in :mod:`kiss.server.agent_file`.
"""

from __future__ import annotations

PRIMARY_MODEL = "claude-fable-5-1"
"""The model that does every task."""

REVIEW_MODEL = "gpt-6-astra"
"""The model that reviews and debugs the primary model's work."""

SYSTEM_PROMPT = f"""\
## Model routing protocol (bestrouter)

Use '{PRIMARY_MODEL}' model for all tasks, including software development. Use
{REVIEW_MODEL} (not codex) using `run_parallel` tool for a thorough read-only review and
debugging of the other model's work. Thoroughly check whether the other model has missed
any code or wiring or introduced any bugs. Use at most 75% of the task budget in
{REVIEW_MODEL} for reviewing and debugging, and ask the model not to invent new problems.
Use the model names literally without hallucinating new model names.
"""
"""The routing protocol added to the system prompt of every run."""


def description() -> str:
    """Return the one-sentence help text shown by ``/bestrouter help``."""
    return (
        f"Runs every task on {PRIMARY_MODEL} and has {REVIEW_MODEL} review and debug the "
        "result read-only through run_parallel on at most 75% of the task budget; pick "
        "`bestrouter` in the model picker, use `/bestrouter <task>` in the chat, or "
        'run_agent(agent="bestrouter", task="...").'
    )


def register_as_model() -> bool:
    """List ``bestrouter`` in the model picker.

    A picked ``bestrouter`` makes the daemon run every task of the tab
    through this SEA on :data:`PRIMARY_MODEL` (``model()`` below), with
    :data:`SYSTEM_PROMPT` added to the system prompt (``add_to_system_prompt()``).
    """
    return True


def add_to_system_prompt() -> str:
    """Add the routing protocol to the default Sorcar system prompt."""
    return SYSTEM_PROMPT


def model() -> str:
    """Run the task on the primary model."""
    return PRIMARY_MODEL
