# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bestrouter agent — one frontier model does the work, a second one reviews it.

The routing protocol (:data:`SYSTEM_PROMPT`) fixes both models by name:
:data:`PRIMARY_MODEL` runs every task, including software development, and
:data:`REVIEW_MODEL` is dispatched through a non-blocking ``run_agent`` for a read-only
review and debugging pass over the primary model's work, on at most 75% of
the task budget.

Three ways to run it::

    pick ``bestrouter`` in the model picker: every task of the tab runs
    through this SEA (see ``register_as_model()`` below)

    /bestrouter add a --json flag to the export command and cover it with tests

    run_agent(agent="src/kiss/agents/seas/bestrouter/bestrouter_sea.py", task="...")

The :class:`BestrouterSea` methods (``settings()``, ``system_prompt()``,
``register_as_model()``, ...) follow the SEA contract in :mod:`kiss.agents.seas.base.base_sea`.
"""

from __future__ import annotations

from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea

PRIMARY_MODEL = "claude-fable-5-1-max"
"""The model that does every task."""

REVIEW_MODEL = "gpt-6-astra"
"""The model that reviews and debugs the primary model's work."""

SYSTEM_PROMPT = f"""\
## Model routing protocol (bestrouter)

Use '{PRIMARY_MODEL}' model for all tasks, including software development. Use
{REVIEW_MODEL} (not codex) through `run_agent(model="{REVIEW_MODEL}", tool_profile="review",
wait="false")` for a thorough read-only review and debugging of the other model's work,
keep working while it runs and collect it with `agent_job(job_id, "wait")`. Check
thoroughly whether the other model missed code or wiring or introduced bugs. Use at most
75% of the task budget in {REVIEW_MODEL}, leave the reviewer's `max_budget` empty (it
follows the size of the diff), and ask the model not to invent new problems. A second
round gets the first round's findings and the diff since then, not the whole change. Use
the model names literally without hallucinating new model names.

Working rules:
- `Read` a file region before its first `Edit`, and `Read` an existing file before a
  `Write` that replaces it; an unread file is refused either way. After an `Edit` is
  rejected as unread, `Read` the file before any further `Edit` of it.
- Before each `run_parallel` / `run_agent` wave, add up the children's `max_budget`s and
  compare with your remaining budget (the `Budget: $spent/$max` footer counts the
  children). Keep at least 15% for the final verification and report. If the wave does
  not fit, do not launch it: report what is done and finish.
- Write a file longer than about 150 lines in `Write` / `Edit` calls of at most 150 lines
  each, the first chunk on disk before you compose the next. Save gathered facts to
  `tmp/` files before composing.
- When the `Steps:` footer of a tool result ends in 9, make your next call `summary`;
  otherwise the 10th, 20th, ... call is refused.
"""
"""The routing protocol added to the system prompt of every run.

A module-level f-string so ``/rsi7d`` can tune it; routers derived from
this one (``orbestrouter``) take it through :func:`routing_prompt`.
"""


def routing_prompt(router: str, primary: str, review: str) -> str:
    """Return :data:`SYSTEM_PROMPT` for a derived router with its own worker and reviewer.

    The heading names *router* and every mention of :data:`PRIMARY_MODEL`
    and :data:`REVIEW_MODEL` becomes *primary* and *review*; the rules are
    shared word for word.

    Args:
        router: The SEA's command name, shown in the heading
            ``## Model routing protocol (<router>)``.
        primary: Catalog name of the model that does every task.
        review: Catalog name of the model that reviews the primary model's work.
    """
    return (
        SYSTEM_PROMPT.replace("(bestrouter)", f"({router})", 1)
        .replace(PRIMARY_MODEL, primary)
        .replace(REVIEW_MODEL, review)
    )


class BestrouterSea(BaseSea):
    """The ``/bestrouter`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/bestrouter help``."""
        return (
            f"Runs every task on {PRIMARY_MODEL} and has {REVIEW_MODEL} review and debug the "
            "result read-only through a non-blocking run_agent on at most 75% of the task "
            "budget; pick "
            "`bestrouter` in the model picker, use `/bestrouter <task>` in the chat, or "
            'run_agent(agent="bestrouter", task="...").'
        )

    def register_as_model(self) -> bool:
        """List ``bestrouter`` in the model picker.

        A picked ``bestrouter`` makes the daemon run every task of the tab
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


