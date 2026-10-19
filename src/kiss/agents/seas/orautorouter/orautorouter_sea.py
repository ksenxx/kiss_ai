# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""OpenRouter autorouter — the ``/autorouter`` protocol restricted to OpenRouter models.

Same routing as :mod:`kiss.agents.seas.autorouter.autorouter_sea` (split the
task into checkable units, run each on the cheapest tier that passes its
check, escalate on a verified failure, log every decision to
``$KISS_HOME/MODEL_DECISIONS.md``), but every candidate of every tier is an
``openrouter/<vendor>/<model>`` catalog entry, so one OpenRouter key runs the
whole session and no direct vendor key is needed.  :data:`TIERS` lists the
OpenRouter twins of the autorouter's candidates in the same preference
order; the orchestrator itself runs on the autorouter's
:data:`~kiss.agents.seas.autorouter.autorouter_sea.ROUTER_MODEL`, which is
already an OpenRouter model.

Three ways to run it::

    /orautorouter add a --json flag to the export command and cover it with tests

    run_agent(agent="src/kiss/agents/seas/orautorouter/orautorouter_sea.py", task="...")

    pick ``orautorouter`` in the model picker: every task of the tab runs
    through this SEA (see ``register_as_model()`` below)

The protocol, the cost estimate, the observed call costs, the ledger and the
weekly ``/rsi7d autorouter`` evidence job are the autorouter's; only the
menu and the pick (``model_menu``, ``pick_model``) read this module's
:data:`TIERS`.
"""

from __future__ import annotations

from typing import Any

from kiss.agents.seas.autorouter.autorouter_sea import (
    DEFAULT_TOKENS_IN,
    DEFAULT_TOKENS_OUT,
    ROUTER_MODEL,
    Tiers,
    estimate_cost,
    log_decision,
    observed_call_costs,
    routing_prompt,
    schedule_weekly_rsi7d,
    tier_menu,
    tier_pick,
)
from kiss.agents.seas.base.base_sea import BaseSea
from kiss.core.brand import HOME_DIR

VENDOR_PREFIX = "openrouter/"
"""Every candidate of :data:`TIERS` starts with this catalog prefix."""

TIERS: Tiers = {
    "small": (
        ("openrouter/z-ai/glm-5.3-flash", "Z.ai flash; #1 on OpenRouter by weekly tokens"),
        ("openrouter/openai/gpt-6-luna", "OpenAI small model, 1M context; cheapest cost per task"),
        ("openrouter/deepseek/deepseek-v4.1-flash", "DeepSeek flash"),
        ("openrouter/qwen/qwen3.8-flash", "Qwen flash"),
        ("openrouter/openai/gpt-5.6-luna", "previous-generation OpenAI small model"),
        (
            "openrouter/anthropic/claude-haiku-4.5",
            "Anthropic small; 10x gpt-6-luna, only when Anthropic is required",
        ),
    ),
    "medium": (
        (
            "openrouter/google/gemini-3.8-flash",
            "Google flash; best coding score per dollar in the tier",
        ),
        ("openrouter/z-ai/glm-5.3", "Z.ai flagship, open weights"),
        ("openrouter/deepseek/deepseek-v4-pro-0813", "DeepSeek pro"),
        ("openrouter/x-ai/grok-4.7", "xAI"),
        (
            "openrouter/openai/gpt-6-sol",
            "OpenAI mid model, 400k context; cheapest measured per step",
        ),
        (
            "openrouter/anthropic/claude-sonnet-5",
            "Anthropic mid model; default when Anthropic is required",
        ),
        (
            "openrouter/anthropic/claude-opus-5.5",
            "Anthropic frontier; cheapest, fastest frontier per step measured",
        ),
        (
            "openrouter/moonshotai/kimi-k3",
            "Moonshot, 1M context. Best cheaper alternative to claude-fable-5.1. Go-to model "
            "for security analysis and hardening.",
        ),
    ),
    "frontier": (
        (
            "openrouter/openai/gpt-6-astra",
            "OpenAI frontier; priciest per step measured, reliable reviewer",
        ),
        (
            "openrouter/anthropic/claude-fable-5.1",
            "Anthropic top model; longest, hardest tasks only; measured stalls",
        ),
    ),
}
"""Ordered ``(model, note)`` OpenRouter candidates per tier; the first runnable one wins.

The OpenRouter twin of each autorouter candidate, in the autorouter's order
(measured coding quality per dollar); prices come from
:mod:`kiss.core.models.model_info` at call time.  Edit the order when models
or prices change.
"""

SYSTEM_PROMPT = routing_prompt(
    "orautorouter",
    "You are the orautorouter: the autorouter restricted to the OpenRouter vendor. Every "
    "model you dispatch to or `set_model` to is an `openrouter/<vendor>/<model>` catalog "
    "entry; `model_menu` and `pick_model` list only those.",
)
"""The autorouter's protocol re-headed for OpenRouter, with the evidence read as this file loads."""


class OrautorouterSea(BaseSea):
    """The ``/orautorouter`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/orautorouter help``."""
        return (
            "The autorouter restricted to OpenRouter models: splits a task into units, runs "
            "each on the cheapest `openrouter/...` tier (small, medium, frontier) that passes "
            "its acceptance check, escalating on failure and logging every decision to "
            f"~/{HOME_DIR}/MODEL_DECISIONS.md; pick `orautorouter` in the model picker, use "
            '`/orautorouter <task>` in the chat, or run_agent(agent="orautorouter", task="...").'
        )

    def register_as_model(self) -> bool:
        """List ``orautorouter`` in the model picker.

        A picked ``orautorouter`` makes the daemon run every task of the tab
        through this SEA on the autorouter's ``ROUTER_MODEL`` (an OpenRouter
        model), with :data:`SYSTEM_PROMPT` added to the system prompt.
        """
        return True

    def on_picked_as_model(self, work_dir: str) -> str:
        """Keep the shared model evidence fresh: ensure the weekly ``/rsi7d autorouter`` job.

        The evidence block of the protocol is the autorouter's
        ``$KISS_HOME/AUTOROUTER.md``, so the same weekly job serves both routers.

        Args:
            work_dir: Work directory of the run.

        Returns:
            The cron tool's note (``exists``, ``resumed`` or ``created``).
        """
        return schedule_weekly_rsi7d(work_dir)

    def system_prompt(self, system_prompt: str) -> str:
        """Add the OpenRouter routing protocol to the default Sorcar system prompt."""
        return system_prompt + "\n\n" + SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Configure a routed session: the OpenRouter router model, web tools and memory on."""
        return settings | {
            "model": ROUTER_MODEL,
            "auto_classify": False,
            "use_web_tools": True,
            "use_memory": True,
        }

    def tools(self, tools: list[Any]) -> list[Any]:
        """Expose the OpenRouter menu and pick, the cost estimate, observed costs, the ledger."""
        return tools + [model_menu, pick_model, estimate_cost, observed_call_costs, log_decision]


def model_menu(tokens_in: int = DEFAULT_TOKENS_IN, tokens_out: int = DEFAULT_TOKENS_OUT) -> str:
    """List every tier's OpenRouter candidates with prices, cost estimate and availability.

    Args:
        tokens_in: Prompt tokens the unit is expected to consume (default 200k).
        tokens_out: Completion tokens the unit is expected to produce (default 20k).

    Returns:
        JSON: tier name -> ordered list of ``{model, input_per_1M, output_per_1M,
        estimated_usd, runnable, note}`` records, every model an ``openrouter/...``
        entry; ``runnable`` is true when the OpenRouter key is configured.
    """
    return tier_menu(TIERS, tokens_in, tokens_out)


def pick_model(
    tier: str,
    tokens_in: int = DEFAULT_TOKENS_IN,
    tokens_out: int = DEFAULT_TOKENS_OUT,
    exclude: str = "",
) -> str:
    """Pick the cheapest runnable OpenRouter model of a routing tier.

    Args:
        tier: ``small``, ``medium`` or ``frontier``.
        tokens_in: Prompt tokens the unit is expected to consume (default 200k).
        tokens_out: Completion tokens the unit is expected to produce (default 20k).
        exclude: Models that already failed this unit, separated by commas or spaces.

    Returns:
        JSON ``{model, tier, input_per_1M, output_per_1M, estimated_usd, note}`` of
        the first ``openrouter/...`` candidate of the tier that is runnable, or an
        ``Error:`` line when the tier is unknown or has no runnable candidate.
    """
    return tier_pick(TIERS, tier, tokens_in, tokens_out, exclude)
