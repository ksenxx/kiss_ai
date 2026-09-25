# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Autoroute agent — runs a task on the cheapest model tier that will finish it.

Frontier models cost 40 to 100 times more per token than small models, and
most agent tokens go to exploration, file reads, test output and mechanical
edits that a small model handles as well.  This SEA takes a task, splits it
into units of work that each have a mechanical acceptance check, classifies
every unit into a tier (``small``, ``medium``, ``frontier``) with the
non-generative ``decide`` tool, picks the cheapest runnable model of that
tier from the local catalog, dispatches the unit to that model through the
built-in ``run_agent`` tool (or switches its own model with ``set_model`` at
a phase boundary), verifies the result through the
acceptance check, escalates one tier up on a verified failure, and logs every
decision to the ledger ``~/.kiss/MODEL_DECISIONS.md`` (``$KISS_HOME`` when
set), which is shared by every task so it accumulates the routing history of
the installation; each row carries the task id of the run that wrote it.  The
objective is cost per accepted task, not cost per token.

Two ways to run it::

    /autoroute add a --json flag to the export command and cover it with tests

    run_agent(agent="src/kiss/agents/seas/autoroute_sea.py", task="...")

The routing protocol is the system prompt (:data:`SYSTEM_PROMPT`); the
deterministic parts — the priced candidate menu, the pick, the cost estimate
and the decision ledger — are the tools this module exposes through
``tools()``.  Candidate order per tier is :data:`TIERS`, ranked by measured
coding quality per dollar (researched 2026-09-24); prices and availability
come from :mod:`kiss.core.models.model_info` at call time, so the menu is
always the one this installation can run.

Module-level getters (``system_prompt()``, ``is_parallel()``, ...) follow the
SEA contract in :mod:`kiss.server.agent_file`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kiss.core.config import kiss_home
from kiss.core.models.model_info import MODEL_INFO, get_available_models
from kiss.server.agent_state import current_agent

TIER_NAMES = ("small", "medium", "frontier")
"""The routing tiers, cheapest first."""

TIERS: dict[str, tuple[tuple[str, str], ...]] = {
    "small": (
        ("zai-org/GLM-5.3-Flash", "Z.ai flash; #1 on OpenRouter by weekly tokens"),
        ("openrouter/z-ai/glm-5.3-flash", "same model via OpenRouter"),
        ("gpt-6-luna", "OpenAI small model, 1M context; cheapest cost per task"),
        ("openrouter/openai/gpt-6-luna", "same model via OpenRouter"),
        ("deepseek-ai/DeepSeek-V4.1-Flash", "DeepSeek flash"),
        ("openrouter/qwen/qwen3.8-flash", "Qwen flash"),
        ("gpt-5.6-luna", "previous-generation OpenAI small model"),
        ("claude-haiku-4-5", "Anthropic small; 10x gpt-6-luna, only when Anthropic is required"),
    ),
    "medium": (
        ("gemini-3.8-flash", "Google flash; best coding score per dollar in the tier"),
        ("openrouter/google/gemini-3.8-flash", "same model via OpenRouter"),
        ("zai-org/GLM-5.3", "Z.ai flagship, open weights"),
        ("openrouter/z-ai/glm-5.3", "same model via OpenRouter"),
        ("deepseek-ai/DeepSeek-V4-Pro-0813", "DeepSeek pro"),
        ("openrouter/x-ai/grok-4.7", "xAI"),
        ("gpt-6-sol", "OpenAI mid model, 400k context"),
        ("claude-sonnet-5", "Anthropic mid model; default when Anthropic is required"),
        ("kimi-k3", "Moonshot, 1M context"),
    ),
    "frontier": (
        ("claude-opus-5-5", "Anthropic frontier; cheapest frontier per token"),
        ("gpt-6-astra", "OpenAI frontier"),
        ("claude-fable-5-1", "Anthropic top model; longest and hardest tasks only"),
    ),
}
"""Ordered ``(model, note)`` candidates per tier; the first runnable one wins.

The order is by measured coding quality per dollar as of 2026-09-24.  Edit
it when models or prices change; prices themselves are read from the
catalog, never stored here.
"""

DEFAULT_TOKENS_IN = 200_000
"""Prompt tokens a sub-agent that reads a medium codebase and runs tests uses."""

DEFAULT_TOKENS_OUT = 20_000
"""Completion tokens such a sub-agent produces."""

LEDGER_NAME = "MODEL_DECISIONS.md"
"""File name of the routing ledger inside the KISS home directory (``~/.kiss``)."""

LEDGER_HEADER = (
    "# Model routing decisions\n\n"
    "| time (UTC) | task_id | unit | tier | model | reason | outcome |\n"
    "|---|---|---|---|---|---|---|\n"
)
"""Title and table header written when the ledger is created."""

SYSTEM_PROMPT = """\
You are the autoroute agent. You receive a task and finish it at the lowest cost per
accepted result by routing every unit of work to the cheapest model tier that will
complete it correctly on the first attempt, escalating only on a verified failure. A
cheap model that fails, retries and then escalates costs more than routing correctly the
first time, so the objective is cost per accepted task, not cost per token.

Tiers, cheapest first: small, medium, frontier. Concrete models, prices and availability
come from your tools (`model_menu`, `pick_model`, `estimate_cost`); never invent a model
name or a price.

## Protocol

1. Split the task into units with a mechanical acceptance check. A unit is dispatchable
   when you can state what proves it done without reading the transcript: a passing test
   command, a `git diff --stat` touching only the expected files, a grep that finds the new
   symbol, a file that exists with the expected sections. Work without such a check stays
   in your own loop. Never route a single tool call; the granularity is one unit with a
   checkable result.

2. Classify each unit with `decide` (non-generative, costs a fraction of a cent). Put the
   unit description, the acceptance check, the file count and size, and any earlier
   failure of the same unit into `state` and ask exactly these questions:
   - `tier`, type `choice`, instructions "Cheapest model tier that completes this unit
     correctly on the first attempt, judged by the acceptance check.", criteria:
     small = "Mechanical or lookup work with an unambiguous spec: read or summarize files
     or logs, grep and report, run tests and report failures, rename or move symbols,
     format, write boilerplate or docstrings from a template, fill a table, translate a
     config between formats."; medium = "Standard engineering with a clear spec:
     implement a function or endpoint from a description, write tests for existing code,
     fix a bug whose cause is known, refactor within one module, document existing
     behaviour, review a small diff against stated rules."; frontier = "Open-ended
     reasoning: find the root cause of a failure, design across modules or services,
     resolve conflicting requirements, security or concurrency review, anything where a
     wrong answer is expensive to detect, and the final acceptance judgement of the whole
     task."
   - `guarded`, type `noul`, instructions "Does the unit touch authentication, secrets,
     payments, data deletion, migrations, or any action the user cannot undo, or does it
     decide whether the whole task is complete?"
   Apply the answer mechanically: `guarded` >= 0.5 means frontier whatever `tier` says;
   `tier` confidence below 0.6 moves one tier up; ties go to the expensive tier; a unit
   that already failed on a tier starts one tier higher with the failed model in
   `exclude`. If `decide` is not among your tools, judge the unit yourself against the
   same criteria and say so in the ledger reason.

3. Pick a concrete model with `pick_model(tier, tokens_in, tokens_out, exclude)`. Pass
   realistic token counts: a sub-agent that reads a medium codebase and runs tests uses
   about 200k prompt and 20k completion tokens. Then dispatch on the sub-agent boundary,
   never mid-context: `run_agent(task=..., model_name=<picked>)`, one call per unit. The
   sub-agent starts a fresh session with the default Sorcar prompt and toolset, so its
   task text must name the files it may touch and the check that ends it; a small-tier
   agent must not expand its own scope. Units run one after another; a unit made of
   independent parts may tell its sub-agent to fan them out with its own `run_parallel`.
   Sub-agents return a summary, never raw output.

4. Phase plan for work you keep in your own loop. Plan and final acceptance on the
   frontier model; execution on medium; volume work (exploration, test runs, log reading,
   documentation drafts) on small sub-agents. Prompt caches are per model, so `set_model`
   only at a phase boundary, never inside a phase: read the code and write the plan and
   the acceptance checks, `set_model(<medium pick>)` once to implement and run the tests,
   `set_model(<original model>)` once for root-cause work after two failed checks and for
   the final verification. Skip the downgrade when the task is short (under roughly 10
   tool calls), when the user pinned a model, or when the user asked for the best result
   rather than the cheapest.

5. Budget gate. `Budget: $spent/$max` follows every tool result. When a pick's
   `estimated_usd` exceeds 25% of the remaining budget, split the unit or drop a tier if
   the classifier allows it (never below the guarded floor); if neither is possible, tell
   the user before dispatching. When the user names a reviewer share ("at most N% for
   reviewing"), reviewers run on the small tier unless the diff touches guarded code.

6. Verify, then escalate up only. Accept a sub-agent's result through its acceptance
   check, never through its own summary; small-tier agents over-report success, so run
   the check yourself (`uv run pytest <impacted tests>`, `git diff --stat`, `grep -n`).
   On failure escalate one tier up with the failure evidence in the new prompt and the
   failed model in `exclude`; two tiers up when the failure is a reasoning failure (wrong
   approach, misunderstood spec) rather than a slip (typo, missed file). Never retry the
   same tier with a longer prompt or higher effort. Never escalate more than twice for
   the same unit; on the third failure stop and report.

7. Log every decision with `log_decision(unit, tier, model, reason, outcome)` when you
   dispatch, and again with the outcome once the check has run. The ledger
   (`~/.kiss/MODEL_DECISIONS.md`) is shared by every task and every row carries the
   task id, so it is what shows, across tasks, whether a tier fails too often for a
   kind of unit.

## Hard rules

- Never route to small or medium: security-sensitive code, credentials, payments,
  deletions, migrations, the final acceptance of the whole task, root-cause analysis
  after two failed fixes, or prose that will be sent on the user's behalf.
- Never downgrade a model the user named explicitly.
- Never spawn a sub-agent to run a shell command; run it inline (`run_commands_parallel`
  for many). A sub-agent costs a full model context.
- Do not invent prices or model names: use `model_menu`.

## Finishing

Finish with the task's result first, then a routing summary: every unit with its tier,
model, estimated and (when known) actual cost, outcome and escalations, and the path of
the ledger.
"""
"""The routing protocol; the operating manual for the orchestrating model."""


def _priced(
    model: str, note: str, runnable: set[str], tokens_in: int, tokens_out: int
) -> dict[str, Any]:
    """Return one candidate record: catalog prices, estimated cost and availability."""
    info = MODEL_INFO.get(model)
    if info is None:
        return {"model": model, "runnable": False, "note": f"{note}; not in the model catalog"}
    cost = (info.input_price_per_1M * tokens_in + info.output_price_per_1M * tokens_out) / 1e6
    return {
        "model": model,
        "input_per_1M": info.input_price_per_1M,
        "output_per_1M": info.output_price_per_1M,
        "estimated_usd": round(cost, 4),
        "runnable": model in runnable,
        "note": note,
    }


def _candidates(tier: str, tokens_in: int, tokens_out: int, exclude: str) -> list[dict[str, Any]]:
    """Return the priced candidates of *tier* in preference order, minus *exclude*."""
    excluded = {name.strip() for name in exclude.replace(",", " ").split()}
    runnable = set(get_available_models())
    return [
        _priced(model, note, runnable, tokens_in, tokens_out)
        for model, note in TIERS[tier]
        if model not in excluded
    ]


def model_menu(tokens_in: int = DEFAULT_TOKENS_IN, tokens_out: int = DEFAULT_TOKENS_OUT) -> str:
    """List every routing tier's candidate models with prices, cost estimate and availability.

    Args:
        tokens_in: Prompt tokens the unit is expected to consume (default 200k).
        tokens_out: Completion tokens the unit is expected to produce (default 20k).

    Returns:
        JSON: tier name -> ordered list of ``{model, input_per_1M, output_per_1M,
        estimated_usd, runnable, note}`` records; ``runnable`` is true when the
        model's provider credential is configured on this installation.
    """
    menu = {tier: _candidates(tier, tokens_in, tokens_out, "") for tier in TIER_NAMES}
    return json.dumps(menu, indent=2)


def pick_model(
    tier: str,
    tokens_in: int = DEFAULT_TOKENS_IN,
    tokens_out: int = DEFAULT_TOKENS_OUT,
    exclude: str = "",
) -> str:
    """Pick the cheapest runnable model of a routing tier.

    Args:
        tier: ``small``, ``medium`` or ``frontier``.
        tokens_in: Prompt tokens the unit is expected to consume (default 200k).
        tokens_out: Completion tokens the unit is expected to produce (default 20k).
        exclude: Models that already failed this unit, separated by commas or spaces.

    Returns:
        JSON ``{model, tier, input_per_1M, output_per_1M, estimated_usd, note}`` of
        the first candidate of the tier whose provider credential is configured, or
        an ``Error:`` line when the tier is unknown or has no runnable candidate.
    """
    if tier not in TIER_NAMES:
        return f"Error: unknown tier {tier!r}; use one of {', '.join(TIER_NAMES)}."
    usable = [c for c in _candidates(tier, tokens_in, tokens_out, exclude) if c["runnable"]]
    if not usable:
        return (
            f"Error: no runnable model in tier {tier!r} (excluded: {exclude or 'none'}); "
            "configure an API key for one of its candidates or route one tier up."
        )
    chosen = dict(usable[0], tier=tier)
    del chosen["runnable"]
    return json.dumps(chosen, indent=2)


def estimate_cost(
    model: str, tokens_in: int = DEFAULT_TOKENS_IN, tokens_out: int = DEFAULT_TOKENS_OUT
) -> str:
    """Estimate the USD cost of one run of a model from the local catalog prices.

    Args:
        model: A model name from the catalog (see ``model_menu``).
        tokens_in: Prompt tokens (default 200k).
        tokens_out: Completion tokens (default 20k).

    Returns:
        JSON ``{model, input_per_1M, output_per_1M, estimated_usd}``, or an
        ``Error:`` line when the model is not in the catalog.
    """
    record = _priced(model, "", set(), tokens_in, tokens_out)
    if "estimated_usd" not in record:
        return f"Error: unknown model {model!r}; it is not in the local model catalog."
    return json.dumps(
        {k: record[k] for k in ("model", "input_per_1M", "output_per_1M", "estimated_usd")}
    )


def ledger_path() -> Path:
    """Return the path of the shared routing ledger: ``<KISS home>/MODEL_DECISIONS.md``.

    The KISS home is ``$KISS_HOME`` when set, else ``~/.kiss``, the same
    directory as ``sorcar.db``, so the ledger outlives the task's work
    directory and worktree and every task appends to the same file.
    """
    return kiss_home() / LEDGER_NAME


def _current_task_id() -> str:
    """Return the persisted task id of the task calling this tool, or ``""`` outside a task.

    The id is the ``task_history`` row id in ``sorcar.db`` (the one
    ``/task_update <task_id>`` takes), read off the agent whose task thread
    is the calling thread (``None`` outside a registered task).
    """
    agent = current_agent()
    return agent.last_task_id if agent is not None else ""


def log_decision(unit: str, tier: str, model: str, reason: str, outcome: str = "pending") -> str:
    """Append one routing decision to the shared ledger ``~/.kiss/MODEL_DECISIONS.md``.

    Call it when a unit is dispatched (outcome ``pending``) and again once its
    acceptance check has run, with the outcome.  Every row carries the id of
    the task that wrote it (``-`` outside a task), so the rows of one run can
    be told apart from the rest of the installation's routing history and
    traced back to the task in ``sorcar.db``.

    Args:
        unit: Short description of the unit of work.
        tier: ``small``, ``medium`` or ``frontier``.
        model: The model the unit was routed to.
        reason: Why this tier: the classifier's answer and confidence, or the escalation.
        outcome: ``pending``, or what the acceptance check showed (default ``pending``).

    Returns:
        The path of the ledger the row was appended to, or an ``Error:`` line for an
        unknown tier.
    """
    if tier not in TIER_NAMES:
        return f"Error: unknown tier {tier!r}; use one of {', '.join(TIER_NAMES)}."
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(LEDGER_HEADER, encoding="utf-8")
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    cells = [
        " ".join(cell.replace("|", "/").split())
        for cell in (stamp, _current_task_id() or "-", unit, tier, model, reason, outcome)
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(cells) + " |\n")
    return f"logged to {path}"


def system_prompt() -> str:
    """Replace the default system prompt with the routing protocol."""
    return SYSTEM_PROMPT


def tools() -> list[Any]:
    """Expose the priced menu, the pick, the cost estimate and the ledger to the model."""
    return [model_menu, pick_model, estimate_cost, log_decision]


def is_parallel() -> bool:
    """Withhold ``run_parallel``: its workers would inherit this protocol as their prompt.

    ``run_parallel`` forwards the parent's custom system prompt to every
    worker, which would turn each routed unit into another router without
    the routing tools.  ``run_agent`` starts a fresh default session, so it
    is the dispatch primitive (one unit per call).
    """
    return False


def classify_tasks() -> bool:
    """Skip the lite/full prompt classifier: the protocol above is the whole prompt."""
    return False


def use_web_tools() -> bool:
    """No browser for the router itself; a routed sub-agent may still get one."""
    return False


def use_memory() -> bool:
    """No persistent memory: the shared ledger in the KISS home is the record."""
    return False
