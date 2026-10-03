# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Autorouter agent — runs a task on the cheapest model tier that will finish it.

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

Three ways to run it::

    /autorouter add a --json flag to the export command and cover it with tests

    run_agent(agent="src/kiss/agents/seas/autorouter/autorouter_sea.py", task="...")

    pick ``autorouter`` in the model picker: every task of the tab runs
    through this SEA (see ``register_as_model()`` below)

The routing protocol (:data:`SYSTEM_PROMPT`) is added to the default Sorcar
system prompt through the ``add_to_system_prompt()`` getter; the
deterministic parts — the priced candidate menu, the pick, the cost estimate
and the decision ledger — are the tools this module exposes through
``tools()``.  Candidate order per tier is :data:`TIERS`, ranked by measured
coding quality per dollar (researched 2026-09-24); prices and availability
come from :mod:`kiss.core.models.model_info` at call time, so the menu is
always the one this installation can run.  The prompt's "Observed model
evidence" section is read from ``~/.kiss/AUTOROUTER.md`` (:func:`evidence_path`)
when this file loads: a dated table plus bullets on what this installation's
own task history shows about each model's cost, speed and reliability, at
most :data:`EVIDENCE_MAX_CHARS` characters of it (:func:`observed_evidence`
cuts a longer file at a line boundary).
:mod:`kiss.agents.seas.rsi7d.rsi7d_sea` rewrites that file from
``~/.kiss/history.db`` (so refreshing the evidence never edits this SEA),
refusing text over the same cap, and the protocol treats it as the
posterior over the tier-order prior.

Module-level getters (``add_to_system_prompt()``, ``register_as_model()``,
``model()``, ``is_parallel()``, ...) follow the SEA contract in
:mod:`kiss.server.agent_file`.  Picking ``autorouter`` in the model picker
also keeps the evidence fresh: the ``on_picked_as_model(work_dir)`` hook
(:func:`schedule_weekly_rsi7d`) makes sure an enabled weekly cron job that
runs ``/rsi7d autorouter`` exists, creating or resuming it when it does not.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kiss.core.brand import HOME_DIR
from kiss.core.config import kiss_home
from kiss.core.models.model_info import MODEL_INFO, get_available_models, get_default_model
from kiss.server.agent_state import current_agent

RSI7D_SEA_RELATIVE = "src/kiss/agents/seas/rsi7d/rsi7d_sea.py"
"""The rsi7d SEA's path inside a KISS checkout; the weekly job relays to it."""

RSI7D_JOB_NAME = "Weekly rsi7d: autorouter evidence and prompt (Sat 1am PT)"
"""Name of the weekly ``/rsi7d autorouter`` cron job; :func:`schedule_weekly_rsi7d`
recognises the job by this name alone, so a user may retune its budget or schedule."""

RSI7D_JOB_SCHEDULE = "0 1 * * 6"
"""Saturday 01:00 America/Los_Angeles (cron expressions are Pacific-evaluated), a day
before the hand-scheduled full ``/rsi7d all`` sweep so the two never share the checkout."""

RSI7D_JOB_MODEL = "claude-fable-5-1"
"""Model of the job and of the rsi7d run when runnable, else :func:`orchestrator_model`."""

RSI7D_JOB_BUDGET_USD = 25.0
"""Budget (USD) passed to the nested rsi7d run.

rsi7d's own ``max_budget()`` getter replaces a passed budget
(``apply_agent_overrides`` applies getters over the wire fields), so the
binding cap is the dollar sentence of :data:`RSI7D_TASK`; this value sizes
the job for a reader of the job list.
"""

RSI7D_JOB_RELAY_BUDGET_USD = 5.0
"""What the relay session that calls ``run_agent`` may spend on top of the rsi7d run."""

RSI7D_JOB_TIMEOUT_SECONDS = 2 * 3600
"""Timeout of the nested rsi7d run; the job gets ten more minutes for its relay."""

RSI7D_TASK = (
    "autorouter. Work in the current work dir; edit only the autorouter SEA through your "
    "tools. Mine the last 7 days of autorouter runs and of the models it dispatched to, rewrite "
    "the observed model evidence with write_autorouter_evidence, and patch the autorouter "
    "prompt only where its own runs show a repeatable failure. Spend at most $20 in total and "
    "at most $8 on replays; replay only tasks that changed nothing on disk, with run_agent. "
    "Delete tmp/rsi7d/replays before finishing. Write the report to "
    "./reports/rsi7d-autorouter-<date>.md and, when the work dir is a git checkout, git add it."
)
"""The ``/rsi7d`` task text of the weekly job: the scope ``autorouter`` first, then the
instructions (see ``rsi7d_sea.parse_scope``)."""

RSI7D_JOB_PROMPT = (
    "Call the run_agent tool IMMEDIATELY, as your very first action, with these arguments "
    "and no others:\n"
    "  agent        = {sea!r}\n"
    "  task         = {task!r}\n"
    "  timeout      = {timeout!r}\n"
    "  max_budget   = {max_budget!r}\n"
    "  model_name   = {model!r}\n"
    "  use_worktree = 'false'\n"
    "  auto_commit  = 'false'\n"
    "Do not explore any source code, do not paraphrase the task, and do not call any other "
    "tool first. When run_agent returns, relay its result (the report path, whether the "
    "autorouter evidence was rewritten and whether its prompt was patched) as your final "
    "summary."
)
"""Prompt of the weekly job: a ``run_agent`` directive to the rsi7d SEA.

A cron prompt job cannot be the literal text ``/rsi7d autorouter``: the
scheduler prepends its unattended-run preamble, so the text no longer
starts with the slash command and the SEA would not be dispatched.  The
relay calls ``run_agent`` on the rsi7d file next to this one instead, in
the job's own work directory (its worktree of the checkout when there is
one), so the child neither nests another worktree nor commits by itself.
"""

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
        ("gpt-6-sol", "OpenAI mid model, 400k context; cheapest measured per step"),
        ("claude-sonnet-5", "Anthropic mid model; default when Anthropic is required"),
        ("claude-opus-5-5", "Anthropic frontier; cheapest, fastest frontier per step measured"),
        (
            "kimi-k3",
            "Moonshot, 1M context. Best cheaper alternative to claude-fable-5-1. Go-to model "
            "for security analysis and hardening.",
        ),
    ),
    "frontier": (
        ("gpt-6-astra", "OpenAI frontier; priciest per step measured, reliable reviewer"),
        ("claude-fable-5-1", "Anthropic top model; longest, hardest tasks only; measured stalls"),
    ),
}
"""Ordered ``(model, note)`` candidates per tier; the first runnable one wins.

The order is by measured coding quality per dollar as of 2026-09-24; the
notes carry what the 7-day task history measured (see the "Observed model
evidence" block of :data:`SYSTEM_PROMPT`, refreshed by ``/rsi7d``).  Edit
the order when models or prices change; prices themselves are read from
the catalog, never stored here.
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

CELL_MAX_CHARS = 120
"""Longest ``unit``, ``reason`` or ``outcome`` cell :func:`log_decision` writes; a longer
one is cut and ends with ``...`` so a ledger row stays one terse line."""

EVIDENCE_NAME = "AUTOROUTER.md"
"""File name, inside the KISS home directory, of the observed model evidence
``/rsi7d`` refreshes: a stamp line, a dated table and bullets."""

EVIDENCE_MAX_CHARS = 2500
"""Most characters of the evidence file that reach the prompt (about 700 tokens).
``/rsi7d``'s ``write_autorouter_evidence`` refuses a longer text; :func:`observed_evidence`
cuts a longer file (hand-edited, or merged across machines) at a line boundary."""

EVIDENCE_CUT = (
    f"_[evidence cut at {EVIDENCE_MAX_CHARS} characters; `/rsi7d` rewrites the file within "
    "that size]_"
)
"""Line appended in place of the part of an over-long evidence file the prompt drops."""

NO_EVIDENCE = (
    f"_No observed evidence yet: `~/{HOME_DIR}/AUTOROUTER.md` is missing or empty. Route on the "
    "tier order alone until `/rsi7d all` has measured this installation's task history._"
)
"""What the prompt says in place of the evidence when the file is missing or empty."""


def evidence_path() -> Path:
    """Return the path of the observed model evidence: ``<KISS home>/AUTOROUTER.md``.

    The KISS home is ``$KISS_HOME`` when set, else ``~/.kiss`` (the directory
    of ``history.db`` and the ledger), so the evidence ``/rsi7d`` measures from
    the task history lives next to that history and travels with it.
    """
    return kiss_home() / EVIDENCE_NAME


def observed_evidence() -> str:
    """Return the observed-model-evidence Markdown spliced into :data:`SYSTEM_PROMPT`.

    The content of :func:`evidence_path` with surrounding blank lines removed,
    or :data:`NO_EVIDENCE` when the file is missing, unreadable or blank.  A
    file over :data:`EVIDENCE_MAX_CHARS` characters is cut at the last line
    break within the cap (mid-line only when its first line alone exceeds
    the cap) and ends with :data:`EVIDENCE_CUT`, so a hand-edited or
    machine-merged file cannot inflate every prompt.  Evaluated when this
    module loads, which the daemon does for every task the SEA runs, so a
    refreshed file reaches the next task's prompt.
    """
    try:
        text = evidence_path().read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    if len(text) > EVIDENCE_MAX_CHARS:
        head = text[: EVIDENCE_MAX_CHARS + 1]  # a break right at the cap keeps its line
        kept = head.rpartition("\n")[0] or head[:EVIDENCE_MAX_CHARS]
        text = f"{kept.rstrip()}\n\n{EVIDENCE_CUT}"
    return text or NO_EVIDENCE


SYSTEM_PROMPT = f"""\
## Model routing protocol (autorouter)

You are the autorouter. Finish the task at the lowest cost per accepted task: route each
unit of work to the cheapest model tier (small, medium, frontier) that completes it
correctly the first time; escalate only on a verified failure. Models, prices and
availability come from `model_menu`, `pick_model` and `estimate_cost`; never invent either.

## Protocol

1. Split the task into units that each have a mechanical acceptance check (a test command,
   `git diff --stat` on the expected files, a grep for the new symbol). Work without such
   a check stays in your own loop. Never route a single tool call.

2. Classify each unit with `decide` (`state`: description, check, size, earlier failures).
   Ask `tier` (type `choice`; small = mechanical work with an unambiguous spec: read, grep,
   summarize, run tests, rename, format, boilerplate; medium = clear-spec engineering:
   implement from a description, tests for existing code, known-cause bug, one-module
   refactor, small diff review; frontier = open-ended reasoning: root cause, cross-module
   design, security or concurrency review, final acceptance) and `guarded` (type `noul`:
   touches auth, secrets, payments, deletion, migrations, anything the user cannot undo,
   or decides whether the whole task is complete). `guarded` >= 0.5 is frontier; `tier`
   confidence below 0.6 moves one tier up; a unit that failed on a tier starts one tier
   higher with the failed model in `exclude`. Without `decide`, judge by the same criteria.

3. Pick with `pick_model(tier, tokens_in, tokens_out, exclude)`; a sub-agent that reads a
   medium codebase and runs tests uses about 200k prompt and 20k completion tokens. Call
   `observed_call_costs(days, model)` once per candidate you will dispatch, never for work
   kept inline, and pass over a candidate whose observed mean cost per call is over twice
   the catalog estimate or far slower than its tier peers. Dispatch with `run_agent(task=..., model_name=<picked>)`, one call per unit,
   never mid-context; the task text names the files the sub-agent may touch and the check
   that ends it. Units run in sequence; a sub-agent may fan out with its own `run_parallel`.

4. Work kept in your own loop: plan and final acceptance on the frontier model, execution
   on medium, volume work (exploration, test runs, log reading) on small sub-agents.
   Caches are per model, so `set_model` only at a phase boundary: to the medium pick once
   the plan is written, back to the original model after two failed checks or for the
   final verification. No downgrade for a short task (under about 10 tool calls), a
   user-pinned model, or a request for the best result.

5. Budget: `Budget: $spent/$max` follows every tool result. A pick whose `estimated_usd`
   exceeds 25% of the remaining budget is split, or dropped one tier when the classifier
   allows it (never below the guarded floor); if neither is possible, tell the user
   first. A user-named reviewer share
   ("at most N% for reviewing") runs reviewers on small unless the diff is guarded.

6. Verify with the acceptance check yourself, never with the sub-agent's summary. On
   failure escalate one tier up with the failure evidence in the new prompt and the failed
   model in `exclude`, two tiers for a reasoning failure (wrong approach, misunderstood
   spec). Never retry the same tier; never escalate more than twice per unit; on the third
   failure stop and report.

7. Log each dispatch with `log_decision(unit, tier, model, reason, outcome)` and log again
   once the check has run. One short clause per cell (cut at {CELL_MAX_CHARS} characters);
   the ledger `~/.kiss/MODEL_DECISIONS.md` is shared by every task.

## Observed model evidence

From this installation's task history (`~/.kiss/AUTOROUTER.md`, rewritten by `/rsi7d`); the
tier order is the prior, this is the posterior. A model with a high observed failure share
for the role goes into `exclude` even when `pick_model` ranks it first; among a tier's
runnable models prefer the lower observed $ and seconds per step when at least 10 tasks
back it. Prices still come from `model_menu`.

{observed_evidence()}

## Hard rules

- Never route to small or medium: security-sensitive code, credentials, payments,
  deletions, migrations, the final acceptance of the whole task, root-cause analysis after
  two failed fixes, or prose sent on the user's behalf.
- Never downgrade a model the user named explicitly.
- Never spawn a sub-agent to run a shell command; run it inline (`run_commands_parallel`
  for many).

## Finishing

Task result first, then a routing summary: each unit's tier, model, estimated and actual
cost, outcome and escalations, and the ledger path.
""" """\


## Lessons from recent runs (rsi7d)

- `rg` is absent on this host and `/bin/sh` does no brace expansion: search with
  `grep -R -n -E <pattern> <dir> --include='*.py'` from the first call.
- Work kept inline: no `estimate_cost`; report it as inline on <model>, no dispatch.
"""
"""The routing protocol; the operating manual for the orchestrating model."""


def description() -> str:
    """Return the one-sentence help text shown by ``/autorouter help``."""
    return (
        "Splits a task into units, runs each on the cheapest model tier (small, medium, "
        "frontier) that passes its acceptance check, escalating on failure and logging every "
        f"decision to ~/{HOME_DIR}/MODEL_DECISIONS.md; pick `autorouter` in the model picker, use "
        '`/autorouter <task>` in the chat, or run_agent(agent="autorouter", task="...").'
    )


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


def observed_call_costs(days: int = 7, model: str = "") -> str:
    """Aggregate the per-call ``llm_call`` events of recent tasks by model.

    Every model call an agent makes is recorded in ``history.db`` as an
    ``llm_call`` event with the call's own tokens, USD cost and duration
    (``KissAgent._print_llm_call``).  This reads the events of the tasks
    launched in the last *days* days and reports, per model, what a call
    actually cost on this installation — the observed price to check
    ``estimate_cost``'s catalog estimate against before dispatching.

    Args:
        days: Look-back window in days over task launch times (default 7).
        model: Restrict the report to one model name; empty for every model.

    Returns:
        JSON list ordered by total cost, one record per model:
        ``{model, calls, tasks, total_usd, mean_usd_per_call,
        median_usd_per_call, mean_input_tokens, mean_output_tokens,
        cache_read_share, mean_seconds}``; an empty list when no calls
        were recorded in the window.
    """
    db_path = kiss_home() / "history.db"
    if not db_path.exists():
        return "[]"
    cutoff = time.time() - max(1, int(days)) * 86400
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
    try:
        task_ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM task_history WHERE timestamp >= ? "
                "ORDER BY timestamp DESC LIMIT 5000",
                (cutoff,),
            )
        ]
        per_model: dict[str, dict[str, Any]] = {}
        for start in range(0, len(task_ids), 500):
            chunk = task_ids[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                "SELECT task_id, event_json FROM events "
                f"WHERE task_id IN ({placeholders}) "
                "AND event_json LIKE '%\"type\": \"llm_call\"%'",
                chunk,
            )
            for task_id, event_json in rows:
                event = json.loads(event_json)
                name = str(event.get("model") or "")
                if event.get("type") != "llm_call" or not name or (model and name != model):
                    continue
                agg = per_model.setdefault(
                    name,
                    {"costs": [], "tasks": set(), "input": 0, "output": 0,
                     "cache_read": 0, "ms": 0},
                )
                agg["costs"].append(float(event.get("cost") or 0.0))
                agg["tasks"].add(task_id)
                # Prompt tokens = uncached input + cache writes (Anthropic
                # reports cache-creation tokens apart from input) + reads.
                agg["input"] += int(event.get("input_tokens") or 0) + int(
                    event.get("cache_write") or 0
                )
                agg["output"] += int(event.get("output_tokens") or 0)
                agg["cache_read"] += int(event.get("cache_read") or 0)
                agg["ms"] += int(event.get("duration_ms") or 0)
    finally:
        conn.close()
    report = []
    for name, agg in per_model.items():
        calls = len(agg["costs"])
        prompt_tokens = agg["input"] + agg["cache_read"]
        report.append(
            {
                "model": name,
                "calls": calls,
                "tasks": len(agg["tasks"]),
                "total_usd": round(sum(agg["costs"]), 4),
                "mean_usd_per_call": round(sum(agg["costs"]) / calls, 6),
                "median_usd_per_call": round(statistics.median(agg["costs"]), 6),
                "mean_input_tokens": round(prompt_tokens / calls),
                "mean_output_tokens": round(agg["output"] / calls),
                "cache_read_share": round(agg["cache_read"] / prompt_tokens, 3)
                if prompt_tokens
                else 0.0,
                "mean_seconds": round(agg["ms"] / calls / 1000, 1),
            }
        )
    report.sort(key=lambda r: r["total_usd"], reverse=True)
    return json.dumps(report, indent=2)


def ledger_path() -> Path:
    """Return the path of the shared routing ledger: ``<KISS home>/MODEL_DECISIONS.md``.

    The KISS home is ``$KISS_HOME`` when set, else ``~/.kiss``, the same
    directory as ``history.db``, so the ledger outlives the task's work
    directory and worktree and every task appends to the same file.
    """
    return kiss_home() / LEDGER_NAME


def _current_task_id() -> str:
    """Return the persisted task id of the task calling this tool, or ``""`` outside a task.

    The id is the ``task_history`` row id in ``history.db`` (the one
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
    traced back to the task in ``history.db``.  A row is one terse line: the
    ``unit``, ``reason`` and ``outcome`` cells are collapsed to single-spaced
    text and cut at :data:`CELL_MAX_CHARS` characters (ending in ``...``).

    Args:
        unit: The unit of work, in a few words.
        tier: ``small``, ``medium`` or ``frontier``.
        model: The model the unit was routed to.
        reason: Why this tier, in one clause: the classifier's answer and confidence,
            or the escalation.
        outcome: ``pending``, or what the acceptance check showed, in one clause
            (default ``pending``).

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
        _terse(cell)
        for cell in (stamp, _current_task_id() or "-", unit, tier, model, reason, outcome)
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(cells) + " |\n")
    return f"logged to {path}"


def _terse(cell: str) -> str:
    """Return *cell* as one table cell: single-spaced, ``|`` replaced, cut at the cell cap."""
    text = " ".join(cell.replace("|", "/").split())
    if len(text) > CELL_MAX_CHARS:
        text = text[: CELL_MAX_CHARS - 3].rstrip() + "..."
    return text


def orchestrator_model() -> str:
    """Return the model the routing agent itself runs on.

    The protocol plans, verifies and accepts on the frontier tier, so this is
    the first runnable candidate of ``TIERS["frontier"]``.  When none is
    runnable it is the best runnable function-calling model in the picker's
    order (the picker offers ``autorouter`` exactly when that list is
    non-empty, so an offered entry always has a runnable orchestrator), and
    on a keyless install the default model, as any run would get.

    Returns:
        A model name from the catalog.
    """
    from kiss.server.autocomplete import ranked_function_calling_models

    runnable = set(get_available_models())
    for name, _note in TIERS["frontier"]:
        if name in runnable:
            return name
    ranked = ranked_function_calling_models()
    return ranked[0] if ranked else get_default_model()


def register_as_model() -> bool:
    """List ``autorouter`` in the model picker.

    A picked ``autorouter`` makes the daemon run every task of the tab
    through this SEA on :func:`orchestrator_model` (``model()`` below), with
    :data:`SYSTEM_PROMPT` added to the system prompt (``add_to_system_prompt()``).
    """
    return True


def kiss_checkout(work_dir: str) -> str:
    """Return the KISS git checkout that contains *work_dir*, or ``""``.

    A checkout is the nearest directory upward from *work_dir* that holds
    both ``.git`` (a directory, or the file of a git worktree) and the
    bundled SEAs (:data:`RSI7D_SEA_RELATIVE`); the weekly job runs there so
    rsi7d edits and commits the checkout's SEAs instead of the installed
    copy the daemon runs from.

    Args:
        work_dir: Work directory of the run in which ``autorouter`` was picked.

    Returns:
        The checkout's absolute path, or ``""`` when *work_dir* is not
        inside a KISS checkout.
    """
    if not work_dir:
        return ""
    start = Path(work_dir).resolve()
    for base in (start, *start.parents):
        if not (base / ".git").exists():
            continue
        checkout = _owning_checkout(base)
        if (checkout / RSI7D_SEA_RELATIVE).is_file():
            return str(checkout)
    return ""


def _owning_checkout(base: Path) -> Path:
    """Return the checkout a linked git worktree at *base* belongs to (*base* itself otherwise).

    A task worktree is discarded when its task ends, so a weekly job must
    point at the durable checkout: the ``.git`` *file* of a linked worktree
    reads ``gitdir: <checkout>/.git/worktrees/<name>``.
    """
    git = base / ".git"
    if git.is_dir():
        return base
    gitdir = git.read_text(encoding="utf-8").strip().removeprefix("gitdir:").strip()
    checkout, linked, _ = gitdir.partition("/.git/worktrees/")
    return (base / checkout).resolve() if linked else base  # gitdir may be relative to base


def weekly_rsi7d_job(work_dir: str) -> dict[str, Any]:
    """Return the ``cron_job("create", ...)`` arguments of the weekly ``/rsi7d autorouter`` job.

    Inside a KISS checkout (:func:`kiss_checkout`) the job runs in a
    worktree of that checkout, auto-commits, and relays to the checkout's
    own rsi7d file; elsewhere it runs in a scratch directory and relays to
    the rsi7d file installed next to this one, which still rewrites the
    observed model evidence in ``$KISS_HOME/AUTOROUTER.md``.

    Args:
        work_dir: Work directory of the run in which ``autorouter`` was picked.

    Returns:
        Keyword arguments for ``cron_job("create", **job)``.
    """
    checkout = kiss_checkout(work_dir)
    rsi7d_sea = (
        RSI7D_SEA_RELATIVE
        if checkout
        else str(Path(__file__).resolve().parents[1] / "rsi7d" / "rsi7d_sea.py")
    )
    model_name = (
        RSI7D_JOB_MODEL if RSI7D_JOB_MODEL in get_available_models() else orchestrator_model()
    )
    prompt = RSI7D_JOB_PROMPT.format(
        sea=rsi7d_sea,
        task=RSI7D_TASK,
        timeout=str(RSI7D_JOB_TIMEOUT_SECONDS),
        max_budget=str(RSI7D_JOB_BUDGET_USD),
        model=model_name,
    )
    return {
        "name": RSI7D_JOB_NAME,
        "schedule": RSI7D_JOB_SCHEDULE,
        "prompt": prompt,
        "model_name": model_name,
        "max_budget": str(RSI7D_JOB_BUDGET_USD + RSI7D_JOB_RELAY_BUDGET_USD),
        "timeout": str(RSI7D_JOB_TIMEOUT_SECONDS + 600),
        "work_dir": checkout,
        "use_worktree": bool(checkout),
        "auto_commit": bool(checkout),
    }


def schedule_weekly_rsi7d(work_dir: str) -> str:
    """Make sure an enabled weekly ``/rsi7d autorouter`` cron job exists.

    ``cron_job("ensure")`` looks the job up by :data:`RSI7D_JOB_NAME` in the
    cron store (``$KISS_HOME/cron/jobs.json``) under the store's lock: an
    enabled one is left as it is (so a retuned schedule or budget
    survives), a paused one is resumed, and when there is none the job
    of :func:`weekly_rsi7d_job` is created.

    Args:
        work_dir: Work directory of the run in which ``autorouter`` was picked.

    Returns:
        The cron tool's reply (YAML): ``exists``, ``resumed`` or ``created``
        with the job, or ``error``.
    """
    from kiss.agents.sorcar.cron_agent import cron_job

    return cron_job("ensure", **weekly_rsi7d_job(work_dir))


def on_picked_as_model(work_dir: str) -> str:
    """Schedule the weekly ``/rsi7d autorouter`` job when ``autorouter`` is picked as the model.

    The daemon runs this hook when ``autorouter`` is picked in the model
    picker and once per run whose model is ``autorouter``
    (``sea_commands.run_picked_hook``), and logs the returned note.

    Args:
        work_dir: Work directory of the run.

    Returns:
        :func:`schedule_weekly_rsi7d`'s note.
    """
    return schedule_weekly_rsi7d(work_dir)


def add_to_system_prompt() -> str:
    """Add the routing protocol to the default Sorcar system prompt."""
    return SYSTEM_PROMPT


def model() -> str:
    """Run the router itself on the frontier orchestrator model."""
    return orchestrator_model()


def tools() -> list[Any]:
    """Expose the priced menu, the pick, the cost estimate, the observed costs and the ledger."""
    return [model_menu, pick_model, estimate_cost, observed_call_costs, log_decision]


def is_parallel() -> bool:
    """Withhold ``run_parallel``: its workers would inherit this protocol in their prompt.

    ``run_parallel`` forwards the parent's system-prompt additions to every
    worker, which would turn each routed unit into another router without
    the routing tools.  ``run_agent`` starts a fresh default session, so it
    is the dispatch primitive (one unit per call).
    """
    return False


def classify_tasks() -> bool:
    """Skip the lite/full prompt classifier: the router always gets the full prompt."""
    return False


def use_web_tools() -> bool:
    """No browser for the router itself; a routed sub-agent may still get one."""
    return False


def use_memory() -> bool:
    """No persistent memory: the shared ledger in the KISS home is the record."""
    return False
