---
title: 'Autoroute SEA: routing task units to the cheapest model tier (autoroute_sea.py,
  MODEL_DECISIONS.md)'
uuid: 2d0112c8-07ca-464d-b17f-6582fe6e2909
summary: '/autoroute SEA: splits tasks into checkable units, decide-based small/medium/frontier
  tiers, cheapest runnable model from TIERS, escalation, MODEL_DECISIONS.md ledger;
  empty ROUTING.md.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Autoroute SEA (cost-based model routing)

`src/kiss/agents/seas/autoroute_sea.py` is a SEA (agent file): run it as the `/autoroute <task>`
slash command or `run_agent(agent="src/kiss/agents/seas/autoroute_sea.py", task=...)`. Goal:
minimum **cost per accepted task**, not per token (frontier models cost 40-100x small models per
token, and most agent tokens are exploration and mechanical work). It replaced the older
`model-cost-router` skill (commit "convert model-cost-router skill into autoroute_sea SEA").

## Tiers (`TIERS`, ordered by coding quality per dollar, researched 2026-09-24)
- small: `zai-org/GLM-5.3-Flash`, `openrouter/z-ai/glm-5.3-flash`, `gpt-6-luna`,
  `openrouter/openai/gpt-6-luna`, `deepseek-ai/DeepSeek-V4.1-Flash`, `openrouter/qwen/qwen3.8-flash`,
  `gpt-5.6-luna`, `claude-haiku-4-5`.
- medium: `gemini-3.8-flash`, `openrouter/google/gemini-3.8-flash`, `zai-org/GLM-5.3`,
  `openrouter/z-ai/glm-5.3`, `deepseek-ai/DeepSeek-V4-Pro-0813`, `openrouter/x-ai/grok-4.7`,
  `gpt-6-sol`, `claude-sonnet-5`, `kimi-k3`.
- frontier: `claude-opus-5-5`, `gpt-6-astra`, `claude-fable-5-1`.
Only the order and notes live here; prices come from `MODEL_INFO` at call time. Edit `TIERS`
when models change (e.g. `claude-opus-5` was removed from frontier when deprecated).

## Tools exposed (`tools()`)
- `model_menu(tokens_in=200_000, tokens_out=20_000)`: JSON per tier of `{model, input_per_1M,
  output_per_1M, estimated_usd, runnable, note}`. `runnable` = in `get_available_models()`
  (provider credential configured). A name missing from the catalog is shown as not runnable.
- `pick_model(tier, tokens_in, tokens_out, exclude)`: first runnable candidate not in `exclude`
  (comma/space separated), or an `Error:` line.
- `estimate_cost(model, tokens_in, tokens_out)`: `(in_price*in + out_price*out)/1e6` from the
  catalog; ignores cache discounts and long-context uplift (a planning estimate, unlike
  `calculate_cost`).
- `log_decision(unit, tier, model, reason, outcome="pending")`: appends a row
  `| time (UTC) | task_id | unit | tier | model | reason | outcome |` to
  `kiss_home()/MODEL_DECISIONS.md` (`$KISS_HOME` or `~/.kiss`), shared by all tasks; task id from
  `current_agent().last_task_id`; `|` in cells replaced by `/`.

## Protocol (`SYSTEM_PROMPT`, replaces the default Sorcar prompt)
1. Split into units with a mechanical acceptance check (test command, diff stat, grep).
2. Classify each with `decide` (see `models-decisions-jev-model`): `tier` choice + `guarded` noul
   (auth, secrets, payments, deletion, migrations, final acceptance). `guarded >= 0.5` -> frontier;
   tier confidence < 0.6 -> one tier up; ties go up.
3. `pick_model`, then dispatch one unit per `run_agent(task=..., model_name=...)`.
4. Own-loop phases: plan on frontier, `set_model` to medium for execution, back for root cause and
   final verification; switch only at phase boundaries because prompt caches are per model.
5. Budget gate: a pick above 25% of remaining budget must be split or confirmed.
6. Verify via the check, not the sub-agent's summary; escalate one tier (two for reasoning
   failures), at most twice per unit.
7. Log every dispatch and outcome.

## SEA getters
`is_parallel()` False (a `run_parallel` worker would inherit the router prompt without its tools),
`classify_tasks()` False, `use_web_tools()` False, `use_memory()` False (the ledger is the record).

## ROUTING.md
The repo-root `ROUTING.md` is a 0-byte placeholder added in commit 3e706dca5 ("docs(speed-audit)
..."). It holds no routing rules; model-name-to-provider routing is code in
`kiss.core.models.model_info.model()` (see `models-provider-resolution`).

## Sources
- `src/kiss/agents/seas/autoroute_sea.py` (`TIERS`, `TIER_NAMES`, `model_menu`, `pick_model`, `estimate_cost`, `log_decision`, `ledger_path`, `SYSTEM_PROMPT`, `is_parallel`)
- `src/kiss/agents/seas/__init__.py`
- `ROUTING.md`; `git log -- ROUTING.md src/kiss/agents/seas/autoroute_sea.py`
