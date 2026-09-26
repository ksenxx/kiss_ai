---
title: How Sorcar enforces budget, steps and sub-agent budget shares
uuid: 5ebe28ec-458a-4186-909b-7f49bd37de92
summary: 'How max_budget, max_steps and max_sub_sessions are enforced: _check_total_budget
  hook, usage ledger, _subagent_budget_share remaining/(n+1), partial results for
  sub-agents.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Budget and step enforcement

## Limits and defaults
| limit | scope | default | enforced by |
|---|---|---|---|
| `max_steps` | per sub-session | 10000 | `KISSAgent._run_agentic_loop` (raises `KISSError "exceeded N steps"`) |
| `max_sub_sessions` | per task | 10000 | `RelentlessAgent.perform_task` loop |
| `max_budget` (USD) | whole task, including sub-agents | `DEFAULT_MAX_BUDGET = 200.0` | executor `_check_limits` + `_check_total_budget` |

When the step limit is hit, that sub-session ends but the task goes on: the summarizer turns the
`KISSError` into a continuation (see `sorcar-relentless-continuation`). The step counter shown to
the model ("Steps: N/max") is per session.

## Budget checks
- Each executor gets `max_budget = remaining_budget` for its own session.
- `KISSAgent._check_limits` runs every step. It raises `BudgetExceededError` when the executor's own
  spend reaches its cap, then calls `budget_check_hook` = `RelentlessAgent._check_total_budget`. That
  hook adds the live executor's spend to `self.budget_used`, which holds the banked sessions plus
  spend attributed mid-session by sub-agents, `decide`, `talk` TTS and `run_agent`. So a parent whose
  children spent the budget stops at its next step.
- Spend is recorded in an append-only `_UsageLedger` (`_attribute_usage`, `_accumulate_usage`,
  `usage_snapshot`). `reset_usage` swaps the ledger atomically in `_reset`. The classifier's spend is
  folded in after the run (`_fold_classifier_usage`) with a keyed record, so a retry never counts it twice.

## Sub-agent budget share
`SorcarAgent._subagent_budget_share(n)` = `(max_budget - budget_used - live executor spend) / (n + 1)`.
The extra share is reserved for the parent, so even a one-child fan-out cannot drain it. With no
remaining budget it raises `BudgetExceededError`. Nested fan-outs split geometrically (each level
divides by n+1). **There is no minimum share any more**: commit ee6ba3d38 removed
`MIN_SUBAGENT_BUDGET` (it used to refuse shares under $0.50, per 2f2f1376b), `ReviewQuota` and
`MAX_REVIEW_ROUNDS`. Deep recursive fan-outs can therefore hand out slivers too small for a
child's first step.

## Exhaustion outcomes
- Top-level task: `BudgetExceededError` propagates and the UI/CLI report "budget exceeded".
- Sub-agent (has `_subagent_info`): `_budget_exhausted_result` returns
  `finish(success=False, is_continue=False)` with `_partial_result_html`, which quotes up to
  `PARTIAL_RESULT_MAX_STEPS = 8` recent steps, so the parent can reuse the work.

## Accounting of children
`run_tasks_parallel` sums each child's `(cost, tokens, steps)` into `totals_out`, and
`_run_tasks_parallel` credits it to the parent through `_attribute_sub_usage`.
`_LiveUsageMonitor` streams the aggregate to the UI while children run. Children abandoned after a stop
are tracked by `_AbandonedSubagent` and banked later by `reclaim_abandoned_subagents`, which is
called at the start of every fan-out.

## Sources
- `src/kiss/agents/sorcar/relentless_agent.py` (`_check_total_budget`, `_UsageLedger`, `_attribute_usage`, `_budget_exhausted_result`, `_partial_result_html`, `DEFAULT_MAX_BUDGET`)
- `src/kiss/agents/sorcar/sorcar_agent.py` (`_subagent_budget_share`, `_attribute_sub_usage`, `_LiveUsageMonitor`, `_AbandonedSubagent`, `reclaim_abandoned_subagents`, `_fold_classifier_usage`)
- `src/kiss/core/kiss_agent.py` (`_check_limits`, `_run_agentic_loop`)
