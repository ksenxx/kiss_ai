---
title: KISSAgent budget and token accounting (_UsageTotals snapshot)
uuid: dc7585fd-15a6-47c6-9970-61d18e591592
summary: KISSAgent token and cost accounting per response (provider cost over catalog
  price), atomic _UsageTotals snapshot of budget/tokens/steps, max_budget and BudgetExceededError.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# KISSAgent budget and token accounting

## Per-response update (`_update_tokens_and_budget_from_response`)
- `model.extract_input_output_token_counts_from_response(response)` returns a tuple of length 4
  `(input, output, cache_read, cache_write)`, 5 (`+ cache_write_1h`) or 7 (`+ audio_input, audio_output`).
- `call_tokens` is the sum of all fields. If it is greater than 0 it becomes `context_tokens_used`, the size
  of the last request, which drives compaction and hand-off. `last_cache_read_tokens` is also stored.
- **Cost**: `model.extract_cost_from_response(response)` (for example OpenRouter's `usage.cost`) is used
  when it returns a number. Otherwise the price comes from `kiss.core.models.model_info.calculate_cost(model_name, ...)`,
  which uses catalog prices (`MODEL_INFO`) including cache and audio rates.
- A `KISSError` from accounting propagates. Any other exception is logged and swallowed.

## Atomic usage snapshot
`budget_used`, `total_tokens_used` and `step_count` are **properties** over one immutable
`_UsageTotals(budget_used, total_tokens_used, step_count)` NamedTuple. One response is published with a
single attribute store, so an asynchronously injected stop (the server's `PyThreadState_SetAsyncExc` watchdog)
cannot leave tokens without their cost. The parent `RelentlessAgent`'s `except BaseException` recovery
bank and Sorcar's live usage monitor read `agent.usage_snapshot()`, which returns one coherent triple.
Only the agent's own thread writes. Keep-alive ping responses are accounted on the agent thread after
the tool returns.

Test: `src/kiss/tests/core/test_conc2026_kiss_agent_usage_atomicity.py`.

## Limits
- `max_budget` defaults to 10.0 USD and `max_steps` to 10000 (set in `_reset`).
- `_check_limits()` raises `BudgetExceededError` when `budget_used >= max_budget`, then calls
  `budget_check_hook()` (RelentlessAgent's total-budget check across sessions), then checks the context
  hand-off (see `core-limits-and-context-handoff`).
- `_execute_tool` re-raises `BudgetExceededError` from a tool (a sub-agent tool running out, for example)
  instead of turning it into a tool error string.
- Non-agentic and CLI runs account through `_generate_once`. When `generate()` fails mid-stream,
  `model.take_partial_usage_response()` is billed before the error propagates.

## Where users see it
- The usage string is appended to the model's next user or tool message and printed as a
  `usage_info` event (with `cache_read` and `model`).
- The `result` event carries `step_count`, `total_tokens` and `cost`.

## Sources
- `src/kiss/core/kiss_agent.py` (`_UsageTotals`, `usage_snapshot`, `_update_tokens_and_budget_from_response`, `_check_limits`, `_get_usage_info_string`, `_generate_once`)
- `src/kiss/core/models/model_info.py` (`calculate_cost`)
