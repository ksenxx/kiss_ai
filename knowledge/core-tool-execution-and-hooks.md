---
title: Tool execution, tool_call_hook, tool_call_guard and other KISSAgent hooks
uuid: 418e4613-408a-4913-b158-d05422062a68
summary: 'KISSAgent tool execution and hooks: tool_call_hook (OK or replacement),
  tool_call_guard, pre_step_hook, llm_call_hook, budget_check_hook, context_reset_hook,
  and who sets them.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Tool execution and KISSAgent hooks

## Per-tool-call pipeline in `_execute_step`
For each function call returned by the model:
1. `tool_calls_made += 1` unless the name is `finish`. Blocked calls count too. `RelentlessAgent` reads
   this counter after a session; a session that only called `finish` made no progress.
2. `tool_call_hook(name, args)` comes from `run(tool_call_hook=...)`. Any return value other than the exact
   string `"OK"` becomes the tool's result, and the tool does not run.
3. If the hook returned `"OK"` (or is unset), `tool_call_guard(name, args)` runs. It returns `None` to allow
   the call or a string to block it. The hook's "OK" does **not** override the guard.
4. A call that nothing blocked and that `is_long_running_call` flags runs through
   `_execute_tool_keeping_cache_warm` (see `models-prompt-caching`). All other calls go through `_execute_tool(fc, blocked)`.
5. A blocked call prints a `tool_call` event and then an error `tool_result`, and it returns the block message.

## `_execute_tool`
- Creates `new_tool_call(name)` before anything can interrupt the call. It registers the token for
  per-panel Stop (except `finish`, which cannot be interrupted) and emits `tool_call` with `call_id`.
- Tool exceptions become error text for the model. `BudgetExceededError` is re-raised.
- `ToolCallInterrupted` becomes the result `"User interrupted the tool call."`. If the interrupt landed while a
  `KeyboardInterrupt` or `BudgetExceededError` was already unwinding (`exc.__context__`), that original
  exception is re-raised so a task Stop is not downgraded. See `core-tool-call-interrupt`.
- `finally` sets `token.closing = True` first and then unregisters the token.

## Other hooks (plain attributes on `KISSAgent`)
| Attribute | Called | Set by |
|---|---|---|
| `pre_step_hook(model)` | start of every step | `SorcarAgent` (`_drain_pending_user_messages`); forwarded to executors by `RelentlessAgent` |
| `tool_call_guard(name, args)` | per tool call, and for implicit finish | `SorcarAgent` (`_block_finish_when_user_message_pending`) |
| `llm_call_hook(msgs) -> msgs` | before each model call on new conversation messages | `run(llm_call_hook=...)` |
| `budget_check_hook()` | inside `_check_limits`, may raise | `RelentlessAgent` (`_check_total_budget`) |
| `context_reset_hook()` | after a compaction was applied | `SorcarAgent` (`useful_tools.forget_reads`, which resets the Read dedupe) |

`run()` resets `llm_call_hook` and `tool_call_hook` from its arguments on every call. The other hooks
persist across runs of the same instance.

## Sources
- `src/kiss/core/kiss_agent.py` (`_execute_step`, `_execute_tool`, `_implicit_finish_allowed`, `KISSAgent.__init__`)
- `src/kiss/agents/sorcar/sorcar_agent.py` (hook assignments)
- `src/kiss/agents/sorcar/relentless_agent.py` (executor hook forwarding)
