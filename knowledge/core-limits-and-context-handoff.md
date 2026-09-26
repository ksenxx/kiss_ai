---
title: Step, budget and context-window limits; ContextWindowExceededError hand-off
  at 70 percent
uuid: 4e49af66-3b45-41f3-b455-0efeee320592
summary: Where KISSAgent enforces max_steps, max_budget and the context limit (CONTEXT_LIMIT_FRACTION
  0.7, KISS_CONTEXT_LIMIT_FRACTION); provider overflow maps to ContextWindowExceededError.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Limits and context-window hand-off

## Step limit
It is owned only by `_run_agentic_loop` (`while step_count < max_steps`). Running out raises
`KISSError("Agent <name> exceeded N steps.")`. `_check_limits` deliberately does not check steps,
because two places used to give the same condition two differently worded errors.

## Context limit
- `CONTEXT_LIMIT_FRACTION` = `DEFAULT_CONFIG.context_limit_fraction` (env `KISS_CONTEXT_LIMIT_FRACTION`,
  default 0.7). A value outside (0, 1] silently becomes 0.7.
- `_handoff_context_tokens()` = fraction * `get_max_context_length(model_name)`. It is `None` when the
  model's window is unknown, and then the check is skipped.
- `_check_limits` raises `ContextWindowExceededError` when `context_tokens_used >= handoff_tokens`.
  The message looks like `conversation reached 350,123 of 500,000 context tokens (limit 70%)`.
- Why 70%: every step re-sends the whole context, so steps near a full window cost several times more
  than early ones. `RelentlessAgent` catches the error and continues in a fresh session with a trajectory
  summary (the Sorcar area covers this).
- Provider-side overflow: an exception whose message contains one of `exceeds the context window`,
  `prompt is too long`, `context_length_exceeded`, `maximum context length` or `exceeds the maximum number of tokens`
  is re-raised as `ContextWindowExceededError`, both in the loop and in `_generate_once`.

## When `_check_limits` runs
1. At the top of each loop iteration, after `step_count += 1`.
2. After the model call, if any call in the turn is not `finish`, so accounting from the response itself can stop the tools.
3. After each tool that is not an unblocked `finish`. The error is **held** (`limit_error`) and the
   loop over the remaining calls breaks. The step's model and user messages are appended to `messages`
   first and the error is raised after that, so a partial result built from the trajectory includes the
   tool that really ran. Calls after the failing one in the same turn are not executed.

## Compaction interplay
Compaction stops once context reaches 80% of the hand-off limit (`HANDOFF_PROXIMITY_FRACTION`); see `core-context-compaction`.

## Sources
- `src/kiss/core/kiss_agent.py` (`CONTEXT_LIMIT_FRACTION`, `_CONTEXT_OVERFLOW_PHRASES`, `_is_context_overflow_error`, `_check_limits`, `_handoff_context_tokens`, `_run_agentic_loop`, `_execute_step`)
- `src/kiss/core/kiss_error.py` (`ContextWindowExceededError`)
- `src/kiss/core/config.py` (`context_limit_fraction`)
