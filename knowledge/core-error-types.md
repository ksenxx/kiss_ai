---
title: KISS error types (KISSError, BudgetExceededError, ContextWindowExceededError,
  ModelRefusalError)
uuid: 10272b46-50e4-423d-b01e-8fe36a7e6ad8
summary: 'kiss_error.py hierarchy: KISSError(ValueError) with code, BudgetExceededError,
  ContextWindowExceededError, ModelRefusalError, plus _EmptyModelResponseError and
  ToolCallInterrupted.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# KISS error types

The public error classes are in `src/kiss/core/kiss_error.py`.

| Class | Base | Meaning / who handles it |
|---|---|---|
| `KISSError(message, code=None)` | `ValueError` | Generic framework error. `str()` gives `"KISS Error: msg"` or `"KISS Error (Code: N): msg"`. Code that catches `ValueError` also catches it. |
| `BudgetExceededError` | `KISSError` | Raised by `_check_limits` when `budget_used >= max_budget`. Orchestrators stop **without** calling recovery LLMs such as RelentlessAgent's summarizer, which would spend more after the limit. `_execute_tool` re-raises it from tools. |
| `ContextWindowExceededError` | `KISSError` | Raised proactively at `CONTEXT_LIMIT_FRACTION` of the window, or when the provider rejects an over-long request. RelentlessAgent starts a fresh session with a summary. Retrying is pointless because each retry adds a message. |
| `ModelRefusalError` | `KISSError` | Raised by the Anthropic adapter on `stop_reason="refusal"` with empty content (seen on claude-fable-5). The loop switches to the fallback model at once. |

Internal to the core area:
- `_EmptyModelResponseError(KISSError)` in `kiss_agent.py`: raised when a turn's response is empty after at least 2 consecutive no-tool turns. Recoverable through fallback.
- `ToolCallInterrupted(BaseException)` in `tool_interrupt.py`: per-tool-call Stop. It never reaches the agent loop.
- A task Stop is a plain `KeyboardInterrupt("Agent stop requested")`.

## Rules of thumb
- In `_run_agentic_loop`, a `KISSError` from a step propagates, apart from the fallback cases. A
  non-KISS `Exception` is classified as context overflow, non-retryable or retryable (see `core-errors-retry-and-model-fallback`).
- Catch the subclass you can handle. `except KISSError` also swallows budget and context errors.

## Sources
- `src/kiss/core/kiss_error.py`
- `src/kiss/core/kiss_agent.py` (`_EmptyModelResponseError`, `_run_agentic_loop`, `_check_limits`)
- `src/kiss/core/tool_interrupt.py` (`ToolCallInterrupted`)
