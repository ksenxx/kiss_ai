---
title: KISSAgent error handling - retries, non-retryable errors, and one-shot model
  fallback
uuid: ae035b1f-9572-4507-be52-5d8597afbadc
summary: 'KISSAgent loop error handling: 3 consecutive retryable errors, non-retryable
  auth/not_found/credit errors, empty or refusal turns, and the one-shot fallback
  model swap.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Error handling, retries and model fallback in the agent loop

## Classification (`_is_retryable_error`)
An error is **non-retryable** when its type name contains `AuthenticationError` or `PermissionDenied`
(`_NON_RETRYABLE_ERROR_TYPES`), or its lower-cased message contains a phrase from `_NON_RETRYABLE_PHRASES`:
`api key`, `api_key`, `invalid key`, `invalid x-api-key`, `incorrect api key`, `unauthorized`,
`permission denied`, `could not resolve authentication`, `is not available`, `not_found_error` or
`credit balance is too low`. Everything else, including rate limits and 5xx errors, is retryable.

## Loop behaviour (`_run_agentic_loop`)
Exceptions from `_execute_step` are handled in this order:
1. **`KISSError`** (including limit errors) propagates. There are two exceptions:
   `_EmptyModelResponseError` (an empty response after at least 2 consecutive no-tool turns) and
   `ModelRefusalError` (Anthropic `stop_reason="refusal"`). For these it tries `_try_switch_to_fallback`
   and continues on success.
2. **Context overflow** phrases lead to `ContextWindowExceededError`.
3. **Non-retryable** errors lead to a fallback swap. Without a fallback it raises `KISSError("Non-retryable error from model: ...")`.
4. **Retryable** errors increment `consecutive_errors`. At `MAX_CONSECUTIVE_ERRORS = 3` it raises
   `KISSError("... failed with 3 consecutive errors ...")`. Otherwise the user message
   `Failed to get response from Model: <e>.\nPlease try again.\n` is appended and the loop continues.
   Each retry uses a step. A successful step resets the counter.

`KeyboardInterrupt` (task Stop) is not an `Exception`, so it always propagates.

## `_try_switch_to_fallback(reason)`
- Only **one** swap per `run()` (`_fallback_used`).
- `get_fallback_model(model_name)` returns a fallback declared in the catalog unchanged. Otherwise it
  returns `None` when `OPENROUTER_API_KEY` is unset (before any twin lookup), else the model's OpenRouter
  twin. `_try_switch_to_fallback` itself rejects a missing fallback or one equal to the current model.
- Config: a declared fallback keeps the caller's `model_config`. The OpenRouter twin drops
  `_ENDPOINT_CONFIG_KEYS = {base_url, api_key, extra_headers}` because those belong to the provider that just failed.
- The new model gets `initialize("")`, then the old `conversation` list and `usage_info_for_messages`.
  The tool schema is rebuilt for the new provider, and a `system_prompt` printer event announces
  `Model X returned <reason>; switching to fallback model: Y`.
- The caller resets the progress trackers (text-only counter and last text) and `consecutive_errors`.

## Related history
Git history includes fixes for recovering from repeated empty turns instead of aborting (aa0ad3631),
detecting Anthropic safety refusals and falling back at once (51ff3ff40), and auto-fallback on
non-retryable provider errors (f8ed33b82).

## Sources
- `src/kiss/core/kiss_agent.py` (`_is_retryable_error`, `_run_agentic_loop`, `_try_switch_to_fallback`, `_EmptyModelResponseError`, `MAX_CONSECUTIVE_ERRORS`)
- `src/kiss/core/models/model_info.py` (`get_fallback_model`, `declared_fallback`)
