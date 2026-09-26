---
title: Prompt caching per provider and keep-alive pings during long tool calls
uuid: 3c21ee99-2d24-4098-8519-8155ac4b6af6
summary: 'Prompt caching: Anthropic top-level cache_control, enable_cache, OpenRouter-Anthropic
  extra_body, automatic OpenAI/Gemini caching, PromptCacheKeepAlive pings (240 s,
  max 10) in long tool calls.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Prompt caching per provider and keep-alive pings

## Switch
`model_config["enable_cache"]` (default `True`) is a framework-only key
(`FRAMEWORK_ONLY_CONFIG_KEYS` in `model.py`), never forwarded to an SDK.

## Anthropic (`AnthropicModel`)
`_build_create_kwargs` adds a single **top-level** `cache_control={"type": "ephemeral"}` when
caching is on. Anthropic places the breakpoint at the last cacheable block automatically and moves
it forward as the conversation grows; no per-block markers are written. The cache TTL is 5 minutes,
counted from the **start** of the request that last read or wrote it; each read renews it for free.

## OpenRouter + Claude (`openrouter/anthropic/*`)
`OpenAICompatibleBase._apply_cache_control_for_openrouter_anthropic` puts the same
`{"cache_control": {"type": "ephemeral"}}` into `extra_body` (Chat Completions and Responses
transports), only when `enable_cache` is on and the name starts with `openrouter/anthropic/`.

## OpenAI, Gemini, Moonshot, others
No request marker is sent: these providers cache prefixes automatically. KISS only accounts for
cached tokens reported in usage (`prompt_tokens_details.cached_tokens` / `cache_write_tokens`,
Gemini `cached_content_token_count`). Pricing rules are in `models-cost-computation`. Cache hits
need a stable prefix, so adapters keep request bytes stable (system instruction, tool schema order)
between turns.

## Keep-alive during long tool calls
Problem: a tool call longer than the 5-minute TTL (tests, builds, sub-agent fan-out) lets the
cache expire, and the next step re-writes the whole context at 1.25x instead of reading it at 0.1x
(0.025x on Claude Fable 5.1).

### Which calls get one (`is_long_running_call(name, args)`)
- `FAN_OUT_TOOLS = {run_parallel, run_commands_parallel, run_agent}`;
- `bash_job` with `action == "wait"` given explicitly;
- any call whose `timeout_seconds` or `timeout` arg is `>= LONG_TOOL_TIMEOUT_SECONDS` (300);
  missing or non-numeric values are ignored.
- In `KISSAgent._execute_step` the call must also not have been blocked by `tool_call_hook` or
  `tool_call_guard`.

### Mechanics (`PromptCacheKeepAlive` in `src/kiss/core/prompt_cache_keepalive.py`)
- `KISSAgent._execute_tool_keeping_cache_warm` wraps `_execute_tool` in
  `with PromptCacheKeepAlive(model, function_map, self._cached_tools_schema, self._prompt_cache_touched_at)`.
- `_prompt_cache_touched_at` is set to `time.time()` just before each model call, matching how the
  TTL is measured.
- Daemon thread `prompt-cache-keepalive` calls `model.keep_prompt_cache_warm(function_map,
  tools_schema)` at `touched_at + KEEP_ALIVE_INTERVAL_SECONDS` (240 s), then 240 s after each
  previous ping start, up to `KEEP_ALIVE_MAX_PINGS` (10, about 40 min; past that a miss is
  cheaper). It stops when the body ends, a ping raises (logged warning), or the model returns `None`.
- `__exit__` sets the stop event and **joins** the thread, possibly waiting for an in-flight ping
  (bounded at 30 s for Anthropic with retries off). Afterwards the model object is single-threaded
  again, so `responses` needs no lock.
- In `finally`, on the agent thread, every ping response goes through
  `_update_tokens_and_budget_from_response` (pings are billed, even if the tool was interrupted or
  hit a limit), and `_prompt_cache_touched_at` moves to `last_ping_at`.

### Provider implementations
- `Model.keep_prompt_cache_warm` (base) returns `None`: no ping. Only `AnthropicModel` overrides it.
  The ping must not change the conversation, which ends with the assistant turn whose tool calls
  are in flight.
- `AnthropicModel.keep_prompt_cache_warm` rebuilds the last request via `_build_create_kwargs`
  (tools, system, thinking, `tool_choice`, messages: the parts the cache is keyed on), returns
  `None` if `cache_control` is absent (caching disabled), appends one user message answering every
  pending `tool_use` with a placeholder `tool_result` (`KEEP_ALIVE_TEXT`; the API rejects a
  dangling tool-use turn), sets `max_tokens` to `max(KEEP_ALIVE_MAX_TOKENS=256, budget_tokens + 1)`
  for manual thinking, and sends with `max_retries=0` and `KEEP_ALIVE_TIMEOUT_SECONDS` (30 s). The
  reply is discarded without touching the conversation; the raw message is returned for billing.

### Thread-safety
- The ping only reads `model.conversation`. Sorcar's `set_model` tool is the only thing that
  reassigns it, and it runs on the agent thread and is not long-running.
- No tool interrupt lands during `__exit__`: `_execute_tool`'s `finally` sets `token.closing` and
  calls `unregister_tool_call` first.

## Cache-write accounting
`cache_creation_tokens(usage, get)` splits Anthropic cache writes into 5-minute and 1-hour buckets;
an aggregate without TTL split is billed as 1-hour (more expensive, never under-bills). The Claude
Code CLI adapter reuses it on the CLI's JSON usage.

## Gotchas
- `enable_cache=False` also disables keep-alive for Anthropic.
- Changing `system_instruction`, the tool list, or thinking config mid-run invalidates the cached
  prefix.

## Sources
- `src/kiss/core/models/anthropic_model.py` (`AnthropicModel._build_create_kwargs`, `keep_prompt_cache_warm`, `KEEP_ALIVE_*`, `cache_creation_tokens`)
- `src/kiss/core/models/openai_compatible_model.py` (`OpenAICompatibleBase._apply_cache_control_for_openrouter_anthropic`)
- `src/kiss/core/models/model.py` (`FRAMEWORK_ONLY_CONFIG_KEYS`, `Model.keep_prompt_cache_warm`)
- `src/kiss/core/prompt_cache_keepalive.py` (`is_long_running_call`, `PromptCacheKeepAlive`, constants)
- `src/kiss/core/kiss_agent.py` (`KISSAgent._execute_step`, `_execute_tool_keeping_cache_warm`, `_execute_tool`)
- `src/kiss/tests/core/test_prompt_cache_keepalive.py`
