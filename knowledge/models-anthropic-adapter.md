---
title: 'Anthropic adapter (AnthropicModel, Claude Messages API): thinking modes, request
  building, hand-off normalization, refusals'
uuid: 7f34c8b9-cd97-4de1-9e5e-aac08aea853b
summary: 'AnthropicModel request building: adaptive vs enabled thinking, max_tokens
  defaults, interleaved-thinking beta, tool_choice any, hand-off normalization from
  OpenAI formats, refusals.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Anthropic adapter (`AnthropicModel`)

Routed for names starting with `claude-` with `ANTHROPIC_API_KEY`
(`src/kiss/core/models/anthropic_model.py`). Uses `client.messages.stream(...)`.

## Client
Built in `initialize()` and reused across runs; rebuilt only when `(api_key,
ANTHROPIC_WORKSPACE_ID)` changes. Timeouts: read = `stream_stall_timeout` (default 180 s), connect
10 s; `_MAX_RETRIES = 1`. Sends `anthropic-workspace-id` when the env var is set.

## Request building (`_build_create_kwargs`)
1. Copy `model_config` minus `FRAMEWORK_ONLY_CONFIG_KEYS`.
2. System prompt = `system_instruction` merged with any `role="system"` messages
   (`merge_system_texts`), sent as top-level `system` (the Messages API rejects the system role).
3. `max_tokens` (or `max_completion_tokens`) else 16384; `stop` -> `stop_sequences`.
4. Unknown keys are dropped and reported once via `_keep_supported_request_params` against the
   SDK signature of `Messages.stream` (`_ANTHROPIC_REQUEST_PARAMS`).
5. Thinking, unless the caller set `thinking`:
   - `_supports_extended_thinking(name)`: catalog `extended_thinking` if set, else
     `_parse_claude_version` family in opus/sonnet/haiku/fable with major >= 4.
   - Then `max_tokens` defaults to 65536 for Opus, 64000 otherwise (unless user-set).
   - `_uses_adaptive_thinking(name)`: catalog `adaptive_thinking` if set, else major >= 5, or Opus
     4.6+. Adaptive sends `{"type": "adaptive", "display": "summarized"}` (the API default
     `display: omitted` returns empty signature-only thinking). Otherwise
     `{"type": "enabled", "budget_tokens": min(10000, max_tokens-1)}` when that is >= 1024.
   - With thinking on, the beta header `anthropic-beta: interleaved-thinking-2025-05-14` is merged
     into `extra_headers`.
6. Tools converted from OpenAI format (`_build_anthropic_tools_schema`: `name`, `description`,
   `input_schema`). If tools are present and neither `tool_choice` nor thinking is set,
   `tool_choice={"type": "any"}` forces a tool call (thinking forbids forced tool choice).
7. `cache_control={"type": "ephemeral"}` when `enable_cache` (see `models-prompt-caching`).

Historic bug: `claude-fable-5` and then `claude-opus-5` never got the `thinking` param because of a
hard-coded `claude-*-4` allowlist; reasoning stayed invisible and `KISSAgent` misread
encrypted-only replies as empty. The version parser fixed this for future majors.

## Conversation normalization (`_normalize_conversation_for_api`)
Conversations can be handed off from other providers (Sorcar `set_model`). Before each request:
Responses-API items are converted (`responses_items_to_chat_messages`); `role="tool"` becomes a user
message with a `tool_result` block; assistant `tool_calls` become `tool_use` blocks; OpenAI
`image_url`/`file` parts become `image`/`document` blocks; `input_audio` is transcribed with Whisper
(`transcribe_audio`, needs `OPENAI_API_KEY`) or dropped; whitespace-only text blocks are dropped
(the API rejects them); consecutive user turns are merged so `tool_result` directly follows its
`tool_use`. An all-whitespace conversation raises `KISSError`.
Response blocks are normalized to dicts; `thinking` blocks keep their `signature` for replay.

## Tool results
`add_function_results_to_conversation_and_return` writes `tool_result` blocks; binary attachments
from tools become image blocks **inside** the `tool_result` (Anthropic is the one provider that
accepts bytes there; others get a follow-up user message).

## Errors
- `stop_reason == "refusal"` with empty content -> `ModelRefusalError` (`_raise_on_refusal`), so the
  agent does not burn retries on a deterministic refusal and the fallback swap is not mislabelled
  as "empty responses".
- `stop_reason == "max_tokens"`: incomplete `tool_use` blocks are discarded.
- Missing workspace id -> `KISSError` with setup hint (see `models-api-keys-and-credentials`).
- Stop/stall -> `KeyboardInterrupt` / retryable `TimeoutError` (see `models-streaming-and-abort`).
- No embeddings: `get_embedding` raises `KISSError`.

## Sources
- `src/kiss/core/models/anthropic_model.py` (`AnthropicModel`, `_build_create_kwargs`, `_supports_extended_thinking`, `_uses_adaptive_thinking`, `_parse_claude_version`, `_normalize_conversation_for_api`, `_normalize_message_for_api`, `_normalize_content_blocks`, `_build_anthropic_tools_schema`, `_raise_on_refusal`, `add_function_results_to_conversation_and_return`, `_openai_part_to_anthropic_block`)
- `src/kiss/core/models/model.py` (`merge_system_texts`, `responses_items_to_chat_messages`, `transcribe_audio`, `_keep_supported_request_params`)
