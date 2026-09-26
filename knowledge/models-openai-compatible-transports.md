---
title: 'OpenAI-compatible transports: Chat Completions (OpenAICompatibleModel) vs
  Responses API (OpenAICompatibleModel2)'
uuid: bd0fd5c9-007e-434f-a8b6-fb710c11b0ec
summary: 'OpenAICompatibleModel (Chat Completions) vs OpenAICompatibleModel2 (Responses
  API): reasoning_effort defaults, tools+effort probe, delegation to Responses, DeepSeek
  R1 text tools.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# OpenAI-compatible transports

Two adapters share `OpenAICompatibleBase` (`openai_compatible_model.py`):
- `OpenAICompatibleModel` (v1): `client.chat.completions.create`.
- `OpenAICompatibleModel2` (v2, `openai_compatible_model2.py`): `client.responses.create`
  (`/v1/responses`). Same public surface, so either can be swapped in.
The factory picks v2 when `use_responses_api` is true (caller flag or catalog flag written by
`src/kiss/scripts/update_responses_api_support.py` after a live probe); see `models-provider-resolution`.

## Shared base (`OpenAICompatibleBase`)
- `_api_model_name = _provider_model_name(name)`: strips thinking-alias suffix (`-xhigh`, or
  `alias_of`-marked `-high`/`-medium`/`-low`/`-max`) then the `openrouter/` prefix
  (OpenRouter wants `<vendor>/<id>`).
- Default reasoning: if the catalog entry has `thinking` and the caller set neither
  `reasoning_effort` nor `reasoning.effort`, `reasoning_effort = thinking` is set on a copy of
  `model_config` (`_model_thinking_level`). Unknown/custom names get none.
- `_ensure_client()`: one `OpenAI(base_url, api_key, timeout=1800, max_retries=1,
  default_headers=extra_headers)` reused until those inputs change (a client per step used to
  create 100 connection pools per 100-step run). `_MAX_RETRIES = 1` because SDK silent retries
  re-billed identical answers (up to 9 per logical turn with the agent's own 3 retries).
- `extract_cost_from_response` (OpenRouter `usage.cost`), OpenRouter-Anthropic `cache_control`,
  embeddings (`get_embedding`, `get_embeddings`), DeepSeek-R1 detection
  (`DEEPSEEK_REASONING_MODELS`).

## v1 Chat Completions specifics
- Kwargs: `model_config` minus framework keys, filtered against the SDK signature of
  `Completions.create` (`_CHAT_REQUEST_PARAMS`), plus `model` and normalized `messages`.
- `tools` + `reasoning_effort` capability per endpoint (`_tools_reasoning_effort_capability`):
  provider's declared `tools_accept_reasoning_effort` (OpenAI False, OpenRouter/Together True),
  else a learned verdict cached per `(base_url, api_model_name)`. Unknown endpoints are probed
  optimistically by `_create_chat_completion_adaptive`: a 400 mentioning `reasoning_effort` records
  False and retries once without it; success records True (a rejection always wins over a racing
  success).
- Delegation: a tool turn with `reasoning_effort` goes to `_generate_with_tools_via_responses`
  when `_should_delegate_to_responses()` (flag `use_responses_api`, else provider
  `delegate_tools_to_responses`, only `api.openai.com`). A cached v2 delegate is rebuilt from the
  chat conversation every turn (the chat conversation stays the single source of truth); raw
  Responses items (incl. reasoning) are cached per `call_id` and pruned when the call leaves the
  conversation. Callbacks, base_url, api_key, extra_headers are re-synced to the delegate each
  turn (stale callbacks once streamed into a finished task's printer).
- DeepSeek R1 models: no native tools; `_generate_with_text_based_tools` injects a tools prompt and
  parses JSON tool calls; `<think>` reasoning is stripped (`_extract_deepseek_reasoning`).
- `finish_reason == "length"` -> `KISSError` (`_raise_for_finish_reason`): truncated tool JSON would
  otherwise parse as `{}` and mislead the agent.
- Reasoning deltas (`_delta_reasoning_text`) are streamed inside a thinking bracket.
- Audio: output audio saved to `last_audio_data`; audio tokens split out for billing.

## v2 Responses specifics (`_shape_responses_kwargs`)
- `system_instruction` -> top-level `instructions` (never a message).
- `reasoning_effort` -> `reasoning.effort`; `reasoning.summary` defaults to `"auto"` when an effort
  is sent, because OpenAI otherwise returns empty reasoning summaries and no thinking is shown.
- `max_tokens`/`max_completion_tokens` -> `max_output_tokens`; `response_format` -> `text.format`;
  Chat-style `tool_choice` flattened; tools flattened (`_flatten_tools_schema`).
- Filtered against `Responses.create` params (`_RESPONSES_REQUEST_PARAMS`); `stream_options`
  sanitized to the Responses shape.
- Content parts `input_text`/`input_image`/`input_file`; tool results are `function_call_output`
  items; pending function calls are validated before a new request
  (`_ensure_no_pending_function_calls`).
- `response.failed` / `status="incomplete"` raise instead of returning partial output.

## Attachment formats
Both transports accept the same set: images PNG/JPEG/WEBP/GIF (`OPENAI_INPUT_IMAGE_MIME_TYPES`),
PDFs (v1 `file` part, v2 `input_file`), audio mp3/wav (`OPENAI_INPUT_AUDIO_FORMATS`); others are
dropped with a warning.

## Sources
- `src/kiss/core/models/openai_compatible_model.py` (`OpenAICompatibleBase`, `OpenAICompatibleModel`, `_provider_model_name`, `_model_thinking_level`, `_create_chat_completion_adaptive`, `_record_tool_effort_verdict`, `_should_delegate_to_responses`, `_generate_with_tools_via_responses`, `_generate_with_text_based_tools`, `_raise_for_finish_reason`, `_stream_chat_completion`, `DEEPSEEK_REASONING_MODELS`)
- `src/kiss/core/models/openai_compatible_model2.py` (`OpenAICompatibleModel2`, `_shape_responses_kwargs`, `_consume_stream`, `_consume_stream_events`, `_ensure_no_pending_function_calls`)
- `src/kiss/core/models/model_info.py` (`OPENAI_COMPATIBLE_PROVIDERS`, `openai_compatible_provider_for_base_url`)
