---
title: 'Gemini adapter (GeminiModel, google-genai SDK): config mapping, thinking,
  thought signatures, streaming'
uuid: 2d6ec48e-8e4f-41ab-b2fd-2dbcc7462991
summary: 'GeminiModel on google-genai: GenerateContentConfig mapping, include_thoughts
  thinking, thought parts, thought_signature replay for tool calls, system hoisting,
  forced httpx timeout, usage chunk.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Gemini adapter (`GeminiModel`)

Routed for names starting with `gemini-` using `GEMINI_API_KEY`
(`src/kiss/core/models/gemini_model.py`). SDK: `google-genai` (`genai.Client`).

## Client and transport
`initialize()` builds `genai.Client(api_key=..., http_options=HttpOptions(httpx_client=...))`
with a `_ResponseTrackingHttpxClient`. That custom httpx client exists for two reasons:
1. **Timeout**: the SDK passes an explicit `timeout=None` to `build_request` unless
   `HttpOptions.timeout` is set, and `None` in httpx means *no timeout*; the client forces a read
   timeout of `stream_stall_timeout` with a short connect timeout.
2. **Abortability**: `generate_content_stream` returns a bare generator, so the client remembers
   the in-flight response and `_AbortableStream` exposes it as `.response`, letting
   `StreamAbortWatchdog` shut down its socket on Stop (see `models-streaming-and-abort`).

## Request config (`_build_config`)
- Every `model_config` key that is a real `GenerateContentConfig` field (`_GEMINI_CONFIG_FIELDS`)
  is forwarded; others are reported once and dropped via `_keep_supported_request_params`
  (forwarding extras would raise a pydantic `ValidationError`).
- Portable aliases: `max_tokens` / `max_completion_tokens` -> `max_output_tokens`; `stop` ->
  `stop_sequences`.
- Default `thinking_config = ThinkingConfig(include_thoughts=True)` so summarized reasoning is
  returned.
- `system_instruction` = config `system_instruction` merged with any OpenAI-style `role="system"`
  messages in the conversation (`merge_system_texts`), because Gemini contents accept only
  `user`/`model` roles (matters after a model hand-off, e.g. Sorcar `set_model`).

## Conversation conversion
The internal conversation is Chat-Completions-shaped. `_chat_messages()` normalizes it (also
converting Responses-API items via `responses_items_to_chat_messages`), and
`_convert_conversation_to_gemini_contents` builds `types.Content` lists: tool calls become
`function_call` parts, tool results `function_response` parts (`_tool_result_response_dict`), media
blocks and data URLs become inline parts (`_media_block_to_part`, `_data_url_to_part`).

## Thinking and thought signatures
- Parts with `thought=True` (`_is_thought`) are streamed through the thinking callback but are
  **never** added to assistant content (`_parse_parts`); otherwise they would be printed twice and
  re-uploaded every step.
- Gemini attaches a `thought_signature` to function-call parts. `_parse_parts` stores it in
  `_thought_signatures[call_id]` (call ids are generated as `call_<8 hex>`), and
  `_function_call_part` / `_function_response_part` re-attach it when the conversation is
  replayed; Gemini requires this for multi-step tool use with thinking. `reset_conversation` and
  `initialize` clear the map so a previous run's reasoning is not replayed.

## Streaming (`_stream_turn`, `_generate_parts`)
Streams when a token callback is set, via `stop_aware_events`; a byte-level `httpx.TimeoutException`
is converted to the same `stall_error`. The turn is billed against the chunk that carries
`usage_metadata` (not the last chunk, which may be a `finishReason`-only chunk and would report the
step as free). If the stream yields nothing, it falls back to unary `generate_content`.

## Tokens and embeddings
- Tokens: input = `prompt_token_count - cached_content_token_count + tool_use_prompt_token_count`;
  output = `candidates_token_count + thoughts_token_count`; cache writes 0.
- `get_embedding(text, embedding_model=None)` uses `client.models.embed_content`; the default model
  is this instance's own name, so a non-embedding model fails instead of silently embedding with
  another model.

## Sources
- `src/kiss/core/models/gemini_model.py` (`GeminiModel`, `_ResponseTrackingHttpxClient`, `_AbortableStream`, `_build_config`, `_resolve_system_instruction`, `_parse_parts`, `_is_thought`, `_function_call_part`, `_function_response_part`, `_stream_turn`, `_generate_parts`, `extract_input_output_token_counts_from_response`, `get_embedding`)
- `src/kiss/core/models/model.py` (`merge_system_texts`, `responses_items_to_chat_messages`)
