---
title: Model providers area map (src/kiss/core/models)
uuid: cf58317d-b0ce-46a8-b220-b4f6189701b6
summary: 'Area map of src/kiss/core/models: MODEL_INFO.json, model_info.py factory/cost/fallbacks,
  Model base, Anthropic/OpenAI/Gemini/CLI/decisions adapters, stream_abort, heif,
  autoroute SEA.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Model providers area map

The model layer turns a model **name** into a `Model` object that streams text, calls tools,
reports token usage, and gets billed. Agents (`KISSAgent`, Sorcar) only use the `Model` interface
and `model_info` helpers.

## Files
| File | Role | Detail page |
|---|---|---|
| `src/kiss/core/models/MODEL_INFO.json` | catalog: 689 models, prices, context, capability flags | `models-model-info-json-schema` |
| `src/kiss/core/models/model_info.py` | catalog loading + `MY_MODELS.json`, `model()` factory, provider table, cost, fallbacks, default model | `models-catalog-loading-and-my-models`, `models-provider-resolution`, `models-cost-computation` |
| `src/kiss/core/models/model.py` | `Model` ABC, `Attachment`, `CLITextModel`, conversation converters, framework-only config keys, Whisper transcription | `models-attachments-and-heif`, `models-cli-backed-cc-codex` |
| `anthropic_model.py` | `AnthropicModel` (`claude-*`) | `models-anthropic-adapter`, `models-prompt-caching` |
| `openai_compatible_model.py` | `OpenAICompatibleBase`, `OpenAICompatibleModel` (Chat Completions) | `models-openai-compatible-transports` |
| `openai_compatible_model2.py` | `OpenAICompatibleModel2` (Responses API) | `models-openai-compatible-transports` |
| `gemini_model.py` | `GeminiModel` (`gemini-*`) | `models-gemini-adapter` |
| `claude_code_model.py`, `codex_model.py` | `cc/*`, `codex/*` CLI-backed models | `models-cli-backed-cc-codex` |
| `decisions_model.py` | `DecisionsModel` for OpenRouter jev decisions | `models-decisions-jev-model` |
| `stream_abort.py` | stall timeout and Stop-button abort for streams | `models-streaming-and-abort` |
| `heif.py` | HEIC/HEIF detection and JPEG transcoding | `models-attachments-and-heif` |
| `src/kiss/agents/seas/autoroute_sea.py` | `/autoroute`: route task units to cheapest model tier | `models-autoroute-cost-routing` |
| `ROUTING.md` (repo root) | empty placeholder, no content | `models-autoroute-cost-routing` |

Cross-cutting pages: `models-api-keys-and-credentials`, `models-embeddings`,
`models-adding-a-new-model`.

## Life of a model call
1. `model(name, model_config, token_callback, thinking_callback)` picks the class
   (`models-provider-resolution`) and API key (`models-api-keys-and-credentials`).
2. `initialize(prompt, attachments)` builds/reuses the SDK client and seeds `conversation`.
3. `generate()` or `generate_and_process_with_tools(function_map, tools)` streams the reply
   through `stop_aware_events` / `StreamAbortWatchdog`; reasoning goes to the thinking callback.
4. `add_function_results_to_conversation_and_return(...)` appends tool results in the provider's
   format.
5. The agent bills via `extract_cost_from_response` or `calculate_cost`.

## Public helpers in `model_info.py`
`model`, `MODEL_INFO`, `calculate_cost`, `get_max_context_length` (raises `KISSError` for unknown
models), `get_available_models`, `get_model_provider`, `get_default_model`, `get_fast_model`,
`model_runs_task_to_completion`, `custom_model_config`, `list_custom_models`,
`save_custom_model`, `delete_custom_model`.

Fallbacks: `get_fallback_model(name)` (used by `KISSAgent._try_switch_to_fallback` on
non-retryable provider errors such as unavailable model or low credit) returns the catalog
`fallback`, else `openrouter_twin(name)`: the `openrouter/<vendor>/<model>` entry for the same model
(dots and dashes matched alike, only when `OPENROUTER_API_KEY` is set), so the model keeps
running and only the billing route changes.
`declared_fallback` returns only the explicit catalog value. The agent-side retry logic belongs to
the core agent area (`core-errors-retry-and-model-fallback`).

## Invariants worth knowing
- `model_config` keys in `FRAMEWORK_ONLY_CONFIG_KEYS` (`system_instruction`, `use_responses_api`,
  `stream_stall_timeout`, `enable_cache`, `stream`, `work_dir`) are never forwarded raw to an SDK
  (adapters translate `system_instruction`, e.g. Gemini puts it in `GenerateContentConfig`);
  other unknown keys are dropped with a one-time report (`_keep_supported_request_params`).
- Anthropic and OpenAI adapters reuse their SDK client until its construction inputs change
  (Anthropic: API key, workspace id; OpenAI: base URL, API key, `extra_headers`); Gemini creates a fresh `genai.Client` on each `initialize`.
- SDK retries are capped at 1 (`_MAX_RETRIES`) so a retried request is not billed twice silently.
- `KISSError` means "do not retry"; stalls raise `TimeoutError` (retryable); user stop raises
  `KeyboardInterrupt`.

## Sources
- `src/kiss/core/models/__init__.py`
- `src/kiss/core/models/model_info.py` (`model`, `get_fallback_model`, `declared_fallback`, `openrouter_twin`, `get_max_context_length`)
- `src/kiss/core/models/model.py` (`Model`, `FRAMEWORK_ONLY_CONFIG_KEYS`)
- `src/kiss/agents/seas/autoroute_sea.py`
