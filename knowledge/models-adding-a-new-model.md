---
title: Adding a new model, provider or custom endpoint to KISS
uuid: f0eee450-67cf-467b-86df-b377593b379d
summary: 'Adding models: update_models.py probes and writes MODEL_INFO.json, hand
  entries, MY_MODELS.json custom endpoints, new OPENAI_COMPATIBLE_PROVIDERS vendor,
  new Model subclass, tests.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Adding a new model, provider or custom endpoint

Pick the lightest option that works.

## 1. Existing provider, new model id: run the updater
`uv run python src/kiss/scripts/update_models.py [--dry-run] [--skip-test] [--test-existing]
[--verbose] [--model-info PATH] [--scrub-only]`
- Fetches price/context from OpenRouter, Together, Gemini, Anthropic and OpenAI listings
  (`fetch_openrouter`, `fetch_together`, `fetch_gemini`, `fetch_anthropic`, `fetch_openai`), and
  OpenRouter decisions models (`?output_modalities=decisions`).
- Live-probes new models: generation (`test_generate`), embeddings, function calling (`fc`),
  highest accepted `reasoning_effort` (`detect_thinking_level` -> `thinking`, plus `-xhigh`/level
  alias entries with `alias_of`), Responses API support (`use_responses_api`), decisions (`dec`).
- Writes atomically (temp file + `os.replace`) because `model_info` loads at import time. New
  entries get a `"comment": "NEW"` marker. When writing the bundled catalog it also refreshes the
  README "Models Supported" totals.
- `src/kiss/scripts/update_responses_api_support.py` re-probes `use_responses_api` flags alone.
Requires the provider API keys in the environment.

## 2. Hand-written catalog entry
Add to `src/kiss/core/models/MODEL_INFO.json` (the source of truth in a checkout) at least
`context_length`, `input_price_per_1M`, `output_price_per_1M`; add `fc`, `thinking`,
`use_responses_api`, cache prices as needed (schema: `models-model-info-json-schema`). The name
must match a routing prefix (`models-provider-resolution`) or `model()` raises
`Unknown model name`. Without an entry, `calculate_cost` raises for any non-zero usage, so an agent
cannot run on the model.

## 3. Private endpoint without code changes
Settings panel -> custom model (or edit `~/.kiss/MY_MODELS.json`):
```json
"my-llama": {"endpoint": "http://localhost:8000/v1", "api_key": "sk-...",
             "headers": "X-Team: research", "context_length": 128000,
             "input_price_per_1M": 0.0, "output_price_per_1M": 0.0}
```
`custom_model_config` turns this into `base_url`/`api_key`/`extra_headers`, and the factory
builds an OpenAI-compatible v1 client for any name (the `base_url` branch runs before prefix
routing). Programmatic equivalent:
`model("my-llama", model_config={"base_url": "http://localhost:8000/v1", "api_key": "x"})`.

## 4. New OpenAI-compatible vendor (code)
1. Add an `OpenAICompatibleProvider(name, label, host, base_url, prefixes, excludes,
   api_key_name, tools_accept_reasoning_effort, delegate_tools_to_responses)` row to
   `OPENAI_COMPATIBLE_PROVIDERS` in `model_info.py`. Order matters: the first match wins.
   `label` is the picker/UI name; `host` is the substring used to find capabilities from a
   `base_url`; `tools_accept_reasoning_effort=None` means "unverified, learn at runtime".
2. Add the key field (`<VENDOR>_API_KEY` with `os.getenv` default) to `config.py`; the settings
   panel / `api_keys.env` handling lives in `kiss.core.vscode_config`.
3. If cache hits are billed specially, add a branch to `_provider_cache_defaults`.
4. Add catalog entries (option 1 or 2). `get_model_provider`, `_configured_providers` and
   routing read the table, so no other routing edits are needed. The Z.AI and Moonshot
   rows were added this way (`src/kiss/tests/test_readme_zai_moonshot.py`). Default/fast model
   choice is separate: `get_default_model`, `get_fast_model` and the credential priority list in
   `_model_for_first_configured_provider` are hardcoded, so extend them for a new provider.

## 5. New native SDK provider (code)
Subclass `Model` (`src/kiss/core/models/model.py`) and implement the abstract methods
`initialize`, `generate`, `generate_and_process_with_tools`,
`extract_input_output_token_counts_from_response`, `get_embedding`; override
`add_function_results_to_conversation_and_return` if the provider's tool-result format differs. Follow the existing adapters:
drop `FRAMEWORK_ONLY_CONFIG_KEYS`, filter kwargs with `accepted_request_params` /
`_keep_supported_request_params`, stream through `stop_aware_events` or `StreamAbortWatchdog`, close
the thinking bracket on every exit, raise `KISSError` only for non-retryable errors. Then add a
branch in `model()`, a `_NATIVE_PROVIDERS` label, an export in `models/__init__.py`, and an optional
dependency.

## Tests
Model tests live in `src/kiss/tests/core/models/`; SSE wire harnesses
(`anthropic_sse_harness.py`, `openai_sse_harness.py`, `gemini_sse_harness.py`) replay recorded
provider streams end-to-end without mocks; updater tests are in `src/kiss/tests/scripts/`.

## Sources
- `src/kiss/scripts/update_models.py` (module docstring, `fetch_*`, `test_generate`, `test_function_calling`, `detect_thinking_level`, `test_responses_api`, `test_decisions`)
- `src/kiss/scripts/update_responses_api_support.py`
- `src/kiss/core/models/model_info.py` (`OpenAICompatibleProvider`, `OPENAI_COMPATIBLE_PROVIDERS`, `_NATIVE_PROVIDERS`, `model`, `custom_model_config`, `_provider_cache_defaults`)
- `src/kiss/core/models/model.py` (`Model`)
- `src/kiss/core/config.py`
- `src/kiss/tests/core/models/`
