---
title: How a model name resolves to a provider class (model() factory routing)
uuid: 6188fab4-daf2-49af-96d3-96166d09c37a
summary: 'How model() maps a name to a class: harbor prefix strip, dec flag, base_url
  override, OPENAI_COMPATIBLE_PROVIDERS prefixes, gemini-/claude-/cc//codex/, v1 vs
  v2 transport choice.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# How a model name resolves to a provider class

Entry point: `model(model_name, model_config=None, token_callback=None, thinking_callback=None)`
in `src/kiss/core/models/model_info.py`. It returns a `Model` subclass instance. Classes are
loaded lazily via `_load_model_class`, which chains the real `ImportError` into the `KISSError`
(an `ImportError` raised inside an installed SDK, e.g. the missing-`brotli` regression in
`urllib3`, used to be misreported as "SDK not installed").

## Dispatch order
1. `_strip_provider_prefix`: harbor-style `openai/`, `anthropic/`, `google/` prefixes are dropped
   when the rest is `claude-*`, `gemini-*` or an OpenAI prefix (not `gpt-oss`). `openrouter/...`,
   `openai/gpt-oss...`, `meta-llama/...` are kept because KISS routes on them.
2. Catalog entry with `dec: true` -> `DecisionsModel` via `_decisions_model` (before any other
   routing, even a `base_url` override). See `models-decisions-jev-model`.
3. `model_config` contains `base_url` -> OpenAI-compatible adapter at that URL with
   `model_config["api_key"]` (both keys removed from the forwarded config). Transport: explicit
   `use_responses_api` wins; otherwise v2 only when the URL is **exactly** (modulo trailing slash)
   the default endpoint of the same vendor the name routes to
   (`_registered_provider_for_exact_base_url`) and the catalog flag says so. Custom gateways stay
   on v1 because their `/v1/responses` support is unverified.
4. `_match_openai_compatible_provider`: first entry of `OPENAI_COMPATIBLE_PROVIDERS` whose
   `prefixes` match and `excludes` do not:

| name | prefixes | base_url | key attribute | tools+effort | delegate to Responses |
|---|---|---|---|---|---|
| openrouter | `openrouter/` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | True | False |
| openai | `gpt`, `text-embedding`, `o1`, `o3`, `o4`, `codex`, `computer-use` (excl. `openai/gpt-oss`, `codex/`) | `https://api.openai.com/v1` | `OPENAI_API_KEY` | False | True |
| together | `meta-llama/`, `Qwen/`, `mistralai/`, `deepseek-ai/`, `moonshotai/`, `zai-org/`, `openai/gpt-oss`, `google/gemma`, ... | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` | True | False |
| zai | `glm-` | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` | None | False |
| moonshot | `kimi-`, `moonshot-` | `https://api.moonshot.ai/v1` | `MOONSHOT_API_KEY` | None | False |

   The class is `OpenAICompatibleModel2` when `_wants_responses_api` is true (caller flag, else
   catalog `use_responses_api is True`), otherwise `OpenAICompatibleModel`.
5. `gemini-*` -> `GeminiModel` (key `GEMINI_API_KEY`).
6. `claude-*` -> `AnthropicModel` (key `ANTHROPIC_API_KEY`).
7. `cc/*` -> `ClaudeCodeModel`, `codex/*` -> `CodexModel` (`model_runs_task_to_completion`).
8. Otherwise `KISSError("Unknown model name: ...")`.

Note the order: `codex-mini` style names hit the OpenAI prefix `codex`, while `codex/...` is
excluded from OpenAI and reaches `CodexModel`.

## Labels and availability
- `get_model_provider(name)` returns the UI label from the same two tables
  (`OPENAI_COMPATIBLE_PROVIDERS`, then `_NATIVE_PROVIDERS`: `cc/` "Claude Code CLI", `codex/`
  "Codex CLI", `claude-` "Anthropic", `gemini-` "Gemini"), else `"Unknown"`.
- `get_available_models()` = catalog names with `gen` true whose provider is configured
  (`_configured_providers`: API key set, or `claude` on PATH / `find_codex_executable()`).
- `get_default_model()` / `get_fast_model()` pick by first configured credential in the order
  Anthropic, OpenAI, Gemini, OpenRouter, Together, Claude Code CLI, Codex CLI
  (`_model_for_first_configured_provider`); `"No model"` if none.

## Where transport capabilities live
`OpenAICompatibleProvider.tools_accept_reasoning_effort` and `delegate_tools_to_responses` are
read by `OpenAICompatibleModel` through `openai_compatible_provider_for_base_url` (substring host
match, so wire tests can embed the host in a local capture URL). Details in
`models-openai-compatible-transports`.

## Sources
- `src/kiss/core/models/model_info.py` (`model`, `_strip_provider_prefix`, `_lookup_model_info`, `_wants_responses_api`, `_registered_provider_for_exact_base_url`, `OpenAICompatibleProvider`, `OPENAI_COMPATIBLE_PROVIDERS`, `_NATIVE_PROVIDERS`, `_match_openai_compatible_provider`, `_openai_compatible`, `_load_model_class`, `get_model_provider`, `_configured_providers`, `get_default_model`, `get_fast_model`, `model_runs_task_to_completion`)
