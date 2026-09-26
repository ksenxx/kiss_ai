---
title: MODEL_INFO.json schema, pricing and capability fields
uuid: 7b087878-52ee-474d-bef7-3007b8c87272
summary: Per-model keys in MODEL_INFO.json / MY_MODELS.json (context_length, prices,
  fc, emb, gen, dec, thinking, alias_of, use_responses_api, cache and audio prices)
  and the ModelInfo attributes they become.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# MODEL_INFO.json schema, pricing and capability fields

`src/kiss/core/models/MODEL_INFO.json` is a flat JSON object: key = catalog model name (e.g.
`claude-opus-4-8`, `gpt-5.5-xhigh`, `openrouter/anthropic/claude-fable-5`, `cc/opus`,
`codex/gpt-5.5`), value = one entry. `_build_model_info_entry` turns each entry into a
`ModelInfo` object; the module-level dict `MODEL_INFO: dict[str, ModelInfo]` is built at import
time (see `models-catalog-loading-and-my-models`).

## Keys (JSON key -> `ModelInfo` attribute)

| JSON key | Required | Default | ModelInfo attribute / meaning |
|---|---|---|---|
| `context_length` | yes | - | `context_length`, max input+output tokens |
| `input_price_per_1M` | yes | - | `input_price_per_1M`, USD per 1M input tokens |
| `output_price_per_1M` | yes | - | `output_price_per_1M` |
| `fc` | no | `True` | `is_function_calling_supported`. `fc=false` = unreliable tool calling, non-agentic use only |
| `emb` | no | `False` | `is_embedding_supported` |
| `gen` | no | `True` | `is_generation_supported` |
| `dec` | no | `False` | `is_decisions_supported`: OpenRouter typed-decision model (see `models-decisions-jev-model`) |
| `thinking` | no | `None` | default/highest `reasoning_effort` (e.g. `"high"`, `"xhigh"`) for OpenAI-compatible models |
| `alias_of` | no | `None` | marks a generated thinking-level alias (`gpt-5.5-xhigh` -> `gpt-5.5`) |
| `use_responses_api` | no | `None` | `True` = build `OpenAICompatibleModel2` (`/v1/responses`); caller's `model_config["use_responses_api"]` overrides |
| `cache_read_price_per_1M`, `cache_write_price_per_1M`, `cache_write_1h_price_per_1M` | no | `None` | explicit cache prices; otherwise filled by `_apply_cache_pricing` (see `models-cost-computation`) |
| `audio_input_price_per_1M`, `audio_output_price_per_1M` | no | `None` | audio token prices (OpenAI audio models) |
| `extended_thinking` | no | `None` | tri-state override for sending Anthropic `thinking` + interleaved-thinking beta header |
| `adaptive_thinking` | no | `None` | tri-state override for Anthropic `{"type": "adaptive", "display": "summarized"}` |
| `fallback` | no | `None` | explicit fallback model name (used by `declared_fallback` / `get_fallback_model`) |
| `comment` | no | - | ignored by the loader; `update_models.py` writes `"NEW"` markers here |

A missing required key raises `KeyError` inside `_build_model_info_entry`, which fails the whole
catalog load (a non-bundled catalog then falls back to the bundled one).

## Shape of the shipped catalog (checked with a script)

689 entries. Every entry carries `context_length`, both prices, `fc`, `emb`, `gen`. About 455 set
`use_responses_api`, 303 `cache_read_price_per_1M`, 107 `cache_write_price_per_1M`, 141
`thinking`, 110 `alias_of`, 8 audio prices, 3 each `extended_thinking`/`adaptive_thinking`, 2 `dec`
(`openrouter/typesafe/jev-1.13`, `openrouter/~typesafe/jev-latest`), 2 `fallback`
(`claude-fable-5`, `openrouter/anthropic/claude-fable-5`).

Examples:
```json
"gpt-5.5":        {"context_length": 500000, "input_price_per_1M": 5.0, "output_price_per_1M": 30.0,
                   "fc": true, "emb": false, "gen": true, "thinking": "high", "use_responses_api": true}
"gpt-5.5-xhigh":  {..., "thinking": "xhigh", "use_responses_api": true, "alias_of": "gpt-5.5"}
"cc/opus":        {"context_length": 500000, "input_price_per_1M": 0.0, "output_price_per_1M": 0.0, ...}
"text-embedding-3-small": {"context_length": 8191, "input_price_per_1M": 0.02, "fc": false, "emb": true, "gen": false}
```
`cc/*` and `codex/*` entries are priced at 0 because the CLI subscription pays, not per-token
billing (see `models-cli-backed-cc-codex`).

## Thinking-level aliases

Entries with `alias_of` exist only so a user can pick a `reasoning_effort` level by name.
`_strip_thinking_alias` maps the alias back to the base id before the request is sent
(`_provider_model_name` in `openai_compatible_model.py`) and before every pricing lookup.
`-xhigh` (`_XHIGH_SUFFIX`) is stripped unconditionally (no upstream model ends in it). `-max`,
`-high`, `-medium`, `-low` (`_LEVEL_ALIAS_SUFFIXES`) collide with real upstream names such as
`openrouter/openai/o3-mini-high`, so they are stripped only when the name is exactly a catalog
key carrying `alias_of` (the recorded base key is returned). Pass the full catalog key, including
any `openrouter/` prefix, before removing routing prefixes.
`_sync_alias_transport_flags` copies a user-overridden base's `use_responses_api` onto its
aliases so selecting an alias does not bypass the user's transport choice.

## Sources
- `src/kiss/core/models/model_info.py` (`ModelInfo`, `_build_model_info_entry`, `MY_MODELS_DEFAULT_CONTENT`, `_strip_thinking_alias`, `_sync_alias_transport_flags`)
- `src/kiss/core/models/MODEL_INFO.json`
- `src/kiss/core/models/openai_compatible_model.py` (`_provider_model_name`, `_model_thinking_level`)
