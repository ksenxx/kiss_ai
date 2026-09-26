---
title: LLM cost computation (calculate_cost, token extraction, cache and long-context
  pricing, OpenRouter usage.cost)
uuid: 0cec663b-0bf9-4a31-93e4-152ee0a08a00
summary: 'How KISS bills a model call: token tuples, OpenRouter usage.cost, calculate_cost
  with cache read/write and 1h cache, audio, long-context uplift, per-provider cache
  price defaults.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# LLM cost computation

## Flow per model call
In `KISSAgent` (`src/kiss/core/kiss_agent.py`, usage accounting after each response):
1. `model.extract_input_output_token_counts_from_response(response)` returns a tuple:
   - 4-tuple `(input, output, cache_read, cache_write)`,
   - 5-tuple adding Anthropic 1-hour cache writes, or
   - 7-tuple `(input, output, cache_read, cache_write, cache_write_1h=0, audio_in, audio_out)`
     for OpenAI audio chat.
   `input` must be the **uncached text** input only; each adapter subtracts cached/audio tokens.
2. `model.extract_cost_from_response(response)`: if it returns a number, that is the cost.
   Base `Model` returns `None`; `OpenAICompatibleBase` returns `usage.cost +
   usage.cost_details.upstream_inference_cost` for `openrouter/` models (the actual bill; the
   catalog price is only a headline estimate because OpenRouter routes to upstreams with
   different prices).
3. Otherwise `calculate_cost(model_name, ...)` from `model_info.py`.

## calculate_cost
- Looks up the entry with `_lookup_model_info` (raw name, then harbor-stripped).
- Unknown model with positive tokens -> `KISSError("Cannot calculate budget for unknown model ...")`;
  zero tokens -> 0.0. So a model used by an agent must be in `MODEL_INFO`/`MY_MODELS.json`.
- Missing cache prices default to the full input price (conservative over-estimate);
  missing 1h-write price defaults to the 5m write price; missing audio prices default to text.
- Long-context uplift (`_long_context_uplift`): when prompt tokens (everything except output)
  exceed the threshold, the **whole request** is billed at multiplied rates:
  `gpt-6-*`, `gpt-5.6-sol/terra/luna`, rolling `gpt-*-latest`, `gpt-5.5` (non-pro), `gpt-5.4`
  (non pro/mini/nano): threshold 272,000, input x2.0, output x1.5; `gemini-3-pro`,
  `gemini-3.1-pro`, `gemini-2.5-pro`: 200,000, x2.0/x1.5. Multipliers (not absolute prices) so
  OpenRouter passthrough entries scale from their own catalog prices.
- Result = sum(tokens x price per 1M) / 1,000,000.

## Cache price defaults (`_apply_cache_pricing`, run over MODEL_INFO at import)
`_provider_cache_defaults(name, input_price)` returns `(read, write_5m, write_1h)`:
- Anthropic (`claude-`, `openrouter/anthropic/`, `openrouter/~anthropic/`): read 0.1x (Fable 5.1 /
  Mythos 5.1: 0.025x; Opus 5.5: 0.05x via `_ANTHROPIC_CACHE_READ_MULTIPLIERS`), write 1.25x,
  1-hour write 2.0x.
- OpenAI (direct and `openrouter/openai/`): read multiplier from `_openai_cache_read_multiplier`
  (GPT-5.x/6.x 0.10, GPT-4.1 and o3/o4 0.25, GPT-4o/o1/o3-mini 0.50, `-pro` 1.0); writes are free
  except GPT-5.6+/GPT-6 and rolling latest aliases, billed 1.25x (`_openai_charges_cache_writes`).
- Gemini (`gemini-`, `openrouter/google/`): read 0.1x, write 0.
- Moonshot (`kimi-`, `moonshot-`): read 0.25x, write 1.0x (writes are excluded from the uncached
  remainder, so a 0 price would give them away).
- Others: no defaults -> full input price.
An explicit catalog `cache_read_price_per_1M` means "published prices": then only the missing
Anthropic 1h tier is derived; an omitted write price stays `None` (billed at input price).
Entries with `gen: false` are skipped.

## Where token counts come from per adapter
- Anthropic: `usage.input_tokens`, `output_tokens`, `cache_read_input_tokens`, and
  `cache_creation_tokens()` which splits `cache_creation.ephemeral_5m/1h_input_tokens`, or puts an
  aggregate `cache_creation_input_tokens` in the 1h bucket (never under-bills). Claude Code CLI
  reuses this parser.
- OpenAI Chat Completions: `prompt_tokens - cached_tokens - cache_write_tokens - audio_tokens`;
  reasoning tokens are inside `completion_tokens` (counted as output). If the turn went through the
  Responses delegate, the delegate's extractor handles `usage.input_tokens`.
- Gemini: input = `prompt_token_count - cached_content_token_count` + `tool_use_prompt_token_count`;
  output = `candidates_token_count + thoughts_token_count`; cache write 0.
- CLI models: see `models-cli-backed-cc-codex` (catalog prices are 0).

## Gotchas
- Thinking-level aliases are priced via their own entry (they copy the base prices); cache and
  uplift rules strip the alias first (`_strip_thinking_alias`).
- `sorcar/decide_tool.py` bills decisions calls with `calculate_cost(model, input, output)`.

## Sources
- `src/kiss/core/models/model_info.py` (`calculate_cost`, `_long_context_uplift`, `_provider_cache_defaults`, `_apply_cache_pricing`, `_openai_cache_read_multiplier`, `_openai_charges_cache_writes`, `_lookup_model_info`)
- `src/kiss/core/models/model.py` (`Model.extract_input_output_token_counts_from_response`, `Model.extract_cost_from_response`)
- `src/kiss/core/models/openai_compatible_model.py` (`OpenAICompatibleBase.extract_cost_from_response`, `OpenAICompatibleModel.extract_input_output_token_counts_from_response`)
- `src/kiss/core/models/anthropic_model.py` (`cache_creation_tokens`, `AnthropicModel.extract_input_output_token_counts_from_response`)
- `src/kiss/core/models/gemini_model.py` (`extract_input_output_token_counts_from_response`)
- `src/kiss/core/kiss_agent.py` (usage accounting calling `extract_cost_from_response` / `calculate_cost`)
