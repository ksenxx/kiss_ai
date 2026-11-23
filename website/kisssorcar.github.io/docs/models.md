# Models Supported by KISS Sorcar

> KISS Sorcar ships a catalog of **993 models** across **7 provider categories**, with built-in prices (including `image_output_price_per_1M`), context lengths, and capability flags (`fc` function calling, `gen` generation, `emb` embedding, `dec` typed decisions via OpenRouter's `/api/alpha/decisions`).

The machine-readable source of truth is [`src/kiss/core/models/MODEL_INFO.json`](https://raw.githubusercontent.com/ksenxx/kiss_ai/main/src/kiss/core/models/MODEL_INFO.json) in the source repository. Models are grouped by the provider that routes them (i.e., whose API key or CLI serves the model).

## Provider Categories

| Provider category | Catalog entries |
|---|---:|
| OpenAI | 99 |
| Anthropic | 82 |
| Amazon Bedrock (`bedrock/*`) | 144 |
| Gemini | 61 |
| OpenRouter | 515 |
| Claude Code CLI (`cc/*`) | 79 |
| Codex CLI (`codex/*`) | 13 |

## Capability Totals

- **941** generation-capable models
- **807** function-calling-capable models
- **6** embedding models
- **38** decision models

## Configuring Model Access

Set one or more environment variables:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export OPENROUTER_API_KEY=...
```

Claude models can also run through Amazon Bedrock: use the model name `bedrock/<Bedrock model or inference-profile id>` (for example `bedrock/us.anthropic.claude-opus-4-6-v1:0`) and set `AWS_BEARER_TOKEN_BEDROCK`, or `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (+ `AWS_SESSION_TOKEN` for temporary credentials), plus `AWS_REGION` (default `us-east-1`). Such a model is priced as its `claude-*` catalog entry; add it under Settings → Custom Models (name only) to pick it in the web app.

Or point at any OpenAI-compatible local/self-hosted endpoint by setting a custom model endpoint and headers in the Settings panel of the web app (e.g. `http://localhost:8000/v1` with an `Authorization: Bearer xxx` header).

## Model Namespaces

- Plain names (e.g. `claude-sonnet-5-5`, `gpt-6.1-sol-medium`) map to the native provider APIs.
- `openrouter/...` routes through OpenRouter (515 entries, including `openrouter/~vendor/model-latest` aliases that always track the newest model).
- `cc/haiku`, `cc/sonnet`, `cc/opus` run on top of the Claude Code CLI.
- `codex/...` (e.g. `codex/gpt-5.6-sol`) run on top of the Codex CLI.

## Multi-Model Workflows

KISS Sorcar can mix models from multiple vendors within a single task, purely via the prompt:

```
Use claude-fable-5 model for all tasks including software development.
Use gpt-5.6-sol (not codex) for thorough review and debugging of the work
done by the other model.
```

A running agent can also switch its own model mid-task with the `set_model` tool, and you can steer running agents by injecting messages on the fly.

## Effort Aliases

Reasoning-capable models are listed once per accepted level on each vendor's effort ladder:

- **OpenAI & Codex:** `-low`, `-medium`, `-high`, `-xhigh`.
- **Claude (direct, Bedrock, `cc/claude-*`) & Moonshot/Kimi:** up to `-max` (`cc/opus`, `cc/sonnet`, and `cc/haiku` carry no effort aliases).
- **Gemini (`gemini-*`):** `-low`, `-medium`, `-high` mapped to `thinking_config.thinking_level` with `include_thoughts` enabled unless disabled in `model_config`.

## Cost and Budget Tracking

A call whose adapter reports a billed cost (such as OpenRouter or API-key-billed `cc/*`/`codex/*` turns) is billed that amount directly; every other call is priced from the catalog's input, cache, output, and image-output rates scaled by context tier.

## Choosing a Default

When no model is selected, KISS Sorcar defaults to the first configured provider in priority order: Anthropic (`claude-opus-5-5`/`claude-sonnet-5-5`), OpenAI (`gpt-6.1-sol-medium`/`gpt-6-luna`), Gemini (`gemini-3.8-flash`/`gemini-3.5-flash-lite`), OpenRouter, Bedrock, Claude Code CLI (`cc/opus`/`cc/haiku`), and Codex CLI (`codex/default`).

The full per-model list is in [MODELS.md](https://github.com/ksenxx/kiss_ai/blob/main/MODELS.md) in the repository.
