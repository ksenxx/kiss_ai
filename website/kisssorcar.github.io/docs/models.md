# Models Supported by KISS Sorcar

> KISS Sorcar ships a catalog of **634 models** across **9 provider categories**, with built-in prices, context lengths, and capability flags (`fc` function calling, `gen` generation, `emb` embedding).

The machine-readable source of truth is [`src/kiss/core/models/MODEL_INFO.json`](https://raw.githubusercontent.com/ksenxx/kiss_ai/main/src/kiss/core/models/MODEL_INFO.json) in the source repository.

## Provider Categories

| Provider category | Catalog entries |
|---|---:|
| OpenAI | 105 |
| Anthropic | 14 |
| Gemini / Google | 27 |
| Together AI | 91 |
| Z.AI | 8 |
| Moonshot AI | 10 |
| OpenRouter | 356 |
| Claude Code CLI (`cc/*`) | 14 |
| Codex CLI (`codex/*`) | 9 |

## Capability Totals

- **614** generation-capable models
- **455** function-calling-capable models
- **11** embedding models

## Configuring Model Access

Set one or more environment variables:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export ZAI_API_KEY=...
export MOONSHOT_API_KEY=...
export TOGETHER_API_KEY=...
export OPENROUTER_API_KEY=...
export GEMINI_API_KEY=...
```

Or point at any OpenAI-compatible local/self-hosted endpoint by setting a custom model endpoint and headers in the Settings panel of the VS Code extension or web app (e.g. `http://localhost:8000/v1` with an `Authorization: Bearer xxx` header).

## Model Namespaces

- Plain names (e.g. `claude-sonnet-4-6`, `gpt-4.1`) map to the native provider APIs.
- `openrouter/...` routes through OpenRouter (356 entries, including `openrouter/~vendor/model-latest` aliases that always track the newest model).
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

## Choosing a Default

When no model is selected, KISS Sorcar defaults to the best available model for the API keys you have configured.

The full per-model list (all 634 entries) is in the [project README](https://github.com/ksenxx/kiss_ai#models-supported).
