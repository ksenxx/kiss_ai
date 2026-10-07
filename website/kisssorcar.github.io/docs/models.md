# Models Supported by KISS Sorcar

> KISS Sorcar ships a catalog of **715 models** across **9 provider categories**, with built-in prices, context lengths, and capability flags (`fc` function calling, `gen` generation, `emb` embedding, `dec` typed decisions via OpenRouter's `/api/alpha/decisions`).

The machine-readable source of truth is [`src/kiss/core/models/MODEL_INFO.json`](https://raw.githubusercontent.com/ksenxx/kiss_ai/main/src/kiss/core/models/MODEL_INFO.json) in the source repository. Models are grouped by the provider that routes them (i.e., whose API key or CLI serves the model); open-weight `openai/gpt-oss-*` and `google/gemma-*` models are served via Together AI.

## Provider Categories

| Provider category | Catalog entries |
|---|---:|
| OpenAI | 112 |
| Anthropic | 16 |
| Gemini | 20 |
| Together AI | 103 |
| Z.AI | 8 |
| Moonshot AI | 10 |
| OpenRouter | 420 |
| Claude Code CLI (`cc/*`) | 16 |
| Codex CLI (`codex/*`) | 10 |

## Capability Totals

- **691** generation-capable models
- **528** function-calling-capable models
- **7** embedding models
- **9** decision models

## Configuring Model Access

You can connect a Claude Code or Codex subscription, configure API keys, or register a custom endpoint.

**Subscription connection.** Use Claude Code 2.1.280+ or Codex 0.162.0+ for subscription isolation. Install the official CLI on the machine running the Sorcar daemon, then use **Settings → CLI Connections → Sign in** and **Refresh status**. Existing logins are reused. The fixed terminal commands are:

```bash
claude auth login
codex login --device-auth
```

Choose **Subscription** and select `cc/opus` or `codex/gpt-6.1-sol`. Sorcar verifies the login before execution and isolates the child process from API credentials and alternate-provider overrides. Codex subscription runs refuse a configured `openai_base_url` or `openai` provider override. New connections use Subscription mode; detected API or enterprise configurations remain available under **Existing CLI configuration**, with their billing shown in Settings. Remote users sign in on the daemon machine; Windows web users can copy the command into a terminal there.

Subscription tasks keep this billing constraint through retries, children, model switches, task updates, and commit helpers. Optional API classification is skipped. A task that requests an API model under that constraint fails with guidance; start a separate API task to use API credits. Fable is optional and requires the saved **Allow Fable usage credits** setting for any Claude subscription login, including one kept as the existing CLI configuration: noninteractive Claude requests can bill usage credits without prompting. Without that opt-in, native Claude child agents use the selected model, and Claude settings files cannot remap model aliases, add a fallback model, or call an advisor model for that run. Provider plan limits and extra-usage settings still apply.

**API connection.** Configure keys in Settings or export them before starting the daemon:

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

- Plain names (e.g. `claude-opus-5-5`, `gpt-6.1-sol-medium`) map to the native provider APIs.
- `openrouter/...` routes through OpenRouter (420 entries, including `openrouter/~vendor/model-latest` aliases that always track the newest model).
- `cc/haiku`, `cc/sonnet`, `cc/opus` run on top of the Claude Code CLI.
- `codex/...` (e.g. `codex/gpt-6.1-sol`) run on top of the Codex CLI.

## Multi-Model Workflows

API tasks can mix vendors when their keys are configured:

```text
Use claude-opus-5-5 for implementation and gpt-6.1-sol-medium for a read-only review.
Report only reproducible defects with file and line evidence.
```

Subscription tasks use CLI models throughout. Select `cc/opus` for Claude or `codex/gpt-6.1-sol` for Codex. Do not ask a subscription task to switch to an API model. Fable requires a saved usage-credit opt-in.

## Choosing a Default

Automatic task selection prefers verified Claude subscriptions, then Codex subscriptions, then configured API providers. Valid saved and explicit selections are retained. API defaults are `claude-opus-5-5`, `gpt-6.1-sol-medium`, and `gemini-3.8-flash`. Outside subscription tasks, lightweight helpers keep using configured API keys first: Haiku 4.5, GPT-6 Luna, and Gemini 3.5 Flash-Lite. Inside a subscription task they use `cc/haiku` or `codex/gpt-6-luna`.

GPT-6.1 Sol tool calls use the Responses API. Its effort aliases include `low`, `medium`, `high`, `xhigh`, and `max`; CLI callers can pass `model_config={"reasoning_effort": "max"}`. Older IDs and aliases remain available. Sorcar retains its intentional context-window cap.

## Gemini 4 Argon

Argon is in a limited rollout through Google's Fairwind program. It is absent from the public Gemini API model catalog, so bundled integration is deferred until a documented API identifier and access contract are available. See [Google's announcement](https://blog.google/intl/en-in/products/gemini-4-argon-our-next-era-of-frontier-intelligence/) and the [public API catalog](https://ai.google.dev/gemini-api/docs/models).

The full per-model list is in [MODELS.md](https://github.com/ksenxx/kiss_ai/blob/main/MODELS.md) in the repository.
