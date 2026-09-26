---
title: Model provider API keys and credentials (env vars, api_keys.env, workspace
  id, CLI logins)
uuid: cb5913f8-34df-44d9-a9a4-6279d64c8383
summary: Provider API key env vars (ANTHROPIC, OPENAI, GEMINI, OPENROUTER, TOGETHER,
  ZAI, MOONSHOT), ANTHROPIC_WORKSPACE_ID, config.DEFAULT_CONFIG, api_keys.env, per-model
  api_key, CLI logins.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Model provider API keys and credentials

## Where keys come from
Keys are fields on `config.DEFAULT_CONFIG` (pydantic settings in `src/kiss/core/config.py`) whose
`default_factory` reads the environment:

| Config attribute / env var | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | `AnthropicModel` (`claude-*`) |
| `ANTHROPIC_WORKSPACE_ID` | `AnthropicModel.initialize` reads `os.environ` directly and sends header `anthropic-workspace-id` (needed for identity-linked keys) |
| `OPENAI_API_KEY` | OpenAI provider (`gpt*`, `o1/o3/o4`, `text-embedding*`, `codex*` not `codex/`); also Whisper audio transcription (`transcribe_audio` in `model.py`) |
| `GEMINI_API_KEY` | `GeminiModel` (`gemini-*`) |
| `OPENROUTER_API_KEY` | `openrouter/*` and the decisions model (`_decisions_model`) |
| `TOGETHER_API_KEY` | Together prefixes (`meta-llama/`, `Qwen/`, `deepseek-ai/`, ...) |
| `ZAI_API_KEY` | `glm-*` |
| `MOONSHOT_API_KEY` | `kimi-*`, `moonshot-*` |

For OpenAI-compatible vendors the mapping is data, not code: `OpenAICompatibleProvider.api_key_name`
names the `DEFAULT_CONFIG` attribute and `model()` does `getattr(keys, provider.api_key_name)`.
Adding a vendor with a new key therefore also needs a new field in `config.py`.

`kiss.core.vscode_config.load_api_keys()` imports `$KISS_HOME/api_keys.env` into `os.environ` at
daemon startup (the settings panel writes that file). Because `DEFAULT_CONFIG` fields are
evaluated at construction, code that changes keys at runtime goes through the config/env update
path in `vscode_config` rather than setting the env var alone.

## Per-model overrides
- `model_config={"base_url": ..., "api_key": ...}` bypasses routing and uses that key
  (see `models-provider-resolution`).
- Custom models in `~/.kiss/MY_MODELS.json` may carry `endpoint`, `api_key`, `headers`;
  `custom_model_config()` converts them to `base_url` / `api_key` / `extra_headers`.
- The decisions model accepts `model_config["api_key"]` / `["base_url"]`, else OpenRouter defaults.

## CLI-backed models have no API key
`cc/*` needs the `claude` executable on `PATH` (logged-in Claude Code subscription) and `codex/*`
needs a Codex executable found by `find_codex_executable()`. `_NATIVE_PROVIDERS` records their key as
`None`; `_configured_providers` checks `shutil.which("claude")` / `find_codex_executable()`
instead. See `models-cli-backed-cc-codex`.

## Anthropic workspace id error
When the API answers with a message containing `anthropic-workspace-id is required`
(`_WORKSPACE_ID_REQUIRED_MARKER`), `AnthropicModel._stream_message` raises a `KISSError` built from
`_WORKSPACE_ID_HINT` telling the user to set `ANTHROPIC_WORKSPACE_ID` (settings panel or
`export ANTHROPIC_WORKSPACE_ID=wrkspc_...`). `KISSError` is not retried by the agent loop, so the
run fails fast with the hint. The client is rebuilt only when `(api_key, workspace_id)` changes
(`_client_inputs`).

## Model availability depends on keys
`get_available_models()` only lists models whose provider credential is present, and
`get_default_model()` picks the first configured provider (Anthropic, OpenAI, Gemini, OpenRouter,
Together, then CLI). With nothing configured the default is the literal string `"No model"`.

## Sources
- `src/kiss/core/config.py` (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`, `ZAI_API_KEY`, `MOONSHOT_API_KEY` fields)
- `src/kiss/core/models/model_info.py` (`OPENAI_COMPATIBLE_PROVIDERS`, `_NATIVE_PROVIDERS`, `_configured_providers`, `model`, `_decisions_model`, `custom_model_config`)
- `src/kiss/core/models/anthropic_model.py` (`AnthropicModel.initialize`, `_WORKSPACE_ID_REQUIRED_MARKER`, `_WORKSPACE_ID_HINT`)
- `src/kiss/core/models/model.py` (`transcribe_audio`)
- `src/kiss/core/vscode_config.py` (`load_api_keys`)
