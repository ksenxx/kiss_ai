<div align="center">

![KISS Framework](assets/KISS-Sorcar.png)

[![Version](https://img.shields.io/badge/version-2026.6.26-blue?style=flat-square)](https://pypi.org/project/kiss-agent-framework/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-blue?style=flat-square)](https://www.python.org/)
[![Website](https://img.shields.io/badge/website-kisssorcar.github.io-1976d2?style=flat-square)](https://kisssorcar.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.23822-b31b1b?style=flat-square)](https://arxiv.org/abs/2604.23822)

*"Everything should be made as simple as possible, but not simpler." — Albert Einstein*

</div>

# KISS Sorcar

### Open-source general-purpose AI agent for long-horizon tasks and AI discovery

**KISS Sorcar is free, simple, local-first, bring-your-own-key AI agent framework.** It runs as a VS Code extension, a Claude-Code-style CLI, and a browser/mobile web app. Your prompts and code are sent directly to the model provider or local endpoint you configure — not through our servers. It supports multi-model workflow just via prompt. All agents run as daemon.

```bash
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash
```

______________________________________________________________________

<details>
<summary><strong>Table of Contents</strong></summary>

- [KISS Sorcar vs Claude Code vs Cursor](#-kiss-sorcar-vs-claude-code-vs-cursor)
- [What is in the Name](#what-is-in-the-name)
- [Installation](#installation)
  - [Full install from source](#full-install-from-source)
  - [Python package / CLI install](#python-package--cli-install)
  - [Configure model access](#configure-model-access)
  - [VS Code Extension Installation](#vs-code-extension-installation)
- [CLI Interface](#cli-interface)
  - [CLI options](#cli-options)
  - [Interactive CLI features](#interactive-cli-features)
  - [`sorcar mcp` subcommand](#sorcar-mcp-subcommand)
- [Messaging & Third-Party Agents](#-messaging--third-party-agents)
- [Models Supported](#-models-supported)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)

</details>

## 🆚 KISS Sorcar vs Claude Code vs Cursor

| Capability | **KISS Sorcar** | **Claude Code** | **Cursor** |
|---|---|---|---|
| **Interfaces** | CLI + VS Code extension + web/mobile app | CLI + mobile app | Custom VS Code |
| **Multiple models from multiple vendors in the same task** | ✅ Mix OpenAI, Anthropic, Gemini, Together, MiniMax, OpenRouter, Claude Code CLI, and Codex CLI | ❌ Anthropic Claude models only | ❌ One model per task |
| **Primary focus** | ✅ **Quality** — rigorous review, end-to-end tests | Speed and developer ergonomics | Speed |
| **Models in bundled catalog** | 501 across 8 provider categories | Claude family only | Subset chosen by Cursor |
| **Bring your own API key / endpoint** | ✅ Yes — keys stay on your machine | ✅ Anthropic key | ⚠️ Routed through Cursor backend |
| **Open source** | ✅ Apache-2.0 | ❌ Proprietary | ❌ Proprietary |
| **Price** | Free framework; pay only your chosen model provider | Subscription / API usage | Subscription |
| **Run on top of Claude Code / Codex CLI** | ✅ `cc/*` and `codex/*` namespaces | N/A | ❌ |
| **Messaging and communication channels** | ✅ 23 third-party agents, including Slack, Gmail, Phone Control, SMS, and WhatsApp | ⚠️ Slack, mobile Remote Control, and research-preview channels for Telegram, Discord, and iMessage; no documented built-in Gmail, WhatsApp, phone-call, or SMS channel | ⚠️ Slack and Microsoft Teams Cloud Agent integrations; no documented built-in Gmail, WhatsApp, phone-call, or SMS channel |
| **Terminal Bench 2.0 score** | **62.2%** | 58% | 61.7% (Cursor agent) |

## What is in the Name

**KISS Agent Framework** is a deliberately small agent runtime organized around the [KISS principle](https://en.wikipedia.org/wiki/KISS_principle) ("Keep it Simple, Stupid").
KISS Sorcar is named after [P. C. Sorcar, the Bengali magician](https://en.wikipedia.org/wiki/P._C._Sorcar).
Note: **Sorcar** also means government in Bengali.

## Installation

### Full install from source

```bash
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash
```

The installer targets macOS and Linux on `x86_64`, `aarch64`, and `arm64`. It installs or checks the tools needed to run KISS Sorcar and build/install the VS Code extension.

### Python package / CLI install

If you only want the Python package and CLI entry points:

```bash
pipx install kiss-agent-framework
# or
uv tool install kiss-agent-framework
```

KISS Sorcar requires **Python 3.13+**.

### Configure model access

Provide at least one model backend. You can use environment variables such as:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export TOGETHER_API_KEY=...
export OPENROUTER_API_KEY=...
export MINIMAX_API_KEY=...
```

You can also configure a custom endpoint with `--endpoint` / `-e` and optional repeated `--header Key:Value` CLI flags.

### VS Code Extension Installation

To install only the KISS Sorcar extension, open Visual Studio Code, search for **KISS Sorcar** in the extension marketplace, install it, and relaunch VS Code. Press ESC if you do not have a specific API key ready, but configure at least one model backend before running tasks.

## CLI Interface

`sorcar` runs in two modes:

- **Interactive** (no `-t/--task` or `-f/--file`) — a Claude-Code-style REPL that connects as a thin terminal client to the local `sorcar web` daemon. Chat-session control (new chat, resume by id, list history) and worktree merge/discard prompts are driven from slash commands. Each task is isolated in a git worktree by default.
- **Non-interactive** (`-t` or `-f` supplied) — runs a plain `SorcarAgent` once on the supplied task and exits. Worktree isolation and chat-session control are unavailable in this mode; display events are still streamed into the chat DB so the run is replayable in the chat webview.

```bash
# Launch the interactive Sorcar CLI, similar to Claude Code.
sorcar

# Run a one-shot task (non-interactive).
sorcar -t "What is 2435*234?"

# Use a specific model.
sorcar -m "claude-sonnet-4-6" -t "What is 2435*234?"

# Custom endpoint and headers for a local or self-hosted model.
sorcar -e "http://localhost:8000/v1" --header "Authorization:Bearer xxx" \
       -t "Summarise this codebase."

# Cap spend at $2 and pin the working directory.
sorcar -b 2.0 -w "$HOME/projects/my-repo" -t "Refactor utils.py for clarity."

# Use the contents of a file as the task.
echo "Can you find the cheapest non-stop flight from SFO to JFK on June 15?" > prompt
sorcar -f prompt

# Disable browser/web tools (terminal-only mode).
sorcar --no-web -t "Lint and fix every Python file under src/."

# Disable parallel sub-agents for a deterministic single-thread run.
sorcar --no-parallel -t 'Run pytest and report which tests fail and why.'

# Ask Sorcar to use desktop/browser/messaging tools.
sorcar -t 'Can you send the message "Hello from Sorcar!" to ksen via the desktop Slack app?'

# Ask Sorcar to explain code.
sorcar -t 'Can you show me the detailed step-by-step workflow of gepa.py?'

# Manage MCP servers (see "sorcar mcp" subcommand below).
sorcar mcp list --ping
```

### CLI options

| Flag | Description |
|------|-------------|
| `-t`, `--task` | Task description; switches to non-interactive mode |
| `-f`, `--file` | Path to a file whose contents are used as the task; switches to non-interactive mode |
| `-m`, `--model_name` | LLM model name; defaults to the best available model for the configured API keys |
| `-e`, `--endpoint` | Custom base URL for a local or self-hosted model |
| `--header` | Custom HTTP header in `Key:Value` form; may be repeated |
| `-b`, `--max_budget` | Maximum spend in USD for the run |
| `-w`, `--work_dir` | Working directory; defaults to the directory where `sorcar` is launched |
| `-v`, `--verbose` | Print Rich panels to the console (`true` by default; pass `false` for quiet mode) |
| `-p`, `--parallel` / `--no-parallel` | Enable/disable parallel sub-agents (enabled by default) |
| `--worktree` / `--no-worktree` | **Interactive only.** Isolate each task in a git worktree branch (enabled by default); use `--no-worktree` to run directly in the working tree |
| `--auto-commit` / `--no-auto-commit` | **Interactive only.** Auto-commit worktree changes when a task finishes (enabled by default); use `--no-auto-commit` to preserve the worktree for manual review |
| `--no-web` | Disable browser/web tools (terminal-only mode) |

`--worktree` / `--no-worktree` / `--auto-commit` / `--no-auto-commit` are rejected with `exit 2` when combined with `-t`/`-f`, since the non-interactive path runs a bare `SorcarAgent` that does not implement them. Argparse prefix abbreviations are disabled, so each flag must be spelled out in full.

### Interactive CLI features

The interactive CLI includes:

- `@` file/folder mentions with ranked project-file completion.
- Slash commands: `/help`, `/clear` (alias `/new`), `/resume`, `/model`, `/model list`, `/cost` (aliases `/usage`, `/context`), `/commands`, `/skills`, `/mcp`, `/autocommit`, and `/exit` (alias `/quit`).
- Custom Markdown slash commands loaded from `~/.kiss/commands`, `<project>/.kiss/commands`, `~/.claude/commands`, and `<project>/.claude/commands`.
- Agent Skills loaded from `~/.kiss/skills`, `<project>/.kiss/skills`, Claude skill directories, `.agents/skills`, and bundled Sorcar skills.
- MCP server discovery from `~/.kiss/mcp.json`, `<project>/.kiss/mcp.json`, and `<project>/.mcp.json`.
- VS Code "Tricks" button entries read from `~/.kiss/INJECTIONS.md` (one per `## Trick` section), seeded on install from the bundled `src/kiss/INJECTIONS.md`. Edit the file to customise the dropdown; remove it to regenerate from the bundled defaults.
- VS Code welcome-screen sample-task chips read from `~/.kiss/SAMPLE_TASKS.md` (one per `## Task` section), seeded on install from the bundled `src/kiss/agents/vscode/SAMPLE_TASKS.md`. Edit the file to customise the chips; remove it to regenerate from the bundled defaults.

### `sorcar mcp` subcommand

Manage Model-Context-Protocol servers used by Sorcar:

| Subcommand | Purpose |
|---|---|
| `sorcar mcp add <name> <cmd…>` | Register a stdio (default) or `--transport http`/`sse` server in `--scope user` (`~/.kiss/mcp.json`) or `--scope project` (`<work_dir>/.kiss/mcp.json`); supports `--env KEY=VALUE` and `--header 'Key: Value'` (repeatable). |
| `sorcar mcp list [--ping]` | List configured servers; `--ping` also connects and reports live status and tool counts. |
| `sorcar mcp get <name>` | Print one server's configuration as JSON. |
| `sorcar mcp remove <name>` | Delete a server from every writable config file. |
| `sorcar mcp auth <name> [--no-browser]` | Run the OAuth 2.1 browser flow (dynamic client registration + PKCE) and persist tokens under `~/.kiss/mcp_auth/`. |
| `sorcar mcp logout <name>` | Delete a server's stored OAuth tokens. |
| `sorcar mcp debug <name>` | Connect and dump capabilities, tools (with input schemas and permission status), resources, and prompts. |

## 💬 Messaging & Third-Party Agents

KISS Sorcar includes 23 third-party messaging agents that can send and receive messages on your behalf:

BlueBubbles · Discord · Feishu · Gmail · Google Chat · iMessage · IRC · LINE · Matrix · Mattermost · Microsoft Teams · Nextcloud Talk · Nostr · Phone Control · Signal · Slack · SMS · Synology Chat · Telegram · Tlon · Twitch · WhatsApp · Zalo

It also ships a **Govee smart-home CLI** for controlling IoT lights (on/off, brightness, color, and color temperature) via the Govee Developer API.

These agents live in `src/kiss/agents/third_party_agents/`.

## 🤖 Models Supported

KISS Sorcar ships a catalog of **501 models** across **8 provider categories**, with built-in prices, context lengths, and capability flags (`fc` function calling, `gen` generation, `emb` embedding). The source of truth is [src/kiss/core/models/MODEL_INFO.json](src/kiss/core/models/MODEL_INFO.json).

| Provider category | Catalog entries |
|---|---:|
| OpenAI | 70 |
| Anthropic | 13 |
| Gemini / Google | 23 |
| Together AI | 77 |
| MiniMax | 5 |
| OpenRouter | 303 |
| Claude Code CLI (`cc/*`) | 3 |
| Codex CLI (`codex/*`) | 7 |

Current catalog capability totals:

- **485** generation-capable models
- **321** function-calling-capable models
- **7** embedding models

Full model list:

<details>
<summary><strong>OpenAI (70)</strong></summary>

- `computer-use-preview`
- `computer-use-preview-2025-03-11`
- `gpt-3.5-turbo`
- `gpt-3.5-turbo-0125`
- `gpt-3.5-turbo-1106`
- `gpt-3.5-turbo-16k`
- `gpt-4`
- `gpt-4-0613`
- `gpt-4-turbo`
- `gpt-4-turbo-2024-04-09`
- `gpt-4.1`
- `gpt-4.1-2025-04-14`
- `gpt-4.1-mini`
- `gpt-4.1-mini-2025-04-14`
- `gpt-4.1-nano`
- `gpt-4.1-nano-2025-04-14`
- `gpt-4o`
- `gpt-4o-2024-05-13`
- `gpt-4o-2024-08-06`
- `gpt-4o-2024-11-20`
- `gpt-4o-mini`
- `gpt-4o-mini-2024-07-18`
- `gpt-4o-mini-search-preview`
- `gpt-4o-mini-search-preview-2025-03-11`
- `gpt-4o-search-preview`
- `gpt-4o-search-preview-2025-03-11`
- `gpt-5`
- `gpt-5-2025-08-07`
- `gpt-5-chat-latest`
- `gpt-5-mini`
- `gpt-5-mini-2025-08-07`
- `gpt-5-nano`
- `gpt-5-nano-2025-08-07`
- `gpt-5.1`
- `gpt-5.1-2025-11-13`
- `gpt-5.1-chat-latest`
- `gpt-5.2`
- `gpt-5.2-2025-12-11`
- `gpt-5.2-chat-latest`
- `gpt-5.3-chat-latest`
- `gpt-5.4`
- `gpt-5.4-2026-03-05`
- `gpt-5.4-mini`
- `gpt-5.4-mini-2026-03-17`
- `gpt-5.4-nano`
- `gpt-5.4-nano-2026-03-17`
- `gpt-5.5`
- `gpt-5.5-2026-04-23`
- `gpt-image-1`
- `gpt-image-1-mini`
- `gpt-image-1.5`
- `gpt-image-2`
- `gpt-image-2-2026-04-21`
- `o1`
- `o1-2024-12-17`
- `o3`
- `o3-2025-04-16`
- `o3-deep-research`
- `o3-deep-research-2025-06-26`
- `o3-mini`
- `o3-mini-2025-01-31`
- `o4-mini`
- `o4-mini-2025-04-16`
- `o4-mini-deep-research`
- `o4-mini-deep-research-2025-06-26`
- `openai/gpt-oss-120b`
- `openai/gpt-oss-20b`
- `text-embedding-3-large`
- `text-embedding-3-small`
- `text-embedding-ada-002`

</details>

<details>
<summary><strong>Anthropic (13)</strong></summary>

- `claude-fable-5`
- `claude-haiku-4-5`
- `claude-haiku-4-5-20251001`
- `claude-opus-4-1`
- `claude-opus-4-1-20250805`
- `claude-opus-4-5`
- `claude-opus-4-5-20251101`
- `claude-opus-4-6`
- `claude-opus-4-7`
- `claude-opus-4-8`
- `claude-sonnet-4-5`
- `claude-sonnet-4-5-20250929`
- `claude-sonnet-4-6`

</details>

<details>
<summary><strong>Gemini / Google (23)</strong></summary>

- `gemini-2.0-flash`
- `gemini-2.0-flash-001`
- `gemini-2.0-flash-lite`
- `gemini-2.0-flash-lite-001`
- `gemini-2.5-flash`
- `gemini-2.5-flash-image`
- `gemini-2.5-flash-lite`
- `gemini-2.5-pro`
- `gemini-3-flash-preview`
- `gemini-3-pro-image`
- `gemini-3-pro-preview`
- `gemini-3.1-flash-image`
- `gemini-3.1-flash-lite`
- `gemini-3.1-flash-lite-preview`
- `gemini-3.1-flash-tts-preview`
- `gemini-3.1-pro-preview`
- `gemini-3.5-flash`
- `gemini-embedding-001`
- `gemini-embedding-2`
- `gemini-embedding-2-preview`
- `google/gemma-2-27b-it`
- `google/gemma-3n-E4B-it`
- `google/gemma-4-31B-it`

</details>

<details>
<summary><strong>Together AI (77)</strong></summary>

- `arcee-ai/trinity-mini`
- `BAAI/bge-base-en-v1.5`
- `deepcogito/cogito-v1-preview-llama-70B`
- `deepcogito/cogito-v1-preview-llama-70B-Turbo`
- `deepcogito/cogito-v1-preview-llama-8B`
- `deepcogito/cogito-v1-preview-qwen-14B`
- `deepcogito/cogito-v1-preview-qwen-32B`
- `deepcogito/cogito-v2-1-671b`
- `deepseek-ai/deepseek-coder-33b-instruct`
- `deepseek-ai/DeepSeek-R1`
- `deepseek-ai/DeepSeek-R1-0528`
- `deepseek-ai/DeepSeek-R1-0528-tput`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- `deepseek-ai/DeepSeek-V3-0324`
- `deepseek-ai/DeepSeek-V3.1`
- `deepseek-ai/DeepSeek-V4-Pro`
- `essentialai/rnj-1-instruct`
- `intfloat/multilingual-e5-large-instruct`
- `meta-llama/Llama-3-70b-chat-hf`
- `meta-llama/Llama-3-8b-chat-hf`
- `meta-llama/Llama-3.1-405B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct-Turbo`
- `meta-llama/Llama-3.3-70B-Instruct-Turbo`
- `meta-llama/Llama-3.3-70B-Instruct-Turbo-test`
- `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`
- `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- `meta-llama/Meta-Llama-3-70B-Instruct-Turbo`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Meta-Llama-3-8B-Instruct-Lite`
- `meta-llama/Meta-Llama-3.1-70B-Instruct-Reference`
- `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`
- `meta-llama/Meta-Llama-3.1-8B-Instruct-Reference`
- `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`
- `mistralai/Ministral-3-14B-Instruct-2512`
- `mistralai/Mistral-7B-Instruct-v0.1`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `mistralai/Mistral-Small-24B-Instruct-2501`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `moonshotai/Kimi-K2-Instruct`
- `moonshotai/Kimi-K2-Instruct-0905`
- `moonshotai/Kimi-K2-Thinking`
- `moonshotai/Kimi-K2.5`
- `moonshotai/Kimi-K2.6`
- `moonshotai/Kimi-K2.7-Code`
- `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF`
- `nvidia/nemotron-3-ultra-550b-a55b`
- `nvidia/NVIDIA-Nemotron-Nano-9B-v2`
- `Qwen/Qwen2-1.5B-Instruct`
- `Qwen/Qwen2-VL-72B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `Qwen/Qwen2.5-72B-Instruct`
- `Qwen/Qwen2.5-72B-Instruct-Turbo`
- `Qwen/Qwen2.5-7B-Instruct-Turbo`
- `Qwen/Qwen2.5-Coder-32B-Instruct`
- `Qwen/Qwen2.5-VL-72B-Instruct`
- `Qwen/Qwen3-235B-A22B-Instruct-2507-tput`
- `Qwen/Qwen3-235B-A22B-Thinking-2507`
- `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`
- `Qwen/Qwen3-Coder-Next-FP8`
- `Qwen/Qwen3-Next-80B-A3B-Instruct`
- `Qwen/Qwen3-Next-80B-A3B-Thinking`
- `Qwen/Qwen3-VL-32B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3.5-397B-A17B`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.6-Plus`
- `Qwen/QwQ-32B`
- `zai-org/GLM-4.5-Air-FP8`
- `zai-org/GLM-4.6`
- `zai-org/GLM-4.7`
- `zai-org/GLM-5`
- `zai-org/GLM-5.1`
- `zai-org/GLM-5.2`

</details>

<details>
<summary><strong>MiniMax (5)</strong></summary>

- `minimax-m2.5`
- `minimax-m2.5-lightning`
- `MiniMaxAI/MiniMax-M2.5`
- `MiniMaxAI/MiniMax-M2.7`
- `MiniMaxAI/MiniMax-M3`

</details>

<details>
<summary><strong>OpenRouter (303)</strong></summary>

- `openrouter/ai21/jamba-large-1.7`
- `openrouter/aion-labs/aion-1.0`
- `openrouter/aion-labs/aion-1.0-mini`
- `openrouter/aion-labs/aion-2.0`
- `openrouter/aion-labs/aion-rp-llama-3.1-8b`
- `openrouter/allenai/olmo-3-32b-think`
- `openrouter/amazon/nova-2-lite-v1`
- `openrouter/amazon/nova-lite-v1`
- `openrouter/amazon/nova-micro-v1`
- `openrouter/amazon/nova-premier-v1`
- `openrouter/amazon/nova-pro-v1`
- `openrouter/anthracite-org/magnum-v4-72b`
- `openrouter/anthropic/claude-3-haiku`
- `openrouter/anthropic/claude-3.5-haiku`
- `openrouter/anthropic/claude-3.7-sonnet:thinking`
- `openrouter/anthropic/claude-fable-5`
- `openrouter/anthropic/claude-haiku-4.5`
- `openrouter/anthropic/claude-opus-4`
- `openrouter/anthropic/claude-opus-4.1`
- `openrouter/anthropic/claude-opus-4.5`
- `openrouter/anthropic/claude-opus-4.6`
- `openrouter/anthropic/claude-opus-4.6-fast`
- `openrouter/anthropic/claude-opus-4.7`
- `openrouter/anthropic/claude-opus-4.7-fast`
- `openrouter/anthropic/claude-opus-4.8`
- `openrouter/anthropic/claude-opus-4.8-fast`
- `openrouter/anthropic/claude-sonnet-4`
- `openrouter/anthropic/claude-sonnet-4.5`
- `openrouter/anthropic/claude-sonnet-4.6`
- `openrouter/arcee-ai/coder-large`
- `openrouter/arcee-ai/trinity-large-thinking`
- `openrouter/arcee-ai/trinity-mini`
- `openrouter/arcee-ai/virtuoso-large`
- `openrouter/baidu/ernie-4.5-vl-424b-a47b`
- `openrouter/bytedance-seed/seed-1.6`
- `openrouter/bytedance-seed/seed-1.6-flash`
- `openrouter/bytedance-seed/seed-2.0-lite`
- `openrouter/bytedance-seed/seed-2.0-mini`
- `openrouter/bytedance/ui-tars-1.5-7b`
- `openrouter/cohere/command-a`
- `openrouter/cohere/command-r-08-2024`
- `openrouter/cohere/command-r-plus-08-2024`
- `openrouter/cohere/command-r7b-12-2024`
- `openrouter/deepcogito/cogito-v2.1-671b`
- `openrouter/deepseek/deepseek-chat`
- `openrouter/deepseek/deepseek-chat-v3-0324`
- `openrouter/deepseek/deepseek-chat-v3.1`
- `openrouter/deepseek/deepseek-r1`
- `openrouter/deepseek/deepseek-r1-0528`
- `openrouter/deepseek/deepseek-r1-distill-llama-70b`
- `openrouter/deepseek/deepseek-v3.1-terminus`
- `openrouter/deepseek/deepseek-v3.2`
- `openrouter/deepseek/deepseek-v3.2-exp`
- `openrouter/deepseek/deepseek-v4-flash`
- `openrouter/deepseek/deepseek-v4-pro`
- `openrouter/essentialai/rnj-1-instruct`
- `openrouter/google/gemini-2.5-flash`
- `openrouter/google/gemini-2.5-flash-image`
- `openrouter/google/gemini-2.5-flash-lite`
- `openrouter/google/gemini-2.5-flash-lite-preview-09-2025`
- `openrouter/google/gemini-2.5-pro`
- `openrouter/google/gemini-2.5-pro-preview`
- `openrouter/google/gemini-2.5-pro-preview-05-06`
- `openrouter/google/gemini-3-flash-preview`
- `openrouter/google/gemini-3-pro-image`
- `openrouter/google/gemini-3-pro-image-preview`
- `openrouter/google/gemini-3.1-flash-image`
- `openrouter/google/gemini-3.1-flash-image-preview`
- `openrouter/google/gemini-3.1-flash-lite`
- `openrouter/google/gemini-3.1-flash-lite-preview`
- `openrouter/google/gemini-3.1-pro-preview`
- `openrouter/google/gemini-3.1-pro-preview-customtools`
- `openrouter/google/gemini-3.5-flash`
- `openrouter/google/gemma-2-27b-it`
- `openrouter/google/gemma-3-12b-it`
- `openrouter/google/gemma-3-27b-it`
- `openrouter/google/gemma-3-4b-it`
- `openrouter/google/gemma-3n-e4b-it`
- `openrouter/google/gemma-4-26b-a4b-it`
- `openrouter/google/gemma-4-31b-it`
- `openrouter/google/lyria-3-clip-preview`
- `openrouter/google/lyria-3-pro-preview`
- `openrouter/gryphe/mythomax-l2-13b`
- `openrouter/ibm-granite/granite-4.0-h-micro`
- `openrouter/ibm-granite/granite-4.1-8b`
- `openrouter/inception/mercury-2`
- `openrouter/inclusionai/ling-2.6-1t`
- `openrouter/inclusionai/ling-2.6-flash`
- `openrouter/inclusionai/ring-2.6-1t`
- `openrouter/inflection/inflection-3-pi`
- `openrouter/inflection/inflection-3-productivity`
- `openrouter/kwaipilot/kat-coder-pro-v2`
- `openrouter/liquid/lfm-2-24b-a2b`
- `openrouter/mancer/weaver`
- `openrouter/meta-llama/llama-3-8b-instruct`
- `openrouter/meta-llama/llama-3.1-70b-instruct`
- `openrouter/meta-llama/llama-3.1-8b-instruct`
- `openrouter/meta-llama/llama-3.2-11b-vision-instruct`
- `openrouter/meta-llama/llama-3.2-1b-instruct`
- `openrouter/meta-llama/llama-3.2-3b-instruct`
- `openrouter/meta-llama/llama-3.3-70b-instruct`
- `openrouter/meta-llama/llama-4-maverick`
- `openrouter/meta-llama/llama-4-scout`
- `openrouter/meta-llama/llama-guard-4-12b`
- `openrouter/microsoft/phi-4`
- `openrouter/microsoft/phi-4-mini-instruct`
- `openrouter/microsoft/wizardlm-2-8x22b`
- `openrouter/minimax/minimax-01`
- `openrouter/minimax/minimax-m1`
- `openrouter/minimax/minimax-m2`
- `openrouter/minimax/minimax-m2-her`
- `openrouter/minimax/minimax-m2.1`
- `openrouter/minimax/minimax-m2.5`
- `openrouter/minimax/minimax-m2.7`
- `openrouter/minimax/minimax-m3`
- `openrouter/mistralai/codestral-2508`
- `openrouter/mistralai/devstral-2512`
- `openrouter/mistralai/ministral-14b-2512`
- `openrouter/mistralai/ministral-3b-2512`
- `openrouter/mistralai/ministral-8b-2512`
- `openrouter/mistralai/mistral-large`
- `openrouter/mistralai/mistral-large-2407`
- `openrouter/mistralai/mistral-large-2512`
- `openrouter/mistralai/mistral-medium-3`
- `openrouter/mistralai/mistral-medium-3-5`
- `openrouter/mistralai/mistral-medium-3.1`
- `openrouter/mistralai/mistral-nemo`
- `openrouter/mistralai/mistral-saba`
- `openrouter/mistralai/mistral-small-24b-instruct-2501`
- `openrouter/mistralai/mistral-small-2603`
- `openrouter/mistralai/mistral-small-3.1-24b-instruct`
- `openrouter/mistralai/mistral-small-3.2-24b-instruct`
- `openrouter/mistralai/mixtral-8x22b-instruct`
- `openrouter/mistralai/voxtral-small-24b-2507`
- `openrouter/moonshotai/kimi-k2`
- `openrouter/moonshotai/kimi-k2-0905`
- `openrouter/moonshotai/kimi-k2-thinking`
- `openrouter/moonshotai/kimi-k2.5`
- `openrouter/moonshotai/kimi-k2.6`
- `openrouter/moonshotai/kimi-k2.7-code`
- `openrouter/morph/morph-v3-fast`
- `openrouter/morph/morph-v3-large`
- `openrouter/nousresearch/hermes-3-llama-3.1-405b`
- `openrouter/nousresearch/hermes-3-llama-3.1-70b`
- `openrouter/nousresearch/hermes-4-405b`
- `openrouter/nousresearch/hermes-4-70b`
- `openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5`
- `openrouter/nvidia/nemotron-3-nano-30b-a3b`
- `openrouter/nvidia/nemotron-3-super-120b-a12b`
- `openrouter/nvidia/nemotron-3-ultra-550b-a55b`
- `openrouter/openai/gpt-3.5-turbo`
- `openrouter/openai/gpt-3.5-turbo-0613`
- `openrouter/openai/gpt-3.5-turbo-16k`
- `openrouter/openai/gpt-3.5-turbo-instruct`
- `openrouter/openai/gpt-4`
- `openrouter/openai/gpt-4-turbo`
- `openrouter/openai/gpt-4-turbo-preview`
- `openrouter/openai/gpt-4.1`
- `openrouter/openai/gpt-4.1-mini`
- `openrouter/openai/gpt-4.1-nano`
- `openrouter/openai/gpt-4o`
- `openrouter/openai/gpt-4o-2024-05-13`
- `openrouter/openai/gpt-4o-2024-08-06`
- `openrouter/openai/gpt-4o-2024-11-20`
- `openrouter/openai/gpt-4o-mini`
- `openrouter/openai/gpt-4o-mini-2024-07-18`
- `openrouter/openai/gpt-4o-mini-search-preview`
- `openrouter/openai/gpt-4o-search-preview`
- `openrouter/openai/gpt-4o:extended`
- `openrouter/openai/gpt-5`
- `openrouter/openai/gpt-5-chat`
- `openrouter/openai/gpt-5-image`
- `openrouter/openai/gpt-5-image-mini`
- `openrouter/openai/gpt-5-mini`
- `openrouter/openai/gpt-5-nano`
- `openrouter/openai/gpt-5.1`
- `openrouter/openai/gpt-5.1-chat`
- `openrouter/openai/gpt-5.2`
- `openrouter/openai/gpt-5.2-chat`
- `openrouter/openai/gpt-5.3-chat`
- `openrouter/openai/gpt-5.4`
- `openrouter/openai/gpt-5.4-image-2`
- `openrouter/openai/gpt-5.4-mini`
- `openrouter/openai/gpt-5.4-nano`
- `openrouter/openai/gpt-5.5`
- `openrouter/openai/gpt-audio`
- `openrouter/openai/gpt-audio-mini`
- `openrouter/openai/gpt-chat-latest`
- `openrouter/openai/gpt-oss-120b`
- `openrouter/openai/gpt-oss-20b`
- `openrouter/openai/gpt-oss-safeguard-20b`
- `openrouter/openai/o1`
- `openrouter/openai/o1-pro`
- `openrouter/openai/o3`
- `openrouter/openai/o3-deep-research`
- `openrouter/openai/o3-mini`
- `openrouter/openai/o3-mini-high`
- `openrouter/openai/o3-pro`
- `openrouter/openai/o4-mini`
- `openrouter/openai/o4-mini-deep-research`
- `openrouter/openai/o4-mini-high`
- `openrouter/perceptron/perceptron-mk1`
- `openrouter/perplexity/sonar`
- `openrouter/perplexity/sonar-deep-research`
- `openrouter/perplexity/sonar-pro`
- `openrouter/perplexity/sonar-pro-search`
- `openrouter/perplexity/sonar-reasoning-pro`
- `openrouter/poolside/laguna-m.1`
- `openrouter/poolside/laguna-xs.2`
- `openrouter/prime-intellect/intellect-3`
- `openrouter/qwen/qwen-2.5-72b-instruct`
- `openrouter/qwen/qwen-2.5-7b-instruct`
- `openrouter/qwen/qwen-2.5-coder-32b-instruct`
- `openrouter/qwen/qwen-plus`
- `openrouter/qwen/qwen-plus-2025-07-28`
- `openrouter/qwen/qwen-plus-2025-07-28:thinking`
- `openrouter/qwen/qwen2.5-vl-72b-instruct`
- `openrouter/qwen/qwen3-14b`
- `openrouter/qwen/qwen3-235b-a22b`
- `openrouter/qwen/qwen3-235b-a22b-2507`
- `openrouter/qwen/qwen3-235b-a22b-thinking-2507`
- `openrouter/qwen/qwen3-30b-a3b`
- `openrouter/qwen/qwen3-30b-a3b-instruct-2507`
- `openrouter/qwen/qwen3-30b-a3b-thinking-2507`
- `openrouter/qwen/qwen3-32b`
- `openrouter/qwen/qwen3-8b`
- `openrouter/qwen/qwen3-coder`
- `openrouter/qwen/qwen3-coder-30b-a3b-instruct`
- `openrouter/qwen/qwen3-coder-flash`
- `openrouter/qwen/qwen3-coder-next`
- `openrouter/qwen/qwen3-coder-plus`
- `openrouter/qwen/qwen3-max`
- `openrouter/qwen/qwen3-max-thinking`
- `openrouter/qwen/qwen3-next-80b-a3b-instruct`
- `openrouter/qwen/qwen3-next-80b-a3b-thinking`
- `openrouter/qwen/qwen3-vl-235b-a22b-instruct`
- `openrouter/qwen/qwen3-vl-235b-a22b-thinking`
- `openrouter/qwen/qwen3-vl-30b-a3b-instruct`
- `openrouter/qwen/qwen3-vl-30b-a3b-thinking`
- `openrouter/qwen/qwen3-vl-32b-instruct`
- `openrouter/qwen/qwen3-vl-8b-instruct`
- `openrouter/qwen/qwen3-vl-8b-thinking`
- `openrouter/qwen/qwen3.5-122b-a10b`
- `openrouter/qwen/qwen3.5-27b`
- `openrouter/qwen/qwen3.5-35b-a3b`
- `openrouter/qwen/qwen3.5-397b-a17b`
- `openrouter/qwen/qwen3.5-9b`
- `openrouter/qwen/qwen3.5-flash-02-23`
- `openrouter/qwen/qwen3.5-plus-02-15`
- `openrouter/qwen/qwen3.5-plus-20260420`
- `openrouter/qwen/qwen3.6-27b`
- `openrouter/qwen/qwen3.6-35b-a3b`
- `openrouter/qwen/qwen3.6-flash`
- `openrouter/qwen/qwen3.6-max-preview`
- `openrouter/qwen/qwen3.6-plus`
- `openrouter/qwen/qwen3.7-max`
- `openrouter/qwen/qwen3.7-plus`
- `openrouter/rekaai/reka-edge`
- `openrouter/rekaai/reka-flash-3`
- `openrouter/relace/relace-apply-3`
- `openrouter/relace/relace-search`
- `openrouter/sao10k/l3-lunaris-8b`
- `openrouter/sao10k/l3.1-70b-hanami-x1`
- `openrouter/sao10k/l3.1-euryale-70b`
- `openrouter/sao10k/l3.3-euryale-70b`
- `openrouter/stepfun/step-3.5-flash`
- `openrouter/stepfun/step-3.7-flash`
- `openrouter/switchpoint/router`
- `openrouter/tencent/hunyuan-a13b-instruct`
- `openrouter/tencent/hy3-preview`
- `openrouter/thedrummer/cydonia-24b-v4.1`
- `openrouter/thedrummer/rocinante-12b`
- `openrouter/thedrummer/skyfall-36b-v2`
- `openrouter/thedrummer/unslopnemo-12b`
- `openrouter/undi95/remm-slerp-l2-13b`
- `openrouter/upstage/solar-pro-3`
- `openrouter/writer/palmyra-x5`
- `openrouter/x-ai/grok-4.20`
- `openrouter/x-ai/grok-4.20-multi-agent`
- `openrouter/x-ai/grok-4.3`
- `openrouter/x-ai/grok-build-0.1`
- `openrouter/xiaomi/mimo-v2.5`
- `openrouter/xiaomi/mimo-v2.5-pro`
- `openrouter/z-ai/glm-4.5`
- `openrouter/z-ai/glm-4.5-air`
- `openrouter/z-ai/glm-4.5v`
- `openrouter/z-ai/glm-4.6`
- `openrouter/z-ai/glm-4.6v`
- `openrouter/z-ai/glm-4.7`
- `openrouter/z-ai/glm-4.7-flash`
- `openrouter/z-ai/glm-5`
- `openrouter/z-ai/glm-5-turbo`
- `openrouter/z-ai/glm-5.1`
- `openrouter/z-ai/glm-5.2`
- `openrouter/~anthropic/claude-fable-latest`
- `openrouter/~anthropic/claude-haiku-latest`
- `openrouter/~anthropic/claude-opus-latest`
- `openrouter/~anthropic/claude-sonnet-latest`
- `openrouter/~google/gemini-flash-latest`
- `openrouter/~google/gemini-pro-latest`
- `openrouter/~moonshotai/kimi-latest`
- `openrouter/~openai/gpt-latest`
- `openrouter/~openai/gpt-mini-latest`

</details>

<details>
<summary><strong>Claude Code CLI (cc/*) (3)</strong></summary>

- `cc/haiku`
- `cc/opus`
- `cc/sonnet`

</details>

<details>
<summary><strong>Codex CLI (codex/*) (7)</strong></summary>

- `codex/codex-auto-review`
- `codex/default`
- `codex/gpt-5.2`
- `codex/gpt-5.3-codex`
- `codex/gpt-5.4`
- `codex/gpt-5.4-mini`
- `codex/gpt-5.5`

</details>

## 🤗 Contributing

Contributions in the form of issues are welcome. KISS Sorcar should be able to help implement and review them.

## 📄 License

Apache-2.0. See [LICENSE](LICENSE).

## 📚 Citation

If you use KISS Sorcar in your research, please cite:

```bibtex
@misc{sen2026kisssorcar,
  title         = {KISS Sorcar: A Stupidly-Simple General-Purpose and Software Engineering AI Assistant},
  author        = {Sen, Koushik},
  year          = {2026},
  eprint        = {2604.23822},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url           = {https://arxiv.org/abs/2604.23822}
}
```
