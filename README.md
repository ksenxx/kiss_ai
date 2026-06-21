<div align="center">

![KISS Framework](assets/KISS-Sorcar.png)

[![Version](https://img.shields.io/badge/version-2026.6.25-blue?style=flat-square)](https://pypi.org/project/kiss-agent-framework/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-blue?style=flat-square)](https://www.python.org/)
[![Website](https://img.shields.io/badge/website-kisssorcar.github.io-1976d2?style=flat-square)](https://kisssorcar.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.23822-b31b1b?style=flat-square)](https://arxiv.org/abs/2604.23822)

*"Everything should be made as simple as possible, but not simpler." — Albert Einstein*

</div>

# KISS Sorcar

### Open-source general-purpose AI agent for long-horizon tasks and AI discovery

**KISS Sorcar is free, simple, local-first, bring-your-own-key AI agent frmework.** It runs as a VS Code extension, a Claude-Code-style CLI, and a browser/mobile web app. Your prompts and code are sent directly to the model provider or local endpoint you configure — not through our servers.  It supports multi-model workflow just via prompt.  All agents run as daemon.

```bash
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash
```

______________________________________________________________________

<details>
<summary><strong>Table of Contents</strong></summary>

- [What is KISS Sorcar?](#what-is-kiss-sorcar)
- [Why KISS Sorcar?](#why-kiss-sorcar)
- [KISS Sorcar vs Claude Code vs Cursor](#-kiss-sorcar-vs-claude-code-vs-cursor)
- [See It in Action](#-see-it-in-action)
- [Installation](#installation)
- [VS Code Extension](#vs-code-extension)
- [CLI Interface](#cli-interface)
- [Web and Mobile App](#web-and-mobile-app)
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
| **Models in bundled catalog** | 499 across 8 provider categories | Claude family only | Subset chosen by Cursor |
| **Bring your own API key / endpoint** | ✅ Yes — keys stay on your machine | ✅ Anthropic key | ⚠️ Routed through Cursor backend |
| **Open source** | ✅ Apache-2.0 | ❌ Proprietary | ❌ Proprietary |
| **Price** | Free framework; pay only your chosen model provider | Subscription / API usage | Subscription |
| **Run on top of Claude Code / Codex CLI** | ✅ `cc/*` and `codex/*` namespaces | N/A | ❌ |
| **Terminal Bench 2.0 score** | **62.2%** | 58% | 61.7% (Cursor agent) |

## What is in the Name

**KISS Agent Framework** is a deliberately small agent runtime organized around the [KISS principle](https://en.wikipedia.org/wiki/KISS_principle) ("Keep it Simple, Stupid"). 
KISS Sorcar is named after [P. C. Sorcar, the Bengali magician](https://en.wikipedia.org/wiki/P._C._Sorcar). 
<sub>Note: **Sorcar** also means government in Bengali.</sub>

## 🎬 See It in Action

<div align="center">

<table>
  <tr>
    <td align="center" width="50%">
      <img src="assets/sorcar-coding.gif" alt="KISS Sorcar writing and refactoring code" width="100%" />
      <br />
      <strong>💻 Coding & Software Engineering</strong>
      <br />
      <sub>Writes, debugs, tests, and refactors code.</sub>
    </td>
    <td align="center" width="50%">
      <img src="assets/sorcar-trip.gif" alt="KISS Sorcar planning a trip" width="100%" />
      <br />
      <strong>✈️ Trip Planning & Research</strong>
      <br />
      <sub>Browses the web, compares options, and assembles itineraries.</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="assets/sorcar-slack.gif" alt="KISS Sorcar sending a Slack message" width="100%" />
      <br />
      <strong>💬 Desktop & Messaging Apps</strong>
      <br />
      <sub>Drives native apps like Slack via the desktop, end-to-end.</sub>
    </td>
    <td align="center" width="50%">
      <img src="assets/sorcar-mobile.gif" alt="KISS Sorcar controlling a mobile device" width="100%" />
      <br />
      <strong>📱 Mobile/Web App</strong>
      <br />
      <sub>Use KISS Sorcar from a browser or mobile device.</sub>
    </td>
  </tr>
</table>

</div>

An older KISS Sorcar video is available at [https://www.youtube.com/watch?v=xnYxWvRqACE](https://www.youtube.com/watch?v=xnYxWvRqACE). We **no longer** recommend explicitly creating a plan in KISS Sorcar; see the paper for the current workflow guidance.

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

## VS Code Extension

To install only the KISS Sorcar extension, open Visual Studio Code, search for **KISS Sorcar** in the extension marketplace, install it, and relaunch VS Code. Press ESC if you do not have a specific API key ready, but configure at least one model backend before running tasks.

You can also manually install the bundled VSIX from [src/kiss/agents/vscode/kiss-sorcar.vsix](src/kiss/agents/vscode/kiss-sorcar.vsix).

## CLI Interface

Use `sorcar` as either an interactive CLI or a one-shot command.

```bash
# Launch the interactive Sorcar CLI, similar to Claude Code.
sorcar

# Run a one-shot task.
sorcar -t "What is 2435*234?"

# Start a new chat session and run directly in the working tree.
sorcar -n --no-worktree -t "What is 2435*234?"

# Use a specific model.
sorcar -m "claude-sonnet-4-6" -t "What is 2435*234?"

# Use the contents of a file as the task.
echo "Can you find the cheapest non-stop flight from SFO to JFK on June 15?" > prompt
sorcar -f prompt

# Ask Sorcar to use desktop/browser/messaging tools.
sorcar -t 'Can you send the message "Hello from Sorcar!" to ksen via the desktop Slack app?'

# Ask Sorcar to explain code.
sorcar -t 'Can you show me the detailed step-by-step workflow of gepa.py?'
```

### CLI options

| Flag | Description |
|------|-------------|
| `-t`, `--task` | Task description |
| `-f`, `--file` | Path to a file whose contents to use as the task |
| `-m`, `--model_name` | LLM model name; defaults to the best available model for configured keys |
| `-e`, `--endpoint` | Custom endpoint for a local or self-hosted model |
| `--header` | Custom HTTP header in `Key:Value` form; may be repeated |
| `-b`, `--max_budget` | Maximum budget in USD |
| `-w`, `--work_dir` | Working directory; defaults to the directory where `sorcar` is launched |
| `-v`, `--verbose` | Print output to console (`true` by default) |
| `-n`, `--new` | Start a new chat session |
| `-c`, `--chat-id` | Resume a chat session by ID |
| `-l`, `--list-chat-id` | List the last 10 chat sessions with tasks and results |
| `-p`, `--parallel` / `--no-parallel` | Enable/disable parallel subagents (`--parallel` by default) |
| `--worktree` / `--no-worktree` | Isolate each task in a git worktree branch (`--worktree` by default) |
| `--auto-commit` / `--no-auto-commit` | Auto-commit worktree changes when a task finishes (`--auto-commit` by default) |
| `--no-web` | Disable browser/web tools for terminal-only mode |
| `--use-chat` | Legacy alias for `--no-worktree` |
| `--use-worktree` | Legacy alias for `--worktree` |
| `--cleanup` | Scan for and clean up orphaned worktree branches |

### Interactive CLI features

The interactive CLI includes:

- `@` file/folder mentions with ranked project-file completion.
- `/help`, `/clear`, `/resume`, `/model`, `/model list`, `/cost`, `/context`, `/commands`, `/skills`, `/mcp`, `/autocommit`, and `/exit` slash commands.
- Custom Markdown slash commands from `~/.kiss/commands`, `<project>/.kiss/commands`, `~/.claude/commands`, and `<project>/.claude/commands`.
- Agent Skills from `~/.kiss/skills`, `<project>/.kiss/skills`, Claude skill directories, `.agents/skills`, and bundled Sorcar skills.
- MCP server discovery from `~/.kiss/mcp.json`, `<project>/.kiss/mcp.json`, and `<project>/.mcp.json`.

## Web and Mobile App

The `kiss-web` entry point starts the browser/mobile web server used for remote KISS Sorcar access. It serves the chat UI over HTTPS/WSS, auto-generates a self-signed certificate under `~/.kiss/tls/` when needed, and authenticates with the `remote_password` setting in `~/.kiss/config.json`.

For access outside your LAN, KISS Sorcar can use a Cloudflare tunnel. Without a named tunnel token, quick tunnels use a random `*.trycloudflare.com` URL that changes on restart; set `CLOUDFLARE_TUNNEL_TOKEN` or `tunnel_token` in `~/.kiss/config.json` for a fixed named tunnel.

## 💬 Messaging & Third-Party Agents

KISS Sorcar includes 23 third-party messaging agents that can send and receive messages on your behalf:

BlueBubbles · Discord · Feishu · Gmail · Google Chat · iMessage · IRC · LINE · Matrix · Mattermost · Microsoft Teams · Nextcloud Talk · Nostr · Phone Control · Signal · Slack · SMS · Synology Chat · Telegram · Tlon · Twitch · WhatsApp · Zalo

It also ships a **Govee smart-home CLI** for controlling IoT lights (on/off, brightness, color, and color temperature) via the Govee Developer API.

These agents live in `src/kiss/agents/third_party_agents/`.

## 🤖 Models Supported

KISS Sorcar ships a catalog of **499 models** across **8 provider categories**, with built-in prices, context lengths, and capability flags (`fc` function calling, `gen` generation, `emb` embedding). The source of truth is [src/kiss/core/models/MODEL_INFO.json](src/kiss/core/models/MODEL_INFO.json).

| Provider category | Catalog entries |
|---|---:|
| OpenAI | 70 |
| Anthropic | 13 |
| Gemini / Google | 23 |
| Together AI | 77 |
| MiniMax | 5 |
| OpenRouter | 301 |
| Claude Code CLI (`cc/*`) | 3 |
| Codex CLI (`codex/*`) | 7 |

Current catalog capability totals:

- **483** generation-capable models
- **320** function-calling-capable models
- **7** embedding models

Examples include OpenAI GPT/o-series/image/search/embedding models, Anthropic Claude Opus/Sonnet/Haiku/Fable models, Gemini and Gemma models, Together-hosted Llama/Qwen/DeepSeek/Kimi/Mistral/Z.AI/MiniMax/DeepCogito/NVIDIA models, direct MiniMax models, Claude Code CLI models, Codex CLI models, and OpenRouter routes to 301 models from 54 provider namespaces.

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
