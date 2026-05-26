<div align="center">

![KISS Framework](assets/KISS-Sorcar.png)

# KISS Sorcar

### The open-source AI coding agent that beats Cursor and Claude Code on Terminal Bench.

**Terminal Bench 2.0:  KISS Sorcar 62.2%  •  Cursor agent 61.7%  •  Claude Code 58%**

*Free. Local. Bring your own API key. Runs as a VS Code extension and as a web/mobile app.*

[![Version](https://img.shields.io/badge/version-2026.5.32-blue?style=flat-square)](https://pypi.org/project/kiss-agent-framework/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-blue?style=flat-square)](https://www.python.org/)
[![Website](https://img.shields.io/badge/website-kisssorcar.github.io-1976d2?style=flat-square)](https://kisssorcar.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.23822-b31b1b?style=flat-square)](https://arxiv.org/abs/2604.23822)

*"Everything should be made as simple as possible, but not simpler." — Albert Einstein*

**Website:** [https://kisssorcar.github.io/](https://kisssorcar.github.io/) · **Paper:** [arXiv:2604.23822](https://arxiv.org/abs/2604.23822)

</div>

______________________________________________________________________

<details>
<summary><strong>Table of Contents</strong></summary>

- [Introduction to KISS Sorcar](#introduction-to-kiss-sorcar)
- [See It in Action](#-see-it-in-action)
- [Full Installation](#full-installation)
- [KISS Sorcar Extension Installation](#kiss-sorcar-extension-installation)
- [CLI Interface](#cli-interface)
- [Messaging & Third-Party Agents](#-messaging--third-party-agents)
- [Models Supported](#-models-supported)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)

</details>

# Introduction to KISS Sorcar

<div align="center">
  <img src="assets/sorcar-main.gif" alt="KISS Sorcar demo" width="100%">
</div>

**KISS Sorcar is the open-source AI coding agent that beats Cursor and Claude Code on Terminal Bench.** On Terminal Bench 2.0 it scored **62.2%**, ahead of **Cursor agent (61.7%)** and **Claude Code (58%)** — while remaining **free**, **open-source**, and **fully local**. You bring your own model API key (Anthropic recommended) or LLM endpoint or Claude Code or OpenAI Codex; nothing about your code or prompts is sent through our servers.

It runs as a **Visual Studio Code extension**, as a **CLI tool** and as a **web/mobile app**, and is built on the **KISS Agent Framework** — a deliberately simple agent runtime that follows the [KISS principle](https://en.wikipedia.org/wiki/KISS_principle) ("Keep it Simple, Stupid"). The agent has **browser** support (Chromium + Playwright), **multimodal** support, **Docker container** support, can **research topics on the web**, and can **run for hours** across multiple sessions. If you have Claude Code or OpenAI Codex in your PATH, you can also use `cc/*` or `codex/*` models for chat.

KISS Sorcar is named after [P. C. Sorcar, the Bengali magician](https://en.wikipedia.org/wiki/P._C._Sorcar). The paper is at [papers/kisssorcar/kiss_sorcar.pdf](papers/kisssorcar/kiss_sorcar.pdf).

> Engineering principles are encoded in the agent's system prompt. See Section 5 of the [paper](papers/kisssorcar/kiss_sorcar.pdf) for details.

An old video on KISS Sorcar can be found at [https://www.youtube.com/watch?v=xnYxWvRqACE](https://www.youtube.com/watch?v=xnYxWvRqACE). We **no longer** recommend explicitly creating a plan in KISS Sorcar. See the paper for details.

<scriptsize>Note that **Sorcar** also means government in Bengali.</scriptsize>

## 🎬 See It in Action

From writing production-grade code to planning your next vacation, KISS Sorcar handles a range of tasks end-to-end. Here are a few examples:

<div align="center">

<table>
  <tr>
    <td align="center" width="50%">
      <img src="assets/sorcar-coding.gif" alt="KISS Sorcar writing and refactoring code" width="100%" />
      <br />
      <strong>💻 Coding & Software Engineering</strong>
      <br />
      <sub>Writes, debugs, and refactors code.</sub>
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
      <sub>You can use KISS Sorcar as a mobile/web app.</sub>
    </td>
  </tr>
</table>

</div>

## Full Installation

```
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash
```

## KISS Sorcar Extension Installation

To install KISS Sorcar, open Visual Studio Code, search for "KISS Sorcar" in the extension marketplace, install, and relaunch VS Code. Press ESC if you don't have a specific API key, but you must provide at least one API key.

You can also manually download the extension from [src/kiss/agents/vscode/kiss-sorcar.vsix](src/kiss/agents/vscode/kiss-sorcar.vsix).

## CLI Interface

If you do not want to use the KISS Sorcar IDE, you can open a terminal and use `sorcar` as a shell command. Some examples:

```
sorcar -t "What is 2435*234"

sorcar -n -t --use-chat "What is 2435*234?" # start a new chat session

sorcar -m "claude-sonnet-4-6" -t "What is 2435*234?" # use a specific model

echo "Can you find the cheapest non-stop flight from SFO to JFK on June 15?" > prompt
sorcar -f prompt # use contents of a file as the task

sorcar -t 'Can you send the message "Hello from Sorcar!" to ksen via the desktop slack app?'

sorcar -t 'Can you show me the detailed step-by-step workflow of gepa.py?'
```

### CLI Options

| Flag | Description |
|------|-------------|
| `-t`, `--task` | Task description |
| `-f`, `--file` | Path to a file whose contents to use as the task |
| `-m`, `--model_name` | LLM model name (default: `claude-opus-4-6`) |
| `-e`, `--endpoint` | Custom endpoint for a local model |
| `--header` | Custom HTTP header in `Key:Value` form; may be repeated |
| `-b`, `--max_budget` | Maximum budget in USD |
| `-w`, `--work_dir` | Working directory |
| `-v`, `--verbose` | Print output to console (default: true) |
| `-n`, `--new` | Start a new chat session |
| `-c`, `--chat-id` | Resume a chat session by ID |
| `-l`, `--list-chat-id` | List the last 10 chat sessions with tasks and results |
| `-p`, `--parallel` | Enable parallel subagents |
| `--no-web` | Disable browser/web tools (terminal-only mode) |
| `--use-chat` | Use chat mode |
| `--use-worktree` | Use chat mode with git worktree isolation (advanced) |
| `--cleanup` | Scan for and clean up orphaned worktree branches |

## 💬 Messaging & Third-Party Agents

KISS Sorcar includes 23 third-party messaging agents that can send and receive messages on your behalf:

BlueBubbles · Discord · Feishu · Gmail · Google Chat · iMessage · IRC · LINE · Matrix · Mattermost · Microsoft Teams · Nextcloud Talk · Nostr · Phone Control · Signal · Slack · SMS · Synology Chat · Telegram · Tlon · Twitch · WhatsApp · Zalo

These agents are in `src/kiss/agents/third_party_agents/`.

## 🤖 Models Supported

**542 models** across 8 provider categories (OpenAI, Anthropic, Gemini, Together AI, MiniMax, OpenRouter, Claude Code CLI, Codex CLI) with built-in pricing, context lengths, and capability flags.

**Generation Models** (text generation with function calling support):

- **OpenAI**: gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-pro, gpt-5.1, gpt-5.2, gpt-5.2-pro, gpt-5.3-chat-latest, gpt-5.4, gpt-5.4-mini, gpt-5.4-nano, gpt-5.4-pro, gpt-5.5, gpt-5.5-pro (+ dated variants)
- **OpenAI (Codex)**: gpt-5-codex, gpt-5.1-codex, gpt-5.1-codex-max, gpt-5.1-codex-mini, gpt-5.2-codex, gpt-5.3-codex
- **OpenAI (Reasoning)**: o1, o1-pro, o3, o3-mini, o3-pro, o3-deep-research, o4-mini, o4-mini-deep-research (+ dated variants)
- **OpenAI (Image)**: gpt-image-1, gpt-image-1-mini, gpt-image-1.5, gpt-image-2
- **OpenAI (Search)**: gpt-4o-search-preview, gpt-4o-mini-search-preview
- **OpenAI (Other)**: computer-use-preview, openai/gpt-oss-20b, openai/gpt-oss-120b
- **Anthropic**: claude-opus-4-7, claude-opus-4-6, claude-opus-4-5, claude-opus-4-1, claude-opus-4, claude-sonnet-4-6, claude-sonnet-4-5, claude-sonnet-4, claude-haiku-4-5 (+ dated variants)
- **Claude Code CLI**: cc/haiku, cc/opus, cc/sonnet
- **Codex CLI**: codex/default, codex/codex-auto-review, codex/gpt-5.2, codex/gpt-5.3-codex, codex/gpt-5.4, codex/gpt-5.4-mini, codex/gpt-5.5
- **Gemini**: gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-image, gemini-2.5-flash-lite, gemini-2.0-flash, gemini-2.0-flash-lite, gemini-3.1-flash-lite
- **Gemini (Preview)**: gemini-3-pro-preview, gemini-3-flash-preview, gemini-3.1-pro-preview, gemini-3.1-flash-lite-preview, gemini-3.1-flash-tts-preview
- **Gemini (Open Models)**: google/gemma-4-31B-it, google/gemma-3n-E4B-it, google/gemma-2-27b-it
- **Together AI (Llama)**: Llama-4-Scout, Llama-4-Maverick (with function calling), Llama-3.x series (generation only)
- **Together AI (Qwen)**: Qwen2-1.5B-Instruct, Qwen2-VL-72B-Instruct, Qwen2.5-72B/14B/7B-Instruct, Qwen2.5-VL-72B, Qwen2.5-Coder-32B, Qwen3-235B series, Qwen3-Coder-480B, Qwen3-Coder-Next, Qwen3-Next-80B, Qwen3-VL-32B/8B, Qwen3.5-397B/9B, Qwen3.6-Plus, QwQ-32B
- **Together AI (DeepSeek)**: DeepSeek-R1, DeepSeek-R1-0528, DeepSeek-R1-0528-tput, DeepSeek-R1-Distill-Llama-70B, DeepSeek-R1-Distill-Qwen-1.5B/14B, DeepSeek-V3-0324, DeepSeek-V3.1, DeepSeek-V4-Pro, deepseek-coder-33b-instruct
- **Together AI (Kimi/Moonshot)**: Kimi-K2-Instruct, Kimi-K2-Instruct-0905, Kimi-K2-Thinking, Kimi-K2.5, Kimi-K2.6
- **Together AI (Mistral)**: Ministral-3-14B, Mistral-7B-v0.1/v0.2/v0.3, Mistral-Small-24B, Mixtral-8x7B
- **Together AI (Z.AI)**: GLM-5, GLM-5.1, GLM-4.5-Air-FP8, GLM-4.6, GLM-4.7
- **Together AI (MiniMax)**: MiniMax-M2.5, MiniMax-M2.7
- **MiniMax**: minimax-m2.5, minimax-m2.5-lightning (via MiniMax direct API)
- **Together AI (DeepCogito)**: cogito-v1-preview (llama-70B/8B/70B-Turbo, qwen-14B/32B), cogito-v2-1-671b
- **Together AI (NVIDIA)**: Llama-3.1-Nemotron-70B, Nemotron-Nano-9B-v2
- **Together AI (Other)**: arcee-ai/trinity-mini, essentialai/rnj-1-instruct
- **OpenRouter**: Access to 330 models from 51 providers via unified API, including OpenAI, Anthropic, Google, Meta Llama, DeepSeek, Qwen, Amazon Nova, Cohere, X.AI Grok, MiniMax, ByteDance Seed, MoonshotAI, Mistral, NVIDIA, Z.AI/GLM, AllenAI, Perplexity, NousResearch, Baidu ERNIE, Xiaomi, Reka AI, Arcee AI, Perceptron, and others

**Embedding Models** (for RAG and semantic search):

- **OpenAI**: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- **Google**: gemini-embedding-001, gemini-embedding-2, gemini-embedding-2-preview
- **Together AI**: BAAI/bge-base-en-v1.5, intfloat/multilingual-e5-large-instruct

Each model entry includes capability flags: `is_function_calling_supported`, `is_generation_supported`, `is_embedding_supported`.

## 🤗 Contributing

Contributions in the form of issues are welcome! KISS Sorcar should be able to take care of them.

## 📄 License

Apache-2.0

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
