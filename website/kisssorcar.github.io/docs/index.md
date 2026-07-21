# KISS Sorcar Documentation

> Pure-Markdown documentation for KISS Sorcar — the free, open-source, local-first, bring-your-own-key general-purpose AI agent framework. These pages are optimized for both humans and LLMs/coding assistants. See also [/llms.txt](https://kisssorcar.github.io/llms.txt) and [/llms-full.txt](https://kisssorcar.github.io/llms-full.txt).

## Contents

- [Overview](overview.md) — What KISS Sorcar is, the name, and how it compares to Claude Code and Cursor
- [Installation](installation.md) — Install from source, pipx/uv, API-key configuration, VS Code extension, Docker
- [CLI Reference](cli.md) — The `sorcar` CLI: modes, options, interactive features, `sorcar mcp`
- [Python API Reference](api.md) — KISSAgent, RelentlessAgent, SorcarAgent, ChatSorcarAgent, WorktreeSorcarAgent, GitWorktreeOps
- [Supported Models](models.md) — 538-model catalog across 9 provider categories
- [Messaging & Third-Party Agents](messaging-agents.md) — 23 messaging agents plus the Govee smart-home CLI
- [Sample Tasks](sample-tasks.md) — Ready-to-use example prompts
- [Prompt Tricks](prompt-tricks.md) — Reusable prompt snippets that boost result quality
- [Tips](tips.md) — Practical tips for getting the highest-quality work

## Quick Links

- Website: <https://kisssorcar.github.io/>
- Source: <https://github.com/ksenxx/kiss_ai>
- PyPI: <https://pypi.org/project/kiss-agent-framework/>
- Paper: <https://arxiv.org/abs/2604.23822>

## Quick Start

```bash
# Full install (macOS/Linux)
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash

# Or Python package + CLI only (Python 3.13+)
pipx install kiss-agent-framework

# Set at least one model API key
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY, GEMINI_API_KEY, ...

# Launch the interactive CLI
sorcar

# Or run a one-shot task
sorcar -t "What is 2435*234?"
```
