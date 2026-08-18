# KISS Sorcar Documentation

> Pure-Markdown documentation for KISS Sorcar — the free, open-source, local-first, bring-your-own-key general-purpose AI agent framework. These pages are optimized for both humans and LLMs/coding assistants. See also [/llms.txt](https://kisssorcar.github.io/llms.txt) and [/llms-full.txt](https://kisssorcar.github.io/llms-full.txt).

## Contents

- [Overview](overview.md) — What KISS Sorcar is, the name, and how it compares to Claude Code and Cursor
- [Installation](installation.md) — Install from source, pipx/uv, API-key configuration, VS Code extension, Docker
- [Client Interfaces](cli.md) — The `kiss-web` daemon, VS Code extension, web/mobile app, and Python client API
- [Python API Reference](api.md) — KISSAgent, RelentlessAgent, SorcarAgent, ChatSorcarAgent, WorktreeSorcarAgent, GitWorktreeOps
- [Supported Models](models.md) — 622-model catalog across 9 provider categories
- [Messaging & Third-Party Agents](messaging-agents.md) — 32 channel agents plus infrastructure agents and the Govee smart-home CLI
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

# Or Python package only (Python 3.13+)
pipx install kiss-agent-framework

# Set at least one model API key
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY, GEMINI_API_KEY, ...

# Start the daemon that serves the VS Code extension and the web app
kiss-web
```

Then open the KISS Sorcar sidebar in VS Code, or open the remote web app URL shown in the Settings panel. From Python:

```python
from kiss.server import sorcar

result = sorcar.run("What is 2435*234?")
print(result.text)
```
