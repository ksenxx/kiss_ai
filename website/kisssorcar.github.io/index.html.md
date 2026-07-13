# KISS Sorcar — Tell it what to do, in English. It picks the best LLM and ships the work.

> Open-source AI agent. Tell it what to do in English — it picks the best LLM, runs on your laptop, and ships the work. Apache 2.0, 530 models. Works as a VS Code extension, a Claude-Code-style CLI, and a web app.

This is the plain-Markdown twin of <https://kisssorcar.github.io/>. Machine-readable entry points: [/llms.txt](https://kisssorcar.github.io/llms.txt) and [/llms-full.txt](https://kisssorcar.github.io/llms-full.txt).

## What Is KISS Sorcar

**KISS Sorcar is a free, simple, local-first, bring-your-own-key AI agent framework** for long-horizon tasks and AI discovery. It runs as a VS Code extension, a Claude-Code-style CLI (`sorcar`), and a browser/mobile web app. Your prompts and code are sent directly to the model provider or local endpoint you configure — not through our servers. It supports multi-model workflows just via prompts. All agents run as daemons. Complex AI systems/techniques can be replaced with a paragraph of prompt.

- **License:** Apache-2.0 · **Source:** <https://github.com/ksenxx/kiss_ai> · **PyPI:** `kiss-agent-framework`
- **530 models** across 9 provider categories (OpenAI, Anthropic, Gemini/Google, Together AI, Z.AI, Moonshot AI, OpenRouter, Claude Code CLI, Codex CLI)
- **23 messaging agents** (Slack, Gmail, WhatsApp, SMS, iMessage, Telegram, Discord, Signal, Phone Control, …)
- Unique features: AI discovery via prompt, GEPA prompt optimization via prompt, multi-vendor multi-model tasks, dynamic `set_model` switching, steering-on-the-fly, git-worktree task isolation, wake-word voice chat

## Install

```bash
# Full install (macOS/Linux)
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash

# Or Python package + CLI only (Python 3.13+)
pipx install kiss-agent-framework

# Run
sorcar -t "What is 2435*234?"
```

## Documentation

- [Documentation index](docs/index.md)
- [Overview & comparison vs Claude Code / Cursor](docs/overview.md)
- [Installation](docs/installation.md)
- [CLI reference](docs/cli.md)
- [Python API reference](docs/api.md)
- [Supported models](docs/models.md)
- [Messaging & third-party agents](docs/messaging-agents.md)
- [Sample tasks](docs/sample-tasks.md)
- [Prompt tricks](docs/prompt-tricks.md)
- [Tips](docs/tips.md)

## Papers

- **KISS Sorcar: A Stupidly-Simple General-Purpose and Software Engineering AI Assistant** — [PDF](assets/kiss_sorcar.pdf) · [arXiv:2604.23822](https://arxiv.org/abs/2604.23822). Introduces KISS Sorcar and the underlying KISS Agent Framework: a deliberately minimal, open-source agent runtime designed for long-horizon tasks, rigorous review, multi-model orchestration, and AI discovery.
- **Software Engineering KISS Sorcar with KISS Sorcar** — [PDF](assets/se_kiss_sorcar.pdf). A case study of building KISS Sorcar with KISS Sorcar itself: over 44 days the developer issued 3,099 tasks through the system's own interface; nine recurring human–AI collaboration patterns are surfaced from the SQLite usage log.
- **Writing a Research Paper with an AI Agent** — [PDF](assets/writing_paper.pdf). A nine-day, hundred-task chronicle of KISS Sorcar drafting, citing, compiling, and debugging its own research paper.

## Citation

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

---

KISS Sorcar · Apache 2.0 · [GitHub](https://github.com/ksenxx/kiss_ai) · Named after [P. C. Sorcar](https://en.wikipedia.org/wiki/P._C._Sorcar), the legendary Bengali magician.
