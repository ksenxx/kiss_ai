# KISS Sorcar Overview

> Open-source general-purpose AI agent for long-horizon tasks and AI discovery.

**KISS Sorcar is a free, simple, local-first, bring-your-own-key AI agent framework.** It runs as a VS Code extension, a Claude-Code-style CLI, and a browser/mobile web app. Your prompts and code are sent directly to the model provider or local endpoint you configure — not through our servers. It supports multi-model workflows just via prompts. All agents run as daemons. Complex AI systems/techniques can be replaced with a paragraph of prompt in KISS Sorcar.

*"Everything should be made as simple as possible, but not simpler." — Albert Einstein*

- **Version:** 2026.7.30
- **License:** Apache-2.0
- **Python:** 3.13+
- **Website:** <https://kisssorcar.github.io/>
- **Source:** <https://github.com/ksenxx/kiss_ai>
- **Paper:** <https://arxiv.org/abs/2604.23822>

## KISS Sorcar vs Claude Code vs Cursor

| Capability | KISS Sorcar | Claude Code | Cursor |
|---|---|---|---|
| Interfaces | CLI + VS Code extension + web/mobile app | CLI + mobile app | Custom VS Code |
| AI Discovery | Yes — simply via prompt | No | No |
| GEPA Prompt Optimization | Yes — simply via prompt | No | No |
| Multiple models from multiple vendors in the same task | Yes — mix OpenAI, Anthropic, Gemini, Together, Z.AI, Moonshot AI, OpenRouter, Claude Code CLI, and Codex CLI | No — Anthropic Claude models only | No — one model per task |
| Primary focus | Quality — rigorous review, end-to-end tests | Speed and developer ergonomics | Speed |
| Core agents lines of code | ~2850 | Unknown | Unknown |
| Models in bundled catalog | 538 across 9 provider categories | Claude family only | Subset chosen by Cursor |
| Bring your own API key / endpoint | Yes — keys stay on your machine | Anthropic key | Routed through Cursor backend |
| Open source | Yes — Apache-2.0 | Proprietary | Proprietary |
| Price | Free framework; pay only your chosen model provider | Subscription / API usage | Subscription |
| Run on top of Claude Code / Codex CLI | Yes — `cc/*` and `codex/*` namespaces | N/A | No |
| Messaging and communication channels | 23 third-party agents, including Slack, Gmail, Phone Control, SMS, and WhatsApp | Slack, mobile Remote Control, and research-preview channels; no documented built-in Gmail, WhatsApp, phone-call, or SMS channel | Slack and Microsoft Teams Cloud Agent integrations; no documented built-in Gmail, WhatsApp, phone-call, or SMS channel |

## Unique Features

- **AI discovery and auto research via prompt.** Describe a discovery or optimization goal in a paragraph; Sorcar iterates over ideas, tracks what worked, and doesn't stop until the target metrics are met.
- **GEPA prompt optimization via prompt.** Run the GEPA reflective prompt-evolution algorithm on your own data with a single task prompt.
- **Multi-model orchestration in one task.** Ask one model to implement and another to review — expressed entirely in the prompt, e.g. "Use claude-fable-5 for development and gpt-5.6-sol for review."
- **Dynamic model switching and steering.** A running agent can change its own LLM mid-task (`set_model`), and you can inject user messages into a running agent to steer it on the fly.
- **Git-worktree task isolation.** Each interactive task runs on an isolated git worktree branch that is auto-committed and squash-merged back when it finishes.
- **Voice interaction.** With the `sorcar` wake word, KISS Sorcar behaves like a super-intelligent Alexa; it distinguishes among different speakers.
- **23 messaging agents.** Slack, Gmail, WhatsApp, SMS, iMessage, Telegram, Discord, Signal, phone control, and more.

## What Is in the Name

**KISS Agent Framework** is a deliberately small agent runtime organized around the KISS principle ("Keep it Simple, Stupid"). The name "Sorcar" pays homage to P. C. Sorcar, the legendary Bengali magician, evoking the idea of an agent that performs feats that appear magical yet are grounded in disciplined engineering. Note: **Sorcar** also means government in Bengali.

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
