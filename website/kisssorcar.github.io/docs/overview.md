# KISS Sorcar Overview

> Open-source general-purpose AI agent for long-horizon tasks and AI discovery.

**KISS Sorcar is a free, simple, local-first, bring-your-own-key AI agent framework.** It runs as a VS Code extension and a browser/mobile web app, both served by a local daemon, and offers a Python client API for scripting tasks. Your prompts and code are sent directly to the model provider or local endpoint you configure — not through our servers. It supports multi-model workflows just via prompts. Agents run as daemons hosted by the local server (a standalone `sorcar` terminal command can also run a task without the daemon). Complex AI systems/techniques can be replaced with a paragraph of prompt in KISS Sorcar.

*"Everything should be made as simple as possible, but not simpler." — Albert Einstein*

- **Version:** 2026.9.14
- **License:** Apache-2.0
- **Python:** 3.13+
- **Website:** <https://kisssorcar.github.io/>
- **Source:** <https://github.com/ksenxx/kiss_ai>
- **Paper:** <https://arxiv.org/abs/2604.23822>

## KISS Sorcar vs Claude Code vs Cursor

| Capability | KISS Sorcar | Claude Code | Cursor |
|---|---|---|---|
| Interfaces | VS Code extension + web/mobile app + Python API | CLI + mobile app | Custom VS Code |
| AI Discovery | Yes — simply via prompt | No | No |
| GEPA Prompt Optimization | Yes — simply via prompt | No | No |
| Multiple models from multiple vendors in the same task | Yes — mix OpenAI, Anthropic, Gemini, Together, Z.AI, Moonshot AI, OpenRouter, Claude Code CLI, and Codex CLI | No — Anthropic Claude models only | No — one model per task |
| Primary focus | Quality — rigorous review, end-to-end tests | Speed and developer ergonomics | Speed |
| Core agents lines of code | ~3000 | Unknown | Unknown |
| Models in bundled catalog | 661 across 9 provider categories | Claude family only | Subset chosen by Cursor |
| Bring your own API key / endpoint | Yes — keys stay on your machine | Anthropic key | Routed through Cursor backend |
| Open source | Yes — Apache-2.0 | Proprietary | Proprietary |
| Price | Free framework; pay only your chosen model provider | Subscription / API usage | Subscription |
| Run on top of Claude Code / Codex CLI | Yes — `cc/*` and `codex/*` namespaces | N/A | No |
| Messaging and communication channels | 43 third-party agents: 32 messaging channels (Slack, Gmail, Email (IMAP/SMTP), Phone Control, SMS, WhatsApp, Home Assistant, …) plus service agents for GitHub, Notion, Postgres, Brave Search, Firecrawl, and Google Workspace | Slack, mobile Remote Control, and research-preview channels; no documented built-in Gmail, WhatsApp, phone-call, or SMS channel | Slack and Microsoft Teams Cloud Agent integrations; no documented built-in Gmail, WhatsApp, phone-call, or SMS channel |
| Scheduled automations | Natural-language cron agent | — | — |
| Wake word for voice interaction | Sorcar | N/A | N/A |

## Unique Features

- **AI discovery and auto research via prompt.** Describe a discovery or optimization goal in a paragraph; Sorcar iterates over ideas, tracks what worked, and doesn't stop until the target metrics are met.
- **GEPA prompt optimization via prompt.** Run the GEPA reflective prompt-evolution algorithm on your own data with a single task prompt.
- **Multi-model orchestration in one task.** Ask one model to implement and another to review — expressed entirely in the prompt, e.g. "Use claude-fable-5 for development and gpt-5.6-sol for review."
- **Dynamic model switching and steering.** A running agent can change its own LLM mid-task (`set_model`), and you can inject user messages into a running agent to steer it on the fly.
- **Git-worktree task isolation.** Each interactive task runs on an isolated git worktree branch that is auto-committed and squash-merged back when it finishes.
- **Voice interaction.** With the `sorcar` wake word, KISS Sorcar behaves like a super-intelligent Alexa; it distinguishes among different speakers.
- **43 third-party agents.** 32 messaging channels — Slack, Gmail, Email (IMAP/SMTP), WhatsApp, SMS, iMessage, Telegram, Discord, Signal, phone control, Home Assistant, and more — plus service agents for GitHub, Notion, PostgreSQL, Brave Search, Firecrawl, and Google Workspace (Calendar, Drive, Docs, Sheets).
- **Pre-run task classification.** Normally one fast non-agentic model call (structured output, with one plain-text retry if that fails) detects whether a task is a development task that requires creating or editing files — non-development tasks (questions, git-only operations) skip worktree isolation, and simple tasks get a lite system prompt for faster starts. Toggleable in the Settings panel.
- **Persistent agent memory.** On by default: standard Sorcar runs get seven `memory_*` tools (search, pull, read, write, list, refresh, delete) and a memory protocol, so agents recall lessons, preferences, and decisions across tasks. Pages are Markdown files under `~/.kiss/memories` with a SQLite vector index; toggle it in the Settings panel or set `KISS_USE_MEMORY=0`.
- **Credential isolation (Muse auth).** On Linux, credentials for the 24 Muse-supported connectors are isolated by default behind a Meta-Muse-style security boundary: legacy tokens auto-migrate into a local auth daemon's vault on first use (a one-time hand-off of the real credential; plaintext copies are then scrubbed on a best-effort basis), after which the agent process holds only opaque surrogate tokens that the daemon swaps for the real ones at the network edge, and every boundary-routed API request is host-allowlisted (credential-free, bodyless `GET`/`HEAD` redirect hops are the one permitted off-list exception), classified read vs. write, and checked against an allow/deny/ask policy with an audit log. Where the provider supports a poll-based grant (GitHub, Twitch, Microsoft Teams, Nextcloud Talk, Matrix, Signal), connecting works like the Muse app's Connect button — the user signs in and approves in their own browser, nothing is pasted back; providers without such a grant (the six Google services on a headless host, Slack, Discord) get a safe paste-back hand-off instead of browser automation, and the agent never asks for a password or 2FA code. Opt out with `KISS_MUSE_AUTH=0`.

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
