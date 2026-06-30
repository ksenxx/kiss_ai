# LinkedIn posts — "Do you really need to build that AI product?"

Two variants. Both grounded in `papers/kisssorcar/kiss_sorcar.tex`:
§2 layered agent architecture, §2.3 `set_model` for on-the-fly model
switching, §3 AI-Discovery / GEPA / Repo-Optimization sample tasks,
§4 Terminal Bench 2.0 score (62.2%, beats Claude Code 58% and Cursor
Composer 2 61.7%), §5 user-facing features (504 models, 23 third-party
messaging agents, MCP, Skills, browser, Docker, worktree isolation).

---

## Option A — feature spotlight (recommended)

**Before you greenlight that "custom AI" build, ask whether KISS Sorcar
could already mimic it from one prompt.**

A pattern I keep seeing: a team writes a glossy spec for a "custom AI
copilot," budgets six engineers and four months, and ships something
that is — honestly — a system prompt, a few tool calls, and a model
behind a logo.

KISS Sorcar (open-source, Apache-2.0, ~2,900 LoC for the core agents)
is the logo-less version of that. Same tools. Same models. Same
engineering rigor. One natural-language prompt instead of a startup.

A non-exhaustive list of AI products that customers have already
mimicked with a single prompt inside the IDE or CLI:

• **Cursor / Claude Code / Codex.** Sorcar already *is* a coding
  agent — it scores **62.2% on Terminal Bench 2.0** with Claude
  Opus 4.6, ahead of Claude Code (58%) and Cursor Composer 2 (61.7%),
  with no benchmark-specific tuning of the prompt or model.

• **Lovable / v0 / Bolt / Replit Agent.** "Scaffold a Next.js SaaS
  with auth, Stripe, and a Supabase backend, run the tests, deploy
  to Vercel, open the live URL in the browser." The streaming shell,
  Playwright browser, and worktree-per-task isolation make this a
  one-prompt workflow.

• **Perplexity / GPT Researcher / Deep Research.** The system prompt's
  `<web_research>` protocol forces the agent to visit at least 10
  distinct sources, track them in a counter file, and cite them. Combined
  with the Playwright browser tool and parallel sub-agents, you get a
  research engine without writing one.

• **Intercom Fin / Decagon / Ada.** Sorcar ships 23 third-party
  messaging agents (Slack, Gmail, WhatsApp, SMS, phone-control,
  iMessage, Telegram, Discord, Microsoft Teams, and more) plus MCP
  connectors for CRMs. A single prompt — "answer support tickets in
  my brand voice, reading the knowledge base, and escalate edge cases
  to #cx-escalations" — gets you a Fin-like customer agent without
  buying one.

• **Clay / 11x / Artisan (AI SDR).** Browser scout for prospect data,
  parallel sub-agents for per-account research, a stronger model for
  personalization, the Gmail/SMS agents for outbound. One prompt.

• **Granola / Otter / Fireflies (AI notetakers).** `Bash` to capture
  computer audio, Whisper for transcription, a summarizer sub-agent,
  Gmail follow-ups, calendar MCP for context. Same product, one prompt.

• **Manus / Genspark / OpenClaw (horizontal task agents).** Sorcar is
  literally the same primitive: Bash + browser + parallel sub-agents
  + persistent chat sessions + worktree isolation. Same surface area.

• **AlphaEvolve / FunSearch / OpenEvolve / ShinkaEvolve (AI Discovery).**
  Ships as a sample task in `SAMPLE_TASKS.md`: one prompt drives an
  evolutionary code search with a held-out 20% eval split, explicit
  numeric stopping rules, anti-reward-hacking clauses, and an HTML
  report — no harness, no programs database, no orchestrator.

• **DSPy / GEPA / PromptBreeder (prompt optimizers).** The shipped
  GEPA Prompt Optimizer template runs the full Pareto-frontier
  algorithm against any `ChatSorcarAgent`, reading trajectory events
  from `sorcar.db` for reflective rewrites. No Python library to wire.

• **KernelBench-style repo optimizers.** The Repository Optimization
  template runs your benchmark in the background, watches streaming
  metrics, edits, reverts via worktrees, journals failed ideas, and
  stops only when your numeric targets are hit.

So why can a single prompt actually replace a product team?

Three reasons, all from §2 of the paper:

1. **A *relentless* loop with excellent software engineering
   discipline.** The Relentless Agent automatically continues across
   context windows by writing a chronologically ordered structured
   summary at the boundary, so a multi-hour or multi-day task does
   not die when the context window does. The system prompt enforces
   the boring things that make code actually work: read-before-modify,
   lint and typecheckers (fixing pre-existing errors too), 100% branch
   coverage, end-to-end tests only (no mocks, no stubs, no structural
   assertions), parallel test shards, explicit anti-reward-hacking and
   anti-fabrication clauses, and a mandatory 10-site web-research
   protocol. The same prompt also forces an HTML report and an explicit
   numeric stopping rule for open-ended tasks.

2. **On-the-fly model switching with `set_model`.** Every Sorcar Agent
   ships a `set_model` tool that swaps the underlying LLM *mid-task*
   without restarting the conversation. The framework copies the
   transcript and budget counters, rebuilds the tool schema in the new
   provider's dialect, and resumes. In practice this means one prompt
   can run a *cheap-scout → strong-builder → independent-reviewer*
   cascade across vendors: a cheap Gemini reads the repo, Claude Opus
   writes the patch, GPT-5.5 from a different vendor audits the diff
   and runs the tests, all in the same conversation under one budget.
   That is the difference between a brittle single-model agent and one
   that catches its own mistakes.

3. **504 models across 9 provider categories**, plus any
   OpenAI-compatible local endpoint via `--endpoint`/`--header`. Pick
   the right model for every step, not for the whole session.

The rest is a small but complete tool layer (Bash with streaming and
cancel, Read/Edit/Write, Playwright browser, parallel sub-agents,
Docker isolation, MCP servers, Agent Skills, 23 third-party messaging
agents, ask-user-question) and a Worktree Sorcar Agent that runs every
task on its own git branch so experiments are reproducible and
rollback-friendly.

The framework is Apache-2.0, distributed on PyPI as
`kiss-agent-framework`, local-first, and bring-your-own-key — your
prompts and code go directly to the model provider you choose, not
through any intermediary server.

📄 Paper: github.com/ksenxx/kiss_ai/blob/main/papers/kisssorcar/kiss_sorcar.pdf
🐙 Repo: github.com/ksenxx/kiss_ai

#AI #LLM #AgenticAI #SoftwareEngineering #DeveloperTools #OpenSource

---

## Option B — narrative / "build vs. prompt" framing

**The build-vs-prompt question is the new build-vs-buy.**

For the last two years, every team I talk to has been quietly drafting
a spec for some "AI thing" that is, on inspection, a model behind a
prompt with a couple of tool calls and a logo. Some of those specs are
real products. Most are not.

Before you greenlight the next one, run the build-vs-prompt test: can
KISS Sorcar mimic the product from a single chat message?

A surprising number of products fail that test (i.e. they *can* be
mimicked), including:

• **AI coding assistants** (Cursor, Claude Code, Codex, Devin, Aider,
  OpenHands). Sorcar already is one — 62.2% on Terminal Bench 2.0,
  ahead of Claude Code (58%) and Cursor Composer 2 (61.7%), with no
  benchmark-specific tuning.

• **AI app builders** (Lovable, v0, Bolt, Replit Agent). "Scaffold a
  CRUD SaaS, write the tests, deploy, screenshot the live URL." One
  prompt.

• **Deep research engines** (Perplexity, GPT Researcher, NotebookLM).
  The system prompt's 10-source web-research protocol with a tracked
  counter file plus the Playwright browser plus parallel sub-agents
  is, in practice, the same product.

• **Customer-support agents** (Intercom Fin, Ada, Decagon). The
  bundled Slack, Gmail, WhatsApp, SMS, iMessage, Telegram, and phone-
  control agents give the channel coverage; one prompt gives the
  policy.

• **AI SDR / outbound tooling** (Clay, 11x, Artisan). A prompt that
  drives the browser to scout prospects, parallel sub-agents to
  research, a stronger model for personalization, the Gmail agent for
  send.

• **AI notetakers** (Granola, Otter, Fireflies). Capture audio with
  the streaming Bash tool, transcribe, summarize, email follow-ups.

• **Horizontal task agents** (Manus, Genspark, OpenClaw, Poke). Same
  primitives. Same surface area.

• **Evolutionary AI discovery** (AlphaEvolve, FunSearch, OpenEvolve,
  ShinkaEvolve) and **prompt optimizers** (DSPy, GEPA, PromptBreeder).
  Already shipped as sample tasks in `SAMPLE_TASKS.md` — single
  prompts that drive the entire loop, with explicit numeric stopping
  rules, anti-reward-hacking clauses, and HTML reports.

Why one prompt is enough comes down to three properties of the agent
framework, all documented in §2 of the paper:

(i) The **Relentless Agent** wraps the inner loop with automatic
continuation. When a sub-session runs out of context window or step
budget, it writes a structured chronologically-ordered summary with
code snippets, then resumes from that summary in a fresh sub-session.
Multi-hour and multi-day runs survive the context-window boundary; the
agent does not give up. The system prompt around it encodes ruthless
software-engineering discipline (lint + typecheck, end-to-end tests
only, 100% branch coverage, parallel test shards, anti-reward-hacking,
anti-fabrication, mandatory web research), which is what stops the
output from devolving into AI slop.

(ii) **On-the-fly model switching** via the `set_model` tool every
Sorcar Agent ships. The agent can swap the underlying LLM *mid-task*
without restarting the conversation — the transcript and usage
counters are copied, the tool schema is rebuilt in the new provider's
dialect, and the loop continues. This makes patterns like *cheap-scout
→ strong-builder → independent-reviewer-from-a-different-vendor*
expressible as a single sentence in the prompt rather than as a
multi-service orchestration. It is the single most under-appreciated
reliability feature in the framework: most "wrong answers" go away the
moment an independent model from a different vendor is forced to
re-read the diff and run the tests.

(iii) A small but complete tool layer: streaming Bash with cancel,
Read/Edit/Write, Playwright browser automation, parallel sub-agents
in a thread pool, Docker isolation, MCP servers, Agent Skills, 23
third-party messaging agents (Slack, Gmail, WhatsApp, SMS, phone-
control, iMessage, Telegram, Discord, Teams, …), ask-user-question,
worktree-per-task git isolation, and a 504-model catalog spanning
OpenAI, Anthropic, Gemini, Together, Z.AI, Moonshot, OpenRouter,
Claude Code CLI, and Codex CLI — plus any local OpenAI-compatible
endpoint via `--endpoint`/`--header`.

Net effect: most "we need to build a custom AI X" projects are
actually "we need to write a really good prompt for X." Free your
engineers; ship the prompt; iterate in the chat.

Apache-2.0. Local-first. Bring your own key.

📄 Paper: github.com/ksenxx/kiss_ai/blob/main/papers/kisssorcar/kiss_sorcar.pdf
🐙 Repo: github.com/ksenxx/kiss_ai

#AI #LLM #AgenticAI #BuildVsBuy #DeveloperProductivity #OpenSource
