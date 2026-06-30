# Twitter / X posts — "Do you really need to build that AI product?"

Theme: most "AI software" you're tempted to build is a thin wrapper over
`tools + model + a prompt`. KISS Sorcar already gives you the tools, 504
models from 9 vendors with mid-task model switching, and rigorous SW-eng
in the system prompt. So you can *mimic* the product with a single prompt
instead of shipping a startup.

Grounded in `papers/kisssorcar/kiss_sorcar.tex` (§2 architecture, §3
AI-Discovery / GEPA / Repo-Optimization templates, §4 Terminal Bench 2.0
results, §5 features).

______________________________________________________________________

## Option A — short hook (recommended)

Do you actually need to *build* that AI product?

Or can KISS Sorcar mimic it from one prompt?

— Cursor / Claude Code → it's already a coding agent (62.2% on
Terminal Bench 2.0, beats Claude Code 58% and Cursor Composer 2 61.7%)
— Lovable / v0 / Bolt → "build me a SaaS habit tracker, ship to Vercel"
— Perplexity / Deep Research → web_research protocol + Playwright
— Intercom Fin → Slack/Gmail/WhatsApp/SMS third-party agents +
knowledge prompt
— Clay / 11x → Claygent = browser scout + Gmail sequencer in one prompt
— Granola → bash + Whisper + Gmail follow-ups
— AlphaEvolve / FunSearch / DSPy-GEPA → already shipped as sample tasks

Why it works: a relentless agent that auto-continues across context
windows, *on-the-fly model switching* across 504 models from 9 vendors
(cheap scout → strong builder → independent reviewer in one task), and
a system prompt that enforces lint, types, end-to-end tests, no AI slop.

Apache-2.0. Local-first. BYO key.
github.com/ksenxx/kiss_ai

______________________________________________________________________

## Option B — thread, 5 tweets

1/ "We need to build a custom AI X for our team."

Before you hire engineers and burn 6 months, ask yourself: is X really
anything more than `tools + a model + a prompt`?

KISS Sorcar gives you the tools and 504 models. You write the prompt.

2/ Things people have asked Sorcar to *mimic* with one chat prompt:

🧑‍💻 Cursor / Claude Code — it already is one (62.2% on TB 2.0)
🛠 Lovable / v0 — "scaffold a Next.js dashboard, deploy, open browser"
🔍 Perplexity — Playwright browser + 10-site research protocol
🧾 Granola — capture audio, transcribe, email follow-ups
📞 Intercom Fin — Slack + Gmail + WhatsApp + SMS + MCP
💼 Clay / 11x — browser scout + Gmail sequencer + Anthropic personalize
🧬 AlphaEvolve / DSPy-GEPA — shipped as `SAMPLE_TASKS.md` templates

3/ Why one prompt is enough?

Sorcar Agent ships *real* tools: streaming Bash with cancel, Read/Edit/
Write, Playwright browser, parallel sub-agents, Docker sandbox, MCP, 23
third-party messaging agents (Slack, Gmail, WhatsApp, SMS, phone, …),
and a `set_model` tool that swaps the LLM *mid-task* without restarting.

4/ Why the output isn't slop?

The system prompt encodes the boring stuff that makes products work:
read-before-modify, lint + typecheck (fix pre-existing too), 100%
branch coverage, end-to-end tests only (no mocks), parallel test
shards, anti-reward-hacking, anti-fabrication, 10-site web research.

Relentless Agent continues across context windows so multi-hour runs
don't just die at the boundary.

5/ Cheap model scouts the repo.

Strong model writes the code.

Independent model from a different vendor reviews the diff and finds the
bugs the first model missed.

All in one task. One conversation. One budget. One sentence in the prompt.

Apache-2.0 · 504 models · BYO key · github.com/ksenxx/kiss_ai

______________________________________________________________________

## Option C — punchier listicle (single tweet)

You don't need to ship a custom AI product. You need one prompt.

KISS Sorcar mimics:
• Cursor (it's already a coding agent — 62.2% TB 2.0)
• Lovable / v0 / Bolt
• Perplexity / Deep Research
• Intercom Fin (Slack + Gmail + WhatsApp + SMS shipped)
• Clay / 11x (browser scout + outreach)
• Granola (bash + Whisper + email)
• Manus / Genspark
• AlphaEvolve / FunSearch
• DSPy / GEPA prompt optimizer

How? Relentless cross-context continuation + on-the-fly switching across
504 models from 9 vendors + a system prompt that enforces real
engineering (lint, types, e2e tests, anti-slop). Apache-2.0.

github.com/ksenxx/kiss_ai

______________________________________________________________________

## Option D — provocateur

Most "AI software" is a prompt with a logo on top.

KISS Sorcar is the logo-less prompt runner.

— `Build a Lovable-style SaaS scaffolder and deploy to Vercel.` ✅
— `Be my Perplexity: cite 10 sources before answering.` ✅
— `Triage my Gmail like Fin, reply in my voice, escalate edge cases.` ✅
— `Run AlphaEvolve on my Python file with this fitness function.` ✅
— `GEPA-optimize this agent's prompt on my dataset.` ✅

Why does this work? Mid-task `set_model` lets a cheap model scout and a
frontier model edit and a different-vendor model review — same
conversation, same budget. Plus a system prompt that refuses to ship
without tests and a Relentless loop that survives 12-hour runs.

Apache-2.0, local-first, bring your own keys.
github.com/ksenxx/kiss_ai
