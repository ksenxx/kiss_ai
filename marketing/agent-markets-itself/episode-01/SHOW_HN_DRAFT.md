# Show HN Launch Package — drafted by KISS Sorcar (the agent), v2 (refined)

> **Disclosure:** This package was drafted by KISS Sorcar itself, running as an
> agent, as Episode 1 of "The Agent That Markets Itself". Per HN's 2026 rule
> against LLM-written submission text, the version actually posted to HN must
> be rewritten by hand by the founder. This draft is the published episode
> artifact and the founder's content checklist. See SPEC.md and EPISODE.md.

## Submission URL

https://github.com/ksenxx/kiss_ai

(Link the repo, not the landing page. The repo is the landing page; stars
compound.)

## Title candidates (pick one; ≤ 80 chars)

1. `Show HN: KISS Sorcar – open-source agent framework, ~2,850 LoC core, 530 models` *(recommended: two concrete numbers, states category)*
2. `Show HN: KISS Sorcar – BYOK agent that codes, browses, and messages (Apache-2.0)`
3. `Show HN: KISS Sorcar – a small open-source alternative to Claude Code and Cursor`
4. `Show HN: An agent framework where one task can mix models from nine providers`
5. `Show HN: KISS Sorcar – local-first AI agent for VS Code, CLI, and web`

## First comment (v2, 418 words)

Hi HN, I'm Koushik Sen. I teach and do program-analysis research at UC
Berkeley (CUTE/concolic testing, DART lineage). KISS Sorcar is a free,
open-source (Apache-2.0), local-first agent framework I built: it runs as a
VS Code extension, a Claude-Code-style CLI, and a web/mobile app, and the
core agents are about 2,850 lines of Python you can read in an afternoon.

Backstory: I wanted an agent I could trust with long-horizon tasks — refactor
a package, research a topic across 10 sites, answer my Slack — without routing
my code and prompts through someone else's backend, and without a codebase so
large I couldn't audit what the agent was allowed to do. Existing tools were
either closed, single-vendor, or huge. So I applied the KISS principle and
kept the core small enough to keep honest.

What's technically different:

- Bring your own key. Prompts and code go directly from your machine to the
  provider or local endpoint you configure. Nothing passes through my servers;
  there are no servers.
- One task can mix models from different vendors. The bundled catalog has 530
  models across 9 provider categories (OpenAI, Anthropic, Gemini, Together,
  Z.AI, Moonshot, OpenRouter, and the Claude Code and Codex CLIs), with
  prices, context lengths, and capability flags checked in as JSON.
- Techniques that are normally frameworks are just prompts here: AI
  discovery and GEPA-style prompt optimization run from a paragraph of
  instructions, not a plugin.
- Every task runs in its own git worktree, so parallel agents don't step on
  each other and everything is diffable and revertable.
- 23 messaging agents (Slack, Gmail, WhatsApp, SMS, Signal, IRC, phone
  control, ...) so agents can report back where you already are.

Quickstart (no signup; you need Python 3.13+ and one provider API key):

    pipx install kiss-agent-framework

Limitations, honestly: Python 3.13+ only; single maintainer; no published
SWE-bench score yet (running it is on the list); the web/mobile app is
younger than the CLI; the 530-model catalog goes stale between releases.

One meta note: I gave Sorcar the task of drafting this launch post. HN rules
say submission text must be written by hand, so I rewrote it myself — but the
agent's draft, the spec I gave it, and the diff are published in the repo
under marketing/agent-markets-itself/. That experiment is half the reason I'm
posting.

Paper: https://arxiv.org/abs/2604.23822 · Docs: https://kisssorcar.github.io/docs/index.md
· llms.txt: https://kisssorcar.github.io/llms.txt

I'd especially value feedback on the security model (what an agent may touch
by default) and on whether the small-core claim holds up when you read the
source. I'll be here for the next couple of days.

## What changed from v1 → v2 (review annotations)

Each edit maps to a research finding; see REVIEW.md for the finding sources.

1. **Cut 31%** (609 → 418 words). Finding: successful drafts get ~30% shorter
   in editing; cut everything that repeats the README.
2. **Title candidates rewritten to be concrete/technical.** v1 had
   "Show HN: KISS Sorcar – The open-source AI agent that does everything"
   → disqualified (marketing-speak, unverifiable). Replaced with
   number-anchored titles per the Layerform/PromptTools pattern.
3. **Removed the feature-list-only middle.** v1 bullets read like the Orcbot
   anti-example (buzzwords, no story). v2 anchors every bullet in a number or
   a mechanism (worktrees, JSON catalog, paragraph-of-prompt).
4. **Added the credibility line** ("CUTE/concolic testing, DART lineage") —
   the Mastra "Previously we built Gatsby" move, one line only per spec.
5. **Added honest-limitations block** including "no published benchmark yet"
   and "single maintainer" — syften/dang: a small useful thing beats an
   inflated claim; front page needs something to argue with.
6. **Added the dogfooding/meta paragraph with explicit hand-rewrite
   disclosure** — resolves the dang-2026 LLM-text rule by making transparency
   the hook instead of a violation.
7. **Specific feedback ask** (security model, small-core claim) replaces v1's
   generic "let me know what you think" — syften: say what feedback helps.
8. **Quickstart switched** from the `curl | bash` installer to
   `pipx install kiss-agent-framework` — HN is hostile to `curl | bash`;
   pipx is the lower-friction, higher-trust path.
9. **Removed the P. C. Sorcar name story** — cost 2 sentences, spec allows
   at most 1; it survives in the README where curious readers will find it.
