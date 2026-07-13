# Hand-Rewrite Guide — for the founder's own-words HN comment

> **Why this file exists.** HN's Show HN tips (item 22336638, edited
> 2026-03-28) require the submission text to be written by hand: *"Don't use
> an LLM to generate any of it (not even a tiny bit, including to edit or
> spruce it up)."* So this guide deliberately contains **no prose to copy** —
> only a beat checklist, verified facts, and evidence. Write the comment in a
> plain text editor, from scratch, in your own voice. Do not paste any
> sentence from SHOW_HN_DRAFT.md, and do not run your text through any AI
> tool afterwards, even "just to polish it".

## How to use this guide

1. Read SHOW_HN_DRAFT.md once for coverage, then close it.
2. Open a blank text file. Write the 8 beats below in order, in your own
   words, aiming for ≤ 450 words total.
3. Check every number you type against the "Verified facts" table — the
   comment must contain only claims verified in this repo.
4. Sleep on it, cut 20–30%, and keep it factual and plain.

## The 8 beats (content checklist, not text)

1. **Intro + one credibility line.** Real name; that you teach/do
   program-analysis research at UC Berkeley (CUTE/concolic testing lineage);
   one sentence stating what KISS Sorcar is (free, Apache-2.0, local-first
   agent framework: VS Code extension + CLI + web/mobile app; core agents
   ~2,850 lines of Python).
2. **Backstory.** The real reason you built it: wanting an agent you could
   trust with long-horizon tasks without routing code/prompts through
   someone else's backend, and small enough to actually audit.
3. **Five mechanism-anchored bullets** — each must carry a number or a
   mechanism, never a bare feature name:
   - BYOK / no servers: prompts go straight from your machine to the
     provider or local endpoint you configure.
   - One task can mix models: 530 models across 9 provider categories,
     catalog checked in as JSON (prices, context lengths, capability flags).
   - AI discovery / GEPA-style prompt optimization is a paragraph of prompt,
     not a plugin.
   - Git-worktree-per-task isolation: parallel agents don't collide;
     everything is diffable/revertable.
   - 23 messaging agents (Slack, Gmail, WhatsApp, SMS, ...) so the agent
     reports back where you already are.
4. **Quickstart:** `pipx install kiss-agent-framework` — no signup, needs
   Python 3.13+ and one provider API key. (Command strings and the title are
   short factual strings; still retype them yourself.)
5. **Honest limitations:** Python 3.13+ only (3.12 fails cleanly at install —
   verified, see evidence below); single maintainer; no published SWE-bench
   score yet; web/mobile app younger than the CLI; model catalog goes stale
   between releases.
6. **Meta-disclosure paragraph.** You gave Sorcar the task of drafting this
   launch post; HN rules require hand-written text, so you rewrote it
   yourself; the agent's draft, spec, and review diff are published in the
   repo under `marketing/agent-markets-itself/` — and that experiment is half
   the reason you're posting. (Ensure this directory is publicly visible
   before submitting; see LAUNCH_CHECKLIST.md blocker.)
7. **Links block:** repo https://github.com/ksenxx/kiss_ai · paper
   https://arxiv.org/abs/2604.23822 · docs
   https://kisssorcar.github.io/docs/index.md · llms.txt
   https://kisssorcar.github.io/llms.txt (all verified returning 200 on
   2026-07-12).
8. **Specific feedback ask:** the security model (what the agent may touch
   by default) and whether the small-core claim holds up when people read
   the source; say you'll be around for the next couple of days.

## Title

Recommended (retype it yourself; ≤ 80 chars):

    Show HN: KISS Sorcar – open-source agent framework, ~2,850 LoC core, 530 models

Submission URL: `https://github.com/ksenxx/kiss_ai` (the repo, not the
landing page). Alternates are in SHOW_HN_DRAFT.md.

## Verified facts you may state (evidence-backed, 2026-07-12)

| Claim | Evidence |
|---|---|
| `pipx install kiss-agent-framework` works first try on a clean machine | Docker `python:3.13-slim`: `pip install pipx && pipx install kiss-agent-framework` → kiss-agent-framework 2026.7.18 installed; `sorcar --help` runs |
| Install completes in well under 5 minutes | Timed end-to-end in the clean container: **34 seconds** |
| Works on Python 3.14 too | Docker `python:3.14-slim`: same install passes |
| Python 3.12 is unsupported and fails gracefully | Docker `python:3.12-slim`: pip reports "No matching distribution found for kiss-agent-framework" (clear error, no crash) |
| 530 models / 9 provider categories | README.md model catalog table (OpenAI 84, Anthropic 14, Gemini 24, Together 79, Z.AI 8, Moonshot 6, OpenRouter 303, cc/* 3, codex/* 9) |
| ~2,850 LoC core agents | README.md |
| 23 messaging agents | README.md; src/kiss/agents/third_party_agents/ |
| Apache-2.0, free, BYOK, local-first | LICENSE, README.md |
| arXiv 2604.23822 | https://arxiv.org/abs/2604.23822 returns 200 |
| docs + llms.txt live | kisssorcar.github.io/docs/index.md and /llms.txt return 200 |

One footnote for honesty if asked in-thread: pipx prints a PATH warning on a
fresh machine (`/root/.local/bin` not on PATH); `pipx ensurepath` fixes it.

## Final pre-post self-check

- [ ] Every sentence written by me, by hand, in one sitting or more — never
      pasted, never AI-edited.
- [ ] ≤ 450 words; no superlatives; no exclamation marks.
- [ ] Every number matches the Verified facts table.
- [ ] The meta paragraph's repo link resolves publicly (blocker cleared).
- [ ] I'm free for the next 48 hours to reply personally to every comment.
