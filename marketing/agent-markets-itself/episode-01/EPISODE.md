# Episode 1 — "I gave my agent one job: write its own Show HN post"

*The Agent That Markets Itself, a build-in-public series about KISS Sorcar —
an open-source agent framework — doing its own marketing, with every prompt,
draft, and edit published. Everything below actually happened; the session
logs live in the Sorcar task database.*

---

## The setup

KISS Sorcar can code, browse, message on 23 platforms, talk, and control a
computer. Its competitors (Cline, OpenHands, Aider) are coding agents; they
can't credibly market themselves. Sorcar can. So this series gives Sorcar its
own marketing tasks, publishes the raw artifacts, and discloses everything.

Episode 1's task, verbatim:

> Draft and record the first episode of the "Agent That Markets Itself"
> series by having KISS Sorcar write its own Show HN launch post, then review
> and refine it for the actual 48-hour launch window.

## Act 1 — The agent does its homework (10 sources)

Before writing a word, the agent researched how Show HN actually works in
2026. It visited ten sources and kept a counter file. The highlights it
extracted:

1. **Official Show HN guidelines** — must be tryable without signup; must be
   personal, non-trivial work; title starts `Show HN:`; never solicit
   upvotes.
2. **dang's official tips thread (item 22336638)** — backstory + what's
   different; factual direct language ("drop any language that sounds like
   marketing — on HN, that is an instant turnoff"); personal username; email
   in profile. And one line, added in an edit dated 2026-03-28, that changed
   this whole episode — more on that in Act 4.
3. **syften.com's 2026 HN guide** — first comment = who/why + the specific
   problem + what's technically interesting + what feedback helps. Timing
   matters only so you can spend the next hours replying (9am–12pm ET,
   weekday). "On Hacker News, the product is rarely the story. The
   interesting thing you learned while building it might be."
4. **lucasfcosta.com** — titles must be specific and technical
   ("Show HN: Layerform – Open-source development environments using
   Terraform files", not "...a staging environment for each engineer");
   link the repo, not a landing page; cut ~30% of words.
5. **HN Algolia, 223 "agent framework" Show HNs** — the winners share an
   anatomy: personal intro with names, one credibility line, pain backstory,
   one-command quickstart, deep technical specifics, honest limitations,
   dogfooding claim, links block, specific feedback ask. Mastra (442 pts),
   AnythingLLM (368), Superset (96 pts / 90 comments with "we use Superset
   to build Superset"). And the anti-example: Orcbot, a buzzword bullet
   list with no story or numbers — 4 points, 0 comments.

## Act 2 — The spec, not the prompt

Following the agent-native GTM pattern (human owns a 300–800-word spec;
agent executes; human reviews outputs against the spec), the founder wrote
[SPEC.md](SPEC.md): hard constraints (factual language only, every claim
verifiable, ≤450 words, quickstart + limitations + disclosure required),
three positive examples (Mastra, Superset, AnythingLLM), three disqualifiers
(marketing-speak, buzzword bullets, hiding AI involvement).

## Act 3 — Draft v1, and why it was rejected

The agent's first pass had the right bones but tripped two disqualifiers:

- Title: *"Show HN: KISS Sorcar – The open-source AI agent that does
  everything"* — "does everything" is unverifiable marketing-speak, and dang's
  tips call that an instant turnoff. Rejected.
- The middle was a feature list ("multi-agent orchestration", "AI discovery",
  "530 models") with no mechanism or story attached — structurally identical
  to the Orcbot anti-example that scored 4 points.
- 609 words, most of them restating the README, which the research explicitly
  warns against ("don't repeat the landing page").

The review didn't edit the draft line-by-line; it updated the spec (added the
≤450-word cap, the "every bullet needs a number or a mechanism" rule, and the
specific-feedback-ask requirement) and re-ran the draft.

## Act 4 — The twist: HN banned the premise

While researching, the agent surfaced this, from dang's official Show HN tips,
edited 2026-03-28:

> "Write your text by hand. Don't use an LLM to generate any of it (not even
> a tiny bit, including to edit or spruce it up). … the community is super
> fussy about this right now … This is a big dividing line at present!"

So the centerpiece of Episode 1 — an agent-written Show HN post — is
disallowed as actual HN submission text. An agent whose launch post
researched itself into a rule against itself.

The resolution the agent proposed (and the founder adopted):

1. The agent's draft is the **episode artifact**, published in full in this
   directory with complete disclosure.
2. The founder **rewrites the HN text by hand**, using the draft only as a
   content checklist (what to cover, in what order, with which numbers).
3. The HN comment itself discloses the experiment and links here — turning
   the constraint into the story. Transparency posts perform on HN
   (Anthropic's post-mortem, Zed's parallel-agents thread), and per syften:
   the product is rarely the story; the interesting thing you learned is.

## Act 5 — Draft v2

The refined package — 5 title candidates, submission URL, a 418-word first
comment with credibility line, mechanism-anchored bullets, pipx quickstart,
honest limitations (single maintainer, no benchmark yet), the meta
disclosure, and a specific feedback ask — is in
[SHOW_HN_DRAFT.md](SHOW_HN_DRAFT.md), with all nine v1→v2 edits annotated
and mapped to research findings.

## Act 6 — Pre-flight verification (2026-07-12)

Before handing off to the human, the agent ran the launch pre-flight itself:

- **Clean-machine quickstart test (Docker):** on `python:3.13-slim`,
  `pip install pipx && pipx install kiss-agent-framework` installed
  2026.7.18 cleanly and `sorcar --help` ran — **34 seconds end-to-end**,
  well inside the 5-minute target. `python:3.14-slim` passes too.
  `python:3.12-slim` fails with a clear "No matching distribution found"
  message, so the draft's "Python 3.13+ only" limitation is verified, not
  assumed.
- **Link check:** repo, arXiv 2604.23822, docs/index.md, and llms.txt all
  return 200. One blocker found: the public repo's `marketing/` path
  404s — the episode directory must be published publicly before the meta
  paragraph's link goes live (tracked in the checklist).
- **Slot picked:** Wednesday 2026-07-15, 9:30am ET (fallbacks Thu 7/16,
  Tue 7/21).
- **Hand-off artifact:** [HAND_REWRITE_GUIDE.md](HAND_REWRITE_GUIDE.md) — a
  beat checklist and verified-facts table with deliberately **no prose to
  copy**, so the founder's hand-written text can't accidentally contain an
  agent sentence.

An agent that runs its own launch QA but is forbidden, by rule, from writing
the launch post: that's the hand-off line for Episode 1.

## Epilogue — what ships and when

The launch itself follows [LAUNCH_CHECKLIST.md](LAUNCH_CHECKLIST.md):
pre-flight repo polish, submit Wed 2026-07-15 9:30am ET, hand-written text,
48 hours of personal replies, then the staggered Reddit/newsletter wave.

**Episode 1 scorecard**

| Item | Status |
|---|---|
| Agent researched Show HN norms (10 sources) | done |
| Founder spec written | done ([SPEC.md](SPEC.md)) |
| Agent draft v1 | done (rejected against spec, 2 disqualifiers) |
| Agent draft v2 | done ([SHOW_HN_DRAFT.md](SHOW_HN_DRAFT.md)) |
| Clean-machine quickstart test | done (34 s on Python 3.13; 3.14 ok; 3.12 fails cleanly) |
| Link pre-flight | done (all 200; one public-repo 404 blocker logged) |
| Submission slot picked | done (Wed 2026-07-15, 9:30am ET) |
| Hand-rewrite guide for the founder | done ([HAND_REWRITE_GUIDE.md](HAND_REWRITE_GUIDE.md)) |
| Hand-rewrite by founder | pending (must be human-written, by rule) |
| Actual HN submission | pending (48-hour launch window) |

*Next episode: the launch itself — what the thread said, what broke, and
whether the small-core claim survived contact with HN.*

---

### A note on video

This written serial is the canonical episode format. A screen-recorded video
version can be produced by replaying this session (task prompt → research →
spec → v1 rejection → v2) in the Sorcar UI and narrating this document as the
script; the section headers above are the shot list.
