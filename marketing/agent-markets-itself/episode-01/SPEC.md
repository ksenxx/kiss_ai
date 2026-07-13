# SPEC — Episode 1: KISS Sorcar writes its own Show HN launch post

This is the natural-language brief given to the agent (KISS Sorcar) before it
drafted its own Show HN submission. Per the agent-native GTM pattern, the
human owns the spec; the agent executes; the human reviews the *output*
against the spec, and iterates the spec (not the prompt) between drafts.

## Goal

Produce a complete Show HN launch package for KISS Sorcar:

1. 3–5 candidate titles beginning with `Show HN:`.
2. The submission URL (the GitHub repo, not a landing page).
3. A full first-comment text (the founder's top-of-thread comment) ready for
   human review and hand-rewrite.

## Hard constraints

- **Factual, direct language only.** No marketing-speak, no superlatives, no
  hype, no exclamation marks, no "revolutionary/game-changing/blazing".
- Every claim must be verifiable from the repo, README, or arXiv paper.
  Concrete numbers over adjectives (530 models, 9 provider categories,
  ~2,850 LoC core agents, 23 messaging agents, Apache-2.0, Python 3.13+).
- Title is concrete and technical, in the pattern
  `Show HN: <Name> – <what it does, specifically>`.
- First comment must include, in roughly this order:
  1. Personal intro (who I am, real name) and one-sentence statement of what
     the project does.
  2. Backstory: why I built it, what problem/constraint led to it.
  3. What is technically different/interesting (multi-model-in-one-task,
     prompt-level AI discovery/GEPA, small core, BYOK/local-first,
     worktree-per-task isolation).
  4. A one-command quickstart that works without signup.
  5. Honest limitations, stated plainly.
  6. A dogfooding/meta note: the agent drafted this launch post itself and
     the full episode (spec, drafts, review) is published.
  7. Links block: repo, arXiv, docs/llms.txt, previous relevant threads if any.
  8. A specific ask for feedback.
- Cut ruthlessly: target ≤ 450 words for the first comment; shorter beats
  longer by ~30%.
- No em-dash-heavy or listicle-cadence "AI voice"; plain sentences.
- Compliance with HN Show HN guidelines: tryable without signup, personal
  work, no upvote solicitation, be around to reply for 48 hours.

## Soft preferences

- Lead the backstory with the researcher angle (program analysis / testing
  research at Berkeley) only briefly; one credibility line, not a bio.
- Prefer "small and auditable" framing over "powerful" framing.
- Mention the name story (P. C. Sorcar) only if it costs < 1 sentence.

## Positive examples (emulate the structure, not the words)

1. **Mastra (442 pts)** — intro + names, credibility line ("Previously we
   built Gatsby"), pain backstory, one-command quickstart, deep technical
   detail, honest license note, links block.
2. **Superset (96 pts / 90 comments)** — "Hey HN, we're X, Y, Z" intro,
   4 concrete feature bullets with footnote links, dogfooding claim ("We use
   Superset to build Superset"), tech-stack rationale, honest lessons.
3. **AnythingLLM (368 pts)** — key-learnings bullets, "privacy by default is
   non-negotiable" positioning.

## Disqualifiers (instant rejection of the draft)

1. Marketing-speak, sales language, or unverifiable claims
   ("best", "fastest", "the future of").
2. Buzzword bullet lists with no story, demo, or numbers (the Orcbot
   anti-pattern: 4 points, 0 comments).
3. Any suggestion of upvote solicitation, booster comments, or hiding the
   AI's involvement in drafting.

## Post-draft rule (the dang 2026 constraint)

HN's official Show HN tips (news.ycombinator.com/item?id=22336638, edited
2026-03-28) now say: *"Write your text by hand. Don't use an LLM to generate
any of it (not even a tiny bit, including to edit or spruce it up)."*
Therefore: the agent's draft below is the **episode artifact**, published
transparently. The text actually submitted to HN must be **rewritten by hand
by the human founder**, using the draft only as a content checklist. The
final hand-rewrite is out of scope for the agent.
