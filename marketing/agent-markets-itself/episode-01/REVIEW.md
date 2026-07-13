# REVIEW — v1 → v2 refinement, each edit mapped to a research source

The agent's draft v1 was reviewed against SPEC.md. Two disqualifiers fired
and seven improvements were identified. Rather than line-editing, the spec
was tightened and the draft regenerated (iterate the spec, not the prompt).

## Draft v1 (rejected) — reproduced for the record

> **Title:** Show HN: KISS Sorcar – The open-source AI agent that does everything
>
> Hi HN! I'm excited to share KISS Sorcar, a revolutionary open-source AI
> agent framework. KISS Sorcar is a general-purpose multi-model, multi-modal,
> multi-agent AI framework that can do software development, control a
> computer, research, discover, write papers, create presentations, chat with
> other agents, shop, bank, message, email, browse, and do data science.
>
> Key features:
>
> - Multi-agent orchestration
> - AI discovery and GEPA prompt optimization
> - 530 models across 9 providers
> - 23 messaging integrations
> - Smart home control
> - Local-first and bring-your-own-key
> - VS Code extension, CLI, and web/mobile app
>
> \[... continued for 609 words, largely restating the README feature list,
> ending with "Let me know what you think!"\]

## Rulings

| # | Edit (v1 → v2) | Research source |
|---|---|---|
| 1 | Title "does everything" → number-anchored concrete titles ("~2,850 LoC core, 530 models") | lucasfcosta.com: specific/technical titles (Layerform pattern); dang: marketing language is an instant turnoff — **disqualifier fired** |
| 2 | "I'm excited to share… revolutionary" → plain intro with real name and one-line what-it-does | dang's tips; syften kill-triggers; SPEC hard constraint |
| 3 | Bare feature bullets → every bullet carries a number or a mechanism (JSON catalog, worktree-per-task, paragraph-of-prompt) | HN Algolia survey: Orcbot anti-example (4 pts, 0 comments) vs Superset's concrete bullets — **disqualifier fired** |
| 4 | Added backstory paragraph (trust + auditability constraint that led to the small core) | dang: include backstory, it seeds discussion; syften: the specific problem behind it |
| 5 | Added one credibility line (Berkeley, CUTE/concolic testing) | Mastra's "Previously we built Gatsby" line (442 pts) |
| 6 | Added honest limitations (single maintainer, no benchmark yet, Python 3.13+ only, young web app, catalog staleness) | syften: "a small useful thing beats an inflated claim"; AnythingLLM's honest-learnings pattern |
| 7 | Added dogfooding/meta paragraph disclosing the agent-drafted experiment and the hand-rewrite | Superset's "we use Superset to build Superset"; dang 2026 LLM-text rule turned into the story; forkoff: transparency posts perform |
| 8 | 609 → 418 words | lucasfcosta: cut ~30%; syften: don't repeat the landing page |
| 9 | "Let me know what you think!" → specific asks (security model, small-core claim) | syften first-comment anatomy: state what feedback would help |
| 10 | `curl \| bash` quickstart → `pipx install kiss-agent-framework` | HN community's documented hostility to `curl \| bash`; official guideline: easy to try |
| 11 | Cut the P. C. Sorcar name story (2 sentences) | SPEC soft preference: ≤ 1 sentence or cut; word budget |

## Spec deltas made during review

1. Added hard word cap (≤ 450 words).
1. Added "every bullet needs a number or a mechanism" to hard constraints.
1. Added "specific feedback ask" to the required first-comment elements.

## Verdict

v2 accepted as the **episode artifact and content checklist**. Not accepted
for direct submission: HN requires the submitted text be hand-written by the
human founder (SPEC.md, post-draft rule). The hand-rewrite is a human task.
