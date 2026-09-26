---
title: INJECTIONS.md bundled tricks for the Inject instruction panel
uuid: ee09f356-55d0-48f1-9bc1-6d2fee0cb82b
summary: INJECTIONS.md bundled Trick snippets for the Inject instruction panel and
  ghost text, merged after ~/.kiss/MY_INJECTION.md by server/tricks.py read_tricks.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# INJECTIONS.md bundled tricks

`src/kiss/INJECTIONS.md` is **not** part of the system prompt. It is a list of reusable instruction snippets,
each under a `## Trick` heading (8 at the time of writing). They appear in the UI's "Inject instruction"
panel and in ghost-text fast-complete suggestions.

## Loading (`src/kiss/server/tricks.py`)
- `read_tricks()` returns one ordered list. User tricks from `~/.kiss/MY_INJECTION.md` come first, then the bundled ones.
- `MY_INJECTION.md` is seeded on first read with a single starter trick ("Write end-to-end 100% coverage tests
  for the feature first. Then implement the feature.") and is never overwritten afterwards.
- The bundled file is read **directly from the package**, never copied into `~/.kiss/`, so upgrades deliver
  new tricks without clobbering user edits. The `KISS_INJECTIONS_PATH` env var overrides the path (tests use it).
- Markdown backslash escapes are parsed the same way as `unescapeMarkdown` in `SorcarTab.ts`. Adding a trick
  from the UI is serialized with `_APPEND_LOCK`.

## Typical bundled tricks
- "Use '<builder model>' for all tasks ... use '<reviewer model>' via `run_parallel` for a read-only review ...
  at most N% of the budget ... do not invent new problems": the builder/reviewer recipe, in several model variants.
- "Reproduce any violation of the invariant by writing end-to-end tests with 100% coverage. Then fix the issue."
- "git pull origin/<current-branch>, merge, and push".
- A channel-authentication trick: check existing credentials first, never drive sign-in pages with the
  agent's browser, never ask for a password or 2FA code.

When model names change, update the tricks too. They name concrete models such as `claude-fable-5-1` and `gpt-6-sol`.

## Sources
- `src/kiss/INJECTIONS.md`
- `src/kiss/server/tricks.py` (`read_tricks`, `_APPEND_LOCK`)
