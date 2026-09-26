---
title: Building and maintaining the repo knowledge base (knowledge/)
uuid: 9f3ca9f8-0b88-43b1-adb7-b5c169945b7a
summary: How knowledge/ was seeded with parallel area agents, normalized, deduplicated,
  search-tested and fact-checked; memory_dir wiring and gotchas.
created: '2026-09-26T19:11:20Z'
updated: '2026-09-26T19:11:20Z'
---
# Building and maintaining the repo knowledge base (knowledge/)

`knowledge/` is a flat `MemoryDir` of domain pages about this repo, used as the agent memory by
setting `memory_dir` in `~/.kiss/config.json` to its absolute path (`_memory_settings()` in
`src/kiss/agents/sorcar/sorcar_agent.py`). It was seeded on 2026-09-26; this page records how, so it
can be redone or extended.

## Recipe that worked
1. One seeding agent per code area, run in parallel, each with a shared brief: area prefix for page
   names (`core-`, `models-`, `git-`, ...), one topic per page, an `<area>-overview` map page,
   facts verified against current code, `## Sources` with file paths and function names (no line
   numbers), no secrets or dated per-session notes, write only inside `knowledge/`.
2. Normalize: rewrite every page through `MemoryDir.write` so the frontmatter is valid YAML with
   `uuid`/`created`/`updated`. Hand-written summaries containing `: ` are invalid YAML and such
   pages get skipped by the index.
3. `VectorIndex.near_duplicates(0.80)` found cross-area duplicates (same topic written by two
   area agents, e.g. JS tests under both `dev-` and `vscode-`). Merge into one page, delete the
   other, then rewrite backtick references to the deleted name. Overview/detail pairs inside one
   area also score 0.85+ and are not duplicates.
4. Test search with `knowledge/eval/questions.yaml` + `knowledge/eval/search_eval.py`, plus
   blind questions written from the code by a different model.
5. Independent fact-check by another model page by page. The first review found about 60
   wrong or overbroad claims in 50 of 176 pages (typical: "always/never/every" statements that
   the code contradicts in one branch). Budget for this step.

## Gotchas
- The index file `<model>.sqlite3` (plus `-wal`/`-shm`) lives inside the memory dir: gitignored.
  So are `.<page>.md-*` staging files that a killed atomic write can leave.
- Uppercase `README.md` and anything in subdirectories (`eval/`) are not pages
  (`MemoryDir.page_stats` accepts only lowercase-hyphen top-level `*.md`).
- Without `OPENAI_API_KEY` the index falls back to `hashed-bow-v1`, which scored R@5 0.71 vs 1.00
  for `text-embedding-3-small` on the same 73 questions.

## Sources
- `src/kiss/core/memoryfield/pages.py` (`MemoryDir.write`, `MemoryDir.page_stats`)
- `src/kiss/core/memoryfield/index.py` (`VectorIndex.near_duplicates`, `VectorIndex.search`)
- `knowledge/eval/search_eval.py`, `knowledge/README.md`
