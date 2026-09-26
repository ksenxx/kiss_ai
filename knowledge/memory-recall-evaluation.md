---
title: Memory recall evaluation harness (evaluate.py)
uuid: fcfa6266-437c-452d-8b34-901f6efa3427
summary: kiss.core.memoryfield.evaluate builds pages from sorcar.db task_history and
  measures Recall@1/3/5 and MRR for vector, hashed, bm25 (FTS5) and hybrid RRF retrievers,
  with family recall
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Memory recall evaluation harness

`src/kiss/core/memoryfield/evaluate.py` measures whether retrieval surfaces the right page, using
real past Sorcar tasks as the corpus. Run:

```
uv run python -m kiss.core.memoryfield.evaluate --limit 300 --llm-probes 40
```

## Corpus
- `load_past_tasks` opens `~/.kiss/sorcar.db` read-only (`?mode=ro` URI) and takes the newest
  top-level rows of `task_history` (`parent_task_id` NULL or empty) whose `result` is at least 300
  chars, skipping failures (results starting with `Task failed`/`Task interrupted`, or exactly
  `Agent Failed Abruptly`, `Task terminated unexpectedly (process killed)`, `Task stopped by user`).
- `task_page_body` renders `# Task (<date>)`, the request, `# Result` and the HTML result converted
  to text (`html_to_text`), truncated to `MAX_PAGE_BYTES - 700` bytes to leave room for frontmatter.
- `build_memory_from_tasks` makes the directory contain exactly one page per task (extra
  frontmatter key `source: sorcar.db task_history <id>`). Existing pages are not rewritten (that would
  only refresh `updated` and force re-embedding); pages for other tasks are **deleted**, so the
  directory must be dedicated to the evaluation. Default `--memory-dir ./tmp/memoryfield-eval/memory`,
  results in `--out ./tmp/memoryfield-eval/results.json`. Never point it at a real memory dir.

## Retrievers compared
- `vector`: `VectorIndex` with `ModelEmbedder` (`text-embedding-3-small`).
- `hashed`: `VectorIndex` with the offline `hashed_embedding`.
- `bm25`: `KeywordIndex`, an in-memory SQLite FTS5 table (`porter unicode61` tokenizer), query words
  OR-ed.
- `hybrid`: `reciprocal_rank_fusion` of vector and bm25 (constant 60).

## Probes and metrics
Probes are hand-written queries for known tasks (`hand_probes`) plus LLM-written paraphrases of
sampled pages (`llm_probes`, model `DEFAULT_PROBE_MODEL`, cached by `cached_llm_probes`). Each probe
has one gold page; metrics are Recall@k for `RECALL_KS = (1, 3, 5)` and MRR. Because users repeat
tasks, the harness also reports *family* recall: a hit counts if its request text equals the gold
request after whitespace/case normalisation (`task_families`, `family_rank_of`). That approximates
what a curated memory with merged repeats would score.

## Sources
- `src/kiss/core/memoryfield/evaluate.py` (`load_past_tasks`, `task_page_body`, `build_memory_from_tasks`, `KeywordIndex`, `reciprocal_rank_fusion`, `run_evaluation`, `main`)
