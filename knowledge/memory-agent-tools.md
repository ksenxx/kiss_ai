---
title: Memory agent tools (memory_search, memory_pull, memory_write, memory_refresh)
  and MEMORY_PROTOCOL
uuid: 1debdda8-f9f8-4b6b-a58a-e69bcd762e75
summary: MemoryTools memory_search, memory_pull, memory_read, memory_write, memory_list,
  memory_delete, memory_refresh; sync before search, PULL_CHAR_LIMIT, stale pages,
  MEMORY_PROTOCOL
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Memory agent tools and MEMORY_PROTOCOL

`MemoryTools(root, embed=None, model_code=None)` builds a `MemoryDir` and a `VectorIndex` over one
directory. `MemoryTools.tools()` returns the seven bound methods that get registered with
`KISSAgent.run(tools=...)`; `MEMORY_PROTOCOL` is appended to the system prompt alongside them (see
`memory-when-enabled` for the Sorcar wiring).

## Tools
| Tool | Behaviour |
|---|---|
| `memory_search(query, k=5)` | Sync, then search. Output lines: `score  name  —  title: summary`. Returns `No memory pages yet.` for an empty index, else `No matches.` |
| `memory_pull(query, k=3)` | Same search, returns full page text (`### name.md  (score ...)` + raw). Capped at `PULL_CHAR_LIMIT = 24_000` chars: the first page is truncated if it alone exceeds the cap, later pages are reported as omitted. Pages deleted after the sync are skipped. |
| `memory_read(name)` | Raw text of one page; errors returned as `Error: ...` strings. |
| `memory_write(name, content, title, summary)` | `MemoryDir.write` (see `memory-page-format`). Warns when the name looks like a per-round note and when the searchable text exceeds 8192 bytes. |
| `memory_list()` | Every page with title and summary. |
| `memory_delete(name)` | Unlink the page; its index row is removed at the next sync. |
| `memory_refresh(stale_days=30, duplicate_threshold=0.9)` | `sync(verify=True)`, then up to 20 near-duplicate pairs and up to 20 stale pages. |

## Sync on every search
There is no separate reindex step: `_search` runs `index.sync()` before searching, so pages written
by any process (another agent, a human editor, `git pull`) are found. The query embedding is
requested on a worker thread (`ThreadPoolExecutor(max_workers=1)`) while the sync scans and embeds
changed pages, so a search costs about one embedding round trip. An embedding failure re-raises.

## Ephemeral-page warning
`_EPHEMERAL_NAME_RE = (^|-)(round|session|iteration|pass)-?\d+(-|$)` flags names such as
`server-fixes-round-7`. The write still succeeds, but the tool result tells the agent to keep such
notes in `./tmp/PROGRESS.md`. An audit found 36 of 260 pages were such notes, which is why rule 3 of
`MEMORY_PROTOCOL` exists.

## Stale pages
`_stale_pages` parses the `updated` frontmatter with format `%Y-%m-%dT%H:%M:%SZ`; pages with missing
or unparseable `updated` are skipped rather than reported, so the agent is not trained to "fix"
pages that may be fresh.

## MEMORY_PROTOCOL (four rules)
1. `memory_search`/`memory_pull` before starting work.
2. Record durable knowledge with `memory_write` (one topic per page, under ~8 KB, cite sources,
   update instead of duplicating).
3. No secrets or transcripts; delete wrong pages; per-round notes go to `./tmp/PROGRESS.md`.
4. Use `memory_refresh` when results look redundant or outdated; merge with write + delete.

## Sources
- `src/kiss/core/memoryfield/tools.py` (`MemoryTools`, `MEMORY_PROTOCOL`, `PULL_CHAR_LIMIT`, `_EPHEMERAL_NAME_RE`, `_search`, `_stale_pages`)
- `src/kiss/core/memoryfield/__init__.py`
