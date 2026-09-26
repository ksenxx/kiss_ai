---
title: 'Persistent agent memory (memoryfield): area overview'
uuid: dd1d15ea-ec41-4ace-be72-3b85abe64f6e
summary: 'Map of kiss.core.memoryfield: pages.py Markdown pages, index.py SQLite vector
  index and embedders, tools.py memory tools and MEMORY_PROTOCOL, evaluate.py recall
  eval, Sorcar gating'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Persistent agent memory (memoryfield): area overview

`src/kiss/core/memoryfield/` implements the memoryfield pattern
(https://github.com/calpaterson/memoryfield-spec): a flat directory of short Markdown pages (the
canonical data) plus a regenerable SQLite embedding index (a cache). It moved from `kiss.agents` to
`kiss.core` in commit "move memoryfield module from kiss.agents to kiss.core".

```
agent --memory_* tools--> MemoryTools --> MemoryDir (pages/*.md)   [pages.py]
                                     \--> VectorIndex (<embedder>.sqlite3 in the same dir) [index.py]
                                             \--> Embedder: text-embedding-3-small | hashed-bow-v1
```

| File | Contents | Page |
|---|---|---|
| `pages.py` | `MemoryDir`, `Page`, frontmatter parse/render, name rules, atomic page writes | `memory-page-format` |
| `index.py` | `VectorIndex` schema, search, `near_duplicates` | `memory-vector-index` |
| `index.py` | `VectorIndex.sync`: stat keys, hashing, batching, CAS writes | `memory-index-sync` |
| `index.py` | `default_embedder`, `ModelEmbedder`, `hashed_embedding` | `memory-embedders` |
| `tools.py` | `MemoryTools` (7 tools), `MEMORY_PROTOCOL` | `memory-agent-tools` |
| `evaluate.py` | recall evaluation on past `sorcar.db` tasks | `memory-recall-evaluation` |
| `__init__.py` | re-exports `MemoryDir`, `MemoryTools`, `VectorIndex`, `MEMORY_PROTOCOL`, embedders | |

Sorcar wiring (whether a run gets memory, where the directory is): `memory-when-enabled`.

## Key invariants
- Pages are the source of truth; deleting `*.sqlite3` loses nothing.
- Every search syncs first, so edits by any process (another agent, an editor, `git pull`) are
  visible without a reindex step. This is what makes a git-versioned memory directory (via the
  `memory_dir` setting) work.
- Many processes share one directory: page writes are atomic, index writes are compare-and-swap on a
  `revision` counter, reads wait out Windows replaces.
- One index file per embedder, so switching between keyed and keyless environments never mixes vectors.
- Only the first 8192 bytes (`MAX_PAGE_BYTES`) of title + summary + body are embedded.

## Using it outside Sorcar
```python
from kiss.core.memoryfield import MemoryTools, MEMORY_PROTOCOL
mem = MemoryTools("/path/to/knowledge")
agent.run(..., tools=mem.tools(), system_prompt=base + "\n\n" + MEMORY_PROTOCOL)
```

## Sources
- `src/kiss/core/memoryfield/__init__.py`, `pages.py`, `index.py`, `tools.py`, `evaluate.py`
- `src/kiss/agents/sorcar/sorcar_agent.py` (`_memory_settings`, `_memory_root_for_run`)
