---
title: Memory vector index (SQLite file, schema, search)
uuid: 6569eb69-c536-4fde-8f4f-957fd1ccc87f
summary: 'VectorIndex: per-embedder SQLite file <model>.sqlite3 inside the memory
  dir, pages/meta tables, SCHEMA_VERSION, float32 BLOB embeddings, numpy cosine search
  with exact rescoring, near_duplicates'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Memory vector index (SQLite file, schema, search)

`VectorIndex` is a cache over a `MemoryDir`; deleting its file loses nothing (`VectorIndex.clear`
unlinks it and the next `sync` rebuilds). No SQLite extension is used: `sqlite-vec` cannot be
loaded by the macOS system Python, so search is an exhaustive scan in numpy.

## File and schema
- Location: inside the memory directory, named after the embedder:
  `<root>/<model_code_for_filename(model_code)>.sqlite3`, e.g. `text-embedding-3-small.sqlite3` or
  `hashed-bow-v1.sqlite3`. Different embedders therefore never mix vectors (see `memory-embedders`).
- `pages` table: `filename` (PK, `name.md`), `frontmatter` (JSON text), `last_modified`, `sha256`
  (BLOB), `embedding` (little-endian float32 BLOB, unit length, the sqlite-vec layout),
  `input_format`, `revision`, `stat_key`. Covering index `pages_sync` on
  `(filename, sha256, input_format, revision, stat_key)` so a sync snapshot does not read the
  embedding overflow pages.
- `meta` table keys: `model_code`, `revision` (monotonic counter), `generation` (random uuid per
  index file), `schema_version`.
- `SCHEMA_VERSION = "3"`. `_connect` runs no DDL and takes no write lock when `schema_version`
  and `generation` are already present; otherwise `_create_schema` creates tables, adds columns
  from `_ADDED_COLUMNS` via `ALTER TABLE` (race-tolerant), and seeds `meta`.
- `_connect` raises `ValueError` if the index path is a symlink or if `meta.model_code` differs
  from the embedder's model code. Connections use `timeout=30.0` and hold no open transaction.

## What gets embedded
`embedding_text`: title + summary + Markdown body. `uuid`, `created` and `updated` are excluded
because they made identical content embed differently and leaked random tokens into scores.
`embedding_input` truncates to `MAX_PAGE_BYTES` (8192) UTF-8 bytes.
`EMBEDDING_INPUT_FORMAT = "2"` is stored per row; bump it whenever this mapping changes and
every older row is re-embedded on the next sync (rows from before the column read as `'1'`).

## Search
`search(query, k=5, min_score=0.0)`:
1. `embed_query` normalizes the query vector, cached in a per-index LRU of `QUERY_CACHE_SIZE = 64`
   (agents often run `memory_search` then `memory_pull` with the same query).
2. `_rows()` loads rows grouped by embedding dimension into float32 numpy matrices, cached until the
   change token `(generation, revision, COUNT(*))` changes.
3. A float32 matrix-vector product selects candidates within `float32_dot_error(dims)` of the k-th
   best score; candidates are rescored exactly with `math.sumprod`, so ties and the `min_score`
   cut match an exact scan. Hits must score strictly above `min_score`; the default drops
   orthogonal pages; `-1.0` returns up to k dimension-compatible hits scoring above -1.0.
4. Rows with a different dimension than the query are skipped with a warning naming the file to
   delete.

`near_duplicates(threshold=0.9)` does an exact pairwise scan in blocks of `_PAIR_BLOCK_ROWS = 1024`
rows; used by `memory_refresh` (see `memory-agent-tools`).

## Sources
- `src/kiss/core/memoryfield/index.py` (`VectorIndex`, `_connect`, `_create_schema`, `_rows`, `search`, `near_duplicates`, `embedding_text`, `EMBEDDING_INPUT_FORMAT`, `SCHEMA_VERSION`)
