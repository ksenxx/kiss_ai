---
title: Memory index sync, staleness detection and concurrent writers
uuid: e6582703-ba61-4c0b-a80a-678ac123f46d
summary: 'VectorIndex.sync incremental refresh: stat_key (size mtime ctime inode,
  git racy-clean RACY_WINDOW_NS), sha256 check, batched embedding, revision compare-and-swap
  for multiple processes, verify mode'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Memory index sync, staleness detection and concurrent writers

`VectorIndex.sync(verify=False)` brings the index in line with the pages on disk and returns a
`SyncReport(added, updated, removed, unchanged)`. Every agent process syncs the same index file,
so the design assumes concurrent writers.

## Deciding what to re-embed
For each page from `MemoryDir.page_stats()` (one scandir pass, `lstat` data):
1. If the row was embedded under the current `EMBEDDING_INPUT_FORMAT` and its stored `stat_key`
   equals the file's current key, the page is **not even read**.
2. Otherwise the file is read and hashed; if the sha256 matches the row, it is unchanged (the
   stat key is refreshed via a "restat" write).
3. Otherwise it is queued for embedding. Rows whose page no longer exists are deleted.

`stat_key` works like git's index: `size:mtime_ns:ctime_ns:ino#sha256hex` on POSIX,
`size:mtime_ns:ino#sha256hex` on Windows (ctime is creation time there and scandir reports inode 0).
It returns `''` ("must hash") while mtime/ctime are within `RACY_WINDOW_NS` (3 s) of the scan
start, git's "racy clean" rule, covering 2 s FAT and 1 s HFS+ timestamp ticks. The digest suffix
means a key only vouches for a row that still holds that content.

`verify=True` hashes every page even when the stat key matches; `memory_refresh` uses it to catch
edits that preserved size and timestamps.

## Embedding
Pending pages are split into requests of at most `EMBED_BATCH_INPUTS = 256` inputs and
`EMBED_BATCH_BYTES = 250_000` bytes (`_batches`); one `embed_many` call per batch when the embedder
has it, else one call per text (`_embed_all`). Embedding happens with **no DB transaction open**;
each batch's rows are then written in one short transaction, so an interrupted build keeps its
finished batches. A page whose hash changed while it was being embedded is skipped and left for the
next sync instead of being stored with a stale vector.

## Compare-and-swap writes
`_write` opens `BEGIN IMMEDIATE`, reads `meta.revision`, and gives each insert/update/delete a fresh
revision number. An insert is `INSERT OR IGNORE` (only if still absent); an update or delete
applies only `WHERE revision = <snapshot revision>`. So a sync never overwrites or removes a row that
a concurrent sync touched after this sync's snapshot, even with identical content. Skipped writes
are logged and counted in no category. Restat updates apply only while the row still has the
verified sha256 and current input format.

Known limits, both healed by the next sync through the sha256 check: a page deleted after its
post-embed check keeps a row until the next sync, and a still-running process on pre-`revision`
code writes rows with revision `0`.

## Upgrades across versions
`EMBEDDING_INPUT_FORMAT` and the `input_format` column let old and new processes share one file
without rebuilding back and forth: `sha256` stays the plain content hash, so older code sees new
rows as unchanged, while new code re-embeds any row with an older format.

## Sources
- `src/kiss/core/memoryfield/index.py` (`VectorIndex.sync`, `VectorIndex._write`, `stat_key`, `RACY_WINDOW_NS`, `_batches`, `_embed_all`, `EMBED_BATCH_INPUTS`, `EMBED_BATCH_BYTES`)
- `src/kiss/core/memoryfield/pages.py` (`MemoryDir.page_stats`)
