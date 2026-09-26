---
title: sorcar.db concurrency - per-thread connections, _RWLock, BEGIN IMMEDIATE, WAL
  and busy_timeout
uuid: 6f3f4883-623d-4e14-9b53-4d287861ba8b
summary: 'persistence.py thread/process safety: threading.local connections, interrupt-safe
  _RWLock, _immediate_txn (BEGIN IMMEDIATE), WAL, busy_timeout=30000, sidecar recovery.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# sorcar.db concurrency

## Connections (`_get_db`, `_open_db_connection`)
- One `sqlite3.Connection` per thread (`threading.local`), so threads never share cursor state.
- Opened with `timeout=10`, `isolation_level=None` (autocommit), then `PRAGMA busy_timeout=30000`,
  `PRAGMA foreign_keys=ON`, and `PRAGMA journal_mode=WAL` (retried up to 30 s while busy); schema migration and
  `_init_tables` run under the process-local `_init_tables_lock`.
- A cached connection is reopened when `_close_db()` bumps `_db_generation`, when `_DB_PATH` changes, when the
  file's `(st_dev, st_ino)` changes (deleted/replaced), or when the `-shm` sidecar this process mapped was replaced
  (`_sidecars_orphaned`); then every connection in the process is closed first (`_recover_orphaned_sidecars`).
- The code never deletes `-wal`/`-shm` files itself: unlinking a live WAL destroys committed data (earlier bug).
- `_close_thread_db()` closes only the caller's connection; short-lived threads (e.g. the startup sweep) call it.

## Locks
- `_rw_lock` (`_RWLock`): writer-preferring read/write lock; SQLite WAL allows one writer anyway. It must survive a
  task stop, which the server delivers as an injected `KeyboardInterrupt` at any bytecode boundary. Ownership is held
  in per-acquisition `_Token` objects tracked by weak references, dead or closed acquisitions stop blocking within
  one bounded wait tick, and the condition mutex is an `RLock` so teardown can check real ownership.
- `_rw_lock` is per process. For cross-process atomicity of read-modify-write sequences, `_immediate_txn(db)` wraps
  them in `BEGIN IMMEDIATE`, which takes SQLite's write lock that all processes respect (e.g. `_save_task_extra`,
  `_mark_legacy_side_channel_rows`, `_recover_orphaned_tasks`).
- Lock order: call `_flush_chat_events` before taking `_rw_lock.write_lock()` (see `db-event-writer`).
- Journals use `_journal_file_lock` (file `flock`) because they are shared by processes.

## Testing
`KISS_RACE_DELAY` (`_concurrency._race_delay`) widens read-modify-write windows in concurrency tests, for example
between attempts in `_repair_and_create_index`.

## Sources
- `src/kiss/agents/sorcar/persistence.py` (`_get_db`, `_open_db_connection`, `_RWLock`, `_Token`, `_immediate_txn`, `_sidecars_orphaned`, `_recover_orphaned_sidecars`, `_close_db`, `_close_thread_db`, `_journal_file_lock`)
- `src/kiss/agents/sorcar/_concurrency.py` (`_race_delay`)
