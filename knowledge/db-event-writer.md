---
title: Chat event persistence - background event writer, seq numbering, failed-event
  journal
uuid: 74772e23-a843-4170-a59a-37937d979806
summary: '_queue_chat_event and the kiss-event-writer thread: batches of 256 in 20
  ms windows, 4 retries, (task_id, seq) numbering, failed_events.jsonl journal and
  replay on flush.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Chat event persistence

## Producers
- `_queue_chat_event(event, task_id, origin_db_path=None)`: non-blocking. Serializes the event (before reserving,
  so a `TypeError` cannot leave an unreleasable reservation), `_reserve_pending(task_id)`, `put_nowait` on the
  unbounded `_event_queue`, and lazily starts the writer.
- `_append_chat_event(...)`: synchronous version (queue + `_flush_chat_events()`), drops the event if
  `origin_db_path` no longer matches the active database.
- `origin_db_path` (`_current_db_path()`) stamps each write with the database it was produced for, so a late
  write after a database swap (tests, daemon pointed at another home) is dropped instead of landing on an
  unrelated row.

## Writer: `_start_event_writer` / `_event_writer_loop`
Daemon thread `kiss-event-writer` takes one item, then keeps collecting for `_BATCH_WINDOW_S` (0.020 s) or until
`_BATCH_MAX` (256) items, and persists the batch in one transaction (`_write_event_batch` ->
`_write_event_batch_locked`, `BEGIN IMMEDIATE`, under `_rw_lock.write_lock()` and `_caches_lock`). A `None`
item is the stop sentinel; each writer has its own stop `Event`.
`_persist_batch_with_retry` tries 4 times (sleep 0.05 s x attempt); after the last failure the batch goes to
`_journal_failed_events` (`<db>.failed_events.jsonl`) for replay. Journalling is best-effort: if writing the sidecar raises
`OSError`, the events are dropped and an error is logged.

## Sequence numbers
Each task's events get increasing `seq` values from the in-memory `_next_seq_cache`. Several processes share the
database, so the `UNIQUE (task_id, seq)` index rejects a stale counter with `IntegrityError`; `_write_event_batch`
then rolls back (`_rollback_event_batch`, which also resets the cached seqs), re-reads `MAX(seq)` and retries once.
Batch rows whose `origin_db_path` differs from the active database are dropped first.

## Ordering barrier: `_flush_chat_events(task_id=None)`
Blocks on `_pending_cond` until the task's (or all) queued events are written, then calls `_replay_failed_events`.
Must be called BEFORE taking `_rw_lock.write_lock()` (the writer takes it per batch, so holding it deadlocks).
`_save_task_extra` and result saves flush first so the final row update lands after the transcript.

## Failed-event journal replay
`_replay_failed_events` runs under `_journal_file_lock` (cross-process). It claims the sidecar by renaming it to a
`.consumed-<claim key>...` snapshot, so appends by a peer go to a fresh file and nothing is replayed twice. Each
snapshot is inserted with a `replayed_journals` marker in the same transaction; a replayer that dies after commit but
before deleting the file is detected by the marker and does not insert duplicates. Failed snapshots keep their claimed
name (not renamed back) to preserve chronological replay order.

Shutdown: `_stop_event_writer` and the atexit `_drain_events_at_exit` drain the queue.

## Sources
- `src/kiss/agents/sorcar/persistence.py` (`_queue_chat_event`, `_append_chat_event`, `_start_event_writer`, `_event_writer_loop`, `_persist_batch_with_retry`, `_write_event_batch_locked`, `_flush_chat_events`, `_journal_failed_events`, `_replay_failed_events`, `_journal_file_lock`, `_current_db_path`, `_drain_events_at_exit`)
