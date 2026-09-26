---
title: Orphaned task recovery - "Agent Failed Abruptly" rows, process killed, final-results
  journal
uuid: 829b383d-91e4-4f82-9d00-d82c1b24b881
summary: Startup sweep _recover_orphaned_tasks turns dead 'Agent Failed Abruptly'
  rows into 'process killed' or restores results from final_results.jsonl; progress
  backfill; shutdown rewrite.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Orphaned task recovery

Every task row starts with `result = "Agent Failed Abruptly"` (`_add_task`). The server's task runner overwrites
it in a cleanup `finally` via `_save_task_result`. If the process is killed (SIGKILL, extension reload, OOM) no
Python code runs, and the sentinel stays.

## Startup sweep: `_recover_orphaned_tasks(active_task_ids, created_before=None)`
Called on a background thread at server start (`src/kiss/server/server.py`) with the ids still running in this
process and `created_before = boot_ts`, so a task started after boot is never mistaken for an orphan.
In one `_rw_lock.write_lock()` + `BEGIN IMMEDIATE`:
1. Select sentinel rows (by `rowid`, since a TEXT primary key can be NULL in rows written by old tools).
2. Keep rows not in `active_task_ids` whose `owner` token is not alive (`_owner_is_alive`). Liveness comes from the
   database owner token, not process memory; before that, a second process (CLI run, reload) marked another
   process's RUNNING tasks as failed.
3. Rows whose id is in the final-results journal get the journalled result back (the task finished but its write was
   lost). Others become `Task terminated unexpectedly (process killed)` and are logged by
   `_log_orphaned_task_forensics`.
4. `_backfill_orphan_progress` fills still-zero `steps`/`tokens`/`cost`/`end_ts` from the task's surviving events
   (`usage_info` text parsed with `_USAGE_STEPS_RE`, `_USAGE_TOKENS_RE`, `_USAGE_COST_RE`); non-zero values are kept.
Afterwards the journal is pruned (`_prune_final_results_journal`, entries older than 30 days).

## Final-results journal
`_save_task_result` also calls `_journal_final_result`, appending to `<db>.final_results.jsonl`. Reason: after a SQLite
`disk I/O error`, WAL frames committed in a task's last minutes were discarded when the next process opened the file,
and the sweep then mislabeled a finished task as killed. A plain append-only file does not share SQLite's failure modes.

## Shutdown safety net
`_shutdown_persist_in_flight_results(task_ids)` (called by the remote server's `_stop_active_agent_tasks` before
signalling workers) rewrites still-sentinel in-flight rows to `Task interrupted by server restart/shutdown`, for
workers stuck in C code or slow cleanup.

## Reading the history
A sentinel row with `end_ts = 0` is running or killed; compare the newest event timestamp of it and its children with
now. `_is_failed_result` gives the red status dot for all of these outcomes.

## Sources
- `src/kiss/agents/sorcar/persistence.py` (`_recover_orphaned_tasks`, `_backfill_orphan_progress`, `_log_orphaned_task_forensics`, `_journal_final_result`, `_load_final_results`, `_prune_final_results_journal`, `_FINAL_RESULTS_MAX_AGE_S`, `_shutdown_persist_in_flight_results`, `_save_task_result`)
- `src/kiss/server/server.py` (startup call of `_recover_orphaned_tasks`)
