---
title: sorcar.db schema - task_history, events, model_usage, file_usage, frequent_tasks,
  steer_inputs, replayed_journals
uuid: 6c145922-e6a9-409d-a57b-5fe01b6f46df
summary: Tables, columns, defaults and indexes created by persistence._init_tables,
  the unique (task_id, seq) index, _add_missing_columns ALTERs and the pre-UUID migration.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# sorcar.db schema

Created by `_init_tables` (idempotent `CREATE TABLE IF NOT EXISTS`) on every new connection.

## `task_history`
`id TEXT PRIMARY KEY` (uuid4 hex), `timestamp REAL` (epoch s, creation), `task TEXT` (prompt), `has_events`,
`result TEXT` (starts as `Agent Failed Abruptly`), `chat_id CHAR(32)`, `model`, `work_dir`, `version`,
`tokens`, `cost REAL`, `steps`, `is_parallel`, `is_worktree`, `auto_commit_mode` (the three toggles default 1,
matching framework defaults), `start_ts`/`end_ts INTEGER` (epoch ms), `is_favorite`, `parent_task_id TEXT`
(sub-agent rows), `max_budget REAL`, `owner TEXT` (creating process token), `is_side_channel` (1 for the /ask
answerer child, whose result goes into the parent transcript).

## `events`
`id INTEGER PRIMARY KEY AUTOINCREMENT`, `task_id TEXT REFERENCES task_history(id)`, `seq INTEGER`,
`event_json TEXT`, `timestamp REAL`. Replay order is `seq`.

## Counters and misc
- `model_usage(model UNIQUE, count, is_last)`: `is_last` kept only for schema compatibility; the last model lives in `config.json`.
- `file_usage(path UNIQUE, count, last_used)`, capped at `_MAX_FILE_USAGE_ENTRIES` (10000).
- `frequent_tasks(task PK, count, timestamp)`, `_MAX_FREQUENT_TASKS` 100.
- `steer_inputs(text PK, timestamp)`: text typed into a RUNNING task's composer (never gets a task row), for autocomplete; cap 1000.
- `replayed_journals(snapshot PK, timestamp)`: see `db-event-writer`.

## Indexes (`_INDEX_DDL`)
`idx_th_timestamp`, `idx_th_task`, `idx_th_chat_id`, `idx_th_parent_task_id`, `idx_ev_task_id`, and
`UNIQUE idx_ev_task_seq ON events(task_id, seq)`. The unique index makes a process with a stale in-memory
seq counter fail its insert instead of writing a duplicate seq. Older databases with duplicates are repaired by
`_dedupe_event_seqs` (renumbers to 0..n-1 by `seq, id`, deletes nothing) inside one `BEGIN IMMEDIATE` with the
index creation (`_repair_and_create_index`, 3 attempts).

## Migrations
- `_add_missing_columns`: `ALTER TABLE` for columns added after first release; "duplicate column name" from a
  concurrent process is ignored.
- `_migrate_old_schema_if_needed`: ports the legacy schema (INTEGER `id` + JSON `extra` column) to uuid ids and
  typed columns, remapping `events.task_id`, then swaps tables.

## Status strings
`_is_failed_result` treats as failed: `Task failed*`, `Agent Failed Abruptly`,
`Task terminated unexpectedly (process killed)`, `Task stopped by user`, `Task interrupted*`.

## Sources
- `src/kiss/agents/sorcar/persistence.py` (`_init_tables`, `_INDEX_DDL`, `_dedupe_event_seqs`, `_repair_and_create_index`, `_add_missing_columns`, `_migrate_old_schema_if_needed`, `_is_failed_result`)
