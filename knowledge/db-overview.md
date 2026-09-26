---
title: Task persistence (sorcar.db) overview
uuid: e7017da4-d6cc-41de-96be-aff8aa2f630c
summary: 'Map of persistence.py: ~/.kiss/sorcar.db (KISS_HOME) SQLite WAL store for
  task_history rows, events transcript, usage counters, owner liveness markers, sidecar
  journals and orphan recovery.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Task persistence (sorcar.db) overview

`src/kiss/agents/sorcar/persistence.py` is the central module that talks to the database; maintenance
scripts in `src/kiss/scripts/` (`db_fingerprint.py`, `sync_db.py`, `cost_report.py`, `running_tasks.py`,
`carry_over_tables.py`, `relocate_work_dir.py`) also open it with `sqlite3.connect`. Everything is in one
SQLite file, `<KISS_HOME>/sorcar.db` (`_DB_PATH`, default `~/.kiss/sorcar.db`, via `kiss.core.config.kiss_home`),
shared by every Sorcar process on the machine (kiss-web daemon, `kiss` CLI runs, VS Code reloads).

## What is stored
- One `task_history` row per task run, including sub-agents (with `parent_task_id`).
- The task's UI transcript as ordered `events` rows (`event_json`).
- Counters: `model_usage`, `file_usage`, `frequent_tasks`, `steer_inputs`.
- `replayed_journals`: dedup markers for replayed failed-event journals.

## Files next to the database
- `sorcar.db-wal`, `sorcar.db-shm`: SQLite WAL sidecars (never deleted manually by the code).
- `sorcar.db.failed_events.jsonl` (+ claimed `.consumed-*` snapshots): events that could not be written.
- `sorcar.db.final_results.jsonl`: every saved final result, used to undo lost WAL writes.
- `task-owners/<pid>-<uuid>.lock`: per-process liveness markers held with `flock`.

## Pages
- `db-schema`: tables, columns, indexes, migrations.
- `db-task-and-chat-ids`: task id, chat id, parent id, the "Agent Failed Abruptly" sentinel, owner token.
- `db-event-writer`: async batched event persistence, seq numbering, journals.
- `db-orphan-task-recovery`: startup sweep, shutdown rewrite, final-results journal.
- `db-concurrency`: per-thread connections, `_RWLock`, `BEGIN IMMEDIATE`, WAL settings.

Reading the database by hand: use Python `sqlite3` with a read-only URI (`?mode=ro`); there may be no
`sqlite3` CLI installed. The `kiss.agents.sorcar.task_digest` module summarizes a task's events.

## Sources
- `src/kiss/agents/sorcar/persistence.py` (module docstring, `_DB_PATH`, `_failed_events_path`, `_final_results_path`, `_owner_dir`)
