---
title: Cron job store (jobs.json) and schedule forms
uuid: 89fce715-c3ca-4878-b277-09bfa8c6528a
summary: 'Cron jobs in $KISS_HOME/cron/jobs.json: job fields, file lock, schedule
  forms (every N, 5-field cron, one-shot duration, ISO time), compute_next_run, duplicates.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Cron job store and schedule forms

Scheduled automations live in `src/kiss/agents/sorcar/cron_agent.py`, modelled on the Hermes
agent's cron: **one JSON file, no database**.

## Files under `$KISS_HOME/cron/`
| path | content |
| --- | --- |
| `jobs.json` | list of job dicts (`load_jobs` drops non-dicts and entries without `id`) |
| `jobs.lock` | inter-process lock (`_jobs_lock`, wraps `useful_tools._file_lock`) |
| `output/<job_id>.md` | append-only local log of every delivered result |
| `runs/<job_id>-<random>/` | per-run scratch dir, removed after the run |
| `work/` | work dir of `run_agent(agent="cron")` management sessions |

`save_jobs` uses `atomic_write_text(..., mode=0o600)` (jobs contain prompts and commands; on
Windows it waits out a concurrent reader). `load_jobs` reads via
`read_bytes_waiting_for_writer` and returns `[]` on any error.

## Locking
The scheduler tick takes the lock **non-blocking** (another tick or a tool edit holds it →
skip; selection takes milliseconds). Tool edits (`create`, `remove`, `pause`, `resume`) and the
result write-back in `_execute_job` take it **blocking**. So an edit is never overwritten by a
tick's stale in-memory save.

## Job fields (`cron_job("create", ...)`)
`id` (8 hex from uuid4), `name`, `prompt` XOR `command`, `schedule`, `deliver` (default
`local`), `model_name`, `max_budget` (float, 0 = default), `work_dir` (resolved existing dir or
""), `use_worktree`, `auto_commit` (only allowed for prompt jobs with `work_dir`), `timeout`
(seconds, 0 = default 3600 prompt / 600 command), `enabled`, `one_shot`, `until_delivered`,
`created_at`, `next_run_at`, `last_run_at`, `last_status` (`ok`/`error`/`silent`),
`last_summary` (≤ `MAX_STORED_SUMMARY_CHARS` = 4000), `last_delivery` (notes). Old records
missing newer fields work because readers use `.get`.

## Schedule forms (`compute_next_run(schedule, now)`)
1. Interval `every 30m` (`s m h d`) → `now + N*unit`.
2. One-shot duration `30m`, `1d`.
3. 5-field cron `0 9 * * 1-5`, **local naive time**. `_parse_cron_field` supports `*`, `*/n`,
   `a`, `a-b`, `a-b/n`, lists; DOW 7 folds to 0. Vixie rule in `_cron_date_matches`: if both
   day-of-month and day-of-week are restricted (neither starts with `*`), either may match;
   otherwise both must. The scan covers `CRON_SCAN_DAYS` = 4*366+1 days (the largest gap
   between leap days, for `0 0 29 2 *`), skipping non-matching days whole. DST can shift a run
   by up to an hour.
4. ISO timestamp `2026-01-15T14:00` (local unless offset given) → `None` if in the past.
Anything else raises `ValueError` with the list of forms. `is_one_shot` = not interval and
not cron. The natural-language translation ("every weekday at 9am") is done by the LLM.

## Duplicate detection
`_find_duplicate` refuses a create whose `prompt`, `command`, `work_dir`, `schedule`, and
`deliver` all match an existing job with non-null `next_run_at` (paused jobs count; finished
one-shots do not). Name/model/budget are ignored. The error includes the existing job and a
hint to `resume` or `remove` it; the agent is told not to retry under another name.

## Other actions
`list`; `remove`/`pause`/`resume` by `job_id` (resuming an already-run one-shot is refused;
resume recomputes `next_run_at` when null); `run_now` executes immediately without changing the
schedule.

## Sources
- `src/kiss/agents/sorcar/cron_agent.py` (`load_jobs`, `save_jobs`, `_jobs_lock`, `compute_next_run`, `_parse_cron_expr`, `_parse_cron_field`, `_cron_date_matches`, `is_one_shot`, `cron_job`, `_find_duplicate`, `_job_view`)
