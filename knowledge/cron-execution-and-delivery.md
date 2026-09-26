---
title: 'Cron execution: scheduler thread, prompt jobs as generated SEA, command jobs,
  delivery'
uuid: e662312f-45f8-498d-819b-de910b030cfc
summary: 'Cron execution: scheduler thread in kiss-web, tick every 60 s, prompt jobs
  as generated SEA via run_agent, command jobs with tree kill, [SILENT], channel delivery.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Cron execution and delivery

## Scheduler
The kiss-web daemon calls `start_scheduler_thread(sock_path=...)` at startup: a daemon thread
runs `run_scheduler`, which calls `tick(wait=False)` immediately and then every
`DEFAULT_TICK_INTERVAL_SECONDS` = 60. The socket is stored in the module global
`_daemon_sock_path` so prompt jobs submit back to the same daemon (read by
`agent_dispatch._daemon_sock_path` from the *canonical* module, because a dispatched cron
session gets a fresh synthetic copy of the module whose global is unset). Outside the daemon,
`kiss-cron --daemon [--interval S]` or `kiss-cron --tick` run the scheduler; command jobs work
standalone but prompt jobs still need a reachable daemon.

## `tick(now, wait)`
Under the non-blocking store lock: for each enabled job not in `running_job_ids()` whose
`next_run_at <= now`, compute the next run (`None` for one-shots, which are also disabled) and
**save before running**, so the same occurrence never fires twice. Malformed jobs are disabled
with `last_summary = "disabled: malformed job: ..."`. Each due job starts in its own thread
(`kiss-cron-job-<id>`), registered in the process-local `_running` dict while holding
`_running_lock`. A job still running from an earlier tick is skipped and stays due, so it runs
again right after it finishes instead of overlapping. The registry is per process: another
process's `--tick` or `run_now` can still overlap.

## `_execute_job(job)`
Creates `runs/<id>-XXXX` via `mkdtemp`, runs the job, and removes the dir in `finally`, except
after an unconfirmed-stop timeout (the task may still be using it). Job and delivery errors are
caught and recorded in `last_status` / `last_summary` / `last_delivery`; the runs-dir creation and the
final outcome persistence are outside those handlers and can still raise. Errors are delivered too, so the user learns
the automation broke.

**Prompt job** (`_run_prompt_job`): `_write_prompt_sea` writes `cron_prompt_sea.py` into the
scratch dir, with getters `prompt()` (= `PROMPT_PREAMBLE` + job prompt), `work_dir()`,
`model()`, `max_budget()`, `use_worktree()`, `auto_commit()`, `classify_tasks() -> False`. Values
are embedded with `repr()` so any prompt text round-trips. It is launched with
`make_run_agent_tool(str(scratch))(agent=<sea path>, task=<placeholder>, timeout=...)`, the same
path-mode dispatch a chat uses for any `.py` agent script, as a top-level task with a fresh
session and no history. Reply handling: `Error:` → error; the exact `stop_unconfirmed_error`
text → `TimeoutError` (keep scratch dir); YAML `{success, summary}`; an empty or silent summary
→ `silent`. Commit `e4351d507` replaced direct daemon-client calls with this SEA approach.

**Unattended rule**: `PROMPT_PREAMBLE` says nobody can answer questions and forbids editing
schedules. `is_unattended(agent)` detects it; children spawned via `run_parallel` get
`unattended_child_prompt` (prepended) and via `run_agent` get `unattended_child_suffix`
(appended through `append_to_prompt`), so sub-tasks never block on `ask_user_question`.

**Command job** (`_run_command_job`, Hermes "no_agent"): runs under the same shell as the Bash
tool (`_popen_kwargs`) in its own process group with cwd = job `work_dir` or scratch dir.
Non-zero exit → error with output; empty or whitespace-only stdout → `silent`; else `ok` with
`stdout.strip()`. On timeout `_kill_command_tree` snapshots descendants from `/proc`
(`_proc_descendants`, Linux) **before** killing the process group, then kills `setsid`/daemonized descendants in other groups (up to 3 passes).
Before this fix `subprocess.run(shell=True, timeout=...)` left orphan trees running.

## Delivery (`_deliver`)
Always appends `## <iso time> — <name>` + text to `output/<id>.md`. Then for each
comma-separated target other than `local`/`none`: `<channel>[:<chat>]` →
`_deliver_to_channel` imports `kiss.agents.third_party_agents.<channel>_sea`, calls
`_make_backend()`, `connect()`, `find_channel(chat) or chat`, `send_message`, `disconnect()`.
Returns notes like `sent to telegram:123` or `error: ...` (unknown module, no
`_make_backend`, `SystemExit` → "channel not authenticated"). Silence: `_is_silent` treats
`[SILENT]` / `NO_REPLY` (HTML tags stripped) as nothing to deliver.

`until_delivered` jobs ("tell me when X happens") are disabled after the first `ok` result that
reached every target without an `error` note, also when triggered by `run_now`.

## Sources
- `src/kiss/agents/sorcar/cron_agent.py` (`start_scheduler_thread`, `run_scheduler`, `tick`, `running_job_ids`, `_execute_job`, `_run_prompt_job`, `_write_prompt_sea`, `PROMPT_PREAMBLE`, `UNATTENDED_CHILD_PREAMBLE`, `is_unattended`, `_run_command_job`, `_kill_command_tree`, `_proc_descendants`, `_deliver`, `_deliver_to_channel`, `_is_silent`)
