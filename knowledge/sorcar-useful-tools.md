---
title: 'UsefulTools: Bash, background jobs, run_commands_parallel, why Edit/Write
  refuse (read before edit)'
uuid: 9f5fd9ce-6548-43eb-b2dc-9b556077d2ad
summary: 'UsefulTools: Read outline and dedupe, why Edit/Write refuse a file not Read
  first ("has not been read in this session"; Write exempt under tmp/), Bash background
  jobs and bash_job, run_commands_parallel, worktree path guard.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# UsefulTools (host shell and file tools)

`SorcarAgent._get_tools` builds one `UsefulTools(stream_callback, stop_event, work_dir, jobs)` per run
(Docker runs use `DockerTools` instead, see `sorcar-tool-set-assembly`).

## Read
- `Read(file_path, max_lines=2000, start_line=1, force=False)`. A whole-file read of a file longer than
  `read_outline_lines` (env `KISS_READ_OUTLINE_LINES`, default 2000) returns an **outline** (`_outline`: def/class/function
  symbols, Markdown headings) instead of the content. Read line ranges after that.
- **Dedupe** (`read_dedupe`, env `KISS_READ_DEDUPE`, default on): `_reads_shown` maps (path, start, max) to the sha1 of
  the text. Re-reading an unchanged window returns "Unchanged since your earlier Read ... Pass force=True".
  `forget_reads()` clears this map. It is installed as `context_reset_hook` and called at each new sub-session, since the
  earlier content is gone from the fresh context.
- A `start_line` past EOF returns an explicit error rather than empty text. Missing files get a "Did you mean"
  suggestion (`_suggest_close_path`). Binary files up to 20 MB become attachments.

## Read-before-modify
`read_files` holds paths this instance has Read or written. `Edit` and `Write` refuse an **existing** file not in the set
(`_unread_error`: "has not been read in this session. Call Read on it before ..."). Viewing a file via Bash does not count.
`Write` only is exempt for scratch paths (`_is_scratch_path`), meaning under a `tmp` directory **inside** work_dir or
under the cron work dir. `Edit` has no exemption: it always requires a prior Read (or Write) of the file. `forget_reads` keeps `read_files`.

## Bash
- It runs with `cwd=work_dir` and a cleaned env (`_clean_env` drops `VIRTUAL_ENV`; `KISS_WORKDIR` is forced to work_dir).
  Output decoding replaces invalid UTF-8, and output streams to the UI via `stream_callback`.
- Stop: `_start_stop_monitor` kills the whole process group (`_kill_process_group`) when the stop event or a
  tool-call interrupt fires.
- `background=True` starts a detached job (`BackgroundJob`) and returns a job id and log path. `bash_job(job_id,
  action="tail"|"wait"|"kill")` reports status plus the log tail. The job registry lives on the agent
  (`SorcarAgent._background_jobs`), so jobs survive into follow-up prompts in the same chat.
- Worktree guard: `_bash_parent_repo_guard` refuses commands that contain the *parent* repo's absolute path while
  running in a `.kiss-worktrees/kiss_wt-*` worktree, because they would silently change the main checkout and skip
  auto-commit. For Read/Write/Edit, `_active_worktree_remap` rewrites such paths into the worktree, and
  `_stale_worktree_fallback` handles a deleted worktree.

## run_commands_parallel
`run_commands_parallel(commands, max_workers=0, timeout_seconds=1800, max_output_chars=5000)`: `commands` is a JSON
array parsed by `fanout_guard.parse_tasks_json`. Each command runs in its own thread under the Bash rules
(`run_commands_pool`). `0` workers means all at once. The report opens with a tally ("N commands: K succeeded, F
failed, T timed out"), then one section per command. Commands never started because of a stop show
`NOT_STARTED`. Use it for test splits and builds, and keep `run_parallel` for LLM work.

## Sources
- `src/kiss/agents/sorcar/useful_tools.py` (`UsefulTools`, `Read`, `Edit`, `Write`, `Bash`, `bash_job`, `run_commands_parallel`, `forget_reads`, `_is_scratch_path`, `_bash_parent_repo_guard`, `_active_worktree_remap`, `run_commands_pool`, `BackgroundJob`)
- `src/kiss/core/config.py` (`read_dedupe`, `read_outline_lines`)
