---
title: Docker command execution, timeouts, kill and output handling
uuid: e9475e42-0a25-47dc-b6f2-74e00fba6cd4
summary: 'DockerManager.Bash and run_commands_parallel: KISS_EXEC_TOKEN-tagged execs,
  kill via /proc/*/environ on timeout, delayed-exec reaper, \r progress collapse,
  exit code, truncation.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Docker command execution, timeouts, kill and output handling

## Entry points

- `DockerManager.Bash(command, description, timeout_seconds=30, max_output_chars=MAX_OUTPUT_CHARS)`:
  one command. If `stream_callback` is set (Sorcar sets it when a printer exists) it uses
  `_bash_streaming`, else the blocking `_exec`. `MAX_OUTPUT_CHARS` is
  `DEFAULT_CONFIG.tool_output_max_chars`, the same cap as host `UsefulTools.Bash`.
- `DockerManager.run_commands_parallel(commands, max_workers=0, timeout_seconds=1800,
  max_output_chars=5000)`: a JSON list of shell commands (parsed by `parse_tasks_json`; shell
  substitutions like `"$(cat cmds.json)"` are rejected), each run as its own exec via
  `_timed_exec`, pooled by the shared `run_commands_pool` from `useful_tools.py`. Output is not
  streamed; the report starts with `N commands: K succeeded, F failed, T timed out`. The task's
  `stop_event` (set by `SorcarAgent._get_tools`) cancels running execs.
- No-container calls raise `KISSError("No container is open...")`. Each method snapshots
  `self.container` once, because `close()` may null it concurrently (command paths
  deliberately do not take the lifecycle lock so a running command cannot block teardown).

## Why execs are tagged

`_tagged_exec_create` builds every exec as `/bin/bash -c <quoted command>` with
`workdir=self.workdir` and environment `KISS_EXEC_TOKEN=<uuid>` (`_EXEC_TOKEN_VAR`). Children
inherit the variable. On timeout or cancel, `_kill_exec(token)` runs a small `/bin/sh` script in
the container that scans `/proc/[0-9]*/environ` for the token and `kill -9`s each match,
killing the whole process tree. `exec_inspect`'s `Pid` is not used because it is a host-namespace
pid that means nothing inside the container. Without this, a hung command kept running and held
the stream open for the life of the container.

`_reap_timed_out_exec` handles a docker daemon that delays the exec start past the deadline:
a daemon thread polls `exec_inspect` with exponential backoff (0.2 s up to 5 s,
`_REAP_POLL_INTERVAL_S`, `_REAP_POLL_MAX_INTERVAL_S`) and kills the process once it appears. It
never gives up while the container lives, since giving up would leave a delayed command running.

## Output shaping

- `_drain_exec_stream` decodes with an incremental UTF-8 decoder so multibyte characters split
  across chunks are not mangled.
- `_collapse_progress` renders `\r` like a terminal (pip/tqdm/wget redraws collapse to their
  final state), which keeps progress bars from flooding the context.
- `_with_exit_code` appends `[exit code: N]` only for non-zero exits.
- Timeout results: `Error: command timed out after Ns`; the streaming path adds
  `and was killed. Output before the timeout:` plus the partial output.
- `_truncate_output` (from `useful_tools.py`) applies the character cap.

## Sources
- `src/kiss/agents/sorcar/docker_manager.py` (`DockerManager.Bash`, `run_commands_parallel`, `_timed_exec`, `_exec`, `_tagged_exec_create`, `_bash_streaming`, `_reap_timed_out_exec`, `_kill_exec`, `_drain_exec_stream`, `_collapse_progress`, `_with_exit_code`)
- `src/kiss/agents/sorcar/useful_tools.py` (`run_commands_pool`, `_truncate_output`)
