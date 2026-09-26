---
title: When Sorcar enables or disables persistent memory (use_memory, KISS_USE_MEMORY,
  memory_dir, docker and CLI-model gates)
uuid: 8295b3ae-92d3-4306-accd-ec468d51a4a9
summary: 'Sorcar memory gating: _memory_settings reads use_memory and memory_dir from
  config.json plus KISS_USE_MEMORY; no memory for docker, cc/codex models, no basic
  tools, custom system_instruction'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# When Sorcar attaches persistent memory

Memory is on by default (commit "enable persistent memory by default with offline fallback"). A
`SorcarAgent.run` that passes the gates gets `MemoryTools(root).tools()` appended to its tools and
`"\n\n" + MEMORY_PROTOCOL` appended to its system prompt; `self._memory_tools` is reset to `None` in
the run's `finally`.

## Where the pages live: _memory_settings()
Returns `(enabled, root)`:
- `enabled`: non-empty `KISS_USE_MEMORY` env var wins (`0`/`false`/`no`/`off`, any case, disables;
  anything else enables); otherwise the `use_memory` key of `config.json` (default True).
- `root`: `memory_dir` from `config.json` (`expanduser`-ed) when non-empty, else
  `kiss_home() / "memories"` (`~/.kiss/memories`). The directory is created lazily on first write.
  `memory_dir` has no settings-panel control; edit `config.json` to set it, e.g. to a repo's
  `knowledge/` folder so the pages are versioned with the code.

## Hard gates: _memory_root_for_run(...)
Returns `None` (no memory) regardless of any override when:
1. `append_basic_tools=False`: the run has only `finish` plus caller tools, so promising `memory_*`
   tools would be a lie;
2. `docker_image` is set: memory tools run in the host process on host paths, which would give a
   containerized task access outside the Docker boundary;
3. the model runs tasks to completion (`model_runs_task_to_completion`, i.e. `cc/*`, `codex/*`
   CLI models), which never see KISS-registered tools;
4. the caller's `model_config["system_instruction"]` is set: it replaces the composed prompt, so
   `MEMORY_PROTOCOL` would never reach the model and tools must not be registered without it.

Past the gates, a boolean `use_memory` argument (the per-run override) beats both the env var and
config; `None` falls back to `_memory_settings`.

## Per-run override plumbing
- `SorcarAgent.run(use_memory=...)` stores it as `_use_memory_override` and forwards it to every
  `run_parallel` sub-agent, so a parent's explicit choice governs its whole task tree.
- Daemon wire field `useMemory` (`task_runner`): absent or non-bool means `None`.
- Agent files (`kiss.server.agent_file`) may define `use_memory()` returning bool or None; several
  bundled SEAs do (`sh_sea`, `task_update_sea`, `autoroute_sea`, `skillopt_sea`).
- The settings panel's memory toggle writes `use_memory` into `config.json` (`main.js`).

## Tool profiles
The `review` profile in `TOOL_PROFILES` includes only the read-side memory tools (`memory_search`,
`memory_pull`, `memory_read`, `memory_list`); `shell`, `assistant` and `bash` include none. Profiles
filter the built tool list, so reviewers can recall but not write memory.

## Sources
- `src/kiss/agents/sorcar/sorcar_agent.py` (`_memory_settings`, `_memory_root_for_run`, `SorcarAgent.run`, `TOOL_PROFILES`)
- `src/kiss/core/vscode_config.py` (`DEFAULTS["use_memory"]`, `DEFAULTS["memory_dir"]`)
- `src/kiss/server/agent_file.py`, `src/kiss/server/task_runner.py` (`useMemory`)
