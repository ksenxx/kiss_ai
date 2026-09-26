---
title: What Docker mode (docker_image) disables or changes in a Sorcar run
uuid: e6246356-3b28-4f85-af4a-62b995b29be7
summary: 'What docker_image disables in a Sorcar run: no bash_job/background Bash,
  no persistent memory, CLI models refused, no Read dedupe; browser stays on host;
  sub-agents share container.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# What Docker mode disables or changes in a Sorcar run

When `docker_image` is set, `SorcarAgent._get_tools` builds a different toolset:

```python
tools = [Bash, self.docker_manager.run_commands_parallel,
         docker_tools.Read, docker_tools.Edit, docker_tools.Write]
```

## Disabled

- **`bash_job` and background Bash.** The Docker `Bash` shim keeps a `background` parameter
  but returns `Error: background=True is not available in Docker mode...`, telling the model to
  use `nohup cmd > /tmp/out.log 2>&1 < /dev/null &` and poll with `tail`. There is no job
  registry. The restricted-profile note removes `bash_job` from the advertised tool list when
  `docker_image` is set (`offered = set(allowed) - {"bash_job"}`).
- **Persistent memory.** `_memory_root_for_run` returns `None` when `docker_image` is set, even
  if `use_memory=True`, so the run gets no `memory_*` tools or memory prompt block. The docstring
  reason: a containerized run should not access host memory outside the container boundary.
- **Run-to-completion CLI models** (`model_runs_task_to_completion`, for example `cc/*`,
  `codex/*`). `RelentlessAgent.run` raises `KISSError` because such a model runs its native
  tools on the host and would bypass the isolation. `set_model` refuses to switch to one mid-task
  with a `Cannot switch to ...` message.
- **Read dedupe / forget-reads hook**: only the host `UsefulTools` branch sets
  `context_reset_hook`; Docker Read has no cache.

## Changed

- The shell and file tools run in the container (see `docker-file-tools`,
  `docker-exec-timeouts-and-streaming`). The work dir is bind-mounted at its host path, so file
  edits land in the host checkout.
- The work-dir line in the prompt is only emitted when the work dir is visible inside the
  container (`work_dir_visible` in `RelentlessAgent.perform_task`); an attached
  `container:<id>` without that mount omits it.
- `run_parallel` children get `docker_image="container:<id>"` and share the parent's container;
  `run_tasks_parallel(docker_image=None)` would put them on the host.

## Not changed (easy to assume otherwise)

- Browser tools are still built (full profile only) and Chromium runs **on the host**; the
  container does not isolate browsing.
- Other tools that live outside `_get_tools`'s shell branch (`run_agent`, `ask_user_question`,
  `talk`, `decide`, `run_parallel`) are unaffected by Docker mode. `run_agent`
  (`agent_dispatch.py`) does not forward `docker_image`, so a dispatched sub-task runs its tools
  on the host unless its agent script defines a `docker_image()` getter.

## Sources
- `src/kiss/agents/sorcar/sorcar_agent.py` (`SorcarAgent._get_tools`, `_memory_root_for_run`, `SorcarAgent.run`, `set_model` tool, `RESTRICTED_PROFILE_NOTE`)
- `src/kiss/agents/sorcar/relentless_agent.py` (`RelentlessAgent.run`, `RelentlessAgent.perform_task`)
- `API.md` (`docker_image` and `use_memory` parameters of `kiss.server.sorcar.run`)
