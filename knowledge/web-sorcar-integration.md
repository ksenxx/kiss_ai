---
title: How SorcarAgent wires the browser tools (web_tools, profiles, sub-agents, Docker)
uuid: 857a1a43-4d41-48f7-8371-de60fb25460f
summary: 'When a Sorcar run gets browser tools: web_tools/use_web_tools toggle, full
  tool profile only, ephemeral profile for sub-agents, closed at run end, Chromium
  stays on host in Docker.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# How SorcarAgent wires the browser tools

## Toggle chain

- `SorcarAgent.run(web_tools=True)` is the in-process switch.
- Daemon runs use `kiss.server.sorcar.run(use_web_tools=...)` / wire field `webTools`; `None`
  means the settings default (the "Use web tools" checkbox, persisted as `use_web_browser`).
- An agent script (SEA) can override it with a `use_web_tools()` getter returning a bool or
  `None` (see `seas-agent-script-contract`).
- `run_tasks_parallel(web_tools=...)` forwards the parent's value, so a parent run without web
  tools cannot hand the browser to its children.
- With `append_basic_tools=False` no basic tools are built, so `web_tools` has no effect.

## Where the tool is built

In `SorcarAgent._get_tools`, after the shell/file tools:

```python
if allowed is None and self._use_web_tools and self.web_use_tool is None:
    self.web_use_tool = WebUseTool(
        work_dir=self.work_dir,
        ephemeral=getattr(self, "_subagent_info", None) is not None,
    )
    tools.extend(self.web_use_tool.get_tools())
```

Consequences:
- `allowed is None` only for the `full` tool profile. Restricted profiles (`review`, `shell`,
  `assistant`, `bash`) never get browser tools.
- Sub-agents (those with `_subagent_info`, i.e. spawned by `run_parallel`) get an ephemeral temp
  profile, because concurrent sub-agents would otherwise contend for the shared profile's
  Chromium lock. They therefore do not share the user's logins.
- One `WebUseTool` per agent run; `SorcarAgent.run` closes it in `finally` and sets
  `self.web_use_tool = None`.
- `screenshot` paths are resolved against the agent's `work_dir` and remapped into the active
  worktree (`_active_worktree_remap`) or away from an already-merged stale worktree
  (`_stale_worktree_fallback`), mirroring `UsefulTools.Write`.

## Docker mode

In a Docker run the shell and file tools are replaced by container versions, but the web tool
code path above is unchanged: Chromium runs on the host, not inside the container, and
screenshots are written to host paths. See `docker-mode-disabled-features`.

## Sources
- `src/kiss/agents/sorcar/sorcar_agent.py` (`SorcarAgent._get_tools`, `SorcarAgent.run`, `TOOL_PROFILES`, `run_tasks_parallel`)
- `src/kiss/server/agent_file.py` (`PARAM_FIELDS`)
- `src/kiss/agents/sorcar/web_use_tool.py` (`WebUseTool.screenshot`)
- `API.md` (`use_web_tools` parameter of `kiss.server.sorcar.run`)
