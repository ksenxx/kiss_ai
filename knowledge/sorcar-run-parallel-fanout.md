---
title: 'run_parallel sub-agent fan-out: engine, stop events, nesting and budget'
uuid: 911c2bf7-3d7f-4502-9c46-1057e465f1fa
summary: 'run_parallel fan-out: run_tasks_parallel engine, ChatSorcarAgent children,
  _SubagentStopEvent, _await_subagents, abandonment, forwarded settings, no nesting
  cap.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# run_parallel fan-out

## Call path
`run_parallel(tasks, max_workers, model_name, tool_profile)`, the tool defined inside `_get_tools`, runs:
1. `parse_tasks_json(tasks)`: the tasks must be a literal JSON array of non-empty strings (see `sorcar-fanout-guard`).
2. In unattended cron runs, each task is wrapped with `cron_agent.unattended_child_prompt`.
3. `max_workers` must be an integer string ≥1. Empty means the ThreadPoolExecutor default. `tool_profile` must be a `TOOL_PROFILES` key.
4. `SorcarAgent._run_tasks_parallel`: `reclaim_abandoned_subagents()`, start a `_LiveUsageMonitor`,
   compute the per-child budget `share` = `_subagent_budget_share(n)`, then call the module-level
   **`run_tasks_parallel`**. That function is the ONE fan-out engine: a duplicate in ChatSorcarAgent once
   drifted and missed four fixes, so keep it single.
5. Results come back as a YAML list of strings in input order; each string is one child's own YAML result (`{success, summary}`).

## What each child gets (`run_tasks_parallel._run_single`)
- A new `ChatSorcarAgent(f"Parallel-{task[:40]}")` that resumes the parent's `chat_id`, so the child sees
  the chat history prefix.
- `_tab_id = f"task-{routing_key}__sub_{idx}"` and `_subagent_info = {parent_task_id, parent_tab_id, reviewer}`.
  The child persists as a nested task_history row and broadcasts its own `new_tab`.
- These run kwargs: `model_name` (the parent's unless overridden), `work_dir`, `printer`, `is_parallel=True`,
  `max_budget=share`, `model_config` (only when the model is the same, since endpoints and keys belong to
  the parent's model), `base_system_prompt`, the `system_prompt` suffix, `web_tools`, `use_memory`,
  `tool_profile`, and `docker_image` (the parent's live container as `container:<id>`, so children act in the same container).
- With `dispatch_path_rewrite` (env `KISS_DISPATCH_PATH_REWRITE`, default on), parent-repo absolute paths in the task
  are rewritten to the active worktree (`useful_tools.rewrite_parent_repo_paths`).

## Stop handling
- `_SubagentStopEvent` is a `threading.Event` chained to a parent event. `is_set()` walks the chain
  iteratively, so deep nesting cannot hit the recursion limit. Each child has its own event under the
  fan-out's event, so the UI can stop one child while a parent stop still kills the whole tree.
- `_await_subagents` waits in slices (`_SUBAGENT_POLL_SECONDS = 1.0`, or 0.4 s inside a tool call) instead
  of `pool.map`, so the parent stays interruptible. A plain `pool.map` once made a parent outlive
  Stop by 3 minutes. After a stop it gives children `_SUBAGENT_STOP_GRACE_SECONDS = 15` before raising
  `KeyboardInterrupt` and abandoning them.
- A child's `KeyboardInterrupt` without a parent stop becomes "Sub-agent task stopped by user." Other
  exceptions become `_yaml_failure(exc)`.
- On abandonment the pool shuts down without waiting, `_collect_unfinished_usage` takes the max of
  the reported and live usage, and `_register_abandoned` lets the parent bank their spend later.

## Nesting
Children run with `is_parallel=True`, so they can fan out again. **No code limits nesting depth or
fan-out count**: commit ee6ba3d38 removed `ReviewQuota` / `MAX_REVIEW_ROUNDS` /
`REVIEWER_SPAWN_REFUSAL` and the `MIN_SUBAGENT_BUDGET` floor. The limit "do not nest run_parallel more
than 2" exists only in `SYSTEM.md`. What bounds recursion in practice is the budget: each level divides the
remaining budget by (n+1) (see `sorcar-budget-and-step-limits`).

## run_parallel vs run_agent vs run_commands_parallel
- `run_commands_parallel`: plain shell commands in threads, no LLM (`useful_tools.run_commands_pool`).
- `run_parallel`: in-process LLM sub-agents that share the parent's printer and chat.
- `run_agent` (agent_dispatch): dispatches ONE task to a channel agent, `cron`, or an agent-script `.py`,
  as a fresh session on the kiss-web daemon. Its spend is attributed back to the caller.

## Sources
- `src/kiss/agents/sorcar/sorcar_agent.py` (`_get_tools.run_parallel`, `SorcarAgent._run_tasks_parallel`, `run_tasks_parallel`, `_SubagentStopEvent`, `_await_subagents`, `_collect_unfinished_usage`, `_register_abandoned`)
- `src/kiss/agents/sorcar/fanout_guard.py` (`parse_tasks_json`)
- `src/kiss/agents/sorcar/agent_dispatch.py` (`make_run_agent_tool`)
- git: ee6ba3d38 "remove review fan-out guardrails"
