---
title: Sorcar agent area overview (files and page map)
uuid: 0b4cc4b6-7c2d-4260-a7e2-8ceb38b575d0
summary: Map of the Sorcar agent area (sorcar_agent, relentless_agent, chat_sorcar_agent,
  task_classifier, tools, skills, MCP, SEAs, task_digest) and links to detail pages.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Sorcar agent: area overview

KISS Sorcar is a general-purpose agent. A task runs as `ChatSorcarAgent` / `WorktreeSorcarAgent`, then `SorcarAgent`,
then `RelentlessAgent`, which creates one fresh `KISSAgent` per continuation sub-session. Tools are plain Python functions.

## Files (all under `src/kiss/agents/sorcar/` unless noted)
| file | role | page |
|---|---|---|
| `relentless_agent.py` | continuation loop, usage ledger, budget hook, Docker wrapper, system-prompt suffix | `sorcar-relentless-continuation`, `sorcar-budget-and-step-limits` |
| `sorcar_agent.py` | `SorcarAgent`: tools, `TOOL_PROFILES`, `run_parallel` engine, classifier hookup, steering, `summary`, CLI `main` | `sorcar-tool-set-assembly`, `sorcar-tool-profiles`, `sorcar-run-parallel-fanout`, `sorcar-summary-and-steering` |
| `chat_sorcar_agent.py` | chat sessions, prior-task prefix, task_history persistence | `sorcar-chat-session-agent` |
| `task_classifier.py` | pre-run is_simple / is_development verdict | `sorcar-task-classifier` |
| `decide_tool.py` | `decide` tool on the Jev decisions model | `models-decisions-jev-model` |
| `fanout_guard.py` | tasks-JSON parsing, review-task heuristics | `sorcar-fanout-guard` |
| `useful_tools.py` | Bash, bash_job, run_commands_parallel, Read/Edit/Write | `sorcar-useful-tools` |
| `skills.py` | Agent Skills (`SKILL.md`) and the `skill` tool | `sorcar-skills` |
| `mcp_servers.py` | MCP server config, tools, OAuth, connection pool | `sorcar-mcp-servers` |
| `sea_commands.py`, `bare_path_task.py` | `/xxx` slash-command and path-only prompt rewrites | `sorcar-slash-commands-and-bare-paths` |
| `task_digest.py` | compact transcript digests for `/ask` and task-update | `sorcar-task-digest` |
| `_concurrency.py` | `_race_delay()` test hook (`KISS_RACE_DELAY`, capped at 0.1 s) and a re-export of `pid_alive` | — |
| `src/kiss/TIPS.md`, `src/kiss/SAMPLE_TASKS.md`, `SORCAR.md` | user-facing tips, sample tasks, user prefs | `sorcar-bundled-docs` |

Cross-cutting pages: `sorcar-agent-class-hierarchy` and `sorcar-system-prompt-assembly`.

## Lifecycle of one task
1. `ChatSorcarAgent.run` writes the task_history row and builds the chat prefix.
2. `SorcarAgent.run` classifies the task (lite or full prompt), appends the profile note and memory protocol, and stores the fan-out settings.
3. `RelentlessAgent.run` calls `_reset` (model, budget, work_dir), optionally inside a Docker container.
4. `SorcarAgent.perform_task` calls `_get_tools` and installs the steering hooks.
5. `RelentlessAgent.perform_task` appends IMPORTANT_INSTRUCTIONS, Task Settings and `~/.kiss/SORCAR.md`, then loops sub-sessions until `finish` without `is_continue`.
6. On unwind: fold the classifier spend, save the result and usage.

## Where related things live (other areas)
- The KISSAgent step loop, `_check_limits`, tool execution: `core-agent-run-loop`, `core-tool-execution-and-hooks`.
- Worktrees and auto-commit (`WorktreeSorcarAgent`, `git_worktree.py`): `git-overview`.
- Persistent memory tools and gating: the `memory-*` pages.
- Model catalog and providers: the `models-*` pages.
- `run_agent` dispatch (`agent_dispatch.py`), cron, the daemon, and the web server belong to other areas.

## Common pitfalls
- No nesting cap on `run_parallel`: the limit is only in the prompt, and the budget shrinks per level.
- `review` profile ≠ sandbox (Bash is unrestricted).
- Edit/Write refuse files not Read by the tool in this session (Write, not Edit, is exempt under `./tmp/`).
- A `KISSError` with a `__cause__` or at step ≤1 is terminal. A step-limit or context overflow continues via the summarizer.

## Sources
- `src/kiss/agents/sorcar/` (files listed above)
- `pyproject.toml` (`sorcar` script entry point)
