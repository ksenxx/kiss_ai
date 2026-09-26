---
title: Sorcar tool profiles (full, review, shell, assistant, bash) and reviewer sub-agents
uuid: 37c5174c-64ef-43ba-a9f8-d2f0bbfb4044
summary: TOOL_PROFILES in sorcar_agent.py, how _tool_profile picks one (explicit,
  reviewer marker + tool_profiles config, is_implementation_task), and the RESTRICTED_PROFILE_NOTE
  added to the prompt.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Tool profiles

`TOOL_PROFILES` (`sorcar_agent.py`) maps a profile name to the allowed tool names. `finish` is always added.

| profile | tools |
|---|---|
| `full` | `None`: everything `_get_tools` can build |
| `review` | Bash, bash_job, Read, run_commands_parallel, memory_search, memory_pull, memory_read, memory_list, decide, summary |
| `shell` | Bash, bash_job, Read, run_commands_parallel |
| `assistant` | shell + ask_user_question, talk, decide, summary, set_model |
| `bash` | Bash only (used by the bundled `/sh` SEA) |

`review` is **not a sandbox**: Bash is unrestricted. It only removes the editing, browser, talk,
dispatch and fan-out tools.

## Selection (`SorcarAgent._tool_profile`)
1. An explicit `_tool_profile_name` wins. It is set by `run(tool_profile=...)`, by
   `run_parallel(tool_profile=...)`, by `run_agent`'s `tool_profile`, or by an agent script's
   `tool_profile()` getter. An unknown name raises `ValueError` in `run` and returns an `Error:` string
   from the `run_parallel` tool.
2. Otherwise `review` when `DEFAULT_CONFIG.tool_profiles` is on, the agent is in a reviewer
   sub-tree (`_subagent_info["reviewer"]`), and the task does not ask for changes
   (`fanout_guard.is_implementation_task`).
3. Otherwise `full`.

The fan-out engine picks the child profile the same way at spawn time: `reviewer = parent_is_reviewer
or is_review_task(task)`. The reviewer marker is inherited by the whole sub-tree (see
`sorcar-fanout-guard`).

## Prompt side
For a non-full profile (with basic tools on), `SorcarAgent.run` appends `RESTRICTED_PROFILE_NOTE`
listing the offered tools, so SYSTEM.md rules that need missing tools (tmp/PROGRESS.md, Write, browser
research, run_parallel) are waived and the child reports everything in `finish`.

## Gotchas
- The word match is lexical: "audit log" makes a task a review task, and a paraphrased review escapes
  it. Name `tool_profile="review"` explicitly when it matters.
- Profiles only filter built-in tools. Caller-supplied `tools` are always appended unfiltered.

## Sources
- `src/kiss/agents/sorcar/sorcar_agent.py` (`TOOL_PROFILES`, `SorcarAgent._tool_profile`, `_is_reviewer_subagent`, `RESTRICTED_PROFILE_NOTE`, `run_tasks_parallel`)
- `src/kiss/agents/sorcar/fanout_guard.py` (`is_review_task`, `is_implementation_task`)
