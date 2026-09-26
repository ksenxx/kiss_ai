---
title: 'fanout_guard: tasks JSON parsing and review-task detection'
uuid: 30164436-33f7-4c89-9d45-57007903c74e
summary: 'fanout_guard: parse_tasks_json (JSON array of strings, lenient backslash
  repair, $(cat) hint) and is_review_task / is_implementation_task heuristics.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# fanout_guard

`src/kiss/agents/sorcar/fanout_guard.py` is shared by `run_parallel`, `run_agent` and the daemon.

## `parse_tasks_json(tasks, name="tasks")`
Parses the `tasks` argument of `run_parallel` and the `commands` argument of `run_commands_parallel`.
- Decoding goes through `_decode_json_leniently`: first `json.loads(strict=False)`, which accepts raw newlines from
  heredocs. If that fails, lone backslashes (`\|`, `\(`, `\.` in grep/sed patterns) are doubled and it retries.
  Strict decoding had failed 21 reviewer calls in one day, each one a wasted step.
- It rejects anything that is not a JSON array, an empty array, or an array with non-string or blank items, raising a
  `ValueError` whose message tells the model how to fix the call. A value starting with `$(` or a backtick gets a
  specific hint: shell substitutions are never expanded, so read the file and paste the array. Before
  this, a literal `"$(cat tasks.json)"` was dispatched as a single sub-agent task.
- The tools turn the `ValueError` into an `Error: ...` string rather than raising.

## Review detection
- `is_review_task(task)`: matches `_REVIEW_WORDS`, stems such as review, audit, critique, inspect, regression,
  read-only, bug-hunt, vulnerability and adversarial.
- `is_implementation_task(task)`: matches `_IMPLEMENTATION_WORDS`, verbs such as implement, fix, patch,
  refactor, create, add, write, modify, edit, update, delete, remove, rename and migrate. "review X and fix
  what you find" therefore keeps the full toolset. When in doubt the child gets full tools.
- Both are **lexical heuristics**. "audit log" trips review, and a paraphrase avoiding every stem escapes it.

## Reviewer marker
`run_tasks_parallel` sets `_subagent_info["reviewer"] = parent_is_reviewer or is_review_task(task)`, so
the marker is inherited down the sub-tree and across daemon dispatches. Since commit ee6ba3d38 the marker
**only** selects the read-only `review` tool profile (see `sorcar-tool-profiles`). It no longer
limits spawning or budget (ReviewQuota, review budget caps and `KISS_REVIEW_BUDGET_FRACTION` were removed).

## Sources
- `src/kiss/agents/sorcar/fanout_guard.py` (`parse_tasks_json`, `_decode_json_leniently`, `is_review_task`, `is_implementation_task`)
- `src/kiss/agents/sorcar/sorcar_agent.py` (`run_tasks_parallel`, `_get_tools.run_parallel`)
- git: ee6ba3d38 "remove review fan-out guardrails"
