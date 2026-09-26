---
title: Task ids, chat ids, parent ids and the "Agent Failed Abruptly" sentinel
uuid: 5d386b92-0b33-46c0-bca1-4c0b7a6ba78a
summary: task_history.id is uuid4 hex (32 chars), chat_id groups tasks of one chat
  session (preallocated by ChatSorcarAgent), parent_task_id marks sub-agents; rows
  start with the sentinel and an owner token.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Task ids, chat ids, parent ids

## Ids
- Task id: `uuid.uuid4().hex`, generated in `_add_task`. `is_task_history_id` checks `^[0-9a-f]{32}$`.
- Chat id: also a 32-char uuid4 hex (`_allocate_chat_id`). `ChatSorcarAgent.run` preallocates `self._chat_id`
  before the first row is written so early consumers (printers, worktree setup) see the final id;
  `_add_task(task, chat_id)` with `""` allocates a new one. A continuation task reuses the chat id.
- Parent id: `parent_task_id` is set on sub-agent rows (`run_parallel`, `run_agent`, merge SEA, /ask). It is
  taken from `extra["parent_task_id"]` or the legacy nested `extra["subagent"]["parent_task_id"]`
  (`_extract_parent_task_id`). `_HISTORY_NOT_SUBAGENT` filters these out of history lists and of
  `_load_chat_context`, which feeds the "Previous tasks and results" section of the chat prompt.
- Worktrees are NOT keyed by chat id (see `git-worktree-lifecycle`).

## Row creation: `_add_task(task, chat_id="", extra=None)`
Inserts under `_rw_lock.write_lock()` with `result = "Agent Failed Abruptly"` (the sentinel), metadata known at
start (model, work_dir, version, toggles, startTs, max_budget), and `owner = _process_owner_token()`. Returns
`(task_id, chat_id)`. `_save_task_result` later overwrites the result (and journals it); `_save_task_extra`
writes usage columns (tokens, cost, steps, end_ts), leaving `is_favorite` alone.

A row with the sentinel and `end_ts = 0` is either still running or was killed; see `db-orphan-task-recovery`.

## Owner token
`_process_owner_token()` returns `<pid>-<uuid hex>` and holds an exclusive `flock` on
`<KISS_HOME>/task-owners/<token>.lock` for the process lifetime. `_owner_is_alive(token)` tests liveness by
trying that lock; the kernel releases it when the process dies. The marker is deleted at normal exit
(`_release_owner_marker`, atexit) and re-minted if `KISS_HOME` is redirected.

## Usage roll-up
A parent's `cost`, `tokens`, `steps` include its sub-agents (children are also separate rows). Spend incurred after the row was finalized (the merge SEA resolving its auto-merge conflict) is added atomically with `_add_task_usage`.

## Sources
- `src/kiss/agents/sorcar/persistence.py` (`_add_task`, `_add_task_usage`, `_allocate_chat_id`, `is_task_history_id`, `_extract_parent_task_id`, `_HISTORY_NOT_SUBAGENT`, `_load_chat_context`, `_process_owner_token`, `_owner_is_alive`, `_save_task_result`, `_save_task_extra`)
- `src/kiss/agents/sorcar/chat_sorcar_agent.py` (`ChatSorcarAgent.run`)
