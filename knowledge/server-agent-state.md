---
title: 'AgentState: per-task live state and the agent_states registry'
uuid: a0353c27-652e-40e2-bf94-ab521b318cb9
summary: 'AgentState per-task live state in agent_states under STATE_LOCK: task_thread,
  stop_event, pending_user_messages, queued_followup_tasks, client_run_token, is_merging,
  frontend_closed.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# AgentState: per-task live state and the agent_states registry

`src/kiss/server/agent_state.py` holds the module-level registry `agent_states: dict[str, AgentState]` (task id -> state) and `STATE_LOCK`, a `threading.RLock`. `STATE_LOCK` is the same object as `VSCodeServer._state_lock` and guards the dict and EVERY field.

## Creation and keys
- A server `run` creates one in `_cmd_run`, keyed by a fresh `_state_key` uuid, with `server_owned=True`.
- The printer bridge creates one when an agent allocates its `task_history` row (sub-agents, standalone runs).
- Either way it is re-keyed to the persisted task id via `rekey` once the row exists. The pre-registered server-owned state is re-keyed, never re-created.
- A tab's state lives until the tab is explicitly closed. A worktree run's state outlives the task while its worktree awaits merge/discard, because the merge flow needs the owning agent.

## Fields
| Field | Meaning |
|---|---|
| `task_thread`, `stop_event` | worker thread and cooperative stop flag |
| `is_task_active` | raised by the worker after it starts (do not gate admission on it; gate on `task_thread`) |
| `user_answer_queue`, `pending_ask_question` | ask-user plumbing |
| `pending_user_messages` | steering text drained before the next model call |
| `queued_followup_tasks`, `followup_queue_closed` | `<task>` follow-ups (see `server-steering-and-followups`) |
| `unattributed_prompt_echoes` | echoes waiting for the owner's task id |
| `is_merging`, `merge_thread` | merge/discard claim and the thread holding it; shutdown waits for merges |
| `is_running_non_wt`, `non_wt_repo_root` | main-tree (non-worktree) task occupying a repo; the busy guard is per repo |
| `wt_merge_deferred_branch` | worktree merge refused because a main-tree task occupied the repo; retried by `_merge_deferred_worktrees` |
| `interrupted_by_shutdown`, `stop_acknowledged` | stop bookkeeping; once `stop_acknowledged` is set, the force-stop watchdog must not inject a second interrupt during cleanup |
| `frontend_closed` | tab closed by the user; suppresses leftover-prompt re-dispatch and makes `_local_tab_shown` treat the tab as not shown |
| `use_worktree`, `use_parallel`, `auto_commit_mode` | per-run toggles |
| `client_run_token` | client-minted `taskId`; scopes `stop` (see below) |
| `parent_task_id` | set for sub-agents (`is_subagent`) |

## Helpers
- `thread_alive()`: a created but not yet started thread counts as alive (`ident is None`), so the startup window is not mistaken for "finished".
- `merge_in_progress()`: a claim whose `merge_thread` died is treated as leaked and self-heals. Before that fix, a thread escaping through a `BaseException` left `is_merging` stuck, refusing all later main-tree runs until restart.
- `find_by_tab` (used by `_local_tab_shown`) looks states up by tab.

## Why `client_run_token`
`daemon_client.run` uses a synthetic `api-<hex>` tab. Its abort cascade sends `stop` with `taskId`. The token keeps a late stop from killing a newer run that reused the tab.

## Sources
- `src/kiss/server/agent_state.py` (`AgentState`, `agent_states`, `STATE_LOCK`, `rekey`, `find_by_tab`)
- `src/kiss/server/commands.py` (`_cmd_run`, `_cmd_stop`)
