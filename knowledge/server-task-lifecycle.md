---
title: 'Task lifecycle in the daemon: run command to terminal status'
uuid: d433b055-7b8d-494f-8f3a-180442387b0d
summary: 'run command to terminal status: _cmd_run queue/steer/fresh run, _run_task
  try/finally, overrides, tools, agent.run subtasks, auto-commit/worktree, leftover
  prompt re-dispatch.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Task lifecycle in the daemon: run command to terminal status

## 1. `_cmd_run` (commands.py)
Under `STATE_LOCK`, `_cmd_run` looks up the tab's previous `AgentState` (`prev`):
- A **merge is in progress** on the tab: the run is refused ("Cannot run a task while a merge is in progress").
- **`prev.task_thread is not None`**: the prompt is NOT a new run. It is routed as steering to the live owner (`_route_prompt_to_owner`, see `server-steering-and-followups`). The gate is the installed thread, not `is_task_active`: the worker raises that flag only after it starts, and gating on the flag once dropped a second submit during the startup window (bug S3-05).
- The agent is blocked in **`ask_user_question`**: the prompt becomes the answer.
- **`_update_installing`** is set (self-update about to restart the daemon): refused with "An update is being installed". See `server-stop-shutdown-and-reset`.
- Otherwise it is a **fresh run**. The chat id is chosen in this order: `prev.chat_id`, the command's `chatId`, the tab's resumed chat view (`_tab_chat_views`), a new uuid. A new `AgentState` is keyed by a fresh `_state_key` uuid with `server_owned=True`, a `stop_event`, a `user_answer_queue` (maxsize 1), and `client_run_token` = the command's `taskId`. `last_user_prompt` is stamped at once, so a reconnect during setup replays the prompt.

`_cmd_run` broadcasts the initial `clear` event synchronously before starting the thread. The chat-id-to-tab mapping is then visible immediately, and a fast `resumeSession` cannot race the worker's first broadcast.

## 2. `_run_task` (task_runner.py)
Runs on a daemon worker thread. The ENTIRE body sits inside an outer try/finally that always clears `state.task_thread` and broadcasts `status running=False`. Agent-script overrides (`apply_agent_overrides(cmd)`) execute untrusted user code inside that try. A Stop-watchdog `KeyboardInterrupt` landing there once skipped the cleanup and bricked the tab until restart, because `_cmd_run` queues instead of running while `task_thread` is set (fix C-RC2). An `AgentFileError` becomes a failed task result.

## 3. `_run_task_inner`
In order:
1. Resolve the state (`_resolve_run_state`).
2. Set the model (command, then tab default).
3. Mark sub-agents via `agent._subagent_info` (`parent_task_id`, `parent_tab_id`, a `reviewer` flag re-checked on the EFFECTIVE prompt after overrides, `side_channel`).
4. Check model availability ("No model available..." when no API key).
5. Admit the worktree or main tree. A non-worktree run is refused only while a worktree merge is in progress on the repo (`_wt_merge_on_repo`) or a main-tree mutator (Discard, manual commit) holds a claim (`_main_tree_claim_reason`); several non-worktree tasks may run in one repo. Admitted runs set `is_running_non_wt`, which merges check via `_any_non_wt_running`.
6. Load tools with `load_tools_file(cmd["toolsFile"])`.
7. Run subtasks. Each call is `agent.run(prompt_template=..., model_name=..., work_dir=..., printer=self.printer, ask_user_question_callback=self._ask_user_question, is_parallel=..., use_worktree=..., ...)`, and queued `<task>` follow-ups run as further sequential subtasks. Each subtask row is persisted (`_persist_subtask_row`).
8. Finish up. A stop is turned into the stopped result by `_cancel_outcome`, and a failure into `_broadcast_failure_result`. Non-worktree runs auto-commit only if `auto_commit_mode` is on and the run succeeded (`_autocommit_changes`, which claims the repo). Worktree runs go to `_handle_worktree_action` / `_present_pending_worktree`. Then `_refresh_files_after_task`, and `_merge_deferred_worktrees` for worktrees whose merge waited on this run.

## 4. Cleanup tail (`_run_task` finally)
`_run_task` snapshots leftover `pending_user_messages` (typed after the agent's last drain) under the lock. After disposal (`_dispose_if_closed`) and deferred merges, `_redispatch_leftover_prompts` re-submits them as the tab's next run via `_cmd_run`: joined by blank lines, inheriting settings but not `taskId`/`_state_key`. It is skipped when the run was stopped, shut down, frontend-closed, or is a sub-agent.

## Terminal signal
Clients treat `status` with `running: false` (tab-stamped) as the end of the run. `daemon_client.run` returns only after that, not on `result`, because persistence, auto-commit and worktree cleanup still run after `result`.

## Result row sentinel
A task killed with the process leaves `task_history.result` at "Agent Failed Abruptly". The next startup's orphan sweep rewrites it to "Task terminated unexpectedly (process killed)". The graceful paths avoid this with `_stop_active_agent_tasks`.

## Sources
- `src/kiss/server/commands.py` (`_cmd_run`, `_reruns_after_teardown`, `_route_prompt_to_owner`)
- `src/kiss/server/task_runner.py` (`_run_task`, `_run_task_inner`, `_redispatch_leftover_prompts`, `_cancel_outcome`, `_persist_subtask_row`)
- `src/kiss/server/agent_file.py` (`apply_agent_overrides`), `src/kiss/server/tools_file.py` (`load_tools_file`)
- `src/kiss/server/web_server.py` (`_stop_active_agent_tasks`)
