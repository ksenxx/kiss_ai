---
title: 'ChatSorcarAgent: chat sessions, prior-task prefix and task_history persistence'
uuid: b3c3a485-3070-49dd-b474-16c33e683c7d
summary: 'ChatSorcarAgent: chat_id, the Previous tasks prefix (MAX_TASKS, digest limits),
  task_history early row and final save, replay events, _subagent_info marker.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# ChatSorcarAgent

`ChatSorcarAgent(SorcarAgent)` reproduces the VS Code server's stateful chat workflow as a reusable agent. Every
`run_parallel` child is one of these.

## Chat state
- `chat_id` (empty means new): `new_chat()` resets it, `resume_chat_by_id(id)` resumes one, and
  `resume_from_task_id(task_id)` is a one-shot seed that loads the parent-task chain
  (`_load_task_chain_context`) instead of the whole chat for the next prompt.
- `build_chat_prompt(prompt)` returns `"# Task\n" + prompt` when there is no history. Otherwise it returns
  `## Previous tasks and results from the chat session for reference` + history + `---` +
  `# Task (work on it now)` + prompt.
  - It keeps at most `MAX_TASKS = 10` prior tasks, **keeping the first two** and dropping from the middle.
  - With `chat_history_digest` (env `KISS_CHAT_HISTORY_DIGEST`, default on), `_digest_history` quotes only the newest
    `DIGEST_FULL_RESULTS = 2` results in full. Older entries are tag-stripped and cut (task 600 chars, result 300),
    and the oldest are dropped until the prefix fits `DIGEST_MAX_PREFIX_CHARS = 6000`. The prefix is re-sent on
    every step, so its size matters.

## run() sequence
1. Allocate a chat id if needed (`_allocate_chat_id`). Build the prompt, applying `bare_path_task.with_open_directive`
   *after* `history_prompt` has been captured, so history shows the raw text.
2. Resolve the model, budget, work_dir and parallel settings exactly as `_reset` will. Write an **early** `task_history` row via
   `_add_task` (with `startTs`, `auto_commit_mode` and `max_budget` in `extra`) and publish `_last_task_id` under
   `_task_id_lock`, which is read by server threads.
3. Printer wiring: set the thread-local `task_id`, call `agent_task_allocated`, broadcast `new_tab` for sub-agents,
   start recording, broadcast `tasks_updated`, call the `_on_task_id_allocated` callback, then emit a `task_settings` event.
4. `_record_frequent_task` (not for sub-agents), then `SorcarAgent.run`.
5. In `finally`: `_save_task_result` (the summary, or "Task failed" / "Task interrupted"), then `_save_task_extra` with
   the model (the launch model: `_launch_model_name` wins over the current `model_name`), tokens, cost, steps and `endTs`. Usage is zero if `super().run` never started. Then
   `_persist_replay_events_if_missing` synthesizes prompt/result/followup events when no recording printer
   stored a transcript (channel agents, headless runs).

Private kwargs: `_skip_persistence`, `_on_task_id_allocated`, `_history_prompt` (raw user text for history when
the LLM sees an internal directive, e.g. slash-command rewrites). The `use_worktree` kwarg is consumed and
ignored here. `is_worktree` is recorded only when `work_dir` is really inside `_wt_dir` (`_dir_inside_worktree`).

## Sub-agent marker
`_subagent_info = {parent_task_id, parent_tab_id, reviewer}` is set by the fan-out engine or `run_agent`. It affects
the persisted `extra.subagent`, the Task Settings ("Is subagent", "Parent task id"), budget-exhaustion partial
results (`sorcar-budget-and-step-limits`) and the ephemeral browser.

`_extract_result_summary` handles every YAML shape a model may emit. For example, a non-string summary is dumped as YAML,
because passing it raw to SQLite would raise `ProgrammingError`.

## Sources
- `src/kiss/agents/sorcar/chat_sorcar_agent.py` (`ChatSorcarAgent`, `build_chat_prompt`, `_digest_history`, `run`, `_persist_replay_events_if_missing`, `_extract_result_summary`, `_dir_inside_worktree`)
- `src/kiss/agents/sorcar/persistence.py` (`_add_task`, `_save_task_result`, `_save_task_extra`, `_load_chat_context`) (persistence area)
