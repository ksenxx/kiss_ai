---
title: Mid-run steering, <task> follow-up queue and ask_user_question answers
uuid: 270d5706-8b61-41cd-bc1d-7f5141964f19
summary: 'Mid-run input: pending_user_messages injected before next model step, <task>
  follow-up queue, followup_queue_closed, viewer-tab routing, ask_user_question answers.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Mid-run steering, <task> follow-up queue and ask_user_question answers

## Three destinations for text typed while a task runs
`_cmd_append_user_message`, and `_cmd_run` when the tab already has a task thread, route the prompt through `_route_prompt_to_owner`. The prompt goes to one of three places:

1. **Steering** (the default): appended to the owner's `AgentState.pending_user_messages` under `STATE_LOCK`. The live agent's pre-step hook drains the list and injects it into the model conversation before the next model call.
2. **Follow-up tasks**: when the owner is `server_owned` and `followup_queue_closed` is not set, a message wrapped in `<task>...</task>` is not injected (this check runs before the ask-user one). Its task blocks go to `AgentState.queued_followup_tasks`, and the task runner's per-subtask loop runs them one by one as additional sequential subtasks of the same submission.
3. **Ask-user answer**: while the agent is blocked inside `ask_user_question`, a plain message is delivered as the answer, because the agent cannot drain steering until the tool returns. Every viewer tab receives `askUserDone`, the same as for `userAnswer`.

## `followup_queue_closed`
Raised under `STATE_LOCK` once the subtask loop decides no more drains will happen: the final drain found the queue empty, a subtask failed, or finalization began. From then on a `<task>` message routes to plain steering, because a task queued after the last drain would be echoed and then silently discarded. It is never reset; each run has a fresh state.

## Prompts typed during teardown
Queued messages are only consumed by the agent's pre-step drain. A prompt that arrives after the last `agent.run` returned would be lost. `commands._reruns_after_teardown(owner)` (server-owned, `followup_queue_closed`, not sub-agent, not frontend-closed) makes the queueing site skip its echo, and `_run_task`'s cleanup re-submits the leftovers as a new run (`_redispatch_leftover_prompts`). See `server-task-lifecycle`.

## Viewer tabs
A tab opened from history while the task runs in another tab has no live task of its own (`is_task_active=False`). The prompt is routed via the printer's per-task subscriber map to the running task's state, so the viewer can steer. Without that the text was silently dropped. `_stop_task` resolves viewer tabs the same way, so a second client can stop the task.

## ask_user_question plumbing
`_ask_user_question` stores the question in `AgentState.pending_ask_question`, so replays re-show the modal to clients that connect mid-question. It then blocks in `_await_user_response`. That waits on the state's `user_answer_queue`, checks `stop_event` periodically (raising `KeyboardInterrupt` on stop), and raises `ToolCallInterrupted` when the user presses the tool's own Stop (`interruptTool`). `_await_user_response` reuses the queue `_ask_user_question` already resolved: re-resolving opened a TOCTOU window in which a closeTab and reopen swapped queues (bug W2-F9). `_cmd_user_answer` clears `pending_ask_question` under the lock when the answer is consumed.

## Echoes
Injected prompts are echoed as `prompt` events. When the owner's persisted task id is not yet known, the prompt is parked in `unattributed_prompt_echoes`.

## Sources
- `src/kiss/server/commands.py` (`_cmd_append_user_message`, `_cmd_run`, `_route_prompt_to_owner`, `_reruns_after_teardown`, `_cmd_user_answer`)
- `src/kiss/server/task_runner.py` (`_ask_user_question`, `_await_user_response`, `_redispatch_leftover_prompts`)
- `src/kiss/server/agent_state.py` (`AgentState`)
