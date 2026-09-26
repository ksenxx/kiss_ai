---
title: summary tool, mid-task user steering and set_model
uuid: b0594f8d-7633-44c7-894e-25841813bb2f
summary: 'summary tool (no-op, 10-step rule is prompt-only), steering via pre_step_hook
  (User says: ...), finish blocked while messages pending, set_model mid-task.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# summary tool, steering and set_model

## summary tool
`sorcar_agent.summary(description)` is a module-level function that discards its argument and returns
`"Summary recorded."`. It exists so the model's progress recap becomes a `tool_call` event that the UI renders
(Markdown) and that transcript digests pick up as SUMMARY entries (`task_digest`). The rule "when the step counter shows
9, 19, 29, ... the next call must be summary" lives **only in `SYSTEM.md`**. No code enforces it or counts steps
for it. It is offered in the `full`, `review` and `assistant` profiles. It moved from ChatSorcarAgent to
SorcarAgent in commit c65a53f83.

## Steering a running task
Wired in `SorcarAgent.perform_task`:
- `pre_step_hook = _drain_pending_user_messages`: at the top of every model step, it drains queued follow-ups
  through the printer's duck-typed `drain_pending_user_messages()` (the server keeps them per task id) and adds each
  one as a user message: `User says: <msg>. Take the message into account and finish your task.`
- `tool_call_guard = _block_finish_when_user_message_pending`: rejects a `finish` call while
  `printer.has_pending_user_messages()` is true. The server accepts `appendUserMessage` while a model call is in
  flight, but draining only happens at the top of a step. Without the guard, a message queued after the last drain
  would be silently dropped when the in-flight response finished. Blocking forces one more step, and that step's drain
  delivers the message.
- Both hooks are copied onto every sub-session's `KISSAgent` and cleared in `SorcarAgent.run`'s finally.

## set_model
`set_model(model_name)` changes the live executor's model mid-task (or defers the change if no model exists yet).
It keeps the conversation, `usage_info_for_messages` and Gemini thought signatures. It drops a
`reasoning_effort` that only matched the old model's default and cleans provider-specific config
(`_sanitize_model_config_for_switch`, `use_responses_api`). It keeps a custom `base_url`/`api_key` unless the old one
was a factory default of a different provider, and rebuilds the cached tools schema. It refuses cc/codex CLI
models when `docker_image` isolation is on. The model picker shows the switch only while the task runs
(`_show_model_in_picker`), and `last_model` is never persisted. `TIPS.md` markets set_model and steering as
"Novel Features".

## Sources
- `src/kiss/agents/sorcar/sorcar_agent.py` (`summary`, `SorcarAgent.perform_task`, `_drain_pending_user_messages`, `_block_finish_when_user_message_pending`, `_get_tools.set_model`, `_sanitize_model_config_for_switch`)
- `src/kiss/SYSTEM.md` (periodic activity summaries rule)
- `src/kiss/TIPS.md` (set_model and Steering-on-the-Fly)
