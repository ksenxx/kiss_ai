---
title: 'Always-on gateway: ChannelRunner poll tick'
uuid: 1dff7c85-630a-4fa2-aef3-d519b67af523
summary: 'Always-on gateway poll tick: ChannelRunner.run_once/_run_tick, cursor, allow
  list, bot-reply dedupe, per-thread chat continuity, [SILENT] replies, model/budget
  overrides.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Always-on gateway: the `ChannelRunner` poll tick

A **gateway** makes a chat a prompt surface: every message to the bot becomes a daemon task
and the answer is posted back in the thread. It is not a long-running process: it is a
**one-shot tick** (`kiss-<channel> --channel CH [--pairing] [--quiet]`) scheduled as a cron
*command* job (see `cron-dispatch-and-gateways`). `channel_main` builds the backend with
`_make_backend(workspace=...)` when the factory accepts `workspace`, resolves
`--allow-users` via `backend.find_user`, and calls `ChannelRunner(...).run_once()`.

## `run_once`
- `state_path=None` → stateless legacy behaviour (`_run_tick` only).
- Otherwise takes the per-channel lock **non-blocking** (`channel_state_lock(..., blocking=False)`):
  an overlapping tick returns 0 immediately. Loads state, returns 0 while
  `paused_until > now` (circuit breaker), runs `_run_tick`, counts an escaping exception as a
  transport failure, resets `failures` on success. See `channels-gateway-state-file`.

## `_run_tick`
1. `backend.connect()` (False → `RuntimeError`); if a channel name was given,
   `find_channel` (not found → `RuntimeError`) and `join_channel`.
2. `_redeliver_pending()` (ledger), `_prune_stale_threads()` (entries older than 7 days by
   `updated_at`).
3. `poll_messages(channel_id, oldest=state["cursor"], limit=50)` → `(messages, new_cursor)`.
   Destructive-read backends (Signal) get the state dict via `bind_channel_state` first and
   park unreturned envelopes in `pending_envelopes`; state is saved right after the poll.
4. Per message: skip `is_from_bot`; skip senders outside `_effective_allow()` (sending a
   pairing code when enabled); skip if `_has_bot_reply` (only checked when `reply_count > 0`);
   else `_handle_message` then `_ack_message` (optional backend hook for non-cursor polls such
   as email's UNSEEN search; failures only logged).
5. `_process_thread_continuations` — for up to 20 known threads per tick (most recently
   updated first, rotating offset `thread_rotation` when there are more), polls the thread,
   selects user follow-ups newer than the stored `last_reply_ts`, joins them into one prompt
   and resumes the thread's daemon chat.
6. The new cursor is stored only if all continuations succeeded, so a failed tick is retried.
7. `finally: backend.disconnect()` (errors logged).

## Running one task (`_launch_task`)
A `KissWebChatAgent` carrier resumes the thread's stored chat id (per-thread continuity),
sends a best-effort `send_typing`, and calls `run_agent_via_kiss_web` with the runner's
model, budget, `work_dir` and `tools=tools_file`. The prompt is the message text
(`strip_bot_mention`) plus `_prompt_context`: channel/thread ids, whether an automatic
summary will be posted, and "finish with exactly [SILENT] and no reply is sent".
After the run the summary is posted unless the agent already replied in the thread
(`_bot_replied_in_thread`, or `_bot_posted_after` for continuations) or `summary_for_reply`
returns `None` for a silence token (`[SILENT]`, `NO_REPLY`, HTML tags stripped). An exception
posts "Error processing your message: ...".

## Model / budget / work dir
`resolve_channel_overrides`: explicit `-m`/`-b` win; otherwise config keys
`channel_model_name` / `channel_max_budget` (positive finite float) from `ChannelConfig.load_metadata()`;
otherwise the default model and `DEFAULT_CONFIG.max_budget`. Not available for Slack
(token-only store) and Google Chat. The runner's `work_dir` falls back to
`$KISS_HOME/channel_work` only when empty; the CLI's `-w` default is the launch directory
(`KISS_WORKDIR` or cwd).

## Sources
- `src/kiss/agents/third_party_agents/_channel_agent_utils.py` (`ChannelRunner.run_once`, `_run_tick`, `_handle_message`, `_launch_task`, `_prompt_context`, `_process_thread_continuations`, `_threads_to_process`, `_ack_message`, `summary_for_reply`, `resolve_channel_overrides`, `channel_main`)
- `src/kiss/agents/third_party_agents/_channel_cli.py` (`_launch_work_dir`)
