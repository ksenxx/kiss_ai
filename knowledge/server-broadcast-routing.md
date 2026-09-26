---
title: 'Event broadcast routing: WebPrinter, JsonPrinter, subscribers and send locks'
uuid: 4e786a7f-09b7-413e-beef-6d93df79745c
summary: 'How daemon events reach clients: task-centric JsonPrinter recording/persistence,
  per-task tab subscribers, tabId/taskId/connId routing rules, FifoSendLock ordering,
  UDS drain timeout drops.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Event broadcast routing: WebPrinter, JsonPrinter, subscribers and send locks

## Task-centric printer
`JsonPrinter` (`json_printer.py`) keys all per-stream state (recordings, usage offsets, bash buffering, persistence) by **task id**, not tab id. A tab watching a task is a subscriber in `_subscribers[task_id] -> {tab_id, ...}`. Several tabs and several clients can watch one task. The agent thread sets `_thread_local.task_id` once its `task_history` row exists. From then on every `broadcast()` from that thread is recorded and persisted under the task and fanned out to subscribers.

`WebPrinter` (`web_server.py`) subclasses it and owns the connected endpoints: WSS `ServerConnection`s and UDS `StreamWriter`s.

## `WebPrinter.broadcast(event)` routing rules
1. **Explicit `tabId`** (status, askUser, commitMessage, ...): a targeted system event. It is sent verbatim to all clients, which filter by `tabId`, and is NOT recorded, except `TAB_STAMPED_TASK_EVENT_TYPES` (`prompt` echoes, `ask_answer`, `result`) that also carry a `taskId`. Their tabId-stripped copy is recorded and persisted under the task.
2. **No `tabId`, thread-local task id**: a task event. `taskId` is injected, the event is recorded and queued for persistence, and one `tabId`-stamped copy goes to each subscribed tab (`_fanout_stamped`). With no subscribers it is still recorded and persisted, but nothing goes on the wire.
3. **Neither**: a global event (`tasks_updated`, `remote_url`, `update_available`), sent verbatim to everyone.
4. **Non-empty `connId`**: a request/reply. The stamp is stripped and the event is sent only to that connection.

`_fanout_stamped` serializes the event once and splices a `"tabId"` into the JSON string per subscriber. This path runs per streamed token. It strips any stale `tabId` first, so duplicate keys cannot occur.

## Ordering and backpressure
- Every write to an endpoint goes through `send_lock(endpoint)`, a per-endpoint `FifoSendLock`, so payloads reach the wire in send-start order even when a `send()` is suspended on backpressure. `FifoSendLock` exists because `asyncio.Lock` removes woken waiters with an O(n) `deque.remove`. With 60 concurrent streaming tasks, thousands of queued sends made that O(n^2), starving the loop and dropping clients.
- `_schedule_send` uses `run_coroutine_threadsafe` and never waits, so `broadcast()` is safe to call from agent threads, even under `STATE_LOCK`. Pending futures are tracked in `_pending_sends` and cancelled on disconnect.
- UDS writes (`_uds_send`) wait `writer.drain()` at most `_uds_drain_timeout` (default `_UDS_DRAIN_TIMEOUT = 30.0` s). On timeout the client is dropped. If the event loop stalls 30+ s (heavy Docker/IO load), every client with a pending drain is dropped. `daemon_client.run` then raises "the sorcar daemon closed the connection before the task finished". Benchmark daemons raise `_uds_drain_timeout`.
- The unified `_watchdog` pings WSS clients every `TUNNEL_CHECK_INTERVAL` (15 s) and closes those that do not answer within `_WS_PING_TIMEOUT` (10 s).

## Replay
The task's recorded events feed session replay (`resumeSession`, `_replay_session`). A client that connects mid-task gets the transcript plus any pending ask-user question (`AgentState.pending_ask_question`).

## Talk events
`talk` events take a special path (`_fanout_talk`). See `server-voice-wake-and-talk`.

## Sources
- `src/kiss/server/json_printer.py` (module docstring, `JsonPrinter`)
- `src/kiss/server/web_server.py` (`WebPrinter.broadcast`, `_fanout_stamped`, `send_lock`, `FifoSendLock`, `_schedule_send`, `_uds_send`, `_UDS_DRAIN_TIMEOUT`, `_watchdog`)
