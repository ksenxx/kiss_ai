---
title: Task Stop signal - per-thread stop event and KeyboardInterrupt propagation
uuid: 1591a66c-53eb-4135-814a-3b2de4cc8528
summary: 'Task Stop: stop_signal.py per-thread stop event set by the server''s _stop_task;
  JsonPrinter._check_stop and model stream watchdogs raise KeyboardInterrupt.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Task Stop signal

The whole-task Stop is separate from the per-tool-call interrupt (see `core-tool-call-interrupt`).

## Pieces
- `src/kiss/core/stop_signal.py` keeps one `threading.local` with `set_thread_stop_event(event | None)`,
  `get_thread_stop_event()` and `stop_requested()`. Each task, or parallel sub-agent, runs on exactly one
  thread, and the server sets that thread's event when the user presses Stop.
- `JsonPrinter._thread_local.stop_event` is a **property over this storage**, so binding a stop event to a
  printer thread publishes it here too. There is a single source of truth.
- Cooperative check: `JsonPrinter._check_stop()` raises `KeyboardInterrupt("Agent stop requested")`. It runs
  only when the agent emits something (a print or a streamed token).
- Below the agent: model streams (`kiss.core.models.stream_abort`, `Model` blocking waits): the requesting thread captures its
  thread-local `get_thread_stop_event()` and passes it to a watchdog thread, which shuts down the socket on
  Stop; the requesting thread then raises the same `KeyboardInterrupt`.
  Before this, a quiet model request kept a stopped task alive until the provider's stall timeout (180 s on
  Anthropic). Post-mortem: `reports/stop_button_delay_2026-08-05.html`.
- When the task does not exit promptly, `_stop_task` in `server/task_runner.py` forces a `KeyboardInterrupt`
  into the task thread with `PyThreadState_SetAsyncExc`. This is why `KISSAgent` publishes usage atomically (see `core-budget-and-token-accounting`).

## How KISSAgent reacts
`KeyboardInterrupt` is a `BaseException`, so `_run_agentic_loop`'s `except Exception` retry path never
catches it. It propagates through `run()` (the trajectory is still saved in `finally`). If it arrives
while a tool is being interrupted, `_execute_tool` re-raises it rather than turning the Stop into
"User interrupted the tool call.".

## Guidance for new blocking code
If a tool or model call can block for a long time, watch `stop_signal.get_thread_stop_event()`, and
`tool_interrupt.current_tool_interrupt_event()` for tools. Abort promptly and raise `KeyboardInterrupt`
for a task stop.

## Sources
- `src/kiss/core/stop_signal.py`
- `src/kiss/server/json_printer.py` (`_check_stop`, `stop_event` property), `src/kiss/server/task_runner.py` (`_stop_task`)
- `src/kiss/core/models/stream_abort.py`, `src/kiss/core/models/model.py`
