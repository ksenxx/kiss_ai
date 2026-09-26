---
title: Model streaming, stall timeout and Stop-button abort (stream_abort.py)
uuid: e3395f35-2777-40a9-aebc-b9220f410c8f
summary: 'Streaming and abort: StreamAbortWatchdog, stop_aware_events, socket shutdown,
  stream_stall_timeout 180 s, KeyboardInterrupt on Stop vs retryable TimeoutError
  on stall, thinking bracket.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Model streaming, stall timeout and Stop-button abort

## Streaming contract
OpenAI and Gemini adapters stream iff a `token_callback` is set (the `stream` key is
framework-only; adapters set it) and otherwise make a non-streaming call. Anthropic always uses
`messages.stream` and only skips forwarding tokens when there is no callback.
Tokens go through `Model._invoke_token_callback`; reasoning is bracketed by
`_invoke_thinking_callback(True/False)` with state in `Model._thinking_open`, and
`_close_thinking_if_open()` closes an open bracket.

## Why a watchdog exists
A thread blocked in `recv()` on a silent httpx connection cannot be stopped: the cooperative stop
flag is read only when the agent emits output, and the injected `KeyboardInterrupt` only lands at
a bytecode boundary. A task was once unstoppable for 178 s (`reports/stop_button_delay_2026-08-05.html`,
commit "make Stop button responsive to wedged model streams"). Closing the response is not
enough: only `socket.shutdown(SHUT_RDWR)` makes the blocked `recv()` return.

## `StreamAbortWatchdog` (`src/kiss/core/models/stream_abort.py`)
Daemon thread per stream; the reader calls `beat()` on every event. It aborts when:
- **stop**: the thread's stop event (`stop_signal.get_thread_stop_event()`) is set; polled every
  0.1 s; sets `.stopped`.
- **stall**: no event for `stall_timeout` seconds; sets `.stalled`. This is event-level, so it
  catches servers that keep the connection alive with pings the SDK filters out.
`_abort` shuts down the TCP socket reached through httpcore's `network_stream` extension and does
NOT call `close()` (freeing the fd under a reader entering `poll()` could leave it polling a
reused descriptor); `close()` is only the fallback. On Windows the socket is also closed because
Winsock `shutdown` does not wake a blocked `recv()`.
`stop()` disarms under the same lock as `_claim_abort` and joins (5 s cap), so no late abort can
shut down a socket httpx already returned to its pool for another request.

## `stop_aware_events(stream, stall_timeout, on_abort, name)`
Generator wrapper for `for event in stream`. After the loop or on an exception it raises:
- `KeyboardInterrupt("Agent stop requested")` for a user stop (not retryable: the whole stack
  unwinds into "Task stopped by user");
- `stall_error(stall_timeout)`, a `TimeoutError` that `KISSAgent._run_agentic_loop` retries.
An aborted socket usually ends iteration at EOF instead of raising, so both flags are checked after
the loop too; otherwise partial text would be returned as a completed answer. `on_abort` (usually
`_close_thinking_if_open`) runs first. The stream is always closed in `finally` (`_close_stream`)
so the connection returns to the pool. Callers whose loop body can raise must `close()` the
generator (OpenAI v1 `_stream_chat_completion` and v2 `_consume_stream` do this in `finally`).

## Per adapter
- OpenAI Chat Completions (`OpenAICompatibleModel._stream_chat_completion`) and Responses
  (`OpenAICompatibleModel2._consume_stream`): `stop_aware_events`. v1 keeps the answer if the
  transport fails **after** `finish_reason` arrived (only the usage tail was lost), to avoid a
  re-sent and re-billed duplicate turn.
- Gemini (`GeminiModel`): `stop_aware_events` over a wrapper stream plus a response-tracking httpx
  client so the socket can be reached.
- Anthropic (`AnthropicModel._stream_message`): runs its own `StreamAbortWatchdog` loop;
  httpx read timeout = `stream_stall_timeout`, connect 10 s, SDK `max_retries=1`. Maps
  `httpx.TimeoutException` / `APITimeoutError` to the stall error, and checks
  `stop_signal.stop_requested()` too because a request silent before headers arrive has no
  watchdog yet. `_create_message` closes the thinking bracket in `finally` on every exit.

## Config
`model_config["stream_stall_timeout"]` (seconds, default `DEFAULT_STREAM_STALL_TIMEOUT = 180.0`),
read into `Model._stream_stall_timeout`; framework-only, never forwarded to the SDK.

## Known bug classes (fixed)
- Retry after a transport error rendered the answer as "thinking" because the bracket stayed open:
  every exit path now closes it.
- OpenAI SDK silent retries duplicated billed turns: `_MAX_RETRIES = 1` in both OpenAI and
  Anthropic adapters.

## Sources
- `src/kiss/core/models/stream_abort.py` (`StreamAbortWatchdog`, `stop_aware_events`, `stall_error`, `DEFAULT_STREAM_STALL_TIMEOUT`)
- `src/kiss/core/models/anthropic_model.py` (`AnthropicModel._create_message`, `_stream_message`, `_stream_was_stopped`, `_stop_error`, `_stall_error`)
- `src/kiss/core/models/openai_compatible_model.py` (`OpenAICompatibleModel._stream_chat_completion`, `_MAX_RETRIES`)
- `src/kiss/core/models/openai_compatible_model2.py` (`OpenAICompatibleModel2._consume_stream`)
- `src/kiss/core/models/gemini_model.py` (`GeminiModel`, `_ResponseTrackingHttpxClient`)
- `src/kiss/core/models/model.py` (`FRAMEWORK_ONLY_CONFIG_KEYS`, `Model._close_thinking_if_open`)
