---
title: GIL stall watchdog for kiss-web (faulthandler thread-stack dumps)
uuid: 67a604b4-54a2-4117-a423-bc0502094fab
summary: 'stall_watchdog: faulthandler.dump_traceback_later re-armed by a heartbeat
  thread (60s/5s) dumps all thread stacks to kiss-web-stderr.log when the GIL is held.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# GIL stall watchdog for kiss-web (faulthandler thread-stack dumps)

## Why it exists
When one thread holds the GIL inside a long C-level call, every other Python thread freezes: no logging, no websocket answers, one core at 100%. The process looks dead. In the 2026-09-12 kiss-web outage, a regex with quadratic backtracking over a 5.8 MB tool result caused 83 minutes of log silence with nothing to point at the culprit (commit "eliminate quadratic regex backtracking that froze the GIL and hung the server"). This module makes the next stall leave a trace.

## How it works
- `faulthandler.dump_traceback_later(timeout, repeat=...)` runs a C watchdog thread that does not need the GIL.
- `StallWatchdog._heartbeat`, a small Python daemon thread, re-arms it every `interval` seconds.
- When the heartbeat cannot get the GIL for `timeout` seconds, the C thread dumps EVERY thread's stack. Dumps repeat every `timeout` while the stall lasts, so a reader can tell a spinning frame from a slowly progressing one.
- Defaults: `_DEFAULT_TIMEOUT_SECS = 60.0`, `_DEFAULT_INTERVAL_SECS = 5.0`.
- Output goes to the process stderr, which under systemd is the appended `kiss-web-stderr.log`. The file must have a real file descriptor, because faulthandler writes with `write(2)`, bypassing Python buffering. `start_stall_watchdog` returns `None` and arms nothing when stderr has no fd (for example a `StringIO` under a test harness).

## Lifecycle
`RemoteAccessServer.start()` calls `start_stall_watchdog()` before `asyncio.run` and calls `stall_watchdog.stop()` at the end of its `finally` block. `stop()` cancels the pending dump (`faulthandler.cancel_dump_traceback_later`) and ends the heartbeat. Before the fix "disarm stall watchdog heartbeat thread on server shutdown", an in-process restart or a test left a heartbeat thread behind.

## Reading a dump
Look for the thread whose frame stays the same across repeated dumps: that is the GIL holder. Typical culprits are regexes over huge strings, giant `json.dumps`, and synchronous C-extension calls on the event-loop thread.

## Related diagnostics
- An event loop that is slow but not GIL-frozen (heavy IO/Docker load) shows up as UDS clients dropped by the 30 s drain timeout, not as dumps (see `server-broadcast-routing`).
- `_handle_shutdown_signal` logs active tabs and RSS on SIGTERM, and startup logs RSS (`_rss_mb`).

## Sources
- `src/kiss/server/stall_watchdog.py` (module docstring, `StallWatchdog`, `start_stall_watchdog`, `_DEFAULT_TIMEOUT_SECS`, `_DEFAULT_INTERVAL_SECS`)
- `src/kiss/server/web_server.py` (`RemoteAccessServer.start`)
- git: `02455a12a`, `75ebcf83d`
