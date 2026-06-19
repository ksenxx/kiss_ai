# Task

When running `sorcar` CLI interactive and sending a task like "hi", the user sees:

```
✗ Daemon connection lost — type /exit to quit
```

Reproduce by writing an end-to-end test, then fix.

Use `gpt-5.5` model (NOT codex) ONLY for thorough review. Use the original model
(self) for all other work.

## Root cause — HIGH CONFIDENCE

The daemon broadcasts a **very large** `system_prompt` event (full system
prompt + injections, > 64 KiB) at the start of every task. The CLI client's
`_main()` reads events with `asyncio.StreamReader.readline()`, whose default
buffer limit is **64 KiB** (set via `asyncio.open_unix_connection(... limit= 2**16)` — see CPython source). A single line that exceeds that limit raises
`asyncio.LimitOverrunError`, which is **not** caught by `_main`'s
`except (asyncio.CancelledError, ConnectionError)`. The exception escapes,
`_main` returns, `_run_loop`'s `finally` sets `client._closed`, and
`_submit_task` immediately prints the "Daemon connection lost" message.

Source code: `src/kiss/agents/sorcar/cli_client.py` ~ line 369

```python
self._reader, self._writer = await asyncio.open_unix_connection(
    str(self.sock_path),
)
```

No `limit=` is passed → default 64 KiB buffer.

I confirmed in `tmp/probe_real.py` that the daemon emits a `system_prompt`
line that is clearly larger than 64 KiB (the JSON includes the entire
`SYSTEM.md` plus injections).

## Reproduction

- `tmp/spawn_daemon3.py` — spawns a real daemon on a fresh UDS, sends
  a raw `run` command with prompt "hi". **Did not reproduce via raw
  socket because raw `recv` has no line-length limit.**
- `tmp/probe_real.py` — connects to the user's currently-running
  kiss-web daemon at `~/.kiss/sorcar.sock`, sends `run`/"hi", sees the
  oversize `system_prompt` event but raw recv works fine.

The repro that DOES reproduce the user bug uses the actual `CliClient`
asyncio reader, exactly like production. Need to write a focused
e2e test:

1. Spin up `RemoteAccessServer` on a temp UDS (existing harness pattern
   from `test_cli_client.py`).
1. Connect a real `CliClient`.
1. Have the daemon broadcast a custom event whose JSON line is > 64 KiB
   (e.g. directly call `self.server._printer.broadcast({"type": "x", "text": "A" * 100_000, "tabId": <tab>})`) — or just send the real
   `run` command and let the daemon emit `system_prompt` itself.
1. Assert that `client._closed.is_set()` does NOT become True.

I drafted `src/kiss/tests/agents/sorcar/test_cli_daemon_connection_lost_repro.py`
which used `_submit_task` with prompt "hi" but it PASSED because the
test harness daemon does not actually emit the large `system_prompt`
event in this isolated case (the agent task didn't actually start
emitting before the test's short window). Need to write a tighter
reproducer that **forces** a large line: have the test inject a raw
broadcast of an oversize event after `ready`, then assert the
connection survives.

## Fix

In `CliClient._main`, pass `limit=` to `asyncio.open_unix_connection`
with a generous buffer (e.g. 16 MiB) so even huge `system_prompt`
events fit:

```python
self._reader, self._writer = await asyncio.open_unix_connection(
    str(self.sock_path),
    limit=16 * 1024 * 1024,
)
```

Also harden `_main`'s outer except clause to catch
`asyncio.LimitOverrunError` and `Exception` more broadly so a future
oversize line cannot silently disconnect. Specifically:

```python
except (asyncio.CancelledError, ConnectionError, asyncio.LimitOverrunError, asyncio.IncompleteReadError):
    return
```

(But the buffer-size fix is the real fix; the broadened except is
defence in depth.)

## What I've done so far in this continuation (chronological)

1. Read PROGRESS.md (prior continuation).
1. Re-read `cli_client.py` to confirm the disconnect message path.
1. Re-read `web_server.py` `_dispatch_client_command`, `_uds_handler`,
   `_translate_webview_command`, `_handle_command` → confirmed the
   daemon does NOT close the connection on `run`.
1. Read `_cmd_run` in `commands.py` to confirm task launches
   asynchronously in a worker thread.
1. Wrote `tmp/spawn_daemon2.py` (port collision) → realised PID
   34767 is the kiss-web extension daemon and own its 8787 port.
1. Wrote `tmp/spawn_daemon3.py` with explicit `uds_path=` and a
   free port → daemon runs, `hi` completes cleanly via raw UDS.
1. Wrote `tmp/probe_real.py` → connected to the real daemon's
   `~/.kiss/sorcar.sock`, raw UDS, sent `run`/"hi", saw the full
   event stream including a very large `system_prompt` event.
1. Wrote `src/kiss/tests/agents/sorcar/test_cli_daemon_connection_lost_repro.py`
   that uses `_submit_task` with prompt "hi" — but it PASSED, so I
   was missing the trigger. **The asyncio readline limit is the
   actual trigger and the test needs to inject an oversize event.**
1. Identified asyncio default 64 KiB `StreamReader` limit as the
   root cause, because `open_unix_connection` is called without
   `limit=` in `cli_client._main`.

## Plan for next continuation

1. Rewrite the reproducer test to inject a single broadcast > 64 KiB
   into the daemon's printer (broadcast directly via the printer) and
   assert the CliClient connection survives.
1. Apply the `limit=16 * 1024 * 1024` fix in `cli_client._main`.
1. Broaden `_main`'s except to also catch
   `asyncio.LimitOverrunError` and `asyncio.IncompleteReadError`.
1. Verify the new test now PASSES with the fix and FAILS without it.
1. Run `uv run check --full` and impacted tests.
1. gpt-5.5 review.
1. Delete `tmp/spawn_daemon.py`, `spawn_daemon2.py`, `spawn_daemon3.py`,
   `probe_real.py`, `launcher.py` etc.

## Operating constraints

- Working dir: `/Users/ksen/work/kiss/.kiss-worktrees/kiss_wt-1781889615-1a0bce7a`
- Current PID 34767 IS the kiss-web daemon — DO NOT KILL.
- Port 8787 owned by it; spawn fresh daemons on a free port.
- `uv run check --full` must be clean before finishing.
