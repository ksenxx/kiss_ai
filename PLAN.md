# Reduce Concurrency in sorcar.py

## Goal

Reduce the number of threads, timers, and other concurrency primitives in
[`src/kiss/agents/sorcar/sorcar.py`](src/kiss/agents/sorcar/sorcar.py) without changing user-visible behavior.
After each change, run `uv run pytest -v` with 900-second timeout to ensure
tests pass. Run `uv run check --full` to ensure lint and type checks pass.

## Current Concurrency Inventory

[`sorcar.py`](src/kiss/agents/sorcar/sorcar.py) creates the following concurrent entities:

### Daemon Threads (3 total at runtime)

| # | Target function | Purpose | Line |
|---|------------------------|----------------------------------------------------------|-------|
| 1 | `_watch_code_server` | Polling loop (5s): restart code-server if it crashes | ~259 |
| 2 | `_watch_periodic` | Polling loop (1s): detect VS Code theme changes + (every 5s) schedule shutdown when no clients | ~443 |
| 3 | `run_agent_thread` | Per-task: runs the agent (created in `run_task` and `run_selection`) | ~828, ~852 |

### `asyncio.to_thread` (offloads blocking work from the event loop — 2 call sites)

| # | Endpoint | Purpose | Line |
|---|----------------------------|---------------------------------|-------|
| 1 | `/complete` | LLM autocomplete generation | ~1070 |
| 2 | `_thread_json_response` | Used by `/commit` and `/generate-commit-message` for blocking git/LLM calls | ~1202 |

### asyncio Tasks and Timers

| # | Function | Purpose | Line |
|---|-------------------------------|----------------------------------------------|-------|
| 1 | `_open_browser_async` | One-shot `asyncio.create_task`: sleep 2s then open browser | ~1351 |
| 2 | `_schedule_shutdown_on_loop` | `loop.call_later(1.0, _do_shutdown)` stored as `asyncio.TimerHandle` | ~716 |
| 3 | `loop.call_soon_threadsafe` | Cross-thread dispatch to schedule shutdown from non-async threads | ~743 |

### Locks and Events (6 primitives)

| # | Name | Type | Purpose |
|---|------------------------|---------------------|-----------------------------------------|
| 1 | `running_lock` | `threading.Lock` | Protects `running`, `agent_thread`, `merging` |
| 2 | `shutting_down` | `threading.Event` | Signals all daemon threads to exit |
| 3 | `current_stop_event` | `threading.Event` | Per-task stop signal for agent thread |
| 4 | `user_action_event` | `threading.Event` | Blocks agent thread waiting for user browser action |
| 5 | `user_question_event` | `threading.Event` | Blocks agent thread waiting for user question answer |
| 6 | `printer._thread_local` | `threading.local` | Thread-local stop event set on agent thread |

### Subprocesses

| # | Type | Purpose |
|---|--------------------------------------|----------------------------------------------|
| 1 | `subprocess.Popen` (long-lived) | code-server child process |
| 2 | `subprocess.Popen` (one-shot) | macOS `open` fallback for browser |
| 3 | Various `subprocess.run` (blocking) | git add/diff/commit, lsof for port kill |

### Thread-kill Mechanism

| # | Mechanism | Purpose |
|---|-------------------------------------------|-------------------------------------|
| 1 | `ctypes.pythonapi.PyThreadState_SetAsyncExc` | Injects `_StopRequested` into agent thread as fallback stop |

## Change to Make

### Change 1: Merge `_watch_code_server` into `_watch_periodic`

**Why**: Both `_watch_code_server` and `_watch_periodic` are daemon threads that
poll in a loop using `shutting_down.wait()` as their sleep mechanism.
`_watch_code_server` checks every 5s if the code-server subprocess crashed and
restarts it. `_watch_periodic` already runs every 1s and performs a 5-tick
counter for its client-count check. The code-server health check can use the
same 5-tick counter pattern. Merging them eliminates one daemon thread.

**How**:

1. Move the body of `_watch_code_server` (the `cs_proc.poll()` check and
   restart logic) into `_watch_periodic`, guarded by the same `tick >= 5`
   condition that already gates the client-count check.
1. Remove the `_watch_code_server` function entirely.
1. Remove the `threading.Thread(target=_watch_code_server, ...).start()` call
   at line ~404.
1. Pass `cs_proc`/`code_server_url` access into `_watch_periodic` via the
   existing closure (both functions already close over the same `nonlocal`
   variables in `run_chatbot`).

**Detailed steps**:

- In `_watch_periodic`, after the existing client-count check block (inside
  `if tick >= 5:`), add the code-server health check:
  ```python
  # Code-server health check (every 5s, same as client check)
  if cs_binary and cs_proc is not None:
      ret = cs_proc.poll()
      if ret is not None:
          # ... restart logic from _watch_code_server ...
  ```
- The restart logic sets `nonlocal cs_proc, code_server_url` — both are already
  accessible from `_watch_periodic`'s closure scope. Add them to the
  `nonlocal` declaration.
- Remove the standalone `_watch_code_server` function and its thread launch.
- The conditional `if cs_binary and code_server_url:` guard before launching
  `_watch_code_server` should be moved into the merged check inside
  `_watch_periodic`.

**Risk**: Low. The check frequency stays at ~5s (same as before). The restart
logic is identical. The only difference is that it shares a thread with theme
and client-count checks, which are non-blocking filesystem operations.

## What NOT to Change

These concurrency mechanisms are genuinely needed and should NOT be removed:

- **`run_agent_thread`**: The agent runs blocking LLM calls for potentially
  minutes. It must be on its own thread to keep the Starlette event loop responsive.
- **`asyncio.to_thread` calls**: These offload blocking git/LLM operations from
  the async event loop. Removing them would block the web server.
- **`running_lock`**: Protects shared mutable state accessed from both the
  async event loop and the agent thread. Essential for correctness.
- **`shutting_down` event**: Needed to cleanly signal daemon threads to exit.
- **`current_stop_event` / `user_action_event` / `user_question_event`**: These
  are per-task coordination primitives between the web server and agent thread.
- **`asyncio.create_task(_open_browser_async())`**: Already the minimal async
  approach (no thread).
- **`loop.call_later` / `call_soon_threadsafe`**: Already the minimal async
  approach for shutdown scheduling (no thread or timer).
- **`subprocess.Popen` for code-server**: The editor process must run as a
  subprocess.

## After the Change

1. Run `uv run pytest -v` (timeout 900s) to verify no test breakage.
1. Run `uv run check --full` to verify lint/type checks.
1. Verify the daemon thread count decreased from 3 to 2 (at runtime with
   code-server enabled).

## Expected Result

| Metric | Before | After |
|-----------------------|--------|-------|
| Daemon threads | 3 (2 watchers + per-task) | 2 (`_watch_periodic` + `run_agent_thread`) |
| Functions | `_watch_code_server` + `_watch_periodic` | `_watch_periodic` (merged) |
| Thread launches | 3 (2 daemon watchers + per-task) | 2 (1 daemon watcher + per-task) |
| Everything else | unchanged | unchanged |
