# Reduce Concurrency in sorcar.py

## Goal

Reduce the number of threads, timers, and other concurrency primitives in
`src/kiss/agents/sorcar/sorcar.py` without changing user-visible behavior.
After each change, run `uv run pytest -v` with 500-second timeout to ensure
tests pass. Run `uv run check --full` to ensure lint and type checks pass.

## Current Concurrency Inventory

`sorcar.py` creates the following concurrent entities (line numbers approximate):

### Daemon Threads (8 total)

| # | Target function         | Purpose                                                | Line  |
|---|-------------------------|--------------------------------------------------------|-------|
| 1 | `_cleanup_stale_cs_dirs`| One-shot: removes stale code-server data dirs at start | ~173  |
| 2 | `_watch_code_server`    | Polling loop (5s): restart code-server if it crashes   | ~383  |
| 3 | `_watch_theme_file`     | Polling loop (1s): detect VS Code theme changes        | ~481  |
| 4 | `_watch_no_clients`     | Polling loop (5s): schedule shutdown when no clients    | ~498  |
| 5 | `run_agent_thread`      | Per-task: runs the agent (created in `run_task`/`run_selection`) | ~753/~789 |
| 6 | `generate_followup`     | One-shot per task: calls LLM for a followup suggestion | ~579  |
| 7 | `refresh_proposed_tasks`| One-shot at startup: calls LLM for proposed tasks      | ~1243 |
| 8 | `_open_browser`         | One-shot: sleeps 2s then opens browser                 | ~1278 |

### Timer

| # | Function            | Purpose                                          | Line  |
|---|---------------------|--------------------------------------------------|-------|
| 1 | `_schedule_shutdown`| `threading.Timer(10s)` → `_do_shutdown` after no clients | ~692 |

### Locks and Events (5)

| # | Name                | Type              | Protects                           |
|---|---------------------|-------------------|------------------------------------|
| 1 | `running_lock`      | `threading.Lock`  | `running`, `agent_thread`, `merging` |
| 2 | `proposed_lock`     | `threading.Lock`  | `proposed_tasks`                    |
| 3 | `shutdown_lock`     | `threading.Lock`  | `shutdown_timer`                    |
| 4 | `shutting_down`     | `threading.Event` | Signals all watcher threads to exit |
| 5 | `current_stop_event`| `threading.Event` | Per-task stop signal                |

### asyncio.to_thread (3 call sites)

| # | Endpoint                   | Purpose                     | Line  |
|---|----------------------------|-----------------------------|-------|
| 1 | `/complete`                | LLM autocomplete generation | ~944  |
| 2 | `_thread_json_response`    | Used by commit, push, generate-commit-message, generate-config-message | ~1029 |

### Subprocesses

| # | Type             | Purpose                       |
|---|------------------|-------------------------------|
| 1 | `subprocess.Popen` (long-lived) | code-server process |
| 2 | `subprocess.Popen` (one-shot)   | macOS `open` fallback for browser |
| 3 | Various `subprocess.run`        | git operations (add, diff, commit, push) |

## Changes to Make (in order)

### Change 1: Make `_cleanup_stale_cs_dirs` synchronous at startup

**Why**: This is a one-shot filesystem cleanup that runs once. There's no reason
for it to be in a background thread — it just deletes old directories and returns.
Making it synchronous eliminates one thread and `_cleanup_stale_cs_dirs` completes
before the server starts, which is actually safer.

**How**: Replace `threading.Thread(target=_cleanup_stale_cs_dirs, daemon=True).start()`
with a direct call `_cleanup_stale_cs_dirs()`.

### Change 2: Merge `_watch_theme_file` and `_watch_no_clients` into one watcher thread

**Why**: Both are simple polling loops that sleep and check a condition. Running
two daemon threads for this is unnecessary. They can share a single thread that
wakes up every 1 second (the faster of the two intervals) and performs both checks.

**How**:
- Create a single `_watch_periodic()` function that combines the logic of
  `_watch_theme_file` (check theme file mtime every 1s) and `_watch_no_clients`
  (check client count every 5s, using a counter to skip 4 out of 5 iterations).
- Replace the two `threading.Thread` launches with one.
- Remove `_watch_theme_file` and `_watch_no_clients` as separate functions.

### Change 3: Inline `generate_followup` into `run_agent_thread` instead of spawning a thread

**Why**: `generate_followup` is called at the end of `run_agent_thread`, which is
already a background thread. Spawning yet another thread for a fire-and-forget LLM
call adds unnecessary concurrency. Since `run_agent_thread` has already set
`running = False` and broadcast `task_done` before calling `generate_followup`,
there is no user-visible delay — the followup suggestion just arrives slightly
later (which it already does today, since the thread was spawned at this point anyway).

**How**: Replace `threading.Thread(target=generate_followup, ...).start()` with a
direct call `generate_followup(task, result_text)` inside the `finally` block of
`run_agent_thread`, wrapped in a try/except. The call already happens after
`running = False` is set, so it won't block new task submission.

### Change 4: Make initial `refresh_proposed_tasks` synchronous or deferred to the agent thread

**Why**: At startup, a daemon thread is spawned just to call `refresh_proposed_tasks`
once. This LLM call could instead be done lazily on the first request to
`/proposed_tasks`, or simply run synchronously during startup (it only takes a
few seconds and happens before the user typically interacts).

**How**: Replace `threading.Thread(target=refresh_proposed_tasks, daemon=True).start()`
with a synchronous call `refresh_proposed_tasks()`, moved to just before
`server.run()`. Alternatively, if startup latency is a concern, make the
`/proposed_tasks` endpoint trigger the refresh on first call (lazy init with a
flag variable).  The simpler approach is synchronous because the browser is opened
in a background thread with a 2-second delay, so the user won't notice.

### Change 5: Replace `_open_browser` thread with `asyncio` startup event

**Why**: The `_open_browser` thread exists solely to sleep 2 seconds then call
`webbrowser.open()`. This can be done with a Starlette `on_startup` or
`lifespan` event that uses `asyncio.get_event_loop().call_later()` or
`asyncio.create_task` with `asyncio.sleep(2)`.

**How**: Add a startup event or lifespan to the Starlette app that schedules
browser opening after 2 seconds using asyncio. Remove the `_open_browser` thread.

### Change 6: Eliminate `shutdown_lock` and `shutdown_timer` by using `asyncio.get_event_loop().call_later()`

**Why**: The `threading.Timer` + `shutdown_lock` pattern adds complexity. Since
the shutdown logic only sets `server.should_exit = True`, it can be done via the
asyncio event loop's `call_later`. This removes `shutdown_timer`, `shutdown_lock`,
and the `threading.Timer`.

**How**: Replace `_schedule_shutdown` to use `loop.call_later(10.0, _do_shutdown)`
and store the handle for cancellation. Replace `_cancel_shutdown` to call
`handle.cancel()`. This removes one lock and the Timer.

## After Each Change

1. Run `uv run pytest -v` (timeout 900s) to verify no test breakage.
2. Run `uv run check --full` to verify lint/type checks.
3. Verify the overall thread/timer/lock count decreased by the expected amount.

## Expected Result

| Metric                | Before | After |
|-----------------------|--------|-------|
| Daemon threads        | 8      | 5 (agent thread + code-server watcher + combined periodic watcher + code-server process watcher remains as-is, _open_browser moved to asyncio) |
| `threading.Timer`     | 1      | 0     |
| `threading.Lock`      | 3      | 2 (remove `shutdown_lock`)  |
| Total thread launches | ~10    | ~6    |

The `run_agent_thread` (per-task), `_watch_code_server`, and `asyncio.to_thread`
calls are genuinely needed and should NOT be removed. The code-server `Popen`
subprocess is also necessary. Focus only on the changes above.
