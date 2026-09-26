---
title: 'Daemon concurrency invariants: lock order, bounded waits, thread-safe broadcast'
uuid: 87472960-fdb9-47e8-b98b-81314fc27cae
summary: kiss-web lock order (STATE_LOCK before printer locks, _ws_lock leaf), non-blocking
  broadcast, bounded flocks, task_thread admission gate, force-stop ownership guards.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Daemon concurrency invariants: lock order, bounded waits, thread-safe broadcast

The daemon mixes one asyncio loop (transport) with many worker threads (one per task, plus executor threads for merges and file IO). These rules come from the code comments and fixed race bugs. Keep them when editing `src/kiss/server/`.

## Lock order (never reverse)
- `agent_state.STATE_LOCK` (an `RLock`; the same object as `VSCodeServer._state_lock`) is taken FIRST.
- Printer locks come after it: `STATE_LOCK` -> `JsonPrinter._model_pick_lock` -> `delivery_lock` -> `_lock` (documented in `JsonPrinter.__init__`). Code running under `_lock` / `_bash_lock` must not take `STATE_LOCK`. It reads the `agent_states` dict directly (a lock-free point read) instead of calling `agent_state.get`.
- `STATE_LOCK` -> tab-registry leaf lock (see `VSCodeServer` comments).
- `WebPrinter._ws_lock` (endpoint sets) is a leaf.

## Broadcasting is non-blocking
`printer.broadcast(...)` may be called from any thread, including under `STATE_LOCK`. `WebPrinter._schedule_send` hands the payload to the loop with `run_coroutine_threadsafe` and never waits. Never add a blocking wait on the loop inside a broadcast path. Per-endpoint order is kept by `FifoSendLock` (see `server-broadcast-routing`).

## Deliberate disk IO under `_state_lock`
The last-model read/persist of `~/.kiss/config.json` runs inside `_state_lock` (`_refresh_default_model`), so a stale on-disk value cannot overwrite a model the user just picked. This is intentional.

## Admission and ownership gates
- **Gate runs on `task_thread`, not `is_task_active`.** The flag rises only after the thread starts (S3-05).
- `_run_task`'s whole body sits in try/finally, so an injected `KeyboardInterrupt` can never leave `task_thread` set (C-RC2).
- Force-stop injections check `still_owns` under `STATE_LOCK` right before `PyThreadState_SetAsyncExc`, and stop once `stop_acknowledged` is set.
- `daemon_client` stops carry a run token (`client_run_token`) so a stale stop cannot hit a newer run on the same tab.
- The self-update barrier: the idle check and `_update_installing` are set in one `STATE_LOCK` section, the same lock `_cmd_run` admits under.
- Merge claims record `merge_thread`. A claim whose thread died is treated as leaked (`AgentState.merge_in_progress`).
- `TabRegistry.open_tab` decides exists/full atomically (D-RC2). Teardowns use generation tokens (`close_tab_if_generation`).
- Local talk visibility is computed at decision time from canonical facts (`_local_tab_shown`), never from mirrored per-connection sets.

## Bounded waits
- Cross-process flocks on startup (`sorcar.sock.lock`, `tls/.tls.lock`) use `_flock_with_deadline`. A blocking `LOCK_EX` in an executor thread cannot be cancelled.
- Tunnel state is published only under `_tunnel_lock` after checking `_tunnel_stopped`.
- `_stop_active_agent_tasks` bounds the aggregate join time. `_SHUTDOWN_EXIT_FAILSAFE` force-exits a wedged shutdown.
- UDS writes are bounded by `_uds_drain_timeout`.
- `queue.get()` without a timeout is used only on daemon threads (talk player, autocomplete, voice worker) or with a watcher that injects a sentinel (`_await_user_response` also watches `stop_event` and tool interrupts).

## Testing races
To confirm a suspected race, temporarily add a small random sleep before the racing statements. Tests live under `src/kiss/tests/server/`. Give each server its own `KISS_HOME` or explicit `uds_path` / `url_file` (see `server-daemon-startup-and-uds`).

## Sources
- `src/kiss/server/agent_state.py` (`STATE_LOCK`, `AgentState.merge_in_progress`)
- `src/kiss/server/json_printer.py` (`JsonPrinter.__init__` lock comments)
- `src/kiss/server/server.py` (`_refresh_default_model`, `_local_tab_shown`)
- `src/kiss/server/web_server.py` (`FifoSendLock`, `_schedule_send`, `_flock_with_deadline`, `_arm_update_barrier_if_idle`)
- `src/kiss/server/task_runner.py` (`_run_task`, `_force_stop_thread`)
- `src/kiss/server/tab_registry.py` (`open_tab`, `close_tab_if_generation`)
