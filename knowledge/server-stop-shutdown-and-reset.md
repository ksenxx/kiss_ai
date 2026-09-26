---
title: Stopping tasks, SIGTERM graceful shutdown, server reset and self-update barrier
uuid: b69c4ddc-1dc6-4a60-851e-c1329e0a1473
summary: stop command (stop_event, then async KeyboardInterrupt at 1s/5s), run-token
  stops, SIGTERM shutdown order, Server reset, update-when-idle barrier, IP-change
  restart.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Stopping tasks, SIGTERM graceful shutdown, server reset and self-update barrier

## Stopping one task
`stop` goes through `_cmd_stop` to `_stop_task(tab_id, run_token)`:
1. An empty `tab_id` is a no-op. A missing tab id is a frontend bug and must not stop everything.
2. A viewer tab with no `stop_event` of its own resolves through the printer's subscriber map to the tab that owns the running task.
3. When `run_token` (the command's `taskId`) is non-empty, the stop applies only if the owner's `AgentState.client_run_token` matches. `daemon_client.run`'s abort cascade relies on this so a late stop never kills a newer run on a reused `api-…` tab. UI stops send no `taskId`.
4. It sets the cooperative `stop_event`, then starts `_force_stop_thread`. After 1 s, if the thread is alive, `inject_keyboard_interrupt` raises `KeyboardInterrupt` in it via `ctypes.pythonapi.PyThreadState_SetAsyncExc`. It retries once after 5 s. Before every injection the `still_owns` guard is evaluated under `STATE_LOCK`. Once the run finished, or `stop_acknowledged` is raised (the runner caught the interrupt and is cleaning up), nothing is injected. Otherwise a reused `ThreadPoolExecutor` worker running an unrelated sibling sub-agent could be hit.

`interruptTool` interrupts only the current tool call (`ToolCallInterrupted`), not the task.

## SIGTERM / SIGHUP
`_handle_shutdown_signal` logs active tabs (`_snapshot_active_tabs`) and memory. On the first SIGTERM it starts the `_shutdown_on_sigterm` thread, so the event loop keeps flushing notifications and answering pings. Later signals during shutdown are ignored: re-raising inside the `finally` sleeps once escaped cleanup. The shutdown order is:
0. `_detach_tunnel`: cloudflared must survive a supervisor's SIGKILL escalation (see `server-cloudflare-tunnel`).
1. `_stop_active_agent_tasks`: set the stop events, inject interrupts and JOIN the workers within an aggregate timeout, so each run persists "Task stopped by user" and broadcasts a final result. Daemon threads killed at exit would leave the row at "Agent Failed Abruptly".
2. `_await_active_merges`, then `_disconnect_mcp_servers`, still on the SIGTERM thread.
3. `_request_loop_shutdown` (via `call_soon_threadsafe`) unwinds `asyncio.run`, and `start()`'s `finally` also runs cleanup (it repeats task stop, merge wait and MCP disconnect, then flushes the registry and stops the watchdog).
4. Failsafe: counted from the loop-shutdown request, a force-exit after `_SHUTDOWN_EXIT_FAILSAFE = 30.0` s if the loop is still running, so the supervisor can respawn.

Before the loop runs, the handler falls back to raising `KeyboardInterrupt`. Raising it mid-loop is unreliable.

## Server reset (settings panel)
`_handle_server_reset(conn_id)` sends a connId-stamped notification, then SIGTERMs its own process after a short delay. The supervisor (macOS LaunchAgent `KeepAlive`, Linux systemd `Restart=always`) respawns `kiss-web`, which re-adopts the port and the cloudflared tunnel, so the public URL is kept.

## Self-update
`_handle_run_update` runs `~/.kiss/kiss_ai/install.sh` detached (output appended to `~/.kiss/update.log`), or the curl bootstrap when the clone is missing. `_watch_update_exit` reports failures such as "another KISS update is already running (pid N)". For update-when-idle, `_run_update_when_idle` polls `_arm_update_barrier_if_idle`, which checks idleness AND sets `_CommandsMixin._update_installing` in one `STATE_LOCK` critical section. `_cmd_run` holds the same lock while admitting, so a run submitted in between is either counted as active or refused ("An update is being installed"). Previously this was a TOCTOU: a run could start and then be killed by the restart. Running tasks can still be steered.

## IP change
The `_watchdog` tick detects a changed LAN IP set, debounced by `_IP_CHANGE_DEBOUNCE_TICKS` (`_watchdog_check_ip_change`). In direct-LAN mode (`use_tunnel=False`) it closes WSS and shuts down so the supervisor restarts on the new address. In tunnel mode it only logs and re-issues the TLS certificate (`_refresh_tls_cert`).

## Sources
- `src/kiss/server/task_runner.py` (`_stop_task`, `_force_stop_thread`, `inject_keyboard_interrupt`)
- `src/kiss/server/commands.py` (`_cmd_stop`, `_cmd_run`, `_update_installing`)
- `src/kiss/server/web_server.py` (`_handle_shutdown_signal`, `_shutdown_on_sigterm`, `_stop_active_agent_tasks`, `_handle_server_reset`, `_handle_run_update`, `_run_update_when_idle`, `_arm_update_barrier_if_idle`, `_watchdog_check_ip_change`)
