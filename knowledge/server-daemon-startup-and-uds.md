---
title: kiss-web daemon startup, port 8787 and the sorcar.sock Unix socket
uuid: 4fedc03c-98ce-4f3e-8cc1-a407957c7197
summary: How kiss-web starts (main, RemoteAccessServer.start), binds $KISS_HOME/sorcar.sock
  with a flock sidecar and liveness probe, serves HTTPS/WSS on port 8787, and is supervised
  by systemd/launchd.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# kiss-web daemon startup, port 8787 and the sorcar.sock Unix socket

## Entry point
`pyproject.toml` maps `kiss-web = kiss.server.web_server:main`. `main()` accepts:
- `--url`: print the active remote URL from `~/.kiss/remote-url.json` and exit
- `--trust-ca`: install the local CA into browser trust stores and exit (see `server-tls-local-ca`)
- `--workdir`: default working directory

Otherwise it resolves tunnel settings (`_resolve_tunnel_settings`: `CLOUDFLARE_TUNNEL_TOKEN` / `CLOUDFLARE_TUNNEL_URL` env, else `tunnel_token` / `tunnel_url` in `~/.kiss/config.json`) and builds `RemoteAccessServer(use_tunnel=True, ...)`. It then calls `_vscode_server.prewarm_worktree_pool()` and `start()`. The default port is `8787` (constructor argument). HTTPS pages and the WSS endpoint `/ws` share that port.

## start()
`RemoteAccessServer.start()` blocks on the main thread. In order it:
1. Raises the open-file limit and configures logging.
2. Arms the stall watchdog (`start_stall_watchdog`, see `server-stall-watchdog`).
3. Installs signal handlers.
4. Runs `asyncio.run(self._serve_async())`. `_serve_async` also starts the cron scheduler thread.

Its `finally` block runs, in order: `_detach_tunnel` (first, so cloudflared survives), `_stop_active_agent_tasks`, `_await_active_merges`, `_disconnect_mcp_servers`, `tab_registry.flush()`, SEA registry-watcher unsubscribe, and `stall_watchdog.stop()`. `start_async()` is the non-blocking variant used by embedders and tests.

## UDS binding (`_setup_server` -> `_bind_uds`)
- Path: `_default_uds_path()` = `$KISS_HOME/sorcar.sock` (overridable via the `uds_path=` constructor argument for tests).
- The probe -> unlink -> bind sequence is serialized across processes with an exclusive flock on a sidecar file `sorcar.sock.lock`. The flock is acquired through `_flock_with_deadline` in the executor, so a wedged sibling cannot hang startup forever (a blocking `LOCK_EX` in a thread cannot be cancelled).
- A live peer on the socket (`_uds_socket_is_live` probe) means another daemon owns it. The new process waits up to `uds_owner_wait_s` (`_wait_for_uds_release`) and refuses to steal a live socket.
- The socket is created with `cleanup_socket=False`, because asyncio's own close-time unlink is not flock-guarded and could delete a successor's socket. Cleanup goes through `_unlink_own_uds_socket`.
- It is chmod `0o600`. POSIX permissions are the only auth for UDS clients: no password handshake.
- If the bind fails, the daemon serves WSS only and local clients fall back to it. Platforms without `AF_UNIX` skip the UDS (`_unix_sockets_supported`).

After the UDS, `_setup_server_after_uds` binds WSS (with retries), starts the tunnel only when `remote_password` is set, and writes `remote-url.json`.

## Supervision
`_handle_server_reset` depends on a supervisor restarting the process after SIGTERM: a macOS LaunchAgent (`KeepAlive`) or a Linux systemd user unit (`Restart=always`). `rsorcar` installs the systemd user service with lingering. The stall watchdog writes to the systemd-appended `kiss-web-stderr.log`.

## Gotchas
- Tests that drive the blocking `start()` on a thread leak the UDS listener: `start()`'s `finally` never closes `_uds_server`/`_ws_server`, because production relies on process exit to free them. A later server then waits `uds_owner_wait_s` for the "live" socket to go away. Isolate tests with `uds_path=`, `url_file=` and a short `uds_owner_wait_s`.
- Only one daemon per `KISS_HOME`. Tests must set their own `KISS_HOME` or pass explicit paths.

## Sources
- `src/kiss/server/web_server.py` (`main`, `_resolve_tunnel_settings`, `RemoteAccessServer.start`, `_setup_server`, `_bind_uds`, `_uds_socket_is_live`, `_wait_for_uds_release`, `_default_uds_path`, `_flock_with_deadline`)
- `pyproject.toml` (`[project.scripts]`)
