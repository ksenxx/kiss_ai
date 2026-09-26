---
title: 'Cloudflare tunnel for kiss-web: quick vs named, surviving restarts, health
  watchdog, ntfy URL'
uuid: a502b6d8-e9b0-4017-a599-f28af61a5f95
summary: cloudflared quick or named tunnel (CLOUDFLARE_TUNNEL_TOKEN), cloudflared.pid
  adoption, stderr drain shim, readyConnections health watchdog, remote-url.json,
  ntfy.sh URL.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Cloudflare tunnel for kiss-web: quick vs named, surviving restarts, health watchdog, ntfy URL

## Modes
- **Quick tunnel** (no token): cloudflared gets a random `*.trycloudflare.com` hostname, parsed from its stderr by `_parse_quick_tunnel_url`. A new process means a new URL.
- **Named tunnel** (fixed URL): the token comes from `CLOUDFLARE_TUNNEL_TOKEN` or `tunnel_token` in `~/.kiss/config.json`. The public URL comes from `CLOUDFLARE_TUNNEL_URL` / `tunnel_url` (`_resolve_tunnel_settings`, `_parse_named_tunnel_url`).

`kiss-web`'s `main()` always passes `use_tunnel=True`, but the tunnel starts only when `remote_password` is non-empty (see `server-remote-access-auth`). With an empty password, a surviving cloudflared from an earlier run is killed (`_terminate_orphan_cloudflared`).

## Keeping the URL across daemon restarts
A daemon restart must not rotate the public URL:
- `_spawn_cloudflared` starts cloudflared with `start_new_session=True` and persists its pid and metrics port to `~/.kiss/cloudflared.pid`.
- On shutdown, `_detach_tunnel` (not `_stop_tunnel`) leaves it running. It is the FIRST shutdown step, both in `_shutdown_on_sigterm` and in `start()`'s `finally`.
- cloudflared's stderr is a pipe held by kiss-web. When kiss-web exits, cloudflared's next stderr write gets EPIPE, and Go turns that into a fatal SIGPIPE. `_detach_tunnel` therefore hands the pipe to a detached drain process (`_spawn_stderr_drain_shim`) before anything slow runs, so a supervisor SIGKILL cannot kill the tunnel.
- The next kiss-web re-adopts the process via `_try_adopt_existing_cloudflared`.
- `_stop_tunnel` kills cloudflared. It is used when the tunnel is unhealthy and must be replaced.

## Health watchdog
`_watchdog` runs every `TUNNEL_CHECK_INTERVAL = 15` s. `_check_and_restart_tunnel` detects:
1. **cloudflared died** (`poll()`), for example killed during macOS sleep.
2. **Alive but deregistered**: the hostname is NXDOMAIN while cloudflared keeps retrying. The watchdog polls the metrics `/ready` endpoint for `readyConnections > 0`. It terminates after `_TUNNEL_UNHEALTHY_LIMIT_NAMED = 3` (named) or `_TUNNEL_UNHEALTHY_LIMIT_QUICK = 40` (quick) consecutive zero readings. The metrics check is skipped during the first `_TUNNEL_STARTUP_GRACE = 120` s.

Failed restarts back off exponentially (`_tunnel_next_retry`) so Cloudflare rate limits are not hammered. Tunnel state (`_tunnel_proc`, `_tunnel_started_at`) is published only under `_tunnel_lock` after checking `_tunnel_stopped`.

The same tick re-writes `remote-url.json` if something deleted it. The VS Code settings panel polls that file every 10 s.

## Publishing the URL
- **`~/.kiss/remote-url.json`** (`_save_url_file`, `_write_url_file_sync`): holds the local and tunnel URLs. `kiss-web --url` prints it (`_print_url`).
- A `remote_url` global broadcast updates connected clients.
- **ntfy.sh**: `_post_url_to_message_board` posts the URL to `https://ntfy.sh/<topic>`. `_get_machine_topic` derives the topic from SHA-256(hostname + MAC), mixing in the KISS home path when it is not `~/.kiss` so test daemons never pollute the production topic. The topic is persisted in `~/.kiss/ntfy_topic`. A post is skipped when the last cached message is the same URL and younger than `_NTFY_REPOST_MAX_AGE = 3600` s. `rsorcar` drops a copied `ntfy_topic` so two machines never share one topic.

## Sources
- `src/kiss/server/web_server.py` (`_resolve_tunnel_settings`, `_start_tunnel`, `_spawn_cloudflared`, `_detach_tunnel`, `_spawn_stderr_drain_shim`, `_stop_tunnel`, `_try_adopt_existing_cloudflared`, `_terminate_orphan_cloudflared`, `_check_and_restart_tunnel`, `_watchdog`, `_save_url_file`, `_get_machine_topic`, `_post_url_to_message_board`)
- `rsorcar` (step 7)
