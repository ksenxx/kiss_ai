---
title: 'Remote webapp access: remote_password, localhost-only lockdown and auth rate
  limiting'
uuid: 7f61eec7-75f6-49ab-b04c-05d17912a9a0
summary: 'Browser access to kiss-web: remote_password, 403 localhost-only lockdown
  when empty, auth/auth_ok/auth_required/auth_locked handshake, per-IP lockout keyed
  on Cf-Connecting-Ip.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Remote webapp access: remote_password, localhost-only lockdown and auth rate limiting

## Who needs a password
- **UDS clients** (VS Code extension, Python `daemon_client`): no handshake. The socket is mode `0o600`.
- **WSS clients** (browser webapp, including via Cloudflare tunnel): must complete `ServerApi.authenticate` before any command is dispatched.

The password is `remote_password` in `~/.kiss/config.json` (read with `load_config()`). It is re-read on every attempt, so a change applies immediately.

## Empty password = localhost only
While `remote_password` is empty:
- `RemoteAccessServer._process_request` answers EVERY parsed request from a non-loopback peer with 403: the HTML page, static assets, trajectory endpoints and the `/ws` upgrade. The server binds `0.0.0.0`, so without this gate any LAN machine could log in with the empty password.
- HEAD health checks are answered before the HTTP parser runs, so `_head_health_response` applies the same rule. Loopback HEADs always get 200, which keeps cloudflared's origin health checks passing.
- No tunnel is started. A cloudflared left over from an earlier password-protected run is terminated at startup.
- `authenticate` re-checks this per attempt as defense in depth.

Tunnel traffic arrives from the local cloudflared, which is a loopback peer. That is why the tunnel only runs when a password is set.

## Handshake (`ServerApi.authenticate`)
The webapp's bootstrap shim (`_WS_SHIM_JS` in web_server.py) sends `{"type":"auth","password":...}` as soon as the socket opens, using the password stored in `localStorage`.
1. An IP that is still locked out gets `{"type":"auth_locked","retry_after":N}` and the socket is closed.
2. An empty configured password with a non-loopback peer gets an error and close.
3. Up to two attempts (30 s, then 60 s to receive). The password is compared in constant time. A correct one gets `auth_ok`. The first wrong one gets `auth_required` (the prompt). The second gets `error` and close. A first frame that is not `auth` closes without counting a failure.
4. Only NON-EMPTY wrong guesses count toward the lockout. Every page load probes with the possibly empty stored password, and penalizing that behind a shared tunnel would lock everyone out.

## Rate limiting
The constants are `_AUTH_FAIL_MAX = 5` failures within `_AUTH_FAIL_WINDOW = 60.0` s, which triggers a lockout of `_AUTH_LOCKOUT = 60.0` s. The key is `_client_ip(websocket)`. When the TCP peer is loopback (cloudflared), it trusts `Cf-Connecting-Ip`, falling back to the first hop of `X-Forwarded-For`. Otherwise every tunnel visitor shares the `127.0.0.1` bucket, and one bad actor locks everyone out ("the remote webapp doesn't ask for a password"). The headers are ignored for non-loopback peers, which could spoof them.

## Other HTTP endpoints
- `/ca.crt` downloads the local CA certificate (`kiss-sorcar-local-ca.crt`) for phones (see `server-tls-local-ca`).
- Trajectory viewer data endpoints are serviced by `ServerApi.trajectory_jobs` / `job_trajectories`. `_job_dir_is_contained` keeps them inside the jobs directory.

## Sources
- `src/kiss/server/sorcar.py` (`ServerApi.authenticate`, `_refuse_no_password_remote`, `trajectory_jobs`, `job_trajectories`, `_job_dir_is_contained`)
- `src/kiss/server/web_server.py` (module docstring, `_process_request`, `_head_health_response`, `_client_ip`, `_peer_is_loopback`, `_AUTH_FAIL_MAX`, `_AUTH_FAIL_WINDOW`, `_AUTH_LOCKOUT`, `_WS_SHIM_JS`)
