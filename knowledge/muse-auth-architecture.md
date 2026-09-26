---
title: 'Muse-auth credential isolation: vault daemon, surrogate tokens, Sentinel policy'
uuid: c00c1b36-a886-4c95-80b8-c2ac13a75cc4
summary: 'Muse-auth credential isolation: vault daemon on a Unix socket, muse-sgt
  surrogate tokens swapped at the boundary, Sentinel read/write policy, grants, audit,
  KISS_MUSE_AUTH.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Muse-auth credential isolation

`src/kiss/agents/third_party_agents/muse_auth/` ports the auth design Meta published for its
Muse agent to the KISS connectors. Goal: real tokens never live in the agent process, and
every outbound connector request passes a deterministic egress policy.

## Components (all state under `$KISS_HOME/muse_auth/`, dir 0700)
- **Daemon** (`daemon.py`, `MuseAuthDaemon`): one process hosting vault + Sentinel + network
  boundary. Listens on `authd.sock` (`_common.socket_path`; falls back to a hashed `/tmp` path
  when the natural path exceeds 90 bytes). One JSON frame in, one out per connection
  (`send_frame`/`recv_frame`, `MAX_FRAME_BYTES` = 64 MiB, so responses cap near 48 MiB). Peers
  are authenticated by `SO_PEERCRED` (UID must match). Ops: `status`, `store_credentials`,
  `clear_credentials`, `mint_surrogate`, `http_request`, `grant`, `revoke`, `stop`.
- **Vault** (`vault.py`): `vault/<service>.json` (0600). Credential kinds: Google authorized-user
  info (refreshed by the daemon), `bearer`, `header` (custom header, e.g. Brave, or a prefix such
  as Discord's `Bot `), `query` (BlueBubbles `password=`), `path` (token inside the URL path,
  e.g. Telegram), `oauth2_client_credentials` (daemon does the token exchange, e.g. Teams
  app-only), `oauth2_refresh_token` (device-grant sign-ins: GitHub, Twitch, Teams). Token
  endpoints are pinned per service in `TOKEN_ENDPOINT_HOSTS` so a secret can only be POSTed to
  the real IdP (loopback allowed for tests).
- **Surrogates**: agents hold opaque `muse-sgt.<service>.<hex>` tokens
  (`SurrogateCredentials`, `bearer_surrogate`). A surrogate minted for one service cannot
  fetch another's credential or reach its hosts.
- **Boundary**: `MuseBoundarySession` (requests-compatible), `MuseHttp` (httplib2 for
  `googleapiclient`), `slack_transport.MuseWebClient` (slack_sdk). They ship each prepared
  request to the daemon (`http_request`), which authorizes it, swaps in the real credential,
  performs the HTTPS call, and scrubs credential spellings from responses.
- **Sentinel** (`sentinel.py`): classifies each request `read`/`write` (`_common.action_class`:
  GET/HEAD/OPTIONS are reads; `request_action` refines per service, e.g. Slack read API
  methods, Firecrawl `/v2/scrape|map|search`, Govee `/device/state`, Telegram read methods),
  checks the host allowlist (`SERVICE_HOSTS` + policy `extra_hosts`; empty tuple = bound to the
  single origin enrolled with the credential), then the policy
  (`policy.json`, default `{"read": "allow", "write": "ask"}`) and grants (`grants.json`;
  session grants in memory only). Files are re-read on every decision. Every decision is
  appended to `audit.jsonl` (decisions, not whether the call succeeded). Bodyless GET/HEAD
  redirect hops are followed off-allowlist after stripping credentials.

## Enablement
`muse_auth_enabled()`: `KISS_MUSE_AUTH` in (0,false,no,off) → off; (1,true,yes,on) → on;
unset/other → `platform_supports_muse_daemon()` = `hasattr(socket, "SO_PEERCRED")`, i.e. on by
default on Linux (commit `01c1dec92`), off on macOS/Windows. A typo keeps the secure default.
Opt out via `KISS_MUSE_AUTH=0` in `$KISS_HOME/api_keys.env`.

## Daemon lifecycle
`client.ensure_daemon()` spawns `python -m kiss.agents.third_party_agents.muse_auth.daemon`
detached (log `daemon.log`), waiting up to 15 s. The check-stop-spawn sequence runs under
`spawn.lock` so concurrent callers start one daemon (fix `fc9417631`); the daemon itself takes
`daemon.lock` around bind. A running daemon with an older `PROTOCOL_VERSION` is stopped and
replaced.

## Migration
Covered connectors migrate legacy plaintext credentials on first use: e.g.
`mint_surrogate_migrating` enrolls a Google `token.json` and deletes it; token channels store
the key then `ChannelConfig.scrub_secrets` removes it from `config.json`. This is a one-time
hand-off through the agent process.

## Honest scope
Both processes run as the same OS user: this is a process boundary, not an OS security
domain. An agent with arbitrary shell could read the vault or run the `grant` CLI; restrict
access to `$KISS_HOME/muse_auth` and the CLI in tool permissions for that threat.

## Sources
- `src/kiss/agents/third_party_agents/muse_auth/__init__.py` (module docstring)
- `src/kiss/agents/third_party_agents/muse_auth/_common.py` (`SERVICE_HOSTS`, `TOKEN_ENDPOINT_HOSTS`, `muse_auth_enabled`, `platform_supports_muse_daemon`, `socket_path`, `action_class`, `request_action`, `MAX_FRAME_BYTES`, `PROTOCOL_VERSION`)
- `src/kiss/agents/third_party_agents/muse_auth/daemon.py` (`MuseAuthDaemon`, `_peer_uid`)
- `src/kiss/agents/third_party_agents/muse_auth/vault.py`, `sentinel.py` (`_DEFAULT_POLICY`)
- `src/kiss/agents/third_party_agents/muse_auth/client.py` (`ensure_daemon`, `mint_surrogate_migrating`, `MuseBoundarySession`, `MuseHttp`)
