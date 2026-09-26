---
title: Muse-auth covered services and the muse_auth CLI (status, enroll, import, grant,
  revoke, export, clear, audit)
uuid: 3d643d7d-293a-40e6-b3f1-13e5668d9a22
summary: The 24 Muse-auth covered services and their host allowlists, migrated legacy
  config keys, and the muse_auth CLI verbs (enroll, import, grant scopes, revoke,
  export, audit).
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Muse-auth covered services and CLI

## Covered services (`muse_auth/_common.SERVICE_HOSTS`)
| service | allowed hosts |
| --- | --- |
| `gmail` | gmail.googleapis.com, www.googleapis.com |
| `google_drive`, `google_calendar` | www.googleapis.com |
| `google_docs` / `google_sheets` | docs. / sheets.googleapis.com + www.googleapis.com |
| `googlechat` | chat.googleapis.com (service-account mode is not covered) |
| `notion` | api.notion.com |
| `github` | api.github.com |
| `slack` (and `slack-<ws>-<hash>`) | slack.com, files.slack.com |
| `brave_search` | api.search.brave.com |
| `discord` | discord.com |
| `govee` | openapi.api.govee.com |
| `twitch` | api.twitch.tv |
| `zalo` | openapi.zalo.me |
| `line` | api.line.me |
| `msteams` | graph.microsoft.com |
| `telegram` | api.telegram.org |
| `homeassistant`, `ntfy`, `firecrawl`, `mattermost`, `nextcloud`, `bluebubbles`, `synology` | none built in: bound to the one origin enrolled with the credential (self-hosted or cloud) |

Channels not listed (email, IRC, Signal, WhatsApp, Matrix, SMS, Postgres, ...) keep the legacy
direct-credential path. `TOKEN_ENDPOINT_HOSTS` pins where secrets may be exchanged: msteams →
login.microsoftonline.com, github → github.com, twitch → id.twitch.tv.

Per-workspace names: `service_root` maps `slack-*`, `telegram-*`, `msteams-*` to their root for
host lookup (`builtin_hosts`). Candidate-validation scratch names
`<telegram|msteams>-pending-<16 hex>` follow the live root's policy (`policy_service`); other
suffixed names (per-workspace Slack) have their own independent policy. Service names must
match `[a-z][a-z0-9_-]{0,63}` (`valid_service_name`, `fullmatch` so `"gmail\n"` fails) because
they become vault file names.

## Legacy key migration (`__main__._TOKEN_SERVICES`)
config key per service: notion/github/homeassistant/ntfy/mattermost `token`; firecrawl
`api_key`; brave_search `api_key` sent as `X-Subscription-Token`; discord `bot_token` as
`Authorization: Bot <token>`; twitch `access_token` (also scrubs `client_secret`); zalo
`access_token` header; line `channel_access_token`; bluebubbles `password` as a query param;
telegram `bot_token` as a path-kind credential (`/bot<token>/`). Self-hosted services
(Firecrawl, Home Assistant, ntfy) enroll their configured base host with the credential;
plain-`http://` bases are enrolled as consent-scoped insecure hosts.

## CLI: `python -m kiss.agents.third_party_agents.muse_auth <verb>`
- `status` — enrolled services and state locations.
- `enroll <service>` — Google OAuth consent straight into the vault; only the six Google
  services (`_SERVICE_MODULES`: gmail, google_drive, google_calendar, google_docs,
  google_sheets, googlechat). Other services enroll by auto-migration or Connect-style sign-in.
- `import <service>` — migrate a legacy token file into the vault and delete the plaintext.
- `grant <service> read|write [--scope once|session|perpetual|ttl] [--ttl S]` — default scope
  `once` (single use). Session grants die with the daemon.
- `revoke <service> [action]` — remove grants (all actions when omitted).
- `export <service>` — print a vault credential, to restore a legacy config after opting out
  with `KISS_MUSE_AUTH=0`.
- `clear <service>` — delete the credential and its surrogates.
- `audit [--tail N]` — recent Sentinel decisions from `audit.jsonl`.
- `daemon` (foreground) / `stop`.

When a write is asked, the connector surfaces `client.approval_hint(...)` telling the user
which `grant` command approves it. Grants are meant to be issued by the human at a terminal.

## Sources
- `src/kiss/agents/third_party_agents/muse_auth/_common.py` (`SERVICE_HOSTS`, `TOKEN_ENDPOINT_HOSTS`, `service_root`, `scratch_root`, `policy_service`, `builtin_hosts`, `valid_service_name`)
- `src/kiss/agents/third_party_agents/muse_auth/__main__.py` (`_SERVICE_MODULES`, `_TOKEN_SERVICES`, `_import_hosts`, argument parser)
- `src/kiss/agents/third_party_agents/muse_auth/client.py` (`grant`, `revoke`, `approval_hint`)
- `src/kiss/agents/third_party_agents/README.md` ("Credential isolation (Muse auth)")
