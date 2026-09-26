---
title: 'Catalog of channel agents: messaging channels, service APIs, infrastructure'
uuid: b6af0e3e-eaaf-4ed1-9253-e8a28315b55c
summary: 'Catalog of the 45 *_sea.py modules: messaging channels, service APIs, infrastructure
  a2a/oai, ask_sea; which ones have gateway/cron delivery (_make_backend).'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Catalog of channel agents

`src/kiss/agents/third_party_agents/` holds 45 `*_sea.py` modules plus `govee.py`. The
channel name used in prompts / `run_agent` is the module stem minus `_sea`. Every module except
`ask_sea.py` has a console script (`kiss-<name>`; exceptions `kiss-gchat` for `googlechat`,
`kiss-ha` for `homeassistant`; see `pyproject.toml`).

## Gateway-capable (define `_make_backend()`; can carry inbound prompts and receive cron delivery)
a2a, bluebubbles, dingtalk, discord, email, feishu, googlechat, irc, line, matrix,
mattermost, msteams, nextcloud, ntfy, phone, qq, signal, simplex, slack, sms, synology,
telegram, webhook, weixin, whatsapp, zalo (26 modules; 25 of them messaging channels, plus a2a).

## Messaging / device channels without a gateway
gmail, homeassistant, imessage, nostr, tlon, twitch, wecom — outbound only.

## Service APIs (outbound only)
brave, firecrawl, github, gcal, gdocs, gdrive, gsheets, notion, overleaf, postgres.
Guardrails in config: GitHub `read_only: "true"` blocks mutating `gh_*` tools; PostgreSQL
defaults to server-enforced read-only (`default_transaction_read_only=on`, extended protocol
rejects multi-statement strings, 60 s `statement_timeout` on reads).

## Infrastructure / non-channels (hidden from `run_agent` via `_NON_CHANNEL_MODULES`)
- `oai_sea.py` — OpenAI-compatible server (`GET /v1/models`, `POST /v1/chat/completions` with
  Bearer `api_key`); a hash of the message prefix maps a conversation to a persistent daemon
  chat (`chat_map.json`).
- `a2a_sea.py` — Agent-to-Agent protocol: embedded server publishes
  `/.well-known/agent-card.json`, queues peer messages as prompts; outbound tools
  `a2a_discover`, `a2a_call`, `a2a_get_task`; optional bearer `token`; 20 messages per
  `contextId` per hour cap; audit in `a2a_audit.jsonl`.
- `ask_sea.py` — the `/ask` slash command: answers questions about a task from its persisted
  events with `task_overview`, `task_transcript`, `task_step`; defines no `BaseChannelAgent`.

`govee.py` is a Govee lights helper (not a channel), Muse-auth covered (service `govee`).

## Embedded-callback caveat
Backends that receive events through an embedded HTTP server (A2A, DingTalk, LINE, QQ,
Synology Chat, Webhook, Weixin, Zalo; helpers in `_backend_utils.py`:
`start_http_server`, `drain_queue_messages`) only receive events while a tick is running.
Polling backends (Slack, Telegram, email, ...) fetch history and catch up. Discord looks back
only ~1 s until a first message is captured.

## Name resolution for `--channel`
Only four adapters override `find_channel` with an API lookup: Slack (channel name without
`#`, or a `C…`/`G…`/`D…` ID), Discord (name searched across guilds), Matrix (`#room:server`
alias), Google Chat (space display name). Others need the platform id (Telegram needs the
numeric chat id).

## Sources
- `src/kiss/agents/third_party_agents/README.md` ("Agent catalog", "Schedules and always-on gateways")
- `src/kiss/agents/sorcar/agent_dispatch.py` (`_NON_CHANNEL_MODULES`, `available_channels`)
- `grep -l "def _make_backend" src/kiss/agents/third_party_agents/*_sea.py`
- `pyproject.toml` (`[project.scripts]`)
