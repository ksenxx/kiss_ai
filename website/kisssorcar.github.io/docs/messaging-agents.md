# Messaging & Third-Party Agents

> KISS Sorcar includes 44 third-party agents that act on messaging services, mailboxes, devices, and web services on your behalf — 32 messaging-channel agents and 10 service agents — plus infrastructure agents and a Govee smart-home CLI.

## The 32 Messaging-Channel Agents

- BlueBubbles
- DingTalk
- Discord
- Email (IMAP/SMTP)
- Feishu
- Gmail
- Google Chat
- Home Assistant
- iMessage
- IRC
- LINE
- Matrix
- Mattermost
- Microsoft Teams
- Nextcloud Talk
- Nostr
- ntfy
- Phone Control
- QQ
- Signal
- SimpleX
- Slack
- SMS
- Synology Chat
- Telegram
- Tlon
- Twitch
- Webhook
- WeCom
- WeiXin
- WhatsApp
- Zalo

## The 10 Service Agents

Ten service agents give Sorcar authenticated API tools for productivity and data services:

- Brave Search (`kiss-brave`)
- Firecrawl (`kiss-firecrawl`)
- GitHub (`kiss-github`)
- Google Calendar (`kiss-gcal`)
- Google Docs (`kiss-gdocs`)
- Google Drive (`kiss-gdrive`)
- Google Sheets (`kiss-gsheets`)
- Notion (`kiss-notion`)
- Overleaf (`kiss-overleaf`)
- PostgreSQL (`kiss-postgres`)

In a chat task, just say what you want ("send 'running late' to Alice on WhatsApp", "list my open GitHub PRs") — Sorcar dispatches the matching agent through its `run_agent` tool (`run_agent(agent="whatsapp", task="...")`). Every channel agent is a SEA with `channel: True`: the sub-task runs in `$KISS_HOME/channel_work` with the channel's authenticated API tools, inherits nothing from the calling task, and waits up to 3600 seconds unless the call passes a `timeout`. Typing `/whatsapp send 'running late' to Alice` in the chat runs the same agent directly in the tab. Each agent also has its own CLI entry point (`kiss-slack`, `kiss-gmail`, `kiss-whatsapp`, ...) for running tasks directly from the shell.

## Inbound Channel Gateways

Channels also work **inbound**: gateway-capable messaging channels can become prompt surfaces of their own. A one-shot `--channel` poll tick (normally scheduled as a recurring cron job — just ask for "an always-on Telegram gateway" in chat) drains new inbound messages and runs each as a Sorcar task, with persisted thread continuity across ticks, a delivery ledger, per-channel model/budget overrides, sender allow-lists (`--allow-users`), and an optional pairing handshake (`--pairing`, `--approve`, `--list-pending`) so only approved senders can drive the agent.

## Infrastructure Agents

Two infrastructure agents round out the set: an **A2A agent** (`kiss-a2a`) exposing Sorcar over the agent-to-agent protocol, and an **OpenAI-compatible server** (`kiss-oai`) that serves Sorcar behind an OpenAI-style HTTP API.

These agents live in `src/kiss/agents/third_party_agents/` in the source repository.

## Smart Home

KISS Sorcar also ships a **Govee smart-home CLI** for controlling IoT lights (on/off, brightness, color, and color temperature) via the Govee Developer API.

## Sign-in & Credential Isolation (Muse Auth)

Authentication runs on demand via each connector's `check_<service>_auth` and `authenticate_<service>` tools or the web app's **Connectors** panel — the agent never asks for a password or 2FA code. The six Google Workspace agents (Gmail, Calendar, Drive, Docs, Sheets, and Google Chat) support Composio one-click OAuth; GitHub, Microsoft Teams, Slack, and Discord use public OAuth device-code or PKCE flows; and interactive consent screens can be completed directly in the web app's Browser tab.

On Linux, credentials for the **18 Muse-covered connectors** (Slack, GitHub, Notion, Discord, Home Assistant, Firecrawl, Brave Search, ntfy, Govee, LINE, Mattermost, Nextcloud Talk, Synology Chat, Twitch, Zalo, BlueBubbles, Microsoft Teams, and Telegram) are isolated by default behind a Meta-Muse-style security boundary: legacy tokens auto-migrate into a vault owned by a local auth daemon on first use, the agent process holds only opaque surrogate tokens swapped for real credentials at the network edge, and every boundary-routed API request is host-allowlisted, classified read vs. write, and checked against an allow/deny/ask policy with an audit log (reads allowed by default; writes ask for a grant). Manage it with `python -m kiss.agents.third_party_agents.muse_auth`; opt out with `KISS_MUSE_AUTH=0`.

## Example Prompts

```text
Authenticate slack workspace <<workspace name>>.
```

```text
Every 2 minutes, run a gateway tick on the Slack channel sorcar, with pairing.
```

```text
Authenticate Gmail [, or gcal, gdrive, gdoc, gsheets]?
```

```text
Can you send "Hello from Sorcar!" to 1-800-999-9999?
```

```text
Can you check my Gmail every hour and ping me on Slack if there is any important
email that needs my immediate attention?
```

See [Sample Tasks](sample-tasks.md) for more ready-to-use prompts.
