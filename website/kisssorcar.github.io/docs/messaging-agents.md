# Messaging & Third-Party Agents

> KISS Sorcar includes 43 third-party agents that act on messaging services, mailboxes, devices, and web services on your behalf — 32 messaging-channel agents and 9 service agents — plus infrastructure agents and a Govee smart-home CLI.

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

## The 9 Service Agents

Nine service agents give Sorcar authenticated API tools for productivity and data services:

- Brave Search (`kiss-brave`)
- Firecrawl (`kiss-firecrawl`)
- GitHub (`kiss-github`)
- Google Calendar (`kiss-gcal`)
- Google Docs (`kiss-gdocs`)
- Google Drive (`kiss-gdrive`)
- Google Sheets (`kiss-gsheets`)
- Notion (`kiss-notion`)
- PostgreSQL (`kiss-postgres`)

In a chat task, just say what you want ("send 'running late' to Alice on WhatsApp", "list my open GitHub PRs") — Sorcar dispatches the matching agent through its `run_agent` tool. Each agent also has its own CLI entry point (`kiss-slack`, `kiss-gmail`, `kiss-whatsapp`, ...) for running tasks directly from the shell.

## Infrastructure Agents

Two infrastructure agents round out the set: an **A2A agent** (`kiss-a2a`) exposing Sorcar over the agent-to-agent protocol, and an **OpenAI-compatible server** (`kiss-oai`) that serves Sorcar behind an OpenAI-style HTTP API.

These agents live in `src/kiss/agents/third_party_agents/` in the source repository.

## Smart Home

KISS Sorcar also ships a **Govee smart-home CLI** for controlling IoT lights (on/off, brightness, color, and color temperature) via the Govee Developer API.

## Credential Isolation (Muse Auth)

On Linux, credentials for the 22 Muse-supported connectors (the six Google services — Google Chat's service-account mode excepted — plus Slack, GitHub, Notion, Discord, Home Assistant, Firecrawl, Brave Search, ntfy, Govee, LINE, Mattermost, Nextcloud Talk, Synology Chat, Twitch, Zalo, and BlueBubbles) are isolated by default behind a Meta-Muse-style security boundary: legacy tokens auto-migrate into a vault owned by a local auth daemon on first use (a one-time hand-off of the real credential; plaintext copies are then scrubbed on a best-effort basis), the agent process holds only opaque surrogate tokens that the daemon swaps for the real ones at the network edge, and every boundary-routed API request is host-allowlisted, classified read vs. write, and checked against an allow/deny/ask policy with an audit log. Reads are allowed by default; writes ask for a grant. Manage it with `python -m kiss.agents.third_party_agents.muse_auth` (`status`, `enroll`, `grant`, `audit`, and an `export` command that reads a vaulted credential back out for recovery); opt out with `KISS_MUSE_AUTH=0`.

## Example Prompts

```text
Authenticate slack workspace <<workspace name>>.
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
