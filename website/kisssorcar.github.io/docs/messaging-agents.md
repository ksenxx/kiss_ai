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
