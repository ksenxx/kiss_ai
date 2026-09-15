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

## Inbound Channel Gateways

Channels also work **inbound**: gateway-capable messaging channels can become prompt surfaces of their own. A one-shot `--channel` poll tick (normally scheduled as a recurring cron job — just ask for "an always-on Telegram gateway" in chat) drains new inbound messages and runs each as a Sorcar task, with persisted thread continuity across ticks, a delivery ledger, per-channel model/budget overrides, sender allow-lists (`--allow-users`), and an optional pairing handshake (`--pairing`, `--approve`, `--list-pending`) so only approved senders can drive the agent.

## Infrastructure Agents

Two infrastructure agents round out the set: an **A2A agent** (`kiss-a2a`) exposing Sorcar over the agent-to-agent protocol, and an **OpenAI-compatible server** (`kiss-oai`) that serves Sorcar behind an OpenAI-style HTTP API.

These agents live in `src/kiss/agents/third_party_agents/` in the source repository.

## Smart Home

KISS Sorcar also ships a **Govee smart-home CLI** for controlling IoT lights (on/off, brightness, color, and color temperature) via the Govee Developer API.

## Credential Isolation (Muse Auth)

On Linux, credentials for the 24 Muse-supported connectors (the six Google services — Google Chat's service-account mode excepted — plus Slack, GitHub, Notion, Discord, Home Assistant, Firecrawl, Brave Search, ntfy, Govee, LINE, Mattermost, Nextcloud Talk, Synology Chat, Twitch, Zalo, BlueBubbles, Microsoft Teams, and Telegram) are isolated by default behind a Meta-Muse-style security boundary: legacy tokens auto-migrate into a vault owned by a local auth daemon on first use (a one-time hand-off of the real credential; plaintext copies are then scrubbed on a best-effort basis), the agent process holds only opaque surrogate tokens that the daemon swaps for the real ones at the network edge, and every boundary-routed API request is host-allowlisted (credential-free, bodyless `GET`/`HEAD` redirect hops are the one permitted off-list exception), classified read vs. write, and checked against an allow/deny/ask policy with an audit log. Reads are allowed by default; writes ask for a grant. For Microsoft Teams, after the one-time enrollment hand-off, the daemon performs the OAuth token exchange itself, keeping the vaulted client secret out of ordinary agent API requests.

Where the provider supports a poll-based grant, connecting works like the Muse app's Connect button — the user signs in and approves in their own browser, nothing is pasted back: GitHub, Twitch, and Microsoft Teams use the OAuth device authorization grant (RFC 8628) with a public client ID, Nextcloud Talk uses Login Flow v2, Matrix uses the OAuth 2.0 device grant of homeservers backed by Matrix Authentication Service (matrix.org included), and Signal links this computer like Signal Desktop via a `signal-cli link` QR code; the refresh tokens these sign-ins produce are renewed by the daemon (`oauth2_refresh_token` credentials) or, for Matrix, by the agent itself. Providers without such a grant get a safe hand-off instead of browser automation: on a headless host the six Google services return the consent URL for the user to approve in their own browser and paste back the resulting `localhost` redirect URL, and Slack/Discord ask the user to create the bot in their own browser and paste back the bot token — the agent never asks for a password or 2FA code.

Manage it with `python -m kiss.agents.third_party_agents.muse_auth` (`status`, `enroll`, `import`, `grant`, `revoke`, `audit`, `clear`, `daemon`, `stop`, and an `export` command that reads a vaulted credential back out for recovery); opt out with `KISS_MUSE_AUTH=0`.

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
