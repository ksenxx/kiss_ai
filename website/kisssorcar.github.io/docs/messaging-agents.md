# Messaging & Third-Party Agents

> KISS Sorcar includes 32 third-party channel agents that act on messaging services, mailboxes, and devices on your behalf, plus infrastructure agents and a Govee smart-home CLI.

## The 32 Channel Agents

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

In a chat task, just say what you want ("send 'running late' to Alice on WhatsApp") — Sorcar dispatches the matching channel agent through its `run_agent` tool. Each channel also has its own CLI entry point (`kiss-slack`, `kiss-gmail`, `kiss-whatsapp`, ...) for running channel tasks directly from the shell.

## Infrastructure Agents

Two infrastructure agents round out the set: an **A2A agent** (`kiss-a2a`) exposing Sorcar over the agent-to-agent protocol, and an **OpenAI-compatible server** (`kiss-oai`) that serves Sorcar behind an OpenAI-style HTTP API.

These agents live in `src/kiss/agents/third_party_agents/` in the source repository.

## Smart Home

KISS Sorcar also ships a **Govee smart-home CLI** for controlling IoT lights (on/off, brightness, color, and color temperature) via the Govee Developer API.

## Example Prompts

```text
Can you authenticate me with the <<workspace name>> workspace on Slack using the Slack agent?
```

```text
Can you authenticate me with Gmail using the Gmail agent? Use the user's default
browser to prompt the user to log in and obtain the authentication token.
```

```text
Can you send "Hello from Sorcar!" to 1-800-772-1213?
```

```text
Can you create a cron job with a name prefixed with "kiss-" which will check every
3 seconds if there are the latest unanswered messages from /<<user name>> in the channel
sorcar using the Slack agent, then it will run the messages as tasks one-by-one in
the order of arrival and respond with the result suitably formatted for Slack.
```

See [Sample Tasks](sample-tasks.md) for more ready-to-use prompts.
