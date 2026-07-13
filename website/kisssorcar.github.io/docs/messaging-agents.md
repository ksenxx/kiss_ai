# Messaging & Third-Party Agents

> KISS Sorcar includes 23 third-party messaging agents that can send and receive messages on your behalf, plus a Govee smart-home CLI.

## The 23 Messaging Agents

- BlueBubbles
- Discord
- Feishu
- Gmail
- Google Chat
- iMessage
- IRC
- LINE
- Matrix
- Mattermost
- Microsoft Teams
- Nextcloud Talk
- Nostr
- Phone Control
- Signal
- Slack
- SMS
- Synology Chat
- Telegram
- Tlon
- Twitch
- WhatsApp
- Zalo

These agents live in `src/kiss/agents/third_party_agents/` in the source repository.

## Smart Home

KISS Sorcar also ships a **Govee smart-home CLI** for controlling IoT lights (on/off, brightness, color, and color temperature) via the Govee Developer API.

## Example Prompts

```text
Can you authenticate me with the <workspace name> workspace on Slack using the Slack agent?
```

```text
Can you authenticate me with Gmail using the Gmail agent? Use the user's default
browser to ask the user to log in and to get the authentication token.
```

```text
Can you send "Hello" to 1-800-772-1213?
```

```text
Can you create a cron job with a name prefixed with "kiss-" which will check every
3 seconds if there are latest unanswered messages from @<user name> in the channel
sorcar using the Slack agent, then it will run the messages as tasks one-by-one in
the order of arrival and respond with the result suitably formatted for Slack.
```

See [Sample Tasks](sample-tasks.md) for more ready-to-use prompts.
