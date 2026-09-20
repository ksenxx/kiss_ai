# Third-Party Agents

This package contains KISS Sorcar's **channel agents**: 43 `*_sea.py` modules plus the
`govee.py` smart-light helper. Each module wraps one external service — a messaging
platform (Slack, Telegram, WhatsApp, ...), a service API (GitHub, Notion, PostgreSQL,
...), or a piece of agent infrastructure (A2A, OpenAI-compatible server) — and exposes
it as a set of authenticated LLM tools.

You do not run these agents directly. You **prompt KISS Sorcar in plain language** on
any of its UI surfaces, name the service you want acted on, and Sorcar dispatches the
work to the right channel agent. This document explains what prompts you can send, on
which surfaces, and what each channel can do.

- [The surfaces: where prompts go](#the-surfaces-where-prompts-go)
- [How a prompt reaches a channel](#how-a-prompt-reaches-a-channel)
- [How a channel agent works](#how-a-channel-agent-works)
- [Authenticate by chatting](#authenticate-by-chatting)
- [Credential isolation (Muse auth)](#credential-isolation-muse-auth)
- [Agent catalog](#agent-catalog)
  - [Messaging and device channels](#messaging-and-device-channels-32)
  - [Service APIs](#service-apis-9)
  - [Infrastructure: two extra surfaces](#infrastructure-two-extra-surfaces)
  - [Home lights (Govee)](#home-lights-govee)
- [Writing good prompts — tips from Meta Muse](#writing-good-prompts--tips-from-meta-muse)
- [Example prompts](#example-prompts)
- [Sources](#sources)

## The surfaces: where prompts go

Every KISS Sorcar surface feeds the same local kiss-web daemon, so the **same
plain-language prompts work everywhere**:

1. **VS Code extension.** Open the KISS Sorcar sidebar and type your prompt in the
   chat box (`@` mentions attach files; the model picker and budget field sit next to
   the input).
2. **Web / mobile app.** The identical chat interface served over a cloudflared
   tunnel — copy the URL and password from the Settings panel and open it on any
   phone, tablet, or browser.
3. **Voice.** Press the mic button and speak, prefixed with the wake word: *"Sorcar,
   tell the eng Slack channel that the deploy is done."* You can also steer a running
   task by voice; Sorcar replies aloud in the language you spoke.
4. **Your existing messaging apps.** Once a channel gateway is running (see
   [Always-on gateways](#always-on-gateways) below), any message you send to the bot
   in Telegram, Slack, Discord, email, WhatsApp, ... *is* a prompt: it becomes a
   Sorcar task and the answer comes back in the same chat or thread. Follow-up
   messages in the same thread continue the same conversation.
5. **Any OpenAI-compatible client.** Open WebUI, LibreChat, or an `openai` SDK script
   pointed at the daemon's OpenAI-compatible server — every user message you type
   there is a prompt to the daemon (see
   [Infrastructure: two extra surfaces](#infrastructure-two-extra-surfaces)).
6. **Other agents.** A peer agent speaking the A2A protocol can send prompts to your
   daemon machine-to-machine (same section).

Unlike a fire-and-forget script, the chat surfaces show you the task's live progress
and its final summary, and you can answer the agent's questions mid-task — which is
exactly what interactive authentication and write-approval flows need.

## How a prompt reaches a channel

Name the service in your prompt and Sorcar routes it. Internally the session calls its
`run_agent` tool with the channel name and your request; the dispatched sub-session
already carries that channel's authenticated tools and is instructed to use them
directly, without exploring source code.

> Send "dinner at 7" to Telegram chat 123456789.

> Post "deploy done" to the #eng channel on Slack.

> Turn off all kitchen lights with Home Assistant.

Channel names are matched forgivingly — case, spaces, hyphens, and underscores are
ignored, so "Home Assistant", "home-assistant", and "HOMEASSISTANT" all resolve to the
`homeassistant` channel. For multi-account
channels, name the workspace in the prompt ("using the acme Slack workspace, ...") and
Sorcar passes it through; you can likewise ask for a specific model or budget for the
sub-task. The two infrastructure modules (`a2a`, `oai`) are not
dispatchable — they are surfaces, not services you ask Sorcar to act on.

Prompts that span several services also work in a single message: the top-level
session orchestrates, dispatching one channel at a time and passing results between
them. Each dispatched channel session handles only its own service (it cannot dispatch
further), so let the session you are chatting with do the coordination — which it does
by default.

When you want a specific channel with no routing guesswork, start the prompt with its
slash command: `/slack post "deploy done" to #eng`. Every `xxx_sea.py` in this
package is registered as `/xxx`, the chat box autocompletes the names, and the daemon
turns the prompt into a direct `run_agent` call on that file. Folders of your own SEAs
listed in `~/.kiss/SEAS.md` are registered the same way; the file syntax and the
dispatch flow are in
[docs/sea-commands.md](https://kisssorcar.github.io/docs/sea-commands.md).

## How a channel agent works

A channel agent is **not** an executable agent itself. Every prompt is submitted to
the kiss-web daemon, and the daemon builds a full chat agent with the standard tools
(bash, file editing, browser automation). The channel agent instance is the *carrier*
of channel identity (see `BaseChannelAgent` in `_channel_agent_utils.py`):

- Each module defines a `tools()` function. The daemon calls it to build the channel's
  tool list: the agent's **auth tools** (always present, e.g. `check_slack_auth`,
  `authenticate_slack`) plus, once authenticated, every public method of the module's
  `*ChannelBackend` class (e.g. `post_message`, `read_messages`, `search_messages`).
- Config lives under `~/.kiss/third_party_agents/<service>/` (`$KISS_HOME` overrides
  `~/.kiss`). On Linux, outbound API secrets for the 24 Muse-covered services (see
  below) migrate out of those files into the `$KISS_HOME/muse_auth/vault` credential
  vault on first use; non-secret settings, OAuth bootstrap files (Google's
  `credentials.json`), and inbound-verification secrets (LINE's `channel_secret`) stay
  in the service directory. Because auth tools are always available,
  a not-yet-configured agent can walk you through authentication *in chat* — you never
  have to hand-edit `config.json` first.
- Backends whose platform has an inbound message stream also implement
  `_make_backend()`, which enables **gateway mode** (the channel itself becomes a
  prompt surface) and **scheduled delivery** (a cron job's result lands in the
  channel). API-only services (GitHub, Notion, the Google Workspace agents, ...) are
  outbound-only: they can be acted on, but cannot carry prompts in or receive
  deliveries.

## Authenticate by chatting

Auth tools are always present, so a fresh, unconfigured channel sets itself up in
conversation:

> Check my GitHub auth; if it's missing, walk me through creating a token and store it.

Each channel's `check_<service>_auth` returns setup instructions when unconfigured,
and several channels go further with a guided sign-in. Three styles exist, and in
every one the sign-in itself stays in your hands — the agent never types or asks for
your password or 2FA code:

- **Connect-style browser sign-in** (GitHub, Twitch, Microsoft Teams, Nextcloud Talk,
  Matrix, Signal). `authenticate_<service>` without a token returns a sign-in link —
  for Signal, a QR code to scan like Signal Desktop — that you open and approve in
  your *own* browser or on your phone, while the agent polls in the background;
  `finish_<service>_auth` collects the credential (answering `pending` until your
  approval lands). WhatsApp has its own variant of this: `start_whatsapp_bridge` +
  `get_whatsapp_qr_code` show a pairing QR that you scan from the phone, and
  `wait_for_whatsapp_pairing` waits for the scan.
- **Consent paste-back** (Gmail and the five other Google agents). On a desktop,
  `authenticate_<service>` simply opens Google's consent window locally; on a
  headless machine it hands you the authorization URL to open in your own browser —
  you approve access and paste the resulting `localhost` redirect URL back into the
  chat, and `finish_<service>_auth` stores the token. (Creating the OAuth client
  itself is separate: `start_gmail_browser_setup` / `start_<service>_browser_setup`
  walks the Google Cloud Console credential setup — Google Chat, whose check tool
  gives the setup instructions instead, has no such tool.)
- **Portal walkthrough with token paste-back** (Slack, Discord).
  `start_slack_browser_auth` / `start_discord_browser_auth` drive the provider's
  developer portal in the built-in browser while its pages load cleanly; at any login
  screen, captcha, or page failure the agent stops and asks you to create the app in
  your own browser and paste the bot token back.

Do interactive auth from a chat surface — the agent may need to ask you questions,
and the chat panel is where you answer them.

One caveat: backend tools are snapshotted when a session starts, so the session that
stores a fresh token cannot call the new backend tools itself. Send the real work as
your **next** prompt:

> List my open pull requests and write them to ./prs.md.

## Credential isolation (Muse auth)

On Linux, credentials for the 24 Muse-supported services are isolated by default behind
a Meta-Muse-style security boundary implemented in the `muse_auth/` package: legacy
tokens auto-migrate into a vault owned by a local auth daemon on first use (a one-time
hand-off of the real credential through the agent process), after which ordinary
boundary-routed requests carry only opaque surrogate tokens that the daemon swaps for
the real ones at the network edge, and every boundary-routed request is checked against
the service's host allowlist (one deliberate exception: bodyless GET/HEAD redirect hops
are followed after stripping credentials, even off-allowlist), classified read vs.
write, and checked against an allow/deny/ask policy. Reads are allowed by default;
writes ask for a grant. The audit log records the Sentinel's allow/deny/ask decisions —
not whether the network call afterwards succeeded.

Covered services (`muse_auth/_common.py` `SERVICE_HOSTS`): the six Google services
(`gmail`, `google_calendar`, `google_docs`, `google_drive`, `google_sheets`,
`googlechat` — Google Chat's service-account mode excepted) plus `slack`, `github`,
`notion`, `discord`, `homeassistant`, `firecrawl`, `brave_search`, `ntfy`, `govee`,
`line`, `mattermost`, `msteams`, `nextcloud`, `synology`, `telegram`, `twitch`, `zalo`,
and `bluebubbles`. Other channels keep their legacy direct-credential path.

The boundary is managed with `python -m kiss.agents.third_party_agents.muse_auth`
(verbs: `status`, `enroll SERVICE`, `import SERVICE`, `grant SERVICE read|write`,
`revoke SERVICE`, `export SERVICE`, `clear SERVICE`, `audit`, `daemon`, `stop`) —
you can run it yourself or simply ask Sorcar to do it:

> Show me the recent Muse auth audit records.

> Grant the github service a single-use write permission.

`enroll` supports only the six Google OAuth services; every other covered service
enrolls itself when its legacy credential auto-migrates on first use — and the
Connect-style browser sign-ins for GitHub, Twitch, and Microsoft Teams store their
grant straight into the vault: as a refresh-token credential the daemon renews itself
when the grant includes a refresh token (Teams requires one; GitHub OAuth apps issue
one only with expiring tokens enabled), otherwise as a plain bearer token.
`grant SERVICE write` defaults to a single-use grant (`--scope once`); use `--scope ttl --ttl 3600`,
`--scope session`, or `--scope perpetual` for a standing one. Opt out with
`KISS_MUSE_AUTH=0` in `$KISS_HOME/api_keys.env` (default `~/.kiss/api_keys.env`).

## Agent catalog

The **Name in prompts** column is the canonical channel name; say it (or any spacing /
casing variant) in your prompt to target the channel. Config paths are relative to
`~/.kiss/third_party_agents/` (override the root with `$KISS_HOME`; exception: Slack's
workspace token store is hard-coded under `~/.kiss`). Token locations name the legacy
(non-Muse) files: on Linux with Muse enabled, the secret moves into the vault on first
use and the legacy file is removed. "Gateway" marks the modules with `_make_backend()`
— they can carry inbound prompts and receive scheduled deliveries. Tool names are the
exact callables the dispatched session sees, i.e. what your prompts can make the
channel do; every agent also gets its auth tools (`check_<service>_auth`,
`authenticate_<service>`, `clear_<service>_auth`, plus the service-specific sign-in
helpers noted below, such as `finish_<service>_auth` and the browser-setup tools).

### Messaging and device channels (32)

| Agent | Name in prompts | Gateway | Auth / config | Backend tools |
| --- | --- | --- | --- | --- |
| BlueBubbles (iMessage via a Mac server) | `bluebubbles` | yes | server URL + password, `bluebubbles/config.json` | `list_chats`, `get_chat`, `get_chat_messages`, `post_message`, `get_server_info`, `mark_chat_read` |
| DingTalk group robots | `dingtalk` | yes | robot webhook (+ optional `secret`, `outgoing_token`), `dingtalk/config.json` | `post_message`, `post_markdown` |
| Discord | `discord` | yes | bot token (also `start_discord_browser_auth`), `discord/config.json` | `list_guilds`, `list_third_party_agents` (channels), `get_channel`, `get_channel_messages`, `post_message`, `edit_message`, `delete_message`, `add_reaction`, `create_thread`, `list_guild_members`, `create_invite` |
| Email (any IMAP/SMTP mailbox) | `email` | yes | IMAP host + SMTP host + address + app-password, `email/config.json` | `send_email`, `list_unread_emails`, `read_email`, `mark_email_read` |
| Feishu / Lark | `feishu` | yes | `app_id` + `app_secret`, `feishu/config.json` | `send_text_message`, `reply_message`, `delete_message`, `list_messages`, `list_chats`, `get_chat`, `get_user_info` |
| Gmail | `gmail` | no | OAuth2 (`start_gmail_browser_setup`, `finish_gmail_auth`), token in `gmail/` | `get_profile`, `list_messages`, `get_message`, `send_email`, `reply_to_message`, `create_draft`, `trash_message`, `untrash_message`, `delete_message`, `modify_labels`, `list_labels`, `create_label`, `get_attachment`, `get_thread` |
| Google Chat | `googlechat` | yes | service account or OAuth2 (`finish_googlechat_auth`), `googlechat/` | `list_spaces`, `get_space`, `list_members`, `list_messages`, `get_message`, `post_message`, `update_message`, `delete_message`, `create_space` |
| Home Assistant | `homeassistant` | no | `base_url` + long-lived token, `homeassistant/config.json` | `ha_get_states`, `ha_call_service`, `ha_list_services`, `ha_get_history`, `ha_render_template`, `ha_fire_event` |
| iMessage (macOS AppleScript) | `imessage` | no | local Messages app, `imessage/config.json` | `send_imessage`, `send_attachment`, `list_conversations`, `get_messages` |
| IRC | `irc` | yes | server/nick (+ NickServ), `irc/config.json` | `connect_irc`, `join_irc_channel`, `leave_channel`, `post_message`, `send_notice`, `get_topic`, `set_topic`, `kick_user`, `whois`, `identify_nickserv` |
| LINE | `line` | yes | channel access token, `line/config.json` | `push_text_message`, `reply_message`, `get_profile`, `get_quota`, `leave_group`, `push_image_message` |
| Matrix | `matrix` | yes | browser sign-in via the homeserver's OAuth 2.0 device grant (`authenticate_matrix(homeserver_url)`, `finish_matrix_auth`; matrix.org and other MAS-backed servers; the agent renews the short-lived token itself) or a hand-supplied access token (matrix-nio), `matrix/config.json` | `list_rooms`, `join_room`, `leave_room`, `send_text_message`, `send_notice`, `get_room_members`, `invite_user`, `kick_user`, `create_room`, `get_profile`, `refresh_if_needed` (renews an OAuth-issued token) |
| Mattermost | `mattermost` | yes | server URL + personal access token, `mattermost/config.json` | `list_teams`, `list_third_party_agents` (channels), `get_channel`, `list_channel_posts`, `create_post`, `delete_post`, `get_user`, `list_users`, `create_direct_message_channel`, `add_reaction` |
| Microsoft Teams | `msteams` | yes | browser sign-in via the Entra device code flow (`authenticate_msteams(tenant_id, client_id)`, `finish_msteams_auth`; delegated token refreshed by the Muse daemon) or app-only client credentials, `msteams/config.json` | `list_teams`, `get_team`, `list_third_party_agents` (channels), `list_channel_messages`, `post_channel_message`, `reply_to_message`, `list_chats`, `post_chat_message`, `list_team_members` |
| Nextcloud Talk | `nextcloud` | yes | browser sign-in via Login Flow v2 (`authenticate_nextcloud(url)`, `finish_nextcloud_auth`; app password issued by the server) or username + app password, `nextcloud/config.json` | `list_rooms`, `get_room`, `create_room`, `list_participants`, `list_messages`, `post_message`, `set_room_name`, `delete_message`, `revoke_app_password` |
| Nostr | `nostr` | no | private key, optional relays (default `wss://relay.damus.io`; pynostr), `nostr/config.json` | `publish_note`, `publish_reply`, `send_dm`, `get_profile`, `set_profile`, `list_relays`, `add_relay`, `remove_relay` |
| ntfy pub-sub | `ntfy` | yes | `topic` (+ optional `server`, `token`), `ntfy/config.json` | `publish_notification`, `poll_topic` |
| Phone control (Android companion app) | `phone` | yes | device IP + optional port/API key of the companion REST app, `phone/config.json` | `send_sms`, `make_call`, `end_call`, `list_sms_conversations`, `get_sms_messages`, `get_call_log`, `get_device_info`, `list_notifications`, `dismiss_notification`, `send_notification_reply` |
| QQ bot platform | `qq` | yes | app id/secret (Ed25519 webhook), `qq/config.json` | `send_group_message`, `send_c2c_message` |
| Signal (signal-cli) | `signal` | yes | link like Signal Desktop: `authenticate_signal()` runs `signal-cli link` and shows a QR code to scan from the phone, `finish_signal_auth` records the account; or an already registered signal-cli number, `signal/config.json` | `send_signal_message`, `receive_messages`, `send_attachment`, `list_contacts`, `list_groups` |
| SimpleX Chat | `simplex` | yes | local `simplex-chat -p 5225` WebSocket, `simplex/config.json` | `send_simplex_message`, `list_simplex_contacts`, `get_simplex_address` |
| Slack | `slack` | yes | bot token (also `start_slack_browser_auth`); one credential set per workspace in `slack/<workspace>/token.json` | `list_third_party_agents` (channels), `read_messages`, `read_thread`, `post_message`, `update_message`, `delete_message`, `list_users`, `get_user_info`, `create_channel`, `invite_to_channel`, `add_reaction`, `search_messages`, `set_channel_topic`, `upload_file`, `get_channel_info` |
| SMS / voice (Twilio) | `sms` | yes | account SID + auth token + from number, `sms/config.json` | `send_sms`, `send_mms`, `list_messages`, `get_message`, `list_phone_numbers`, `get_account_info`, `send_whatsapp_message`, `create_call`, `list_calls`, `get_call`, `cancel_message` |
| Synology Chat | `synology` | yes | incoming/outgoing webhooks, `synology/config.json` | `post_message`, `send_file_message` |
| Telegram | `telegram` | yes | @BotFather bot token, `telegram/config.json` | `send_text`, `send_photo`, `send_document`, `edit_message_text`, `delete_message`, `pin_message`, `unpin_message`, `get_chat`, `get_chat_members_count`, `get_chat_member`, `ban_chat_member`, `unban_chat_member`, `get_updates`, `send_poll`, `forward_message` |
| Tlon / Urbit | `tlon` | no | Eyre HTTP server + code, `tlon/config.json` | `list_groups`, `list_third_party_agents` (channels), `get_messages`, `post_message`, `get_profile`, `poke`, `scry` |
| Twitch | `twitch` | no | browser sign-in via the device code grant (`authenticate_twitch(client_id)`, `finish_twitch_auth`; rotating refresh token kept by the Muse daemon) or client ID + OAuth access token (all calls, chat included, via Helix), `twitch/config.json` | `get_stream_info`, `get_channel_info`, `get_user_info`, `get_chatters`, `send_chat_message`, `ban_user`, `search_third_party_agents` (channels), `get_clips`, `create_clip` |
| Webhook routes (inbound HMAC webhooks) | `webhook` | yes | listener `port` (routes added via `add_webhook_route`), `webhook/config.json` | `add_webhook_route`, `remove_webhook_route`, `list_webhook_routes` |
| WeCom group robots | `wecom` | no | robot webhook, `wecom/config.json` | `post_message`, `post_markdown` |
| Weixin / WeChat Official Accounts | `weixin` | yes | app id/secret, optional callback token (enables callback verification), `weixin/config.json` | `send_text_message`, `get_user_info` |
| WhatsApp (personal, QR-paired bridge) | `whatsapp` | yes | whatsapp-mcp Go bridge (auth tools also: `start_whatsapp_bridge`, `get_whatsapp_qr_code`, `wait_for_whatsapp_pairing`, `stop_whatsapp_bridge`), `whatsapp/` | `search_whatsapp_contacts`, `list_whatsapp_chats`, `get_whatsapp_chat`, `get_whatsapp_direct_chat_by_contact`, `get_whatsapp_contact_chats`, `get_whatsapp_last_interaction`, `list_whatsapp_messages`, `get_whatsapp_message_context`, `send_whatsapp_message`, `send_whatsapp_file`, `send_whatsapp_audio_message`, `download_whatsapp_media` |
| Zalo Official Account | `zalo` | yes | OA access token, `zalo/config.json` | `send_text_message`, `send_image_message`, `get_follower_profile`, `get_followers`, `get_oa_info`, `get_recent_messages`, `get_conversation`, `upload_image` |

Platform notes: BlueBubbles and iMessage are macOS-only (BlueBubbles needs a Mac running
the BlueBubbles server; iMessage drives the local Messages app via `osascript`). The
email agent drops automated mail (no-reply senders, `Auto-Submitted`, `Precedence:
bulk/junk/list`) from the gateway loop so it only answers real people. The webhook
agent verifies GitHub (`X-Hub-Signature-256`) or generic timestamped HMAC signatures,
caps bodies at 1 MB, suppresses duplicate deliveries, rate-limits to 60 events per
route per minute, and can either queue events as agent tasks or push them straight
through another channel's backend (`deliver_module` routes).

### Service APIs (9)

| Agent | Name in prompts | Auth / config | Backend tools |
| --- | --- | --- | --- |
| Brave Search | `brave` | subscription token, `brave_search/config.json` | `brave_web_search`, `brave_news_search`, `brave_image_search`, `brave_video_search` |
| Firecrawl (scraping/crawling) | `firecrawl` | API key (+ optional self-hosted `base_url`), `firecrawl/config.json` | `firecrawl_scrape`, `firecrawl_map`, `firecrawl_search`, `firecrawl_start_crawl`, `firecrawl_get_crawl_status`, `firecrawl_cancel_crawl` |
| GitHub | `github` | browser sign-in via the OAuth device flow (`authenticate_github(client_id=...)` with a device-flow-enabled OAuth app, `finish_github_auth`) or a personal access token (+ optional `read_only: "true"`), `github/config.json` | `gh_get_me`, `gh_search_repositories`, `gh_get_repository`, `gh_list_issues`, `gh_get_issue`, `gh_list_issue_comments`, `gh_search_issues`, `gh_search_code`, `gh_list_pull_requests`, `gh_get_pull_request`, `gh_get_pull_request_diff`, `gh_get_file_contents`, `gh_list_commits`, `gh_list_branches`, `gh_create_issue`, `gh_comment_on_issue`, `gh_update_issue`, `gh_create_pull_request`, `gh_merge_pull_request` |
| Google Calendar | `gcal` | OAuth2 quintet (`check_google_calendar_auth`, `authenticate_google_calendar`, `clear_google_calendar_auth`, `start_google_calendar_browser_setup`, `finish_google_calendar_auth`), `google_calendar/` | `gcal_list_calendars`, `gcal_list_events`, `gcal_get_event`, `gcal_create_event`, `gcal_update_event`, `gcal_delete_event`, `gcal_quick_add` |
| Google Docs | `gdocs` | OAuth2 quintet (as above, for `google_docs`), `google_docs/` | `gdocs_create_document`, `gdocs_read_document`, `gdocs_append_text`, `gdocs_replace_text`, `gdocs_insert_text`, `gdocs_batch_update`, `gdocs_list_documents` |
| Google Drive | `gdrive` | OAuth2 quintet (for `google_drive`), `google_drive/` | `gdrive_search_files`, `gdrive_get_file`, `gdrive_read_file`, `gdrive_download_file`, `gdrive_upload_file`, `gdrive_create_folder`, `gdrive_share_file`, `gdrive_move_file`, `gdrive_trash_file` |
| Google Sheets | `gsheets` | OAuth2 quintet (for `google_sheets`), `google_sheets/` | `gsheets_create_spreadsheet`, `gsheets_get_info`, `gsheets_get_values`, `gsheets_update_values`, `gsheets_append_values`, `gsheets_clear_values`, `gsheets_add_sheet`, `gsheets_batch_update`, `gsheets_list_spreadsheets` |
| Notion | `notion` | internal-integration token, `notion/config.json` | `notion_search`, `notion_get_page`, `notion_get_block_children`, `notion_append_paragraph`, `notion_append_blocks`, `notion_create_page`, `notion_update_page`, `notion_get_database`, `notion_query_database`, `notion_list_users`, `notion_create_comment`, `notion_get_comments` |
| PostgreSQL | `postgres` | `postgresql://` URI, `postgres/config.json` | `pg_query`, `pg_execute`, `pg_list_schemas`, `pg_list_tables`, `pg_describe_table`, `pg_list_indexes`, `pg_explain` |

All nine are outbound-only (no gateway mode). PostgreSQL defaults to **read-only
enforced server-side**: connections open with `default_transaction_read_only=on` and
`pg_query` uses the extended query protocol so multi-statement strings are rejected;
read paths run under a 60 s server-side `statement_timeout` while `pg_execute` (write
mode only) has none. GitHub's `read_only: "true"` config key blocks every mutating
`gh_*` tool.

### Infrastructure: two extra surfaces

These two modules are hidden from prompt dispatch — they are not services you ask
Sorcar to act on, but ways for *other software* to send prompts to your daemon.

- **OpenAI-compatible server** (`oai_sea.py`). Turns kiss-web into an
  OpenAI-style backend: unauthenticated `GET /v1/models` and `POST
  /v1/chat/completions` (requires Bearer `api_key`). Point Open WebUI, LibreChat, or
  any `openai` SDK script at it and every user message typed there becomes a daemon
  task; system messages become the task's system prompt, and a hash of the message
  prefix maps each conversation back to the same persistent daemon chat across
  requests (`chat_map.json`). A one-time configure-and-serve setup from a terminal is
  required (see the module docstring); after that the connected client is just another
  chat surface.
- **A2A (Agent-to-Agent protocol)** (`a2a_sea.py`). Inbound, it embeds an HTTP
  server publishing this agent's card at `/.well-known/agent-card.json` and queues
  peer messages as prompts for the channel runner; outbound, its tools
  (`a2a_discover`, `a2a_call`, `a2a_get_task`) let a session talk to a remote peer.
  Peer input is treated as untrusted text (queued, never executed); an optional bearer
  `token` gates inbound JSON-RPC POSTs (the agent card `GET` is public); a
  20-messages-per-`contextId`-per-hour cap (per backend instance) stops ping-pong
  loops; inbound JSON-RPC requests land in `a2a_audit.jsonl`. Config:
  `a2a/config.json`.

### Home lights (Govee)

`govee.py` is a small helper for Govee smart lights (the Developer API), not a channel
agent — but it is Muse-auth covered and it is the preferred way to act on home lights.
This repository's `SORCAR.md` tells every session to use it for light actions, so
plain prompts on any surface just work:

> Turn off the living room lamp.

> Set the living room lamp to 40% brightness and a warm 2700K white.

> List all my Govee devices and their current state.

It can list devices, query state, switch on/off, set brightness (1–100), set an RGB
color, and set a color temperature in kelvin. The key comes from `$GOVEE_API_KEY` and
is enrolled into the Muse vault on first use; device-state queries classify as reads,
`/device/control` calls follow the write policy.

## Writing good prompts — tips from Meta Muse

Meta's Muse personal agent (launched September 8, 2026) popularized a set of habits for
delegating work to an agent that acts on your behalf. KISS's channel agents follow the
same trust architecture Muse does — a credential vault held by a separate daemon
(day-to-day API calls see only surrogate tokens), a Sentinel policy that allows reads
and asks before writes, and an audit log — so Muse's prompt-writing guidance transfers
directly:

1. **Define the result *and* the stopping point.** Muse's documentation and every
   launch guide converge on this: a good prompt states the outcome and where the agent
   must stop. "Draft replies to unanswered client emails — do not send anything" beats
   "handle my email."
2. **Give a specific, bounded job.** State the outcome, the constraints, the criteria,
   and what you want to approve. "Find three options under $200 and compare them" is
   better than "find me something good."
3. **Start read-only, then widen.** Muse's suggested first prompt pattern is
   deliberately read-only ("Review my calendar for next week and suggest three
   45-minute blocks... Don't create or move any events yet"). Judge the agent's work on
   something reversible before granting write access.
4. **Least privilege.** Ask "what is the minimum access this exact task needs?", not
   "how much access can the agent have?" In KISS terms: keep GitHub's
   `read_only: "true"` and Postgres's default read-only mode on until a task actually
   needs writes, and grant Muse-auth write access per service, not globally.
5. **Draft before send, compare before buy, plan before book.** Let the agent do the
   reversible preparation; keep yourself at the decision point before anything sends,
   spends, books, submits, or deletes. On a chat surface this is natural: ask for a
   draft, read it in the chat panel, then send a follow-up prompt to transmit it. The
   KISS Sentinel enforces exactly this split — reads proceed silently, writes ask —
   which also keeps approvals meaningful instead of becoming a button you tap without
   reading.
6. **Anchor schedules in plain language with exact times, and name them.** Muse's
   reminder help center uses "Remind me to call the dentist tomorrow at 10am" and
   "Every Monday at 9am, remind me to submit my timesheet", then cancels by name
   ("Cancel my dentist reminder"). Phrase scheduled prompts to Sorcar the same way: a
   name, an exact schedule, a bounded prompt, a delivery target.
7. **Meet people in the channel they already use.** Muse's biggest distribution bet is
   living inside WhatsApp rather than a new app. The KISS equivalent: have scheduled
   results delivered where the audience already is (a Telegram chat, `#eng` on Slack,
   an ntfy topic), and run gateways on existing channels instead of inventing new
   inboxes.
8. **Verify in the real service, then review the trail.** Check the actual Slack
   channel / calendar / repository after the task, and ask Sorcar to show the recent
   Muse-auth audit records to see which API calls the Sentinel allowed, denied, or
   asked about (it records decisions, not network outcomes).

## Example prompts

Everything below is a prompt you can type — or speak, with the wake word "sorcar, …" —
into any Sorcar chat surface (VS Code sidebar, web/mobile app, a channel gateway, an
OpenAI-compatible client).

### Getting started

**1. One-shot message.** The smallest useful prompt — one channel, one action, an
explicit target:

> Post "deploy of v2.3 finished, all green" to the Slack channel #eng.

**2. Authenticate in chat, not in config files:**

> Check my GitHub auth; if it's missing, walk me through creating a token and store it.

Then, as a second prompt (the session that stores a fresh token cannot use the new
backend tools itself — they are snapshotted at session start):

> List my open pull requests in the chat.

**3. Probe read-only before you trust writes** (Muse tip 3). First prompt against a
new Postgres config:

> List the schemas and tables in my Postgres database, describe the orders table, and
> show me the ten most recent orders. Do not modify the database.

The default read-only mode makes the "do not modify" clause server-enforced, not a
polite request.

**4. Push a notification to yourself:**

> Notify me on ntfy that the benchmark finished: 42.3 seconds, 0 failures.

**5. Speak it.** Press the mic button on any chat surface:

> Sorcar, send "dinner at 7" to my family Telegram chat.

### Cross-service pipelines

The most powerful pattern: one prompt that names two or three services. The session
you are chatting with orchestrates, dispatching each channel in turn and passing
results between them.

**6. GitHub → Slack standup digest:**

> Summarize the pull requests merged into acme/api in the last 24 hours (title,
> author, one-line summary each), then post the digest to the Slack channel #standup.

**7. Postgres → Google Sheets weekly report:**

> Query the orders table for the last 7 days: one row per day with order count and
> revenue (read-only). Then append the rows to the "Weekly KPIs" Google Sheets
> spreadsheet, one row per day.

**8. Firecrawl → Notion research capture:**

> Use Firecrawl to scrape https://example.com/pricing and its /docs subpages (map the
> site first, then scrape the top 5 relevant pages). Then create a Notion page
> "Competitor pricing — example.com" under my Research page with a structured summary
> and the source URLs.

**9. Brave Search → Email newsletter, with a human checkpoint.** `send_email`
transmits immediately, so split draft and send (Muse tip 5) into two prompts and
review the draft in the chat panel between them:

> Find the 5 most significant news stories about RISC-V from the past week with Brave
> News search and show me a plain-text digest with links. Do not send anything.

After you have read the digest:

> Looks good — email that digest to team@acme.dev with the subject "RISC-V weekly".

**10. Gmail → Google Calendar.** Read one service, write another, keep the stop point
explicit (Muse tips 1 and 5):

> Find emails from the last 3 days that propose meetings. For each, extract the
> proposed time and attendees, create a Google Calendar event, and then read the
> created events back to me so I can verify them.

Then check the new events in Google Calendar itself (Muse tip 8).

**11. Home Assistant + Govee evening scene.** Two device backends, one instruction:

> Set the living room to movie mode: turn off the ceiling lights via Home Assistant,
> then dim the Govee living room lamp to 15% at 2700K.

(The Muse write policy still gates Govee's `/device/control` calls; run this from a
workspace whose `SORCAR.md` points at `govee.py`, as this repository's does.)

**12. Twitch → Discord stream announcement:**

> Get the stream info for the Twitch channel "mychannel". If it is live, post "We are
> live: {stream title} — https://twitch.tv/mychannel" to the #announcements Discord
> channel; if it is offline, do nothing.

(`get_stream_info` returns title and viewer data but no URL — the prompt builds the
link from the channel login.)

**13. Google Drive backup, link shared to Mattermost:**

> Upload ./reports/q3-summary.pdf to the "Team Reports" folder on Google Drive, share
> it read-only with team-lead@acme.dev, then post the file link to the town-square
> Mattermost channel.

**14. Slack thread → Google Docs minutes:**

> Read the full Slack thread in #planning at ts 1726300000.000100, then create a
> Google Doc "Planning sync 2026-09-14" with the decisions, action items (owner + due
> date), and open questions.

**15. Phone → Email SMS triage:**

> List the SMS conversations on my phone with unread messages from today, summarize
> each in one line, and email the triage list to me@example.com. Do not reply to any
> SMS.

### Schedules and always-on gateways

The built-in cron agent understands plain-language schedules; the kiss-web daemon
ticks the scheduler automatically, and a job's result can be delivered to any
gateway-capable channel (25 of the 32 messaging channels; a `[SILENT]` or `NO_REPLY`
result suppresses delivery).

**16. Scheduled brief delivered where you already read** (Muse tips 6 and 7):

> Every weekday at 9am, summarize today's HN front page in 10 bullets and deliver it
> to my Telegram chat 123456789.

**17. Silent health check — alert only on failure:**

> Every 15 minutes, fetch https://api.acme.dev/health with curl. If it returns HTTP
> 200, reply [SILENT]; otherwise describe the failure. Deliver the result to ntfy and
> name the job "api healthcheck".

**18. Manage schedules by name**, exactly as you created them:

> List my scheduled jobs. Pause "api healthcheck" and cancel the morning brief.

<a id="always-on-gateways"></a>
**19. Always-on gateway: make a messaging channel itself a prompt surface.** A
gateway is a recurring poll tick of a channel: each tick fetches pending messages,
runs a daemon task per pending top-level message (pending follow-ups in the same
thread are batched into one continuation task), and replies in-channel. Since a tick
is just a shell command, you set one up as a scheduled command job:

> Every 2 minutes, run the command `kiss-telegram --channel=-1001234567890 --pairing`.

You do not have to write that command or know the numeric chat ID yourself. Ask for
the gateway in plain language and let the chat session do the plumbing: it looks up
the chat identifier through the channel agent, composes the tick command, and hands it
to the cron agent as the scheduled command job:

> Find the chat ID of my Telegram group "Sen family" from the bot's recent updates,
> then schedule a gateway tick of that chat with pairing every 2 minutes.

(Send any message in the bot's group first so a recent update exists to read the ID
from.) On the four adapters whose `find_channel` resolves names through the platform
API, even that lookup is unnecessary — the tick's `--channel` value can be the name
itself, so nothing in the setup ever mentions a number: Slack takes a public channel
name (without the `#`), Discord a channel name (searched across your guilds), Matrix a
`#room:server` alias, and Google Chat a space display name:

> Every 2 minutes, run a gateway tick on the Slack channel eng, with pairing.

From then on, anything anyone types to the bot in the gatewayed chat is a prompt to
Sorcar, and Sorcar answers in the same chat. With pairing enabled, unknown senders receive
a one-time approval code in-channel (a sender allowlist is the stricter alternative);
on adapters that implement thread polling (Slack), follow-ups in the same thread
resume the same daemon chat. Gateway state — per-thread chat continuity, an
at-least-once delivery ledger with `(recovered reply)` redelivery, and a circuit
breaker that pauses the channel after repeated tick crashes (only errors that escape a
tick count) — persists next to the adapter's config (or under
`$KISS_HOME/third_party_agents/channel_state/`), and overlapping ticks exit
immediately thanks to a non-blocking per-channel lock.

Two caveats. Adapters that receive messages through an **embedded callback server**
(A2A, DingTalk, LINE, QQ, Synology Chat, Webhook, Weixin, Zalo) only receive events
*while a tick is running*, so schedule their ticks accordingly; polling-based adapters
(Slack, Telegram, email, ...) fetch history from the platform, so scheduled ticks
catch up on anything sent in between. And Discord is a partial exception: until a
first message has been captured, each tick looks back only about one second, so
messages that arrive between ticks in a quiet channel can be missed.

**20. What you can send *into* a gateway.** Once the gateway runs, the channel is a
full chat surface: any prompt in this document works there, replies land in the same
chat or thread, and thread follow-ups continue the same conversation. The email
gateway additionally drops no-reply and bulk/list mail before the agent sees it, so an
email autoresponder only answers real people. Per-channel defaults for gateway tasks
(`channel_model_name`, `channel_max_budget` in the adapter's `config.json`) let
routine chatter run on an inexpensive model — supported on all gateway adapters except
Slack (token-only store) and Google Chat (credential-only store).

**21. GitHub push → ntfy via a webhook route.** The webhook agent verifies inbound
HMAC-signed events and can push them straight through another channel's backend:

> Add a webhook route named "gh-push" with the github signature scheme, secret
> "rotate-me-7f3a", prompt template "Repo {repository.full_name}:
> {head_commit.message}", and deliver_module
> "kiss.agents.third_party_agents.ntfy_sea".

Include the (required, nonempty) secret in the prompt, then point GitHub's webhook at
`http://<host>:<port>/hook/gh-push`. `deliver_module` must be the full module path;
deliver-only routes push through that module's backend and return 502 on delivery
failure so GitHub retries. The embedded-server caveat above applies: the listener
accepts events only while a webhook gateway tick is running.

### Multi-account, budgets, and guardrails

**22. Two Slack workspaces, cleanly separated.** Each workspace keeps its own
credential set under `slack/<workspace>/` (legacy file `token.json`; vaulted
per-workspace under Muse). Name the workspace in the prompt:

> Using the acme Slack workspace, post the release notes to #general. Then do the same
> in the sideproj workspace.

Concurrent dispatches with different workspaces are serialized so one account's
credentials never leak into the other's session.

**23. Cheap model for chatter, big model on demand.** Gateway ticks use the
per-channel defaults from example 20; for a task that matters, pick the model and
budget in the chat surface's model picker and budget field — or say so in the prompt:

> Using claude-fable-5 with a $2 budget, analyze the last 200 messages in #incidents
> and post a post-mortem outline back to #incidents.

**24. Guardrails are config, not prompts** (Muse tip 4). Prefer the enforced switch
over an instruction: set `"read_only": "true"` in `github/config.json` for a
triage-only bot, keep Postgres in its default read-only mode, and grant Muse-auth
writes per service only when a workflow actually needs them:

> Grant the github service a single-use Muse write permission, then show me the recent
> audit records.

### Other surfaces

**25. Any OpenAI client as a Sorcar frontend.** After the one-time server setup (see
[Infrastructure](#infrastructure-two-extra-surfaces)), point Open WebUI, LibreChat, or
the `openai` SDK (with `base_url`) at the daemon and simply chat: the last user
message becomes a daemon task, system messages become the task's system prompt, and
the conversation-prefix hash maps each chat back to the same persistent daemon chat
across requests.

**26. Prompts from another agent over A2A.** A peer that fetches your agent card can
send your daemon a message such as "What does your project do?" via JSON-RPC
`message/send`; it queues as a prompt for the gateway runner (embedded-server caveat
above) and the peer polls for the reply. JSON-RPC POSTs require your bearer token when
one is configured, peer input is never executed, and the per-instance
20-turns-per-`contextId`-per-hour cap prevents two agents from ping-ponging forever.

### Sources

- Meta Muse product page: https://ai.meta.com/muse/
- Meta launch announcement: https://about.fb.com/news/2026/09/introducing-muse-personal-ai-agent/
- Meta Help Center, reminders and scheduled tasks with Muse: https://www.meta.com/help/artificial-intelligence/1484325780075655/
- The Neuron, "How to get started with Meta Muse": https://www.theneuron.ai/explainer-articles/how-to-get-started-with-meta-muse/
- Designs24hr, "How to Use Meta Muse AI for Everyday Tasks": https://www.designs24hr.com/how-to-use-meta-muse-ai/
- AI Agents Library, "Meta Muse AI: How to Use It": https://www.aiagentslibrary.com/blog/meta-muse-ai/
