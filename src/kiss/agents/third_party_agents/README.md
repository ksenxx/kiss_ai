# Third-Party Agents

This package contains KISS Sorcar's **channel agents**: 43 `*_agent.py` modules plus the
`govee.py` CLI. Each module wraps one external service — a messaging platform (Slack,
Telegram, WhatsApp, ...), a service API (GitHub, Notion, PostgreSQL, ...), or a piece of
agent infrastructure (A2A, OpenAI-compatible server) — and exposes it as a set of
authenticated LLM tools plus a command-line gateway.

- [How a channel agent works](#how-a-channel-agent-works)
- [Five ways to use an agent](#five-ways-to-use-an-agent)
- [Common CLI flags](#common-cli-flags)
- [Credential isolation (Muse auth)](#credential-isolation-muse-auth)
- [Agent catalog](#agent-catalog)
  - [Messaging and device channels](#messaging-and-device-channels-32)
  - [Service APIs](#service-apis-9)
  - [Infrastructure agents](#infrastructure-agents-2)
  - [The govee.py CLI](#the-goveepy-cli)
- [Writing good tasks — tips from Meta Muse](#writing-good-tasks--tips-from-meta-muse)
- [26 examples and tips for combining agents](#26-examples-and-tips-for-combining-agents)

## How a channel agent works

A channel agent is **not** an executable agent itself. Every task is submitted to the
kiss-web daemon through the public `kiss.server.sorcar.run` API, and the daemon builds a
full chat agent with the standard tools (bash, file editing, browser automation). The
channel agent instance is the *carrier* of channel identity (see `BaseChannelAgent` in
`_channel_agent_utils.py`):

- Each module defines a `tools()` function. The daemon calls it to build the channel's
  tool list: the agent's **auth tools** (always present, e.g. `check_slack_auth`,
  `authenticate_slack`) plus, once authenticated, every public method of the module's
  `*ChannelBackend` class (e.g. `post_message`, `read_messages`, `search_messages`).
- Config and tokens live under `~/.kiss/third_party_agents/<service>/` (`$KISS_HOME`
  overrides `~/.kiss`). Because auth tools are always available, a not-yet-configured
  agent can walk you through authentication *in chat* — you never have to hand-edit
  `config.json` first.
- Backends whose platform has an inbound message stream also implement
  `_make_backend()`, which enables **poll mode** (`--channel`) and **cron delivery**
  (see below). API-only services (GitHub, Notion, the Google Workspace agents, ...)
  are outbound-only: their `--channel` poll mode is disabled.

## Five ways to use an agent

### 1. `run_agent` from a Sorcar session (recommended)

Inside any Sorcar task, dispatch a sub-task to a channel by name:

```
run_agent(agent="slack",    task="Post 'deploy done' to #eng")
run_agent(agent="telegram", task="Send 'dinner at 7' to chat 123456789")
run_agent(agent="Home Assistant", task="Turn off all kitchen lights")
```

Names are matched forgivingly — case, spaces, hyphens, and underscores are ignored, so
`"Home Assistant"`, `"nextcloud-talk"`, and `"phone control"` resolve to the
`homeassistant`, `nextcloud_talk`, and `phone_control` channels. The dispatched session
already carries the authenticated channel tools; its preamble tells it to use them
directly without exploring source code. The optional `workspace=` argument selects a
per-account credential set for multi-account channels, and `max_budget=` / `model_name=`
override the defaults. The two infrastructure modules (`a2a`, `openai_compat`) are
hidden from `run_agent`; use their CLIs instead.

### 2. One-shot CLI task

Every agent has a console script (installed by `pyproject.toml`):

```bash
kiss-slack    -t 'List all public channels in my workspace'
kiss-gmail    -t 'List my 5 most recent emails'
kiss-postgres -t 'List the ten most recent orders'
kiss-github   -t 'List the open issues in octocat/Hello-World'
```

The task runs through the kiss-web daemon with that channel's tools attached, then
prints run statistics. `-f FILE` reads the task from a file instead of `-t`.

### 3. Channel (poll) mode — a Hermes-style gateway

For the 26 poll-capable modules, `--channel` turns the CLI into a one-shot inbound
message processor: it fetches pending messages from the named channel/chat, runs a
daemon task per message, and replies in-channel.

```bash
kiss-telegram --channel 123456789                      # process pending messages once
kiss-slack    --channel eng --allow-users alice,bob    # only these senders
kiss-discord  --channel general --pairing              # unknown senders get a pairing code
kiss-discord  --channel general --list-pending         # list user_id:code pairs
kiss-discord  --channel general --approve 1f2e3d4c     # approve a pairing code (8 hex chars)
```

Poll mode persists Hermes-gateway state next to the adapter's config (or under
`$KISS_HOME/third_party_agents/channel_state/`): per-thread daemon-chat continuity, an
at-least-once delivery ledger with `(recovered reply)` redelivery, a circuit breaker
that pauses the channel after repeated transport failures, and DM pairing. Run it from
cron (or let the kiss-web daemon's scheduler tick it) to get an always-on channel bot;
overlapping ticks exit immediately thanks to a non-blocking per-channel lock.

One caveat: adapters that receive messages through an **embedded callback server**
(A2A, DingTalk, LINE, QQ, Synology Chat, Webhook, Weixin, Zalo) start that HTTP server
on `connect()` and stop it on `disconnect()`, so they only receive events **while a
tick is running**; a one-shot tick drains the in-memory queue and exits. Polling-based
adapters (Slack, Telegram, Discord, email, ...) fetch history from the platform, so
scheduled ticks miss nothing.

### 4. Python API

```python
from kiss.agents.third_party_agents.slack_agent import SlackAgent

agent = SlackAgent()                       # workspace="default"
result = agent.run(prompt_template="List all public channels in my workspace")
print(result)                              # YAML with 'success' and 'summary'
print(agent.budget_used, agent.total_steps)
```

`run()` submits the task through `kiss.server.sorcar.run` exactly like the CLI does.
Keyword arguments accepted include `model_name`, `max_budget`, `work_dir`, `timeout`,
and `use_worktree`.

### 5. Cron delivery target

The cron agent (`kiss-cron`, or `run_agent("cron", ...)` from a session) can deliver any
scheduled job's result to any module that defines `_make_backend()`:

```bash
kiss-cron --create "morning brief" --schedule "0 9 * * *" \
    --prompt "Summarize today's HN front page" --deliver telegram:123456789
```

Delivery targets look like `<channel>[:<chat>]` — `telegram:123`, `slack:eng`, `ntfy`.
Of the 32 messaging channels, 25 can receive deliveries (all poll-capable ones; `a2a`
is poll-capable but is infrastructure, and outbound-only modules such as `wecom` cannot
be delivery targets because delivery reuses the same `_make_backend()` factory). A
summary that is exactly `[SILENT]` (or `NO_REPLY`) suppresses delivery.

## Common CLI flags

All agents share the same argument parser (`_channel_cli.py`):

| Flag | Meaning |
| --- | --- |
| `-t, --task TEXT` | Task description (one-shot mode). |
| `-f, --file PATH` | Read the task from a file. |
| `-m, --model_name NAME` | LLM model (default: best model for the configured API keys). |
| `-b, --max_budget USD` | Max budget, a positive finite number. |
| `-w, --work_dir DIR` | Working directory (default: launch directory). |
| `-e, --endpoint URL` | Custom endpoint for a local model. |
| `--header 'Key:Value'` | Custom HTTP header; repeatable. |
| `-p, --parallel` / `--no-parallel` | Enable/disable parallel mode. |
| `--no-web` | Disable browser/web tools. |
| `--workspace WS` | Credential workspace for multi-account channels (default `default`). |
| `--channel CH` | Poll mode: process pending messages in this channel/chat. |
| `--allow-users A,B` | Poll mode: restrict to these senders. |
| `--pairing` | Poll mode: unknown DM senders get a one-time approval code. |
| `--approve CODE` / `--list-pending` | Approve / list pending pairing requests (need `--channel`). |
| `-V, --version` | Print version. |

Per-channel defaults: when `-m` / `-b` are omitted in poll mode, the
`channel_model_name` / `channel_max_budget` keys of the adapter's `config.json` apply
before the global defaults — on adapters with a module-level config store (all poll
adapters except Slack, whose store holds only workspace tokens).

## Credential isolation (Muse auth)

On Linux, credentials for the 24 Muse-supported services are isolated by default behind
a Meta-Muse-style security boundary implemented in the `muse_auth/` package: legacy
tokens auto-migrate into a vault owned by a local auth daemon on first use (a one-time
hand-off of the real credential through the agent process), after which ordinary
boundary-routed requests carry only opaque surrogate tokens that the daemon swaps for
the real ones at the network edge, and every boundary-routed request is host-allowlisted, classified
read vs. write, and checked against an allow/deny/ask policy with an audit log. Reads
are allowed by default; writes ask for a grant.

Covered services (`muse_auth/_common.py` `SERVICE_HOSTS`): the six Google services
(`gmail`, `google_calendar`, `google_docs`, `google_drive`, `google_sheets`,
`googlechat` — Google Chat's service-account mode excepted) plus `slack`, `github`,
`notion`, `discord`, `homeassistant`, `firecrawl`, `brave_search`, `ntfy`, `govee`,
`line`, `mattermost`, `msteams`, `nextcloud`, `synology`, `telegram`, `twitch`, `zalo`,
and `bluebubbles`. Other channels keep their legacy direct-credential path.

Manage the boundary with:

```bash
python -m kiss.agents.third_party_agents.muse_auth status   # daemon + enrollment state
python -m kiss.agents.third_party_agents.muse_auth enroll SERVICE       # enroll a credential
python -m kiss.agents.third_party_agents.muse_auth grant SERVICE write   # grant write access
python -m kiss.agents.third_party_agents.muse_auth audit                 # inspect the audit log
python -m kiss.agents.third_party_agents.muse_auth export SERVICE        # recover a vaulted credential
```

Opt out with `KISS_MUSE_AUTH=0` in `~/.kiss/api_keys.env`.

## Agent catalog

Config paths below are relative to `~/.kiss/third_party_agents/` (override the root
with `$KISS_HOME`; exception: Slack's workspace token store is hard-coded under
`~/.kiss`). "Poll" marks the modules with `_make_backend()` — usable with
`--channel` and as cron delivery targets. Tool names are the exact callables the agent
session sees; every agent also gets its auth tools (`check_<service>_auth`,
`authenticate_<service>`, `clear_<service>_auth`, plus service-specific browser-setup
helpers noted below).

### Messaging and device channels (32)

| Agent | CLI | Poll | Auth / config | Backend tools |
| --- | --- | --- | --- | --- |
| BlueBubbles (iMessage via a Mac server) | `kiss-bluebubbles` | yes | server URL + password, `bluebubbles/config.json` | `list_chats`, `get_chat`, `get_chat_messages`, `post_message`, `get_server_info`, `mark_chat_read` |
| DingTalk group robots | `kiss-dingtalk` | yes | robot webhook (+ optional `secret`, `outgoing_token`), `dingtalk/config.json` | `post_message`, `post_markdown` |
| Discord | `kiss-discord` | yes | bot token (also `start_discord_browser_auth`), `discord/config.json` | `list_guilds`, `list_third_party_agents` (channels), `get_channel`, `get_channel_messages`, `post_message`, `edit_message`, `delete_message`, `add_reaction`, `create_thread`, `list_guild_members`, `create_invite` |
| Email (any IMAP/SMTP mailbox) | `kiss-email` | yes | host/user/app-password, `email/config.json` | `send_email`, `list_unread_emails`, `read_email`, `mark_email_read` |
| Feishu / Lark | `kiss-feishu` | yes | `app_id` + `app_secret`, `feishu/config.json` | `send_text_message`, `reply_message`, `delete_message`, `list_messages`, `list_chats`, `get_chat`, `get_user_info` |
| Gmail | `kiss-gmail` | no | OAuth2 (`start_gmail_browser_setup`, `finish_gmail_auth`), token in `gmail/` | `get_profile`, `list_messages`, `get_message`, `send_email`, `reply_to_message`, `create_draft`, `trash_message`, `untrash_message`, `delete_message`, `modify_labels`, `list_labels`, `create_label`, `get_attachment`, `get_thread` |
| Google Chat | `kiss-gchat` | yes | service account or OAuth2 (`finish_googlechat_auth`), `googlechat/` | `list_spaces`, `get_space`, `list_members`, `list_messages`, `get_message`, `post_message`, `update_message`, `delete_message`, `create_space` |
| Home Assistant | `kiss-ha` | no | `base_url` + long-lived token, `homeassistant/config.json` | `ha_get_states`, `ha_call_service`, `ha_list_services`, `ha_get_history`, `ha_render_template`, `ha_fire_event` |
| iMessage (macOS AppleScript) | `kiss-imessage` | no | local Messages app, `imessage/config.json` | `send_imessage`, `send_attachment`, `list_conversations`, `get_messages` |
| IRC | `kiss-irc` | yes | server/nick (+ NickServ), `irc/config.json` | `connect_irc`, `join_irc_channel`, `leave_channel`, `post_message`, `send_notice`, `get_topic`, `set_topic`, `kick_user`, `whois`, `identify_nickserv` |
| LINE | `kiss-line` | yes | channel access token, `line/config.json` | `push_text_message`, `reply_message`, `get_profile`, `get_quota`, `leave_group`, `push_image_message` |
| Matrix | `kiss-matrix` | yes | homeserver + user + token (matrix-nio), `matrix/config.json` | `list_rooms`, `join_room`, `leave_room`, `send_text_message`, `send_notice`, `get_room_members`, `invite_user`, `kick_user`, `create_room`, `get_profile` |
| Mattermost | `kiss-mattermost` | yes | server URL + personal access token, `mattermost/config.json` | `list_teams`, `list_third_party_agents` (channels), `get_channel`, `list_channel_posts`, `create_post`, `delete_post`, `get_user`, `list_users`, `create_direct_message_channel`, `add_reaction` |
| Microsoft Teams | `kiss-msteams` | yes | Azure AD client credentials, `msteams/config.json` | `list_teams`, `get_team`, `list_third_party_agents` (channels), `list_channel_messages`, `post_channel_message`, `reply_to_message`, `list_chats`, `post_chat_message`, `list_team_members` |
| Nextcloud Talk | `kiss-nextcloud` | yes | server URL + username/password, `nextcloud/config.json` | `list_rooms`, `get_room`, `create_room`, `list_participants`, `list_messages`, `post_message`, `set_room_name`, `delete_message` |
| Nostr | `kiss-nostr` | no | private key + relays (pynostr), `nostr/config.json` | `publish_note`, `publish_reply`, `send_dm`, `get_profile`, `set_profile`, `list_relays`, `add_relay`, `remove_relay` |
| ntfy pub-sub | `kiss-ntfy` | yes | `topic` (+ optional `server`, `token`), `ntfy/config.json` | `publish_notification`, `poll_topic` |
| Phone control (Android companion app) | `kiss-phone` | yes | device IP + optional port/API key of the companion REST app, `phone/config.json` | `send_sms`, `make_call`, `end_call`, `list_sms_conversations`, `get_sms_messages`, `get_call_log`, `get_device_info`, `list_notifications`, `dismiss_notification`, `send_notification_reply` |
| QQ bot platform | `kiss-qq` | yes | app id/secret (Ed25519 webhook), `qq/config.json` | `send_group_message`, `send_c2c_message` |
| Signal (signal-cli) | `kiss-signal` | yes | registered signal-cli number, `signal/config.json` | `send_signal_message`, `receive_messages`, `send_attachment`, `list_contacts`, `list_groups` |
| SimpleX Chat | `kiss-simplex` | yes | local `simplex-chat -p 5225` WebSocket, `simplex/config.json` | `send_simplex_message`, `list_simplex_contacts`, `get_simplex_address` |
| Slack | `kiss-slack` | yes | bot token (also `start_slack_browser_auth`); `--list-workspaces` / `--delete-workspace WS` manage accounts; token in `slack/<workspace>/token.json` | `list_third_party_agents` (channels), `read_messages`, `read_thread`, `post_message`, `update_message`, `delete_message`, `list_users`, `get_user_info`, `create_channel`, `invite_to_channel`, `add_reaction`, `search_messages`, `set_channel_topic`, `upload_file`, `get_channel_info` |
| SMS / voice (Twilio) | `kiss-sms` | yes | account SID + auth token, `sms/config.json` | `send_sms`, `send_mms`, `list_messages`, `get_message`, `list_phone_numbers`, `get_account_info`, `send_whatsapp_message`, `create_call`, `list_calls`, `get_call`, `cancel_message` |
| Synology Chat | `kiss-synology` | yes | incoming/outgoing webhooks, `synology/config.json` | `post_message`, `send_file_message` |
| Telegram | `kiss-telegram` | yes | @BotFather bot token, `telegram/config.json` | `send_text`, `send_photo`, `send_document`, `edit_message_text`, `delete_message`, `pin_message`, `unpin_message`, `get_chat`, `get_chat_members_count`, `get_chat_member`, `ban_chat_member`, `unban_chat_member`, `get_updates`, `send_poll`, `forward_message` |
| Tlon / Urbit | `kiss-tlon` | no | Eyre HTTP server + code, `tlon/config.json` | `list_groups`, `list_third_party_agents` (channels), `get_messages`, `post_message`, `get_profile`, `poke`, `scry` |
| Twitch | `kiss-twitch` | no | OAuth2 tokens (Helix + twitchio chat), `twitch/config.json` | `get_stream_info`, `get_channel_info`, `get_user_info`, `get_chatters`, `send_chat_message`, `ban_user`, `search_third_party_agents` (channels), `get_clips`, `create_clip` |
| Webhook routes (inbound HMAC webhooks) | `kiss-webhook` | yes | `port` + `routes` map, `webhook/config.json` | `add_webhook_route`, `remove_webhook_route`, `list_webhook_routes` |
| WeCom group robots | `kiss-wecom` | no | robot webhook, `wecom/config.json` | `post_message`, `post_markdown` |
| Weixin / WeChat Official Accounts | `kiss-weixin` | yes | app id/secret + callback token, `weixin/config.json` | `send_text_message`, `get_user_info` |
| WhatsApp (personal, QR-paired bridge) | `kiss-whatsapp` | yes | whatsapp-mcp Go bridge (auth tools also: `start_whatsapp_bridge`, `get_whatsapp_qr_code`, `wait_for_whatsapp_pairing`, `stop_whatsapp_bridge`), `whatsapp/` | `search_whatsapp_contacts`, `list_whatsapp_chats`, `get_whatsapp_chat`, `get_whatsapp_direct_chat_by_contact`, `get_whatsapp_contact_chats`, `get_whatsapp_last_interaction`, `list_whatsapp_messages`, `get_whatsapp_message_context`, `send_whatsapp_message`, `send_whatsapp_file`, `send_whatsapp_audio_message`, `download_whatsapp_media` |
| Zalo Official Account | `kiss-zalo` | yes | OA access token, `zalo/config.json` | `send_text_message`, `send_image_message`, `get_follower_profile`, `get_followers`, `get_oa_info`, `get_recent_messages`, `get_conversation`, `upload_image` |

Platform notes: BlueBubbles and iMessage are macOS-only (BlueBubbles needs a Mac running
the BlueBubbles server; iMessage drives the local Messages app via `osascript`). The
email agent drops automated mail (no-reply senders, `Auto-Submitted`, `Precedence:
bulk/junk/list`) from the channel loop so it only answers real people. The webhook
agent verifies GitHub (`X-Hub-Signature-256`) or generic timestamped HMAC signatures,
caps bodies at 1 MB, suppresses duplicate deliveries, rate-limits to 60 events per
route per minute, and can either queue events as agent tasks or push them straight
through another channel's backend (`deliver_module` routes).

### Service APIs (9)

| Agent | CLI | Auth / config | Backend tools |
| --- | --- | --- | --- |
| Brave Search | `kiss-brave` | subscription token, `brave_search/config.json` | `brave_web_search`, `brave_news_search`, `brave_image_search`, `brave_video_search` |
| Firecrawl (scraping/crawling) | `kiss-firecrawl` | API key (+ optional self-hosted `base_url`), `firecrawl/config.json` | `firecrawl_scrape`, `firecrawl_map`, `firecrawl_search`, `firecrawl_start_crawl`, `firecrawl_get_crawl_status`, `firecrawl_cancel_crawl` |
| GitHub | `kiss-github` | personal access token (+ optional `read_only: "true"`), `github/config.json` | `gh_get_me`, `gh_search_repositories`, `gh_get_repository`, `gh_list_issues`, `gh_get_issue`, `gh_list_issue_comments`, `gh_search_issues`, `gh_search_code`, `gh_list_pull_requests`, `gh_get_pull_request`, `gh_get_pull_request_diff`, `gh_get_file_contents`, `gh_list_commits`, `gh_list_branches`, `gh_create_issue`, `gh_comment_on_issue`, `gh_update_issue`, `gh_create_pull_request`, `gh_merge_pull_request` |
| Google Calendar | `kiss-gcal` | OAuth2 quintet (`check_google_calendar_auth`, `authenticate_google_calendar`, `clear_google_calendar_auth`, `start_google_calendar_browser_setup`, `finish_google_calendar_auth`), `google_calendar/` | `gcal_list_calendars`, `gcal_list_events`, `gcal_get_event`, `gcal_create_event`, `gcal_update_event`, `gcal_delete_event`, `gcal_quick_add` |
| Google Docs | `kiss-gdocs` | OAuth2 quintet (as above, for `google_docs`), `google_docs/` | `gdocs_create_document`, `gdocs_read_document`, `gdocs_append_text`, `gdocs_replace_text`, `gdocs_insert_text`, `gdocs_batch_update`, `gdocs_list_documents` |
| Google Drive | `kiss-gdrive` | OAuth2 quintet (for `google_drive`), `google_drive/` | `gdrive_search_files`, `gdrive_get_file`, `gdrive_read_file`, `gdrive_download_file`, `gdrive_upload_file`, `gdrive_create_folder`, `gdrive_share_file`, `gdrive_move_file`, `gdrive_trash_file` |
| Google Sheets | `kiss-gsheets` | OAuth2 quintet (for `google_sheets`), `google_sheets/` | `gsheets_create_spreadsheet`, `gsheets_get_info`, `gsheets_get_values`, `gsheets_update_values`, `gsheets_append_values`, `gsheets_clear_values`, `gsheets_add_sheet`, `gsheets_batch_update`, `gsheets_list_spreadsheets` |
| Notion | `kiss-notion` | internal-integration token, `notion/config.json` | `notion_search`, `notion_get_page`, `notion_get_block_children`, `notion_append_paragraph`, `notion_append_blocks`, `notion_create_page`, `notion_update_page`, `notion_get_database`, `notion_query_database`, `notion_list_users`, `notion_create_comment`, `notion_get_comments` |
| PostgreSQL | `kiss-postgres` | `postgresql://` URI, `postgres/config.json` | `pg_query`, `pg_execute`, `pg_list_schemas`, `pg_list_tables`, `pg_describe_table`, `pg_list_indexes`, `pg_explain` |

All nine are outbound-only (no `--channel` poll mode). PostgreSQL defaults to
**read-only enforced server-side**: connections open with
`default_transaction_read_only=on` and `pg_query` uses the extended query protocol so
multi-statement strings are rejected; read paths run under a 60 s server-side
`statement_timeout` while `pg_execute` (write mode only) has none. GitHub's
`read_only: "true"` config key blocks every mutating `gh_*` tool.

### Infrastructure agents (2)

| Agent | CLI | What it does |
| --- | --- | --- |
| A2A (Agent-to-Agent protocol) | `kiss-a2a` | Outbound: `a2a_discover` (fetch a peer's agent card), `a2a_call` (JSON-RPC `message/send`), `a2a_get_task` (poll `tasks/get`). Inbound: embeds an HTTP server publishing this agent's card at `/.well-known/agent-card.json` and queueing peer messages for the channel runner. An optional bearer `token` gates inbound JSON-RPC POSTs (the agent card `GET` is public); a 20-messages-per-`contextId`-per-hour cap (per backend instance) stops ping-pong loops; inbound JSON-RPC requests land in `a2a_audit.jsonl`. Config: `a2a/config.json`. |
| OpenAI-compatible server | `kiss-oai` | Turns kiss-web into an OpenAI-style backend: unauthenticated `GET /v1/models` and `POST /v1/chat/completions` (requires Bearer `api_key`), so Open WebUI, LibreChat, or any `openai` SDK script becomes a chat surface for the daemon. Conversations map to persistent daemon chats via a hash of the message prefix (`chat_map.json`). Configure with `kiss-oai -t 'configure the OpenAI-compatible API server'`, then run `kiss-oai --serve`. Tool: `openai_compat_status`. |

Both are hidden from `run_agent` (they are infrastructure, not services a user asks
Sorcar to act on).

### The govee.py CLI

`govee.py` is a tiny standalone CLI for Govee smart lights (the Developer API), not a
channel agent — but it is Muse-auth covered and it is the preferred way to act on home
lights:

```bash
./govee.py list                             # show all devices
./govee.py state "Living room lamp"         # query current state
./govee.py on "Living room lamp"
./govee.py off "Living room lamp"
./govee.py brightness "Living room lamp" 40 # 1..100
./govee.py color "Living room lamp" ff8800  # hex RGB
./govee.py kelvin "Living room lamp" 4000   # color temperature
```

The key comes from `$GOVEE_API_KEY` and is enrolled into the Muse vault on first use;
device-state queries classify as reads, `/device/control` calls follow the write policy.

## Writing good tasks — tips from Meta Muse

Meta's Muse personal agent (launched September 8, 2026) popularized a set of habits for
delegating work to an agent that acts on your behalf. KISS's channel agents follow the
same trust architecture Muse does — a credential vault held by a separate daemon (day-to-day API calls see only
surrogate tokens; the `export` CLI can still read a vaulted credential back for
recovery), a Sentinel policy that allows reads and asks before writes, and an audit log — so
Muse's task-writing guidance transfers directly:

1. **Define the result *and* the stopping point.** Muse's documentation and every
   launch guide converge on this: a good task states the outcome and where the agent
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
   reversible preparation; keep the human at the decision point before anything sends,
   spends, books, submits, or deletes. The KISS Sentinel enforces exactly this split:
   reads proceed silently, writes ask — which also keeps approvals meaningful instead
   of becoming a button you tap without reading.
6. **Anchor schedules in plain language with exact times, and name them.** Muse's
   reminder help center uses "Remind me to call the dentist tomorrow at 10am" and
   "Every Monday at 9am, remind me to submit my timesheet", then cancels by name
   ("Cancel my dentist reminder"). Phrase `kiss-cron` jobs the same way: a name, an
   exact schedule, a bounded prompt, a delivery target.
7. **Meet people in the channel they already use.** Muse's biggest distribution bet is
   living inside WhatsApp rather than a new app. The KISS equivalent: deliver results
   where the audience already is (`--deliver telegram:...`, `slack:eng`, `ntfy`), and
   run poll-mode gateways on existing channels instead of inventing new inboxes.
8. **Verify in the real service, then review the trail.** Check the actual Slack
   channel / calendar / repository after the task, and use
   `python -m kiss.agents.third_party_agents.muse_auth audit` to see exactly which API
   calls were made.

## 26 examples and tips for combining agents

Everything below is real, runnable usage of the agents in this directory. Shell
examples use the CLIs; the same tasks also work as `run_agent("<channel>", "<task>")`
from a Sorcar session or as `Agent().run(prompt_template=...)` from Python — except
examples 25 and 26, whose infrastructure agents are hidden from `run_agent`.

### Getting started

**1. One-shot message.** The smallest useful task — one channel, one action, an
explicit target:

```bash
kiss-slack -t 'Post "deploy of v2.3 finished, all green" to #eng'
```

**2. Authenticate in chat, not in config files.** Auth tools are always present, so a
fresh agent can set itself up:

```bash
kiss-github -t 'Check my GitHub auth; if missing, walk me through creating a token,
store it, then list my open pull requests'
```

Each agent's `check_<service>_auth` returns setup instructions when unconfigured, and
Slack/Discord/Gmail can even drive the provider's console in the browser
(`start_slack_browser_auth`, `start_discord_browser_auth`, `start_gmail_browser_setup`).

**3. Probe read-only before you trust writes** (Muse tip 3). First task against a new
Postgres config:

```bash
kiss-postgres -t 'List schemas and tables, describe the orders table, and show the ten
most recent orders. Do not modify anything.'
```

The default read-only mode makes the "do not modify" clause server-enforced, not a
polite request.

**4. Long tasks from a file.** Keep reusable task briefs in files:

```bash
kiss-notion -f briefs/weekly-notes-cleanup.md
```

**5. Dispatch from a Sorcar session.** Inside any running task, don't import agent
modules — dispatch:

```
run_agent(agent="ntfy", task="Notify me that the benchmark finished: 42.3s, 0 failures")
```

### Cross-agent pipelines

The most powerful pattern: give **one task** that names two or three services, and let
the session use both tool sets (dispatch the combined task with `run_agent` to the
primary channel, or run the primary CLI with a task that calls `run_agent` for the
second service).

**6. GitHub → Slack standup digest.**

```bash
kiss-github -t 'Summarize the pull requests merged into acme/api in the last 24 hours
(title, author, one-line summary each), then use run_agent to post the digest to the
Slack channel #standup'
```

**7. Postgres → Google Sheets weekly report.**

```bash
kiss-postgres -t 'Query the orders table for the last 7 days: one row per day with
order count and revenue (read-only). Then use run_agent("google_sheets", ...) to
append the rows to the "Weekly KPIs" spreadsheet, one row per day.'
```

**8. Firecrawl → Notion research capture.**

```bash
kiss-firecrawl -t 'Scrape https://example.com/pricing and its /docs subpages
(firecrawl_map first, then scrape the top 5 relevant pages). Then run_agent("notion",
...) to create a page "Competitor pricing — example.com" under my Research page with a
structured summary and the source URLs.'
```

**9. Brave Search → Email newsletter.**

```bash
kiss-brave -t 'Find the 5 most significant news stories about RISC-V from the past
week (brave_news_search). Then run_agent("email", ...) to send a plain-text digest
with links to team@acme.dev. Show me the draft in the summary before sending.'
```

**10. Gmail → Google Calendar.** Read one service, write another, keep the stop point
explicit (Muse tips 1 and 5):

```bash
kiss-gmail -t 'Find emails from the last 3 days that propose meetings. For each,
extract the proposed time and attendees, then run_agent("google_calendar", ...) to
create an event for each. List every event you created in the summary so I can verify.'
```

**11. Home Assistant + Govee evening scene.** Two device backends, one instruction:

```bash
kiss-ha -t 'Set the living room to movie mode: turn off ceiling lights via Home
Assistant, then run bash: ./src/kiss/agents/third_party_agents/govee.py brightness
"Living room lamp" 15 && ./src/kiss/agents/third_party_agents/govee.py kelvin
"Living room lamp" 2700'
```

(`govee.py` is a plain CLI, so any agent session with bash can call it; the Muse write
policy still gates `/device/control`.)

**12. Twitch → Discord stream announcement.**

```bash
kiss-twitch -t 'Get stream info for channel "mychannel". If live, run_agent("discord",
...) to post "We are live: <title> — <url>" to the #announcements channel; if offline,
do nothing and say [SILENT] in your summary.'
```

**13. Google Drive backup, link shared to Mattermost.**

```bash
kiss-gdrive -t 'Upload ./reports/q3-summary.pdf to the "Team Reports" folder, share it
read-only with team-lead@acme.dev, then run_agent("mattermost", ...) to post the file
link to the town-square channel.'
```

**14. Slack thread → Google Docs minutes.**

```bash
kiss-slack -t 'Read the full thread at #planning ts 1726300000.000100, then
run_agent("google_docs", ...) to create a doc "Planning sync 2026-09-14" with
decisions, action items (owner + due date), and open questions.'
```

**15. Phone → Email SMS triage.**

```bash
kiss-phone -t 'List SMS conversations with unread messages from today, summarize each
in one line, and run_agent("email", ...) to mail the triage list to me@example.com.
Do not reply to any SMS.'
```

### Always-on gateways and schedules

**16. A Telegram bot in one cron line.** Poll mode is a one-shot tick, so an always-on
bot is just a schedule:

```bash
# every 2 minutes, process pending messages in the group
*/2 * * * * kiss-telegram --channel=-1001234567890 --pairing
```

With `--pairing`, unknown senders receive a one-time code; approve with
`kiss-telegram --channel=-1001234567890 --approve CODE`. On adapters that implement
thread polling (Slack), follow-ups in the same thread resume the same daemon chat.

**17. Scheduled brief delivered where you already read** (Muse tips 6 and 7):

```bash
kiss-cron --create "morning brief" --schedule "0 9 * * *" \
    --prompt "Summarize today's HN front page in 10 bullets" \
    --deliver telegram:123456789
```

Or conversationally from a session: `run_agent("cron", "Every weekday at 9am, summarize
today's HN front page and deliver it to telegram chat 123456789")`.

**18. Silent health check — alert only on failure.** `[SILENT]` suppresses delivery,
so a check that passes stays quiet:

```bash
kiss-cron --create "api healthcheck" --schedule "*/15 * * * *" \
    --prompt "GET https://api.acme.dev/health with curl. If HTTP 200, reply [SILENT].
Otherwise describe the failure." \
    --deliver ntfy
```

**19. GitHub push → ntfy via a webhook route.** The webhook agent verifies inbound
HMAC-signed events and can push them straight through another channel's backend:

```bash
kiss-webhook -t 'Add a webhook route named "gh-push" with the github signature scheme,
secret from my input, prompt template "Repo {repository.full_name}: {head_commit.message}",
and deliver_module "kiss.agents.third_party_agents.ntfy_agent". Then tell me the URL
to configure on GitHub.'
```

Point GitHub's webhook at `http://<host>:<port>/hook/gh-push`. `deliver_module` must be
the full module path; deliver-only routes push through that module's backend and return
502 on delivery failure so GitHub retries. Remember the embedded-server caveat above:
the listener accepts events only while a poll tick is running, so schedule
`kiss-webhook --channel <route>` ticks accordingly.

**20. Email autoresponder for humans only.** The email backend drops no-reply and
bulk/list mail before the agent sees it:

```bash
*/5 * * * * kiss-email --channel INBOX --allow-users boss@acme.dev,client@example.com
```

### Multi-account, budgets, and guardrails

**21. Two Slack workspaces, cleanly separated.** Tokens live under
`slack/<workspace>/token.json`:

```bash
kiss-slack --list-workspaces                  # show configured workspaces
kiss-slack --workspace acme     -t 'Post the release notes to #general'
kiss-slack --workspace sideproj -t 'Post the release notes to #general'
kiss-slack --delete-workspace sideproj        # remove a workspace's token
```

From a session: `run_agent(agent="slack", task=..., workspace="acme")`. Concurrent
dispatches with different workspaces are serialized so one account's credentials never
leak into the other's session.

**22. Cheap model for chatter, big model on demand.** Set per-channel defaults in the
adapter's `config.json` (`channel_model_name`, `channel_max_budget` — e.g. in
`telegram/config.json`; Slack's token-only store does not support these keys) so
gateway ticks run on an inexpensive model, and override per task when it matters:

```bash
kiss-slack -m claude-fable-5 -b 2.0 -t 'Deep-dive: analyze the last 200 messages in
#incidents and write a post-mortem outline'
```

**23. Guardrails are config, not prompts** (Muse tip 4). Prefer the enforced switch
over an instruction: set `"read_only": "true"` in `github/config.json` for a triage-only
bot, keep Postgres in its default read-only mode, and grant Muse-auth writes per
service only when a workflow actually needs them:

```bash
python -m kiss.agents.third_party_agents.muse_auth grant github write  # per-service write grant
python -m kiss.agents.third_party_agents.muse_auth audit                # verify what actually ran
```

**24. Restrict who can talk to a gateway.** `--allow-users` resolves names via the
backend and drops everyone else; combine with `--pairing` for a controlled onboarding
flow instead of an open bot.

### Agent-to-agent and custom frontends

**25. Talk to another agent over A2A.** Machine A publishes its agent card while its
channel runner is ticking (embedded-server caveat above); from machine B:

```bash
kiss-a2a -t 'Discover the agent at http://machine-a:8710 (a2a_discover), then a2a_call
it with "What does your project do?", passing token="<machine A bearer token>", and
poll a2a_get_task with the same token until the reply completes.'
```

Peer input is treated as untrusted text (queued, never executed), JSON-RPC POSTs
require machine A's bearer token when one is configured, and the per-instance
20-turns-per-`contextId`-per-hour cap prevents two agents from ping-ponging forever.

**26. Any OpenAI client as a Sorcar frontend.**

```bash
kiss-oai -t 'configure the OpenAI-compatible API server'   # set port + api_key
kiss-oai --serve                                           # run the server
```

Then point Open WebUI/LibreChat (or the `openai` SDK with `base_url`) at it: the last
user message becomes a daemon task, system messages become the task's system prompt,
and the conversation-prefix hash maps each chat back to the same persistent daemon
chat across requests.

### Sources

- Meta Muse product page: https://ai.meta.com/muse/
- Meta launch announcement: https://about.fb.com/news/2026/09/introducing-muse-personal-ai-agent/
- Meta Help Center, reminders and scheduled tasks with Muse: https://www.meta.com/help/artificial-intelligence/1484325780075655/
- The Neuron, "How to get started with Meta Muse": https://www.theneuron.ai/explainer-articles/how-to-get-started-with-meta-muse/
- Designs24hr, "How to Use Meta Muse AI for Everyday Tasks": https://www.designs24hr.com/how-to-use-meta-muse-ai/
- AI Agents Library, "Meta Muse AI: How to Use It": https://www.aiagentslibrary.com/blog/meta-muse-ai/
