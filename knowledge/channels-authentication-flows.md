---
title: 'Channel authentication flows: check/authenticate/finish tools, device grant,
  Google consent, portal hand-off'
uuid: 90a5e742-3c7e-4456-803d-84cf1c95ce98
summary: 'Channel sign-in in chat: check/authenticate/finish tools, RFC 8628 device
  grant (GitHub, Twitch, Teams, Matrix), Nextcloud flow, Google consent, portal token
  paste-back.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Channel authentication flows

Auth tools are **always** in a channel's tool list (`BaseChannelAgent._get_tools`), so an
unconfigured channel sets itself up in conversation. Standard names: `check_<svc>_auth`
(returns status or setup instructions), `authenticate_<svc>`, `clear_<svc>_auth`, plus
`finish_<svc>_auth` and `start_<svc>_browser_setup` / `start_<svc>_browser_auth` where the flow
needs them. Rule in every flow: the agent never drives the sign-in page and never asks for a
password or 2FA code; the user approves in their own browser.

## 1. Connect-style (poll-based consent) — `_device_auth.py`
Used by GitHub, Twitch, Microsoft Teams, Matrix (RFC 8628 device grant; each module builds a
`DeviceFlowProvider(device_url, token_url, scope_param, token_scope_param)`; Twitch spells the
field `scopes`) and Nextcloud (`NextcloudLoginSession`, Login Flow v2, no client registration).
`authenticate_<svc>()` starts a `ConsentSession` subclass (`DeviceFlowSession`), opens the URL
in the user's browser if possible, and returns the URL plus user code
(`consent_required` / `consent_instructions`). A background thread polls the provider;
`finish_<svc>_auth()` answers `pending` until approval, then enrolls the credential
(`TokenGrant` maps the token response onto a Muse vault credential: refresh-token kind when a
refresh token is issued, otherwise bearer). When the OAuth client id is missing (GitHub,
Twitch, Teams) the tool opens the provider's app-registration page instead. GitHub's client id
may come from `oauth_client_id` in `github/config.json` or `$KISS_GITHUB_CLIENT_ID`.
Signal (`signal-cli link` QR code) and WhatsApp (`start_whatsapp_bridge`,
`get_whatsapp_qr_code`, `wait_for_whatsapp_pairing`) are QR variants.

## 2. Google consent hand-off — `_google_workspace_utils.py`
Gmail, Calendar, Drive, Docs, Sheets, Google Chat. `make_google_auth_tools` builds the
standard set. `authenticate_<svc>` → `start_google_consent` starts a `RemoteOAuthSession`
(loopback redirect server in a background thread) and returns `remote_oauth_instructions`.
Same-machine approval completes by itself; approval on another device ends at an unreachable
`http://localhost:PORT/?state=...&code=...` URL that the user pastes back and the agent replays
locally (e.g. with `curl`). `start_<svc>_browser_setup` opens the Google Cloud Console page for
creating the OAuth client (`credentials.json`). In Muse mode API calls go through
`google_api_session` (the daemon boundary).

## 3. Portal hand-off with token paste-back
Slack (`start_slack_browser_auth`), Discord (`start_discord_browser_auth`), and API-key
channels (Brave, Notion, Firecrawl, Twilio SMS, LINE, Feishu, QQ, Weixin, Zalo, Telegram):
`_browser_handoff.portal_handoff` opens the developer portal; the user creates the app and
pastes the token back; the agent stores it with `authenticate_<svc>(...)` (e.g.
`authenticate_telegram(bot_token=...)`). Overleaf uses the `overleaf_session2`
cookie the same way.

## Browser hand-off helper
`open_in_default_browser(url)` honours `$BROWSER` (`webbrowser` convention, `%s` placeholder,
`os.pathsep`-separated list), else `open` (macOS), `os.startfile` (Windows), `xdg-open`.
Returns `False` without raising when headless (no display, Docker, or `KISS_HEADLESS=1` via
`_backend_utils.is_headless_environment`). `browser_handoff_note` tells the agent what
happened so it always shows the URL in chat as a fallback.

## Gotcha
Tools are snapshotted at session start: after a successful `finish_*`/`authenticate_*`, the
backend tools appear only in the **next** session.

## Sources
- `src/kiss/agents/third_party_agents/_device_auth.py` (`DeviceFlowProvider`, `ConsentSession`, `DeviceFlowSession`, `NextcloudLoginSession`, `TokenGrant`, `consent_required`, `connect_prompt`)
- `src/kiss/agents/third_party_agents/_google_workspace_utils.py` (`make_google_auth_tools`, `start_google_consent`, `RemoteOAuthSession`, `remote_oauth_instructions`, `google_api_session`)
- `src/kiss/agents/third_party_agents/_browser_handoff.py` (`open_in_default_browser`, `browser_handoff_note`, `portal_handoff`)
- `src/kiss/agents/third_party_agents/_backend_utils.py` (`is_headless_environment`)
- `src/kiss/agents/third_party_agents/README.md` ("Authenticate by chatting")
