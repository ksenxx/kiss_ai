---
title: KISS channel authentication flows after the KISS-owned-apps rewrite (Sep 2026)
uuid: 52c760aa-6e31-4d9b-ac44-94075d3f5428
summary: 'How third-party agents sign in now: KISS-owned public OAuth apps (device
  flow / PKCE loopback), Composio proxy for Google, MCP OAuth login for remote MCP
  servers; what was removed and known limits.'
created: '2026-09-26T21:46:46Z'
updated: '2026-09-26T21:46:46Z'
---
# Channel authentication flows (rewritten 2026-09-26)

Source of truth: `src/kiss/agents/third_party_agents/README.md` ("Authenticate by chatting") and `knowledge/channels-authentication-flows.md`.

## KISS-owned public OAuth apps (`_oauth_apps.py`)
- `KISS_OAUTH_CLIENT_IDS` table (github, msteams, slack, discord), `oauth_client_id(provider)` honours `KISS_<PROVIDER>_CLIENT_ID`. **The embedded IDs are still empty**: nobody has registered the vendor apps yet; until then the env var is required and `missing_client_id_error` is returned.
- GitHub + Teams: device flow (`DeviceFlowSession`). Teams tenant defaults to `organizations`; Teams works only with Muse-auth on. PAT / client-secret paths removed.
- Slack + Discord: `LoopbackPkceSession` in `_device_auth.py` (S256, public client, fixed redirect `http://localhost:53682/callback`, one session at a time, RLock port ownership, handler socket timeout 5 s). Slack yields a *user* token (xoxp; localhost redirect forbids bot scopes) with rotating refresh; channel mode ignores the signed-in user's own messages. Discord user token = identify+guilds+webhook.incoming; bot-only tools still need `authenticate_discord(bot_token=...)`; webhook URL stored 0600 in `discord/webhook/config.json` and posts bypass Muse.
- Muse `TOKEN_ENDPOINT_HOSTS` pins slack.com and discord.com.

## Google via Composio (`_composio_google.py`)
- `connected_accounts.link()` -> Connect Link; `finish_connect` waits on INITIATED/INITIALIZING/INACTIVE, re-reads state (superseded / already-recorded checks); state `<service>/composio.json`, API key `google/composio_api_key.json` or `COMPOSIO_API_KEY`.
- All calls go through `composio.client.without_retries.tools.proxy` (`ComposioSession` requests-shaped, `ComposioHttp` httplib2-shaped); JSON -> `body`, else base64 `binary_body` (Drive multipart uploads work); googleapiclient's `x-http-method-override: GET` form POST is rewritten back to GET.
- Google Chat has no Composio-managed app: needs `KISS_COMPOSIO_AUTH_CONFIG_GOOGLECHAT` or the service-account path. `auth_status` ignores stale Google vault enrollments.
- `composio>=0.24.0` dependency; `openai` locked at 2.54 (composio needs >=2.48; 3.x avoided). `google-auth-oauthlib` removed.

## Remote MCP servers (`sorcar/mcp_oauth.py`)
- `connect_mcp_server` / `finish_mcp_server_connect` Sorcar tools, CLI `python -m kiss.agents.sorcar.mcp_oauth <name>`; presets notion, linear, asana (v1 sse), zoom (needs `KISS_MCP_ZOOM_CLIENT_ID/_SECRET`, its AS allows no DCR/CIMD); redirect `http://localhost:53683/callback`; `KISS_MCP_CLIENT_METADATA_URL` for CIMD.
- `FileTokenStorage` stores `expires_at` and the AS `oauth_metadata`; expired tokens come back with an empty access_token so the SDK refreshes; `drop_stale_registration` removes registrations lacking the current redirect URI (older versions registered `localhost:0`). Known gap: a token that expires mid-run is not refreshed (SDK 1.23 keeps no expiry for loaded tokens).
- New tool names are in `_RESERVED_TOOL_NAMES`.

## Testing notes
- Slack/Discord/loopback tests bind fixed port 53682 and MCP tests 53683: run those files serially, others in parallel.
- Fake servers: `slack_oauth_test_utils.py`, Discord emulator in `test_discord_oauth.py`, `composio_test_utils.py` (COMPOSIO_BASE_URL), real FastMCP OAuth server in `test_mcp_oauth_login.py`.
