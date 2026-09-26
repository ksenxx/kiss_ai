---
title: 'Remote MCP servers: URLs and OAuth registration support (probed Sep 2026)'
uuid: fa98459e-d5f6-44a2-9d1f-370908b6e556
summary: Live-probed MCP endpoints for Notion, Linear, Asana, Zoom and whether their
  auth servers allow DCR/CIMD public clients; plus Slack/Discord/Composio OAuth facts.
created: '2026-09-26T20:11:35Z'
updated: '2026-09-26T20:11:35Z'
---
Probed 2026-09-26 with curl (POST to URL -> 401 WWW-Authenticate resource_metadata; then AS /.well-known/oauth-authorization-server).

| Service | MCP URL | transport | AS | DCR | CIMD | public client |
|---|---|---|---|---|---|---|
| Notion | https://mcp.notion.com/mcp | http | mcp.notion.com | yes | yes | yes (none) |
| Linear | https://mcp.linear.app/mcp | http | mcp.linear.app | yes | yes | yes |
| Asana v1 | https://mcp.asana.com/sse | sse | mcp.asana.com | yes | no | yes |
| Asana v2 | https://mcp.asana.com/v2/mcp | http | app.asana.com | NO | no | secret only |
| Zoom | https://mcp.zoom.us/mcp/zoom/streamable | http | zoom.us | NO | no | client_secret_basic only |

Zoom and Asana v2 need a pre-registered OAuth app (client id + secret), so no click-Allow without an owned app.

Other facts (see tmp/information-mcpoauth.md in the MCP OAuth task):
- Slack PKCE GA 2026-03-30: public client, no secret; localhost redirect => user scopes only; rotating refresh tokens (30-day refresh expiry). Initial oauth.v2.access nests user token under `authed_user`.
- Discord: "Public Client" toggle + PKCE lets token exchange omit secret; user bearer tokens cannot send/read messages (bot token needed); `webhook.incoming` scope returns a channel webhook.
- Composio: tokens are masked in connected_accounts; call provider APIs via `composio.tools.proxy(endpoint, method, body, connected_account_id, parameters=[{name,type,value}])`; `connected_accounts.link(user_id, auth_config_id)` for managed OAuth (initiate() retired for managed auth). composio 0.24 needs openai>=2.48.
- MCP SDK 1.23.3 OAuthClientProvider supports client_metadata_url (CIMD), DCR, pre-registered via storage.get_client_info.
