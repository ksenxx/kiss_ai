---
title: Managed OAuth brokers / click-Allow auth options for third-party agents (Sep
  2026)
uuid: 0c888bf4-38b2-4ce6-897b-21397bc1ec26
summary: Survey of Nango, Composio, Arcade, Pipedream Connect, Auth0 Token Vault,
  Klavis, Activepieces and MCP OAuth (DCR/CIMD) for letting users sign in and click
  Allow without registering OAuth apps.
created: '2026-09-26T19:48:37Z'
updated: '2026-09-26T19:48:37Z'
---
Question (Sep 2026): is there OSS/service so KISS third-party agents (src/kiss/agents/third_party_agents/) can auth by "user signs in, clicks Allow"?

Key constraint: someone must own a registered OAuth client per provider. Click-Allow without your own app exists only when a vendor lends its apps, or the provider supports MCP OAuth with Dynamic Client Registration / Client ID Metadata Documents.

- **Nango** (github.com/NangoHQ/nango, Elastic License, ~12k stars): 1,000+ APIs, Connect UI, token refresh, proxy. Free self-host = auth + proxy only, and you must register your own OAuth apps (callback <instance>/oauth/callback). Cloud is paid.
- **Composio** (SDK MIT, ~30k stars; backend is hosted SaaS): Composio-managed OAuth apps for dev (no setup), your own creds for prod. connected_accounts.link(user_id, auth_config_id) returns a redirect_url; wait_for_connection(). Rube MCP (rube.app/mcp) = OAuth2.1 with DCR.
- **Arcade.dev** (arcade-mcp MIT; engine is Cloud or Helm self-host): default Arcade OAuth apps only work with the Arcade user verifier (user must sign in to Arcade as a project member), so they suit dev or single-tenant use. Prod needs your own apps + a custom verifier (auth.confirmUser).
- **Pipedream Connect** (SaaS): defaults to Pipedream's own OAuth clients, but with those you cannot retrieve raw tokens (only actions + API proxy). A custom client lets you retrieve them.
- **Auth0 Token Vault** (commercial): Connected Accounts plus token exchange for Google/MS/Slack/GitHub/Box.
- **Klavis** (Apache-2.0, 5.8k, slowing): hosted MCP with OAuth; self-host means bringing your own tokens/apps.
- **Activepieces** CE (MIT core): env AP_CLOUD_AUTH_ENABLED=true (default) = "Use Activepieces-hosted OAuth2 apps for piece connections". This is the only self-hostable OSS found that gives click-Allow without registering apps, but it is a whole automation platform.
- **MCP auth spec** (modelcontextprotocol.io/specification/.../basic/authorization): client registration priority is pre-registered, then CIMD, then DCR, then ask the user. Official remote MCPs with DCR: Notion, Linear, Asana, Airtable, Zoom, Intercom, Sentry, Stripe, etc. Without DCR: Slack, GitHub, Atlassian, Box, Figma (source: github.com/jaw9c/awesome-remote-mcp-servers). No Gmail/Drive OAuth MCP.
- Bot-token channels (Telegram, Discord bots, WhatsApp, Signal, iMessage...) are not OAuth consent flows at all.
