---
title: Loopback HTTP server port release and MCP OAuth e2e testing lessons
uuid: 80be3789-130c-4a70-92ff-954c0f7f9ebd
summary: Closing an http.server socket from another thread while handle_request blocks
  in select keeps the port bound; MCP SDK 1.23 never refreshes tokens loaded from
  storage; how to build a real OAuth-protected FastMCP server and a fake Composio
  API for tests.
created: '2026-09-26T20:41:23Z'
updated: '2026-09-26T20:41:23Z'
---
# Lessons (verified 2026-09-26, KISS repo)

## Port release for loopback redirect servers
- On Linux, `server.server_close()` called from thread A while thread B is inside
  `server.handle_request()` (blocked in select/poll with `timeout=0.25`) does NOT free the
  port right away: the in-flight poll holds a kernel reference until it returns. A new
  bind on the same port fails with `EADDRINUSE`.
- Fix used in `src/kiss/agents/third_party_agents/_device_auth.py` (`LoopbackPkceSession`):
  the serve thread closes its own socket in a `finally`, and `_close_server()` sets the stop
  condition and then `join()`s the serve thread.

## MCP SDK 1.23 OAuth client
- `OAuthClientProvider._initialize` loads tokens from storage without setting
  `token_expiry_time`. An expired stored access token is sent as-is, and on a 401 the SDK
  runs a full browser authorization instead of refreshing. KISS's `FileTokenStorage` keeps
  no expiry, so a non-interactive `build_oauth_provider` cannot refresh. This is recorded
  as a strict xfail in `src/kiss/tests/agents/sorcar/test_mcp_oauth_login.py`.
- anyio wraps transport errors in `ExceptionGroup`. Flatten them with
  `mcp_servers.describe_exception`, or the real message (such as the login hint) is lost.

## Test harness patterns
- Real OAuth MCP server: `FastMCP(name, host, port, auth_server_provider=<in-memory provider>,
  auth=AuthSettings(issuer_url=base, resource_server_url=base+"/mcp",
  client_registration_options=ClientRegistrationOptions(enabled=True)))`, then
  `uvicorn.Server(Config(mcp.streamable_http_app(), ...))` on a thread (poll `.started`).
  `authorize()` returns `construct_redirect_uri(redirect_uri, code=..., state=...)`, so
  `httpx.get(auth_url, follow_redirects=True, trust_env=False)` plays the browser.
- Composio SDK: set `COMPOSIO_BASE_URL` to a local server. Paths:
  `/api/v3.1/auth_configs` (GET list, POST create -> `{"auth_config": {...}}`),
  `/api/v3.1/connected_accounts` (GET list; `link()` lists first),
  `/api/v3.1/connected_accounts/link` (POST -> `connected_account_id`, `redirect_url`),
  `/api/v3.1/connected_accounts/{id}` (GET status, DELETE), and
  `/api/v3.1/tools/execute/proxy`. Response validation is non-strict, so partial JSON is
  accepted. `wait_for_connection` polls once per second.
