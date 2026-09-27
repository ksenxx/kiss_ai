---
title: Slack click-Allow sign-in in KISS (PKCE, user tokens)
uuid: 891155ca-2f26-477f-a560-678f1954626e
summary: How slack_sea.py authenticates via the KISS-owned Slack app with PKCE, token
  storage/refresh, env vars, and test helpers.
created: '2026-09-26T20:35:31Z'
updated: '2026-09-26T20:35:31Z'
---
- `src/kiss/agents/third_party_agents/slack_sea.py`: `authenticate_slack()` (no args) -> `LoopbackPkceSession('slack', PkceProvider(<base>/oauth/v2/authorize, <base>/api/oauth.v2.access), oauth_client_id('slack'), LOOPBACK_REDIRECT_URI, {'user_scope': _USER_SCOPES})`; `finish_slack_auth()` flattens `authed_user` into TokenGrant, validates with auth.test BEFORE storing (failed sign-in keeps the old credential).
- Localhost redirect => Slack grants only USER scopes (xoxp token, rotating ~12h, refresh via oauth.v2.access grant_type=refresh_token + client_id, no secret).
- Env: `KISS_SLACK_BASE_URL` (default https://slack.com) drives authorize, token and Web API base; `KISS_SLACK_CLIENT_ID` overrides the embedded client ID.
- Muse mode: vault `oauth2_refresh_token` credential (daemon refreshes, 60s margin). Legacy mode: token.json holds access_token/refresh_token/expires_at/client_id as STRINGS (load_json_config stringifies); `_RefreshingWebClient.api_call` refreshes 300s before expiry.
- Poll-mode caveat: `is_from_bot` compares against auth.test user_id, which is now the signed-in human, so their own messages are ignored in channel mode.
- Tests: `tests/agents/third_party_agents/slack_oauth_test_utils.py` (SlackOAuthState, SlackApiServer, sign_in replays the redirect to 127.0.0.1:53682), `test_slack_pkce_auth.py`, Slack sections of `test_muse_auth_channels.py`.
- Gotcha: port 53682 is fixed; concurrent pytest runs from other agents cause EADDRINUSE. Wait until `ss -ltn | grep :53682` is empty. A blocker socket in tests needs SO_REUSEADDR because callbacks leave TIME_WAIT entries.
