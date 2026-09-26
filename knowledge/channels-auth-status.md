---
title: 'auth_status: connected/not-connected badges for the Apps panel'
uuid: 60558d5b-d7e7-4578-8b51-2232c648b77d
summary: 'auth_status subprocess for the Apps panel: probes every channel concurrently
  (20 s), reads Muse vault enrollments, then KISS_MUSE_AUTH=0 so probing does not
  touch the vault.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# `auth_status`: channel connection badges

The kiss-web sidebar "Apps" panel shows each channel as connected or not. Deciding that means
importing each channel module and constructing its agent, which is heavy and can hang on
network calls, so the daemon runs it as a **short-lived subprocess**
(`src/kiss/server/sidebar_panels.py` spawns `[sys.executable, "-m", "kiss.agents.third_party_agents.auth_status"]`).

## Output
One JSON array on stdout, one object per channel in name order:
`{"name": "slack", "label": "Slack", "authenticated": true, "error": ""}`.
`authenticated` is `null` when the agent could not be built or the probe timed out; `error`
carries `"<ExcType>: <msg>"` truncated to 300 chars.

## How it works
1. `main()` calls `vault_services()` first: the set of Muse vault enrollments
   (`client.enrolled_services()`), or `None` when Muse-auth is off or the daemon does not answer.
2. It then sets `os.environ["KISS_MUSE_AUTH"] = "0"` for the rest of the process. Building an
   agent in Muse mode mints a surrogate and migrates legacy plaintext credentials into the
   vault; a status check must not do that, so agents are built in legacy mode. Legacy mode is
   not strictly read-only: Slack's `_load_token` may mkdir/rename a legacy token file, and
   `load_google_credentials` refreshes an expired Google token and re-saves `token.json`.
3. `all_channel_statuses()` probes `available_channels()` on a `ThreadPoolExecutor`
   (`_MAX_WORKERS` = 8) with `PROBE_TIMEOUT_SECONDS` = 20 for the whole run.
4. `channel_status(name, enrolled)`: imports the module, finds the agent class with
   `agent_dispatch._agent_class`, maps the Muse service name (module `_SERVICE` attribute, e.g.
   Google agents; `_MUSE_SERVICES = {"brave": "brave_search"}`; else the channel name), and
   reports connected when the service is in the vault **or** `agent_cls()._is_authenticated()`.
5. Exits with `os._exit(0)` so a probe thread stuck in a hung constructor cannot keep the
   process alive.

Labels: `_BRAND_LABELS` for spellings that class-name splitting gets wrong (GitHub, iMessage,
LINE, Microsoft Teams, ...); otherwise `HomeAssistantAgent` → "Home Assistant".

## Adding a channel
Nothing to register: the probe uses `available_channels()`. If the Muse service name differs
from the channel name, set `_SERVICE` in the module or extend `_MUSE_SERVICES`.

## Sources
- `src/kiss/agents/third_party_agents/auth_status.py` (`main`, `vault_services`, `channel_status`, `all_channel_statuses`, `channel_label`, `_MUSE_SERVICES`, `_BRAND_LABELS`)
- `src/kiss/server/sidebar_panels.py`
