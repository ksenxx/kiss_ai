---
title: Where channel credentials and settings live (ChannelConfig, KISS_HOME)
uuid: 4764c8da-02a2-49fc-8bb4-e90f347329f0
summary: 'Where channel credentials live: $KISS_HOME/third_party_agents/<service>/config.json,
  ChannelConfig load/save/scrub_secrets, 0600 atomic writes, config_file_lock.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Where channel credentials and settings live

## Layout
- Root: `$KISS_HOME/third_party_agents/<service>/` (`$KISS_HOME` defaults to `~/.kiss`).
  Most channels keep one `config.json`; Google services keep `credentials.json` (OAuth client)
  and `token.json` (user token); Slack keeps `slack/<workspace>/token.json` under a path
  hard-coded to `Path.home()/.kiss`.
- Gateway state files sit next to `config.json` (see `channels-gateway-state-file`).
- On Muse-capable platforms (Linux) secrets of the 24 covered services move into
  `$KISS_HOME/muse_auth/vault/<service>.json` on first use (see `muse-auth-architecture`);
  non-secret settings, Google's `credentials.json`, and inbound-verification secrets (LINE's
  `channel_secret`) stay in the service directory.

## `ChannelConfig(channel_dir, required_keys)`
Replaces per-module `_config_path/_load/_save/_clear` boilerplate.
- `path` is resolved lazily: a dir under the default `~/.kiss` is rebased onto
  `kiss_home()`, so setting `KISS_HOME` (the test suite uses a per-process temp dir) isolates
  configs from the user's real ones.
- `load()` → dict or `None` when missing, malformed, or any required key is empty.
- `load_metadata()` → dict without required-key enforcement. Needed because Muse migration
  removes the secret key while `base_url`, `channel_model_name`, `channel_max_budget` survive;
  `load()` would report such a file invalid. `channel_override_config` uses it.
- `save(data)` / `clear()` → `save_json_config` / `clear_json_config`, both under
  `config_file_lock(path)`.
- `scrub_secrets(secret_keys)` — removes vault-migrated keys; deletes the file if nothing
  else remains. Runs the whole read-filter-write under the lock, using raw
  `write_private_file`/`unlink` inside because the lock is not reentrant.

## Write safety
`write_private_file` → `kiss.core.utils.atomic_write_text(mode=0o600)`: staged in a
`mkstemp` sibling and renamed, so readers never see a torn file and secrets are never briefly
world-readable (on Windows the rename waits out a concurrent reader). `config_file_lock` is a
`<name>.lock` sibling file lock shared across threads and processes; never call
`save_json_config` while holding it (deadlock: it is not reentrant across opens).

## Google Workspace specifics
`_google_workspace_utils.credentials_path(service)` falls back from
`<service>/credentials.json` to `google/credentials.json` to `gmail/credentials.json`, so one
Google Cloud OAuth client can serve every Google agent. With Muse-auth off,
`save_google_credentials` writes `token.json` with 0600 and `load_google_credentials` loads it;
with Muse-auth on, both use the vault (a leftover `token.json` is migrated into it).

## Sources
- `src/kiss/agents/third_party_agents/_channel_agent_utils.py` (`ChannelConfig`, `load_json_config`, `save_json_config`, `clear_json_config`, `config_file_lock`, `write_private_file`, `channel_override_config`)
- `src/kiss/agents/third_party_agents/_google_workspace_utils.py` (`google_service_dir`, `token_path`, `credentials_path`, `save_google_credentials`)
- `src/kiss/agents/third_party_agents/README.md` ("How a channel agent works")
