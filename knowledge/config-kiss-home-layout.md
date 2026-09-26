---
title: KISS_HOME and the ~/.kiss directory layout
uuid: f3786e09-1766-4827-b94b-1b8f5e90fc66
summary: 'kiss_home() is $KISS_HOME or ~/.kiss; layout: config.json, api_keys.env,
  sorcar.db, sorcar.sock, tabs.json, memories, MODEL_INFO.json, skills, mcp.json,
  cron; import-time path gotchas'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# KISS_HOME and the ~/.kiss directory layout

All per-user state lives under one directory, returned by `kiss.core.config.kiss_home()`:
`$KISS_HOME` when set, else `~/.kiss`. It is resolved on every call. Setting `KISS_HOME` redirects
config, keys, DB and daemon socket together, which is how the test suite and
`src/kiss/scripts/cost_levers_experiment.py` isolate themselves from the real home.

## Entries (path relative to KISS_HOME, owner)
| Entry | Purpose / owner |
|---|---|
| `config.json` (+ `.config.lock`) | user settings, `vscode_config.save_config` (see `config-vscode-config-json`) |
| `api_keys.env` (+ `.api_keys.env.kiss.lock`) | canonical key store (see `config-api-keys-store`) |
| `sorcar.db` (+ `-wal`, `-shm`) | SQLite task history and events, `agents/sorcar/persistence.py` |
| `sorcar.sock` | daemon Unix socket (`KISS_SORCAR_SOCK` overrides), `daemon_client`, `web_server` |
| `tabs.json` | tab registry, `server/tab_registry.py` |
| `memories/` | default persistent memory dir unless `memory_dir` is set (see `memory-when-enabled`); holds `*.md` pages and `<embedder>.sqlite3` indexes |
| `MODEL_INFO.json` | installed model catalog, seeded by the installer (`model_info.user_model_info_path`) |
| `MY_MODELS.json` | user's own model entries (may hold API keys) |
| `SEAS.md` | user agent-command registry, `agents/sorcar/sea_commands.py` |
| `SORCAR.md` | read by `relentless_agent` |
| `MODEL_DECISIONS.md` | autoroute ledger, `agents/seas/autoroute_sea.py` |
| `task_classifier_cache.json` | `agents/sorcar/task_classifier.py` |
| `MY_TASK_TEMPLATES.md`, `MY_INJECTION.md` | welcome chips and inject-trick panel, `server/user_assets.py`, `server/tricks.py` |
| `skills/<name>/SKILL.md` | user skills, `agents/sorcar/skills.py` |
| `mcp.json`, `mcp_auth/` | user MCP servers and their auth, `agents/sorcar/mcp_servers.py` |
| `cron/` (and `cron/work`) | cron agent state and job notes, `agents/sorcar/cron_agent.py` |
| `agent_work/`, `channel_work/` | default work dirs for dispatched agents, `agent_dispatch.py` |
| `browser_profile/` | Chromium profile for web tools, `web_use_tool.py` |
| `models/` | voice-wake model files, `server/voice_wake.py` |
| `third_party_agents/`, `muse_auth/`, `connectors/` | third-party agent state |
| `task-owners/<token>.lock` | per-task owner locks, `persistence.py` |

## Gotchas
- Never cache `kiss_home()` in a module-level constant: it would freeze the path at import and
  ignore a later `KISS_HOME` (tests set it in conftest). Three known exceptions exist today:
  `persistence._KISS_DIR`/`_DB_PATH` are computed at import (tests and daemon restarts reassign
  `_DB_PATH`; `_current_db_path()` stamps async writes with the DB they belong to), and
  `model_info.USER_MY_MODELS_PATH` is hard-coded to `Path.home() / ".kiss" / "MY_MODELS.json"`, so it
  does not follow `KISS_HOME`, and `slack_sea._SLACK_DIR` hard-codes
  `Path.home() / ".kiss" / "third_party_agents" / "slack"` for Slack workspace tokens.
- `vscode_config.CONFIG_DIR`/`CONFIG_PATH` are lazy module attributes; tests may assign them.
- Files holding secrets are written 0600 (`atomic_write_text` default); see
  `config-atomic-writes-and-locks`.

## Sources
- `src/kiss/core/config.py` (`kiss_home`)
- `src/kiss/core/vscode_config.py` (`_config_dir`, `api_keys_env_path`)
- `src/kiss/agents/sorcar/sorcar_agent.py` (`_memory_settings`)
- `src/kiss/agents/sorcar/persistence.py` (`_default_kiss_dir`, `_DB_PATH`, `_current_db_path`)
- `src/kiss/core/models/model_info.py` (`user_model_info_path`, `USER_MY_MODELS_PATH`)
