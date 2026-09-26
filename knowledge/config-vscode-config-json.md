---
title: '~/.kiss/config.json user settings: DEFAULTS, sanitize_config, RETIRED_KEYS,
  save_config'
uuid: 96cf3ce6-122e-4e89-9534-64d5b2e25ded
summary: 'config.json user settings in vscode_config.py: DEFAULTS keys (max_budget,
  use_memory, memory_dir, is_worktree...), sanitize_config types, RETIRED_KEYS, atomic
  0600 save_config'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# ~/.kiss/config.json user settings

The settings panel (VS Code extension and web app) and the daemon persist user preferences in
`$KISS_HOME/config.json` through `src/kiss/core/vscode_config.py`. (Process-level env toggles are in
`config-core-config-module`; provider API keys exported as env vars live in `api_keys.env`, see `config-api-keys-store`; the
one exception is `custom_api_key`, the custom-endpoint key, which is a persisted setting here.)

## DEFAULTS keys
`max_budget` (= `DEFAULT_MAX_BUDGET`, 100.0), `custom_endpoint`, `custom_api_key`, `custom_headers`,
`use_web_browser` (True), `remote_password`, `auto_commit_mode` (True), `is_worktree` (True),
`classify_tasks` (True: pre-run task classifier picks the lite prompt and worktree on/off for that
run only), `classify_with_decisions` (True: try OpenRouter `~typesafe/jev-latest` first, fall back to
the LLM classifier), `use_memory` (True), `memory_dir` (`""` = `$KISS_HOME/memories`), `work_dir`,
`last_model`.

Non-DEFAULTS keys pass through untouched: extension-owned `tunnel_token`, `email`,
`skill_permissions`, `mcp_permissions`, and `recent_work_dirs` (list of `{"path", "ts"}`, capped at
`MAX_RECENT_WORK_DIRS = 30`, via `record_recent_work_dir`/`recent_work_dirs`).

## Reading: load_config()
`dict(DEFAULTS)` overlaid with `_read_stored_config(path)`, then `sanitize_config`. The reader treats
a missing file, `OSError`, invalid JSON, invalid UTF-8, or a non-object top level as `{}`. It uses
`read_bytes_waiting_for_writer` so a reader on Windows waits out a concurrent replace instead of
seeing an empty config.

## sanitize_config (why it exists)
Values come from untrusted clients (`saveConfig` payloads) and a hand-editable file. Junk types used
to crash handlers: a non-string `custom_endpoint` broke `get_custom_model_entry` (killing the models
reply in every window), `custom_headers` as a list crashed `splitlines`, a non-string `work_dir`
corrupted the daemon working dir, and a truthy non-string `remote_password` looked like a password
change and restarted the daemon, killing all tasks. Rules by the default's type:
- bool: any value coerced with `bool()`;
- number: finite int/float or numeric string; booleans, non-numeric and `NaN`/`Infinity` fall back
  to the default (a NaN budget would silently disable every `cost > max_budget` check);
- string: only real `str`, else the default.

## RETIRED_KEYS = {"demo_mode", "is_parallel"}
Removing a key from DEFAULTS is not enough, because load overlays the file and save rewrites the
file from its previous contents, so the key would be re-read, broadcast in `configData` and
re-persisted forever. Retired keys are dropped by `sanitize_config` and popped by `save_config`.
`is_parallel` never had a reader: parallelism comes from the run command's `useParallel` flag. To
retire a setting, add it here as well as deleting it from DEFAULTS.

## Writing: save_config(data)
Sanitizes, then under `_config_lock` (in-process `RLock`) and
`exclusive_file_lock($KISS_HOME/.config.lock)` (cross-process) merges into the stored dict (keys
absent from *data* are preserved; `API_KEY_ENV_VARS` are skipped; retired keys purged) and writes
with `atomic_write_text(path, json, mode=0o600)`. Mode 0600 is **forced**, not just the new-file
default, because the file holds `remote_password` and `tunnel_token`; this also repairs files a past
release published as 0644.

## Applying settings
`apply_config_to_env(cfg)` sets `DEFAULT_CONFIG.max_budget` (sanitized). `build_model_config(cfg)`
returns `{"base_url", "api_key"?, "extra_headers"?}` for a custom endpoint; `get_custom_model_entry`
makes a `custom/<last path segment>` model entry. `custom_headers` is parsed as `Key: Value` lines.

## Test redirection
`CONFIG_DIR` / `CONFIG_PATH` are lazy module attributes (module `__getattr__`); tests may assign them
to override, otherwise they follow `kiss_home()`.

## Sources
- `src/kiss/core/vscode_config.py` (`DEFAULTS`, `RETIRED_KEYS`, `sanitize_config`, `load_config`, `save_config`, `_read_stored_config`, `record_recent_work_dir`, `apply_config_to_env`, `build_model_config`, `get_custom_model_entry`)
