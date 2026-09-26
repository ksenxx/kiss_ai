---
title: 'Configuration and core utilities: area overview'
uuid: f7b28027-f57a-4b78-937b-365f10fd3f7d
summary: 'Map of kiss.core config and utility modules: config.py, vscode_config.py
  (config.json, api_keys.env), utils.py atomic writes, file_lock.py, processes.py,
  brand.py, speech, _version.py'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Configuration and core utilities: area overview

Files under `src/kiss/core/` that hold settings, per-user state and cross-platform primitives. Memory
(`src/kiss/core/memoryfield/`) has its own overview: `memory-overview`.

| File | What it owns | Page |
|---|---|---|
| `config.py` | `kiss_home()`, process-level `Config` (API keys from env, `KISS_*` cost-lever toggles), `DEFAULT_MAX_BUDGET`, per-process artifact dir | `config-core-config-module` |
| `vscode_config.py` | `~/.kiss/config.json` settings (`DEFAULTS`, `sanitize_config`, `RETIRED_KEYS`, `save_config`), custom endpoint model config | `config-vscode-config-json` |
| `vscode_config.py` | `~/.kiss/api_keys.env` key store, RC migration, key deletion | `config-api-keys-store` |
| `utils.py` | `atomic_write_text`, Windows-safe replace/read | `config-atomic-writes-and-locks` |
| `file_lock.py` | `exclusive_file_lock`, `lock_exclusive` (fcntl / msvcrt) | `config-atomic-writes-and-locks` |
| `utils.py` | `finish`, `ensure_html`, `substitute_prompt_args`, `is_root_dir`, `config_to_dict`, `dump_yaml` | `config-core-utils` |
| `processes.py` | `pid_alive`, `popen_process_group`, `kill_process_group`, `process_identity` | `config-process-helpers` |
| `brand.py`, `speech_synthesis.py` | product name from `brand.json`; `talk` TTS via `gpt-audio-1.5` | `config-branding-and-speech` |
| `_version.py` | `__version__`, the single version source | `config-versioning` |

Directory layout of `$KISS_HOME` (`~/.kiss`): `config-kiss-home-layout`.

## Two kinds of configuration
1. **Environment / process** (`kiss.core.config.DEFAULT_CONFIG`): read from env when `Config()` is
   built; API keys reach the environment through `vscode_config.load_api_keys()` at daemon start.
2. **Persisted user settings** (`config.json`): read with `vscode_config.load_config()` on demand,
   written by the settings panel through `save_config`. `apply_config_to_env` copies `max_budget`
   into `DEFAULT_CONFIG`.

Precedence examples: `KISS_USE_MEMORY` env beats `config.json` `use_memory`; a per-run argument beats
both (see `memory-when-enabled`).

## Recurring bug classes in this area (from history and docstrings)
- Lost updates between threads/processes: fixed with `_config_lock` + sidecar flocks around
  read-modify-write, and snapshots taken under the lock (`load_api_keys`).
- Readers seeing empty or half-written files: fixed with `atomic_write_text` everywhere.
- Windows breakage: `fcntl` imports, `os.kill(pid, 0)` meaning Ctrl+C, `os.replace` sharing
  violations; fixed by `file_lock.py`, `processes.py` and the retry helpers.
- Junk-typed or non-finite values from clients or hand edits crashing handlers or disabling
  budget checks: fixed by `sanitize_config` and `_env_*` helpers.
- Paths frozen at import ignoring `KISS_HOME`: resolve through `kiss_home()` at call time.
- Secrets leaking: config.json forced 0600, provider env keys in api_keys.env (only `custom_api_key` is in config.json), `config_to_dict` drops key
  fields from trajectories.

## Sources
- `src/kiss/core/config.py`, `src/kiss/core/vscode_config.py`, `src/kiss/core/utils.py`, `src/kiss/core/file_lock.py`, `src/kiss/core/processes.py`, `src/kiss/core/brand.py`, `src/kiss/core/speech_synthesis.py`, `src/kiss/core/_version.py`
