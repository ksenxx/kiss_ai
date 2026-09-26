---
title: API key store ~/.kiss/api_keys.env (save_api_key, load_api_keys, RC migration)
uuid: f5fb77dd-9f58-4e59-8207-f79987d4402a
summary: 'API key store $KISS_HOME/api_keys.env (export lines, mode 0600): save_api_key
  deletes on empty value, load_api_keys at daemon start, shell RC migration, flock
  sidecar'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# API key store ~/.kiss/api_keys.env

Credentials exported as environment variables (provider API keys) live in one file,
`$KISS_HOME/api_keys.env` (`API_KEYS_ENV_FILE`), managed by
`src/kiss/core/vscode_config.py`. Format: one `export NAME=value` line per key, bash syntax, values
`shlex.quote`-d, mode 0600. The settings panel writes it, the daemon loads it at startup,
`./rsorcar` ships it to deploy targets and `scripts/install-api-keys.sh` makes `~/.bashrc` source it.

Known key names (`API_KEY_ENV_VARS`): `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`ANTHROPIC_WORKSPACE_ID`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`, `ZAI_API_KEY`,
`MOONSHOT_API_KEY`. The store may hold other assignments (channel tokens etc.); they are loaded too.

## load_api_keys() (daemon startup)
1. `_migrate_legacy_rc_keys()`: for each known key missing from the store, look in the shell RCs
   (textual scan first, which sees past a stock `.bashrc` non-interactive `return` guard, then one
   sourcing of the user's RC) and copy it in. Additive only; nothing is read when nothing is missing.
2. `_remove_systemd_mirror()`: delete legacy `api_keys.systemd.env` so a deleted key can never be
   re-injected by an old unit's `EnvironmentFile=-`.
3. Under `_config_lock`: parse the store **in Python** (no shell) with `_parse_env_assignment`, set
   `os.environ`, then `_refresh_config()`. The lock spans the file read, otherwise a concurrent save
   could be reverted (or a deleted key resurrected) by a stale snapshot.

`load_api_keys_readonly()` is the fallback for CLI entry points when `$KISS_HOME` is read-only: parse
and import, but skip migration and mirror removal.

## save_api_key(name, value)
- Writes/replaces the line in the store and removes the assignment from every supported shell RC
  (`~/.bashrc`, `~/.zshrc`, fish `config.fish`). Bash/zsh users get a sourcing hook block between
  `# >>> sorcar-cloud API keys >>>` and `# <<< sorcar-cloud API keys <<<`; fish cannot source bash
  syntax, so it gets no hook.
- Empty value means delete everywhere: store, RCs, systemd mirror, `os.environ`, `DEFAULT_CONFIG`.
- Values containing newlines are refused: edits are line-oriented, so a multi-line quoted value
  would leave an unterminated quote after a later edit.
- Updates `os.environ` and refreshes `DEFAULT_CONFIG` in place.

## Locking
Every store edit holds `_config_lock` **and** `_api_keys_store_flock()`, an `exclusive_file_lock` on
the sidecar `.api_keys.env.kiss.lock`. The sidecar is locked because the store itself is replaced
atomically (a lock on the old inode would not exclude a writer of the new one). A process must not
flock the sidecar twice through two descriptors (flock exclusion is per open file description, so
that self-deadlocks); `_config_lock` guarantees this. Helpers with a `_locked` suffix expect both
locks held.

## _refresh_config updates in place
It sets each key attribute on the existing `DEFAULT_CONFIG` instead of rebuilding `Config()`, because
a rebuild reset `max_budget` (not env-backed) and lost a budget change saved in the same payload as a
key.

## Sources
- `src/kiss/core/vscode_config.py` (`API_KEYS_ENV_FILE`, `SYSTEMD_ENV_FILE`, `load_api_keys`, `load_api_keys_readonly`, `save_api_key`, `_edit_api_keys_env_file_locked`, `_api_keys_store_flock`, `_migrate_legacy_rc_keys`, `_refresh_config`, `RC_HOOK_BEGIN`)
