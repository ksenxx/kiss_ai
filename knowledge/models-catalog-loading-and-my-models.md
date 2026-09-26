---
title: How the model catalog is loaded (KISS_MODEL_INFO_PATH, ~/.kiss/MODEL_INFO.json,
  MY_MODELS.json)
uuid: cab0d8de-081e-4964-b28d-a69327ebaa7b
summary: Which MODEL_INFO.json is loaded (KISS_MODEL_INFO_PATH, ~/.kiss copy when
  installed, bundled), import-time retries, MY_MODELS.json overlay, custom model CRUD,
  custom_model_config.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# How the model catalog is loaded

`MODEL_INFO = _load_model_info()` runs at **import time** of
`kiss.core.models.model_info`. Everything that asks about a model (factory, cost, context length,
picker) reads this dict.

## Which catalog file (`_select_catalog_path`)
1. `KISS_MODEL_INFO_PATH` env var, if it names an existing file. `update_models.py --model-info`
   sets it so the updater reads the exact file it rewrites. A non-existent override falls back to
   the bundled file (lets the updater bootstrap a new target).
2. `$KISS_HOME/MODEL_INFO.json` (`user_model_info_path`, `KISS_HOME` defaults to `~/.kiss`) but
   **only when the package is installed** and the file exists.
3. The bundled `src/kiss/core/models/MODEL_INFO.json` (`PACKAGE_MODEL_INFO_PATH`).

"Installed" (`_is_installed_package`) means the project root four directories above
`model_info.py` has no `.git` entry (the packaged VS Code extension bundle). A git checkout or
worktree (`.git` file) always uses the bundled catalog, so a stale `~/.kiss/MODEL_INFO.json` never
shadows the source of truth during development. The installer seeds the user copy; the settings
panel's "Update Models" refreshes it via `kiss.scripts.update_models --model-info`.

## Robust reading
`_read_model_info_json` retries 5 times, 50 ms apart (`_CATALOG_READ_ATTEMPTS`,
`_CATALOG_RETRY_SECONDS`) on `OSError`/`ValueError` or a non-object top level, then raises
`KISSError` naming the file. Reason: an import-time `JSONDecodeError` would kill every subagent or
CLI process while some tool is rewriting the file. `_load_model_info` catches schema errors
(`KISSError, KeyError, TypeError, ValueError, AttributeError`) from a **non-bundled** catalog and
falls back to the bundled one with a warning; errors in the bundled catalog propagate.

## MY_MODELS.json overlay
`~/.kiss/MY_MODELS.json` (`USER_MY_MODELS_PATH`) is merged on top in `_load_catalog_file`:
matching keys replace the catalog entry wholesale, new keys are added. Keys starting with `_` and
non-object values are skipped (documentation / inert `_example/...` entry). The file is
auto-seeded from `MY_MODELS_DEFAULT_CONTENT` via `_seed_file_atomically` (temp file + `os.link`,
never clobbers) because a plain `write_text` seed let a concurrent reader see an empty file and
silently lose every user model. Unreadable/corrupt file -> treated as `{}` by the loader.

## Custom models from the settings panel
- `list_custom_models()` -> rows `{name, endpoint, api_key, headers}` for the UI.
- `save_custom_model(name, endpoint, api_key, headers, original_name)`: add refuses an existing
  name; edit merges into the existing entry (hand-written `thinking`/prices survive); rename refuses
  an existing target. New entries get loader-required keys from `_seed_new_custom_entry`: a
  name shadowing a bundled model copies that model's context length and prices (not the generic
  128K/$0 of `_CUSTOM_MODEL_DEFAULTS`, which would break budget accounting). Names starting with
  `_` are refused. Edits hold `_MY_MODELS_EDIT_LOCK` plus a cross-process flock
  (`_my_models_flock`) and write atomically (`atomic_write_text`); a corrupt file is refused, not
  "recovered".
- `delete_custom_model(name)`.
- `custom_model_config(model_name)` returns `{"base_url", "api_key"?, "extra_headers"?}` when the
  entry has an `endpoint` (headers parsed from `Key: Value` lines by
  `kiss.core.vscode_config._parse_custom_headers`). The task runner passes this as
  `model_config`, so the `model()` factory routes the name to an OpenAI-compatible client at that
  endpoint (see `models-provider-resolution`).

## Gotchas
- `MODEL_INFO` is a snapshot: edits to the JSON files after import are not seen by a running
  process.
- An override entry replaces the whole bundled entry; it must repeat the three required keys.

## Sources
- `src/kiss/core/models/model_info.py` (`_select_catalog_path`, `_is_installed_package`, `user_model_info_path`, `_read_model_info_json`, `_load_model_info`, `_load_catalog_file`, `_read_my_models`, `_seed_file_atomically`, `save_custom_model`, `delete_custom_model`, `list_custom_models`, `custom_model_config`, `_seed_new_custom_entry`)
