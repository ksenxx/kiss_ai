# PROGRESS

## Task

> Move `MODEL_INFO` from `src/kiss/core/models/model_info.py` into a JSON file
> `src/kiss/core/models/MODEL_INFO.json`. `src/kiss/scripts/update_models.py`
> MUST update `MODEL_INFO.json`. During installation, copy `MODEL_INFO.json`
> to `~/.kiss/` and make `model_info.py` load the table from that copied file.

## Status: CODE COMPLETE — awaiting GPT‑5.5 review

## Work completed (chronological)

1. Inspected the existing `model_info.py` (≈900 lines: 499 entries + helpers
   `_mi`/`_emb`, cache-pricing logic, `model()` dispatch).

1. Inspected `update_models.py` — confirmed it was a Python-source rewriter
   that called `_make_entry_line` to emit `_mi(...)` lines.

1. Inspected `install.sh` and `DependencyInstaller.ts` to find the right
   places to seed `~/.kiss/MODEL_INFO.json`.

1. **Extracted MODEL_INFO to JSON.** Wrote a one-shot AST-walk script in
   `tmp/dump_to_json.py` that visits each `_mi(...)`/`_emb(...)` call and
   writes the explicit fields (`context_length`, prices, `fc`/`emb`/`gen`,
   `thinking`) into `src/kiss/core/models/MODEL_INFO.json`. 499 entries.
   Cache prices are NOT persisted (they are computed at load time).

1. **Refactored `model_info.py` to load from JSON:**

   - Added `PACKAGE_MODEL_INFO_PATH` and `USER_MODEL_INFO_PATH` constants.
   - Added `_ensure_user_model_info_path()` that copies the package copy
     to `~/.kiss/MODEL_INFO.json` when missing OR when the package copy
     is newer (mtime-based refresh); falls back to the package path on
     `OSError` so import never fails on read-only filesystems.
   - Added `_build_model_info_entry()` and `_load_model_info()`.
   - Removed the `_mi` / `_emb` helper functions and the literal
     `MODEL_INFO: dict[str, ModelInfo] = {…}` (replaced by a single
     `MODEL_INFO: dict[str, ModelInfo] = _load_model_info()`).

1. **Refactored `update_models.py` to update the JSON file:**

   - Changed `_EXPECTED_SUBPATH` and `MODEL_INFO_PATH` to point at
     `src/kiss/core/models/MODEL_INFO.json`.
   - Added `USER_MODEL_INFO_PATH = Path.home() / ".kiss" / "MODEL_INFO.json"`.
   - Replaced `_make_entry_line` / `apply_updates_to_file` (Python source
     rewriter + line-span parser) with:
     - `_build_entry(...)` — builds one JSON entry dict.
     - `_read_model_info_json` / `_write_model_info_json` helpers.
     - `apply_updates_to_file(...)` — reads JSON, removes deprecated names,
       applies updates (including `thinking=None` → key-removal), inserts
       new models with a `comment` field ("NEW" / "NEW: needs pricing"),
       sorts keys, writes back, and ALSO refreshes
       `~/.kiss/MODEL_INFO.json` when it already exists.
   - Removed the now-unused `fmt_price` helper.

1. **Updated installation:**

   - `install.sh`: added a `cp` step right before writing
     `~/.kiss/.extension-updated` to seed the user-local copy.
   - `DependencyInstaller.ts`: added `installModelInfoJson(kissProjectPath)`,
     called from `runFinalization`, mirroring the bash behaviour.

1. **Updated/added tests:**

   - Rewrote `test_update_models_xhigh.py` to exercise the new
     `_build_entry` helper (replacing the removed `_make_entry_line`).
   - Rewrote `test_update_models_write_target.py` to operate on JSON
     fixtures and to redirect `USER_MODEL_INFO_PATH` so the test does
     not touch a developer's real `~/.kiss/`.
   - Added `test_model_info_json_loader.py` with 9 new end-to-end tests
     covering: package JSON validity, fresh-copy creation, stale-copy
     mtime refresh, fresh-copy preservation, OSError fallback to
     package path, `_build_model_info_entry` with full + minimal
     payloads, `_load_model_info` return type, and the documented
     `~/.kiss/MODEL_INFO.json` location invariant.
   - Updated `test_model_implementations.py` to drop its `_mi` import
     (the helper is gone) and use a local `_mi_for_test` wrapper that
     calls `ModelInfo(...)` directly.

1. **Verified:**

   - `uv run check --full` → all checks pass.
   - `uv run pytest src/kiss/tests/core/models/test_model_info_completion.py src/kiss/tests/core/models/test_model_info_json_loader.py src/kiss/tests/core/models/test_model_implementations.py src/kiss/tests/scripts/ src/kiss/tests/core/test_coverage_integration.py`
     → 142 passed, 1 skipped.
   - Smoke test: 499 models loaded, cache pricing correctly applied
     (`gpt-5.5` cache_read = 0.5 = 0.10 × input, Claude 1h cache_write =
     2× input, etc.), `calculate_cost` returns expected values.
