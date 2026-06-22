# Progress

## Task: Always overwrite ~/.kiss/INJECTIONS.md and ~/.kiss/SAMPLE_TASKS.md on new version install

### Summary

The user wants `~/.kiss/INJECTIONS.md` and `~/.kiss/SAMPLE_TASKS.md` to always be
overwritten when a new version is installed — same behavior as `MODEL_INFO.json`.

### Changes Made

1. **`install.sh`** (root): Removed no-clobber guard (`if [ -f "$DST" ]; ... fi`).
   Changed to always `cp` INJECTIONS.md and SAMPLE_TASKS.md unconditionally,
   matching the MODEL_INFO.json pattern directly above.

1. **`src/kiss/agents/vscode/src/DependencyInstaller.ts`**:
   Added `installMarkdownAssets(kissProjectPath)` — always copies INJECTIONS.md
   and SAMPLE_TASKS.md from the bundled package to `~/.kiss/`. Called from
   `runFinalization` (right after `installModelInfoJson`).

1. **`src/kiss/agents/vscode/user_assets.py`**: Updated module docstring to
   note that install-time (via `install.sh` / `DependencyInstaller.ts`) always
   overwrites; runtime lazy-seed function remains unchanged.

1. **`src/kiss/agents/vscode/src/userAssets.ts`**: Matching docstring update.

1. **`src/kiss/tests/agents/vscode/test_user_assets.py`**: Updated docstring of
   `test_user_edits_survive_newer_package_copy` to clarify it tests the runtime
   `ensure_user_asset` function (which preserves user edits between reads),
   not the install-time behavior (which now always overwrites).
