# Task

Move `src/kiss/agents/vscode/SAMPLE_TASKS.md` to `src/kiss/SAMPLE_TASKS.md` and adjust all code using it.

# Progress

- Read `SORCAR.md` first as required; it is empty.
- Cleared the previous task log in `PROGRESS.md` and started this task-specific log.
- Searched for `SAMPLE_TASKS.md` references across the repo to identify runtime, install, packaging, documentation, and test touch points.
- Read the relevant files before editing:
  - `src/kiss/agents/vscode/src/SorcarTab.ts`
  - `src/kiss/agents/vscode/src/userAssets.ts`
  - `src/kiss/agents/vscode/copy-kiss.sh`
  - `install.sh`
  - `src/kiss/agents/vscode/test/sampleTasks.test.js`
  - `src/kiss/agents/vscode/package.json`
  - `src/kiss/agents/vscode/SAMPLE_TASKS.md`
  - `README.md`
  - `src/kiss/agents/vscode/.vscodeignore`
  - `src/kiss/agents/vscode/src/kissPaths.ts`
  - `src/kiss/agents/vscode/user_assets.py`
  - `src/kiss/tests/agents/vscode/test_user_assets.py`
- Planned the changes: move the Markdown file, resolve packaged defaults from `kiss_project/src/kiss/SAMPLE_TASKS.md`, preserve a source-checkout fallback, update install/copy scripts, update tests, and verify.
- Moved `src/kiss/agents/vscode/SAMPLE_TASKS.md` to `src/kiss/SAMPLE_TASKS.md` with `git mv`.
- Updated `src/kiss/agents/vscode/src/SorcarTab.ts` so `readSampleTasks(extensionRoot)` now seeds from the packaged `extensionRoot/kiss_project/src/kiss/SAMPLE_TASKS.md` when present, falling back to `extensionRoot/../../SAMPLE_TASKS.md` in a source checkout.
- Updated `install.sh` to seed `~/.kiss/SAMPLE_TASKS.md` from `$PROJECT_DIR/src/kiss/SAMPLE_TASKS.md`.
- Updated `src/kiss/agents/vscode/copy-kiss.sh` comments/path handling so the new `src/kiss/SAMPLE_TASKS.md` is copied by the default `src/kiss/` copy flow into `kiss_project/src/kiss/SAMPLE_TASKS.md`.
- Updated `README.md` to document the new bundled path.
- Updated `src/kiss/agents/vscode/test/sampleTasks.test.js` to write package fixture sample tasks under `extensionRoot/kiss_project/src/kiss/SAMPLE_TASKS.md` and to validate the real source-checkout file at `src/kiss/SAMPLE_TASKS.md`.
- Re-searched for old `src/kiss/agents/vscode/SAMPLE_TASKS.md` and `extensionRoot/SAMPLE_TASKS.md` references; none remain outside this progress log.
- Counted the focused JavaScript sample-task tests before running them: 12 tests, so no splitting was needed.
- Attempted to run `npm run compile && node test/sampleTasks.test.js`; it failed because `tsc` was not installed in this isolated worktree, so npm dependencies needed to be installed with `npm ci`.
- Installed extension npm dependencies with `npm ci`.
- Re-ran `npm run compile && node test/sampleTasks.test.js`; all 12 tests passed.
- Switched to `gpt-5.5` as requested and re-read the changed files for thorough review, then switched back to the original model for further coding/testing.
- Ran `npm run copy-kiss` and verified it packages the moved file at `kiss_project/src/kiss/SAMPLE_TASKS.md` and not at the old `kiss_project/src/kiss/agents/vscode/SAMPLE_TASKS.md` path.
- Ran `node test/sampleTasks.test.js` again with generated `kiss_project` present; all 12 tests passed.
- Ran `uv run check --full`; Python checks, TypeScript checks, linting, and extension tests passed, but markdown formatting failed on `PROGRESS.md`, so this log was simplified into mdformat-friendly Markdown.
- Re-ran `uv run check --full`; all checks passed.
