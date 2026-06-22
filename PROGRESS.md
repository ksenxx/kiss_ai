# Task

Rename the welcome page suggestion chip label from "Suggested" to "Suggested prompt".

# Progress

- Started by reading `SORCAR.md` as required.
- Searched the VS Code webview files for user-visible "Suggested" labels and identified the welcome-page chip label in `src/kiss/agents/vscode/media/main.js` inside `renderWelcomeSuggestions`.
- Changed the welcome chip label markup in `renderWelcomeSuggestions` from `Suggested` to `Suggested prompt`.
- Updated the existing jsdom welcome suggestion regression test to assert that `.chip-label` now renders `Suggested prompt`, in addition to preserving the full-text tooltip check.
- Counted the impacted test file before running the focused test; the rough count was below 100, so no parallel splitting was needed.
- The first focused test run failed because `jsdom` was unavailable in this isolated worktree, so I installed the VS Code extension dependencies with `npm ci`.
- Re-ran `node test/welcomeSuggestionsTooltip.test.js`; it passed.
- Ran `uv run check --full`; code checks and tests passed, but markdown formatting failed on this `PROGRESS.md` file, so I rewrote it in mdformat-friendly form.
- Re-ran `uv run check --full`; all checks passed.
