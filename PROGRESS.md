# Task

Add tooltips showing the entire suggested text for each suggestion.

# Progress

- Started by reading SORCAR.md as required.
- Cleared prior task progress and began investigating the VS Code suggestion UI.
- Located welcome suggestion rendering in `src/kiss/agents/vscode/media/main.js` (`renderWelcomeSuggestions`) and the existing custom tooltip system based on `[data-tooltip]`.
- Added an end-to-end jsdom test `src/kiss/agents/vscode/test/welcomeSuggestionsTooltip.test.js` that sends a production `welcome_suggestions` event, hovers the rendered chip, and expects the custom tooltip to show the full suggestion text.
- Ran the new test directly before installing Node dependencies in the worktree; it failed because `jsdom` was unavailable in this isolated worktree, but the test documents the missing tooltip behavior.
- Implemented the fix by adding `chip.dataset.tooltip = s.text` in `renderWelcomeSuggestions`, reusing the existing custom tooltip code that watches `[data-tooltip]`.
- Added `welcomeSuggestionsTooltip.test.js` to the VS Code extension `npm test` script so it runs with the normal extension test suite.
- Ran `npm ci` in the VS Code extension directory to install test dependencies in this isolated worktree, then verified the impacted test with `node test/welcomeSuggestionsTooltip.test.js`.
- Ran `uv run check --full`; it passed code checks and the VS Code extension test suite but failed markdown formatting on pre-existing `src/kiss/INJECTIONS.md` and `src/kiss/agents/vscode/SAMPLE_TASKS.md`.
- Formatted those markdown files with `uv run mdformat ...` and reran `uv run check --full`, which passed all checks.
