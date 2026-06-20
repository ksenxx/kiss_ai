# PROGRESS

## Task

> In a task panel of the task history panel, show the time spent on the task in
> hh:mm:ss format after showing the cost at the bottom of the panel. Reproduce
> the issue by writing an integration test. Then fix the issue.

## Status: DONE

## Work completed

1. Read the project instructions in `SORCAR.md`.

1. Located the History sidebar rendering in
   `src/kiss/agents/vscode/media/main.js`, specifically `renderHistory()` and
   the `.running-item-metrics` row that previously rendered:

   ```text
   <steps> steps • <tokens> tok • $<cost> [ • <date>]
   ```

1. Confirmed the backend history payload already includes `startTs` and `endTs`
   in milliseconds, so no backend change was needed.

1. Added the JSDOM integration test
   `src/kiss/agents/vscode/test/historyTaskDuration.test.js`. The test drives
   the real `chat.html`, `panelCopy.js`, and `main.js`, sends a `history` event,
   and asserts that history rows render duration immediately after cost:

   ```text
   <steps> steps • <tokens> tok • $<cost> • hh:mm:ss [ • <date>]
   ```

   The fixture covers 10 seconds (`00:00:10`), 65 seconds (`00:01:05`), 3725
   seconds (`01:02:05`), a running task using a frozen `Date.now()`, and a
   legacy row with no usable timestamps that must omit the duration token.

1. Reproduced the issue by running the new node integration test before the fix;
   it failed because the metrics row did not include the duration token.

1. Fixed `src/kiss/agents/vscode/media/main.js` by adding
   `formatDurationHms(ms)` and computing duration inside `renderHistory()` from
   `endTs - startTs` for finished rows or `Date.now() - startTs` for running
   rows. Non-positive or legacy durations are omitted, so `00:00:00` is not
   shown for missing timestamp data.

1. Wired the JS integration test into
   `src/kiss/agents/vscode/package.json` so `npm test` and the VS Code extension
   check path run it.

1. Added `src/kiss/tests/agents/vscode/test_history_task_duration.py`, a pytest
   wrapper that spawns the node integration test so the regression is also
   collected by pytest.

1. Used `gpt-5.5` (non-codex) only for the requested thorough review of the diff
   and tests, then switched back to the original coding model. Review conclusion:
   no code changes were needed.

## Verification

1. Direct integration test:

   ```bash
   cd src/kiss/agents/vscode && node test/historyTaskDuration.test.js
   ```

   Result: passed.

1. Counted the targeted pytest set before running it: 9 tests, below the
   100-test parallelization threshold.

1. Targeted pytest run:

   ```bash
   uv run pytest -v \
     src/kiss/tests/agents/vscode/test_history_task_duration.py \
     src/kiss/tests/agents/vscode/test_history_filter_panel.py \
     src/kiss/tests/agents/vscode/test_history_title_full_text.py \
     src/kiss/tests/agents/vscode/test_history_failed_flag.py
   ```

   Result: 9 passed.

1. Full check:

   ```bash
   uv run check --full
   ```

   Result: all code/type/lint/extension checks passed, including the new
   history duration integration test.
