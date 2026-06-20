# PROGRESS

## Task

> Make sure that the From label, the date textbox, and the date picker
> MUST not get split between new lines. Reproduce the issue by writing
> an integration test. Then fix the issue.

## Status: DONE

## Work completed

1. Located the offending markup in
   `src/kiss/agents/vscode/media/chat.html` (the History sidebar's
   `.history-filter-bar`). The "From" trio (`<label for="hf-from">`,
   `<input type="date" id="hf-from">`, `<button id="hf-from-btn">`)
   and the matching "To" trio were six direct children of
   `.history-filter-bar`, which is declared as
   `display: flex; flex-wrap: wrap;`. Because each child was its own
   flex item, a narrow sidebar could wrap the label, the textbox, and
   the calendar-picker button onto three different rows.

1. Reproduced the bug by writing a JSDOM integration test
   `src/kiss/agents/vscode/test/historyFilterDateGroup.test.js` that
   loads `media/chat.html` + `media/main.js` and asserts:

   - `label[for="hf-from"]`, `#hf-from`, and `#hf-from-btn` share a
     single direct parent element (i.e. the trio is wrapped in one
     grouping span);
   - that parent is **not** `.history-filter-bar` itself;
   - that parent carries the class `.history-filter-date-group`;
   - the same three invariants hold for the "To" trio;
   - the order inside each group is label → input → button;
   - the group has exactly 3 children (no stray nodes);
   - `media/main.css` declares a `.history-filter-date-group` rule
     that uses `display: inline-flex` (or `inline-block`) AND uses
     `flex-wrap: nowrap` or `white-space: nowrap`, guaranteeing the
     three pieces stay glued on a single line even when the sidebar
     wraps.

   Running the test against the unfixed code failed at step 1, exactly
   as expected (TDD).

1. Fixed `src/kiss/agents/vscode/media/chat.html` by wrapping each
   trio in a `<span class="history-filter-date-group">…</span>`:

   ```html
   <span class="history-filter-date-group">
     <label for="hf-from" class="history-filter-date-lbl">From:</label>
     <input type="date" id="hf-from" class="history-filter-date" …>
     <button type="button" id="hf-from-btn" class="history-filter-date-btn" …>📅</button>
   </span>
   <span class="history-filter-date-group">
     <label for="hf-to" class="history-filter-date-lbl">To:</label>
     <input type="date" id="hf-to" class="history-filter-date" …>
     <button type="button" id="hf-to-btn" class="history-filter-date-btn" …>📅</button>
   </span>
   ```

   With the wrapper in place the bar sees just one flex item per
   date filter, so the bar's `flex-wrap: wrap` can still break
   between groups when the sidebar is very narrow, but it can never
   tear a label/input/button trio apart.

1. Added the matching CSS rule to
   `src/kiss/agents/vscode/media/main.css` (next to the existing
   `.history-filter-date-lbl` rule):

   ```css
   .history-filter-date-group {
     display: inline-flex; align-items: center; gap: 4px;
     flex-wrap: nowrap; white-space: nowrap;
   }
   ```

   `inline-flex` makes the wrapper one flex item of
   `.history-filter-bar`; `flex-wrap: nowrap` + `white-space: nowrap`
   keep the children on a single line.

1. Added a pytest wrapper
   `src/kiss/tests/agents/vscode/test_history_filter_date_group.py`
   that spawns `node` on the JSDOM test so `uv run pytest` and CI
   pick it up alongside the rest of the VS Code-extension tests.

## Verification

1. Direct JSDOM integration test (the source of truth):

   ```bash
   cd src/kiss/agents/vscode && node test/historyFilterDateGroup.test.js
   ```

   Result:

   ```
   ok - From and To label/input/button trios share a single
        .history-filter-date-group parent inside .history-filter-bar
   ok - .history-filter-date-group CSS keeps label/input/button on
        the same line
   historyFilterDateGroup.test.js: all assertions passed.
   ```

1. Regression sweep of all related History-sidebar JSDOM tests
   (`historyTaskMeta`, `historyTaskWorkspace`, `historyTaskDuration`,
   `historyWorkspaceFilter`, `historyFilterDateGroup`) — all green.

1. Targeted pytest (impacted tests only, well under the 100-test
   threshold):

   ```bash
   uv run pytest -v \
     src/kiss/tests/agents/vscode/test_history_filter_date_group.py \
     src/kiss/tests/agents/vscode/test_history_task_meta.py \
     src/kiss/tests/agents/vscode/test_history_task_workspace.py
   ```

   Result: 3 passed.

1. Full check:

   ```bash
   uv run check --full
   ```

   Result: all code / type / lint / extension / markdown checks
   passed.
