# PROGRESS — History panel "Workspace" filter checkbox

## Task

Add a new "Workspace" checkbox at the top of the task history panel,
positioned BEFORE the "Favorites" checkbox, checked by default. When
checked, only show tasks whose `work_dir` matches the client's
configured `work_dir`. Write an end-to-end test for the feature.

## Changes

### 1. `src/kiss/agents/vscode/media/chat.html`

Added the `Workspace` checkbox in the history filter bar, immediately
before the `Favorites` checkbox. `checked` by default.

```html
<label class="history-filter-chk" title="...">
  <input type="checkbox" id="hf-workspace" checked>Workspace
</label>
<label class="history-filter-chk" title="...">
  <input type="checkbox" id="hf-favorite">Favorites
</label>
```

### 2. `src/kiss/agents/vscode/server.py`

`_get_history()` now propagates the persisted `extra.work_dir` to the
session payload so the client can apply the Workspace filter without
re-querying:

```python
"work_dir": "",  # default
...
wd_raw = extra_obj.get("work_dir", "")
if isinstance(wd_raw, str):
    session["work_dir"] = wd_raw
```

`work_dir` is already persisted on every completed task by
`_TaskRunnerMixin._run_task_inner` via `_save_task_extra({"work_dir":
work_dir, ...})` (task_runner.py:842).

### 3. `src/kiss/agents/vscode/media/main.js`

* `renderHistory()` stamps each row with `div.dataset.workDir = s.work_dir
  || ''` so the filter helper can use plain DOM lookups.
* The filter-bar listener wiring now includes `hf-workspace`.
* `applyHistoryFilterVisibility()` reads `hfWorkspace` + `configWorkDir`
  and applies a strict-equality filter:

  ```js
  const wsOk = !onlyWorkspace || rowWorkDir === clientWorkDir;
  ```
* `populateConfigForm()` re-runs `applyHistoryFilterVisibility()` when
  `configWorkDir` actually changes so an in-place reconfiguration of
  the client work_dir immediately re-filters the already-rendered
  history list.

### 4. `src/kiss/tests/agents/vscode/test_history_filter_panel.py`

Updated structural test:

* Asserts the `Workspace` checkbox (`hf-workspace`) is rendered between
  the search box and the history list AND sits immediately before the
  Favorites checkbox.
* Asserts `Workspace` is `checked` by default.
* Asserts the JS listener-binding array references `hf-workspace`.

### 5. `src/kiss/agents/vscode/test/historyWorkspaceFilter.test.js`

New jsdom end-to-end test (`5 passed, 0 failed`). It drives the
production `media/main.js` and `media/chat.html` and verifies:

1. Workspace checkbox markup, default `checked` state, ordering before
   Favorites.
2. With Workspace ON and `configWorkDir=/repo/alpha`, only the two
   `/repo/alpha` rows are visible (the `/repo/beta` and legacy
   empty-work_dir rows are hidden).
3. Unchecking Workspace reveals every row.
4. Reconfiguring `configWorkDir` to `/repo/beta` re-filters the
   already-rendered list in place.
5. An empty client `work_dir` matches only legacy rows whose persisted
   `work_dir` is empty.

Wired into the `npm test` script next to the other jsdom E2E tests.

## Verification

* `node test/historyWorkspaceFilter.test.js` → `5 passed, 0 failed`.
* `uv run pytest -v -k "history or workspace or favorite"
  src/kiss/tests/agents/vscode/` → `38 passed`.
* `uv run pytest -q -k "server or vscode"
  src/kiss/tests/agents/vscode/` → `1050 passed`.
* `uv run check --full` → `All checks passed!`
