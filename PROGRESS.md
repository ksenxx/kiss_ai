# PROGRESS

## Task

> Can you also add model name, wt/no-wt, sequential/parallel, auto commit /
> manual commit separated by dot in the same line? Use gpt-5.5 model (not codex)
> for thorough review and use the original model for all tasks including coding,
> bug fixing, and test creation.

## Status: DONE

## Work completed

1. Read `PROGRESS.md` from the previous task (which added the workspace line
   beneath the metrics row in each History sidebar item) to confirm the
   anchor point: the new four-item dot-separated metadata line must go
   right under that workspace row.

1. Surveyed the persisted task metadata in
   `src/kiss/agents/vscode/task_runner.py::_run_task_inner` →
   `_save_task_extra(...)`. The `extra` JSON column already carries every
   field the user asked for:

   ```python
   _save_task_extra(
       {
           "model": model,
           "is_parallel": tab.use_parallel,
           "is_worktree": use_worktree,
           "auto_commit_mode": tab.auto_commit_mode,
           ...
       },
       task_id=tab.task_history_id,
   )
   ```

   So no schema change was needed — the new line is a pure read/render
   feature on top of fields already present in the DB.

1. Wrote the failing end-to-end JSDOM integration test FIRST:
   `src/kiss/agents/vscode/test/historyTaskMeta.test.js`. It drives the
   real `chat.html`, `panelCopy.js`, and `main.js` in jsdom and asserts:

   - row A (all flags true) renders
     `gpt-5 • wt • parallel • auto-commit`;
   - row B (all flags false) renders
     `claude-3.7-sonnet • no-wt • sequential • manual-commit`;
   - row C (model present, NO workspace) renders the meta span as the
     metrics span's immediate next sibling (no workspace gap);
   - row D (no `model` field) renders NO `.running-item-meta` span at all;
   - row E (`model` only, every boolean missing) renders
     `legacy-model • no-wt • sequential • manual-commit` (defaults);
   - the production `main.css` declares `flex-basis: 100%` on
     `.running-item-meta` so the line drops onto its own visual row in
     the flex container.

   First run with `node test/historyTaskMeta.test.js` failed exactly as
   expected: "row A must render a .running-item-meta span".

1. Implemented the frontend in `src/kiss/agents/vscode/media/main.js` in
   `renderHistory()`, immediately after the workspace span:

   ```js
   const modelName = typeof s.model === 'string' ? s.model : '';
   if (modelName) {
     const meta = document.createElement('span');
     meta.className = 'running-item-meta';
     const wtLabel = s.is_worktree ? 'wt' : 'no-wt';
     const parLabel = s.is_parallel ? 'parallel' : 'sequential';
     const acLabel = s.auto_commit_mode ? 'auto-commit' : 'manual-commit';
     const metaText =
       modelName + ' • ' + wtLabel + ' • ' + parLabel + ' • ' + acLabel;
     meta.textContent = metaText;
     meta.title = metaText;
     div.appendChild(meta);
   }
   ```

   `textContent` is XSS-safe. `title` exposes the full text on hover so
   long model names clipped by `overflow: hidden` stay reachable.
   Booleans default to `false` (omitted payload → "no-wt" / "sequential"
   / "manual-commit"). An empty / missing `model` suppresses the whole
   line — no placeholder.

1. Added a sibling CSS rule in `src/kiss/agents/vscode/media/main.css`
   mirroring `.running-item-workspace`:

   ```css
   .running-item-meta {
     flex-basis: 100%;
     font-size: var(--fs-sm);
     color: color-mix(in srgb, #000 55%, transparent);
     margin-top: 2px;
     white-space: nowrap;
     overflow: hidden;
     text-overflow: ellipsis;
   }
   ```

   `flex-basis: 100%` + the existing `.running-item { flex-wrap: wrap; }`
   guarantees the meta row gets its own visual line inside the
   `.sidebar-item` flex container, the same trick metrics and workspace
   already use.

1. Wired the new JS test into `src/kiss/agents/vscode/package.json` so
   `npm test` (and the VS Code extension check path) run it.

1. Added the backend half in `src/kiss/agents/vscode/server.py`:

   - Defaulted `model = ""`, `is_worktree = False`, `is_parallel = False`,
     `auto_commit_mode = False` on every session dict in `_get_history`.
   - Read each field back out of the `extra` JSON with safe type / bool
     coercion (`isinstance(str)` for the model; `bool(...)` for the three
     flags). Garbage values (e.g. `None`) fall back to the documented
     defaults.
   - For live (still-running) tasks, overlaid the same four fields in
     `_overlay_live_metrics` using `agent.model_name` (the model the
     task was launched with) with a tab-level `selected_model` fallback,
     and `tab.use_worktree / use_parallel / auto_commit_mode`. This makes
     the meta line render correctly for in-flight tasks whose `extra`
     row hasn't been written yet (the writer fires at task end).

1. Added `src/kiss/tests/agents/vscode/test_history_task_meta_server.py`,
   a pytest module with 7 end-to-end test methods that drive the real
   `VSCodeServer._get_history` against a fresh sqlite DB. Cases: no
   `extra` (all defaults), all flags on, all flags off, model-only
   (legacy partial), non-string model (None → empty), truthy non-bool
   booleans coerced via `bool()`, falsy values coerced to False.

1. Added `src/kiss/tests/agents/vscode/test_history_task_meta.py`,
   the pytest wrapper that spawns the JSDOM test from `uv run pytest`
   so the regression is also collected by Python's test runner.

1. Fixed two Prettier errors (single-line meta-text expression) and a
   single mypy error (`Returning Any` from `_history_sessions`) reported
   by `uv run check --full`.

1. Performed the requested gpt-5.5 thorough review of the full diff.
   Reviewed for:

   - missing model field → no meta line (correct).
   - non-string model field (e.g. `None`) → empty model → no meta line.
   - non-bool boolean fields → coerced via `bool()` so the frontend
     receives `true` / `false` only.
   - workspace-absent rows (`work_dir = ""`) still render the meta line,
     immediately after the metrics row (no awkward gap).
   - Click on the meta span bubbles up to the row's click handler
     (opens the task) — same behaviour as the metrics and workspace
     spans.
   - `applyHistoryFilterVisibility()` toggles the entire `.sidebar-item`
     so hiding a row also hides its meta span.
   - Long model names clipped by `overflow: hidden`; `title` gives the
     full text on hover.
   - Running tasks: live overlay reads from `agent.model_name`
     (in-flight) with `tab.selected_model` fallback. This guards against
     the user changing the model picker mid-task.

   No code changes resulted from the review.

## Verification

1. Direct integration test (failed before fix, passes after):

   ```bash
   cd src/kiss/agents/vscode && node test/historyTaskMeta.test.js
   ```

   Result: `historyTaskMeta.test.js: all assertions passed.`

1. Targeted pytest (impacted tests only, well under the 100-test
   threshold):

   ```bash
   uv run pytest -v \
     src/kiss/tests/agents/vscode/test_history_task_meta.py \
     src/kiss/tests/agents/vscode/test_history_task_meta_server.py \
     src/kiss/tests/agents/vscode/test_history_task_workspace.py \
     src/kiss/tests/agents/vscode/test_history_task_duration.py
   ```

   Result: 10 passed.

1. Full check:

   ```bash
   uv run check --full
   ```

   Result: all code / type / lint / extension checks passed.
