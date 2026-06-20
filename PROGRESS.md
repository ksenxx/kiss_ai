# PROGRESS

## Task

> Can you also show the workspace, which is the work_dir of the task, in a line
> after the line you just modified in the task panel? Use gpt-5.5 model
> (not codex) for thorough review and use the original model for all tasks
> including coding, bug fixing, and test creation.

## Status: DONE

## Work completed

1. Read project instructions in `SORCAR.md` and the previous task summary in
   `PROGRESS.md`. The "line you just modified" refers to the
   `.running-item-metrics` row in `renderHistory()` in
   `src/kiss/agents/vscode/media/main.js`, which renders:

   ```text
   <steps> steps • <tokens> tok • $<cost> • <hh:mm:ss>[ • <date>]
   ```

   The requirement: render the task's workspace (`work_dir`) on its own line
   immediately below the metrics row.

1. Confirmed the backend already surfaces `work_dir` on every history session
   payload in `src/kiss/agents/vscode/server.py` (sets `session["work_dir"]`
   from `extra_obj["work_dir"]`). No backend change needed.

1. Wrote an end-to-end JSDOM integration test FIRST to reproduce the missing
   workspace line:

   - `src/kiss/agents/vscode/test/historyTaskWorkspace.test.js` drives the
     real `chat.html`, `panelCopy.js`, and `main.js` in jsdom; sends a
     `history` event with four fixture rows:

     - row A: unix workspace path → must render
       `.running-item-workspace` span with verbatim text.
     - row B: windows workspace path (backslashes) → same.
     - row C: empty `work_dir` → no workspace span at all.
     - row D: `work_dir` field missing entirely → no workspace span.

   - Asserts the workspace span is the immediate next sibling of the
     metrics span (so the workspace shows on the line right under
     metrics), and that the production `main.css` declares
     `flex-basis: 100%` on `.running-item-workspace` so the row drops onto
     its own visual line inside `.sidebar-item` (which already wraps via
     `.running-item { flex-wrap: wrap; }`).

   First run with `node test/historyTaskWorkspace.test.js` failed exactly as
   expected: "row A must render a .running-item-workspace span for its
   work_dir".

1. Implemented the fix in `src/kiss/agents/vscode/media/main.js` inside
   `renderHistory()`, immediately after `div.appendChild(metrics)`:

   ```js
   const workDir = typeof s.work_dir === 'string' ? s.work_dir : '';
   if (workDir) {
     const workspace = document.createElement('span');
     workspace.className = 'running-item-workspace';
     workspace.textContent = workDir;
     workspace.title = workDir;
     div.appendChild(workspace);
   }
   ```

   - `textContent` is XSS-safe.
   - Defensive `typeof === 'string'` guards against missing/`null`/non-string
     payloads → falls back to empty (no workspace line).
   - `title` exposes the full path on hover when overflow clipping kicks in.

1. Added the matching CSS rule in `src/kiss/agents/vscode/media/main.css`,
   mirroring `.running-item-metrics`:

   ```css
   .running-item-workspace {
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
   guarantees the workspace span occupies its own row beneath the metrics
   row. Long paths are clipped with an ellipsis; the `title` attribute
   gives the full value on hover.

1. Wired the new JS integration test into
   `src/kiss/agents/vscode/package.json` so `npm test` (and the VS Code
   extension check path) run it.

1. Added `src/kiss/tests/agents/vscode/test_history_task_workspace.py`, a
   pytest wrapper that spawns the node integration test so the regression
   is also collected by `uv run pytest`.

1. Fixed one Prettier error (multi-line `const` should be a single line)
   reported by `npm run lint:ts`.

1. Performed the requested gpt-5.5 thorough review of the diff. Checked:

   - work_dir = null/undefined/non-string → workDir = '' → no line.
   - work_dir = '' → falsy → no line.
   - work_dir = valid unix/windows path → rendered verbatim via
     `textContent`.
   - XSS via `textContent` (safe).
   - Click on the workspace span still bubbles to the row's click handler
     (opens the task) — matches metrics-row behaviour.
   - `.sidebar-item` is `display: flex; align-items: center;`. History rows
     additionally carry `.running-item` which sets `flex-wrap: wrap`, so
     `flex-basis: 100%` correctly breaks both metrics and workspace onto
     their own lines.
   - History filter visibility toggles the whole `.sidebar-item`, so
     hiding a row also hides its workspace span.
   - Sub-agent rows: `extra.work_dir` is absent for them, so the workspace
     line is omitted automatically.

   Review conclusion: no code changes required.

## Verification

1. Direct integration test (failed before fix, passes after):

   ```bash
   cd src/kiss/agents/vscode && node test/historyTaskWorkspace.test.js
   ```

   Result: `historyTaskWorkspace.test.js: all assertions passed.`

1. Targeted pytest (impacted tests, well under the 100-test threshold):

   ```bash
   uv run pytest -v \
     src/kiss/tests/agents/vscode/test_history_task_workspace.py \
     src/kiss/tests/agents/vscode/test_history_task_duration.py
   ```

   Result: 2 passed.

1. Full check:

   ```bash
   uv run check --full
   ```

   Result: all code/type/lint/extension checks passed (including the new
   `historyTaskWorkspace.test.js` invoked via the VS Code extension's
   `npm run check`).
