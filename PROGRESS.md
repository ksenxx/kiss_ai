# PROGRESS

## Task

> I wanted them in the same line as the workspace and a dot between them.

(Follow-up to the previous "model name, wt/no-wt, sequential/parallel,
auto-commit/manual-commit" task — instead of rendering on their own
row, the four metadata items must share the workspace line with a `•`
separator.)

## Status: DONE

## Work completed

1. Reviewed the previous implementation in
   `src/kiss/agents/vscode/media/main.js` (`renderHistory`) where two
   separate spans were rendered under each history row's metrics line:

   - `.running-item-workspace` carrying just the `work_dir`;
   - `.running-item-meta` carrying
     `<model> • <wt|no-wt> • <par|seq> • <auto|manual>`.

   Each span used `flex-basis: 100%` so they took their own visual rows
   inside the `.sidebar-item` flex container.

1. Replaced both spans with a SINGLE merged `.running-item-workspace`
   span whose text is built as `parts.join(' • ')`:

   ```js
   const parts = [];
   if (workDir) parts.push(workDir);
   if (modelName) {
     const wtLabel = s.is_worktree ? 'wt' : 'no-wt';
     const parLabel = s.is_parallel ? 'parallel' : 'sequential';
     const acLabel = s.auto_commit_mode ? 'auto-commit' : 'manual-commit';
     parts.push(modelName, wtLabel, parLabel, acLabel);
   }
   if (parts.length > 0) {
     const workspace = document.createElement('span');
     workspace.className = 'running-item-workspace';
     const text = parts.join(' • ');
     workspace.textContent = text;
     workspace.title = text;
     div.appendChild(workspace);
   }
   ```

   This puts the workspace path and the four metadata items on the SAME
   visual line separated by ` • `, exactly as requested. Edge cases:

   - workspace + model present → `<wd> • <model> • wt • parallel • auto-commit`;
   - workspace only (legacy rows, no `model` field) → just `<wd>`;
   - model only (no workspace) → `<model> • <wt|no-wt> • ...` alone (no
     leading bullet);
   - neither → no span at all (no blank line, no placeholder);
   - missing booleans default to `false` → `no-wt`/`sequential`/`manual-commit`.

1. Removed the now-unused `.running-item-meta` CSS rule from
   `src/kiss/agents/vscode/media/main.css` and rewrote the comment above
   `.running-item-workspace` to describe the merged "workspace + meta"
   line. The single `flex-basis: 100%` rule still drops the line onto
   its own row below the metrics row.

1. Rewrote the JSDOM integration test
   `src/kiss/agents/vscode/test/historyTaskMeta.test.js` to assert the
   merged behaviour. Added a sixth fixture row (F: neither workspace nor
   model) and updated the existing rows so the test now verifies:

   - row A (workspace + all flags on) →
     `${WS_A} • gpt-5 • wt • parallel • auto-commit`;
   - row B (workspace + all flags off) →
     `${WS_B} • claude-3.7-sonnet • no-wt • sequential • manual-commit`;
   - row C (no workspace, model + flags) →
     `gpt-5-mini • wt • sequential • auto-commit` (no leading bullet);
   - row D (workspace, no model) → workspace alone (legacy);
   - row E (model only, every boolean missing) →
     `legacy-model • no-wt • sequential • manual-commit` (defaults);
   - row F (no workspace, no model) → NO `.running-item-workspace` span;
   - every row asserts that no stray `.running-item-meta` span survives;
   - the workspace+meta span is the metrics span's immediate next
     sibling (so the line lands directly below metrics);
   - the production `main.css` declares `flex-basis: 100%` on
     `.running-item-workspace` so the line drops onto its own visual
     row in the flex container.

1. Updated the pytest wrapper docstring in
   `src/kiss/tests/agents/vscode/test_history_task_meta.py` to reflect
   the new combined-line contract. No backend change was needed — the
   server still emits `model` / `is_worktree` / `is_parallel` /
   `auto_commit_mode` on each session row exactly as before, so the
   server-side pytest module
   `test_history_task_meta_server.py` continued to pass unchanged.

1. `npm install` inside `src/kiss/agents/vscode` (needed because the
   worktree did not yet have `node_modules/`), then verification.

## Verification

1. Direct integration test (single source of truth):

   ```bash
   cd src/kiss/agents/vscode && node test/historyTaskMeta.test.js
   ```

   Result:
   ```
   ok - workspace + metadata render on the same line, separated by " • "
   ok - workspace+meta span has flex-basis: 100%
   historyTaskMeta.test.js: all assertions passed.
   ```

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
