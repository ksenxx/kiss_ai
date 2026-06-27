# Progress

## Task: Fix "time does not update every second when the panel is active"

Goal: while the model is thinking but has not produced any thought tokens yet,
the Thoughts (`.llm-panel`) panel must still be shown with a live `.panel-time`
footer. More generally, while a Thoughts (`.llm-panel`) or Tool-call (`.tc`)
panel is still open, the `.panel-time` footer must tick every second. Before
the fix, the footer was only rendered when the panel closed.

### Reproduction (end-to-end test)

Wrote `src/kiss/agents/vscode/test/panelTimeActiveTick.test.js` (jsdom-based,
loads the production `chat.html` + `panelCopy.js` + `media/main.js`):

1. `clear` + `status running` + `thinking_start` only — deliberately no
   `thinking_delta`, so the panel is open while the model has produced zero
   thought tokens.
1. Immediately asserts the `.llm-panel` exists, its `.ev.think .cnt` is empty,
   and exactly one direct-child `:scope > .panel-time` footer is already shown.
1. `await sleep(2200)` → asserts exactly one `:scope > .panel-time` footer
   remains on the tokenless `.llm-panel` and its parsed-ms value is `>= 1000`
   (proves the live ticker ran).
1. `await sleep(1200)` → asserts the footer's parsed-ms grew by `>= 800`
   (proves the tick keeps firing).
1. Asserts `.panel-time` stays the LAST child while ticking and `.ev.think .cnt`
   is still empty, proving the test remains in the no-thought-token state.
1. Sends `thinking_end` + `result` + `status running:false`,
   `await sleep(1300)` → asserts the footer is frozen (Δ < 200 ms).

Wired the new test into `npm test` in
`src/kiss/agents/vscode/package.json`.

### Fix

Edited `src/kiss/agents/vscode/media/main.js`:

- Added module-private `_activePanels = new Set()` and
  `_activePanelTickIv = null` after `fmtElapsedMs`.
- Refactored the original `finalizePanelTime` body into a shared
  `_renderPanelTime(el)` that creates/refreshes the `.panel-time` footer
  as the LAST direct child using `fmtElapsedMs(Date.now() - startMs)`.
- `stampPanelStart(el)` now: stamps `data-start-ms`, adds `el` to
  `_activePanels`, calls `_renderPanelTime(el)` so the footer appears
  immediately, then `_startActivePanelTick()`.
- `_startActivePanelTick()` lazily starts a single
  `setInterval(..., 1000)` that iterates `_activePanels`, prunes
  disconnected elements, re-renders each via `_renderPanelTime`, and
  stops the interval when the set empties.
- `finalizePanelTime(el)` now calls `_renderPanelTime(el)` once for the
  final value, then removes `el` from `_activePanels` and clears the
  interval if it was the last active panel — so the footer freezes when
  the panel closes.
- `_deferHighlight` exclusion preserved: replayed events still never
  stamp/tick.

Relevant snippet:

```js
function stampPanelStart(el) {
  if (!el || _deferHighlight) return;
  if (el.dataset.startMs) return;
  el.dataset.startMs = String(Date.now());
  _activePanels.add(el);
  _renderPanelTime(el);
  _startActivePanelTick();
}

function _startActivePanelTick() {
  if (_activePanelTickIv) return;
  if (_activePanels.size === 0) return;
  _activePanelTickIv = setInterval(() => {
    for (const el of Array.from(_activePanels)) {
      if (!el || !el.isConnected) {
        _activePanels.delete(el);
        continue;
      }
      _renderPanelTime(el);
    }
    if (_activePanels.size === 0) {
      clearInterval(_activePanelTickIv);
      _activePanelTickIv = null;
    }
  }, 1000);
}
```

### Verification

- `node test/panelTimeActiveTick.test.js` → `All tests passed`.
- `npm test` (full suite) → all suites pass; no regression in
  `panelTimeSpent.test.js`.
- `uv run check --full` → all checks pass (compileall, ruff, mypy,
  pyright, VS Code typecheck+lint, mdformat).

### Models used

- Coding / bug-fix / tests: `claude-opus-4-7`.
- Thorough review of the diff and test: `gpt-5.5` (NOT codex). Review
  found no issues to fix — single shared ticker, correct
  pruning/auto-stop, no race in single-threaded JS event loop,
  last-child anchoring preserved, replay exclusion preserved.
