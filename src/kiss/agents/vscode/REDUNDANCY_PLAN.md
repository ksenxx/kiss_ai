# Redundancy & Simplification Plan for `src/kiss/agents/vscode/`

## Summary

After analyzing all 11 source files (8,055 lines total), I identified **14 concrete redundancies and simplification opportunities** across Python, TypeScript, and JavaScript.
Estimated net savings: **~107 lines** plus significant maintainability improvement.

______________________________________________________________________

## R1. `server.py`: `_get_last_session` duplicates `_replay_session` logic

**Files:** `server.py` (lines ~470–500)

**Problem:** `_get_last_session` manually loads events, calls `resume_chat`, broadcasts `task_events`, and calls `_emit_pending_worktree` — the same sequence that `_replay_session` already does. The only added logic is loading the most recent task from history and restoring a pending merge.

**Current code (redundant):**

```python
def _get_last_session(self) -> None:
    with self._state_lock:
        if self._task_thread and self._task_thread.is_alive():
            return
    entries = _load_history(limit=1)
    if entries:
        task = str(entries[0].get("task", ""))
        if task:
            events = _load_task_chat_events(task)       # ← duplicated
            self.agent.resume_chat(task)                 # ← duplicated
            self.printer.broadcast({"type": "task_events", "events": events, "task": task})  # ← duplicated
            self._emit_pending_worktree()                # ← duplicated
    self._restore_pending_merge()
```

**Fix:** Have `_get_last_session` call `_replay_session(task)` instead of duplicating the event-load + resume + broadcast + emit logic.

**Note:** `_replay_session` has an early return `if not events: return` that `_get_last_session` does not. This means tasks with no stored events would no longer trigger `resume_chat()` or `_emit_pending_worktree()`. This is acceptable — with no events there is nothing to replay, and the worktree state wouldn't have been created.

```python
def _get_last_session(self) -> None:
    with self._state_lock:
        if self._task_thread and self._task_thread.is_alive():
            return
    entries = _load_history(limit=1)
    if entries:
        task = str(entries[0].get("task", ""))
        if task:
            self._replay_session(task)
    self._restore_pending_merge()
```

**Savings:** ~5 lines, eliminates a subtle maintenance hazard (changes to replay logic must be mirrored).

______________________________________________________________________

## R2. `server.py`: Duplicated `worktree_done` event construction

**Files:** `server.py` (lines ~290, ~460)

**Problem:** The `worktree_done` event dict is constructed nearly identically in two places:

1. In `_run_task_inner` after the agent finishes
1. In `_emit_pending_worktree` for session replay

Both build nearly identical dicts — the only difference is `hasConflict`:

- `_run_task_inner`: `"hasConflict": self._check_merge_conflict()` (unconditional, but only called when `changed` is non-empty)
- `_emit_pending_worktree`: `"hasConflict": self._check_merge_conflict() if changed else False` (conditional)

```python
self.printer.broadcast({
    "type": "worktree_done",
    "branch": wt._wt_branch,
    "worktreeDir": str(wt._wt_dir),
    "originalBranch": wt._original_branch,
    "changedFiles": changed,
    "hasConflict": ...,  # differs between sites (see above)
})
```

**Fix:** Extract a `_broadcast_worktree_done(changed: list[str])` method.

```python
def _broadcast_worktree_done(self, changed: list[str]) -> None:
    wt = self._worktree_agent
    self.printer.broadcast({
        "type": "worktree_done",
        "branch": wt._wt_branch,
        "worktreeDir": str(wt._wt_dir),
        "originalBranch": wt._original_branch,
        "changedFiles": changed,
        "hasConflict": self._check_merge_conflict() if changed else False,
    })
```

Both call sites shrink to `self._broadcast_worktree_done(changed)`.

**Savings:** ~10 lines, one source of truth for the event shape.

______________________________________________________________________

## R3. `server.py`: `_get_adjacent_task` — redundant branching

**Files:** `server.py` (lines ~590–610)

**Problem:** Both the found and not-found branches broadcast the same event type with the same keys — only the values differ.

**Current:**

```python
if result:
    self.printer.broadcast({
        "type": "adjacent_task_events",
        "direction": direction,
        "task": result["task"],
        "events": result["events"],
    })
else:
    self.printer.broadcast({
        "type": "adjacent_task_events",
        "direction": direction,
        "task": "",
        "events": [],
    })
```

**Fix:** Collapse to a single broadcast:

```python
def _get_adjacent_task(self, task: str, direction: str) -> None:
    result = _get_adjacent_task_in_chat(task, direction)
    self.printer.broadcast({
        "type": "adjacent_task_events",
        "direction": direction,
        "task": result["task"] if result else "",
        "events": result["events"] if result else [],
    })
```

**Savings:** ~6 lines.

______________________________________________________________________

## R4. `browser_ui.py`: `stop_recording` and `peek_recording` share identical filter+coalesce logic

**Files:** `browser_ui.py` (lines ~100–130)

**Problem:** Both methods have an identical two-line suffix:

```python
filtered = [e for e in raw if e.get("type") in _DISPLAY_EVENT_TYPES]
return _coalesce_events(filtered)
```

**Assessment:** The proposed helper would be a 1-line method body, which violates the "no 1–2 line functions for the sake of abstraction" guideline. The duplicated two-liner is clear and self-documenting at each call site. **Skip — not worth the indirection.**

______________________________________________________________________

## R5. `browser_ui.py`: `_check_stop` has two near-identical branches

**Files:** `browser_ui.py` (lines ~180–185)

**Problem:**

```python
def _check_stop(self) -> None:
    ev = getattr(self._thread_local, "stop_event", None)
    if ev is not None:
        if ev.is_set():
            raise KeyboardInterrupt("Agent stop requested")
    elif self.stop_event.is_set():
        raise KeyboardInterrupt("Agent stop requested")
```

The thread-local stop event is checked first, with a fallback to the instance-level one. This can be simplified.

**Fix:**

```python
def _check_stop(self) -> None:
    ev = getattr(self._thread_local, "stop_event", None) or self.stop_event
    if ev.is_set():
        raise KeyboardInterrupt("Agent stop requested")
```

**Savings:** ~3 lines, clearer intent.

______________________________________________________________________

## R6. `SorcarTab.ts` + `TabManager`: Duplicated commit-message timeout Promise pattern

**Files:** `SorcarTab.ts` (lines ~485–500, ~730–745)

**Problem:** `SorcarTab.generateCommitMessage()` and `TabManager.generateCommitMessage()` both contain an identical ~15-line Promise pattern:

```typescript
return new Promise<void>((resolve) => {
    let resolved = false;
    const done = () => {
        if (resolved) return;
        resolved = true;
        this._commitPending = false;
        disposable.dispose();
        clearTimeout(timer);
        resolve();
    };
    const disposable = this._onCommitMessage.event(() => done());
    token?.onCancellationRequested(() => done());
    const timer = setTimeout(done, 30_000);
});
```

**Fix:** Extract a private helper function at module level:

```typescript
function commitMessagePromise(
    commitEvent: vscode.Event<{ message: string; error?: string }>,
    onDone: () => void,
    token?: vscode.CancellationToken,
): Promise<void> {
    return new Promise<void>((resolve) => {
        let resolved = false;
        const done = () => {
            if (resolved) return;
            resolved = true;
            onDone();
            disposable.dispose();
            clearTimeout(timer);
            resolve();
        };
        const disposable = commitEvent(() => done());
        token?.onCancellationRequested(() => done());
        const timer = setTimeout(done, 30_000);
    });
}
```

Both callers become:

```typescript
return commitMessagePromise(
    this._onCommitMessage.event,
    () => { this._commitPending = false; },
    token,
);
```

**Savings:** ~15 lines, eliminates duplicated bug-prone timeout logic.

______________________________________________________________________

## R7. `main.js`: Massive duplication in `renderAdjacentTask` — event processing logic

**Files:** `main.js` (lines ~130–200)

**Problem:** `renderAdjacentTask` contains a full copy of the "LLM panel" state management and event routing logic from `processOutputEvent`. It creates its own `adjState`, `adjLlmPanel`, `adjLlmPanelState`, `adjLastToolName`, `adjPendingPanel` variables and processes events with identical branching logic. It also duplicates the rendering of `task_done`/`task_error`/`task_stopped` and `followup_suggestion` events.

**Current pattern (duplicated ~40 lines):**

```javascript
var adjState = mkS();
var adjLlmPanel = null;
var adjLlmPanelState = mkS();
var adjLastToolName = '';
var adjPendingPanel = false;
events.forEach(function(ev) {
    var t = ev.type;
    if (t === 'task_done' || t === 'task_error' || ...) { /* render */ return; }
    if (t === 'followup_suggestion') { /* render */ return; }
    // Mirror processOutputEvent logic
    if (t === 'tool_call') { adjLastToolName = ev.name || ''; ... }
    if (t === 'tool_result' && adjLastToolName !== 'finish') { adjPendingPanel = true; }
    if (adjPendingPanel && (t === 'thinking_start' || t === 'text_delta')) { ... }
    var target = container, tState = adjState;
    if (adjLlmPanel && ...) { target = adjLlmPanel; tState = adjLlmPanelState; }
    handleOutputEvent(ev, target, tState);
});
```

This is nearly identical to `processOutputEvent` and `replayTaskEvents`.

**Fix:** Extract a `replayEventsInto(container, events, opts)` function that encapsulates the full event replay loop. Parameters: `container` (the DOM element to render into), `events` (the event list), and an `opts` object with flags like `{ collapseAll: true }`.

The extracted function handles:

- Creating local state (`mkS`, llmPanel, etc.)
- The tool_call → tool_result → thinking_start panel logic
- `task_done`/`task_error`/`task_stopped` banner rendering
- `followup_suggestion` rendering
- Calling `handleOutputEvent` for each streaming event

This function is then used by:

1. `replayTaskEvents(events)` → calls `replayEventsInto(O, events, { collapseAll: true })`
1. `renderAdjacentTask(direction, task, events)` → calls `replayEventsInto(container, events, { collapseAll: true })`

**Savings:** ~50 lines, eliminates the biggest single redundancy in the codebase.

______________________________________________________________________

## R8. `main.js`: History reset pattern repeated 3 times

**Files:** `main.js` (lines ~960, ~1110, ~1120)

**Problem:** The same 4-line sequence appears in `refreshHistory()`, the `historyBtn` click handler, and the `historySearch` input handler:

```javascript
historyOffset = 0;
historyHasMore = true;
historyLoading = false;
historyGeneration++;
```

**Fix:** Extract `resetHistoryPagination()` and call it from all three sites.

```javascript
function resetHistoryPagination() {
    historyOffset = 0;
    historyHasMore = true;
    historyLoading = false;
    historyGeneration++;
}
```

**Savings:** ~8 lines.

______________________________________________________________________

## R9. `main.js`: Repeated `inp.focus()` with retry pattern

**Files:** `main.js` (multiple locations)

**Problem:** The pattern `inp.focus(); setTimeout(function() { inp.focus(); }, 100); setTimeout(function() { inp.focus(); }, 300);` appears in the handlers for both `appendToInput` and `focusInput` events.

**Fix:** Extract `focusInputWithRetry()`:

```javascript
function focusInputWithRetry() {
    inp.focus();
    setTimeout(function() { inp.focus(); }, 100);
    setTimeout(function() { inp.focus(); }, 300);
}
```

**Savings:** ~6 lines.

______________________________________________________________________

## R10. `main.js`: `task_error` / `task_stopped` rendering duplicated between `handleEvent` and `renderAdjacentTask`

**Files:** `main.js`

**Problem:** Both `handleEvent` (for the main view) and `renderAdjacentTask` (for adjacent tasks) render error/stopped banners with identical HTML construction. This is addressed by R7 (the extracted `replayEventsInto` function).

**Already covered by R7.** No separate action needed.

______________________________________________________________________

## R11. `SorcarTab.ts`: `_handleMessage` mergeAction routing duplicates method dispatch

**Files:** `SorcarTab.ts` (lines ~380–395)

**Problem:** The mergeAction handler in `_handleMessage` has an 8-way if/else chain mapping action strings to MergeManager method calls:

```typescript
if (mAction === 'accept') this._mergeManager.acceptChange();
else if (mAction === 'reject') this._mergeManager.rejectChange();
else if (mAction === 'prev') this._mergeManager.prevChange();
else if (mAction === 'next') this._mergeManager.nextChange();
else if (mAction === 'accept-all') this._mergeManager.acceptAll();
else if (mAction === 'reject-all') this._mergeManager.rejectAll();
else if (mAction === 'accept-file') this._mergeManager.acceptFile();
else if (mAction === 'reject-file') this._mergeManager.rejectFile();
```

**Fix:** Use a dispatch map:

```typescript
case 'mergeAction': {
    const mergeDispatch: Record<string, () => void> = {
        'accept': () => this._mergeManager.acceptChange(),
        'reject': () => this._mergeManager.rejectChange(),
        'prev': () => this._mergeManager.prevChange(),
        'next': () => this._mergeManager.nextChange(),
        'accept-all': () => this._mergeManager.acceptAll(),
        'reject-all': () => this._mergeManager.rejectAll(),
        'accept-file': () => this._mergeManager.acceptFile(),
        'reject-file': () => this._mergeManager.rejectFile(),
    };
    const action = (message as any).action;
    const handler = mergeDispatch[action];
    if (handler) handler();
    else if (action === 'all-done') {
        this._agentProcess.sendCommand({ type: 'mergeAction', action: 'all-done' });
    }
    break;
}
```

**Savings:** ~4 lines net, better maintainability, clearer mapping.

______________________________________________________________________

## R12. `DependencyInstaller.ts`: Shell-type detection logic duplicated between `addToShellRc` and `ensurePathInShellRc`

**Files:** `DependencyInstaller.ts` (lines ~590, ~650)

**Problem:** Both functions compute `isPs1` and `isFish` from the rc path, and both have shell-type-specific formatting for export lines. This is a minor duplication (2 lines each) but the shell-specific string formatting patterns are substantively duplicated.

**Assessment:** The functions do different things (one sets a variable, one adds a PATH entry). They share the detection logic but the formatting is distinct enough that extracting a helper would obscure the code. **Skip — not worth the indirection.**

______________________________________________________________________

## R13. `main.js`: `followup_suggestion` rendering duplicated

**Files:** `main.js` (two locations in `handleEvent` and `renderAdjacentTask`)

**Problem:** The followup suggestion bar is rendered with identical HTML in both places:

```javascript
var fu = mkEl('div', 'followup-bar');
fu.innerHTML = '<span class="fu-label">Suggested next</span>'
    + '<span class="fu-text">' + esc(ev.text) + '</span>';
```

The main event handler version also adds a click handler; the adjacent version does not.

**Fix:** Already addressed by R7 — the extracted `replayEventsInto` function will handle followup_suggestion rendering once.

______________________________________________________________________

## R14. `browser_ui.py`: `print()` method — long if/elif chain could use early returns more consistently

**Files:** `browser_ui.py` (lines ~200–290)

**Problem:** The `print()` method has 9 branches in a long `if`/`elif` chain. Each branch returns `""`. The method is already well-structured with early returns, but the final `return ""` at the bottom is redundant since every branch already returns. This is a style issue, not a real redundancy.

**Assessment:** The current structure is readable. **Skip.**

______________________________________________________________________

## Execution Plan (Priority Order)

| # | Item | File(s) | Savings | Risk |
|---|------|---------|---------|------|
| 1 | R7 | `main.js` | ~50 lines | Medium — large refactor, needs careful testing |
| 2 | R6 | `SorcarTab.ts` | ~15 lines | Low |
| 3 | R1 | `server.py` | ~5 lines | Low |
| 4 | R2 | `server.py` | ~10 lines | Low |
| 5 | R3 | `server.py` | ~6 lines | Low |
| 6 | R5 | `browser_ui.py` | ~3 lines | Low |
| 7 | R8 | `main.js` | ~8 lines | Low |
| 8 | R9 | `main.js` | ~6 lines | Low |
| 9 | R11 | `SorcarTab.ts` | ~4 lines | Low |

**Skipped:** R4 (1-line method body, not worth indirection), R10 (covered by R7), R12 (not worth indirection), R13 (covered by R7), R14 (style only).

**Total estimated savings:** ~107 lines net (after adding new helper functions/methods).
