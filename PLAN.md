# Race Condition Fix Plan — VS Code Extension

Files in scope: `src/kiss/agents/vscode/src/*.ts`, `src/kiss/agents/vscode/media/main.js`,
`src/kiss/agents/vscode/server.py`

---

## Race 1 — Python TOCTOU on `_stop_event` and `_user_answer_event`

**File:** `server.py` — `_stop_task()`, `_handle_command()` for `userAnswer`

**Bug:** Both `_stop_event` and `_user_answer_event` are read-then-used without
holding a lock, while the task thread can set them to `None` in its `finally` block.

```python
# _stop_task (main thread)
def _stop_task(self) -> None:
    if self._stop_event:            # ← reads attribute, sees Event
        # THREAD SWITCH: task thread sets self._stop_event = None
        self._stop_event.set()      # ← reads attribute AGAIN → NoneType has no .set()
```

Same pattern for `_user_answer_event`:

```python
elif cmd_type == "userAnswer":
    self._user_answer = cmd.get("answer", "")
    if self._user_answer_event:     # ← sees Event
        # THREAD SWITCH: task finishes, sets self._user_answer_event = None
        self._user_answer_event.set()  # ← AttributeError
```

**Fix:** Capture the reference in a local variable before using it:

```python
def _stop_task(self) -> None:
    ev = self._stop_event
    if ev:
        ev.set()
    # ...
```

```python
elif cmd_type == "userAnswer":
    self._user_answer = cmd.get("answer", "")
    ev = self._user_answer_event
    if ev:
        ev.set()
```

---

## Race 2 — Stale autocomplete popup after `@` is deleted

**File:** `main.js` — `renderAutocomplete()`, `checkAutocomplete()`

**Bug:** When the user types `@foo`, a `getFiles` message is sent. If the user then
deletes the `@` before the response arrives, the `input` event calls
`checkAutocomplete()` → `hideAC()`. But then the stale `files` response arrives and
`renderAutocomplete()` shows the dropdown again, causing a flash of stale results.

Sequence:
1. User types `@foo` → sends `getFiles(prefix:'foo')`
2. User deletes `@foo` → `input` event → `checkAutocomplete()` → `hideAC()`
3. Stale `files` response arrives → `renderAutocomplete(files)` → shows dropdown

**Fix:** Before rendering, check that the `@` context still exists:

```js
case 'files':
    if (getAtCtx()) {
        renderAutocomplete(ev.files || []);
    }
    break;
```

---

## Race 3 — `newConversation()` sends `newChat` while task is still stopping

**File:** `SorcarPanel.ts` — `newConversation()`

**Bug:** `newConversation()` calls `stop()` then immediately sends `newChat`.
On the Python side, `newChat` is guarded:

```python
elif cmd_type == "newChat":
    if not (self._task_thread and self._task_thread.is_alive()):
        self.agent.new_chat()
```

If the task thread hasn't exited yet (stop is async), `newChat` is silently
ignored. The UI shows "new chat" but the agent didn't actually start one.
When the task eventually finishes, `tasks_updated` is broadcast, but the
webview has already been cleared, causing a confusing state.

**Fix:** Don't send `newChat` synchronously. Instead, let the `status: running: false`
event (which the agent sends when the task ends) trigger the `newChat` command.
Use a flag to indicate a pending new-chat:

```typescript
public newConversation(): void {
    this.sendToWebview({ type: 'status', running: false });
    this.sendToWebview({ type: 'clearChat' });
    if (this._isRunning) {
        this._pendingNewChat = true;
        this._agentProcess.stop();
    } else {
        this._agentProcess.sendCommand({ type: 'newChat' });
    }
    this._isRunning = false;
}
```

Then in the `'status'` handler in the constructor:

```typescript
if (msg.type === 'status') {
    this._isRunning = msg.running;
    if (!msg.running && this._pendingNewChat) {
        this._pendingNewChat = false;
        this._agentProcess.sendCommand({ type: 'newChat' });
    }
    // ...
}
```

---

## Race 4 — `_last_active_file` written without `_state_lock` in `_run_task_inner`

**File:** `server.py` — `_run_task_inner()`, `_handle_command()` for `complete`

**Bug:** `_run_task_inner` sets `self._last_active_file` without acquiring
`_state_lock`, but `complete` command handling reads it under `_state_lock`.
If a `complete` request arrives while a task is starting, the two threads
race on `_last_active_file`.

```python
# _run_task_inner (task thread) — NO lock
self._last_active_file = active_file or ""

# _handle_command complete (main thread) — with lock
with self._state_lock:
    if active_file:
        self._last_active_file = active_file
```

**Fix:** Acquire `_state_lock` in `_run_task_inner` when writing these fields:

```python
with self._state_lock:
    self._last_active_file = active_file or ""
```

---

## Race 5 — MergeManager `_afterHunkAction` doesn't await navigation

**File:** `MergeManager.ts` — `_afterHunkAction()`, `nextChange()`, `prevChange()`

**Bug:** `nextChange()` and `prevChange()` return `void` (not `Promise<void>`)
but call the async `_navigateHunk()`. Since `_afterHunkAction` calls
`nextChange()` without awaiting, the `_withHunkGuard` releases
`_hunkOpInProgress` before navigation completes. A rapid second
accept/reject could start while the first navigation is still opening
the document.

```typescript
// nextChange doesn't propagate the promise
nextChange(): void {
    this._navigateHunk(1);  // fire-and-forget
}

// _afterHunkAction doesn't await
private _afterHunkAction(fp: string): void {
    this._refreshDeco(fp);
    if (Object.keys(this._ms).length > 0) {
        this.nextChange();  // not awaited → guard releases early
    }
    // ...
}
```

This is mitigated by `_navSeq` (stale navigations are cancelled), and
`_curHunk` is set synchronously before awaits, so data corruption is
unlikely. But the user can observe jumpy navigation: two documents
opening in quick succession, decorations flashing.

**Fix:** Make `nextChange()`, `prevChange()`, and `_afterHunkAction` async and
propagate the promise:

```typescript
async nextChange(): Promise<void> {
    await this._navigateHunk(1);
}

async prevChange(): Promise<void> {
    await this._navigateHunk(-1);
}

private async _afterHunkAction(fp: string): Promise<void> {
    this._refreshDeco(fp);
    if (Object.keys(this._ms).length > 0) {
        await this.nextChange();
    } else {
        this._checkAllDone();
    }
}
```

And in `_applyHunkAction`, await `_afterHunkAction`:

```typescript
private async _applyHunkAction(...): Promise<void> {
    // ... splice hunks ...
    if (!s.hunks.length) delete this._ms[fp];
    this.emit('hunkProcessed');
    await this._afterHunkAction(fp);
}
```

---

## Race 6 — Two providers sharing one MergeManager

**File:** `extension.ts`, `SorcarPanel.ts`

**Bug:** Both `primaryProvider` and `secondaryProvider` share the same
`MergeManager` instance. If both providers have active agent processes
and both produce `merge_data` events, `openMerge()` would be called
twice from different sources. While `openMerge`'s `_mergeInProgress`
flag serializes calls, the second merge would overwrite the first's state
(`_ms` is cleared in `_doOpenMerge` at the top).

More practically: merge actions (accept/reject) go through
`SorcarPanel._handleMessage` → `this._mergeManager.acceptChange()`.
Both providers route to the same merge manager, so both can issue
accept/reject — but they also both send `mergeAction: all-done` to
their respective Python agent processes. Only one of them owns the
merge session.

**Fix:** Track which provider owns the current merge session. Only allow
merge actions from the owning provider. In `SorcarPanel.ts`, when
`merge_data` arrives, have the panel claim ownership of the merge session.
In `extension.ts`, route the `allDone` event only to the owner.

Alternatively (simpler): since `getActiveProvider()` already returns
the preferred provider, ensure merge actions only go through the active
provider. This is already partially done for `allDone`. The remaining
risk is the user interacting with the non-active panel's merge toolbar.

Simplest fix: when a merge_data message arrives for a provider, ignore
it if that provider is not the active provider.

---

## Race 7 — `_doOpenMerge` revert may target wrong editor

**File:** `MergeManager.ts` — `_doOpenMerge()`

**Bug:** After `showTextDocument(doc)`, the code checks `doc.isDirty` and
runs `workbench.action.files.revert`. This command reverts the *active*
editor, which might have changed if another `showTextDocument` call
or user action happened between the two lines.

```typescript
const ed = await vscode.window.showTextDocument(doc, { preview: false });
if (doc.isDirty) {
    await vscode.commands.executeCommand('workbench.action.files.revert');
}
```

**Fix:** Use the `TextEditor` instance returned by `showTextDocument` to
verify the active editor hasn't changed, or use a file-specific revert
approach (write the file to disk and reopen). The simplest fix:

```typescript
if (doc.isDirty) {
    // Ensure this document is still the active one before reverting
    if (vscode.window.activeTextEditor?.document === doc) {
        await vscode.commands.executeCommand('workbench.action.files.revert');
    }
}
```

---

## Race 8 — `AgentProcess.dispose()` vs. buffered stdout events

**File:** `AgentProcess.ts` — `dispose()`

**Bug:** After `dispose()` sets `this.process = null`, buffered stdout data
may still trigger `handleStdout()` → `emit('message', ...)`. The listeners
are removed *after* the null assignment, so there's a window where events
are emitted to listeners of a logically-disposed object.

```typescript
dispose(): void {
    if (this.process) {
        const proc = this.process;
        this.process = null;
        // ← buffered stdout callback can fire here
        try { proc.stdin?.end(); } catch {}
        try { proc.kill('SIGTERM'); } catch {}
    }
    this.removeAllListeners();  // ← too late?
}
```

**Fix:** Remove listeners first, then kill the process:

```typescript
dispose(): void {
    this.removeAllListeners();
    if (this.process) {
        const proc = this.process;
        this.process = null;
        try { proc.stdin?.end(); } catch {}
        try { proc.kill('SIGTERM'); } catch {}
        setTimeout(() => { try { proc.kill('SIGKILL'); } catch {} }, 2000);
    }
}
```

---

## Race 9 — `_resolveAll` partial failure leaves documents inconsistent

**File:** `MergeManager.ts` — `_resolveAll()`

**Bug:** If `_deleteFileHunks()` throws for one file, the `finally` block
clears *all* state (`this._ms = {}`). Remaining files' hunks are lost,
but the edits weren't applied, leaving those documents with inserted
old-lines that can never be removed via the merge UI.

```typescript
try {
    for (const fp of fps) {
        await this._deleteFileHunks(fp, countProp, startProp);  // might throw
    }
} finally {
    this._ms = {};  // ← remaining files' hunks are now lost
}
```

**Fix:** Track which files were successfully processed and only clear those.
On error, keep remaining files in `_ms` so the user can still resolve them:

```typescript
private async _resolveAll(...): Promise<void> {
    const fps = Object.keys(this._ms);
    const processedFps: string[] = [];
    try {
        for (const fp of fps) {
            await this._deleteFileHunks(fp, countProp, startProp);
            processedFps.push(fp);
        }
    } finally {
        for (const fp of processedFps) {
            delete this._ms[fp];
        }
        // ... refresh decos, check all done ...
    }
}
```

---

## Race 10 — `deactivate()` doesn't prevent post-dispose message handling

**File:** `extension.ts` — `deactivate()`, `SorcarPanel.ts` — `sendToWebview()`

**Bug:** After `deactivate()` calls `dispose()` on both providers, the
`_view` field is never cleared. If a pending event fires on the event
loop after dispose, `sendToWebview()` would call `postMessage()` on a
disposed webview, which could throw.

**Fix:** Clear `_view` in `SorcarPanel.dispose()`:

```typescript
public dispose(): void {
    this._view = undefined;
    this._activeEditorDisposable?.dispose();
    this._agentProcess.dispose();
    this._mergeManager.dispose();
    this._onCommitMessage.dispose();
}
```

---

## Race 11 — `_await_user_response` clear/set ordering

**File:** `server.py` — `_await_user_response()`, `_handle_command()` for `userAnswer`

**Bug:** Between the `askUser` broadcast and `_user_answer_event.clear()`,
a very fast (automated or programmatic) `userAnswer` could arrive and set
the event. The subsequent `clear()` would erase this signal, and `wait()`
would block forever.

```python
def _ask_user_question(self, question: str) -> str:
    self.printer.broadcast({"type": "askUser", "question": question})
    # ← userAnswer arrives here, sets _user_answer_event
    self._await_user_response()  # ← clear() erases the signal, wait() blocks forever
```

**Fix:** Create a fresh Event for each question instead of reusing and clearing:

```python
def _ask_user_question(self, question: str) -> str:
    self._user_answer_event = threading.Event()
    self.printer.broadcast({"type": "askUser", "question": question})
    self._user_answer_event.wait()
    return self._user_answer
```

This way there's no clear/set race — the event is brand new and not set.
The `userAnswer` handler remains unchanged (it calls `.set()` on whatever
event is current).

---

## Race 12 — `generateCommitMessage` can match stale `commitMessage` event

**File:** `SorcarPanel.ts` — `generateCommitMessage()`

**Bug:** The promise listens for *any* `commitMessage` event on
`_onCommitMessage`. If a previous in-flight commit generation finishes
after a new one starts, the new promise could resolve with the old result.
There's no correlation ID.

```typescript
const disposable = this._onCommitMessage.event(() => done());
```

**Fix:** Add a generation counter and include it in the command. The
Python server would echo it back. The listener would check the generation
matches. Alternatively, since `_commitPending` prevents concurrent calls,
and the Python `_generate_commit_message` is simple enough, this race
requires very specific timing. A simpler fix: the `_commitPending` guard
already prevents concurrent calls, so the only way to get a stale result
is if the previous generation's response arrives after cancellation.
Clear the listener on cancellation, which is already done via `done()`.

**Verdict:** Low risk. The `_commitPending` guard + `done()` cleanup is
sufficient. Document the design assumption that only one generation runs
at a time.

---

## Summary — Priority Order

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | **Critical** | `server.py` | TOCTOU on `_stop_event`/`_user_answer_event` → crash |
| 11 | **Critical** | `server.py` | `_await_user_response` clear/set ordering → deadlock |
| 2 | **High** | `main.js` | Stale autocomplete popup after `@` deleted |
| 3 | **High** | `SorcarPanel.ts` | `newConversation` + `newChat` while task stopping |
| 4 | **Medium** | `server.py` | `_last_active_file` unprotected write |
| 5 | **Medium** | `MergeManager.ts` | `_afterHunkAction` doesn't await navigation |
| 8 | **Medium** | `AgentProcess.ts` | `dispose()` listener removal ordering |
| 10 | **Medium** | `SorcarPanel.ts` | `_view` not cleared on dispose |
| 9 | **Low** | `MergeManager.ts` | `_resolveAll` partial failure |
| 7 | **Low** | `MergeManager.ts` | `_doOpenMerge` revert wrong editor |
| 6 | **Low** | `extension.ts` | Two providers sharing MergeManager |
| 12 | **Info** | `SorcarPanel.ts` | Stale `commitMessage` after cancellation |
