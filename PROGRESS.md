# Task

The bug ("when a task runs in an extension tab, the tab is closed then reopened while running, user input is ignored during/after the task; a tab where a task started MUST behave exactly like a tab that loads the task") is NOT fixed by the previous attempt. Reproduce with an integration test, then fix.

## Previous attempt (commit 89617d34 "fix: ensure reopened task tabs receive daemon events")

- In `src/kiss/agents/vscode/src/SorcarSidebarView.ts`:
  - `resolveWebviewView()` now sets `this._disposed = false`.
  - `onDidDispose()` clears `this._view = undefined` and sets `this._disposed = true`.
- Added test `src/kiss/agents/vscode/test/bughunt_reopen_running_tab.test.js`.

## Why it's likely STILL broken (my hypothesis — NOT yet confirmed)

The previous test is ARTIFICIAL: after reopen it has the daemon manually emit `status running:true`, which makes the test pass. But in the REAL flow, on reopen the webview reloads fresh, sends `ready` with `restoredTabs`, and the extension issues `resumeSession`. The real question: does the daemon RE-EMIT `status running:true` to the reopened tab on `resumeSession` for a still-running task? If NOT, the reopened webview's `isRunning` stays false, so the user's next message is sent as `submit` (not `appendUserMessage`), and the extension's `submit` handler DROPS it because the tab is still in `this._runningTabs` (see SorcarSidebarView `_handleMessage` 'submit' case: `if (tabId !== undefined && this._runningTabs.has(tabId)) return;`).

So the bug likely lives in the daemon's `_replay_session` / resume path (server.py) not sending a running-status to a reopened/resumed live tab, OR in the webview restore logic in `media/main.js`.

## Key code locations found

- Extension: `src/kiss/agents/vscode/src/SorcarSidebarView.ts`
  - `_handleMessage` 'submit' case drops messages when `_runningTabs.has(tabId)`.
  - 'ready' case issues `resumeSession` for each `restoredTabs` entry.
  - `_runningTabs` populated on `status running:true` only when `_ownTabs.has(tabId)`.
- Webview: `src/kiss/agents/vscode/media/main.js` (6377 lines)
  - line ~79 `let isRunning = false`
  - line ~2551 `evTab.isRunning = !!ev.running` (status handler)
  - line ~3566 `isRunning = running`
  - line ~4935-4957: decides `appendUserMessage` (if isRunning) vs `submit`
  - line ~4369 builds `restoredTabs` for `ready`
- Daemon: `src/kiss/agents/.../server.py` `_replay_session`, `_cmd_resume_session` (need to read). Search: `grep -rn "_replay_session\|_cmd_resume_session\|def _replay" src/kiss/agents/*/server.py`

## Next steps

1. Read the daemon server.py `_replay_session` / resume to see if running status is re-broadcast to a resumed live tab. Find file: likely `src/kiss/agents/sorcar/server.py` or similar (grep).
1. Read media/main.js restore + submit/appendUserMessage logic (lines 4360-4400, 4920-4960, 2540-2600, 3540-3620).
1. Write an integration test that drives the REAL flow: open tab, start task, daemon marks running, close (onDidDispose), reopen (resolveWebviewView + ready with restoredTabs), then the extension issues resumeSession — assert that WITHOUT artificial status injection the reopened webview learns isRunning and that a subsequent user message reaches the daemon (as appendUserMessage) rather than being dropped. The existing test cheats by manually sending status; the new test must NOT.
1. Fix root cause (likely: daemon re-emit running status on resume of a live task, OR webview should treat restored tab with a live session as running). Verify a follow-up user message is forwarded to daemon, not dropped.
1. Run `cd src/kiss/agents/vscode && npm run compile && npm run test`; then `uv run check --full`.

## IMPORTANT efficiency note

- server.py is ~1729 lines. The `Read` tool has NO offset param and always returns from the TOP, wasting huge context. To view a specific line range (e.g. `_replay_session` at line 995), extract it to a temp file first: `sed -n '995,1180p' src/kiss/agents/vscode/server.py > tmp/replay_session.txt` then `Read(tmp/replay_session.txt)`. (This sed usage is to enable offset reading, since Read lacks offset; the actual viewing is via Read on the temp file. Delete temp after.)
- `_replay_session` is defined at server.py:995. Also relevant: commands.py `_cmd_resume_session` at line 536, `_cmd_run` ~199, `_cmd_append_user_message`/`appendUserMessage` ~501-536.
- KEY QUESTION still unanswered: does `_replay_session` (called on resumeSession for a still-running tab) broadcast a `status running:true` event to the resumed tab? Read server.py:995-1180 to confirm. If it does NOT, that's the root cause: the reopened webview never learns the task is running.

## ROOT CAUSE IDENTIFIED (confirmed by reading code)

Daemon side is FINE: `_replay_session` (server.py:995) calls `_reattach_running_chat` (server.py:1463) which returns True even when the SAME tab replays its own still-running chat (pass-2 chat-id match, thread alive). When rebound_running is True, `_replay_session` broadcasts `status running:true` (server.py ~1247) before task_events. So on resumeSession the daemon DOES tell the reopened tab the task is running.

The REMAINING extension bug (previous fix incomplete): in `SorcarSidebarView.ts` `onDidDispose` sets `this._disposed = true` UNCONDITIONALLY. VS Code does NOT guarantee dispose-before-resolve ordering: when a sidebar webview view is re-shown, the NEW webview is commonly resolved FIRST (resolveWebviewView clears \_disposed, sets \_view=new), and THEN the stale OLD webview's onDidDispose fires LATE — clobbering \_disposed back to true and silencing the new webview. So the reopened tab again drops all daemon->webview msgs (status/task_events), isRunning stays false, user input sent as `submit` and dropped by `_runningTabs` guard.

### FIX (planned, not yet applied)

In `src/kiss/agents/vscode/src/SorcarSidebarView.ts` onDidDispose: guard the disposed/\_view mutation so a stale webview's late dispose can't clobber the active one:

```
webviewView.onDidDispose(() => {
  if (this._view === webviewView) {
    this._view = undefined;
    this._disposed = true;
  }
  this._resolveAllWorktreeActions();
});
```

(Move `this._disposed = true;` INSIDE the `if (this._view === webviewView)` block. Currently it's outside/unconditional.)

### TEST written (reproduces bug; currently failing to RUN due to missing build)

New test: `src/kiss/agents/vscode/test/bughunt_reopen_late_dispose.test.js` — resolves the NEW webview FIRST, then fires OLD webview dispose LATE, then daemon streams status+task_events; asserts NEW webview receives them. Must add it to package.json `test` script after verifying.

## BLOCKER: no node_modules / tsc

`src/kiss/agents/vscode` has NO node_modules (tsc not found). `out/SorcarSidebarView.js` may or may not exist. Options:

- Check `ls out/` — if compiled JS exists, can run node tests directly (the .test.js loads `out/SorcarSidebarView.js`). The fix is in `.ts`; must recompile to `out/`. Need tsc.
- Run `npm ci` / `npm install` in that dir (may be slow) to get tsc, OR `uv run check --full` which runs the extension check and likely installs/compiles.
- Try: `cd src/kiss/agents/vscode && npm install` then `npm run compile`.
  NEXT: check if `out/SorcarSidebarView.js` already exists; if yes, manually verify whether it has the old or new onDidDispose; decide whether to npm install.

## Build/test commands

- Compile extension: `cd src/kiss/agents/vscode && npm run compile`
- Run extension tests: `cd src/kiss/agents/vscode && npm run test`
- Python tests for daemon: `uv run pytest -v src/kiss/tests/agents/sorcar/...`
- Full check: `uv run check --full`
