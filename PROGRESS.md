# PROGRESS — Reopen running tab ignores user input (Task 3)

## Task

Bug NOT fixed after 2 prior attempts: a Sorcar extension tab that started a running task,
when closed and reopened while running, ignores user input during/after the task.
Must reproduce with an integration test, then fix.

## Prior fixes (already in tree, on SorcarSidebarView.ts)

1. `resolveWebviewView` sets `_disposed = false` on reopen.
1. `onDidDispose` guards `_view === webviewView` before setting `_disposed=true` / clearing `_view`
   (handles late dispose of stale webview).
   Both prior reproduction tests STUB the daemon and only assert the EXTENSION forwards
   `status`/`task_events` to the new webview. They pass. So the remaining bug is NOT pure
   extension forwarding.

## Key architecture facts learned

- Extension: `src/kiss/agents/vscode/src/SorcarSidebarView.ts` (compiled to out/SorcarSidebarView.js).
- Webview JS: `media/main.js` (6377 lines).
  - line ~79 `isRunning`; ~263 isActiveTabRunning; ~2551 `evTab.isRunning = !!ev.running` on status.
  - line ~4935-4960: on send, if `isRunning` -> sends `appendUserMessage`, else `submit`.
  - line ~4369 builds `restoredTabs` for `ready`; ~5475 resumeSession for sub-agents.
- Extension `_handleMessage`:
  - `submit` handler: `if (tabId in _runningTabs) return;` (DROPS submit while running).
  - `appendUserMessage` is in FORWARDED_COMMANDS -> forwarded to daemon.
  - `ready` with restoredTabs -> sends `resumeSession {chatId, tabId}` per tab.
- Extension registered with `retainContextWhenHidden: true` (extension.ts:31).
- Daemon: `server.py` `_replay_session` (line 995), `_reattach_running_chat` (1463),
  `_cmd_resume_session`. These reattach a still-running chat and broadcast status+task_events.

## Hypothesis for remaining bug (NEEDS VERIFICATION)

The prior tests stub the daemon, so they don't test the REAL daemon replay OR the webview JS
state machine. Likely remaining bug is one of:
(a) Daemon `_replay_session`/`_reattach_running_chat`: for a tab that STARTED the task and
resumes its OWN running chat, it may NOT re-broadcast `status running:true` / re-subscribe,
so reopened tab keeps isRunning=false -> next input sent as `submit` -> dropped by
`_runningTabs`. OR after task finishes the desync persists.
(b) Webview JS: after reopen + restore, the restored tab's `isRunning` and the global
`isRunning` may desync, or `_runningTabs` in extension never cleared.

## NEXT STEPS (do in fresh context)

1. Write a REAL integration test against the actual daemon (server.py) — NOT a stub — that:
   starts a task, closes the tab (webview dispose, keep AgentClient), reopens (new webview +
   ready+restoredTabs+resumeSession), then sends user input and asserts it reaches the agent
   (during running -> appendUserMessage injected; after finish -> submit accepted).
   Look at existing python daemon tests for harness; or extend the JS harness to spawn the
   real `python -m kiss ... web server` daemon instead of net.createServer stub.
1. Confirm which layer drops the input. Add sleeps to expose races if needed.
1. Fix root cause. Re-run `npm run compile`, `npm run test`, `uv run check --full`.
1. Clean up tmp files.

## Daemon dispatch map (commands.py)

- Dispatch table: `commands.py:863 _HANDLERS` (mixed into server via server.py:433).
- `_cmd_append_user_message` = commands.py:500-535. Line ~525 logs
  "appendUserMessage dropped: tab %s has no live task" — KEY: if reopened tab's
  appendUserMessage arrives but daemon has no live task for that tab, it's DROPPED.
- `_cmd_run` = commands.py:202; `_cmd_resume_session` = commands.py:536; `_cmd_stop`=283.
- task_runner.py:191 `tab.pending_user_messages.clear()`; :315 drains pending_user_messages
  before each model call. So appendUserMessage -> tab.pending_user_messages -> injected.
- \_replay_session=server.py:995, \_reattach_running_chat=server.py:1463.

## STRONG hypothesis (verify next)

The reopened tab's appendUserMessage is keyed by tabId. After reopen, if the daemon's running
task/tab state is keyed under a DIFFERENT tab identity (or \_reattach rebinds chat but the live
task's tab object that \_cmd_append_user_message looks up is not the same one running the loop),
the appendUserMessage is dropped at commands.py:525 ("has no live task"). Need to read
\_cmd_append_user_message + how it resolves the tab + \_reattach_running_chat to confirm the tab
object identity / chat_id linkage survives close+reopen for a tab that STARTED the task.

## NEXT (fresh context)

1. Read commands.py 500-560 (\_cmd_append_user_message, \_cmd_resume_session) fully.
1. Read server.py 995-1140 (\_replay_session) and 1463-1600 (\_reattach_running_chat).
1. Determine where reopened-tab input is dropped; likely daemon tab/chat rebinding.
1. Write integration test spawning REAL daemon (look at existing pytest daemon harness in
   src/kiss/agents/vscode/ or repo) reproducing close+reopen+input-during-run + input-after-finish.
1. Fix root cause; npm run compile; npm run test; uv run check --full; clean tmp.

## Files

- src/kiss/agents/vscode/src/SorcarSidebarView.ts
- src/kiss/agents/vscode/media/main.js
- src/kiss/agents/vscode/server.py
- existing tests: test/bughunt_reopen_running_tab.test.js, test/bughunt_reopen_late_dispose.test.js
- package.json `test` script wires the bughunt tests.

## UPDATE (session 2 findings)

- npm deps installed in src/kiss/agents/vscode (node_modules now present). `npm run compile` works.
- Both existing reopen tests PASS (extension forwarding fixed by prior commits).
- jsdom harness exists (see test/tab_timer_per_tab.test.js): loads real chat.html+panelCopy.js+main.js,
  stub acquireVsCodeApi. main.js exposes window.\_\_kissTest exports (setRunningState, getActiveTabId,...).
- WROTE test/bughunt_reopen_input_webview.test.js: real main.js, carries vscode.getState across two
  webviews (close+reopen). It PASSES: after reopen, during-run typed msg -> appendUserMessage; after
  status running:false -> submit. So webview layer alone is CORRECT too.
- GOTCHA: assert.deepStrictEqual on jsdom-realm objects fails on prototype mismatch; use JSON.stringify compare.
- sendMessage() in main.js (line ~4925): uses GLOBAL isRunning; isRunning -> appendUserMessage else submit.
- restoredTabs builder (main.js ~4369): filters tabs with backendChatId, sends {tabId:t.id, chatId:t.backendChatId}.
- backendChatId set in 'clear' handler (main.js 2691-2693) which DOES call persistTabState() right after.
- Daemon \_replay_session broadcasts status running:true (rebound_running branch) BEFORE task_events when
  \_reattach_running_chat finds a live source (chat-id fallback matches the started tab's own running_agent_states).
- \_cmd_append_user_message (commands.py 500): looks up running_agent_states.get(tab_id); drops if tab None
  or not is_task_active.

## CONCLUSION: bug is a CROSS-LAYER gap, not in either layer alone.

NEXT: build a COMBINED integration test: real SorcarSidebarView (out/) + real main.js (jsdom) wired together
(webview.postMessage -> view.\_handleMessage; view.\_sendToWebview -> webview 'message' event) + stub UDS daemon.
Do full close+reopen-during-run and assert the DAEMON receives appendUserMessage (during) and run (after finish).
This should expose the real drop. Then fix root cause.

## ROOT CAUSE FOUND + FIXED (session 2)

Reproduction probe test_reopen_started_tab_resume.py::test_run_for_busy_tab_injects_prompt_instead_of_dropping
FAILED on original code -> confirmed bug: daemon \_cmd_run SILENTLY DROPS a `run` for a tab whose
task_thread is not None. After close+reopen, a re-opened webview can momentarily still think the task
is idle (before resume's status:true arrives) and send the typed text as submit->run; the daemon dropped it.

FIXES:

1. commands.py \_cmd_run: when tab.task_thread is not None, instead of `return`, inject the prompt into
   tab.pending_user_messages (if non-blank and is_task_active) and broadcast a `prompt` echo OUTSIDE the
   lock; never start a 2nd task. Restructured guard into if-busy/else-normal with thread/inject_prompt
   locals; broadcast clear+thread.start() only on the normal path.
1. src/SorcarSidebarView.ts submit handler: instead of `return` when tabId in \_runningTabs, forward the
   text to the daemon as appendUserMessage (if non-blank). Belt-and-suspenders for the extension-layer
   desync. Recompiled.

NEW TESTS (all pass):

- src/kiss/tests/agents/sorcar/test_reopen_started_tab_resume.py (2 tests: resume broadcasts running+accepts
  append; run-for-busy-tab injects).
- src/kiss/agents/vscode/test/bughunt_reopen_input_webview.test.js (jsdom real main.js close+reopen).
- src/kiss/agents/vscode/test/bughunt_reopen_input_e2e.test.js (real ext + real webview jsdom + UDS daemon).
- src/kiss/agents/vscode/test/bughunt_submit_while_running.test.js (submit-while-running -> appendUserMessage).
  All 3 JS tests wired into package.json `test` script. `npm run test` PASSES fully.

REMAINING: run python tests + `uv run check --full`; clean tmp; finish.
