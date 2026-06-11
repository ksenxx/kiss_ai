# Task: Bug-hunt iteration 3, group E (vscode backend: server.py / commands.py / task_runner.py)

Find NEW bugs (16 already fixed in rounds 1-2 — do not re-report), reproduce each with a
failing-first integration test `src/kiss/tests/agents/vscode/test_bughunt3_<short>.py`,
then fix root cause, run impacted tests + `uv run check --full`.

## Session 1 state (context exhausted at investigation phase — NO code changes made yet)

### Files fully read & understood
- src/kiss/agents/vscode/server.py (1595 lines)
- src/kiss/agents/vscode/commands.py (743 lines)
- src/kiss/agents/vscode/task_runner.py (1148 lines)
- src/kiss/agents/sorcar/running_agent_state.py
- web_server.py dispatch path: `_ws_handler`/`_uds_handler` loop → `_dispatch_client_command`
  → `_run_cmd` → executor → `VSCodeServer._handle_command`. The `try` in both handlers wraps
  the whole `async for` receive loop, so ANY exception escaping `_handle_command`
  **terminates the entire client connection** (whole VS Code window / browser tab) and
  triggers deferred closeTab of every tab on it.

### CONFIRMED new bugs (reproduced via tmp/repro1.py, since deleted; output below)

1. **BUG-A — server.py `_handle_command` (~line 330): unhashable `type` field kills connection.**
   `cmd_type = cmd.get("type", "")` then `self._HANDLERS.get(cmd_type)` raises
   `TypeError: unhashable type` for `{"type": []}` / `{"type": {}}`.
   Repro output: `A: RAISED TypeError cannot use 'list' as a dict key`.
   Fix: `if not isinstance(cmd_type, str):` route to the existing unknown-command error
   branch (use `f"Unknown command: {cmd_type!r}"`-safe formatting).

2. **BUG-B — commands.py: 4 handlers crash (→ connection death) on malformed numeric fields,
   inconsistent with `_cmd_get_adjacent_task` which guards its `int()` with try/except.**
   - `_cmd_delete_task`: `int(task_id)` — `{"taskId": "abc"}` → ValueError
   - `_cmd_set_favorite`: `int(task_id)` — same
   - `_cmd_resume_session`: `int(raw_task_id)` — same
   - `_cmd_get_frequent_tasks`: `int(cmd.get("limit", 50))` — same
   All confirmed raising ValueError. Fix: guarded int parse (small shared helper in
   commands.py, e.g. `_parse_int(value) -> int | None`), ignore/None on garbage like
   `_cmd_get_adjacent_task` does (for limit fall back to 50).

3. **BUG-C — commands.py `_cmd_run` concurrent-start race (NOT yet repro'd, by code reading).**
   Guard is `if tab.task_thread is not None and tab.task_thread.is_alive(): return`,
   but `tab.task_thread = thread` is set under `_state_lock` while `thread.start()`
   happens AFTER releasing the lock and after `printer.broadcast({"type":"clear",...})`
   (network I/O — wide window). A created-but-not-started thread has `is_alive() == False`,
   so a second concurrent `_cmd_run` for the same tab passes the guard, **clobbers
   `tab.stop_event` / `tab.user_answer_queue` / `tab.task_thread`** and both threads then
   start → two concurrent tasks on one tab/chat; first task becomes unstoppable
   (its stop_event/task_thread references were overwritten; `_run_task`'s finally then
   nulls the second task's state while it still runs).
   Fix: treat the tab as busy whenever `tab.task_thread is not None` (the `_run_task`
   outer finally always resets it to None under the lock, so non-None ⇔ task in flight
   or about to start). Verified no other production code assigns `task_thread` besides
   `_cmd_run` (commands.py:186) and the finally (task_runner.py:160).
   Test idea (deterministic, no mocks): duck-typed printer whose `broadcast` blocks on
   the first "clear" event until a second `_cmd_run` (from another thread) has executed;
   pre-fix both runs proceed (2 "clear" events / task_thread clobbered), post-fix the
   second submit is silently dropped (1 "clear"). Stub agent via the existing precedent
   `SorcarAgent.__mro__[1].run` replacement (see test_bughunt_server_runner.py `_BugHuntBase`).

4. **BUG-D — task_runner.py `_run_task_inner` (~line 240): malformed attachment kills the
   task thread with no user-visible error.** `base64.b64decode(att.get("data",""))` raises
   `binascii.Error` (and non-dict att → AttributeError) BEFORE the big try; `_run_task`
   has try/finally but **no except**, so the exception propagates to threading excepthook:
   user sees spinner stop (`status running:false` from the finally) with no result/error
   event, nothing persisted. Fix: wrap per-attachment decode in try/except, skip bad
   attachments (log) — or broadcast an error result; pick skip+log as root-cause-minimal.
   Test: call `server._run_task({...,"attachments":[{"data":"%%%not-b64%%%","mimeType":"x"}]})`
   synchronously with capture printer; pre-fix raises binascii.Error out of `_run_task`.

5. **BUG-E — server.py `_new_chat` / commands.py `_cmd_new_chat`: empty tabId creates a
   permanent phantom registry entry keyed `""` (+ a WorktreeSorcarAgent via `_get_tab`).**
   Confirmed: `{"type": "newChat"}` → registry gains key `''`. `_cmd_close_tab` guards
   empty tabId, so the phantom can NEVER be disposed. Inconsistent with `_stop_task` /
   `_replay_session`, which both no-op on empty tab_id. Fix: early-return in `_new_chat`
   when `not tab_id` (mirror `_replay_session`'s logger.debug pattern). Check whether
   `_cmd_select_model` has the same hole (it calls `_get_tab(cmd.get("tabId",""))` —
   confirm and fix consistently, but check existing tests that may call selectModel
   without tabId first: `grep -rn selectModel src/kiss/tests`).

6. **BUG-F — commands.py `_cmd_user_answer`: non-string `answer` (e.g. None) is put on the
   `queue.Queue[str]`** and returned verbatim by `_await_user_response` → agent's
   `ask_user_question` callback returns None where str promised. Confirmed: queue item
   `None`. Fix: coerce — `ans = cmd.get("answer"); ans = "" if ans is None else str(ans) if not isinstance(ans, str) else ans`.

### Plan for next session (chronological)
1. Re-read the exact regions before editing (Read-before-Edit rule): server.py
   `_handle_command` + `_new_chat`; commands.py `_cmd_delete_task`/`_cmd_set_favorite`/
   `_cmd_resume_session`/`_cmd_get_frequent_tasks`/`_cmd_run`/`_cmd_user_answer`/
   `_cmd_select_model`; task_runner.py attachment-decode block in `_run_task_inner`.
2. Write failing-first tests (one file per bug, style of test_bughunt_server_runner.py:
   `VSCodeServer()` + capture-printer, `_RunningAgentState.running_agent_states.clear()`
   in tearDown; root tests/conftest.py already isolates KISS_HOME):
   - test_bughunt3_dispatch_malformed.py  (BUG-A + BUG-B: assert `_handle_command` does
     not raise and, for BUG-A, emits an "error" event; for BUG-B handlers assert no raise)
   - test_bughunt3_run_start_race.py      (BUG-C, blocking duck-typed printer)
   - test_bughunt3_bad_attachment.py      (BUG-D)
   - test_bughunt3_newchat_phantom_tab.py (BUG-E)
   - test_bughunt3_useranswer_nonstring.py(BUG-F)
   Run each, confirm FAIL pre-fix.
3. Apply fixes listed above; confirm tests pass.
4. Run impacted existing tests: `uv run pytest src/kiss/tests/agents/vscode -v` (count
   tests first; if >100 split across cores-2 via run_parallel) plus
   `src/kiss/tests/agents/sorcar` if persistence-adjacent (not expected).
5. `uv run check --full`; fix all errors.
6. Update this PROGRESS.md with results; clean tmp/ (tmp/repro1.py already created —
   DELETE it); finish with per-bug report (file:line, root cause, fix, test name).

### Notes
- NO source edits made yet; only tmp/repro1.py scratch (delete it).
- Conventions: tests live in src/kiss/tests/agents/vscode/; existing precedent allows
  replacing `SorcarAgent.__mro__[1].run` with a stub fn and assigning
  `server.printer.broadcast = capture` (duck-typed), per `_BugHuntBase`.
- Previous rounds' bugs (do not re-report): see list earlier in git history of this file
  (16 bugs, tests named test_bughunt_* / test_bughunt*2*).

---

# Bug-hunt iteration 3, group G (vscode helpers + frontend protocol consistency)

Scope: autocomplete.py, helpers.py, json_printer.py, vscode_config.py,
media/*.js, src/*.ts vs Python backend protocol. New tests: Python
src/kiss/tests/agents/vscode/test_bughunt3_<short>.py, JS
src/kiss/agents/vscode/test/bughunt3_<short>.test.js (jsdom pattern of
bughunt2_status_timer.test.js: load chat.html + panelCopy.js + main.js, stub
acquireVsCodeApi, dispatch MessageEvent; run `node <file>`).

## Session 1 (investigation, no code changes yet)

Fully read: autocomplete.py, helpers.py, json_printer.py, vscode_config.py,
core/printer.py helpers (parse_result_yaml guarantees 'summary' key →
`_broadcast_result`'s `parsed["summary"]` is safe). Diffed backend-emitted
event types (grep '"type": "..."' in vscode/sorcar *.py) against main.js
`case` list (lines 1674-3354), types.ts union, SorcarSidebarView.ts handlers.
Handled elsewhere (NOT bugs): activeTasksResponse/auth_* (web/extension layer),
autocommit_progress + worktree_created/progress (SorcarSidebarView.ts:375-425),
closeTab (SorcarSidebarView.ts:1150).

### CONFIRMED BUG G1 — helpers.clip_autocomplete_suggestion echo-strips
already-suffix suggestions. Only 2 call sites, both pass pure suffixes
(autocomplete.py:180 `fast`; cli_repl.py:240 via `_active_file_suffix`), no
LLM in pipeline anymore. `if s.lower().startswith(query.lower()): s =
s[len(query):]` double-strips suffixes that begin with the query: active file
"x = quxqux_token", query "qux" → suffix "qux_token" → clipped "_token" →
accepting ghost yields "qux_token" instead of "quxqux_token". Same for
history tasks with doubled words ("do it " + "do it do it again").
Fix plan: remove echo-strip + fix docstring; update stale unit test
src/kiss/tests/agents/sorcar/test_remaining_branches.py:184
(test_clip_autocomplete_suggestion_echo_prefix: ("hello","hello world") must
now expect "hello world").
Tests: test_bughunt3_autocomplete_echo_strip.py —
(a) UDS e2e modeled on test_per_window_autocomplete.py (RemoteAccessServer,
temp uds_path, _generate_self_signed_cert, redirect th._DB_PATH/_db_conn/
_KISS_DIR; send {"type":"complete","query":"qux","activeFile":...,
"activeFileContent":"x = quxqux_token\n"}; expect ghost "qux_token");
(b) CliCompleter._predictive_matches with real temp active_file → expect
["quxqux_token"].

### CONFIRMED BUG G2 — backend `warning` events silently dropped by frontend.
worktree_sorcar_agent.py:531-539 `_flush_warnings` broadcasts
{"type":"warning","message":...} (stash-pop failure + merge-conflict
warnings) but NO handler exists in main.js / demo.js / types.ts /
SorcarSidebarView.ts. main.js has error→addError(ev.text) ~2596,
notice→addNotice(ev.text) ~2600; renderers main.js:3529 ('ev tr err',
'<strong>Error:</strong> '+esc) / 3537 ('ev tr note'); CSS main.css:623
(.tr.err), 626 (.tr.note). Warning uses field `message` (not `text`).
Fix plan: main.js `case 'warning':` (tabId-gated like notice) →
addWarning(ev.message || ev.text); addWarning renderer 'ev tr warn'
'<strong>Warning:</strong> '; CSS .tr.warn amber mirror of .tr.err;
types.ts union `| {type: 'warning'; message: string; tabId?: string}`.
demo.js unaffected (warning not in _DISPLAY_EVENT_TYPES → never persisted).
Test: bughunt3_warning_event.test.js (assert .ev element with warning text
appears; also assert foreign-tabId warning does NOT render).

### TODO next
- Verify SorcarSidebarView forwards all backend events to webview (grep
  postMessage) — needed for G2 to reach main.js in the VS Code path.
- Write failing tests → verify fail → fix → verify pass.
- Still uninspected: json_printer malformed/streaming/unicode rendering paths
  in main.js (~1674-2050, esc()/XSS), vscode_config round-trips,
  rank_file_suggestions edge cases.
- Then: `npx tsc -p .`, node JS tests (all bughunt* + existing 6), impacted
  Python tests, `uv run check --full`.
