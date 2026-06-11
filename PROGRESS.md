# Task: Repeatedly bug-hunt src/kiss/agents/vscode/ + src/kiss/agents/sorcar/ until no new bugs

Find/reproduce inconsistencies and obvious bugs via integration tests, then fix them.
Repeat until an iteration finds zero bugs.

## History

- Rounds 1-2 (commits up to 13f9516f): 16 bugs found+fixed; tests `test_bughunt_*`,
  `test_bughunt2_*`, JS `bughunt2_*.test.js`. See git history of PROGRESS.md.
- Iteration 3 (7 parallel groups, commits ad7cd058..19c75837): 26 NEW bugs found+fixed,
  each with a failing-first integration test (`test_bughunt3_*`, `bughunt3_*.test.js`):
  - Group A (persistence/running_agent_state): 2 bugs — subagent SQL LIKE filter false
    positives vs `_is_subagent_row` (fixed with JSON1 predicate); timestamp-tie wrong-row
    resolution in `_most_recent_task_id`/`_load_task_chat_id` (added `id` tiebreaks).
  - Group B (git_worktree/worktree_sorcar_agent): 4 bugs — merge() skipped
    `_finalize_worktree` for vanished wt dirs (leaked branches); unguarded
    copy_dirty_state PermissionError killed task + leaked half worktree; `remove()`
    failed on corrupt/locked worktrees (escalate --force --force → rmtree+prune);
    cleanup_orphans ran without repo_lock (race vs task start).
  - Group C (CLI): 4 bugs — UTF-8 chars split across reads destroyed; CSI/SS3 escapes
    split across reads typed garbage (`_pending_esc` buffer); cli_panel width math used
    len() not display columns (CJK/emoji broke box); terminal resize never re-emitted
    DECSTBM scroll region.
  - Group D (agents/tools): 4 bugs — Edit with empty old_string corrupted files;
    `_coerce_tasks` treated JSON `[]`/non-str lists as one literal task; profile-lock
    pid 0 ⇒ os.kill(0,0) always "in use"; WebUseTool atexit registration leak.
  - Group E (vscode server/commands/task_runner): 6 bugs — unhashable `type` killed
    connection; unguarded int() in 4 handlers; `_cmd_run` busy-guard race vs
    thread.start(); malformed attachment killed task thread silently; empty tabId
    phantom registry entries; non-string user answers on Queue[str].
  - Group F (web_server/diff_merge/merge_flow): 4 bug classes — git C-quoted paths
    misattributed hunks/invisible files/missed conflicts; deleted binary file invisible
    in merge view; rejecting binary merge entry crashed (UnicodeDecodeError) /
    truncated; rejecting one hunk rewrote CRLF file with LF.
  - Group G (helpers/frontend): 2 bugs — autocomplete echo-strip corrupted suffixes
    starting with the query; backend `warning` events silently dropped by main.js /
    types.ts (now rendered, HTML-escaped).
  - Verification: all 68 Python bughunt3 tests + JS test pass; full vscode+sorcar
    suites green in shards (only documented pre-existing forkpty/load flakes);
    `uv run check --full` passes.

## Iteration 4 (current)

- Since iteration 3 still found bugs, launching iteration 4: same 7 functional groups,
  parallel sub-agents, tests named `test_bughunt4_*` / `bughunt4_*.test.js`.
  Stop condition: an iteration that finds zero new bugs.

### Iteration 4, group B (git_worktree / worktree_sorcar_agent): 2 NEW bugs found+fixed

- BUG-4B-1 (worktree_sorcar_agent.py `_do_merge`, was ~line 247): a FAILED
  `git stash push` was indistinguishable from "tree clean" — `stash_if_dirty`
  returns False for both. `_do_merge` then ran `git merge --squash` on a still
  dirty main tree, (a) silently committing the USER's staged changes into the
  agent's squash-merge commit, and (b) on merge failure running
  `git reset --hard HEAD`, permanently DESTROYING the user's staged+unstaged
  edits. Reproduced with a mode-000 untracked file (makes stash push fail:
  "Cannot save the untracked files") plus a staged user edit. Fix: new
  `MergeResult.STASH_FAILED`; `_do_merge` aborts before any mutation when
  `not did_stash and has_uncommitted_changes(repo)`; `merge()` returns a
  "Cannot merge: ... could not be stashed" message (keeps `_wt` for retry;
  merge_flow.py's `"Successfully merged" in msg` check yields success=False);
  `_release_worktree` sets a stash-failure `_merge_conflict_warning` and keeps
  the branch. Tests: test_bughunt4_stash_fail_merge.py (3 tests, all failed
  pre-fix).
- BUG-4B-2 (worktree_sorcar_agent.py `_try_setup_worktree` baseline commit):
  `commit_staged(..., no_verify=True)` can still fail — `--no-verify` skips
  only pre-commit/commit-msg, NOT prepare-commit-msg (also: stale index.lock,
  missing identity). The old code silently continued with
  `baseline_commit=None` while the user's dirty files sat UNCOMMITTED in the
  worktree → later auto-committed as (and attributed to) agent work and
  squash-merged back, duplicating the user's edits into the original branch.
  Fix: when the baseline commit fails AND the worktree still has uncommitted
  changes, `cleanup_partial` + return None (run()'s documented direct-execution
  fallback). The `elif has_uncommitted_changes` guard keeps the phantom-dirty
  case (e.g. CRLF-smudge: stage_all stages nothing) on the worktree path.
  Tests: test_bughunt4_baseline_commit_fail.py (failing test failed pre-fix +
  clean-tree control test).
- Investigated and verified NOT bugs (with throwaway-repo experiments):
  rename old-path recreated as untracked (`R a -> b` precedes `?? a` in
  porcelain → removal happens before re-copy, final state correct); untracked
  nested git repo shown as `?? nested/` is skipped by copy_dirty_state (kin to
  the iter-3 "dirty submodules" not-bug; git cannot merge it back anyway);
  sparse-checkout main repo (new worktree inherits sparseness — mirrors the
  user's view; merges handled by git); merge while main branch moved ahead
  (cherry-pick baseline..branch / squash merge both do proper 3-way merges,
  user dirty state not duplicated); post-checkout hook failure during checkout
  (self-heals on merge retry); index.lock contention (all git calls fail
  gracefully, no partial mutation).
- Verification: 5/5 bughunt4 group-B tests pass; all 475 worktree/bughunt/
  autocommit/workflow/baseline sorcar tests pass in 8 parallel shards (only
  the documented pre-existing forkpty pty flake
  test_bughunt_cli.py::test_ctrl_c_abort_actually_stops_the_running_agent
  failed under load, passes in isolation); `uv run check --full` passes.
- NOTE: the working tree also contains UNRELATED in-progress group-C (CLI)
  changes from a previously crashed session (cli_steering.py paste/SIGCONT
  work + test_bughunt4\_{paste,sigcont,prompt_markers,interrupt_lock}.py,
  test_bughunt4_parallel_stop_event.py); left untouched and uncommitted.

### Iteration 4 — Group D (sorcar agents/tools) — COMPLETE: 3 NEW bugs found+fixed

- BUG-4D-1 (sorcar_agent.py `run_tasks_parallel`, ~line 916): module-level
  parallel executor never copied the parent thread's
  `printer._thread_local.stop_event` into worker thread-locals (the
  `ChatSorcarAgent._run_tasks_parallel` override did) — sub-agents spawned via
  plain `SorcarAgent` resolved `self._stop_event = None`, so Stop could not
  kill their Bash process groups. Fix: capture `parent_stop_event` next to
  `parent_key` and set `tl.stop_event = parent_stop_event` at the top of
  `_run_single`. Test: test_bughunt4_parallel_stop_event.py (failed pre-fix).
- BUG-4D-2 (chat_sorcar_agent.py `run()`, ~line 427): result strings that
  parse as valid NON-dict YAML (plain string / list / number) persisted
  `result_summary = ""` to task history, while unparseable results fell back
  to `result[:500]`. Fix: added `else: result_summary = result[:500]` branch.
  Test: test_bughunt4_summary_nondict_yaml.py (2 of 3 tests failed pre-fix;
  dict-result regression guard passed).
- BUG-4D-3 (useful_tools.py `_expand_pwd_prefix`, ~line 40): `"PWD//etc/x"`
  computed `os.path.join(base, "/etc/x")` → `os.path.join` discards *base*
  when the 2nd component is absolute, silently ESCAPING the work_dir. Fix:
  `lstrip("/")` on the suffix (`PWD//` alone still → work_dir). Test:
  test_bughunt4_pwd_double_slash.py (4 of 5 tests failed pre-fix).
- Investigated, NOT bugs (do not re-report): `_truncate_output` marker math
  (dropped-count exact; `max_chars < marker` head-slice fallback is the
  documented degenerate case); Read `\r`/splitlines counting + truncation
  marker; Write parent-dir creation; Edit guards (iter-3); `_coerce_tasks`
  JSON forms (iter-3); profile-lock pid\<=0 + WebUseTool atexit re-arm
  (iter-3); go_to_url tab:list / tab:N int() errors caught per-call;
  `_resolve_locator` re-snapshot path; `run_parallel` tool `int(max_workers)`
  on junk/`"0"` raises but kiss_agent.py:412 wraps every tool call in
  try/except → clean error string to the model; update_settings round-trips
  (each key persists via `_apply_setting`; auto_commit one-shot semantics
  correct); `number_of_cores` (`os.process_cpu_count() or 1`).
- Verification: 9/9 bughunt4 group-D tests pass; all 758 impacted sorcar
  tests (88 files importing useful_tools/sorcar_agent/chat_sorcar_agent) pass
  in 8 parallel shards (90+64+114+70+125+71+104+120, zero failures);
  `uv run check --full` passes (also auto-fixed an unrelated pre-existing
  ruff UP041 in test_bughunt4_merge_replay_on_reconnect.py).

Original session-1 findings (now all fixed above, kept for context):

1. **CONFIRMED (known a)** `sorcar_agent.py` module-level `run_tasks_parallel()`:
   `_run_single` does NOT set `tl.stop_event = parent_stop_event` on the worker
   thread-local, while `ChatSorcarAgent._run_tasks_parallel._run_single` DOES
   (chat_sorcar_agent.py ~line 195: `if tl is not None: tl.stop_event = parent_stop_event`).
   Module-level only captures `parent_key = getattr(parent_tl, "task_id", "")`.
   Effect: sub-agents spawned via plain `SorcarAgent` never see the parent stop event
   (`SorcarAgent.run` reads `self._stop_event = getattr(tl, "stop_event", None)` in the
   worker thread) so Stop doesn't kill sub-agent Bash process groups.
   FIX: in `run_tasks_parallel`, capture `parent_stop_event = getattr(parent_tl, "stop_event", None)`
   next to `parent_key`, and at top of `_run_single` do
   `tl = getattr(printer, "_thread_local", None) if printer else None; if tl is not None: tl.stop_event = parent_stop_event`.
   Test: `test_bughunt4_parallel_stop_event.py` — build a real Printer-like object with
   `_thread_local = threading.local()`; set `stop_event` in calling thread; instead of a
   real agent run, the test must exercise the plumbing: pattern — see how existing
   tests test this for the chat override (grep `stop_event` in
   test_race_conditions.py / test_run_parallel_integration.py / test_vscode_stop.py and
   copy the no-LLM pattern, e.g. monkeypatching is NOT allowed (no mocks); existing
   tests likely subclass ChatSorcarAgent overriding run() — subclassing in the test to
   record `getattr(tl, "stop_event", None)` inside worker run() is acceptable since
   module-level run_tasks_parallel hardcodes ChatSorcarAgent — alternative: assert via
   the worker thread-local directly by passing a printer and a task list and overriding
   ChatSorcarAgent? Module-level imports ChatSorcarAgent inside the function, so test
   can't substitute class without patching. Simplest no-mock test: call
   run_tasks_parallel with tasks=[] is useless; instead test via SorcarAgent subclass?
   PRAGMATIC: existing bughunt tests in repo DO use light monkeypatching of agent.run;
   check test_run_parallel_integration.py first and mirror its approach.)

1. **CONFIRMED (known b)** `chat_sorcar_agent.py` `run()` (~line 405-412):

   ```python
   result_yaml = yaml.safe_load(result)
   if isinstance(result_yaml, dict):
       result_summary = result_yaml.get("summary", "")
   except Exception: result_summary = result[:500] if result else ""
   ```

   When `result` parses as non-dict YAML (e.g. plain string "all done"), summary stays ""
   and `_save_task_result` persists '' — inconsistent with parse-failure fallback
   `result[:500]`. FIX: add `else: result_summary = result[:500] if result else ""`.
   Also consider: dict but summary is None → persists None; coerce to "" or str.
   Test: `test_bughunt4_summary_nondict_yaml.py` — subclass ChatSorcarAgent whose
   SorcarAgent.run returns a plain-string YAML (override `SorcarAgent.run` via subclass
   method calling object-level shim — pattern: define subclass with
   `def run(self, ...)`? No — ChatSorcarAgent.run calls super().run; so subclass
   overriding nothing won't help. Existing tests (e.g. test_stateful_sorcar_agent.py,
   test_history_continuation_context.py) already run ChatSorcarAgent without LLM —
   COPY THEIR PATTERN. Then assert sqlite task_history row's result == "all done"
   (use temp KISS db via env var / persistence module's db path fixture used by
   test_persistence.py).

1. **NEW candidate** `useful_tools.py::_expand_pwd_prefix`: `"PWD//foo"` →
   `suffix = "/foo"` → `os.path.join(base, "/foo")` returns `"/foo"` (absolute!),
   escaping work_dir entirely. FIX: `suffix.lstrip("/")` before join (keep `PWD` and
   `PWD/` behavior). Test: `test_bughunt4_pwd_double_slash.py` asserts
   `_expand_pwd_prefix("PWD//sub/f.txt", "/base") == "/base/sub/f.txt"` and Read/Write
   round-trip through UsefulTools(work_dir=tmp) with "PWD//x.txt" lands inside tmp.

1. Checked and found OK (do NOT re-report): `_truncate_output` marker math (dropped
   count correct; max_chars\<marker fallback returns plain head slice — documented-ish);
   Read \\r splitlines counts fine; Write parent-dir creation present; Edit guards
   (empty old_string, equal strings, count) all present from iter-3; `_coerce_tasks`
   JSON handling fixed in iter-3; profile pid\<=0 fixed; WebUseTool atexit re-arm +
   unregister present; go_to_url tab:list/tab:N int errors caught; \_resolve_locator
   re-snapshots; scroll invalid direction defaults down (minor, not fixing).
   Possible minor: `run_parallel` tool `int(max_workers)` raises on "0"/"-1"/junk →
   propagates to framework tool-error handling (decide in next session whether to
   guard with friendly error — leaning yes, cheap: validate and return error string).
   Possible minor: ChatSorcarAgent.run finally sets `tl.task_id = ""` instead of
   restoring previous — investigate only if time permits (vscode runner may re-set).

NEXT STEPS (exact):

- Read src/kiss/tests/agents/sorcar/test_run_parallel_integration.py and
  test_stateful_sorcar_agent.py (or test_history_continuation_context.py) to copy the
  no-LLM ChatSorcarAgent test pattern + temp-db fixture.
- Write failing tests: test_bughunt4_parallel_stop_event.py,
  test_bughunt4_summary_nondict_yaml.py, test_bughunt4_pwd_double_slash.py
  (in src/kiss/tests/agents/sorcar/). Verify each fails.
- Apply the 3 fixes above. Verify tests pass.
- Run impacted tests: `uv run pytest src/kiss/tests/agents/sorcar -k "bughunt4 or parallel or stateful or useful_tools or tools" -x -q` (count first; shard if >100).
- `uv run check --full` must pass. Report with file:line.

### Iteration 4 — Group F (web_server/diff_merge/merge_flow) — session 2 (CONTINUATION) — 2 NEW BUGS FOUND+FIXED

Read in full: diff_merge.py, merge_flow.py, web_server.py merge regions (\_WebMergeState
~396-520, \_restore_base_bytes/\_reject_hunk_in_file/\_reject_all_hunks_in_file ~530-670,
\_apply_web_merge_action ~3625-3725, \_augment_merge_data ~2401, dispatch ~3144).

**BUG F4-1 (fixed)**: `web_server._augment_merge_data` read `base_text`/`current_text`
with `Path.read_text()` → universal-newline translation (CRLF→LF, lone CR→LF).
Inconsistent with hunk math which splits preserved bytes on "\\n" only
(`_read_lines_preserved`): CRLF files' displayed text differed from what reject writes
back; lone-"\\r" files got MORE lines in the browser than hunk cs/cc coordinates →
misaligned hunk highlighting. FIX: read with `open(path, newline="")`.
Test: test_bughunt4_augment_newlines.py (2 tests, failed before fix, pass after).

**BUG F4-2 (fixed, data loss)**: rejecting a merge entry whose workspace path is a
SYMLINK wrote THROUGH the link (`open(write_to, "w")` / `Path.write_bytes`), truncating
or overwriting the pointed-to file (possibly outside the repo) while the rejected link
survived. Repro: agent creates untracked symlink link.txt→data.txt; \_prepare_merge_view
lists it as a new file (is_file()/reads follow links); reject-all wrote "" through the
link → data.txt emptied. Same for binary-flagged path via `_restore_base_bytes`.
FIX: in `_reject_hunk_in_file` and `_restore_base_bytes`, unlink `write_to` first when
`Path(write_to).is_symlink()` (git tracks the link, never write through it).
Test: test_bughunt4_symlink_reject.py (3 tests: reject-all text, per-hunk, binary;
all failed before fix, pass after).

**BUG F4-3 (fixed)**: merge review lost forever on browser reload. `merge_data` events
are tab-stamped → `WebPrinter.broadcast` forwards to connected clients only, never
persisted/replayed; `_handle_ready` re-claimed reloaded tabs (cancel deferred close +
resumeSession) but never re-emitted the in-flight review. Result: after a mid-review
page reload the merge UI is gone, the unresolved server-side `_WebMergeState` and the
backend tab's `is_merging` stay stuck, all-done/_finish_merge/autocommit never fire.
(VS Code extension unaffected: its TS MergeManager survives webview reloads.)
FIX: `_WebMergeState` now keeps the full `data` payload (`self.data`, hunks shared so
reject cs-offset mutations stay live); new `RemoteAccessServer._replay_merge_review`
re-sends augmented `merge_data` + `merge_started` + `merge_nav` (with resolutions and
current position) TARGETED at the reconnecting endpoint; called from `_handle_ready`
for the claimed tabId and every restoredTabs entry.
Test: test_bughunt4_merge_replay_on_reconnect.py (real wss:// reconnect; 2 tests:
replay with one hunk pre-accepted via real action handler asserts remaining=2 +
resolved list; clean reconnect asserts NO merge events; failed before fix, pass after).

**Verified NOT bugs (do NOT re-report)**:

- Interleaved/out-of-order per-hunk accept/reject offset bookkeeping in
  `_apply_web_merge_action` ("reject" branch adjusts later UNRESOLVED hunks' cs by
  bc-cc; `_reject_all_hunks_in_file` adjusts pending-only): verified correct end-to-end
  with a real repo + RemoteAccessServer driving next/prev/reject sequences out of order
  (4 hunks incl. insert/delete/append, delta != 0) — file restored byte-exact both
  orders (scratch-verified, both cases OK).
- `_reject_all_hunks_in_file` vs per-hunk reject delta consistency (pending-set vs
  is_resolved): consistent because reject-all marks resolved BEFORE calling and passes
  the previously-unresolved indices.
- `_WebMergeState.current()` can never return a resolved hunk (advance() after every
  resolution; remaining==0 → None); accept-file/reject-file/accept-all/reject-all
  resolution bookkeeping consistent.
- empty\<->nonempty transitions for pre-dirty untracked files (saved-base diff produces
  correct {bs,bc,cs,cc} both directions, reject restores byte-exact).
- cc==0 / bc==0 hunk start conventions (`_hunk_to_dict` 1-based→0-based with
  count-0 exception) agree between git -U0 parse, `_diff_files` (difflib), and the
  reject splice.

Still-open candidates for a future session (NOT verified, NOT fixed):

- merge_data replay to a reconnecting browser may carry stale hunks (cs mutated by
  rejects) without resolved markers (needs reading replay path).
- `_main_dirty_files` strips unquoted trailing-space filenames (git doesn't quote
  trailing spaces) — very marginal.
- new empty file / empty-file deletion invisible in merge view (documented behavior of
  `_file_as_new_hunks`; deleting empty tracked file produces no hunks → not in
  post_hunks) — by-design-ish.
- rejecting an agent-created NEW file leaves an empty file rather than deleting it
  (consistent VS Code-side? would need manifest flag to distinguish new-file vs
  emptied-file) — deliberate-looking, did not change.

### Iteration 4 — Group G (vscode helpers + frontend consistency) — session 1 notes

Files read in full: autocomplete.py, helpers.py, json_printer.py, vscode_config.py.
Targeted greps done. Candidate bugs to verify next session (NOT yet confirmed/tested):

1. **vscode_config cross-process lost update** (explicitly hinted in task): `_config_lock`
   in vscode_config.py is a `threading.Lock` — serialises save_config only within ONE
   process. Two daemons (e.g. kiss-web daemon + a VS Code window daemon) doing concurrent
   read-merge-replace of `~/.kiss/config.json` can drop each other's keys. Likely fix:
   `fcntl.flock` on a sidecar lock file (CONFIG_DIR/".config.lock") held across the whole
   load-merge-store in save_config. Test: spawn 2 subprocesses each saving a different
   key N times; assert both keys survive.
1. **demo.js does NOT handle 'warning' events**: grep shows `warning` appears in
   media/main.js (case 'warning' at :2607, renderer :3552) and src/types.ts:208, but
   NOWHERE in media/demo.js (389 lines). Need to read demo.js to decide contract — if
   demo.js groups/replays recorded display events by type, iteration-3's new warning
   events may break/get dropped during demo replay. Also check: `warning` is NOT in
   json_printer.py `_DISPLAY_EVENT_TYPES` → warnings are never recorded/persisted, so
   chat history replay (viewer reopen) silently loses warnings even though live view
   shows them. Decide contract: probably add "warning" to \_DISPLAY_EVENT_TYPES + demo
   replay rendering.
1. Autocomplete `_complete` in autocomplete.py: when `_prefix_match_task(query)` matches,
   `fast = match[len(query):]` — verify \_prefix_match_task case behaviour (if match is
   case-insensitive, slicing by len(query) is fine but suggestion may duplicate case
   issues); also no dedup/ranking across history-vs-file sources (history always wins).
   Check persistence.\_prefix_match_task.
1. commands.py `_cmd_complete` (~:510-533) sets `self._complete_seq_latest[conn_id] = seq`
   — seq comes from frontend? Verify monotonicity: if frontend restarts (webview reload)
   seq resets to 0 < stale latest → all new requests dropped as stale? server.py:290 pops
   on disconnect — check webview reload uses same conn or new conn.
1. extension.ts vs package.json: registers `kissSorcar.${cmd}` for Object.values(MERGE_ACTIONS)
   — verify MERGE_ACTIONS values exactly match package.json contributes (acceptChange,
   rejectChange, prevChange, nextChange, acceptAll, rejectAll, acceptFile, rejectFile).
   Also `kissSorcar.focusEditor`/`runSelection`/`insertSelectionToChat`/`toggleFocus`
   present both sides per greps — looks consistent so far.
1. Still to inspect: panelCopy.js copy formatting, status-bar/timer edges in main.js,
   settings fields in package.json `contributes.configuration` + webview settings UI vs
   vscode_config DEFAULTS, json_printer parallel sub-agent interleaved events.

Test naming: Python src/kiss/tests/agents/vscode/test_bughunt4\_<short>.py; JS
src/kiss/agents/vscode/test/bughunt4\_<short>.test.js (run `node <file>`, use
\_vscode-stub.js patterns). Failing test FIRST, then fix. Finish with JS tests +
`npx tsc -p .` (in src/kiss/agents/vscode) + impacted Python tests + `uv run check --full`.
No code modified yet in this session.

### Session-1 verification results (step 12)

- CONFIRMED BUG A: `warning` missing from `_DISPLAY_EVENT_TYPES` in json_printer.py
  (~line 38). Live view renders warnings (main.js:2607) but `_persist_event`,
  `stop_recording`, `peek_recording` all filter to `_DISPLAY_EVENT_TYPES` → warnings
  vanish on viewer reopen / chat-history replay / demo replay. Fix: add "warning" to
  the frozenset. demo.js + main.js need NO change (warning falls into current group in
  groupEventsIntoPanels; processEvent/handleOutputEvent has case 'warning'). Test:
  src/kiss/tests/agents/vscode/test_bughunt4_warning_persist.py — JsonPrinter:
  set `printer._thread_local.task_id = "t1"`, start_recording(), broadcast
  {"type":"warning","message":"x"} + a text_delta, stop_recording() must include the
  warning event (fails pre-fix). Also peek_recording variant.
- CONFIRMED BUG B: vscode_config.save_config `_config_lock` is threading-only; two
  daemon PROCESSES doing read-merge-replace on config.json lose updates. Fix: fcntl.flock
  on CONFIG_DIR/".config.lock" held across load-merge-store inside save_config (keep
  threading lock too). Test: test_bughunt4_config_cross_process.py — KISS_HOME=tmp,
  spawn 2 subprocesses; A repeatedly save_config({"work_dir": str(i)}) then load_config
  and assert its own key == just-written; B same with "last_model"; any mismatch printed
  → parent asserts no mismatch (fails pre-fix with high probability, ~300 iters each).
- NOT bugs (verified): autocomplete seq is server-generated monotonic global counter
  (commands.py \_cmd_complete, server.py 240-300) — webview reload safe; per-conn state
  dropped via drop_connection_state. \_prefix_match_task uses case-sensitive GLOB —
  `match[len(query):]` slicing correct. MERGE_ACTIONS values (SorcarSidebarView.ts:54)
  exactly match package.json contributes (acceptChange/rejectChange/prevChange/
  nextChange/acceptAll/rejectAll/acceptFile/rejectFile); other commands
  (openPanel/newConversation/stopTask/toggleFocus/focusEditor/runSelection/
  insertSelectionToChat/generateCommitMessage) all registered both sides.
- Still optionally check (lower priority): panelCopy.js formatting, package.json
  contributes.configuration (only kissSorcar.defaultModel + kissSorcar.kissProjectPath)
  vs webview settings, json_printer interleaved sub-agent events.

### Session-1 DONE (steps 16-27)

- Wrote failing tests FIRST, verified failures:
  - `src/kiss/tests/agents/vscode/test_bughunt4_warning_persist.py` (3 tests) —
    failed: warning dropped by stop_recording/peek_recording.
  - `src/kiss/tests/agents/vscode/test_bughunt4_config_cross_process.py` — failed
    with genuine "LOST UPDATE: wrote 'last_model-…' read back …" from real
    2-subprocess hammer.
- FIX A: json_printer.py — added "warning" (with comment) to `_DISPLAY_EVENT_TYPES`.
- FIX B: vscode_config.py — `import fcntl`; save_config now holds
  `fcntl.flock(CONFIG_DIR/".config.lock", LOCK_EX)` (inside `_config_lock`, released
  in finally) across the whole read-merge-replace.
- Post-fix: all 4 new tests pass. Impacted tests pass: test_replay_event_coalescing,
  test_printer_equivalence, test_config_race, test_config_save_on_close,
  test_bughunt3_autocomplete_echo_strip (28 passed); JS: bughunt3_warning_event.test.js
  PASS, bughunt2_demo_continue.test.js PASS. demo.js/main.js need no change (warning
  falls into current panel group; main.js case 'warning' renders it on replay).
- Verified NOT bugs additionally: panelCopy.js (getRawText/normalise/clipboard fallback
  fine); settings DEFAULTS keys all referenced in main.js settings UI (last_model is
  backend-only); MERGE_ACTIONS/commands package.json parity OK.

REMAINING: grep tests asserting on \_DISPLAY_EVENT_TYPES contents (ensure no structural
test broke), run `uv run check --full`, git add+commit the two fixes + two tests +
PROGRESS.md, then finish with bug report (2 NEW bugs).

### Iteration 4 — Group A (sorcar persistence.py / running_agent_state.py) — COMPLETE: 2 NEW bugs fixed

Read persistence.py (2052 lines) + running_agent_state.py in full; reviewed all prior
persistence bughunt tests to avoid duplicates.

1. **BUG (fixed)** `persistence.py:_delete_task` — cascade only deleted DIRECT
   sub-agent rows (`_subagent_child_ids(db, task_id)`, one level). A sub-agent is a
   full `ChatSorcarAgent` with the `run_parallel` tool, so nested fan-out creates
   grandchild rows whose `extra.subagent.parent_task_id` points at the CHILD id.
   Deleting the top-level parent leaked grandchildren + their events as permanently
   unreachable zombies, and `_chat_has_tasks` kept reporting the visually-empty chat
   as non-empty. Fix: breadth-first cascade over `_subagent_child_ids` with a `seen`
   set (also defends against corrupt self/cyclic parent references).
   Test: `src/kiss/tests/agents/sorcar/test_bughunt4_delete_nested_subagents.py` (3 tests).

1. **BUG (fixed)** `persistence.py:_list_recent_chats` — chat-selection query
   (`GROUP BY chat_id ... LIMIT ?`) selected the most recent *limit* chats BEFORE
   sub-agent-only chats were dropped in Python (`continue`), so an omitted chat still
   consumed a limit slot and a real older chat silently disappeared from the listing
   (CLI `--list-chats`). Also chat recency was anchored to MAX(timestamp) including
   sub-agent rows. Fix: added `AND {_HISTORY_NOT_SUBAGENT}` to the chat-selection
   query so only chats with ≥1 real task count against the limit and recency uses the
   latest REAL task.
   Test: `src/kiss/tests/agents/sorcar/test_bughunt4_recent_chats_limit.py` (2 tests).

Investigated and ruled NOT bugs (do not re-report):

- `_recover_orphaned_tasks` multi-process clobber of another live process's sentinel
  row: transient and self-healing — the live task's `_save_task_result` overwrites
  unconditionally by row id at completion; if the task IS later killed the message is
  accurate.
- `_get_db` stale `-wal`/`-shm` unlink when DB file missing: only reachable when the
  main DB file was externally deleted; the open-fd holder keeps its own inode; the
  cross-process creation window is not deterministically testable and SQLite recreates
  WAL on demand.
- `_record_frequent_task` eviction at cap, `_prefix_match_task` GLOB escaping,
  `_search_history` LIKE escaping, `_write_event_batch` dangling-event defense,
  `_load_chat_context_text` generation-guarded cache, `_delete_task` rowcount
  semantics (parent-row only — correct), `_shutdown_persist_in_flight_results`.
- Pre-existing (NOT caused by these fixes): ordering flake in
  `test_orphan_task_recovery.py::test_concurrent_boot_does_not_corrupt` when run after
  the full persistence suite — fails identically with the fixes stashed; passes alone.

Verification: 5 new tests fail pre-fix, pass post-fix; 120-test focused persistence
regression suite green (only the pre-existing flake above, reproduced on pristine
code); `uv run check --full` clean.
