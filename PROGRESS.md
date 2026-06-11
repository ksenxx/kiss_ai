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
- Group C (sorcar CLI: cli_repl/cli_steering/cli_helpers/cli_panel) DONE — 5 NEW bugs
  found + fixed, each with a failing-first integration test (commit 4f3390cd):
  1. Bracketed paste (`cli_steering._InputBox`): mode 2004 never enabled and
     `ESC[200~`/`ESC[201~` swallowed as generic CSI — a multi-line paste submitted
     every line as a separate queued instruction; pasted ANSI/control chars acted as
     keys. Fix: enable `?2004h`/`?2004l` in start/stop, buffer paste content (incl.
     newlines, CRLF normalised, ANSI stripped, DEL/C0 dropped) with split-across-reads
     handling (`_pasting`, `_partial_suffix_len`). Tests: test_bughunt4_paste.py (9).
  1. Tiny-terminal resize (`_draw_locked`/`stop`/`_park_cursor_locked`): rows \<=
     \_BOX_H produced invalid `ESC[1;-1r`/`ESC[0;1H` (zero/negative rows). Fix:
     `_box_top_row` clamp (>=1). Tests: test_bughunt4_tiny_resize.py (3).
  1. No SIGCONT handler: after Ctrl+Z + fg the raw mode/paste mode/scroll region/box
     were stale until the next keypress. Fix: `_on_sigcont` re-applies raw termios,
     re-enables paste mode, forces scroll-region re-anchor + redraw; handler installed
     in start (main thread only), restored in stop. Test: test_bughunt4_sigcont.py
     (pty.fork end-to-end).
  1. Worker leak on Ctrl+C outside select: `_loop` only caught KeyboardInterrupt
     around `select.select`; SIGINT while the main thread was blocked on the terminal
     RLock (feed→redraw vs a worker `_StdoutProxy` write blocked on a full pty)
     escaped without `_on_abort`, so `agent.run` kept executing in the background.
     Fix: `SteeringSession.run` catches KeyboardInterrupt from `_loop` → `_on_abort`.
     Test: test_bughunt4_interrupt_lock.py (deterministic pty flood + lock-park).
  1. readline prompt width (`cli_repl._read_line`): ANSI SGR codes in the prompt
     lacked `\x01`/`\x02` ignore markers, so GNU readline thought the 4-col prompt
     was ~26 cols and redrew/scrolled after the 2nd typed char on narrow terminals
     (measured empirically). Fix: `_readline_prompt` wraps SGR runs in markers when
     stdin+stdout are TTYs and readline is active. Test:
     test_bughunt4_prompt_markers.py (pty, 30 cols, clean 10-char echo).
  - Also: `cli_panel.clip_buf` renders tabs (now reachable via paste) as a space.
  - Regression sweep: all 1352 sorcar tests run in 8 parallel shards — only failure
    was test_print_to_browser lockdown missing the `warning` display type added by a
    parallel iter-4 group (expected set updated), plus a mypy error in group F's
    untracked test_bughunt4_replay_worktree_flag.py (cast added). `uv run check --full` passes.
  - Verified NOT bugs this round: `_prefix_match_task` GLOB escaping (\[,\*,? escaped —
    no wildcard prefix-violation), combining-char backspace (width-0 consistent),
    history read/write guards, `/model list` precedence, `\r\n` double-submit
    (empty second submit ignored / Queue.Full guarded).

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

### Iteration 5 — Group C (sorcar CLI) — session 1 (context exhausted, CONTINUE)

Read in full: cli_steering.py, cli_panel.py, cli_repl.py, cli_helpers.py. No test
written yet. Candidate bugs to REPRODUCE next session (write failing test first,
src/kiss/tests/agents/sorcar/test_bughunt5_<short>.py; check existing test_bughunt4_paste.py
style for harness — real _InputBox with a fake out stream / pyte / pty.fork):

1. C1 control chars pass `_append_paste` filter (cli_steering.py `_append_paste`:
   `ch >= " "` keeps U+0080–U+009F, e.g. U+009B = single-char CSI) → pasted C1
   lands in buf and is written RAW to the terminal in the body row (clip_buf only
   replaces \n and \t) → can corrupt the box/terminal. Same hole in `feed`'s typed
   path (`elif ch >= " ": buf += ch`). Test: paste b"\x1b[200~a\xc2\x9bXb\x1b[201~",
   assert buf has no U+009B / rendered body has no raw C1.
2. `clip_buf` "⏎" replacement char U+23CE: verify `char_width("⏎")` —
   unicodedata.east_asian_width(U+23CE) is "W"?? If char_width=2 but real terminals
   render 1 col, `body_cursor_col` parks the caret off-by-one per newline in buf
   (multi-line paste). MUST check actual EAW value in python first; compare with
   wcwidth if available. If EAW=N width 1, fine — drop.
3. `clip_buf` tail-clip can return a slice STARTING with a combining mark (backward
   loop keeps cw=0 combining char, breaks on its base char) → mark combines with the
   border space on screen. Check severity; width math itself is consistent.
4. `_pending_esc` 64-cap drop (feed end): pending dropped wholesale; if next chunk
   then begins with the CONTINUATION of that escape, its tail bytes are typed as
   literal buf text. Per code comment this is by-design for absurd sequences —
   probably NOT bug; only pursue if a real ≤64 sequence (paste marker straddling
   the cap with a long preceding CSI) can trigger it.
5. `SteeringSession.run`: if `box.start()` raises (termios.error race: tty closed
   after supports_steering()), the _StdoutProxy stays installed (sys.stdout never
   restored) and termios state half-applied — start() is called OUTSIDE the
   try/finally. Fix: move start() inside try or wrap. Verify reproducible: call
   run() with stdin not a real tty so tcgetattr raises.
6. `_on_submit` when task just finished (`_done` set between draw and Enter):
   message appended to state.pending_user_messages but never drained → silently
   lost (no warning). Compare with VS Code path behavior; maybe print "task
   already finished" or run queued as new task. Decide if bug.
7. `CliCompleter._model_matches`: `/model name extra` → query "name extra"; minor.
   `_build_matches` at-mention regex `@([^\s]*)$` only matches at EOL (cursor at
   end) — readline passes text up-to-cursor, so fine.
Ruled out by reading (do NOT re-report): paste-start/Shift+Enter/CSI split across
reads (pending_esc covers all; verified by trace); partial _PASTE_END suffix len
clamp; `keep` math; ESC-as-paste-content stripping consistent with unsplit; \x7f
ordering in elif chain; body_cursor_col ≤ cols-1 incl. wide chars; panel_top/bottom
ASCII-only inputs; Ctrl+C-during-question unblock (covered by _interrupt_worker);
resize-while-pasting; tab typed directly swallowed (by design).

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
backend tab's `is_merging` stay stuck, all-done/\_finish_merge/autocommit never fire.
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

## Iteration 4 — group E (vscode server.py / commands.py / task_runner.py)

Bugs found and fixed (failing test first, then fix, then verified pass):

1. `_cmd_run` phantom run on empty/missing `tabId` (commands.py ~175): registered a
   `_RunningAgentState` under key `""` and started a real task thread, but
   `_stop_task`, `_cmd_close_tab` and `_dispose_if_closed` all ignore empty ids, so
   the task was unstoppable/undisposable. Fix: early-return guard mirroring
   `_stop_task`. Tests: `test_bughunt4_run_empty_tabid.py` (committed in 7188834b).
1. `_replay_session` clobbered `use_worktree` mid-flight (server.py ~1035): resuming a
   different chat into a tab unconditionally overwrote `tab.use_worktree` with the
   loaded chat's `is_worktree`, even while a worktree task was running on the tab or
   while a finished worktree run awaited merge/discard (`agent._wt_pending`). The
   end-of-task cleanup (task_runner.py ~187) keeps the agent alive only when
   `tab.use_worktree and tab.agent._wt_pending`, so the flip disposed the agent
   holding the pending worktree (merge/discard then fails; worktree branch leaks);
   the merge-busy guard (`t.is_merging and t.use_worktree`) broke the same way.
   Fix: skip the `use_worktree` overwrite when `tab.is_task_active` or `_wt_pending`;
   idle tabs still adopt the resumed chat's flag.
   Tests: `test_bughunt4_replay_worktree_flag.py` (3 tests; 2 fail pre-fix).

Candidates investigated and closed as NOT bugs:

- Stop-during-startup window: a pre-start `_stop_task` sets the cooperative
  `stop_event`; `JsonPrinter._check_stop` raises `KeyboardInterrupt` at the agent's
  very first `print` — honored immediately.
- `selectModel` persistence vs `_new_chat` `_load_last_model()` re-read: consistent.
- Lock-order inversion state_lock↔printer.\_lock: only state→printer nesting exists.
- task_runner exception paths: failure broadcasts `result(success=False)` live and
  persists exactly ONE end event via `_append_chat_event(task_end_event)` — no
  duplicated persisted events; symmetric with the success flow.
- `_replay_session` dropping the owner tab's own-task subscription via
  `cleanup_tab`: intentional (documented) — the tab now renders a different chat.

Also: formatted PROGRESS.md (mdformat was failing `uv run check --full`).

## Iteration 4 — COMPLETE (verified by orchestrator)

19 NEW bugs found+fixed across 7 groups (A:2 nested-subagent delete cascade +
recent-chats limit slots; B:2 stash-failure merge data loss + baseline-commit-failure
fallback; C:5 bracketed paste, tiny-resize clamps, SIGCONT, Ctrl+C-under-lock leak,
readline prompt markers; D:3 parallel stop_event propagation, non-dict-YAML summary,
PWD// escape; E:2 empty-tabId run phantom, replay clobbering use_worktree; F:3
merge-view newline translation, symlink write-through on reject, merge replay on
reconnect; G:2 warning not in \_DISPLAY_EVENT_TYPES, cross-process config lost update).
All 50 Python bughunt4 tests pass; `uv run check --full` green; committed through
cedf3fcf + PROGRESS.md format commit.

## Iteration 5 (current)

- Iteration 4 still found bugs → launching iteration 5, same 7 groups, tests
  `test_bughunt5_*` / `bughunt5_*.test.js`. Stop when an iteration finds zero bugs.

### Iteration 5 — Group D (sorcar agents/tools) — session 1 (in progress)

Files fully read so far: useful_tools.py, sorcar_agent.py. 2 NEW bugs found+fixed
(failing tests written FIRST, 6 tests failed pre-fix, all 7 pass post-fix):

- BUG-5D-1 (useful_tools.py `_spawn`): Popen used `encoding="utf-8"` with STRICT
  errors. Any command emitting invalid UTF-8 (cat/grep on a binary, compiler
  latin-1 diagnostics) made non-streaming `communicate()` raise
  UnicodeDecodeError → process group killed, tool returned
  "Error: 'utf-8' codec can't decode..." losing ALL output even for exit-0
  commands; on the STREAMING path the exception escaped `_bash_streaming`
  entirely (no outer try/except) and leaked out of the tool. Fix: add
  `errors="replace"` to `_spawn`. Test:
  test_bughunt5_bash_binary_output.py (3 tests, all failed pre-fix).
- BUG-5D-2 (useful_tools.py `Read`/`Write`): Read on an existing NON-regular
  file (FIFO/char device/socket) called `Path.read_text()` — opening a FIFO
  with no writer blocks FOREVER (no timeout) hanging the whole agent;
  /dev/zero streams endlessly. Same hang for Write to a reader-less FIFO.
  Fix: guard `resolved.exists() and not resolved.is_file()` in Read (after
  the is_dir branch) and in Write (before mkdir/write_text), returning an
  immediate error string. Symlink-to-regular-file still works (regression
  test included). Test: test_bughunt5_read_nonregular_hang.py (3 bug tests
  failed pre-fix by 5s-hang detection + 1 symlink regression guard).

Examined in sorcar_agent.py, NOT bugs (do not re-report): update_settings
max_budget float() / framework arg coercion; set_model deferred path +
schema rebuild + conversation hand-off; run() finally cleanup
(web_use_tool.close, callback/pre_step_hook reset); _run_single sets
tl.stop_event each task (pool thread reuse safe); _coerce_tasks (iter-3/4);
sub_usage aggregation order (results from pool.map are input-ordered;
sub_usage indexed by idx — ordering correct).

STILL TO DO in this group-D session (next session if out of context):
read chat_sorcar_agent.py (471 lines: chat-context construction/token
budgeting vs persistence, _run_tasks_parallel override, model-switch
plumbing, tl.task_id restore in finally) and web_use_tool.py (624 lines:
screenshot/get_page_content error paths, profile lock, _resolve_locator,
tab switching); then run impacted tests + `uv run check --full`, commit.

### Iteration 5 — Group G (vscode helpers + frontend) — session 1: 1 NEW bug found+fixed

- **BUG 5G-1 (fixed)** media/main.js `handleOutputEvent` (switch ending ~line 2015)
  had NO `case 'warning'`. Iter-4 made `warning` persisted (`_DISPLAY_EVENT_TYPES`),
  but BOTH replay paths route through `handleOutputEvent`:
  `replayEventsInto` (task_events on chat reopen + background/sub-agent tab
  fragments) and `processOutputEvent` (demo.js `api.processEvent`). A persisted
  warning therefore silently vanished on chat reopen and demo replay, while the
  LIVE path rendered it (top-level switch case 'warning' :2607 → addWarning).
  FIX: added `case 'warning'` to handleOutputEvent rendering the identical
  `div.ev.tr.warn` + `<strong>Warning:</strong> ` + esc(ev.message||ev.text||'')
  banner into `target`. Live path can't double-render (top-level switch breaks
  before the display default route — regression-guarded in test).
  Test: src/kiss/agents/vscode/test/bughunt5_warning_replay.test.js (4 tests:
  replay-on-reopen, replay-identical-to-live + XSS escape, demo processEvent,
  live single-render). Failed pre-fix (first assert), all pass post-fix;
  bughunt3_warning_event.test.js still passes.
- Verified so far NOT bugs: demo.js grouping of warning/autocommit_done (fall
  into current panel group → processEvent renders them post-fix);
  package.json contributes.configuration only has kissSorcar.defaultModel +
  kissSorcar.kissProjectPath (no DEFAULTS overlap → no drift).
- REMAINING for session 2 (not yet checked): run eslint + `npx tsc -p .` + full
  JS test dir + `uv run check --full`; commit. Then continue surfaces:
  (a) autocomplete multi-line/unicode (`_complete` slicing vs `_prefix_match_task`
  on multi-line queries; `[A-Za-z_]`-anchored candidate regex vs unicode `\w`
  partials in `_complete_from_active_file` — a unicode-start partial like "héllo"
  can never match candidates anchored at `[A-Za-z_]` BUT partial regex `[\w][\w.]*`
  accepts it → always returns "" (probably fine), check instead a partial whose
  matched candidate slice misbehaves with astral chars in JS frontend accept path);
  (b) helpers.rank_file_suggestions Windows separators (marginal);
  (c) vscode_config save_config on read-only HOME (open of .config.lock raises
  OSError — does any caller crash a handler? grep save_config callers);
  (d) json_printer sub-agent panel interleaving with warning/persisted types
  (openSubagentTab :3268 / subagentDone :3362 replay fragments — covered by 5G-1
  fix? verify warning inside subagent tab fragment renders);
  (e) src/*.ts daemonHealth/restart flows vs iter 3-5 server changes
  (daemonHealth.js, AgentClient.ts reconnect/restart, reloadGuard).
  Then: 'clear' persisted event dropped by replay paths — likely by design
  (replay starts with fresh container), do NOT report without demonstrating
  user-visible inconsistency.

### Iteration 5 — Group F (web_server/diff_merge/merge_flow) — session notes

Read in full: diff_merge.py, merge_flow.py; web_server.py regions: \_WebMergeState +
reject helpers (396-710), \_schedule/\_cancel/\_fire_pending_tab_close + ws/uds
handlers (2865-3081), \_handle_ready/\_replay_merge_review/\_handle_submit/
\_register_merge_state/\_handle_web_merge_action/\_apply_web_merge_action (3440-3815),
\_http_response/trajectory responses/\_augment_merge_data/\_translate_webview_command
(2357-2510), \_process_request + auth (2716-2865).

**CANDIDATE BUG 5F-1 (web_server.\_fire_pending_tab_close ~line 2956)**: when the
deferred tab-close grace fires while a merge review is STILL IN FLIGHT, it silently
pops `_merge_states[tab_id]` + `_merge_action_locks` and dispatches `closeTab`.
Backend `_close_tab` sees `is_merging=True` → flips `frontend_closed=True` and defers
disposal until "the lifecycle ends" — but the lifecycle can now NEVER end: no merge
state remains, so every future mergeAction returns early, `all-done` is never
dispatched, `_finish_merge` never runs. Result: backend tab stuck `is_merging=True`
forever, agent leak, `_merge_data_dir(tab_id)` artifacts never cleaned,
`_present_pending_worktree` never fires (pending worktree orphaned), and a stuck
`is_merging` (worktree) tab can block other tabs' merges via busy guards.
FIX IDEA: in `_fire_pending_tab_close`, when a merge state was present, after the
`closeTab` cmd also dispatch `{"type":"mergeAction","action":"all-done","tabId":...,
"workDir": state.work_dir}` via `_run_cmd` (treat close-mid-review as accept-remaining:
no disk writes needed) so `_finish_merge` clears is_merging, cleans merge dir,
presents pending worktree, and `_dispose_if_closed` disposes the closed tab.
Need failing test first: real RemoteAccessServer + wss client, start merge via
\_register_merge_state path (like test_bughunt4_merge_replay_on_reconnect.py), drop
connection, fast-forward grace (patch \_TAB_CLOSE_GRACE small), assert backend tab
is_merging cleared / disposed.

Verified-looking-OK so far (no test yet): \_apply_web_merge_action accept path never
writes (correct: workspace already holds agent content); accept-file/accept-all no
writes; reject-all delta bookkeeping (iter-4 verified); \_replay_merge_review checks
remaining and pops happen before all-done; auth two-attempt flow + rate limiting;
\_process_request /media/ traversal guard; \_trajectory_job_response name validation;
\_augment_merge_data newline='' reads + binary skip.

Still to check: multiple simultaneous clients nav broadcast (merge_nav broadcast to
all — looks consistent since broadcast goes to everyone); \_merge_action_locks leak
for unknown tabIds (marginal); external file change mid-review (reject splices on
stale coordinates — likely wontfix/marginal); websocket backpressure on huge diffs
(\_schedule_send / \_discard_pending_send); upload/attachment handling in \_handle_submit
(clamps look fine); commands.py mergeAction all-done routing (verify _finish_merge
workDir plumbing).

### Iteration 5 — Group A (persistence.py / running_agent_state.py) — IN PROGRESS

- Session 1 (context exhausted early): read PROGRESS.md history, confirmed all
  iter-3/4 group-A fixes are committed (cd5d0b7b tip). Read persistence.py lines
  1-450 (RW lock, cache, \_get_db, \_most_recent_task_id, \_add_task) and the full
  function index (grep '^def'). NO new bugs identified yet — investigation has not
  started in earnest.
- Session 2: read persistence.py IN FULL + running_agent_state.py IN FULL.
  Analysis so far:
  - CANDIDATE BUG A (NaN extra divergence): every extra write uses
    `json.dumps(extra)` with default `allow_nan=True` → a float('nan')/inf value
    (e.g. cost) serialises as bare `NaN` which is INVALID JSON. The SQL predicate
    `_HISTORY_NOT_SUBAGENT` treats invalid JSON as NOT-subagent (ELSE 1), but
    Python `_is_subagent_row` uses `json.loads` which ACCEPTS NaN → says IS
    subagent. Divergence: a subagent row with NaN cost leaks into
    `_load_history`/`_search_history`/`_get_history_entry`/`_prefix_match_task`
    and consumes a `_list_recent_chats` limit slot then gets `continue`d
    (resurrects iter-4 bug). Need to verify subagent extra can carry float
    cost/tokens (check chat_sorcar_agent.\_run_tasks_parallel + task_runner
    \_save_task_extra payloads), then test+fix (json.dumps(..., allow_nan=False)
    with sanitisation, or make readers consistent).
  - running_agent_state.py: pure state container + registry with RLock; no logic
    to break — confirms prior "not a bug" verdict.
  - Still TODO: favorites round-trip consumers (commands.py/server.py/main.js
    `is_favorite` shaping), \_search_history shaping vs frontend, chat title
    paths, event replay ordering (events ORDER BY seq vs FIFO queue under
    writer restart), concurrent reader/writer under `_close_db` swap
    (`_add_task` captures `db = _get_db()` BEFORE write lock — stale conn write
    after swap?), `_record_frequent_task` eviction already ruled NOT bug.
- TODO for next session (remaining surface from task spec):
  favorites round-trip (`_set_task_favorite` L1020, `_save_task_extra` L1068
  preserve-is_favorite path) and interaction with delete/search; frequent-task
  ranking math (`_record_frequent_task` L1990, `_load_frequent_tasks` L2051,
  `_delete_frequent_task` L2031); `_search_history` L676 result shaping vs frontend
  expectations (compare with `_load_history` L625 and vscode frontend consumers);
  event coalescing/replay ordering (`_queue_chat_event` L1291, `_event_writer_loop`
  L1185, `_write_event_batch` L1225, `_flush_chat_events` L1326,
  `_fetch_events_for_task_id` L1498 — check seq vs id ordering); chat title
  generation/update paths (grep title in vscode server + persistence); concurrent
  reader/writer under db swap (`_maybe_reset_caches` L1148, `_close_db`);
  malformed `extra` JSON tolerance in every reader (`_is_subagent_row` L1821,
  `_load_history`, `_list_recent_chats` L1442, `_get_adjacent_task_by_chat_id`
  L1706, `_load_subagent_rows_by_parent_task_id` L1638). Write failing tests
  first in src/kiss/tests/agents/sorcar/test_bughunt5_<short>.py; no mocks;
  `uv run check --full` at the end. Do NOT re-report items listed as already
  fixed / already-verified-not-bugs in the task description above.

### Iter 5 group E (vscode server/commands/task_runner) — session 1 notes (context ran out)

- Read in full: commands.py (802 lines), task_runner.py (1161 lines),
  test_bughunt3_run_start_race.py (harness recipe: VSCodeServer(), override
  printer.broadcast, stub SorcarAgent.__mro__[1].run, stub
  _server_module.generate_followup_text, clear _RunningAgentState.running_agent_states
  in tearDown). server.py (1625 lines) NOT yet read — read next.
- Candidate leads spotted while reading (NOT yet verified — verify in next session):
  1. `_cmd_merge_action` only handles "all-done" and ignores workDir-less calls;
     check `_finish_merge` with empty tabId / double all-done (double-click) — does
     `_finish_merge` guard re-entry / missing tab?
  2. `_cmd_worktree_action`: no busy/empty-tabId guard at the command layer — relies
     on `_handle_worktree_action`'s `_check_worktree_busy`; check double merge clicks
     (two threads both passing busy check before `is_merging` set?) and empty tabId
     minting phantom via `_get_tab` inside `_handle_worktree_action` (server.py —
     must read).
  3. `_cmd_append_user_message`: broadcasts "prompt" event AFTER releasing state lock
     — task may end between append and broadcast (benign?); also no size limit.
  4. Attachments: no size limit on base64 data (task mentions "attachment size
     limits") — a huge attachment could OOM; check what frontend limits.
  5. `_run_task_inner`: `tab.chat_id = tab.agent.chat_id or tab.chat_id` — agent
     property; check stale agent chat_id from a kept-alive `_wt_pending` agent after
     `_new_chat` (new chat id on tab, old agent kept? does _new_chat dispose agent?)
  6. `_await_user_response`: q.get() with no timeout; if tab closed after queue
     resolution, watcher only wakes on stop event — closeTab does not set stop_event?
     (check `_close_tab` in server.py).
  7. `_subscribe_chat_viewers` broadcasts clear+status to viewers — viewer tabs of
     SUBAGENT events surface (task hint) — check subscribe of subagent task ids.
  8. Task hints remaining: _wt_pending vs daemon restart/crash recovery; resume of
     chat whose worktree branch deleted; daemon shutdown with running tasks;
     getFrequentTasks/favorites vs iter-4/5 persistence semantics.
- Next session: Read server.py fully (esp. _handle_worktree_action,
  _check_worktree_busy, _finish_merge, _close_tab, _new_chat, _replay_session,
  _present_pending_worktree, merge_flow imports), then write failing tests
  test_bughunt5_<short>.py.
- UPDATE (session 1b): server.py now read IN FULL (all 1625 lines). Ruled out:
  `_save_task_extra` preserves is_favorite (already merged, persistence.py:1102);
  `_handle_command` non-string type guarded; `_get_models` default-model refresh ok;
  `_replay_session` use_worktree guard present (iter4 fix).
- VERIFY NEXT (concrete candidates, none confirmed yet):
  (a) `_generate_followup_async`: worker sets thread_local.task_id then broadcasts
      followup_suggestion. If tab closed (printer.cleanup_tab dropped subscribers)
      BEFORE LLM returns, broadcast with task_id having NO subscribers — check
      WebPrinter.broadcast: does it fall back to global fan-out (leak into unrelated
      active tabs) or drop? Also JsonPrinter path.
  (b) double worktree merge click race: two `_cmd_worktree_action("merge")` threads —
      read merge_flow.py `_handle_worktree_action`/`_check_worktree_busy`: is busy
      check + is_merging set atomic under _state_lock? Window between check and set?
  (c) `_cmd_merge_action` all-done with empty tabId → `_finish_merge("")` behavior;
      double all-done (double click) re-entry.
  (d) `_new_chat` on a tab with pending worktree (_wt_pending agent kept alive):
      resets chat_id but keeps agent? does NOT release worktree and does not guard
      busy tab — docstring claims newChat only arrives for fresh tab ids; check
      main.js/extension whether newChat can target an existing tab (e.g. "New chat"
      button on existing tab) → leaked _wt_pending agent + use_worktree flag kept?
  (e) `_teardown_tab_resources` releases _wt_pending via _ensure_wt_agent(tab) — but
      `_close_tab` busy path defers; OK. Check _ensure_wt_agent for resurrect bugs.
  (f) resume chat whose worktree BRANCH deleted externally: `_emit_pending_worktree`
      (merge_flow.py) on _replay_session; and `_present_pending_worktree` after task.
  (g) daemon shutdown with running tasks: web_server `_stop_active_agent_tasks` sets
      interrupted_by_shutdown; check task_runner._cancel_outcome path persists; and
      check _wt_pending lifecycle vs daemon restart (worktree branch left, on restart
      _recover_orphaned_tasks only fixes result rows — replay of wt-pending chat after
      restart: tab.agent gone, `_emit_pending_worktree` reads agent=None → user loses
      merge/discard ability silently? expected: warning).
  (h) attachments: no size cap on base64 decode in _run_task_inner (hint: "attachment
      size limits") — check web_server max message size; UDS transport unbounded?
  (i) `_subscribe_chat_viewers`: does ChatSorcarAgent._run_tasks_parallel forward
      `_on_task_id_allocated` to sub-agent run()? (line ~219 agent.run(...) — check
      kwargs). If YES → viewer tabs of parent chat get 'clear' wiping parent view.
      If NO → hint (i) is N/A.
- Harness recipe (test_bughunt3_run_start_race.py): VSCodeServer(); override
  server.printer.broadcast with recording fn; stub SorcarAgent.__mro__[1].run to
  return "success: true\nsummary: ok\n"; stub _server_module.generate_followup_text;
  tearDown: _RunningAgentState.running_agent_states.clear().
