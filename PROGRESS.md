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

### Iteration 5 — Group C (sorcar CLI) — COMPLETE: 2 NEW bugs found+fixed

- BUG-5C-1 (cli_steering.py `_append_paste` + `feed` typed path): C1 control
  characters U+0080–U+009F (category Cc) passed both input filters — the
  `ch >= " "` guards only exclude C0/DEL (`"\x9b" >= " "` is True). U+009B is
  the one-character CSI introducer; once in `buf` it is emitted RAW to the
  terminal by `cli_panel.clip_buf` (which only rewrites newline/tab),
  corrupting the box row, and the queued instruction carries the raw C1
  bytes. Fix: paste filter excludes `"\x7f" <= ch <= "\x9f"`; typed path
  excludes `"\x80" <= ch <= "\x9f"` (DEL already eaten by the backspace
  branch). Test: test_bughunt5_c1_controls.py (4 tests, all failed pre-fix,
  incl. a clip_buf end-to-end render check).
- BUG-5C-2 (cli_repl.py `_read_line`): Ctrl+C at the idle prompt escaped
  `input()` with the cursor still on the panel's BODY row; `run_repl`'s
  handler then printed "\\n(Press Ctrl+C again …)" over the bottom border row
  without erasing it — the screen showed
  `(Press Ctrl+C again or type /exit to quit)────────…────╯` (and the second
  Ctrl+C's `Goodbye.` overprinted the next panel's rule the same way). Fix:
  both interactive `input()` calls in `_read_line` now catch
  KeyboardInterrupt, write `"\n" + ESC[2K` (step onto the rule row, erase
  it), flush, and re-raise. Test: test_bughunt5_ctrlc_prompt_border.py —
  pty.fork end-to-end `run_repl`, child byte stream replayed through a mini
  VT100 screen model; asserts the "(Press Ctrl+C" and "Goodbye." screen rows
  carry no `─`/`╯` remnants. Failed pre-fix with exactly the garbled row.
- Additionally investigated and ruled out this round (do NOT re-report):
  leftover pending_user_messages when the task finishes mid-typing are
  dropped by documented design (vscode task_runner.py ~line 167 clears them
  the same way — CLI consistent); ⏎ (U+23CE) is EAW=N width 1 so multi-line
  buf cursor math is exact; property sweep over panel math (body width ==
  cols-4, cursor col in [1, cols-1]) for emoji/CJK/combining/multi-line/tab
  buffers at cols 10..80 — all exact; wide-char paste split byte-by-byte
  across reads + backspaces — buffer exact; SIGINT during paste covered by
  the bughunt4 KeyboardInterrupt path (\\x03 bytes inside a paste dropped,
  already tested); prompt text with `{braces}` is used verbatim (no .format);
  `box.start()` failure leaving the proxy installed has no realistic trigger
  after supports_steering() (not reproducible without test doubles).
- Verification: 141/141 CLI tests pass (every test_cli\_\* / CLI bughunt file,
  incl. the 5 new bughunt5 tests); `uv run check --full` passes.

Superseded session-1 scratch notes (kept for audit; conclusions above):

1. C1 control chars pass `_append_paste` filter (cli_steering.py `_append_paste`:
   `ch >= " "` keeps U+0080–U+009F, e.g. U+009B = single-char CSI) → pasted C1
   lands in buf and is written RAW to the terminal in the body row (clip_buf only
   replaces \\n and \\t) → can corrupt the box/terminal. Same hole in `feed`'s typed
   path (`elif ch >= " ": buf += ch`). Test: paste b"\\x1b\[200~a\\xc2\\x9bXb\\x1b\[201~",
   assert buf has no U+009B / rendered body has no raw C1.
1. `clip_buf` "⏎" replacement char U+23CE: verify `char_width("⏎")` —
   unicodedata.east_asian_width(U+23CE) is "W"?? If char_width=2 but real terminals
   render 1 col, `body_cursor_col` parks the caret off-by-one per newline in buf
   (multi-line paste). MUST check actual EAW value in python first; compare with
   wcwidth if available. If EAW=N width 1, fine — drop.
1. `clip_buf` tail-clip can return a slice STARTING with a combining mark (backward
   loop keeps cw=0 combining char, breaks on its base char) → mark combines with the
   border space on screen. Check severity; width math itself is consistent.
1. `_pending_esc` 64-cap drop (feed end): pending dropped wholesale; if next chunk
   then begins with the CONTINUATION of that escape, its tail bytes are typed as
   literal buf text. Per code comment this is by-design for absurd sequences —
   probably NOT bug; only pursue if a real ≤64 sequence (paste marker straddling
   the cap with a long preceding CSI) can trigger it.
1. `SteeringSession.run`: if `box.start()` raises (termios.error race: tty closed
   after supports_steering()), the \_StdoutProxy stays installed (sys.stdout never
   restored) and termios state half-applied — start() is called OUTSIDE the
   try/finally. Fix: move start() inside try or wrap. Verify reproducible: call
   run() with stdin not a real tty so tcgetattr raises.
1. `_on_submit` when task just finished (`_done` set between draw and Enter):
   message appended to state.pending_user_messages but never drained → silently
   lost (no warning). Compare with VS Code path behavior; maybe print "task
   already finished" or run queued as new task. Decide if bug.
1. `CliCompleter._model_matches`: `/model name extra` → query "name extra"; minor.
   `_build_matches` at-mention regex `@([^\s]*)$` only matches at EOL (cursor at
   end) — readline passes text up-to-cursor, so fine.
   Ruled out by reading (do NOT re-report): paste-start/Shift+Enter/CSI split across
   reads (pending_esc covers all; verified by trace); partial \_PASTE_END suffix len
   clamp; `keep` math; ESC-as-paste-content stripping consistent with unsplit; \\x7f
   ordering in elif chain; body_cursor_col ≤ cols-1 incl. wide chars; panel_top/bottom
   ASCII-only inputs; Ctrl+C-during-question unblock (covered by \_interrupt_worker);
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

### Iteration 5 — Group B (git_worktree / worktree_sorcar_agent) — COMPLETE: 2 NEW bugs found+fixed

- BUG-5B-1 (git_worktree.py `_git`, ~line 95): repo-scoped `GIT_*` environment
  variables (GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE / GIT_OBJECT_DIRECTORY /
  GIT_COMMON_DIR / ...) inherited from the parent process — e.g. when KISS is
  launched from a git hook (`post-commit` starting an agent), `git rebase --exec`, or a user shell export — OVERRIDE git's `-C <cwd>` repository
  discovery, so EVERY worktree git call silently targeted the WRONG repo:
  `discover_repo(B)` returned the hook repo A, `has_uncommitted_changes(B)`
  reported A's state (dirty B → "clean" → user dirty state never copied into
  the worktree), `stage_all`/commits mutated A's index, kiss branches were
  created in A. Reproduced with two real repos + real env vars. Fix: `_git`
  now scrubs the repo-scoped variable list (mirrors git's own
  `local_repo_env` cleared by `git submodule`; author/committer/config vars
  kept) from the subprocess env. Test: test_bughunt5_git_env_leak.py
  (4 tests, all failed pre-fix).
- BUG-5B-2 (git_worktree.py `_git`): stdout/stderr decoded with STRICT
  `encoding="utf-8"` while `-c core.quotepath=false` makes git emit bytes
  > 0x7F verbatim — a legal non-UTF-8 path in the repo (e.g. a Latin-1
  > filename committed on Linux; injectable via `git update-index --cacheinfo`
  > with bytes argv even on macOS) made EVERY git call whose output mentions
  > the path raise UnicodeDecodeError: `has_uncommitted_changes`,
  > `copy_dirty_state`, `stash_if_dirty`... The error (a ValueError, NOT the
  > OSError that `_try_setup_worktree` guards) propagated out of
  > `WorktreeSorcarAgent.run()` and killed the whole task. Also internally
  > inconsistent: `_unquote_git_path` already decodes with surrogateescape.
  > Fix: `errors="surrogateescape"` in `_git` (round-trips through
  > `os.fsencode` for filesystem ops, matching `_unquote_git_path`). Test:
  > test_bughunt5_invalid_utf8_path.py (3 tests, all failed pre-fix).
- Investigated and verified NOT bugs (do not re-report): autocommit message
  paths (empty LLM output → `clean_llm_output(raw) or fallback`, empty diff →
  fallback, so `commit_staged -m ""` is unreachable); smudge/clean filter
  failure during worktree staging (stage_all stages nothing → BUG-4B-2's
  has_uncommitted_changes fallback catches it); promisor/partial-clone offline
  (`worktree add` fails → create() False → documented direct-execution
  fallback); branch checked out in a user-made second worktree (delete_branch
  returns False → discard() emits the existing delete_warning); merge()
  CONFLICT keeps `_wt` for retry and `_finalize_worktree` is idempotent on a
  vanished wt dir (iter-3/4 coverage); wt-dir name collisions after
  crash-restart impossible in practice (time + uuid4-hex8 slug, and
  cleanup_orphans holds repo_lock).
- Verification: 7/7 new tests fail pre-fix, pass post-fix; 513 impacted
  worktree/bughunt/autocommit/baseline/git sorcar tests run in 8 parallel
  shards — only failures were the documented pre-existing forkpty/pty load
  flakes (test_bughunt_cli ctrl-c, test_bughunt4_interrupt_lock; both pass in
  isolation) and parallel group A's test_bughunt5_nan_extra order-dependent
  flake (passes in isolation, persistence-only, unrelated); `uv run check --full` clean. Fixes+tests committed (with parallel groups' work) in
  1de07413.

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
(web_use_tool.close, callback/pre_step_hook reset); \_run_single sets
tl.stop_event each task (pool thread reuse safe); \_coerce_tasks (iter-3/4);
sub_usage aggregation order (results from pool.map are input-ordered;
sub_usage indexed by idx — ordering correct).

- BUG-5D-3 (chat_sorcar_agent.py `run()` summary extraction): when the
  result YAML is a dict whose `summary` value is a LIST or nested MAPPING
  (LLMs routinely emit `summary:\n  - did x\n  - did y`), the raw Python
  object was passed to `_save_task_result` → sqlite3.ProgrammingError
  ("type 'list' is not supported") raised FROM THE `finally` BLOCK,
  replacing the task's successful return value with an exception and
  skipping `_save_task_extra` (tokens/cost lost). Fix: coerce non-string
  summary — str kept, None → "", list/dict → `yaml.safe_dump(...).strip()`.
  Test: test_bughunt5_summary_nonstring.py (4 tests; list+dict cases
  failed pre-fix with ProgrammingError; None + plain-string guards).
- BUG-5D-4 (web_use_tool.py `screenshot` + sorcar_agent.py `_get_tools`):
  screenshot resolved paths against the DAEMON PROCESS cwd, not the agent
  work_dir, and had no `PWD/` expansion — inconsistent with
  Read/Write/Edit (`_expand_pwd_prefix`) and Bash (`cwd=work_dir`). In
  worktree mode `screenshot("shot.png")` silently escaped the worktree;
  `screenshot("PWD/tmp/x.png")` created a junk literal `PWD/` dir in the
  process cwd (the `**_kwargs` swallowed any work_dir). Fix: WebUseTool
  gains explicit `work_dir` param; screenshot expands PWD/ and anchors
  relative paths at work_dir; `_get_tools` passes
  `WebUseTool(work_dir=self.work_dir)`. Test:
  test_bughunt5_screenshot_workdir.py (real headless Chromium, 3 tests;
  2 failed pre-fix; absolute-path regression guard).

Examined in chat_sorcar_agent.py / web_use_tool.py, NOT bugs (do not
re-report): build_chat_prompt MAX_TASKS middle-deletion (keeps first 2 +
last 8, intentional) and renumbering; tl.task_id restore-to-"" in finally
(guarded by == task_key; nested same-thread chat runs don't exist);
go_to_url tab:N int() inside try; tab:list crashed-page title() caught;
\_check_for_new_tab; \_resolve_locator re-snapshot + visibility scan;
scroll negative amount (empty range); close() idempotent + atexit
unregister; \_is_profile_in_use EPERM=true / pid\<=0=false (iter-3);
\_clean_singleton_locks unconditional-safe; ask_user_question callback
plumbing (str() coercion; queue semantics live in vscode group E scope).

DONE this session: all 14 bughunt5 group-D tests pass. REMAINING: run
impacted test sweep (shards), `uv run check --full`, commit.

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
- **BUG 5G-2 (fixed)** media/main.js top-level `case 'warning'` dropped LIVE
  warnings stamped for a BACKGROUND tab (`if tabId !== activeTabId break`),
  while every other display event reaches the owning tab's `outputFragment`
  via the `default:` route → `processOutputEventForBgTab`. A worktree task
  finishing with a stash-pop warning while the user viewed another tab lost
  the warning forever (tab switch showed the result but not the warning).
  FIX: tabId-mismatch now mirrors the default route (`findTabByEvt` →
  `processOutputEventForBgTab`, which renders via the new handleOutputEvent
  warning case); unknown/foreign-window tab ids still dropped.
  Test: src/kiss/agents/vscode/test/bughunt5_warning_bgtab.test.js (3 tests:
  bg-tab warning survives tab switch (failed pre-fix; control system_output
  passed), foreign-window drop guard, active-tab single-render guard).
- Group G iteration-5 COMPLETE: 2 NEW bugs (commits 8e170445, c037fd12).
  Investigated, NOT bugs (do not re-report):
  (a) autocomplete multi-line/unicode — ghost flow is exact-echo
  (`ev.query === inp.value` string compare, unicode/astral-safe; backend
  `match[len(query):]` slices code points; `clip_autocomplete_suggestion`
  stops at the first newline, whitespace-gap rules hold for newline-ending
  queries);
  (b) helpers.rank_file_suggestions separators — substring match against
  paths produced by the same `_scan_files` cache the user picks from,
  separator-agnostic;
  (c) vscode_config save_config on read-only HOME — `.config.lock` open
  fails at the same point pre-flock `mkstemp` already did (no regression;
  load_config catches OSError);
  (d) package.json contributes.configuration (kissSorcar.defaultModel,
  kissSorcar.kissProjectPath) has zero overlap with vscode_config DEFAULTS —
  no drift;
  (e) demo.js grouping of persisted warning/autocommit_done — they stay in
  the current panel group and render via processEvent post-5G-1;
  (f) src/\*.ts daemon health/restart — iters 3-5 changed handler robustness
  only, no protocol/response shape change; all 21 daemonHealth tests pass;
  (g) persisted 'clear' not replayed by replayEventsInto — by design (replay
  starts with a fresh container).
  Verification: all 11 JS test files pass (`node`), `npx tsc -p .` clean,
  eslint clean on main.js, impacted Python tests pass
  (test_bughunt4_warning_persist.py, test_replay_event_coalescing.py),
  `uv run check --full` passes.

### Iteration 5 — Group F (web_server/diff_merge/merge_flow) — session notes

Read in full: diff_merge.py, merge_flow.py; web_server.py regions: \_WebMergeState +
reject helpers (396-710), \_schedule/\_cancel/\_fire_pending_tab_close + ws/uds
handlers (2865-3081), \_handle_ready/\_replay_merge_review/\_handle_submit/
\_register_merge_state/\_handle_web_merge_action/\_apply_web_merge_action (3440-3815),
\_http_response/trajectory responses/\_augment_merge_data/\_translate_webview_command
(2357-2510), \_process_request + auth (2716-2865).

**BUG 5F-1 FIXED (web_server.\_fire_pending_tab_close ~line 2956)**: fix = new
`_finish_merge_and_close_tab(tab_id, merge_state)` coroutine: when a merge state was
popped, dispatch `mergeAction all-done` (workDir=state.work_dir) BEFORE `closeTab`
(all-done first so a non-busy tab isn't popped then re-created as a phantom by
`_finish_merge`'s `_get_tab`). Test
test_bughunt5_close_mid_merge.py::test_deferred_close_mid_review_ends_merge_and_disposes
failed pre-fix (leaked is_merging=True frontend_closed=True tab), passes post-fix;
reconnect-within-grace control test passes.

**BUG 5F-2 FIXED (web_server.\_dispatch_client_command)**: explicit `closeTab` from a
STILL-CONNECTED web client mid-review left `_merge_states[tab_id]` registered and the
backend tab stuck is_merging/frontend_closed forever (web UI lets you close a chat tab
any time; closing destroys the only review UI). Fix: intercept `closeTab` for non-UDS
endpoints (`not isinstance(endpoint, asyncio.StreamWriter)`), pop merge state + action
lock, delegate to `_finish_merge_and_close_tab`. UDS (VS Code) exempt: TS MergeManager
owns reviews in real editor tabs that survive chat-tab closure and still sends
all-done. Test: ::test_explicit_close_tab_mid_review_ends_merge (failed pre-fix).

**BUG 5F-3 FIXED (web_server.\_replay_merge_review)**: the reconnect/reload replay
read the reviewed files (`_augment_merge_data`) and the shared hunk dicts WITHOUT the
per-tab `_merge_action_lock` that serialises every merge action — a browser reloading
while another client's reject/reject-all was mid file-rewrite received a torn
`current_text` (open(w) truncates before writing) and/or mid-mutation cs offsets that
no later merge_nav can repair (merge_nav carries no text). Fix: wrap the replay body
in `async with self._merge_action_lock(tab_id)` with a state re-check under the lock
(review may have finished while waiting). Tests:
test_bughunt5_replay_action_race.py (2 tests: replay must not emit while the action
lock is held + must emit after release; post-wait re-check sends nothing for a
finished review; both failed pre-fix).

**BUG 5F-4 FIXED (web_server.\_dispatch_client_command mergeAction all-done)**: the
VS Code TS MergeManager runs per-hunk review entirely in the extension host and sends
ONLY `mergeAction all-done` to the backend (SorcarSidebarView.sendMergeAllDone); the
dispatch path forwarded it to `_cmd_merge_action`/`_finish_merge` but never popped the
server-side shadow `_WebMergeState` registered at merge_data-broadcast time. The
stale fully-unresolved state (a) replayed a ZOMBIE merge review to the VS Code webview
on the next reload (`ready` → `_replay_merge_review`), (b) leaked one state with full
file payloads per finished review, (c) made the deferred-close path fire a spurious
second all-done (phantom `_get_tab` + spurious autocommit_prompt). Fix: when
`mergeAction action=all-done` arrives from a client, pop `_merge_states[tab_id]` +
`_merge_action_locks[tab_id]` before falling through to the backend command. Test:
test_bughunt5_stale_state_after_all_done.py (real UDS transport like the extension;
zombie-replay test failed pre-fix; web-driven-review control passed).

ORIGINAL 5F-1 analysis: when the
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
`closeTab` cmd also dispatch `{"type":"mergeAction","action":"all-done","tabId":..., "workDir": state.work_dir}` via `_run_cmd` (treat close-mid-review as accept-remaining:
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
(clamps look fine); commands.py mergeAction all-done routing (verify \_finish_merge
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
- Session 2 results — BUG 5A-1 CONFIRMED, REPRODUCED, FIXED:
  - **BUG 5A-1** (persistence.py): every `extra` write (`_add_task`,
    `_save_task_extra`, `_set_task_favorite`) used plain `json.dumps`
    (`allow_nan=True`) → a non-finite float (NaN cost, e.g.
    `round(tab.agent.budget_used, 6)` in task_runner.py:724) serialises as the
    bare `NaN` token = INVALID RFC-8259 JSON. SQLite `json_valid` (used by
    `_HISTORY_NOT_SUBAGENT`) rejects it ⇒ SQL says "not subagent"; Python
    `json.loads` (in `_is_subagent_row`) ACCEPTS NaN ⇒ Python says "subagent".
    Effects reproduced: NaN-cost subagent row leaks into `_load_history`
    sidebar; subagent-only chat with NaN cost eats a `_list_recent_chats`
    limit slot (resurrects iter-4 bug); legacy NaN rows classified
    inconsistently end-to-end.
  - Fix: new `_dumps_extra` (json.dumps allow_nan=False; on ValueError
    recursively replaces non-finite floats with None via
    `_sanitize_non_finite`) used by all 3 extra writers; new strict
    `_parse_extra_dict` (json.loads with `parse_constant` raising → mirrors
    `json_valid` semantics) now used by `_is_subagent_row`,
    `_subagent_child_ids`, `_load_subagent_rows_by_parent_task_id` so legacy
    corrupt rows are classified identically by SQL and Python sides.
  - Test: src/kiss/tests/agents/sorcar/test_bughunt5_nan_extra.py — 6 tests,
    ALL 6 failed pre-fix, all pass post-fix.
  - Investigated and ruled NOT bugs this session (do not re-chase):
    favorites round-trip (`_cmd_set_favorite` uses guarded `_parse_int`;
    `_save_task_extra` preserves is_favorite; `_get_history` shapes
    is_favorite from extra correctly; delete/search interactions clean);
    `_search_history` shaping identical to `_load_history`, consumed by
    `server._get_history` with per-field guarded coercion; chat title = task
    text (no separate title path in persistence); event replay ordering
    (single FIFO queue + single writer thread + per-task seq cache seeded
    under write lock ⇒ `ORDER BY seq` replay matches enqueue order; sync
    `_append_chat_event` funnels through the same queue);
    `_event_writer_loop` shutdown-sentinel handling; `_get_history`'s
    is_subagent marking is dead code (rows pre-filtered by SQL) — harmless;
    `_record_frequent_task` eviction + `_load_frequent_tasks` ranking
    (count DESC, timestamp DESC — consistent with eviction order);
    `_log_orphaned_task_forensics` malformed-extra tolerance (broad except);
    db-swap stale-conn write in `_add_task` (test-fixture-only scenario,
    previously ruled under RW-lock discipline).
  - Verification: all 6 new tests pass; 1284-test regression sweep (all 127
    test files importing persistence, 8 parallel shards) — only failures were
    2 tests in a PARALLEL group-E commit's
    test_bughunt5_replay_phantom_state.py that pass in isolation
    (shared-registry ordering flake, unrelated to persistence);
    `uv run check --full` clean. Code+tests landed in commit 1de07413.
  - Group A iteration 5: COMPLETE — 1 new bug (BUG 5A-1) found and fixed.
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
  first in src/kiss/tests/agents/sorcar/test_bughunt5\_<short>.py; no mocks;
  `uv run check --full` at the end. Do NOT re-report items listed as already
  fixed / already-verified-not-bugs in the task description above.

### Iter 5 group E (vscode server/commands/task_runner) — session 1 notes (context ran out)

- Read in full: commands.py (802 lines), task_runner.py (1161 lines),
  test_bughunt3_run_start_race.py (harness recipe: VSCodeServer(), override
  printer.broadcast, stub SorcarAgent.__mro__[1].run, stub
  \_server_module.generate_followup_text, clear \_RunningAgentState.running_agent_states
  in tearDown). server.py (1625 lines) NOT yet read — read next.
- Candidate leads spotted while reading (NOT yet verified — verify in next session):
  1. `_cmd_merge_action` only handles "all-done" and ignores workDir-less calls;
     check `_finish_merge` with empty tabId / double all-done (double-click) — does
     `_finish_merge` guard re-entry / missing tab?
  1. `_cmd_worktree_action`: no busy/empty-tabId guard at the command layer — relies
     on `_handle_worktree_action`'s `_check_worktree_busy`; check double merge clicks
     (two threads both passing busy check before `is_merging` set?) and empty tabId
     minting phantom via `_get_tab` inside `_handle_worktree_action` (server.py —
     must read).
  1. `_cmd_append_user_message`: broadcasts "prompt" event AFTER releasing state lock
     — task may end between append and broadcast (benign?); also no size limit.
  1. Attachments: no size limit on base64 data (task mentions "attachment size
     limits") — a huge attachment could OOM; check what frontend limits.
  1. `_run_task_inner`: `tab.chat_id = tab.agent.chat_id or tab.chat_id` — agent
     property; check stale agent chat_id from a kept-alive `_wt_pending` agent after
     `_new_chat` (new chat id on tab, old agent kept? does \_new_chat dispose agent?)
  1. `_await_user_response`: q.get() with no timeout; if tab closed after queue
     resolution, watcher only wakes on stop event — closeTab does not set stop_event?
     (check `_close_tab` in server.py).
  1. `_subscribe_chat_viewers` broadcasts clear+status to viewers — viewer tabs of
     SUBAGENT events surface (task hint) — check subscribe of subagent task ids.
  1. Task hints remaining: \_wt_pending vs daemon restart/crash recovery; resume of
     chat whose worktree branch deleted; daemon shutdown with running tasks;
     getFrequentTasks/favorites vs iter-4/5 persistence semantics.
- Next session: Read server.py fully (esp. \_handle_worktree_action,
  \_check_worktree_busy, \_finish_merge, \_close_tab, \_new_chat, \_replay_session,
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
  check + is_merging set atomic under \_state_lock? Window between check and set?
  (c) `_cmd_merge_action` all-done with empty tabId → `_finish_merge("")` behavior;
  double all-done (double click) re-entry.
  (d) `_new_chat` on a tab with pending worktree (\_wt_pending agent kept alive):
  resets chat_id but keeps agent? does NOT release worktree and does not guard
  busy tab — docstring claims newChat only arrives for fresh tab ids; check
  main.js/extension whether newChat can target an existing tab (e.g. "New chat"
  button on existing tab) → leaked \_wt_pending agent + use_worktree flag kept?
  (e) `_teardown_tab_resources` releases \_wt_pending via \_ensure_wt_agent(tab) — but
  `_close_tab` busy path defers; OK. Check \_ensure_wt_agent for resurrect bugs.
  (f) resume chat whose worktree BRANCH deleted externally: `_emit_pending_worktree`
  (merge_flow.py) on \_replay_session; and `_present_pending_worktree` after task.
  (g) daemon shutdown with running tasks: web_server `_stop_active_agent_tasks` sets
  interrupted_by_shutdown; check task_runner.\_cancel_outcome path persists; and
  check \_wt_pending lifecycle vs daemon restart (worktree branch left, on restart
  \_recover_orphaned_tasks only fixes result rows — replay of wt-pending chat after
  restart: tab.agent gone, `_emit_pending_worktree` reads agent=None → user loses
  merge/discard ability silently? expected: warning).
  (h) attachments: no size cap on base64 decode in \_run_task_inner (hint: "attachment
  size limits") — check web_server max message size; UDS transport unbounded?
  (i) `_subscribe_chat_viewers`: does ChatSorcarAgent.\_run_tasks_parallel forward
  `_on_task_id_allocated` to sub-agent run()? (line ~219 agent.run(...) — check
  kwargs). If YES → viewer tabs of parent chat get 'clear' wiping parent view.
  If NO → hint (i) is N/A.
- CONFIRMED CANDIDATE BUG-5E-1 (merge_flow.py:759 `_check_worktree_busy`): the guard
  checks `tab.is_task_active` and `_any_non_wt_running()` but NEVER `tab.is_merging`.
  Double-click on Merge (or merge+discard concurrently) on the SAME tab: thread 1
  sets is_merging=True, releases \_state_lock, blocks in repo_lock/wt.merge() (slow:
  LLM commit message); thread 2 passes busy check (is_merging not checked), sets
  is_merging=True again, queues on repo_lock, then re-runs wt.merge()/wt.discard()
  on the already-merged worktree; ALSO whichever finishes first clears is_merging in
  its finally while the other is still merging → non-wt task can start mid-merge.
  FIX: add `if tab.is_merging: return {...refused...}` to `_check_worktree_busy`
  (check happens under \_state_lock in `_handle_worktree_action`, so check+set
  atomic). TEST plan: VSCodeServer harness; tab with use_worktree=True and a stub
  agent (real WorktreeSorcarAgent with \_wt_pending=True and merge() overridden via
  subclass to block on an Event + count calls); thread1 worktreeAction merge,
  wait until inside merge(), thread2 worktreeAction merge → assert second result
  success=False/"already in progress" and merge called exactly once.
  NOTE: sub-checks also needed? `_check_worktree_busy` is also used elsewhere?
  (grep: only \_handle_worktree_action). Also `_run_task_inner`'s non-wt start guard
  scans `t.is_merging and t.use_worktree` for ALL tabs — already fine.
- Also noted: `_finish_merge` uses `_get_tab` (mints registry entry+agent for unknown
  tab id) — phantom but disposable (non-empty id) — marginal, skip.
- STATUS (session 1b): BUG-5E-1 FIXED — merge_flow.py `_check_worktree_busy` now
  refuses when `tab.is_merging` (test_bughunt5_double_merge_click.py, 2 tests,
  failed pre-fix, pass post-fix). BUG-5E-2 FIXED — merge_flow.py
  `_present_pending_worktree` now uses non-creating registry lookup instead of
  `_get_tab` (test_bughunt5_replay_phantom_state.py, 2 tests, failed pre-fix, pass
  post-fix; C2/C3 invariant restored for history-click viewer tabs).
  Committed as cab64b7e (incl. updates to test_detach_tab_and_reattach.py and
  test_no_git_error_on_empty_worktree.py which had codified pre-fix behavior).
- BUG-5E-3 FIXED (task_runner.py `_ask_user_question`, commit 2a81b4b1): a stale
  multi-viewer duplicate answer (viewer B's still-open askUser modal submitted after
  viewer A's answer was consumed) sat in the maxsize=1 queue and instantly answered
  the NEXT ask_user_question — the user never saw question 2. Fix: extracted
  `_resolve_task_answer_queue()`; `_ask_user_question` drains the queue (under
  \_state_lock) BEFORE broadcasting the new question. Test:
  test_bughunt5_stale_user_answer.py (failed pre-fix). Updated
  test_vscode_tabs.py::test_ask_user_broadcasts_question which pre-seeded the
  answer before asking (now delivers it after the askUser broadcast).
- Verified NOT bugs in iter-5 group E (do NOT re-report): followup_suggestion after
  tab close (WebPrinter fan-out is subscriber-only — no global leak; origin-db-path
  guard present); attachment size (64MB `_MAX_LINE_BYTES` cap on both WS max_size
  and UDS readline is a deliberate transport limit); `_wt_pending` vs daemon
  restart (documented design: "no cross-process restoration", branch deliberately
  preserved for manual resolution); double mergeAction all-done (`_finish_merge`
  idempotent); `_save_task_extra` vs setFavorite on a running task (is_favorite
  explicitly merged back, persistence.py:1102); sub-agents do not forward
  `_on_task_id_allocated` (no viewer-wipe path for subagent task allocation);
  `_handle_command` non-string type; getFrequentTasks limit parse.
- Verification: full vscode suite (991 tests) green in 8 parallel shards post-fixes;
  ask-flow tests (sorcar+vscode) green; `uv run check --full` passes.
- Iter-5 group E result: 3 NEW bugs found+fixed (BUG-5E-1, BUG-5E-2, BUG-5E-3).
- mf1/mf2/mf3 extracts in tmp/: merge_flow regions; tmp/server_part2/3.txt = server.py
  820-1625; tmp/csa_parallel.txt = chat_sorcar_agent \_run_single (sub-agents do NOT
  forward \_on_task_id_allocated → candidate (i) N/A).
- Harness recipe (test_bughunt3_run_start_race.py): VSCodeServer(); override
  server.printer.broadcast with recording fn; stub SorcarAgent.__mro__[1].run to
  return "success: true\\nsummary: ok\\n"; stub \_server_module.generate_followup_text;
  tearDown: \_RunningAgentState.running_agent_states.clear().

## Iteration 5 — COMPLETE (verified by orchestrator)

18 NEW bugs found+fixed across 7 groups (A:1 NaN/Infinity in `extra` JSON breaking
SQL-vs-Python subagent classification; B:2 GIT\_\* env leakage in `_git` + strict UTF-8
decode crash on non-UTF-8 paths; C:2 C1 control chars passing input filters + Ctrl+C
overprinting panel border; D:4 Bash strict-UTF-8 output loss, Read/Write FIFO/device
hang, non-string summary sqlite crash in finally, screenshot escaping work_dir; E:3
double-merge-click guard missing is_merging, \_present_pending_worktree phantom
registry entries, stale multi-viewer user answers; F:4 deferred close mid-merge-review
leak, explicit web closeTab mid-review limbo, replay racing in-flight merge actions,
zombie merge state after client all-done; G:2 persisted warnings dropped on replay,
live warnings for background tabs dropped). All 44 Python bughunt5 tests + 2 JS tests
pass; `uv run check --full` green.

## Iteration 6 (current)

- Iteration 5 still found bugs → launching iteration 6, same 7 groups, tests
  `test_bughunt6_*` / `bughunt6_*.test.js`. Stop when an iteration finds zero bugs.

### Iteration 6 — Group G (vscode helpers + frontend consistency) — COMPLETE: 3 NEW bugs found+fixed

- BUG 6G-1 (commit 94d7b422, media/main.js `case 'files'` + autocomplete.py
  `_emit_files`/`_get_files` + types.ts): the `files` reply for a `getFiles`
  cache miss arrives ASYNCHRONOUSLY after a background directory scan
  (potentially seconds on a big work dir), and main.js rendered every `files`
  event unconditionally — a late reply re-opened the @-mention picker (with
  `acIdx = 0`) after the user had deleted the mention and typed a plain task,
  so the next Enter was swallowed by the phantom picker instead of submitting.
  Protocol gap: unlike `ghost` (echoes `query`), `files` carried no `prefix`,
  so the frontend COULDN'T staleness-check. Fix: backend stamps `prefix` on
  every files event; main.js ignores replies when no @-mention is being typed
  (hideAC) and drops replies whose prefix no longer matches the typed query
  (prefix-less events still render for back-compat); types.ts files event
  gains `prefix?`/`loading?`. Tests: test/bughunt6_files_stale.test.js (4 JS
  tests, 2 failed pre-fix) + test_bughunt6_files_prefix.py (2 tests, both
  failed pre-fix).
- BUG 6G-2 (commit ce1db9b0, helpers.py `clip_autocomplete_suggestion`):
  vestigial `strip('"').strip("'")` from the LLM-suggestion era corrupted
  history suffixes whose boundary holds a REAL quote character: history
  `run "make test"` typed as `run "make` suggested ` test` instead of
  ` test"` (accepting typed the unbalanced `run "make test` the user never
  submitted); history `echo "hi" done` typed as `echo ` suggested `hi" done`.
  Same invariant as the iter-3 echo-strip fix: accepting a history ghost must
  reproduce the matched task exactly. Removed the quote-strip and the now
  unused `_strip_surrounding_quotes` (quote-stripping belongs only to
  `clean_llm_output`). Test: test_bughunt6_ghost_quote_suffix.py (3 tests,
  2 failed pre-fix with exactly the corrupted completions; gap-normalisation
  regression guard).
- BUG 6G-3 (commit e83e54fa, vscode_config.py `apply_config_to_env`): bare
  `float(cfg["max_budget"])` — the value comes from the user-editable
  `~/.kiss/config.json` AND from any client's `saveConfig` payload; a
  non-numeric value (`"abc"`, `None`) raised ValueError/TypeError out of
  `_cmd_save_config` → `_handle_command` → transport receive loop, killing
  the whole client connection (same escape path as the iter-3 unguarded-int
  handler bugs); SorcarAgent's startup caller swallowed it (`except: pass`),
  silently skipping budget application. Fix: fall back to
  `DEFAULTS["max_budget"]` when not float-convertible (numeric strings still
  apply). Test: test_bughunt6_budget_junk.py (4 tests, 3 failed pre-fix).
- Investigated and verified NOT bugs this round (do NOT re-report):
  `parse_result_yaml` guarantees the `summary` key (no KeyError in
  `_broadcast_result`); `usage_info`(total_steps)/`result`(step_count) keys
  match the frontend exactly (live + bg-tab paths); connId routing is
  server-side (web_server.py:1751 pops connId and targets the requesting
  connection; UDS per window) so main.js needn't filter ghost/files;
  `autocommit_progress`/`worktree_progress`/`worktree_created` unhandled in
  main.js are VS Code-native notifications handled by SorcarSidebarView
  (browser drops them harmlessly via the default route — feature gap, not
  incorrect behavior); demo.js read in full (grouping, result streaming,
  esc/sanitize — consistent with post-iter-5 main.js contract); ghost clear
  flow (requestGhost clears + 300ms debounce; cursor-at-end + @-ctx guards);
  `acceptGhost` append assumes cursor-at-end (guaranteed by requestGhost);
  extension relays whole backend messages so new fields pass through;
  main.js default branch ignores unknown event types safely (taskId-adoption
  guard intact).
- Verification: all 13 JS test files pass (`node`), `npx tsc -p .` clean,
  eslint clean on main.js, 25-test focused Python sweep + 105+251-test
  impacted sweeps green, `uv run check --full` passes.

### Iteration 6 — Group B (git_worktree / worktree_sorcar_agent) — COMPLETE: 2 NEW bugs found+fixed

- BUG-6B-1 (git_worktree.py `ensure_excluded`, ~line 668): the existing
  `<git_common_dir>/info/exclude` was read with STRICT
  `Path.read_text()`. Git treats exclude files as raw BYTES — non-UTF-8
  patterns/comments (e.g. a Latin-1 filename pattern, the exact
  use-case of BUG-5B-2) are legal and honored by git. The
  UnicodeDecodeError was swallowed by `_try_setup_worktree`'s broad
  except, so the `.kiss-worktrees/` entry was silently NEVER added:
  `?? .kiss-worktrees/...` polluted the user's `git status` forever,
  `has_uncommitted_changes(repo)` misreported a clean repo as dirty,
  and every merge ran a junk "kiss: auto-stash before merge"
  push/pop cycle (git prints "Ignoring path .kiss-worktrees/...").
  Fix: read via `read_bytes().decode("utf-8", errors="surrogateescape")`
  (mirrors `_git`'s policy); append opened with explicit utf-8.
  Test: test_bughunt6_exclude_nonutf8.py (3 tests; 2 failed pre-fix,
  idempotency guard).
- BUG-6B-2 (worktree_sorcar_agent.py `_try_setup_worktree`, ~line 460):
  `released_branch` (the ORIGINAL branch of the PREVIOUS pending
  worktree, returned by `_release_worktree()`) was reused as the new
  worktree's `original_branch` even when the new task targets a
  DIFFERENT git repo (the user changed `work_dir` between two runs of
  the same agent). Reproduced WRONG MERGE RESULT: task 1 in repoA
  (original branch `develop`), task 2 run from repoB's `main` —
  `merge()` checked out repoB's unrelated `develop` branch,
  squash-merged the agent's work into it (work missing from `main`)
  and left the user's checkout switched to `develop`; when repoB has
  no same-named branch, merge failed with a bogus "Cannot checkout"
  error. Root cause dates from the BUG-30 (audit 7) optimization,
  which never anticipated a repo switch. Fix: capture
  `prev_repo_root = self._wt.repo_root` before the release and use
  `released_branch` only when `prev_repo_root.resolve() == repo.resolve()`; otherwise fall back to `current_branch(repo)`
  (read inside `repo_lock(repo)`, preserving the BUG-30 contract).
  Test: test_bughunt6_cross_repo_release.py (3 tests; 2 failed
  pre-fix + same-repo regression guard).
- Investigated and verified NOT bugs this round (do NOT re-report):
  `_split_rename_tail` " -> "-in-filename ambiguity (git C-quotes ANY
  rename side containing a space, so the unquoted-side first-index
  split is provably unambiguous — verified with real repos);
  `git worktree list --porcelain` path quoting vs `cleanup_orphans`
  resolve-matching (paths emitted verbatim even with `"` in them);
  cherry-pick of an already-applied/empty agent commit
  (`--no-commit` exits 0, nothing staged → SUCCESS, no bogus
  conflict); partial multi-commit cherry-pick conflict → `--abort`
  restores byte-exact incl. staged first pick; typechange (`T`)
  status codes in copy_dirty_state (handled by the symlink/file
  branches); untracked nested-repo stash push (creates a no-op stash
  entry, pop clean — noise only, and only reachable post-6B-1 fix
  failure); `status.showUntrackedFiles=no` (merge --squash refuses
  overwrite, reset --hard never deletes untracked); user on a
  different branch at auto-release time (stash→checkout→pop contract
  explicitly documented in `_do_merge`); `has_uncommitted_changes`
  ignoring git-status returncode and `worktree list` failure in
  `_cleanup_orphans_locked` (no realistic trigger without manual
  gitdir corruption); `ensure_excluded` rev-parse returncode
  (unreachable after discover_repo succeeded); save_baseline_commit
  failure ignored (in-memory GitWorktree keeps the SHA; config never
  read back cross-process by design).
- Verification: 6/6 new tests (2 bug tests each failed pre-fix); 806
  worktree/bughunt/git/baseline/autocommit sorcar tests run in 8
  parallel shards — ZERO failures; `uv run check --full` passes.

### Iteration 6 — Group D (sorcar agents/tools) — COMPLETE: 1 NEW bug found+fixed

All four files read IN FULL this session (sorcar_agent.py, chat_sorcar_agent.py,
useful_tools.py, web_use_tool.py).

- BUG-6D-1 (useful_tools.py `Edit`, ~line 463): `Edit` read the file with
  universal-newline translation (`Path.read_text()` turns `\r\n` into `\n`)
  and wrote the edited text back as-is — a ONE-LINE edit on a CRLF file
  silently rewrote EVERY line ending in the file as LF (massive spurious git
  diffs; breaks CRLF-required files like `.bat`). Reproduced: Edit of
  `b"line one\r\nline two\r\nline three\r\n"` left `b"line one\nLINE 2\nline three\n"`.
  Fix: read with `read_text(newline="")` (no translation); when the raw match
  fails AND the file contains `\r\n` AND old_string carries only LF (what the
  model sees through Read's translated output), retry with CRLF-normalised
  old/new strings (new_string normalised collapse-then-expand,
  `replace("\r\n","\n").replace("\n","\r\n")`, so a partially-CRLF new_string
  is not corrupted into `\r\r\n`); write back with `write_text(..., newline="")`.
  Only the edited region changes; mixed-ending files keep every untouched
  byte. Test: test_bughunt6_edit_crlf.py (8 tests; 5 failed pre-fix, 3
  regression guards: LF-file unchanged, CRLF-verbatim old_string, uniqueness
  count after normalisation).
- Investigated and verified NOT bugs (do not re-report): Edit on lone-`\r`
  (classic-Mac) files now returns "String not found" instead of silently
  destroying the CR endings (safer; no normalisation attempted); Write
  `write_text` newline translation (identity on POSIX; Windows-only concern,
  untestable without mocks); Read translating CRLF→LF for display (standard
  text-mode semantics, now consistent with Edit's normalisation); Read
  binary fallback for UTF-16 text files (extension-based mime guess —
  by-design limit of the attachment whitelist); `_bash_streaming` unbounded
  `chunks` accumulation before truncation (inherent to output capture, same
  as non-streaming `communicate()`); `scroll(amount=huge)` long-but-finite
  loop (model-controlled, bounded, no corruption); `go_to_url("tab:-1")`
  message wording; `update_settings(use_web_browser=True)` only effective
  next run (documented "subsequent sub-sessions"); ChatSorcarAgent
  KeyboardInterrupt persisting `result_summary=""` (overwritten by the task
  runner's cancel outcome); stale `self.printer` fallback in
  `ChatSorcarAgent.run` (callers consistent); `Read(max_lines<=0)` /
  `Bash(max_output_chars<=0)` degenerate inputs (model never emits them;
  framework coerces types).
- Verification: 8/8 new tests pass post-fix; all 219 tests across the 23
  test files importing useful_tools/UsefulTools pass; `uv run check --full`
  clean (also auto-fixed parallel group files: ruff UP037 in
  test_bughunt6_malformed_fields.py, mypy ignore-code in
  test_bughunt6_files_prefix.py — both untracked files from concurrent
  iteration-6 sessions).

### Iteration 6 — Group C (sorcar CLI) — session 1 (in progress)

All four files read IN FULL (cli_repl.py, cli_steering.py, cli_helpers.py,
cli_panel.py). 2 NEW bugs found, reproduced (failing tests first), FIXED:

- BUG-6C-1 (cli_steering.py `SteeringSession.run`): only `sys.stdout` was
  swapped for the lock-guarded `_StdoutProxy`; `sys.stderr` was NOT. Any
  worker/library stderr write (logging default handlers incl.
  `logging.lastResort`, `warnings`, LLM SDK noise) was emitted at the VISIBLE
  cursor parked inside the box body row — pty capture showed
  `ESC[23;5H XSTDERRNOISEX` overprinting the panel body, outside the scroll
  region. Fix: capture `_real_stderr` in `__init__`; `run()` swaps stderr with
  a second `_StdoutProxy` (same lock/box) and restores it in `finally`.
  Test: test_bughunt6_stderr_proxy.py (pty.fork end-to-end; asserts the marker
  is wire-prefixed by ESC 8 i.e. routed via the restore/emit/re-save dance, and
  stderr restored after the session). Failed pre-fix, passes post-fix.
- BUG-6C-2 (cli_repl.py `run_repl`): `_handle_slash` was called bare in the
  REPL loop while task errors were guarded (`_run_one`). `/resume` (lists
  recent chats) over a corrupt `sorcar.db` raised
  `sqlite3.DatabaseError: file is not a database` (from `_get_db`'s PRAGMA —
  no corruption recovery exists) and KILLED the whole interactive session
  with a traceback. Fix: wrap the `_handle_slash` call in try/except
  Exception → `✗ Command failed: {exc}` + re-prompt (mirrors `_run_one`).
  Test: test_bughunt6_slash_survives_error.py (subprocess run_repl, corrupt
  db, `/resume` then `/help` then `exit`; asserts /help output + clean exit).
  Failed pre-fix, passes post-fix.

Investigated and ruled NOT bugs this session (do NOT re-report):

- `/resume <bogus-id>` prints "Resumed chat" without validating existence —
  `resume_chat_by_id` just sets `_chat_id` (attach semantics, same as `-c`);
  borderline UX, not incorrect behavior.
- CliCompleter `_file_cache` cached forever → files created by a task are not
  @-mentionable later; CONSISTENT with the extension (vscode `_file_cache` is
  only wiped on work_dir switch, commands.py:766) — not an inconsistency.
- `/clear` prints "Started a new chat" even for non-Chat agents — unreachable:
  run_repl only ever receives ChatSorcarAgent/WorktreeSorcarAgent (main() in
  worktree_sorcar_agent.py forces use_chat in interactive mode).
- stale `_answer_q` item race (CLI analog of vscode BUG-5E-3): single input
  source; double-Enter in one chunk hits `queue.Full` → dropped; the
  get()-to-clear() window is microseconds and unreachable by a human — the
  realistic multi-viewer path of 5E-3 does not exist in the CLI.
- X10 mouse-report bytes (ESC\[M + 3 raw bytes) could type 3 garbage chars,
  but the box never enables mouse reporting — unreachable without external
  terminal state corruption.
- `_partial_suffix_len`/pending-paste clamp math re-verified; ESC-at-chunk-end
  inside paste consistent with unsplit stripping; CSI param/intermediate/final
  byte ranges correct; `stop()` erase-row math correct for \_BOX_H.
- panel_top/panel_bottom use len() not display width — only ASCII titles ever
  passed (STEER_TITLE/IDLE_TITLE/queued-status/answer-title), prior iterations
  already ruled this OK.
- cli_helpers `_print_recent_chats` timestamp float() — values come from the
  DB REAL column (NaN extra sanitised in 5A-1); `_build_run_kwargs`,
  `_print_result`, `print_outcome` re-checked clean.

REMAINING (next session): run full CLI test sweep (all test_cli\_\* +
test_bughunt*cli*/paste/sigcont/tiny_resize/prompt_markers/interrupt_lock/
c1_controls/ctrlc_prompt_border + the 2 new bughunt6 files), `uv run check --full`, commit. Optionally probe: anything in `_read_line` continuation
cursor math under resize (deemed marginal), Ctrl+C during `_handle_slash`
prints no Goodbye (marginal, not fixing).

### Iteration 6 — Group F (web_server/diff_merge/merge_flow) — session 1 notes

- Read all PROGRESS history for groups F (iter 3/4/5 fixed + verified-not-bug lists).
- Read diff_merge.py IN FULL (827 lines). Candidate leads from this read (NOT yet
  verified — verify with throwaway repos next session before writing tests):
  1. `_load_gitignore_dirs`: negation lines (`!keep/`) are skipped entirely — but a
     PRIOR broad ignore (e.g. `build`) still hides re-included dirs in `_scan_files`;
     also an ignore entry like `foo` matches dirs at ANY depth by name — files named
     in .gitignore (not dirs) also added to skip → harmless for dirs[:] filter but
     `skip` only filters DIRS, never files: a .gitignore'd FILE (e.g. `secret.env`)
     still appears in `_scan_files` output (inconsistency vs docstring "respecting
     .gitignore"). Marginal — check who consumes `_scan_files` (autocomplete file
     list?) before deciding.
  1. `_prepare_merge_view` deleted-TEXT-file placeholder is written with
     `write_text("")` BEFORE `_write_base_copy` — fine. But `current` placeholder for
     deleted text file = empty file; `hunks` were computed from git diff (post_hunks)
     coordinates of base vs EMPTY — check reject-single-hunk on a DELETED file via
     web_server `_reject_hunk_in_file(write_to=target)` writes to target (restores) —
     iter-3 covered deleted-binary; deleted-text per-hunk reject probably covered.
  1. `_file_changed` compares md5 but `pre_file_hashes` may contain hash for file
     that agent DELETED: `(Path/work_dir/fname).read_bytes()` raises OSError →
     `return True` (treated changed) — ok.
  1. `_capture_untracked` with `core.quotepath=false` + `_unquote_git_path` — covered
     iter 3.
  1. `_save_untracked_base` atomic swap: on os.replace(staging→base_dir) failure
     rolls back; but `shutil.copy2` of a file with size check TOCTOU — marginal.
  1. `_parse_diff_hunks`: mode-only changes produce `diff --git` header with NO hunks
     and NO "Binary files" line → fname never enters post_hunks (current_file set but
     no hunk lines) — wait, it DOES enter only via hunks.setdefault on Binary/hunk
     lines; mode-only → absent → invisible. By-design-ish (nothing to review?) but a
     chmod +x by agent is silently unreviewable and merge view says "No changes" if
     that's the only change. CHECK: worktree flow would still commit it. Candidate.
  1. `_diff_header_path` fallback `^diff --git a/.* b/(.*)` — for rename with
     " b/" inside OLD name could mis-split, but --no-renames is used in
     \_parse_diff_hunks... ub_dir copy `saved_base.is_file()` follows symlinks —
     symlink handling fixed iter-4 on reject path only.
- NOT yet read this session: merge_flow.py (904 L), web_server.py merge regions.
  web_server regions of interest (line refs from iter-5 notes): \_WebMergeState +
  reject helpers 396-710, pending-close 2865-3081, ready/replay/submit/merge-action
  3440-3815, \_augment_merge_data ~2401.
- Next session plan: read merge_flow.py + web_server merge regions; probe candidates
  6 (mode-only change invisible ⇒ "No changes") and 1 (gitignored FILES leak into
  \_scan_files) with real repos; then untested surfaces: merge view with agent
  changing file→directory or directory→file; concurrent merges in two TABS same
  work_dir (per-tab data dirs exist since iter-3); reject of file whose parent dir
  agent deleted (target dir missing on restore — does _reject_hunk_in_file mkdir
  parents?); `_handle_submit` upload path. Tests:
  src/kiss/tests/agents/vscode/test_bughunt6_<short>.py, failing-first.
- Git state: clean tree at fd65f0eb (iteration-5 complete, all committed).

### Iter 6 group E (vscode server/commands/task_runner) — session 1 notes (context exhausted early)

- Working tree clean at fd65f0eb. Read IN FULL: commands.py (802 lines).
  Read server.py lines 1-560 only. NOT yet read: server.py 560-1625,
  task_runner.py (1189), merge_flow.py routing regions.
- Existing bughunt tests listed (3/4/5 series in src/kiss/tests/agents/vscode/);
  do not duplicate anything in PROGRESS history above.
- Candidate leads from commands.py + server.py(1-560) — NONE verified yet:
  1. `_cmd_select_model` with a tabId but EMPTY model: model falls back to
     tab.selected_model, then `_record_model_usage(model)` runs → inflates
     usage count without a real user selection? Check what frontend sends
     (test_model_usage_on_select_only.py exists — read it first; may be
     covered/by-design).
  1. `_resolve_user_answer_queue` multi-viewer fallback: candidate_tabs
     iteration order arbitrary; if a viewer tab shares subscriptions with TWO
     owner tabs (stale subscription to an old task id + current), answer could
     route to the WRONG owner's queue. Check whether cleanup of `_subscribers`
     on task end (bughunt-srv2 stale-subscriptions fix) makes this impossible.
  1. `_cmd_complete` with empty query still bumps `_complete_seq_latest[conn]`
     → marks an in-flight request stale with no replacement (ghost text never
     arrives). Probably by design (input cleared) — verify worker behavior.
  1. `_cmd_get_adjacent_task`: falls back to `_tab_chat_views` — fine; but
     `raw_task_id` parse duplicated instead of `_parse_int` (cosmetic only).
  1. `_cmd_save_config`: `self.work_dir = new_work_dir` without `_state_lock`
     (other mutations of work_dir take the lock in `_cmd_set_work_dir`) —
     benign str assignment? Also `cfg` may contain non-dict (cmd.get("config",
     {}) could be a list) → `cfg.get` raises AttributeError → kills transport
     receive loop? CHECK: `_handle_command` callers wrap exceptions? (verify
     how web_server dispatches — if no try/except, malformed saveConfig kills
     the connection: would mirror iter-3 unhashable-type bug class).
  1. `_cmd_record_file_usage`: `path` unvalidated non-str (e.g. dict) →
     `_record_file_usage` may raise → same transport-kill class. Check
     guards in persistence.
  1. `_cmd_user_answer` with ans_tab="" — registry.get("") None → fallback
     scans subscriber sets for "" membership; harmless? verify.
- SESSION 1 (continued) RESULTS — 3 NEW bug classes CONFIRMED + FIXED, tests
  written FIRST (all failed pre-fix, all pass post-fix):
  - **BUG-6E-1** (server.py `_handle_command` + commands.py handlers): the
    ws/uds receive loops wrap the whole `async for` in ONE try — any exception
    escaping `_handle_command` kills the ENTIRE client connection. 15+
    malformed payloads still raised (probe-verified): non-str `tabId` ([1])
    in run/userAnswer/appendUserMessage/selectModel/complete/closeTab/newChat/
    stop/getAdjacentTask/mergeAction (TypeError unhashable);
    `selectModel` non-str model corrupted tab.selected_model/\_default_model
    to a list BEFORE raising sqlite ProgrammingError; `setWorkDir` non-str
    workDir silently corrupted daemon-global self.work_dir+printer.work_dir;
    `recordFileUsage` non-str path (ProgrammingError); `saveConfig` non-dict
    config/apiKeys (AttributeError); `getHistory` non-str query
    (AttributeError) / non-int offset (IntegrityError); `getFiles` non-str
    workDir (unhashable cache key). FIXES: dispatch-boundary coercion in
    `_handle_command` (non-str tabId/workDir/connId → ""), isinstance guards
    in \_cmd_select_model/\_cmd_record_file_usage/\_cmd_save_config/
    \_cmd_get_history (uses `_parse_int` for offset/generation)/\_cmd_get_files
    (prefix)/\_cmd_complete (query/activeFile/activeFileContent).
    Test: test_bughunt6_malformed_fields.py (8 tests; 7 failed pre-fix).
  - **BUG-6E-2** (autocomplete.py `_complete_worker_loop`): the lazily-started
    SINGLETON worker had no try/except — one malformed `complete` query (dict
    with ≥2 keys passes the `len(query)<2` guard, AttributeError in
    `_prefix_match_task`) killed the thread; `_ensure_complete_worker` never
    restarts it (non-None check) → ghost-text autocomplete dead for the
    daemon's WHOLE lifetime, all windows. Also `getFiles` non-str prefix
    crashed the per-request `_do_refresh` thread (TypeError in
    rank_file_suggestions) → file picker never replied. FIX: try/except in
    worker loop (+ logger added to autocomplete.py) + the query/prefix guards
    above. Tests: same file (worker-kill test uses distinct connIds so the
    per-conn staleness check doesn't skip the poisoned item).
  - **BUG-6E-3** (task_runner.py `_run_task`): NO exception handler around
    `_run_task_inner` — any exception raised before its big `try` (non-list
    `attachments` field → TypeError at `for att in raw_attachments` before
    even the model check; non-str prompt → TypeError at `prompt[:200]` in
    the "Task started" log; a git failure re-raised by the
    `_capture_pre_snapshot` guard) killed the worker thread SILENTLY:
    spinner stopped (finally broadcasts running=False) but NO
    result/task_error/task_done event was broadcast or persisted. FIXES:
    (a) `except Exception` in `_run_task` broadcasting a
    `result success=False "Task failed: ..."` event; (b) isinstance-list
    guard on `raw_attachments` (iter-3 only fixed malformed ENTRIES).
    Test: test_bughunt6_silent_task_death.py (2 tests, both failed pre-fix;
    prompt test skips when no models available).
- Verified in passing (not bugs): `_finish_merge` guards empty tab_id;
  attachments malformed ENTRIES skipped (iter-3); `deleteFrequentTask`
  non-str task guarded; `_cmd_worktree_action`/`_cmd_autocommit_action`
  wrap/route safely; `_translate_webview_command` non-str type safe;
  `_resolve_user_answer_queue` multi-owner mis-route requires stale
  subscriptions already fixed in bughunt-srv2.
- REMAINING for this iteration: run impacted vscode test sweep + `uv run check --full`, commit. (Candidate leads list below was session-1 planning;
  items 1-7 are now resolved by the fixes/probes above except: model-usage
  inflation on empty selectModel (verified harmless — `_record_model_usage`
  only runs when a model string exists; empty model returns early), complete
  empty-query staleness bump (by design: clears pending ghost).)
- OLD SESSION-1 PLAN (superseded): (a) check how `_handle_command` exceptions are handled by
  web_server dispatch (UDS + WS) — if unprotected, malformed-payload bugs in
  6/7 above are real (write tests per bughunt3_dispatch_malformed.py recipe,
  which may already cover some — READ IT FIRST); (b) read server.py 560-1625
  (esp. \_stop_task, \_close_tab, \_new_chat, \_replay_session,
  \_teardown_tab_resources, \_generate_commit_message/\_generate_followup_async,
  \_subscribe_chat_viewers, \_get_adjacent_task); (c) read task_runner.py fully
  (pending_user_messages drain, _ask_user_question, attachments, autocommit,
  result persistence); (d) write failing tests
  src/kiss/tests/agents/vscode/test_bughunt6_<short>.py (harness recipe:
  VSCodeServer(); override printer.broadcast; stub SorcarAgent.__mro__[1].run;
  stub \_server_module.generate_followup_text; tearDown clears
  \_RunningAgentState.running_agent_states); (e) fix, verify, run impacted
  tests, `uv run check --full`, commit.

## Iteration 6 — group A (sorcar persistence/running_agent_state)

- Scope: persistence.py (2147 lines, read in full) + running_agent_state.py
  (read in full; pure state container, no logic to break — consistent with
  prior verdicts). ~50 candidate surfaces audited.
- **BUG 6A-1 (FIXED)** — `persistence.py:_set_task_favorite` (~line 1100):
  parsed stored `extra` with lenient default `json.loads` (accepts bare NaN).
  A legacy corrupt row (e.g. `'{"subagent": {...}, "cost": NaN, ...}'`,
  written pre-iter-5) is uniformly classified NOT-subagent (strict
  `_parse_extra_dict` and the SQL `json_valid` predicate both reject it) and
  is VISIBLE in the history sidebar. Starring it merged `is_favorite` into
  the leniently-recovered dict — including the never-effective `subagent`
  key — and re-encoded via `_dumps_extra` as VALID JSON, so the row flipped
  to subagent classification and permanently VANISHED from
  `_load_history`/`_search_history`/`_list_recent_chats`.
  Fix: parse stored extra with strict `_parse_extra_dict` first; when
  strict-invalid, recover metadata leniently (the sidebar displays extra via
  lenient json.loads in server.py) but `extra_obj.pop("subagent", None)` so
  the rewrite can never change the row's classification.
  Test: test_bughunt6_favorite_corrupt_extra.py (5 tests; 3 failed pre-fix:
  star-stays-visible, keeps-chat-in-recent-chats, unstar-stays-visible;
  2 controls passed: metadata-preserved, valid-subagent-stays-hidden).
  Regression: 116 impacted persistence/favorite/history tests pass (2
  parallel shards 46+70); mypy/pyright 0 errors.
- Ruled out (verified NOT bugs — do NOT re-chase): LIKE prefilter false
  negatives (all writers use json.dumps default separators); `_load_history`
  `LIMIT -1 OFFSET` semantics (verified in sqlite); RWLock nesting/deadlock
  paths (no caller flushes events while holding read lock);
  `_write_event_batch` seq-cache vs `_delete_task` interleavings (all under
  write lock); `_event_writer` shutdown sentinel; chat-context cache
  generation race; `_record_file_usage`/`_record_frequent_task` eviction
  ties; `_get_history_entry` negative idx (no production callers);
  `_save_task_extra` lenient `is_favorite` preservation (payload is
  caller-controlled; cannot flip classification).
- Note: the htmlhint errors printed by `uv run check --full` for
  src/kiss/agents/vscode/media/chat.html (`{{NONCE_ATTR}}` template
  placeholders) are non-blocking BY DESIGN — package.json `lint:html` is
  `htmlhint ... || true` (commit e57db0dd); chat.html is committed/clean and
  vscode-group territory, left untouched.
- Iteration 6 group A result: 1 NEW bug (6A-1) ⇒ loop continues.
