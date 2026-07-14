# PROGRESS — Horizontally resizable docked history sidebar (remote webapp)

## Task

The agent history panel (`#sidebar`, docked LEFT on desktop-wide ≥900px remote
webapp windows) MUST be horizontally resizable. Reproduce the issue with
end-to-end tests first (failing), then fix. Dev model `claude-fable-5`; review
model `gpt-5.6-sol`.

## What was done

1. Internet research (10/10 sources): pointer-capture drag pattern
   (`setPointerCapture` on `pointerdown`, `touch-action: none` on the handle,
   `pointercancel` handled like `pointerup`), the W3C ARIA Window Splitter
   pattern (`role=separator`, `aria-orientation=vertical`,
   `aria-valuenow/min/max`, ArrowLeft/ArrowRight keyboard resize), the
   single-CSS-custom-property rule to keep sidebar width and content margin in
   sync, localStorage persistence conventions, and jsdom caveats (no
   `PointerEvent` constructor / element pointer-capture methods → tests
   dispatch `MouseEvent`s with pointer event types and stub
   `setPointerCapture`/`releasePointerCapture`; implementation guards with
   `typeof` + try/catch).

1. Tests FIRST — `src/kiss/agents/vscode/test/remoteSidebarResize.test.js`
   (NEW, 11 jsdom e2e tests running real `chat.html` + `panelCopy.js` +
   `main.js`): resizer existence + ARIA attributes; drag 300→420 sets
   `--sidebar-w: 420px` with pointer capture/release; clamping to 220px/600px;
   localStorage `kiss-sidebar-w` persistence on pointerup and restore on load;
   garbage/out-of-range sanitization; ArrowLeft/ArrowRight ±16px keyboard
   resize; dblclick reset to 300px + key removal; mobile inertness; non-remote
   (VS Code webview) isolation; pointercancel ends the drag; `sidebar-resizing`
   body class during drag with the dock kept open. The suite FAILED before the
   implementation (`#sidebar-resizer handle must exist in chat.html`),
   reproducing the issue. `src/kiss/tests/agents/vscode/test_codex_mobile_layout.py`
   extended to 40 tests (resizer HTML/CSS/JS-wiring assertions, CONTROL_IDS,
   `var(--sidebar-w, 300px)` in the dock block).

1. Implementation — `media/chat.html`: `#sidebar-resizer` separator element as
   last child of `#sidebar`. `media/remote-codex.css`: dock block now uses
   `width: var(--sidebar-w, 300px)` and `margin-left: var(--sidebar-w, 300px)`
   (one variable drives both, no desync); resizer hidden by default,
   `display:block` 6px `cursor: col-resize; touch-action: none` strip in the
   ≥900px block with hover/active/focus-visible highlight;
   `body.remote-chat.sidebar-resizing { user-select: none; }`. `media/main.js`:
   resizer wiring after the remote-desktop matchMedia block, guarded by
   `remote-chat` + `remote-desktop` classes — clamp 220–600px, pointer-capture
   drag, localStorage persist/restore (`/^\d+$/` validated), ArrowLeft/Right
   ±16px, dblclick reset. `package.json`: test registered in the chain.

1. Review — `gpt-5.6-sol` independent pass over the full diff: guard ordering,
   IIFE scope, box-sizing, closeSidebar(force) compatibility, ARIA correctness,
   eslint/stylelint clean. No missed wirings or bugs found.

1. Verification — 11/11 resize tests green; 10/10 remoteDesktopSidebar dock
   tests green (no regression); 40/40 pytest layout tests green; npm lint
   green; `uv run check --full` green. Full JS chain (~100 test files) run to
   completion: every executed test file reported `0 failed`; one PRE-EXISTING
   environment hang was found in `installFailureNoCompleteNotification.test.js`
   (its sandbox empties `PATH`, `xcode-select -p` then fails and the compiled
   installer polls up to 10 minutes for git) — verified it hangs identically on
   a clean tree with the feature stashed, so it is unrelated to this change;
   the remaining chain segments after it (including the two remote sidebar
   suites) were run separately and exited 0. Live validation: real
   `RemoteAccessServer` over HTTPS in a desktop-width Chromium window —
   docked sidebar with the resizer present, and the served page/main.js/CSS
   verified end-to-end to carry the resizer element, wiring, and
   `var(--sidebar-w)` rules.

______________________________________________________________________

# PROGRESS — Local tree-sitter `code_graph` tool — Sessions 1–4 (current task)

## Task

Implement Graphify-inspired features #1, #3, and #5 in KISS Sorcar with minimal
coupling: a per-worktree tree-sitter knowledge graph, the “deny message is the
answer” query-before-grep pattern, and SHA256-incremental Git-hook freshness.
Write end-to-end tests first with 100% branch coverage; use `claude-fable-5` for
development and `gpt-5.6-sol` for an independent review/debug pass.

## What was done

1. Researched 10 current primary/technical sources: Graphify README,
   `ARCHITECTURE.md`, and `SKILL.md`; tree-sitter and
   tree-sitter-language-pack documentation/package pages; Aider’s repository-map
   implementation; Claude Code’s `PreToolUse` hook contract; and Git’s
   `post-commit` hook documentation. The resulting design follows Graphify’s
   top-three seed / three-hop BFS query, confidence-tagged edges, SHA256 cache,
   and detached post-commit update patterns.

1. Added `tree-sitter>=0.24.0` and
   `tree-sitter-language-pack>=1.10.0` to `pyproject.toml` and refreshed
   `uv.lock`. Parser acquisition is optional and failure-safe, so an unavailable
   grammar cannot break Sorcar startup or graph builds.

1. Wrote the end-to-end feature suite first in
   `src/kiss/tests/agents/sorcar/test_code_graph.py`. It builds real Python,
   JavaScript, TypeScript/TSX, Go, Rust, Java, Ruby, C, and C++ fixture projects;
   executes real Git commits/hooks and real Bash commands; verifies persistence,
   incremental add/change/delete behavior, graph query/path/explain formats,
   local/streaming/Docker interception, CLI behavior, and graceful failure paths.

1. Implemented the standalone
   `src/kiss/agents/sorcar/code_graph.py` with only stdlib plus the optional
   tree-sitter language pack. Each worktree stores versioned, atomically-written
   state in `.kiss/code_graph/{graph.json,cache.json}` and adds that generated
   directory to Git’s local exclude file. Nodes represent files, modules,
   classes, interfaces, structs, traits, functions, and methods. Edges represent
   `defines`, `contains`, `imports`, and `calls`, tagged `EXTRACTED` for direct
   same-file/source relationships or `INFERRED` for unique cross-file call
   resolution; ambiguous relationships are omitted rather than invented.

1. Implemented graph operations:

   - `query(question)`: term-ranked top-three seeds and a deterministic
     three-hop BFS rendered as compact `NODE`/`EDGE` lines under a character
     budget.
   - `path(a, b)`: shortest undirected connection while preserving each stored
     edge’s direction, relation, and confidence.
   - `explain(name)`: source location, kind, degree, and directional neighbors.
   - Agent actions: `build`, `query`, `path`, `explain`, `install_hook`, and
     `uninstall_hook`; querying auto-builds when no graph exists.

1. Kept production coupling to two guarded seams: `SorcarAgent._get_tools()`
   lazily adds `make_code_graph_tool(...)`, and local plus Docker Bash wrappers
   lazily consult `grep_hint(...)`. An exact graph identifier hit returns the
   inline graph answer without spawning `grep`/`rg`:

   ```python
   hint = grep_hint(command, self.work_dir) or ""
   if hint:
       return hint  # “the deny message IS the answer”
   ```

   Literal text and regex searches that the graph cannot answer continue to the
   real command. This exact-label guard fixed a review-discovered false positive
   where prose containing `app` incorrectly matched the `Application` node and
   suppressed a legitimate grep.

1. Implemented idempotent post-commit hook install/uninstall without overwriting
   existing shell hooks. The hook runs a fully detached incremental update via
   the current Python executable. A PID lock deduplicates concurrent hook jobs
   and reclaims dead/stale locks; a full worktree hash pass avoids unsafe shell
   expansion, including filenames containing spaces/newlines, while reparsing
   only changed files.

1. The `gpt-5.6-sol` independent review added regression tests that exposed and
   fixed: duplicate method-label scope collapse, incorrect reverse-path arrows,
   empty self-paths, two-hop instead of three-hop queries, dropped recursive
   calls, missing `grep -e`/`--regexp` parsing, whitespace-unsafe hook arguments,
   non-shell/corrupt hook handling, generated graph artifacts in Git status, a
   missed Docker Bash seam, stale cache-schema reuse, and non-atomic persistence.
   Definition/caller records now use stable per-file integer identities and
   lexical-scope resolution, and persisted state carries a cache schema version.

1. Final verification:

   - `103 passed`; `code_graph.py`: **628 statements, 250 branches, 100% branch
     coverage**.
   - Impacted Bash/Sorcar regressions: **83 passed, 2 deselected**.
   - `uv run check --full`: dependency sync, API generation, compileall, Ruff,
     mypy, Pyright, and Markdown formatting all passed (0 errors/warnings from
     static analysis).

______________________________________________________________________

# PROGRESS — Codex-mobile remote webapp restyle — Sessions 4–6

## Task

Restyle the standalone remote webapp (served by `RemoteAccessServer` in
`src/kiss/agents/vscode/web_server.py`) to look like the OpenAI Codex
mobile / ChatGPT app. Constraints: every existing control and DOM id
must remain identical (visual/layout change only), the change must be
scoped to the remote view only (`body.remote-chat`; shared `chat.html`
and `main.css` used by the VS Code webview stay untouched), e2e tests
written first, claude-fable-5 for development and gpt-5.6-sol for an
independent review pass, full suite run 8-way parallel via
`run_parallel` with failure classification, and final validation by
running the remote webview and taking screenshots.

## What was done

1. **Web research (10/10 sources)** on the Codex app / ChatGPT mobile
   design language; extracted tokens: `#0d0d0d` page, `#ececec` text,
   `#212121` composer card at 28px radius with inset white edge shadow,
   36px circular composer controls, white circular send button, 22px
   light user bubble right-aligned at max 75% width, 768px centered
   content column, pill tab chips, `#171717` drawers/panels.

1. **TDD tests first**: added
   `src/kiss/tests/agents/vscode/test_codex_mobile_layout.py` (16
   tests, red before implementation): cache-busted
   `remote-codex.css` link ordered after `main.css`, no unresolved
   placeholders, control-id parity (explicit inventory ∪ every id
   parsed from `chat.html`), `body.remote-chat` retained, no Codex CSS
   leakage into shared `chat.html`/`main.css`, every selector in the
   new stylesheet scoped under `body.remote-chat` (including inside
   `@media`), design-token assertions (palette, composer, circular
   send, pill tabs, task-panel bubble), a real-`RemoteAccessServer`
   HTTPS e2e test (production `start_async`/`_process_request`
   lifecycle, MIME + byte-for-byte body checks), and `_media_url`
   cache-bust hashing.

1. **Implementation**: new file
   `src/kiss/agents/vscode/media/remote-codex.css` (all selectors
   prefixed `body.remote-chat`); `_build_html()` in `web_server.py` now
   prepends a cache-busted `<link href="/media/remote-codex.css?v=...">`
   to the remote-only `HEAD_STYLE` placeholder (the VS Code extension
   passes `HEAD_STYLE=''`, so the webview is provably unaffected).
   `chat.html` and `main.css` untouched; packaging includes the new css
   automatically (no media allowlist exists).

1. **gpt-5.6-sol review pass** found and fixed: the HTTP test
   originally used a look-alike server (replaced with the production
   `RemoteAccessServer`), the control-parity test was hardened to also
   include every id parsed from `chat.html`, and the pinned task panel
   lacked the Codex light-bubble treatment (test added first, then CSS:
   `#ececec` bg, `#0d0d0d` text, 22px radius, right-aligned 75%).

1. **Full suite 8-way parallel run** (6138 tests, splits of ~767 via
   `run_parallel`): splits 0/3/6 green. Five individual failures in
   splits 2/4/5/7 (`test_bughunt4_merge_replay_on_reconnect`,
   `test_chat_agent_state_registration` ×2, `test_talk_tool`,
   `test_cli_history_click_resumes_live_stream`) each re-run 3× in
   isolation — all passed → classified as pre-existing parallel-load
   timing flakes (test-environment issues, not project bugs; the CSS
   change cannot affect WS replay or threading).

1. **split_1 SEGFAULT root-caused and fixed (pre-existing TEST BUG)**:
   the `orphan-task-sweep` daemon thread was executing `db.execute`
   against a redirected/deleted temp DB while
   `test_web_server_bugs.py`'s `IsolatedAsyncioTestCase.asyncTearDown`
   closed the connection and `rmtree`'d the tmpdir (the conftest
   join hook runs only after the pytest call phase, i.e. too late for
   asyncTearDown-based classes). Fix in `asyncTearDown`:

   ```python
   sweep = self.server._vscode_server._orphan_sweep_thread
   if sweep is not None and sweep.is_alive():
       await asyncio.to_thread(sweep.join, 30)
   ```

   Full split_1 re-run: **764 passed, 4 skipped — GREEN**. Net full
   suite result: 6138 tests, 0 project bugs from the change.

1. **Visual validation**: launched the real `RemoteAccessServer`
   locally (hermetic temp persistence, TLS disabled for localhost
   automation), loaded the page in Chromium: accessibility tree showed
   every control present and functional (tabs, status, composer, model
   pill, upload/tricks/voice, send, history filters, frequent/tricks/
   settings panels, API key fields, auth). Screenshots confirmed the
   Codex look: near-black `#0d0d0d` page, pill "new chat" tab chip,
   centered welcome, bottom rounded `#212121` composer card, pill model
   button, circular icon buttons, white circular send button, and the
   `#171717` history drawer with rounded search field.

1. **Final checks**: fixed two mypy `[misc]` errors in the new test
   (raise-from on `state["error"]` typed `object` — narrowed with
   `isinstance(..., BaseException)`); `uv run check --full` fully
   green; feature + bug-fix suites green (23 passed).

______________________________________________________________________

# PROGRESS — Run all tests in parallel, diagnose & fix failures — Session 3

## Task

Same task as Sessions 1–2: run all tests split by test-method count into
(cores − 2 = 8) parallel splits via `run_parallel`, report causes of
failing tests, classify each as project bug vs test bug, fix accordingly.

## Session 3 (this run)

1. Collected 6122 tests (`uv run pytest --collect-only -q --no-cov`,
   73 slow deselected), split node IDs round-robin into 8 files
   (766/766/765×6), ran all 8 splits concurrently via `run_parallel`
   (`uv run pytest --no-cov -q -p no:cacheprovider @./tmp/split_N.txt`).
1. Results verified directly from the 8 result logs: **ALL 8 SPLITS
   GREEN — 6074 passed, 48 skipped, 0 failed, 0 errors** (760+6, 762+4,
   758+7, 757+8, 756+9, 760+5, 758+7, 763+2; totals add up to all 6122
   collected tests). 445 subtests also passed. Only non-fatal warnings
   (supabase/pydantic deprecations, one asyncio `__del__`
   PytestUnraisableExceptionWarning) were observed.
1. Nothing to fix this session: the bugs found and fixed in Sessions
   1–2 (persistence.py `_get_db()` TOCTOU project bug, commit c64acc16;
   SIGINT SIG_IGN-inheritance test bugs in
   test_install_script_npm_ignore_scripts.py and
   test_bughunt9_c_sigterm_during_cleanup.py, commit 6726ee6e) held —
   no recurrence of the disk-I/O poisoning or the SIGINT hang under the
   backgrounded `run_parallel` environment.
1. No source or test code modified; temp split/result files cleaned up.

______________________________________________________________________

# PROGRESS — Run all tests in parallel, diagnose & fix failures — Session 2

## Task

Run all tests split by test-method count into (cores − 2 = 8) parallel
splits via `run_parallel`, report causes of failing tests, classify
each failure as a project bug vs a test bug, and fix accordingly.
(Session 1's fixes — the persistence.py `_get_db()` TOCTOU project-bug
fix and the SIGINT `preexec_fn` test-bug fix in
test_install_script_npm_ignore_scripts.py — were already committed as
c64acc16.)

## Session 2

1. Collected 6114 tests, split round-robin into 8 files (765/764 each),
   ran all 8 splits concurrently via `run_parallel`
   (`uv run pytest --no-cov -q -p no:cacheprovider @./tmp/split_N.txt`).
1. Results: splits 1–7 fully green (761+757+756+755+759+757+762 passed,
   plus skips/subtests). Split 0: exactly ONE failure — a HANG at test
   #410, `test_bughunt9_c_sigterm_during_cleanup.py::TestSigtermDuringSigintCleanup::test_sigint_shutdown_arms_sigterm_guard`
   (`pytest-timeout` dump: main thread parked forever in
   `KqueueSelector.select()` inside `srv.start()` →
   `asyncio.run(self._serve_async())`; server had printed its
   listening banner; only idle helper threads alive). All 6113 other
   tests passed. Confirmed the previous 5 disk-I/O persistence
   failures are gone (fixed in c64acc16).
1. Root cause of the hang — **TEST BUG**, the same SIG_IGN inheritance
   family as Session 1's install-script flake but in a different test:
   `run_parallel` workers launch pytest as a background job of a
   non-job-control shell (`sh -c 'pytest … &'`), which per POSIX starts
   the job with SIGINT set to `SIG_IGN`. CPython then never installs
   its default `KeyboardInterrupt` handler (verified:
   `signal.getsignal(SIGINT) == SIG_IGN` inside a backgrounded pytest;
   a backgrounded python survives a self-delivered SIGINT). The test's
   interrupter thread `os.kill(pid, SIGINT)` is therefore a silent
   no-op, `srv.start()` never unwinds, and the test hangs. Not a
   web_server.py bug: `RemoteAccessServer` deliberately relies on
   Python's default SIGINT→KeyboardInterrupt path, which is correct in
   a real terminal (test passes instantly in the foreground and passed
   in the earlier serial re-run).
1. Fix (test code only): in
   `test_bughunt9_c_sigterm_during_cleanup.py`, after snapshotting the
   old handlers, install `signal.signal(signal.SIGINT, signal.default_int_handler)` for the duration of the test (restored
   in the existing `finally`), with a comment documenting the POSIX
   background-job SIG_IGN semantics. This gives the test the same
   signal environment as a real terminal regardless of how pytest was
   launched.
1. Verification: test passes in foreground and 4/4 iterations as a
   `sh -c '… &'` background job (previously hung/failed 100% when
   backgrounded); the first-410-tests-of-split-0 run that reproduced
   the hang now passes (406 passed, 4 skipped); full split_0 re-run
   green (759 passed, 6 skipped); `uv run check --full` all checks
   pass.

## FINAL RESULT

- 6114 collected tests (non-slow), 8 parallel splits: **6113 passed /
  skipped cleanly; 1 failure**.
- The 1 failure was an order-independent, environment-dependent hang:
  a **test bug** (SIGINT test assumed the pytest process never inherits
  SIGINT=SIG_IGN from a backgrounding shell). Fixed in the test.
- No project-code bugs found this session (Session 1's TOCTOU
  persistence bug was the only project bug, already fixed).

______________________________________________________________________

# PROGRESS — Run all tests in parallel, diagnose & fix failures — Session 1

## Task

Run all 6113 tests split by test-method count into (cores − 2 = 8)
parallel splits via `run_parallel`, report causes of failing tests,
classify each failure as a project bug vs a test bug, and fix
accordingly.

## What was done

1. Collected 6113 tests (`uv run pytest --collect-only -q`; addopts add
   `-m 'not slow'`, 73 slow tests deselected); split node IDs
   round-robin into 8 files (~764 each); ran all 8 splits concurrently
   with `run_parallel`, each worker running
   `uv run pytest --no-cov -q -p no:cacheprovider @./tmp/split_N.txt`.
1. Results: splits 0, 2–7 ALL PASSED. Split 1: 5 FAILED, all with
   `sqlite3.OperationalError: disk I/O error` raised from
   `src/kiss/agents/sorcar/persistence.py` against the shared
   per-process test DB:
   - test_bughunt_srv2_stale_subscriptions.py::TestReplayOtherChatDropsOldStream::test_replay_back_to_running_chat_restores_stream
   - test_non_git_workdir.py::TestNonGitCommandsDoNotCrash::test_resume_session_unknown_chat
   - test_orphan_task_recovery.py::TestOrphanTaskRecovery::test_sweep_ignores_rows_created_after_cutoff
   - test_simplify_server_cmds_regr.py::test_get_input_history_conn_id_stamping
   - test_wave3_runner_merge_bugs.py::test_b2_warm_agent_second_run_still_attributes_own_metrics
     All 5 passed when re-run serially → order/race-dependent poisoning,
     not a deterministic logic bug. One additional flake surfaced during
     subset reruns: test_install_script_npm_ignore_scripts.py::
     test_run_with_heartbeat_double_sigint_aborts ("first SIGINT trap
     never fired" — empty log after the 10 s `_wait_for_log_text`
     default timeout).
1. Root cause of the 5 disk-I/O failures (confirmed with targeted
   SQLite experiments + a full-stack race harness): **PROJECT BUG** — a
   TOCTOU race in `_get_db()` (persistence.py ~line 910). The stale
   side-file cleanup `if not _DB_PATH.exists(): unlink _DB_PATH+'-wal'/'-shm'` re-read the `_DB_PATH` **global** between
   the existence check and the unlink. Tests that redirect `_DB_PATH`
   to a scratch dir, delete the scratch DB, and restore the shared path
   (e.g. test_persistence_db_deleted_file.py) let a background thread
   (kiss-event-writer / orphan-sweep) observe "missing" for the SCRATCH
   file while computing unlink targets from the freshly RESTORED shared
   path — unlinking the live shared DB's `-shm`. SQLite then fails
   every NEW connection to that DB with a permanent `disk I/O error`
   while any old connection keeps the unlinked `-shm` mapped, poisoning
   later vscode-server tests in the same process.
1. Fix (project code): in `_get_db()` both the existence check and the
   unlink targets are now derived from the single `current_path`
   snapshot (`os.path.exists(current_path)` /
   `os.unlink(current_path + suffix)` in try/except OSError), never a
   re-read of `_DB_PATH`; detailed comment documents the TOCTOU.
1. Regression test added (e2e, no mocks):
   `src/kiss/tests/agents/sorcar/test_persistence_wal_unlink_toctou.py`
   — hammers the redirect/delete/restore pattern with background
   `_get_db()` threads for 5 s, watches the shared `-shm` inode, then
   health-checks a fresh thread. Verified to FAIL on pre-fix code
   ("shared -shm was unlinked") and PASS with the fix.
1. Sigint flake — TRUE root cause found (the earlier "timeout too
   tight" theory was wrong; the test failed again with a 30 s timeout
   and a completely empty harness log even in a solo run): the failure
   reproduces 100% whenever pytest is launched as a **background job of
   a non-job-control shell** (`sh -c 'pytest … &'` — exactly how
   `run_parallel` workers and `nohup … &` invoke it). POSIX requires
   such a shell to start background jobs with SIGINT set to `SIG_IGN`;
   the ignored disposition is inherited across fork/exec down to the
   harness bash, and bash cannot trap signals ignored on entry
   ("Signals ignored upon entry to the shell cannot be trapped or
   reset") — so install.sh's `trap handle_interrupt INT` silently
   became a no-op and `handle_interrupt` never ran (empty log,
   `sleep 30` unaffected). Confirmed empirically: identical harness
   loop fails 36/36 when backgrounded (`getsignal(SIGINT) == SIG_IGN`), passes 13/13 in the foreground. Classification:
   **TEST BUG** — install.sh is correct in a real terminal (SIGINT
   default); the test assumed the pytest process never inherits
   SIGINT=SIG_IGN. Fix: `_reset_signal_dispositions()` helper passed as
   `preexec_fn=` to both harness `subprocess.Popen` calls in
   test_install_script_npm_ignore_scripts.py, resetting SIGINT/SIGTERM
   to `SIG_DFL` between fork and exec so the harness always gets the
   same signal environment as a real terminal. (The earlier 30 s
   `_wait_for_log_text` timeouts were kept — harmless robustness.)
1. Verification: toctou regression test + test_persistence_db_deleted_file
   (5 passed); the 5 previously failing tests + sigint test file +
   persistence neighborhood (32 passed); full split_1 re-run green
   (758/759, only the sigint test failing pre-final-fix); after the
   preexec_fn fix the two sigint tests pass 5/5 iterations when pytest
   runs as a background job (previously 0/5) and the whole sigint test
   file passes in the foreground (8 passed); `uv run check --full` all
   checks pass.

______________________________________________________________________

# PROGRESS — Resolve stranded-branch merge conflict (kiss/wt-1783912825-c143d801)

## Task

The framework's auto-merge of worktree branch `kiss/wt-1783912825-c143d801`
(which carried the previous merge-conflict-resolution session's 5 commits:
merge 984137bf of the gmail/googlechat KISS_HOME fixes, the Episode 1
PROGRESS restore, marketing mdformat, and the PROGRESS log) reported
"Merge conflict detected. Resolve manually" with `git merge --squash`
instructions. Resolve it and land the branch.

## What was done

1. Investigated: the branch held 6 commits on top of merge-base 886d06b1;
   main had meanwhile advanced by one commit (f56e7a4e, Episode 1
   pre-flight). `git merge-tree` showed exactly ONE conflicted file:
   `marketing/agent-markets-itself/episode-01/LAUNCH_CHECKLIST.md` — main's
   f56e7a4e rewrote its pre-flight section (verified quickstart, launch
   slot, 404 blocker) while the branch's 049df754 mdformatted the older
   version of the same lines.
1. Ran `git merge --no-ff kiss/wt-1783912825-c143d801` (merge commit
   7d7c7fac) and resolved the single conflict by keeping main's newer
   pre-flight CONTENT (verified items, slot, blocker) and applying the
   branch's mdformat indentation style; re-ran `mdformat` on the file.
1. Post-merge review: the ort merge of PROGRESS.md had dropped main's
   "Show HN launch pre-flight" section (from f56e7a4e, absent on the
   branch); restored it verbatim + mdformat (commit 900fc669).
1. `uv run check --full` failed only on the pre-existing unformatted
   `HAND_REWRITE_GUIDE.md` (new in f56e7a4e, never mdformatted);
   formatted it (commit aac21482). Full check now passes.
1. Verified merged-in tests still pass (test_gmail_gchat_isolation.py,
   test_gmail_agent.py, test_tlon_config_isolation.py — 39 passed).
1. Verified the auto-stashes hold nothing unique (every differing blob in
   stash@{0} is the stale dc59fed9 version of PROGRESS.md / docs/api.md /
   index.html.md); left them for the user to drop.
1. Deleted the fully-merged branch `kiss/wt-1783912825-c143d801`
   (tip cb8a0d0b is an ancestor of HEAD).

______________________________________________________________________

# PROGRESS — Resolve stranded-branch merge conflict (kiss/wt-1783911737-192f8ce6) (previous task)

## Task

The framework's auto-merge of worktree branch `kiss/wt-1783911737-192f8ce6`
(which carried commit c8bd1176, the gmail/googlechat KISS_HOME isolation
fixes) reported "Merge conflict detected. Resolve manually" with cherry-pick
instructions. Resolve it and land the branch.

## What was done

1. Investigated: the stranded branch held exactly one commit (c8bd1176) on
   top of dc59fed9; main had advanced to 886d06b1 (Episode 1 marketing
   artifacts). Both sides edited `PROGRESS.md`. `git merge-tree --write-tree`
   showed the merge is now clean (the original conflict came from the
   framework's cherry-pick path against a dirty/stashed main).
1. Merged the branch into the current worktree branch (based on main):
   merge commit 984137bf — brought in c8bd1176 (gmail/googlechat/\_channel
   work_dir KISS_HOME fixes + `test_gmail_gchat_isolation.py` + website
   mdformat fixes) with zero conflicts.
1. Post-merge review of `PROGRESS.md` showed the merged branch's version
   (written before the Episode 1 task) had dropped the Episode 1 section
   that main added; restored it verbatim (commit 2834c7f1) and mdformatted.
1. `uv run check --full` initially failed on 6 pre-existing unformatted
   marketing markdown files from main's 886d06b1; mdformatted them
   (commit 049df754). Full check now passes.
1. Verified merged-in tests: `test_gmail_gchat_isolation.py`,
   `test_gmail_agent.py`, `test_tlon_config_isolation.py` — 39 passed.
1. Verified both auto-stashes (`kiss: auto-stash before merge`) contain no
   unique content: every blob in stash@{0} matches either HEAD or the
   pre-merge commits (its PROGRESS.md/api.md/index.html.md copies are the
   older dc59fed9 versions). The user's uncommitted changes were the same
   c8bd1176 content that is now merged. Stashes left in place for the user
   to drop (`git stash drop`) after confirming.
1. Advised: the stranded branch `kiss/wt-1783911737-192f8ce6` can now be
   deleted (`git branch -D kiss/wt-1783911737-192f8ce6`) since its commit
   is contained in the merge.

______________________________________________________________________

# PROGRESS — Audit third-party agent dirs for KISS_HOME test isolation (previous task)

## Task

Audit the ~20 third-party agent integrations sharing the
`Path.home()`-based directory pattern of `slack_agent.py` and apply the
KISS_HOME/tmp-dir test-isolation fix to prevent cross-process races in
parallel test runs (follow-up to the slack token fix in a4e6c58b).

## Audit results

All 23 module-level `_*_DIR` globals in `src/kiss/agents/third_party_agents/`
were audited:

1. **Already isolated (20 agents, no change needed):** bluebubbles, discord,
   feishu, imessage, irc, line, matrix, mattermost, msteams, nextcloud_talk,
   nostr, phone_control, signal, sms, synology_chat, telegram, tlon, twitch,
   whatsapp, zalo. Each module's `_*_DIR` is used only to construct a
   `ChannelConfig`, whose `.path` property already resolves lazily via
   `_kiss_home()` (honours `$KISS_HOME`, which `src/kiss/tests/conftest.py`
   sets to a fresh temp dir per pytest process).
1. **Gap — gmail_agent.py:** `_GMAIL_DIR = Path.home()/...` was used directly
   by `_token_path()`/`_credentials_path()`, bypassing `KISS_HOME`. Tests
   (`test_gmail_agent.py`, `test_bughunt_gmail_whatsapp.py`) backed up,
   overwrote, and restored the REAL user `token.json`/`credentials.json` —
   the same cross-process race that broke slack, plus a risk of clobbering
   real Gmail credentials on a crash.
1. **Gap — googlechat_agent.py:** same direct-path pattern for
   `token.json`/`credentials.json`/`service_account.json` (latent — no test
   currently writes them).
1. **Gap (minor) — \_channel_agent_utils.py:** `ChannelRunner` and
   `channel_main` defaulted `work_dir` to `Path.home()/".kiss"/"channel_work"`
   instead of `_kiss_home()/"channel_work"`.
1. **Intentionally unchanged:** `slack_sorcar_poller.py` /
   `slack_channel_sorcar_poller.py` `STATE_DIR` constants (live production
   cron state; tests already monkeypatch `mod.LOCK_FILE` etc., and the
   code paths exercised by tests never touch the real state files).

## Fixes applied

1. `gmail_agent.py`: replaced the `_GMAIL_DIR` module global with a lazy
   `_gmail_dir()` helper returning
   `_kiss_home() / "third_party_agents" / "gmail"`; `_token_path()` and
   `_credentials_path()` now call it.
1. `googlechat_agent.py`: same pattern — `_gchat_dir()` helper; the three
   path helpers now resolve lazily.
1. `_channel_agent_utils.py`: both `work_dir` defaults now use
   `_kiss_home() / "channel_work"`.
1. New regression test `src/kiss/tests/agents/channels/test_gmail_gchat_isolation.py`
   (mirrors `test_tlon_config_isolation.py`): proves gmail/googlechat paths
   and the `ChannelRunner` default `work_dir` follow `$KISS_HOME` changes
   made after import, and that a token written under one `KISS_HOME` never
   leaks into another.

## Verification

- New isolation tests: 4 passed.
- Full channels test dir: 620 passed, 32 skipped.
- Real `~/.kiss/third_party_agents/gmail/` untouched (empty before and after).
- `uv run check --full` clean.

______________________________________________________________________

# PROGRESS — Ship llms.txt + pure-Markdown docs on kisssorcar.github.io (previous task)

## Task

Ship **llms.txt** at the kisssorcar.github.io root plus pure-Markdown docs so
LLMs and coding assistants can index and recommend KISS Sorcar, and update the
website.

## Session 1 (research + planning)

1. Read `SORCAR.md`; explored repo. The live site is a separate repo,
   `https://github.com/kisssorcar/kisssorcar.github.io`, mirrored locally at
   `website/kisssorcar.github.io/`. `gh` is authenticated with push permission.
1. Completed the mandatory 10-site web research on the llms.txt convention
   (llmstxt.org spec, Mintlify/Anthropic/Vite practice, `.md` twin pages,
   `llms-full.txt`, `.well-known/` mirror, GitHub Pages static serving).
1. Read source-of-truth content: `README.md`, `API.md`,
   `src/kiss/SAMPLE_TASKS.md`, `src/kiss/INJECTIONS.md`, `src/kiss/TIPS.md`.

## Session 2 (implementation + deploy) — COMPLETE

1. Re-cloned the live site repo to a scratch dir (previous clone was lost with
   the old worktree).
1. Authored new files (all pure Markdown / plain text, no HTML):
   - `llms.txt` — spec-compliant: H1 `# KISS Sorcar`, `>` blockquote summary,
     "Key facts" detail block, then `## Docs` (10 links), `## Papers` (4),
     `## Source & Install` (5), `## Optional` (4) — all `- [name](url): notes`
     lines with absolute URLs. Validated structure with a small Python check.
   - `docs/` — 10 pages: `index.md`, `overview.md` (incl. comparison table +
     citation), `installation.md`, `cli.md` (options + mcp subcommand),
     `api.md` (condensed from API.md), `models.md` (530 models / 9 provider
     categories table), `messaging-agents.md` (23 agents + Govee),
     `sample-tasks.md` (all 12 SAMPLE_TASKS), `prompt-tricks.md` (all 12
     INJECTIONS tricks), `tips.md` (all TIPS content).
   - `index.html.md` — Markdown twin of the homepage.
   - `llms-full.txt` — generated by concatenating the 10 docs pages with
     `<!-- Source: url -->` markers.
   - `.well-known/llms.txt` — copy of `llms.txt`.
   - `robots.txt` (allow all, references llms.txt + sitemap), `sitemap.xml`
     (homepage + all md/txt URLs), `.nojekyll`.
1. Edited `index.html` head:
   `<link rel="alternate" type="text/markdown" href=".../index.html.md">` and
   `<link rel="llms-txt" type="text/plain" href=".../llms.txt">`; footer
   gained `Docs` (docs/index.md) and `llms.txt` links.
1. Verified locally (python3 -m http.server): all 16 URLs returned 200.
1. Committed ("Ship llms.txt + pure-Markdown docs for LLM/coding-assistant
   indexing", b11a7a2) and pushed to `kisssorcar/kisssorcar.github.io` main.
1. Verified LIVE after Pages deploy: 200 for `/llms.txt`, `/llms-full.txt`,
   `/robots.txt`, `/sitemap.xml`, `/index.html.md`, `/.well-known/llms.txt`,
   `/docs/index.md`; rendered llms.txt in browser; homepage contains the two
   llms.txt references.
1. Mirrored everything into the main repo artifact dir
   `website/kisssorcar.github.io/` via rsync (also picked up previously
   missing live assets: og-card.png, sorcar-main.gif, KISS-Sorcar-UI.png,
   swedefend.pdf, cleverest_plus.pdf, .gitignore) and rewrote
   `website/README.md` to document the shipped llms.txt work.
1. Cleaned up `./tmp/site` and research notes; staged website changes in git.

## Possible follow-ups (not required)

- Submit https://kisssorcar.github.io/llms.txt to llms.txt directories
  (llmstxt.site, directory.llmstxt.cloud). — DONE, see next section.

# Task: Submit llms.txt to directories (2026-07)

Submitted https://kisssorcar.github.io/llms.txt to both canonical llms.txt
directories (per https://llmstxt.org/#directories):

1. **llmstxt.site** — submitted via https://llmstxt.site/submit with
   Product Name "KISS Sorcar", Website https://kisssorcar.github.io/,
   llms.txt + llms-full.txt URLs, contact ksen@berkeley.edu, and notes
   (Apache-2.0 framework, `pipx install kiss-agent-framework`, docs at
   /docs/index.md, arXiv 2604.23822). Confirmed via redirect to
   https://llmstxt.site/thankyou. Listing appears after their moderation /
   site refresh.
1. **directory.llmstxt.cloud** — submitted via their Tally form
   (https://tally.so/r/wAydjB): name "KISS Sorcar", llms.txt URL,
   Category "AI", email ksen@berkeley.edu for approval notification,
   adoption note. Confirmed "Form submitted / Thanks for completing this
   form!". Pending curation-team approval; notification will go to
   ksen@berkeley.edu.

Both live URLs (https://kisssorcar.github.io/llms.txt and /llms-full.txt)
were re-verified in the browser before and after submitting.

# Task: Episode 1 of "The Agent That Markets Itself" — Show HN launch post (2026-07)

Drafted and "recorded" (written-serial format) Episode 1: KISS Sorcar wrote
its own Show HN launch post, then the draft was reviewed and refined for the
48-hour launch window.

## Session 1 (research)

Completed the mandatory 10-site web research on Show HN norms 2026:
official Show HN guidelines; dang's tips (item 22336638 — **edited
2026-03-28: HN submission text must be hand-written, no LLM text at all**);
syften.com May-2026 guide (first-comment anatomy, kill-triggers, 9am–12pm ET
weekday); lucasfcosta.com (concrete titles, link the repo, cut 30%);
HN Algolia survey of 223 agent-framework Show HNs (winners: Mastra 442 pts,
AnythingLLM 368, Superset 96/90 comments; anti-example Orcbot 4 pts);
forkoff.xyz agent-native GTM (spec-driven agent workflow); indiehackers KTool
(second-chance pool, reply to everything).

## Session 2 (artifacts) — COMPLETE

Created `marketing/agent-markets-itself/` (git-tracked artifact directory):

- `README.md` — series index + ground rules (always disclose agent work;
  HN text hand-written by the founder; no upvote solicitation).
- `episode-01/SPEC.md` — the 300–800-word brief given to the agent: goal
  (titles + URL + first comment), hard constraints (factual language,
  verifiable numbers, ≤450 words, quickstart + limitations + disclosure),
  3 positive examples (Mastra/Superset/AnythingLLM), 3 disqualifiers,
  and the dang-2026 post-draft rule (agent draft = episode artifact only;
  final HN text must be hand-rewritten by the human).
- `episode-01/SHOW_HN_DRAFT.md` — the agent's refined v2 package:
  5 concrete title candidates (recommended: "Show HN: KISS Sorcar –
  open-source agent framework, ~2,850 LoC core, 530 models"), submission URL
  = the GitHub repo, and a 418-word first comment (intro + credibility line,
  backstory, 5 mechanism-anchored bullets, `pipx install kiss-agent-framework` quickstart, honest limitations incl. single
  maintainer + no benchmark yet, meta disclosure of the agent-drafted
  experiment, links block, specific feedback asks on security model and
  small-core claim), plus 9 annotated v1→v2 edits.
- `episode-01/REVIEW.md` — reproduces rejected draft v1 ("does everything"
  title + buzzword bullets = 2 disqualifiers) and maps all 11 rulings to
  research sources; records the 3 spec deltas (word cap, number-or-mechanism
  rule, specific-ask requirement).
- `episode-01/EPISODE.md` — the episode itself in 5 acts, including the
  twist: while researching, the agent discovered HN's 2026 rule banning
  LLM-written submission text, so the episode's resolution is
  "agent draft = published artifact, human hand-rewrites the HN text,
  the HN comment discloses the experiment" — transparency becomes the hook.
  Includes scorecard and video shot-list note.
- `episode-01/LAUNCH_CHECKLIST.md` — 48-hour runbook: T-7 pre-flight
  (hand-rewrite, README GIF, clean-machine quickstart test, personal HN
  account with profile email), T-0 submission (Tue–Thu 9am–12pm ET, repo
  URL, no upvote solicitation), T+48h rules (reply to every comment
  personally, never AI-written replies, second-chance pool), staggered
  Reddit/newsletter wave after traction, and a metrics table that feeds
  Episode 2.

All artifacts committed to git. Remaining human-only steps: hand-rewrite the
first comment and execute the launch window per the checklist.

# Task: Show HN launch pre-flight — hand-rewrite prep, quickstart test, slot (2026-07-12)

Follow-up to Episode 1 of "The Agent That Markets Itself" (see
marketing/agent-markets-itself/episode-01/). Three deliverables:

1. **Hand-rewrite handled correctly.** Per HN's 2026-03-28 rule, any
   agent-written text is LLM-generated and disallowed as submission text —
   so the agent CANNOT produce the final comment. Instead, created
   `marketing/agent-markets-itself/episode-01/HAND_REWRITE_GUIDE.md`: an
   8-beat content checklist (intro/credibility, backstory, 5
   mechanism-anchored bullets, pipx quickstart, honest limitations,
   meta-disclosure, links, feedback asks) plus a verified-facts table and a
   pre-post self-check — deliberately containing NO prose to copy. The
   founder writes the actual comment by hand from this guide.
1. **Clean-machine quickstart test — PASSED.** Docker as clean machine:
   - `python:3.13-slim`: `pip install pipx && pipx install kiss-agent-framework` → 2026.7.18 installs; `sorcar --help` works;
     **34 seconds end-to-end** (≤5-min target met).
   - `python:3.14-slim`: passes (3.13+ claim holds upward).
   - `python:3.12-slim`: fails gracefully ("No matching distribution
     found") — validates the "Python 3.13+ only" limitation claim.
   - Caveat: pipx PATH warning; `pipx ensurepath` fixes.
1. **Slot picked:** primary **Wed 2026-07-15, 9:30am ET**; fallbacks Thu
   2026-07-16 and Tue 2026-07-21 (Tue 7/14 too soon for human pre-flight).

Also: link pre-flight all 200 (repo, arXiv 2604.23822, docs/index.md,
llms.txt); found one **BLOCKER** — the public repo path
https://github.com/ksenxx/kiss_ai/tree/main/marketing returns 404; the
episode dir must be published publicly (push marketing/ to the public repo
or mirror to kisssorcar.github.io) before T-0, since the comment's meta
paragraph links to it. Updated LAUNCH_CHECKLIST.md (checked off verified
items, added slot + blocker), EPISODE.md (new Act 6: pre-flight
verification + scorecard rows), and the series README (guide + slot).
Remaining human-only steps: hand-rewrite, HN-account karma warm-up, README
hero GIF check, clear the public-repo blocker, submit Wed 9:30am ET.

# PROGRESS — Issue #43: explicit UTF-8 encoding for all text file I/O

## Task

Fix https://github.com/ksenxx/kiss_ai/issues/43: `read_text()`,
`write_text()`, and text-mode `open()` calls without `encoding=` use the
platform locale encoding, so on non-UTF-8 locales (Windows cp1252,
`LC_ALL=C`) the agent mis-reads UTF-8 files or raises
`UnicodeEncodeError` on write. Fix by passing `encoding="utf-8"`
explicitly everywhere (same precedent as issue #33 for subprocess pipes).

## Changes (session of 2026-07)

1. Added `encoding="utf-8"` to **57 `Path.read_text()`/`write_text()`
   call sites** across 22 non-test files under `src/kiss`, keeping any
   existing `newline=` argument (`useful_tools.py` Write/Edit round-trip
   semantics unchanged; the `UnicodeDecodeError` binary-detection
   try/except around `Read` kept intact since strict UTF-8 decoding
   still raises it):
   - core: `base.py` (SYSTEM_PROMPT), `relentless_agent.py`,
     `models/model_info.py` (3 sites)
   - agents/sorcar: `cli_helpers.py` (`-f` task file), `cli_repl.py`,
     `useful_tools.py` (Read/Write/Edit tools, 4 sites)
   - agents/third_party_agents: `_channel_agent_utils.py`,
     `googlechat_agent.py`, `gmail_agent.py`, `slack_sorcar_poller.py`,
     `slack_channel_sorcar_poller.py`
   - agents/vscode: `diff_merge.py`, `web_server.py` (13 sites),
     `tips.py`, `tricks.py`, `vscode_config.py`
   - benchmarks: `generate_dashboard.py`, `terminal_bench/run.py`,
     `deepscholar_bench/run.py`
   - scripts: `check.py`, `update_models.py`, `generate_api_docs.py`
1. Added `encoding="utf-8"` to **15 text-mode `open()`/`Path.open()`
   sites** (incl. lock files for lint cleanliness):
   `slack_sorcar_poller.py`, `slack_channel_sorcar_poller.py`,
   `autocomplete.py`, `diff_merge.py`, `merge_flow.py`,
   `voice_wake.py`, `vscode_config.py` (the `save_config` lock `with`
   was reformatted to a parenthesized multi-item `with`),
   `web_server.py` (3 sites), `generate_dashboard.py`,
   `redundancy_analyzer.py`.
1. `docker/docker_tools.py`: the generated Python code strings executed
   *inside* containers now use `open(path, encoding='utf-8').read()` and
   `open(path, 'w', encoding='utf-8').write(...)` so in-container edits
   are locale-safe too (f-string quoting verified).
1. **New e2e regression test**
   `src/kiss/tests/agents/sorcar/test_utf8_encoding.py` (no mocks):
   each test runs a child interpreter with `LC_ALL=C`, `LANG=C`,
   `PYTHONUTF8=0` (ASCII default encoding) and asserts
   (a) `UsefulTools.Write/Read/Edit` round-trips "café ☕ — ünïcode"
   byte-exactly, (b) the `sorcar -f` task-file path
   (`cli_helpers._resolve_task`) loads non-ASCII task text intact,
   (c) `kiss.core.base.SYSTEM_PROMPT` imports non-empty under C locale.
   Verified the Write/Read/Edit test FAILS on the pre-fix tree
   (git stash → 1 failed) and passes after.

## Verification

- `grep -rnE '\.(read_text|write_text)\(' src/kiss --include='*.py' | grep -v encoding= | grep -v /tests/ | grep -v test_` → only the two
  multi-line `write_text(` calls whose `encoding="utf-8"` sits on a
  later line (diff_merge.py:977, web_server.py:4716) — zero real
  violations.
- `uv run pytest src/kiss/tests/agents/sorcar/test_utf8_encoding.py -v`
  → 3 passed.
- `uv run check --full` → ✅ All checks passed (ruff, mypy, pyright,
  mdformat).
- Impacted suites: sorcar file-tool tests (76 passed),
  `src/kiss/tests/core` (1114 passed, 1 skipped),
  `src/kiss/tests/vscode` (153 passed),
  `src/kiss/tests/scripts` + `benchmarks` + `docker` (104 passed).

## Review pass (gpt-5.6-sol)

### Findings and fixes

- Re-derived file-I/O sites independently with lexical searches and an AST
  audit. Found **two missed locale-dependent text opens**: the
  `logging.basicConfig(filename=...)` calls in
  `slack_sorcar_poller.py::_setup_logging` and
  `slack_channel_sorcar_poller.py::_setup_logging`. Added
  `encoding="utf-8"` to both. Verified each poller in a fresh child process
  under `LC_ALL=C`, `LANG=C`, `PYTHONUTF8=0`, and
  `PYTHONCOERCECLOCALE=0`; each wrote a non-ASCII log message that decoded
  byte-exactly as UTF-8.
- Found no other missed non-test text I/O after auditing all
  `read_text`/`write_text`, builtin `open`/`Path.open`, tempfile text modes,
  `TextIOWrapper`, configparser/csv/toml file APIs, and logging file handlers.
  The remaining unencoded `open`-named AST hits are binary opens or unrelated
  APIs (`os.open`, `wave.open`, `webbrowser.open`, and `self.open`). Confirmed
  no binary-mode open received an `encoding` argument.
- Found no introduced behavior bugs. An AST comparison of every changed Python
  module against `HEAD`, after removing only added UTF-8 keywords, found no
  semantic differences except the intentional generated-code string changes
  in `docker_tools.py`. `UsefulTools` retains `newline=""`, strict decoding,
  and the reachable `UnicodeDecodeError` fallback. Lock-file descriptor and
  `fcntl` semantics are unchanged. Executed the exact shell command generated
  by `DockerTools.Edit` against a path containing a space and non-ASCII
  replacement text; shell quoting, embedded Python syntax, and UTF-8 output
  all succeeded.
- Reviewed the new regression test: it uses real child interpreters and no
  mocks; its replacement environment overrides `PYTHONUTF8` and omits (thus
  clears) inherited `PYTHONIOENCODING`. A detached clean `HEAD` worktree
  failed the test as expected with ASCII `UnicodeDecodeError`, while the
  patched tree passes all three cases. The temporary worktree was removed and
  the original working tree restored unchanged.

### Verification

- `uv run pytest src/kiss/tests/agents/sorcar/test_utf8_encoding.py -v`:
  **3 passed**.
- `uv run pytest src/kiss/tests/agents/sorcar -x -q`: reached **1325 passed**
  before one unrelated credential-enabled live OpenRouter test failed because
  its remote provider returned `None` (not an encoding-code regression).
- Targeted changed-path suites (`test_useful_tools.py`, `test_cli_repl.py`,
  `test_bughunt9_cli_files.py`, `test_bughunt_matrix_pollers.py`, and
  `test_kiss_web_launch.py`): **70 passed, 2 deselected**.
- Both poller UTF-8 logging C-locale smoke tests passed; the exact generated
  `DockerTools.Edit` command smoke test passed.
- `uv run check --full`: **all checks passed** (dependency sync, API docs,
  compileall, ruff, mypy, pyright, and mdformat).

# PROGRESS — Fix adjacent-task overscroll ("side scrolling of chats") in the chat webview

## Task

When a chat tab's chat-id has multiple tasks, overscrolling at the
top/bottom of `#output` must load the previous/next task
(Cursor-style adjacent-task navigation). This "worked several days
ago" and regressed. Reproduce with e2e tests first, then fix.
Dev model: claude-fable-5; independent review: gpt-5.6-sol.

## Root cause (confirmed by bisecting behavior against `df31a0e8~1`)

Regression introduced by commit `df31a0e8` ("show prompt/system-prompt
panels immediately on task submit"), which added
`_broadcast_early_prompts` to `src/kiss/agents/vscode/task_runner.py`.
It broadcasts optimistic `system_prompt`/`prompt` events with
`taskId: ''` (EMPTY STRING) and `early: true` at submit time, BEFORE
the task's DB row exists.

Poisoning chain in `src/kiss/agents/vscode/media/main.js`:

1. `setTaskText` resets `currentTaskId = null` and calls
   `resetAdjacentState()` → both scroll anchors
   (`oldestLoadedTaskId`/`newestLoadedTaskId`) become `null`.
1. The early prompt event falls into the message-switch DEFAULT-case
   taskId-adoption block, whose guard only rejected
   `undefined`/`null` — the empty string passed → `currentTaskId = ''`
   and (both anchors being null) the anchors were seeded to `''`.
1. The real taskId (e.g. `'123'`) streamed moments later updates
   `currentTaskId`, but the anchor re-seed only fired when BOTH
   anchors were still `null` — they were `''`, so they stayed `''`
   forever.
1. Overscroll posted `getAdjacentTask {taskId: ''}`; backend
   `commands.py` `_opt_str('')` → `None` → persistence returns no
   task → server broadcast an EMPTY `adjacent_task_events` →
   `renderAdjacentTask` latched `noPrevTask = true` PERMANENTLY.
   Adjacent scrolling dead for the tab.

Secondary bug: `task_events` replays carrying a valid `task_id` but an
empty/missing `task` title (server.py resume-race replay path) never
synced `currentTaskId`/anchors/`currentTaskName`, so overscroll stayed
blocked after such a replay.

## Changes

All in `src/kiss/agents/vscode/media/main.js`:

1. DEFAULT-case adoption guard: added `ev.taskId !== ''` so early
   empty-string taskIds are never adopted; anchor re-seed now also
   fires when anchors are `''` (defensive), not only `null`.
1. `accumulateOverscroll()`: early-return when the anchor taskId is
   `undefined`/`null`/`''` so `getAdjacentTask` is never posted with
   an unknown row id (prevents the empty-reply noPrev/noNext latch).
1. `task_events` active-tab path: new `else if` branch — when
   `ev.task` is empty but `ev.task_id` is present, still set
   `currentTaskId = ev.task_id`, derive `currentTaskName` from the tab
   title (fallback `'Task'`) only when it is empty, and call
   `resetAdjacentState()`.

New permanent e2e test `src/kiss/agents/vscode/test/adjacentTaskScroll.test.js`
(jsdom harness on the real `chat.html` + `panelCopy.js` + `main.js`,
same pattern as `sideScrollWhileRunning.test.js`), wired into the
`package.json` `test` chain. Covers:

- REGRESSION: full live-task lifecycle with early `taskId:''` prompts →
  overscroll must post `getAdjacentTask` with the REAL task id.
- Never post `getAdjacentTask` with an empty taskId.
- `task_events` with `task_id` but no `task` title still enables
  overscroll.
- History-load → wheel-up prev request → `adjacent_task_events` reply
  renders `.adjacent-task` → chained prev uses the new oldest id.
- Wheel-down at bottom requests `next`.
- Touch pull-down at top requests `prev`.

## Verification

- New test FAILS on pre-fix code (verified via
  `git stash push media/main.js`: posted `taskId:""`), PASSES after fix.
- Full vscode test chain (96 `node test/*.js` files) run in 8 parallel
  batches: all passed except `installFailureNoCompleteNotification.test.js`,
  which is unrelated (exercises `DependencyInstaller` in a PATH-empty
  sandbox; on this machine it triggers the Xcode CLT install poll loop
  and a `uv sync` pyproject fixture parse error — fails identically on
  the untouched base commit; no files it touches were modified).
- `npm run typecheck` (tsc): clean. `npm run lint:ts` (eslint on
  `src/**/*.ts` + `media/**/*.js`): clean. `npm run compile`: clean.
- No Python files changed.

## Review pass (gpt-5.6-sol)

Independent review conducted on gpt-5.6-sol (via `set_model`):

- Audited ALL taskId/anchor write sites in `media/main.js`
  (lines 582-584 declarations, 847 `restoreTab` — sets
  `currentTaskId` from `tab.currentTaskId` and calls
  `resetAdjacentState()` last so anchors seed correctly; 2154-2155
  `resetAdjacentState`; 2228/2232 `renderAdjacentTask` anchor
  updates — guarded against ''/null taskIds; 4835 background-tab
  `teTab.currentTaskId`; 4922 active `task_events` with title;
  4937 the NEW no-title branch; 5009 `setTaskText` clear;
  5651/5655/5666-5667 default-case adoption + re-seed): consistent,
  no site can reintroduce a '' anchor.
- The default-case adoption change cannot regress the
  result/usage_info misroute guard: early events have `taskId: ''`
  and `ev.taskId` being falsy also skips the guard's drop branch, so
  behavior for early events is unchanged (they render, as intended by
  df31a0e8's promptPanelEarlyReplace design — that test still passes).
- Wheel and touch handlers both funnel through `accumulateOverscroll`
  (single choke point), so the ''-taskId protection covers both input
  paths; `oldestLoadedTaskId != null` guards in the handlers remain
  correct because `''` now never reaches the anchors.
- Backend wiring verified intact end-to-end: main.js posts
  `getAdjacentTask {tabId, taskId, direction}` → SorcarSidebarView.ts
  FORWARDED_COMMANDS forwards exactly those fields → commands.py
  `_cmd_get_adjacent_task` resolves chat_id from the
  `_RunningAgentState` registry with `_tab_chat_views` fallback and
  parses taskId via `_opt_str` (non-empty string else None) →
  server.py `_get_adjacent_task` → persistence
  `_get_adjacent_task_by_chat_id` ((timestamp, rowid) total order,
  sub-agent rows filtered) → broadcast `adjacent_task_events` stamped
  with tabId → extension forwards (not in the merge_data drop list) →
  main.js drops if `ev.tabId !== activeTabId`, else renders.
  web_server.py routes browser commands through the same
  `VSCodeServer._handle_command`, so the remote webapp inherits the fix.
- The new `task_events` `else if` branch sits in the active-tab
  section (after the `teTabId !== activeTabId` early-break) and only
  derives `currentTaskName` when it is empty, so it cannot clobber a
  real title; server.py's resume-race replay (line ~1428, `task: ""`
  with valid task_id) now correctly re-seeds the anchors.
- Sub-agent tabs remain excluded (isSubagentTab guards in both wheel
  and touch handlers untouched).
- Verified the new test fails pre-fix (git stash → posts `taskId:""`)
  and passes post-fix; existing `sideScrollWhileRunning.test.js` (4/4)
  and `promptPanelEarlyReplace.test.js` still pass.
- Minor pre-existing (not introduced) nits, no action needed: types.ts
  declares `taskId: number | null` for getAdjacentTask while the
  webview actually sends string row ids (same looseness as
  deleteTask/setFavorite; backend accepts strings), and the
  adjacent_task_events type omits the `task_id` field it carries at
  runtime. Both predate this change and are cast through
  `as unknown as AgentCommand` at the single forwarding site.

No missed wirings or introduced bugs found.

______________________________________________________________________

# PROGRESS — Long-running `code_graph` cost/speed/accuracy benchmark

## Task

Evaluate the new Graphify-inspired `code_graph` feature against the previous
Sorcar behavior on a large project. Measure cost, speed, and accuracy; research
modern agent-evaluation methodology extensively; use `claude-fable-5` for the
benchmark/development work and `gpt-5.6-sol` for an independent review that
checks the harness, production wiring, trajectories, and implementation bugs.

## Method

1. Researched 10 current sources before designing the evaluation: Anthropic's
   agent-eval and tool-design guidance, Databricks' multi-million-line codebase
   benchmark, SWE-bench, CodexGraph, Graphify's README/architecture, Mem0's
   memory evaluation, and deterministic gold-fact grading references. Applied
   their core recommendations: same harness in both arms, sealed Git history,
   isolated environments, deterministic grading rather than an LLM judge,
   retained trajectories, cost/tokens/latency reported together, and repeated
   trials.

1. Created a controlled feature ablation over two isolated 222 MB copies of the
   KISS repository. Both copies excluded `.git`, `.kiss`, `.venv`, `tmp`,
   `node_modules`, caches, and benchmark output. Baseline used the real
   `UsefulTools.Bash`/`Read` path with no graph; treatment used those same tools
   plus the production `code_graph` tool and a prebuilt graph. The graph had
   18,008 nodes and 47,480 edges from 1,036 source files and took 2.17 seconds
   and $0 of LLM spend to build.

1. Added `benchmarks/code_graph_eval/`:

   - `tasks.json`: 14 deterministic code-comprehension tasks (11 structure and
     3 general) with reviewed gold-fact checklists.
   - `runner.py`: reproducible corpus preparation, arm/trial selection,
     resumable incremental results, real `KISSAgent` execution using
     `claude-fable-5`, full trajectories, wall time, steps, tokens, model cost,
     graph interception/call counters, and arm-purity assertions.
   - `analyze.py`: matrix completeness, duplicate/missing-cell, grade, error,
     and baseline-contamination audits; aggregate statistics; and 50,000-sample
     deterministic task-cluster bootstrap intervals.
   - `results.json`, `transcripts/`, `metadata.json`, and `RESULTS.md`: the 84
     final runs, all raw trajectories, setup metadata, and full report.
   - `results_pre_review.json`/`transcripts_pre_review/` and
     `results_query_fix.json`/`transcripts_query_fix/`: archived diagnostic
     stages so the original regression and review fixes remain reproducible.

1. Ran 14 tasks × 2 arms × 3 trials (84 final independent agent runs). The
   final matrix cost $5.4448175. All 42 baseline trajectories had zero graph
   hints; treatment generated 25 grep interceptions and no explicit graph-tool
   calls. Both arms had zero run errors and 100% deterministic gold-fact
   coverage.

## Results

Per task-trial mean:

| Metric | Baseline | `code_graph` | Treatment delta |
|---|---:|---:|---:|
| Gold-fact accuracy | 100.0% | 100.0% | tie / ceiling |
| Tokens | 6,413 | 7,188 | +12.1% |
| Model cost | $0.06428 | $0.06536 | +1.7% |
| Wall time | 18.81 s | 20.84 s | +10.8% |
| Agent steps | 3.05 | 3.24 | +6.3% |

Task-cluster bootstrap 95% intervals all crossed zero: tokens −4.9% to +33.5%,
cost −8.9% to +13.6%, time −5.9% to +33.6%, and steps −7.5% to +24.5%.
Therefore this workload **does not verify an overall improvement** and also
does not establish a statistically stable regression. Accuracy is inconclusive
because both arms reached the suite's 100% ceiling. Structure tasks—the subset
where the graph should help—used 15.0% more treatment tokens and took 15.9%
longer, although individual caller tasks had meaningful wins.

## Independent `gpt-5.6-sol` review and fixes

1. Verified corpus identity and arm isolation, absence of Git-history leakage,
   baseline purity, identical prompts/model/budget/step limits, real treatment
   tool registration, model token/cost accounting, every saved answer, and
   representative paired trajectories. A stricter word-boundary audit found no
   false-positive grades. The reviewer added the omitted valid `Skill.source`
   value `user`; all scores remained 100% because both answers had included it.
1. Fixed benchmark instrumentation that conflated `[code_graph]` tool labels
   with grep interception and missed textual `code_graph(action=...)` model
   calls. Reprocessed the archived raw trajectories with separate exact-marker
   and model-message counters.
1. Found a production relevance bug: a three-hop query globally sorted every
   discovered node and rendered all nodes before all edges, so high-degree hubs
   could consume the 1,500-character hint budget before the exact seed, direct
   callers, or relationships appeared. Added a failing tight-budget regression
   first, then ranked by BFS distance and interleaved each ring's nodes with
   edges back toward earlier rings. The released-feature one-trial treatment
   dropped from 15,500 to 9,349 tokens/task after this fix.
1. Found repeated deny-message loops: repeated verification greps received the
   same graph answer forever. Added tests and changed local/streaming/Docker Bash
   wiring so each distinct graph answer intercepts once per tool instance; a
   repeat falls through to real grep/Docker. The reviewed trial-1 treatment
   dropped further to 7,801 tokens/task.
1. Found a second relevance edge case: substring matches such as
   `AnotherUsefulToolsTest` displaced the exact `UsefulTools` seed. Added a
   failing test, exact-label scoring, and stable seed-rank rendering. One-letter
   identifiers still bypass interception as intended.
1. Reviewed missed wiring explicitly: one-shot behavior covers normal,
   streaming, and Docker Bash; treatment receives the tool while baseline does
   not; graph failures still degrade to normal Bash; the final benchmark saw
   the production interception path; and no new startup or fallback bug was
   found.

## Verification

- `uv run python benchmarks/code_graph_eval/analyze.py`: audited 84 unique
  cells, 14 tasks, trials 1–3, zero errors/contamination, $5.4448175 total final
  matrix spend, and reproduced every report aggregate and bootstrap interval.
- Focused feature suite: **106 passed**; `code_graph.py`: **656 statements, 266
  branches, 100% statement and branch coverage**.
- Impacted Bash/UsefulTools/Sorcar suites: **78 passed, 2 deselected**.
- Focused Ruff, mypy, and Pyright: clean (0 errors/warnings).
- `uv run check --full`: all dependency, API generation, compileall, Ruff,
  mypy, Pyright, and Markdown checks passed.
- The final report honestly recommends keeping interception opt-in, returning
  compact operation-shaped caller/callee/definition answers, suppressing test
  and hub noise, avoiding unused tool-schema overhead, and next running 20–50
  harder tasks on a repository of at least one million lines with randomized
  arm order.

______________________________________________________________________

# PROGRESS — Desktop width defaults: 1/4-screen history panel, 90% chat column

## Task

In the remote web app on desktop browsers: (1) the history panel must occupy
1/4 of the browser screen by default; (2) the chat panels and the fixed task
panel (and composer) must occupy 90% of the chat webview. Reproduce with
jsdom-based e2e tests first, then fix. Dev model `claude-fable-5`; review
model `gpt-5.6-sol`.

## What was done

1. Internet research (10/10 sources): `clamp(min, 25vw, max)` is the canonical
   quarter-screen sidebar default (CSS-Tricks/MDN, baseline widely available);
   `25vw` = 25% of the viewport; `max-width: 90%` resolves against the
   containing block (the chat webview column); ChatGPT width-fix userscripts
   use exactly `maxWidth: 90%` for wide chat columns; jsdom does no layout so
   tests assert parsed CSS rules + runtime seeding.

1. Tests FIRST — NEW `src/kiss/agents/vscode/test/remoteDesktopWidths.test.js`
   (9 jsdom e2e tests, real `chat.html` + `panelCopy.js` + `main.js`):
   CSS `var(--sidebar-w, clamp(220px, 25vw, 600px))` on BOTH the docked
   sidebar and `#app` margin; 90% max-width for `#output` children,
   `#task-panel`, `#input-container`; JS default seeding
   (`aria-valuenow` = 256 at jsdom innerWidth 1024), keyboard baseline
   256→272, dblclick reset to 256, persisted width still wins, clamp range
   unchanged, VS Code webview isolation. FAILED before the fix (found
   `width: var(--sidebar-w, 300px)` and 75%/85%/768px caps), reproducing the
   issue. Pytest `test_codex_mobile_layout.py` extended to 43 tests
   (quarter-screen fallback, 90% column rules, `window.innerWidth * 0.25`
   in main.js).

1. Implementation — `media/remote-codex.css`: dock block fallback
   `300px` → `clamp(220px, 25vw, 600px)` in both `width` and `margin-left`
   (single `--sidebar-w` variable still drives both, no desync);
   `#task-panel` 75% → 90%; `#input-container` 768px → 90%;
   `#output > *:not(#welcome)` `min(85%, 768px)` → 90%. `media/main.js`:
   `SB_DEF = 300` replaced by `sidebarDefaultW()` =
   `clamp(220, round(window.innerWidth * 0.25), 600)`; used for the initial
   `aria-valuenow`, the keyboard baseline, and the dblclick reset. Existing
   `remoteSidebarResize.test.js` expectations updated (300 → 256 baseline).
   `package.json`: `remoteDesktopWidths.test.js` appended to the test chain.

1. Review — `gpt-5.6-sol` independent pass: searched for stale
   300px/75%/85%/768px remnants (none in rules; only comments/tests), checked
   `* { box-sizing: border-box }` (90% of padded parents correct), confirmed
   percentage max-widths resolve against `#output`/`#input-area` content
   boxes (the chat webview column), verified jsdom parses the nested
   `var(...clamp(...))` syntax, mobile (\<600px) rules untouched, VS Code
   webview isolation intact, localStorage persistence flow unchanged. One
   stale comment in remoteSidebarResize.test.js fixed.

1. Verification — 9/9 new widths tests, 11/11 resize tests, 10/10 dock tests,
   adjacentTaskScroll, `npm run lint` exit 0, 43/43 pytest layout tests,
   `uv run check --full` ✅ (twice, incl. after review fixes).
