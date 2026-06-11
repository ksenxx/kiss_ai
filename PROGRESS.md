# Task: bug-hunt in src/kiss/agents/vscode (TS/JS scope) — COMPLETE

## Steps done

1. Read every scoped file fully: src/*.ts, src/*.js, media/main.js, media/panelCopy.js,
   media/demo.js (≈12.4k lines), cross-checked read-only against web_server.py, server.py,
   commands.py, diff_merge.py, json_printer.py, task_runner.py, model_info.py.
1. Found REAL bug B1: MergeManager.\_doOpenMerge `isNewFile = every(h.oc===0)` mis-classifies
   existing files with insertion-only hunks (e.g. appends) as new ⇒ Reject File/Rest DELETED the
   user's pre-existing file. Python writes an empty base copy only for brand-new files, so the
   correct check is base-file size (mirrors the binary branch's hasBase).
1. Wrote test/bughunt_isNewFile.test.js (real compiled out/MergeManager.js + vscode stub via
   Module.\_resolveFilename, real filesystem). Verified it FAILS on pre-fix code (restored
   c09ab07b version) and PASSES with the fix. 3 cases: reject-keeps-existing-file,
   reject-still-deletes-genuinely-new-file, accept-keeps-file.
1. Fixed src/MergeManager.ts (hasTextBase stat check), corrected daemonHealth.js docstring
   (ECONNRESET ⇒ 'unknown', not 'dead'), added startTs/is_continue to types.ts protocol types
   (main.js + python emitters use them).
1. Added the new test to package.json "test"; ran npm test (all 6 suites pass),
   npx tsc --noEmit (clean), npm run lint:ts / lint:css (clean).
1. Wrote tmp/findings-vscode-ts.md (bug list, fixes, cross-checks, OUT OF SCOPE: none).

Note: an auto-commit (b4628973) captured the source fixes mid-task. Pre-existing uncommitted
changes to API.md / cli_steering.py / web_server.py / test_bughunt_web_server.py are from an
earlier session and were left untouched.

# Task: bug-hunt in sorcar CLI (cli_helpers/cli_steering/cli_repl/cli_panel/running_agent_state/__init__) — COMPLETE

## Steps done

1. Read all 6 scoped files fully; cross-checked persistence.py, sorcar_agent.py (drain hook),
   chat_sorcar_agent.py (\_chat_id), worktree_sorcar_agent.py (main/registration),
   vscode/helpers.py, vscode/server.py:239 (lock binding), model_info.py; read existing
   test_cli_panel.py, test_cli_steering.py, test_cli_repl.py to avoid flagging intended behavior.
1. Bug 1 (cli_steering.py SteeringSession.run abort path): Ctrl+C raised KeyboardInterrupt to the
   caller but left the daemon worker running agent.run forever (budget burn, output over next
   prompt; pending ask_user_question waiter stuck in Queue.get). Wrote failing-first PTY test
   test_ctrl_c_abort_actually_stops_the_running_agent in
   src/kiss/tests/agents/sorcar/test_bughunt_cli.py (verified FAILED: worker status 'missing').
   Fix: new SteeringSession.\_interrupt_worker — unblocks pending question with "", injects
   KeyboardInterrupt via ctypes PyThreadState_SetAsyncExc (same as vscode task_runner), joins 5s.
   Test now PASSES.
1. Bug 2 (cli_steering.py run_with_steering finally): registry read-then-modify without
   \_RunningAgentState.\_registry_lock, violating the documented mandatory locking discipline
   (TOCTOU vs peer producers). Fix: wrapped check-then-unregister in the lock. Deterministic
   failing test impossible without fakes (documented in findings); path exercised by the PTY test
   and existing fallback tests.
1. Regressions all green: test_cli_panel.py + test_cli_steering.py + test_cli_launch_work_dir.py
   (37 passed), test_cli_repl.py (27 passed). No test_steer\*.py files exist.
1. Report: tmp/findings-sorcar-cli.md (bugs + rejected candidates: cli_panel math all consistent,
   reversed recent-chats order intentional, /clear-on-non-chat unreachable, \_term_size docstring
   correct, \_resolve_task docstring nit only).
1. uv run check --full → all checks passed. Files changed: src/kiss/agents/sorcar/cli_steering.py,
   src/kiss/tests/agents/sorcar/test_bughunt_cli.py (new, committed), tmp/findings-sorcar-cli.md.

## Task B — COMPLETE

- 2 real bugs found, reproduced with failing integration tests, fixed, verified:
  - BUG-1 startTs=0 on resume-of-running-task (server.py:1088 fallback + \_live_task_start_ms helper;
    task_runner.py:277 stamps tab.agent.\_task_start_ms).
  - BUG-2 autocommit_done persisted under previous task id (merge_flow.py:428 prefers tab.task_history_id).
- Tests: src/kiss/tests/agents/vscode/test_bughunt_server_runner.py (2 passed; both failed pre-fix).
- Regressions: 217 passed (all test\_*merge*/test\_*task* in both dirs) + 81 passed (startTs/autocommit/replay suites).
- uv run check --full: all checks passed.
- Findings report: tmp/findings-vscode-server.md (tmp/ is gitignored; path mandated by task).

## Final Verification (Continuation 1)

- API.md diff inspected: legitimate (regenerated from corrected `WorktreeSorcarAgent.run()` docstring). Kept.
- `uv run check --full`: ALL checks pass (ruff, mypy, pyright, compileall, VS Code tsc+lint, mdformat, API docs).
- All 15 new bughunt integration tests pass together.
- Full regression sweep: 2163 tests across src/kiss/tests/agents/sorcar + src/kiss/tests/agents/vscode split into 8 parallel groups — ALL PASSED (0 failures).
- `npm test` in src/kiss/agents/vscode: all suites pass incl. new bughunt_isNewFile.test.js (3 tests).
- tmp/ findings files deleted; all changes committed.
