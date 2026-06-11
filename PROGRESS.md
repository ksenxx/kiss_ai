# Task: Find & reproduce inconsistencies/bugs in src/kiss/agents/vscode/ and src/kiss/agents/sorcar/ via integration tests, then fix

## Baseline

- Compiled TS (`npx tsc -p .`); all 6 existing JS test files pass.
- Ran full sorcar+vscode Python suite (246 files, ~2200 tests) in 8 parallel shards.
  3 failures, all passing in isolation — traced to cross-test event pollution (see bug 16).
- Diffed frontend/backend message-type protocol surfaces — consistent.

## Bugs found (each reproduced by a failing-first integration test, then fixed)

1. `persistence._list_recent_chats` leaked sub-agent rows into the CLI resume menu
   (fix: filter `_is_subagent_row`; test `test_bughunt_persist2_list_recent_chats.py`).
1. `persistence._delete_task` orphaned sub-agent rows + events; `_chat_has_tasks`
   stayed true for visually-empty chats (fix: cascade delete children;
   test `test_bughunt_persist2_delete_cascade.py`).
1. `persistence._get_adjacent_task_by_chat_id` unreachable tasks on timestamp ties
   (fix: total order `(timestamp, id)`; test `test_bughunt_persist2_adjacent_ts_tie.py`).
1. `git_worktree.GitWorktreeOps.copy_dirty_state` mishandled symlinks (5 modes: dropped
   broken links, materialized links as files, wrote through links corrupting targets,
   crashed on renamed dir-symlinks via rmtree, left stale deleted links). Fix: check
   `is_symlink()` first + `_remove_path` helper; test `test_bughunt_wt2_symlinks.py`.
1. `web_use_tool._is_profile_in_use` treated `PermissionError` from `os.kill(pid, 0)`
   as "process dead", handing out a live Chromium profile and deleting its locks
   (fix: EPERM ⇒ in use; test `test_bughunt_agent2_profile_lock.py`).
1. `cli_steering._InputBox.feed` didn't swallow SS3 escapes (`ESC O x`) — arrows/F1–F4
   typed literal `OA`/`OP` into the box (test `test_bughunt_cli2_ss3_keys.py`).
1. `cli_repl._run_one` let any agent exception kill the whole REPL session
   (fix: catch, print `✗ Task failed`, return to prompt;
   test `test_bughunt_cli2_repl_survives_error.py`).
1. `commands._cmd_get_adjacent_task` created a phantom registry entry for viewer tabs
   and always resolved an empty chat id ⇒ prev/next arrows broadcast empty payloads
   (fix: read registry without creating, fall back to `_tab_chat_views`;
   test `test_bughunt_srv2_adjacent_viewer_tab.py`).
1. `server._new_chat` / `_replay_session` left stale live-task subscriptions, leaking
   another chat's stream into tabs that navigated away (fix: `cleanup_tab` on
   new-chat/replay, re-subscribe owner; test `test_bughunt_srv2_stale_subscriptions.py`).
1. `diff_merge._parse_diff_hunks` ran `git diff` without `--no-renames`: renamed files
   produced inconsistent merge manifests (hunks past end of empty base; reject corrupted
   files) (test `test_bughunt_mrg2_rename_merge_view.py`).
1. `merge_flow` `--name-only` diffs (4 sites) also missed `--no-renames`: changed-file
   list omitted the deleted old path and the conflict check missed dirty-main edits to
   renamed paths (test `test_bughunt_mrg2_worktree_rename_files.py`).
1. `web_server` "ready" handler didn't add `restoredTabs` to `tabs_seen`, so restored
   tabs escaped deferred-close on disconnect ⇒ permanent backend tab-state leak
   (test `test_bughunt_web2_restored_tabs_close.py`).
1. `web_server._read_url_from_file` crashed (AttributeError) on valid-JSON-but-non-dict
   content despite docstring promising None (test `test_bughunt_web2_url_file.py`).
1. `media/main.js` `status` handler set the global running-timer anchor for
   broadcast/background-tab events, clobbering the active tab's timer
   (test `vscode/test/bughunt2_status_timer.test.js`).
1. `media/demo.js` rendered `is_continue` results as "Status: FAILED" instead of
   "Status: Continue" (mismatch with main.js/types.ts contract;
   test `vscode/test/bughunt2_demo_continue.test.js`).
1. Stale async events misrouted across database swaps: the background event writer and
   the fire-and-forget `followup_suggestion` thread wrote events carrying numeric task
   ids from the OLD `_DB_PATH` into the NEW database, attaching them to an unrelated
   task with the same row id (root cause of the recurring cross-test pollution).
   Fix: every queued write is stamped with `origin_db_path`
   (`_queue_chat_event`/`_append_chat_event`/`_write_event_batch` in persistence.py;
   `_generate_followup_async` in server.py captures the path at scheduling time).
   Test `test_bughunt_stale_db_events.py` (verified failing pre-fix via git stash).

## Coordination fixes after parallel sub-agent merge

- `server._new_chat`: guarded `cleanup_tab` via `getattr` for duck-typed test printers
  (fixed `test_worktree_audit17` regression from bug-9 fix).
- `test_flush_chat_events_dead_writer.py`: updated raw queue-item simulation to the new
  4-tuple shape.

## Verification

- All new bughunt tests pass (39 Python + 4 JS assertions).
- Full sharded suite re-run: 2217 passed; only `test_bughunt_cli.py::test_ctrl_c_abort…`
  failed under shard load (pre-existing forkpty flake, fails identically on baseline,
  passes in isolation).
- `npx tsc -p .` + `npm test` + new JS tests: all pass.
- `uv run check --full`: All checks passed.
