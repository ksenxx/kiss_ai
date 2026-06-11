# Bug-hunt task: vscode server.py / commands.py / task_runner.py / json_printer.py — COMPLETE

## Steps done (chronological)

1. Read all 4 scoped source files fully; built candidate-bug list.
1. Verified persistence contracts (`_get_adjacent_task_by_chat_id` returns None for empty
   chat_id; `_record_model_usage` does persist last model — not a bug; loader dict keys and
   `_on_task_id_allocated(int, str)` callback contract all match — no key-mismatch bugs).

## BUG-1 (FIXED): adjacent-task navigation broken in pure-viewer tabs

- File: src/kiss/agents/vscode/commands.py, `_cmd_get_adjacent_task` (~line 504).
- `_replay_session` deliberately creates NO `_RunningAgentState` for viewer tabs (C2/C3),
  only `_tab_chat_views[tab_id]=chat_id`. The handler called `self._get_tab(tab_id)` which
  CREATED a fresh state with `chat_id=""` → `_get_adjacent_task_by_chat_id` returned None →
  empty `adjacent_task_events` broadcast.
- Failing test (now passes):
  src/kiss/tests/agents/vscode/test_bughunt_srv2_adjacent_viewer_tab.py
- Fix: resolve chat id from existing registry entry, falling back to `_tab_chat_views`,
  under `_state_lock`; never create a registry entry (read-only view op).

## BUG-2 (FIXED): stale task-stream subscriptions when a tab navigates away

- Files: src/kiss/agents/vscode/server.py — `_new_chat` (~line 880), `_replay_session`
  (~line 976), `_reattach_running_chat` (~line 1431).
- Only unsubscribe path was tab CLOSE. So: (a) "New Chat" left the tab subscribed to the
  old chat's running task (events streamed onto welcome screen); (b) replaying a different
  chat into the tab kept the old chat's live stream flowing in, mixing two task streams.
- Failing tests (now pass) + 2 regression guards:
  src/kiss/tests/agents/vscode/test_bughunt_srv2_stale_subscriptions.py
- Fix: `printer.cleanup_tab(tab_id)` in `_new_chat`; same (getattr-guarded for duck-typed
  test printers) in `_replay_session` before `_reattach_running_chat`; and
  `_reattach_running_chat` now re-subscribes even when source tab == new tab (owner
  self-replay), since replay just dropped its subscriptions. Lint auto-fix removed the
  now-unused `source_tab_id` local.

## Verification

- `uv run pytest src/kiss/tests/agents/vscode/test_bughunt_server_runner.py src/kiss/tests/agents/vscode/test_bughunt_srv2_*.py -q --no-cov` → 7 passed.
- Impacted neighbors (resume_running_symmetry, multi_viewer_streaming, detach_tab_and_reattach,
  chat_viewer_live_stream, new_chat_model_picker, + 11 more subscription/replay test files)
  → all pass (119 passed earlier run; 27 passed final run).
- `uv run check --full` → all checks pass.
- New tests + fixes staged in git.
