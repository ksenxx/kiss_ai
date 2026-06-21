# Progress

## Task

When a chat id's tasks are loaded into a tab, if the user has changed the global settings (`is_worktree`, `is_parallel`, `auto_commit_mode`, `model`) since the last task ran, the NEXT task launched in the same tab must use the user's CURRENT global settings instead of the loaded task's historical snapshot.

## Root cause

1. `_TaskRunnerMixin._run_task_inner` persists per-task snapshots of `model`, `is_worktree`, `is_parallel`, `auto_commit_mode` into `task_history.extra` (JSON).
1. `VSCodeServer._replay_session` (backend, `src/kiss/agents/vscode/server.py`) broadcasts a `task_events` event carrying the unmodified persisted `extra` blob to the frontend on a chat load.
1. The frontend's `task_events` handler (`src/kiss/agents/vscode/media/main.js`, active-tab branch ~lines 2940–2980 and background-tab branch ~lines 2880–2900) read those keys back out of `extra` and clobbered the live UI: `selectedModel`, `modelName.textContent`, `worktreeToggleBtn.checked`, `parallelToggleBtn.checked`, `autocommitToggleBtn.checked`, and the per-tab `teTab.selectedModel`.
1. The toggles + `selectedModel` are the SOURCE OF TRUTH for the next `submit` (`useWorktree`, `useParallel`, `autoCommit`, `model` flags sent to the backend's `_cmd_run`).
1. Backend `_replay_session` additionally seeded `tab.use_worktree` from the loaded task's `extra.is_worktree`, mirroring the same stale-state path in Python state.

Net effect: after loading a chat, the live UI silently reverted to the loaded task's historical settings, so any new task in the same tab ran with those stale settings — even after the user had explicitly changed the global config.

## Fix

### Backend (`src/kiss/agents/vscode/server.py`)

- Added `_REPLAY_STRIPPED_EXTRA_KEYS = ("model", "is_worktree", "is_parallel", "auto_commit_mode")` and `_extra_for_replay(extra) -> str`. The helper parses the persisted JSON, removes the four keys, and reserializes (passing through non-string / non-dict-JSON payloads unchanged so it is safe to apply unconditionally).
- `_replay_session` now broadcasts `task_events` with `"extra": _extra_for_replay(result.get("extra", ""))` — preserves `startTs` / `endTs` / `work_dir` / `tokens` / `cost` / `steps` / `version` / `subagent` for header/timer/routing.
- `_open_persisted_subagent_tabs` applies the same strip to each sub-agent's `task_events.extra` so the live model picker cannot inherit a stale per-sub-agent model snapshot when the user switches to a reopened sub-tab.
- Removed the `tab.use_worktree = bool(extra_raw.get("is_worktree") ...)` seeding block in `_replay_session`. The follow-up `_run_task` always overwrites `tab.use_worktree` from `cmd.useWorktree`, which now reflects the user's live (= global) toggle state.

### Frontend (`src/kiss/agents/vscode/media/main.js`)

- Active-tab `task_events` handler no longer reads `extra.model`, `extra.is_worktree`, `extra.is_parallel`, `extra.auto_commit_mode`. Still reads `extra.startTs`, `extra.endTs`, `extra.work_dir`.
- Background-tab `task_events` handler no longer reads `bgExtra.model` (which would otherwise be promoted to the live `selectedModel` via `restoreTab` when the user switched to the tab). Still reads `bgExtra.work_dir` / `bgExtra.startTs` / `bgExtra.endTs`.
- Detailed comments at both sites explain the invariant (historical extras must not clobber live state) and reference the backend's defensive strip.

### Test (`src/kiss/tests/agents/sorcar/test_replay_uses_global_settings.py`)

End-to-end (Python, no JS) integration test with three cases:

1. `test_replay_strips_global_setting_keys_from_extra`: runs a task with `(use_worktree=True, use_parallel=True, auto_commit=True, model="claude-opus-4-6")`, loads it via `newChat` + `resumeSession`, then asserts the captured `task_events.extra` no longer contains any of the four stripped keys while still carrying `work_dir` / `startTs` / `endTs`.
1. `test_followup_task_uses_new_global_settings`: after the load, asserts `loaded_tab.use_worktree` is False (no historical seeding), then runs a follow-up task with the NEW global settings and verifies the persisted second-task `extra` reflects the new global values.
1. `test_persisted_subagent_extras_are_stripped`: seeds a parent + 2 sub-agent rows whose `extra` carries the stripped keys, calls `_replay_session(parent)`, and asserts every emitted sub-tab `task_events.extra` has the four keys stripped but preserves `startTs` / `endTs` / `work_dir` / `subagent` metadata.

Bug reproduction verified: tests fail against pre-fix code (verified with `git stash`) and pass against the fix.

### Verification

- `uv run pytest src/kiss/tests/agents/sorcar/test_replay_uses_global_settings.py` — 3 passed.
- `uv run pytest` for related neighbors (`test_history_continuation_context`, `test_load_task_opens_subagent_tabs`, `test_subagent_history_click`, `test_reopen_started_tab_resume`, `test_subagent_result_not_in_parent_webview`, `test_autocommit_off_on_failure`, `test_subagent_events_after_followup`, `test_update_settings`, `test_vscode_tab_isolation_fixes`, `test_history_scroll_to_task`) — 101 passed.
- `uv run check --full` — all checks pass (ruff, mypy, pyright, mdformat, generate-api-docs, compileall).

### gpt-5.5 thorough review

- Spawned a parallel review under `gpt-5.5`. Review confirmed the active-tab bug is fixed and identified two additional defects which were then addressed:
  1. `_open_persisted_subagent_tabs` was bypassing the strip — fixed by routing its `task_events.extra` through `_extra_for_replay`.
  1. Background-tab `task_events` was still reading `bgExtra.model` into `teTab.selectedModel` — removed for defense-in-depth.
- Test gap noted: the test does not directly exercise the JS DOM. Mitigated by the additional sub-agent test that exercises the `_open_persisted_subagent_tabs` path. The DOM clobber path is structurally impossible after the JS edits (assignments removed).
