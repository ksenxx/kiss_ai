# Progress

## Task

Review the commit `cc6b47b5` ("prevent historical task settings from
clobbering global config") using gpt-5.5 (non-codex); reproduce any
reported bugs with end-to-end tests under claude-opus-4-7; fix and
iterate until gpt-5.5 reports no defects.

## Result — `Defects found: none.` after round 4.

## Loop history

- **Round 1 (review):** original fix correct; flagged 3 residual
  defects: Medium (`tab.use_parallel`/`auto_commit_mode` not reset
  on history load — symmetric to `use_worktree`); Low
  (`_extra_for_replay` non-dict pass-through); Low
  (`tab.selected_model` not reset).
- **Round 1 (fix):** in `src/kiss/agents/vscode/server.py`,
  `_extra_for_replay` returns `""` for non-dict JSON;
  `_replay_session` adds an explicit per-tab reset gated on a
  `tab_alive` predicate.
- **Round 2 (review):** flagged Medium that `tab_alive` omitted
  `tab.is_merging`, and Low that the docstring was inconsistent.
- **Round 2 (fix):** replaced ad-hoc `tab_alive` disjunction with the
  shared `_tab_busy(tab)` predicate (covers `is_task_active`,
  `is_merging`, `task_thread.is_alive()`); updated
  `_extra_for_replay` docstring; added two tests:
  `test_history_load_preserves_state_during_merge_review` and
  `test_history_load_preserves_state_when_thread_alive`.
- **Round 3 (review):** confirmed round-2 fixes; flagged Low that
  `_tab_busy` docstring was now stale (didn't mention
  `_replay_session` as a caller).
- **Round 3 (fix):** updated `_tab_busy` docstring with the third
  caller + lock-safety contract.
- **Round 4 (review):** **Defects found: none.**

## Files touched

- `src/kiss/agents/vscode/server.py`
  - `_extra_for_replay`: non-dict JSON → `""`; docstring updated.
  - `_replay_session`: state-reset block gated by `_tab_busy(tab)`.
  - `_tab_busy`: docstring updated for the third caller and the
    lock-safety contract.
- `src/kiss/tests/agents/sorcar/test_replay_uses_global_settings.py`
  - +4 tests in `TestReplayUsesGlobalSettings`:
    `test_history_load_resets_use_parallel_and_auto_commit`,
    `test_history_load_resets_selected_model`,
    `test_history_load_preserves_state_during_merge_review`,
    `test_history_load_preserves_state_when_thread_alive`,
    `test_history_load_preserves_state_during_live_run`.
  - +new `TestExtraForReplayUnit` class with 5 edge-case tests.

## Verification

- `uv run pytest src/kiss/tests/agents/sorcar/test_replay_uses_global_settings.py`
  — 13 passed.
- 11 impacted neighbor suites — 104 passed total.
- `uv run check --full` — all green.
