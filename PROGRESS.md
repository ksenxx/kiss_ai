# Task: Fix 4 failing tests — COMPLETED

## Summary

All 4 failing tests have been fixed and verified (commit `a9a9a442`).

### Fix 1: test_subagent_history_click.py

- **Test:** `TestReplaySessionOpensSubagentTab::test_replay_subagent_does_not_invoke_reattach_running_chat`
- **Root cause:** Stale assertion expected `"tab-history-click" in server._running_agent_states` but replay is a VIEW operation that never creates registry entries.
- **Fix:** Replaced with `assert "tab-history-click" not in server._running_agent_states`.

### Fix 2–4: test_tab_switch_race_regression.py

Three brittle source-substring tests refactored to behavioral Node.js tests using `_extract_function` + `_run_node`:

1. **`TestSetReadyResetsRunningTabId::test_set_ready_resets_tab_running_state`** — Now evaluates real `setReady` in Node, verifying `isRunning=false`, `t0`/`endTs` anchoring, and background-tab UI isolation.
1. **`TestPerTabT0::test_switch_to_non_running_tab_clears_t0`** — Now evaluates real `switchToTab` in Node, verifying timer stops and `t0`/`endTs` anchors are preserved.
1. **`TestPerTabT0::test_set_ready_clears_running_tab_t0`** — Now evaluates real `setReady` in Node, verifying active-tab completion clears running state, re-anchors timestamps, and handles legacy fallback.

## Verification

- All 4 tests pass: `uv run pytest ... -v` → 4 passed
- Full test files pass: 73 passed
- `uv run check --full` passes (except pre-existing markdown lint on PROGRESS.md, now fixed)
