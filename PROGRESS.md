# Task: Fix 4 failing tests (no production code changes needed)

## Status

### DONE: test_subagent_history_click.py (1 of 4 fixed)
File: `src/kiss/tests/agents/sorcar/test_subagent_history_click.py`
Test: `TestReplaySessionOpensSubagentTab::test_replay_subagent_does_not_invoke_reattach_running_chat`
Fix applied: replaced the stale assertion that `"tab-history-click" in server._running_agent_states`
with the current design's invariant (replay is a VIEW operation — `_replay_session` in
`src/kiss/agents/vscode/server.py` ~line 1040 only updates an existing registry entry, never creates one):
```python
        assert "tab-parent" in server._running_agent_states
        assert server._running_agent_states["tab-parent"] is parent_tab
        # Replay is a VIEW operation: the viewer tab must NOT get its
        # own ``_RunningAgentState`` registry entry (no agent runs
        # there), and in particular the parent's state must not have
        # been rebound under the new tab id.
        assert "tab-history-click" not in server._running_agent_states
        parent_thread.join(timeout=1)
```
NOT YET VERIFIED by running the test — must run:
`uv run pytest src/kiss/tests/agents/sorcar/test_subagent_history_click.py -v`

### TODO: test_tab_switch_race_regression.py (3 remaining failures)
File: `src/kiss/tests/agents/sorcar/test_tab_switch_race_regression.py` (1765 lines)
The file already has a node-based behavior-test harness:
- `_JS_PREAMBLE` (lines ~29-148): minimal DOM stubs
- `_run_node(script)` (line 151): runs `node -e script`
- `_make_test_script(body)` (line 161): preamble + body
- `_MAIN_JS` points to `src/kiss/agents/vscode/media/main.js`
Many existing tests use `_run_node(_make_test_script(r"""..."""))` with small JS replicas
of main.js logic and print PASS/FAIL — follow that pattern (see e.g.
`TestClearGuard::test_clear_skipped_on_wrong_tab` line ~243 for style).

Failing tests to refactor into behavior tests (verify behavior, not source substrings):

1. `TestSetReadyResetsRunningTabId::test_set_ready_resets_tab_running_state` (line ~884):
   currently asserts `"doneTab.isRunning = false"` and `"doneTab.t0 = null"` within first
   400 chars of `function setReady(` — fails because setReady was rewritten (now begins
   with comments + doneStartTs/doneEndTs handling).

2. `TestPerTabT0::test_switch_to_non_running_tab_clears_t0` (line ~1288):
   asserts `'t0 = null'` inside `function switchToTab(tabId)` — t0 reset logic was
   refactored out of switchToTab.

3. `TestPerTabT0::test_set_ready_clears_running_tab_t0` (line ~1291):
   searches `function setReady(label, tabId)` — signature changed to
   `function setReady(label, tabId, doneStartTs, doneEndTs)` (main.js line 3587).

Key main.js locations (CURRENT source, must read before writing behavior tests):
- `function makeTab(title)` line 159
- `function saveCurrentTab()` line 276
- `function restoreTab(tab)` line 338
- `function switchToTab(tabId)` line 553
- `function setRunningState(running)` line 3552
- `function setReady(label, tabId, doneStartTs, doneEndTs)` line 3587

Plan for refactor: read setReady (line 3587+) and switchToTab (line 553+) bodies from
main.js, then write `_run_node` behavior tests that replicate the relevant logic
extraction OR (better) extract the actual function source from main.js and evaluate it
in node with stubbed globals, asserting:
- after setReady('Done', <runningTabId>, ...): the done tab's isRunning is false and its
  t0/timer state is cleared (behavioral: tab object fields).
- after switching to a non-running tab: the live timer state is not running (status text
  not ticking / t0-equivalent cleared).
Alternative simpler approach used elsewhere in the file: extract the real function text
from main.js via string slicing and eval it inside node with stub globals, then assert
post-conditions on tab objects.

## Remaining steps
1. Read main.js setReady body (~line 3587) and switchToTab body (~line 553) and
   surrounding timer-state code (find where t0/doneStartTs/doneEndTs/timer reset lives).
2. Rewrite the 3 failing tests as behavior tests.
3. Run: `uv run pytest src/kiss/tests/agents/sorcar/test_tab_switch_race_regression.py src/kiss/tests/agents/sorcar/test_subagent_history_click.py -v`
4. `uv run check --full` (a .py test file was modified; .js untouched so far).
5. Clean tmp files I created (none created yet in tmp/), update PROGRESS.md, finish.
