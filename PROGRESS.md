# Task: Fix restore of agent-with-sub-agents tabs after VSCode restart

## Problem analysis

After a VSCode restart, the webview restores tabs from `vscode.getState()` and
sends `resumeSession {chatId, tabId}` (NO taskId) for every restored tab
(media/main.js `init()` → SorcarSidebarView.ts `ready` handler).

`VSCodeServer._replay_session(chat_id, tab_id, task_id=None)` then calls
`_load_latest_chat_events_by_chat_id(chat_id)` which picks the latest
`task_history` row in the chat **including sub-agent rows** (sub-agents share
the parent's chat_id, see `ChatSorcarAgent._run_tasks_parallel`). Since
sub-agent rows are inserted after the parent row, the restored PARENT tab:

1. loads the last SUB-AGENT's events instead of its own,
1. gets converted into a sub-agent tab (`openSubagentTab` broadcast via the
   `subagent_info` branch),
1. never reopens its sub-agent tabs (`_open_persisted_subagent_tabs` is only
   called when `subagent_info is None`).

Additionally, restored sub-agent tabs from persisted webview state resume with
only the (shared) chatId, so they too load the wrong row.

## Plan

1. Integration test reproducing the bug:
   `src/kiss/tests/agents/sorcar/test_restore_tabs_with_subagents.py`
1. Backend fix: `_load_latest_chat_events_by_chat_id` must skip sub-agent rows
   (use `_is_subagent_row`) so a chatId-only resume always lands on the
   parent's own latest task; `_replay_session` then reopens each persisted
   sub-agent row in its own tab via `_open_persisted_subagent_tabs`.
1. Frontend fix (media/main.js):
   - `persistTabState()` / restore IIFE: do not persist/restore sub-agent tabs;
     the parent's `resumeSession` deterministically reopens them with fresh
     events (avoids duplicates and wrong content).
   - `openSubagentTab` handler: insert the sub-agent tab immediately to the
     RIGHT of its parent tab (after existing siblings) instead of appending at
     the far end of the tab bar.

## Steps done

- Explored server.py `_replay_session`, `_open_persisted_subagent_tabs`,
  persistence.py loaders, main.js persist/restore/openSubagentTab,
  SorcarSidebarView.ts restoredTabs → resumeSession flow.
- WROTE failing integration test
  `src/kiss/tests/agents/sorcar/test_restore_tabs_with_subagents.py`
  (7 tests; reproduced: restored parent tab loaded sub-agent row task_id=3
  instead of its own task_id=1 and was converted to a sub-agent tab).
- FIXED backend: `_load_latest_chat_events_by_chat_id` in
  src/kiss/agents/sorcar/persistence.py now iterates rows
  `ORDER BY timestamp DESC, id DESC` and skips rows where
  `_is_subagent_row(extra)` is true; returns None if only sub-agent rows.
  All 7 new tests PASS now.

## Remaining TODO

1. Frontend media/main.js fixes (NOT yet done):
   a. `persistTabState()` (~line 752): filter out `t.isSubagentTab` tabs from
   the serialized list (parent's resumeSession deterministically reopens
   them via `_open_persisted_subagent_tabs`); activeTabIndex computed on
   the filtered list (fallback to 0 if active tab was a sub tab).
   b. Restore IIFE (~line 776): skip `st.isSubagentTab` entries from older
   persisted states so they don't send chatId-only resumeSession (which
   would now load parent events + duplicate sub tabs).
   c. `case 'openSubagentTab':` handler (~line 3204): when creating a NEW
   sub tab, insert it immediately to the RIGHT of its parent tab (after
   existing contiguous siblings with same parentTabId) instead of
   `tabs.push(subTab)` at the end. Note: parentId is currently computed
   AFTER the push — must compute it before insertion.
1. Optionally add a JS test (test/ dir uses node --test with \_vscode-stub.js)
   for the persist-filter; check eslint via `cd src/kiss/agents/vscode && npx eslint`.
1. Run impacted Python tests: test_restore_tabs_with_subagents.py,
   test_subagent_history_click.py, test_resolve_parent_tab_id_for_sub.py,
   test_event_persistence_no_duplicate.py, test_subagent_events_after_followup.py,
   plus grep for other users of `_load_latest_chat_events_by_chat_id`.
1. Run `uv run check --full` and fix all errors.
1. Delete any tmp files; finish.
