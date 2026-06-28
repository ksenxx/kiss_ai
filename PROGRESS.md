# Progress

## Task

> When a tab has an associated chat-id of a real task, changing the
> working directory in the settings page MUST not change the directory
> of that tab and chat id. Reproduce a violation of the invariant by
> writing an end-to-end test. Then fix the issue.

Models:

- **claude-opus-4-7** drove the analysis, repro test, and code fix.
- **gpt-5.5** reviewed coverage, looked for missed sites, and verified
  no regressions before finish.

## Repro / fix (claude-opus-4-7)

### Root cause

The chat webview routes follow-up commands (`submit`,
`autocommitAction`, `worktreeAction`, `mergeAction`, …) through
`workDirForTab(tabId)`:

```js
function workDirForTab(tabId) {
  const tab = getTab(tabId);
  if (tab && tab.workDir) return tab.workDir;
  return configWorkDir || '';
}
```

`tab.workDir` was only pinned when a `task_events` replay carried
`extra.work_dir` (i.e. when the user reopened a chat from history and
the persisted `extra` happened to include the directory). Two
production paths bind a chat-id of a real persisted task to a tab
without ever pinning `tab.workDir`:

1. `clear` event — broadcast for a freshly-submitted task as soon as
   `ChatSorcarAgent` persists it in the DB. No `task_events`
   replay fires.
1. `task_events` event whose `extra` is missing `work_dir` —
   older persisted rows.

Both leave `tab.workDir = ''`. A later settings-panel change to the
work directory updates `configWorkDir` — and the very next command
from that tab is routed to the **new** directory even though the
bound chat-id still belongs to the **original** task. Auto-commit /
merge then run on the wrong repo; follow-up submits inherit a wrong
`work_dir` snapshot.

### Reproduction test

`src/kiss/agents/vscode/test/tabWorkDirSettingsInvariant.test.js`
drives production `media/chat.html` + `media/panelCopy.js` +
`media/main.js` in JSDOM. Three sub-tests:

1. `testInvariantHoldsAfterSettingsChange_ClearBind` — bind chat-id
   via `clear`, change `configData.work_dir`, raise an
   `autocommit_prompt`, click "Auto commit"; the posted
   `autocommitAction.workDir` MUST equal the pre-change work_dir.
   **Failed before the fix** (returned the new settings value),
   passes after.
1. `testInvariantHoldsAfterSettingsChange_TaskEventsBindNoExtraWorkdir` —
   same flow but binding via `task_events` whose `extra` omits
   `work_dir` (older rows).
1. `testTaskEventsExtraWorkDirStillWinsOverConfig` — sanity: when
   `extra.work_dir` IS present, it pins the tab and survives a later
   settings change. Guards against future regression of the existing
   path.

The first test reproducibly failed before the fix:

```
AssertionError [ERR_ASSERTION]: INVARIANT: ... — observed workDir = "/path/new"
+ actual - expected
+ '/path/new'
- '/path/initial'
```

### Fix — `src/kiss/agents/vscode/media/main.js`

Three small additions:

1. `case 'clear'` — when a chat-id is bound to a tab, pin
   `tab.workDir` from `configWorkDir` if it is still empty:

```js
if (ev.chat_id && clearTab) {
  clearTab.backendChatId = ev.chat_id;
  if (!clearTab.workDir && configWorkDir) {
    clearTab.workDir = configWorkDir;
  }
  persistTabState();
}
```

2. `case 'task_events'` — same fallback at chat-id bind. The
   `extra.work_dir` branch downstream still takes priority and
   overwrites the value with the task's recorded directory; the
   fallback only matters when `extra` carries no `work_dir`.

1. `persistTabState` + initial restore — also persist
   `tab.workDir` so a window reload that restores the tab keeps the
   same effective work_dir even before `resumeSession` re-pins it
   from the next `task_events` replay, even for older persisted
   rows whose `extra` carries no `work_dir`.

`package.json`'s `test` script now runs
`test/tabWorkDirSettingsInvariant.test.js` so it's part of `npm test`
and `uv run check --full`.

## Review (gpt-5.5)

- Verified every `backendChatId =` assignment site
  (`grep -n "backendChatId =" media/main.js`) is now covered by a
  matching `workDir` pin: lines 1267 (restore), 3245 (clear), 3408
  (task_events).
- Verified no other code path overwrites `tab.workDir` from
  `configWorkDir` (greps for `tab.workDir =` / `.workDir =` show only
  the deliberate `extra.work_dir` assignments in the active and
  background `task_events` branches — both keep priority over the
  bind-time fallback).
- Verified `saveCurrentTab`/`restoreTab` do not touch `workDir` —
  tabs live in the `tabs` array and survive tab switches without
  losing the pin.
- Ran the full JSDOM test suite individually
  (`historyClickSwitchExistingChat`, `historyWorkspaceFilter`,
  `historyTaskWorkspace`, `historyTaskMeta`, `historyBurgerMenuRunningTask`,
  `historyRunningPulsingDot`, `ask_user_*`, `bughunt5_*`,
  `bughunt6_files_stale`, `bughunt_submit_while_running`,
  `bughunt_reopen_running_tab`, `syncWorkDir`, `panelTimeSpent`,
  `panelTimeActiveTick`, `bashHeaderCyan`, `multiSessionResultOrder`,
  `historyTaskDuration`, `tab_timer_per_tab`, `historyFilterDateGroup`,
  `historyTaskRowIndicators`, …) plus the new
  `tabWorkDirSettingsInvariant` test — all pass.
- `uv run check --full` passes (ruff, mypy, pyright, VS Code TS
  check + lint, mdformat).
