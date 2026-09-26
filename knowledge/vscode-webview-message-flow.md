---
title: Webview message flow (webview, extension host, daemon)
uuid: e8bc4f6d-2608-4274-b99b-9685e92c05d3
summary: How chat webview messages travel - main.js api.js whitelist vs postToHost,
  SorcarSidebarView _handleMessage and FORWARDED_COMMANDS, daemon events relayed back,
  tabId ownership, types.ts unions.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Webview message flow (webview, extension host, daemon)

```
media/main.js --postMessage--> SorcarSidebarView._handleMessage --AgentClient--> kiss-web daemon
      ^                                                                            |
      +------------- _sendToWebview <---- client 'message' listener <--------------+
```

## Webview side (`media/main.js`, `media/api.js`)
- `main.js` starts with `const vscode = acquireVsCodeApi(); const api = createSorcarApi(msg => vscode.postMessage(msg));`.
- `api.js` defines `SORCAR_API_COMMANDS`, the list of **daemon commands** the webview may send; `api.send(msg)` throws on an unknown type and each name gets a helper (`api.getConfig()`, `api.saveConfig({...})`). The list mirrors the daemon's API catalog in `kiss/server/sorcar.py`; `test/sorcarApi.test.js` and the Python test `src/kiss/tests/agents/vscode/test_server_api.py` guard that relation.
- **Host-only** messages (not daemon commands) go through `postToHost(msg)`, e.g. `setEditorTabsMode`, `openChatPanel`, `closePanel`, `panelTitle`, `metaUpdate`, `activeTask`, `revealPanel`, `openWorkDir`, `pickWorkDir`. They are intentionally absent from `api.js`.
- Incoming messages are handled in the big `window.addEventListener('message', ...)` switch in `main.js` (`case 'tabs_state'`, `case 'configData'`, `case 'openSettings'`, ...).

## Host side (`src/SorcarSidebarView.ts`)
`_handleMessage(message: FromWebviewMessage)`:
1. Drops everything after `dispose()` (`_terminated`), because handling could rebuild the client after teardown.
2. Records tab ownership: any message with `tabId` adds it to `_ownTabs` (`closeTab` removes it); `ready.restoredTabs` are added too.
3. If `FORWARDED_COMMANDS[message.type]` exists (a table of type -> field list), it copies exactly those fields and `forward`s the command to the daemon. Most simple commands take this path.
4. Otherwise a `switch` handles host-side logic: `ready`, `submit`, `stop`, `openFile`, `checkPaths`, `resumeSession`, `complete`, `worktreeAction`, `voiceToggle`, `closeTab`, `panelTitle`, `metaUpdate`, `openChatPanel`, `closePanel`, `setEditorTabsMode`, `openWorkDir`, `pickWorkDir`, and others.

Notable handlers:
- `ready`: sends `daemonStatus`, welcome suggestions, remote URL, cached `metaState`/`activeTask`, then forwards `ready` (with `restoredTabs`, `singleTabId`) to the daemon, which answers with models/history/config, a canonical `tabs_state` and a replay of chat transcripts. A pending `insertAndSubmit` is flushed afterwards.
- `submit`: if the tab is running, the prompt becomes `appendUserMessage` (follow-up). A single-line prompt that resolves to an existing **file** (`_resolveTabFile(..., fileOnly=true)`) opens the file instead of starting a task. Otherwise `_startTask` posts `status running:true` and calls `api.run({...})` with `workDir`, `tabScopeWorkDir`, `attachments`, `useWorktree`, `useParallel`, `autoCommit`, `webTools`, `tabId`.

## Daemon -> webview
`_installClientListener` receives every daemon message, applies a few host-side hooks, and relays it with `_sendToWebview`; there is no allow-list. Hooks include:
- `configData`: `msg.config.work_dir` is overwritten with the window's workspace folder (the webview scopes tabs and history by it).
- `commitMessage`, `worktree_result`, `main_tree_result`: notifications and progress are handled only when `_isOwnTab(msg.tabId)` (no tabId counts as own). Exception: a successful non-kept `worktree_result` drops the tab's worktree dir and closes it in SCM regardless of ownership.
- `tabs_state`: updates the host's registry mirror and fires `onRegistryTabsState` (used by editor-tabs adoption).

## Types
`src/types.ts`: `FromWebviewMessage` (webview -> host), `ToWebviewMessage = ToWebviewMessageBody & {tabId?}` (host -> webview), `AgentCommand` (host -> daemon). Adding a message means updating these unions (see `vscode-adding-a-webview-command`).

## Sources
- `src/kiss/agents/vscode/media/api.js` (`SORCAR_API_COMMANDS`, `createSorcarApi`)
- `src/kiss/agents/vscode/media/main.js` (`postToHost`, message listener)
- `src/kiss/agents/vscode/src/SorcarSidebarView.ts` (`FORWARDED_COMMANDS`, `_handleMessage`, `_startTask`, `_installClientListener`, `_isOwnTab`)
- `src/kiss/agents/vscode/src/types.ts`
