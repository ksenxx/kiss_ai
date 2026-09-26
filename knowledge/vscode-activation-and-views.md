---
title: VS Code extension activation, views, commands and settings
uuid: 970bc4b0-6f2d-4358-97ea-ed3bb9f16899
summary: How extension.ts activate() wires the History, Chat and Task Info webview
  views, the four SorcarSidebarView controllers, commands, keybindings, kissSorcar.*
  settings and first-launch behavior.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# VS Code extension activation, views, commands and settings

## Activation
`package.json` `activationEvents`: `onStartupFinished` and `onWebviewPanel:kissSorcar.chatTab` (the second lets a serialized chat editor tab revive the extension). `activate()` in `src/extension.ts` runs, in order:
1. `ensureLocalBinInPath()` (prepends `~/.local/bin` to `PATH`).
2. Creates `sidebarView = new SorcarSidebarView(...)` and registers it for view `kissSorcar.chatViewSecondary` with `retainContextWhenHidden: true`.
3. Reads workspaceState `kissSorcar.editorPanelTabIds` (tab ids of chat panels open at the previous shutdown) and builds `SorcarPanelManager` with a retire callback that goes through the sidebar controller's long-lived client.
4. `metaView` (Task Info, `kissSorcar.metaViewSecondary`, body attrs `metaPanelBodyAttrs()`), fed by `panelManager.setMetaSink`.
5. `onDidChangeConfiguration` handler for `kissSorcar.editorTabsMode` (mode switch, see `vscode-editor-tabs-mode`) and `window.customTitleBarVisibility`.
6. `historyView` (`kissSorcar.historyView`, body attrs `historyPanelBodyAttrs()`), whose `openChat` events go to the panel manager (editor-tabs mode) or `sidebarView.openChatFromHistory` (sidebar mode).
7. Commands, SCM commit-message hooks (`vscode-scm-commit-message`), the `.extension-updated` reload watcher (`vscode-update-and-window-reload`), one-time sidebar widening, first-launch auto-open, `ensureDependencies()`, `checkForExtensionUpdate()`.

So up to four kinds of `SorcarSidebarView` exist: the sidebar chat, the history panel, the Task Info panel, and one per chat editor panel. Each owns its own `AgentClient` connection to the daemon. Only the sidebar controller (the one without `panelHooks`) sends `getTabsState` on every (re)connect, so the window always has a registry baseline.

`deactivate()` calls `panelManager.markShutdown()` first, so panel disposals during teardown never retire chats from the daemon registry.

## Views (`contributes.views`)
- Activity bar container `kissSorcarContainer`: webview `kissSorcar.historyView` ("History"), used in both modes.
- Secondary sidebar container `kissSorcarSecondary` (`order: -100`): `kissSorcar.chatViewSecondary` ("Chat", `when: !config.kissSorcar.editorTabsMode`) and `kissSorcar.metaViewSecondary` ("Task Info", `when: config.kissSorcar.editorTabsMode`).
All three views render the same `media/chat.html`; `<body>` classes select the surface (see `vscode-chat-html-build`).

## Commands and keybindings
Commands: `kissSorcar.openPanel`, `newConversation`, `openSettings`, `stopTask`, `generateCommitMessage`, `gitCommit`, `toggleFocus`, `focusEditor`, `runSelection`, `insertSelectionToChat`, `showHistory`. Keybindings: `ctrl+t` new conversation, `ctrl+d` toggle focus, `ctrl+e` run selection, `ctrl+l` insert selection into chat (`cmd+` variants on macOS). Commands act on `chatController(createIfMissing)`: the active chat panel in editor-tabs mode, otherwise the sidebar view.

## Settings (`contributes.configuration`)
| Key | Type | Default | Use |
|---|---|---|---|
| `kissSorcar.defaultModel` | string | `claude-opus-4-6` | model preselected in the webview |
| `kissSorcar.kissProjectPath` | string | `""` | override for the Python project (trusted workspaces only) |
| `kissSorcar.editorTabsMode` | boolean | `true` | chats as editor tabs instead of the sidebar view |
Agent settings (budget, worktree, API keys, ...) are NOT VS Code settings; they live in `~/.kiss/config.json` via the webview settings panel (`vscode-settings-panel`).

## First launch and widening
- workspaceState `firstLaunchDone`: on a genuine first launch (or when `~/.kiss/.extension-updated` exists) a 1 s timer closes the built-in auxiliary bar (sidebar mode) or reveals Task Info (editor-tabs mode), then focuses the chat composer.
- workspaceState `sidebarWidened`: first resolve of the sidebar chat view widens it to a third of the window once (sidebar mode only).
- Both timers capture `sidebarView` and re-check it after every `await`, because deactivation can clear the module slot mid-callback (an earlier version dereferenced `undefined` and rejected unhandled).

## Sources
- `src/kiss/agents/vscode/src/extension.ts` (`activate`, `deactivate`, `chatController`, `revealMetaView`)
- `src/kiss/agents/vscode/package.json` (`contributes`)
- `src/kiss/agents/vscode/src/SorcarSidebarView.ts` (`_getClient`)
- `src/kiss/agents/vscode/src/SorcarTab.ts` (`historyPanelBodyAttrs`, `metaPanelBodyAttrs`)
