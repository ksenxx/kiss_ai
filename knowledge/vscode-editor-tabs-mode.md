---
title: Editor-tabs mode (SorcarPanelManager, chat editor tabs)
uuid: 6461759d-3a5f-48c0-b137-2b12119526bf
summary: kissSorcar.editorTabsMode chat editor tabs (kissSorcar.chatTab) - SorcarPanelManager
  serializer revival, tabs_state registry adoption, one-chat invariant, spinner icon,
  mode switch.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Editor-tabs mode (SorcarPanelManager, chat editor tabs)

`kissSorcar.editorTabsMode` (default `true`) moves chats from the secondary-sidebar Chat view into editor tabs. Each tab is a `WebviewPanel` with viewType `kissSorcar.chatTab` (`CHAT_PANEL_VIEW_TYPE`), created with `retainContextWhenHidden: true` and `localResourceRoots` `media/` and `out/`. Each panel gets its own `SorcarSidebarView` controller with `panelHooks` (`rootTabId`, `onEvent`) and its own daemon connection. `SorcarPanelManager.modeEnabled()` reads the setting.

## Chat identity and the daemon registry
The daemon keeps the canonical chat-tab registry and broadcasts `tabs_state` snapshots. Rule: one tab per chat; the newest bind wins and displaces the older tab. A panel tracks `tabId`, `chatId`, and `registryBound` (a snapshot confirmed the binding).

- `enterMode(entries, workspaceDir)`: on switch-on, open a panel per registry tab scoped to this workspace (`isDirInside(scopeWorkDir || workDir, workspaceDir)`), then `ensureChatOpen()`.
- `adoptRegistryTabs(added, workspaceDir, listed)`: tabs another client created (remote webapp, other window) open as background panels. A chat already shown by a panel is skipped unless the registry displaced that panel's tab. A skip caused by an unconfirmed claim is remembered in `_pendingAdoptions` and adopted later if the claim is dropped (`registrationDropped` event) or the panel closes.
- On activation, `extension.ts` filters the first snapshot through workspaceState `kissSorcar.editorPanelTabIds` so tabs that serialized placeholders will revive are not opened twice. The record is updated per tab (`recordPanelTab(tabId, open)`), not as a whole set, because revival happens one panel at a time.

## Serializer revival
`registerSerializer()` revives panels after a reload using the webview state field `editorRootTabId` (persisted by `main.js` `persistTabState`). A revived panel whose id is already open is disposed; if the mode was switched off meanwhile, the panel is disposed. `dispose()` (teardown) disposes controllers but leaves panels standing so the workbench can persist them.

## One-chat invariant
There is always at least one chat editor tab, like the sidebar strip always keeps one tab. `ensureChatOpen({preserveFocus})` opens one when `hasChatEditorTab()` is false. `hasChatEditorTab()` counts live panels and restored placeholders from `vscode.window.tabGroups` (`isChatEditorTab`). It is called after every panel dispose, on `enterMode`, at activation, on history-view reveal, and from `watchEditorTabs()` (a `tabGroups.onDidChangeTabs` backstop for placeholders closed before revival). It is a no-op while `_shuttingDown` or `_closingAll`.

## Closing and retiring
- A user close retires the chat from the registry through the sidebar controller's long-lived client (`_retireTab`), since the panel's own client dies with the panel.
- `closeAll()` (mode switched off) sets `suppressCloseTab` so chats stay registered and reappear in the sidebar view.
- `closeSelf` event: `retire` means the user closed the root chat inside the webview; without it, the registry dropped the tab elsewhere.

## Status icon and title
`_applyPanelTitle`: while running, `panel.iconPath` is `media/spinner-running.svg` (an SVG with SMIL animation, since `ThemeIcon` spin is not supported for webview tab icons on engine `^1.98`), else `kiss-icon.svg`; finished tasks get a `✅ `/`❌ ` title prefix. Icon paths go through `mediaIconPath` (hashed names in a VSIX). `STATUS_PREFIX_RE` strips legacy prefixes from revived titles.

## Mode switch (extension.ts)
- ON: `syncEditorActionsLocation(context, true)` moves the four editor-title buttons into the window title bar (setting `workbench.editor.editorActionsLocation` = `titleBar`, prior value saved under globalState `kissSorcar.priorEditorActionsLocation`), `enterMode`, reveal Task Info.
- OFF: restore the actions location, `closeAll()`, focus the sidebar chat composer.
- The webview's settings toggle posts `setEditorTabsMode`; the host writes to the most specific configuration scope that already has a value, so a workspace override does not swallow the toggle.

## Task Info and history relays
Panels post `metaUpdate` and `activeTask`; the manager forwards the ACTIVE panel's values to the Task Info view (`setMetaSink`) and to the history view highlight (`setActiveTaskSink`). `openChat(event)` reveals the panel already bound to a chat or creates one resuming it; `autoSubmit` (Apps subpanel "connect" launch) submits once after `ready` via `submitWhenReady`.

## Sources
- `src/kiss/agents/vscode/src/SorcarPanelManager.ts` (`SorcarPanelManager`, `adoptRegistryTabs`, `ensureChatOpen`, `registerSerializer`, `_applyPanelTitle`, `_onPanelEvent`, `openChat`)
- `src/kiss/agents/vscode/src/extension.ts` (`onDidChangeConfiguration` handler, `onRegistryTabsState` wiring, `PANEL_TAB_IDS_KEY`)
- `src/kiss/agents/vscode/src/editorActionsLocation.ts` (`syncEditorActionsLocation`, `PRIOR_LOCATION_KEY`)
- `src/kiss/agents/vscode/src/SorcarSidebarView.ts` (`case 'setEditorTabsMode'`)
- `src/kiss/agents/vscode/test/editorTabsAlwaysOneChat.test.js`
