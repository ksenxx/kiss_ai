---
title: In-webview notifications and progress toasts (WebviewNotifications.ts)
uuid: 92fdfb85-445e-40a0-a5c1-79c254c9ed92
summary: WebviewNotifications.ts toasts inside the chat webview - single poster, showInformationNotification,
  withWebviewNotificationProgress, notificationAction, native fallback.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# In-webview notifications and progress toasts (WebviewNotifications.ts)

The extension does not call `vscode.window.show*Message` directly for its own notifications. It calls the wrappers in `src/WebviewNotifications.ts`, which render a toast inside the chat webview when one is available and fall back to native notifications otherwise.

## API
- `showInformationNotification(message, ...items)`, `showWarningNotification`, `showErrorNotification`: `items` are action strings and an optional `MessageOptions` object, like the native API. Returns a promise of the chosen action (or `undefined`).
- `withWebviewNotificationProgress(options, task)`: for `ProgressLocation.Notification`, posts a sticky progress toast and forwards `progress.report({message})` updates; other locations, or no poster, use `vscode.window.withProgress`.
- `resolveWebviewNotificationAction(id, action)`: called by `SorcarSidebarView` on the webview's `notificationAction` message.

## The poster
A module-level `poster` function sends `{type: 'notification', id, severity, message, actions, sticky, progress?, progressMessage?}` to one webview. A toast is sticky when it is modal or has actions. Actions resolve through `pendingActions` (id -> resolver).
- The sidebar chat controller (no `panelHooks`) installs its poster in `attachWebviewHost` and clears it on webview dispose.
- In editor-tabs mode `SorcarPanelManager._refreshPoster` installs one poster that routes to the currently active panel, and releases it when the last panel closes. Per-panel controllers never install their own, otherwise they would steal each other's toasts.
- `clearWebviewNotificationPoster(p)` clears only if `p` is the installed poster. An unconditional clear let the sidebar webview's late dispose silence the poster the panel manager had just installed.
- Replacing or clearing the poster resolves all pending actions with `undefined`, so callers awaiting a choice never hang.

## Gotchas
- Code that awaits an action from a toast must handle `undefined` (poster replaced, webview closed).
- Progress toasts must be settled on every path. A past bug left a fallback timer alive after settlement; the fix clears the timeout inside the idempotent settle function (see `_showActionProgress` in `SorcarSidebarView.ts` and `test/actionProgressTerminalPaths.test.js`).
- Messages use the `PRODUCT_NAME` prefix from `brand.ts` instead of a hard-coded name.

## Sources
- `src/kiss/agents/vscode/src/WebviewNotifications.ts` (`setWebviewNotificationPoster`, `clearWebviewNotificationPoster`, `showNotification`, `withWebviewNotificationProgress`, `resolveWebviewNotificationAction`)
- `src/kiss/agents/vscode/src/SorcarPanelManager.ts` (`_refreshPoster`)
- `src/kiss/agents/vscode/src/SorcarSidebarView.ts` (`attachWebviewHost`, `case 'notificationAction'`, `_showActionProgress`)
