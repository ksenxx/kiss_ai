---
title: Recurring bug classes in the VS Code extension and chat webview
uuid: 2a84b053-d022-41ba-9f5a-338cec14b909
summary: Recurring VS Code extension/webview bugs - stale async after deactivate,
  shared poster ownership, dropped queued commands, duplicate editor tabs, kiss-web
  drift, cached icons.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Recurring bug classes in the VS Code extension and chat webview

Each entry: the pattern, where it bit, and the fix now in the code. Check new code against this list.

## Host (TypeScript) side
1. **Async continuation outlives its activation.** Timers and `.then` callbacks in `activate()` ran after `deactivate()` cleared module slots and dereferenced `undefined`. Fix pattern: capture the object (`const view = sidebarView`) and re-check `sidebarView !== view` / `panelManager !== revealManager` after every `await` (first-launch auto-open, sidebar widening, Task Info reveal in `extension.ts`). Also check whether the mode flipped meanwhile (`editorTabsMode()`).
2. **Work after dispose.** A webview message queued before `dispose()` could rebuild the daemon client. Fix: `_terminated` guard at the top of `_handleMessage`, and `_getClient()` hands out an inert disposed client after termination.
3. **Shared singleton stolen or cleared by the wrong owner.** The toast poster is shared by the sidebar and the panel manager; an unconditional clear on one surface's teardown silenced the other. Fix: identity-checked `clearWebviewNotificationPoster(p)`; per-panel controllers never install a poster (`vscode-webview-notifications`).
4. **Optimistic UI never undone.** The webview shows "running" when a `run` is sent; if `AgentClient` drops the queued command (TTL 10 s or overflow), nothing ever sent `status running:false`. Fix: `commandDropped` event -> `_handleDroppedCommand` (`vscode-daemon-transport`). Queued commands also expire so a `run` is not replayed into a restarted daemon.
5. **Many windows hammer one daemon.** Fixed-interval reconnects from every window produced connect storms during daemon restarts. Fix: exponential backoff with jitter, reset only after a 5 s stable connection.
6. **Duplicate or lost editor tabs across reloads.** Serialized placeholders are invisible until revived, so registry adoption could open a second tab for the same chat. Fix: `hasChatEditorTab()` counts tabGroups placeholders, the first snapshot is filtered by workspaceState `kissSorcar.editorPanelTabIds` (updated per tab, not as a whole set), and `_pendingAdoptions` handles claims that never bind (`vscode-editor-tabs-mode`).
7. **Cross-target writes.** SCM commit messages from one repository landed in another's input box, and stale countdown ticks overwrote newer text. Fix: per-repository `scm:<root>` ids and a serialized `scmWriteChain` (`vscode-scm-commit-message`).
8. **Two processes, two paths.** `KISS_HOME` set in the window but not in the daemon (or the update terminal) made the extension watch or probe a different socket/marker than the one written. Fix: `sorcarSockPath()` mirrors the daemon; `runUpdate` pins `KISS_HOME`; the systemd unit and launchd plist propagate `KISS_HOME`.
9. **Lost deferred work.** A daemon restart deferred for active tasks had no durable record after the update marker was deleted. Fix: `~/.kiss/.kiss-web.restart-pending` plus a retry timer (`vscode-dependency-installer-and-daemon-restart`).
10. **Hung child processes.** A probe that ignores the default SIGTERM can block activation. Fix: explicit timeouts with `killSignal: 'SIGKILL'` (`findUvPath` in `kissPaths.ts`), process-group SIGKILL for installer steps, and SIGTERM-then-SIGKILL (after 3 s) when freeing port 8787 (`DependencyInstaller.ts`).

## Webview (`media/main.js`) side
11. **Stale replies.** Async replies (`dirListing`, `gitStatus`, `gitLog`, history pages) can arrive after the user moved on. Fix pattern: send a token or generation with the request and drop replies whose token is not current (`test/activityBarViews.test.js`).
12. **Surface drift.** VS Code and the remote webapp build the same `chat.html`; placeholders added on one side only render as raw `{{KEY}}` on the other, and a daemon running old Python can serve new media. Keep `buildChatHtml` and `web_server._build_html` in sync (`vscode-remote-webapp-relation`).
13. **Automatic panel collapse hiding content.** Task finish and replays collapsed or hid live panels and `/ask` answers. Fixes (commits "stop task-finish from collapsing or hiding live event panels", "never auto-collapse or hide /ask answer panels"): live finishes leave panels as streamed; `answerPanelStaysOpen()` exempts answers from every automatic pass (`collapseOlderPanels`, `applyChevronState`).
14. **DOM rebuilds swallowing input.** History list refreshes replaced rows under the pointer and ate clicks and focus (commit "prevent history list clicks/focus from being swallowed by rebuilds").
15. **Unbounded work in no-folder windows.** The `@` file picker scanned the whole disk when the window had no folder (commit "stop @-mention file picker from scanning the whole disk in no-folder windows"); a filesystem root (`/`, `C:\`) is never treated as a real workspace, neither in `main.js` nor in `SorcarSidebarView.ts`.
16. **Prompt heuristics.** A one-line prompt naming an existing file opens it instead of running a task; it must be a regular file (`_resolveTabFile(..., fileOnly=true)`), otherwise prompts like "src" or "tmp" would reveal directories instead of starting tasks.

## Stale-cache class
17. **Icons cached for a year** by the VS Code server under a version-only URL. Fix: content-hashed icon names in the VSIX (`vscode-packaging-and-branding`). Webview assets already carry `?v=<sha256>`.

## Sources
- `src/kiss/agents/vscode/src/extension.ts`, `src/SorcarSidebarView.ts` (`_handleMessage`, `_getClient`, `_handleDroppedCommand`, `_resolveTabFile`)
- `src/kiss/agents/vscode/src/AgentClient.ts`, `src/WebviewNotifications.ts`, `src/SorcarPanelManager.ts`, `src/kissPaths.ts`, `src/DependencyInstaller.ts`
- `src/kiss/agents/vscode/media/main.js` (`answerPanelStaysOpen`, `collapseOlderPanels`, `applyChevronState`)
- `git log --oneline -- src/kiss/agents/vscode` (fix commits quoted above)
