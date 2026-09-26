---
title: Server-canonical tab registry (tabs.json) shared by all clients
uuid: c319a592-3dee-4d0b-908a-a9dd398ef9df
summary: 'TabRegistry in KISS_HOME/tabs.json: shared tabs for all clients, one chat
  per tab, tabs_state snapshots, openTab/closeTab, 512 cap, openTabRejected, ready
  replay.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Server-canonical tab registry (tabs.json) shared by all clients

## What it is
Every client (VS Code chat webviews via the extension's UDS, remote web apps over WSS) shares ONE tab registry. The daemon's `TabRegistry` owns the canonical ordered list. Clients keep no tab set of their own: they reconcile against the full `tabs_state` snapshot broadcast after every mutation. A client may render only a workspace-filtered subset: `media/main.js` (`tabMatchesWorkspace`, `tabScopeWorkDir`, `isTabHidden`) hides tabs whose scope lies outside its workspace, but they stay in the registry and keep receiving events. Tabs are global, so a client disconnect never disposes a tab. Only an explicit `closeTab` does.

Entries hold `tabId`, `chatId`, `title` (capped at `_MAX_TITLE_CHARS = 200`), `workDir`, `scopeWorkDir` and `taskId`. `scopeWorkDir` pins a tab to a client workspace separately from `workDir` (e.g. a standalone API dispatch runs in a scratch dir but shows in the calling workspace); empty means clients fall back to `workDir`.

## Invariants
- **A chat id is bound to at most one tab.** Binding a chat to a tab atomically removes any other tab bound to that chat (the newest bind wins). Load and merge skip duplicate chat bindings. No client ever shows two tabs for one chat.
- Only top-level chat tabs are registered. Sub-agent tabs are derived (rebuilt from `openSubagentTab` broadcasts and replays), and content/editor tabs are client-local.
- At most `_MAX_TABS = 512` entries. An `openTab` over the cap is answered with `openTabRejected` to the sender. Without that answer the client would keep a local tab no one else sees. `open_tab` returns an `OpenTabOutcome` that separates "already exists" from "full" atomically. An unlocked `has_tab` re-probe once turned a benign re-announce into a spurious "Tab limit reached" when racing a close (D-RC2).
- **One owner per file.** The registry loads `tabs.json` once and writes its whole list on every save, so a second live registry on the same file would clobber it. One daemon per `KISS_HOME` (UDS liveness check) guarantees this. An embedded server sharing the home (the channel launcher's private-UDS daemon) uses `VSCodeServer.use_private_tab_registry`.

## Persistence
Writes are atomic (`kiss.core.utils.atomic_write_text`: unique temp file + `os.replace`). A failed save leaves unsaved state that `flush()` retries. `RemoteAccessServer.start()` calls `tab_registry.flush()` on shutdown.

## Generation tokens
`generation`, `republished_since`, `reopened_since`, `close_tab_if_generation`, `finalize_removal` and `stamp_unregistered` / `retire_unregistered` give each publication of a tab a token from a monotonic clock. Asynchronous teardowns can then tell whether the tab was reopened meanwhile, and they avoid removing a tab that a concurrent `resumeSession` just republished.

## Commands
- `openTab` (`_cmd_open_tab`): idempotent register, broadcasting `tabs_state`.
- `closeTab` (`_cmd_close_tab`): backend cleanup for a closed tab. It raises the state's `frontend_closed`, and the state is disposed when no run or merge needs it.
- `getTabsState`: the current snapshot.
- `ready` (`_handle_ready`): `merge_if_empty` adopts a legacy client's `restoredTabs` only when the registry is empty, then replays chat-bound tabs (`bound_tabs`) to the connecting client.

## Related: local UDS talk visibility
Which tabs a local webview "shows" (used to route `talk` audio) is decided at event time from canonical facts by `VSCodeServer._local_tab_shown`, not by mirroring registry state into per-connection sets. The mirror approach had unfixable races between close, `ready` and `resumeSession`.

## Sources
- `src/kiss/server/tab_registry.py` (`TabRegistry`, `OpenTabOutcome`, `_MAX_TABS`, `_MAX_TITLE_CHARS`)
- `src/kiss/server/commands.py` (`_cmd_open_tab`, `_cmd_close_tab`)
- `src/kiss/server/web_server.py` (`_handle_ready`)
- `src/kiss/server/server.py` (`_local_tab_shown`)
