---
title: Daemon wire protocol, API catalog and ServerApi.dispatch
uuid: c37669a1-9537-45a4-bb3f-e54ef9dc016a
summary: 'JSON command protocol over UDS (NDJSON) and WSS: API catalog, ApiCommand
  forward/drop handlers, ServerApi.dispatch stamping connId/tabId/workDir, ready handshake.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Daemon wire protocol, API catalog and ServerApi.dispatch

## Framing
Both transports carry the same JSON objects, dispatched on the `"type"` field:
- **UDS** (`RemoteAccessServer._uds_handler`): newline-delimited JSON lines. The VS Code extension host uses it (`src/AgentClient.ts`, wrapped by `src/SorcarApi.ts`), relaying its chat webview's `postMessage` traffic, and so do Python clients (`daemon_client`). No password.
- **WSS** (`_ws_handler`): one JSON object per WebSocket frame on `/ws`. Only the remote browser webapp uses it. The first message must be the `auth` handshake (see `server-remote-access-auth`).

The frame limit is `_MAX_LINE_BYTES = 64 MiB` in `web_server.py`. `daemon_client._MAX_LINE_BYTES` must equal it. A smaller client cap splits oversized lines (for example a `system_prompt` event carrying all of SYSTEM.md), and when the split frame is the terminal `result`, a successful task reads as an empty failure.

Each connection gets a fresh `conn_id` (uuid hex) and a per-connection `work_dir`. The UDS writer is registered with the printer via `bind_conn`.

## The catalog (`kiss.server.sorcar.API`)
`API` is a `dict[str, ApiCommand]` (built by `_catalog`) mapping each name to its `ApiCommand(name, required=(...), handler=...)`, the single source of truth for the wire API (about 70 commands). `handler` is one of:
- `"forward"` (the default): routed unchanged to the backend `VSCodeServer`, whose `_HANDLERS` table in `commands.py` maps names to `_cmd_*` methods (`run`, `stop`, `interruptTool`, `userAnswer`, `appendUserMessage`, `resumeSession`, `openTab`, `closeTab`, `getTabsState`, `newChat`, `complete`, `worktreeAction`, `setWorkDir`, `getConfig`, `saveConfig`, ...).
- a `ServerApi` method name (for example `ready`, server reset, update, voice wake start/stop).
- `"drop"`: accepted and discarded. These messages are consumed by the extension host or the WSS handshake (`auth`, `voiceToggle`, `voiceSensitivity`, `voiceAck`).

`validate_command(cmd)` returns `None` for a valid command, or an error string for an unknown `type` or a missing required field.

## `ServerApi.dispatch(cmd, ctx)`: the only router
The transport (`_dispatch_client_command`) never routes by itself. `dispatch` does, in order:
1. Silently drops `DROPPED_COMMANDS` before validation.
2. Strips whitespace from a string `tabId` and writes it back, so every registry keys the tab identically.
3. Validates the command. An invalid one gets a direct `{"type":"error","text":...}` sent only to that connection.
4. Records the `tabId` in the connection's bookkeeping (`_record_tab`).
5. Stamps `connId` = the connection's `conn_id`, overwriting any client value (anti-spoofing). The stamp keys per-connection state and reply routing.
6. Keeps the per-window work_dir invariant. `setWorkDir` updates the connection's `work_dir`. Any other command without a usable `workDir` gets it stamped. A `workDir` that is a filesystem root (`/`, `C:\`, per `kiss.core.utils.is_root_dir`) is blanked first, so a client can never pin or run against the whole disk. This fixed a bug where root work dirs poisoned the daemon.
7. Invokes the catalog's handler.

## Reply routing conventions
- Events stamped with a non-empty `connId` are request/replies (`models`, `history`, `files`, `configData`, error): they go ONLY to the requesting connection, so one VS Code window never changes another's UI.
- Events with a `tabId` are addressed to a tab. Clients filter by `tabId`.
- Task events carry `taskId` and are fanned out per subscribed tab (see `server-broadcast-routing`).

## The `ready` handshake
A (re)connecting client sends `ready`. `_handle_ready` fans it out into `getModels` / `getInputHistory` / `getConfig`, each connId-stamped. It adopts the client's legacy `restoredTabs` only into an EMPTY registry (a one-time migration), broadcasts a `tabs_state` snapshot, and replays every chat-bound registry tab to that client only (`replayConnId`). A client that mirrors one tab (a VS Code editor-tab panel, `singleTabId`) gets only that tab's replay.

## Adding a command
Add an `ApiCommand` to `API` in `sorcar.py`. Add the handler (a `_cmd_*` entry in the `commands.py` `_HANDLERS` table for `forward`, or a `ServerApi` method). Add the client facade method in `media/api.js` / `src/SorcarApi.ts` (VS Code area).

## Sources
- `src/kiss/server/sorcar.py` (`API`, `ApiCommand`, `validate_command`, `ServerApi.dispatch`, `DROPPED_COMMANDS`)
- `src/kiss/server/web_server.py` (`_uds_handler`, `_ws_handler`, `_dispatch_client_command`, `_handle_ready`, `_MAX_LINE_BYTES`)
- `src/kiss/server/commands.py` (`_HANDLERS`)
- `src/kiss/agents/sorcar/daemon_client.py` (`_MAX_LINE_BYTES`)
