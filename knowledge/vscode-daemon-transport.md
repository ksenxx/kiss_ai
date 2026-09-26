---
title: Extension-to-daemon transport (AgentClient over sorcar.sock)
uuid: fa0a30d2-dcc4-41c8-98e7-91ddd845561d
summary: AgentClient links VS Code to the kiss-web daemon over sorcar.sock (KISS_SORCAR_SOCK/KISS_HOME)
  - JSON lines, jittered reconnect backoff, 10s TTL queue, commandDropped.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Extension-to-daemon transport (AgentClient over sorcar.sock)

## Socket path
`sorcarSockPath()` in `src/userAssets.ts`: `$KISS_SORCAR_SOCK` if set, else `<kissHomeDir()>/sorcar.sock`, where `kissHomeDir()` is `$KISS_HOME` or `~/.kiss`. It must match the daemon's own resolution (`kiss/server/web_server.py` binds under `$KISS_HOME`); a mismatch made a window with `KISS_HOME` set kill a healthy daemon and poll a path it never binds. Tests point `KISS_SORCAR_SOCK` at a fake daemon (`test/fakeSock.js` `fakeSockPath` uses a named pipe on Windows).

## Wire format
Newline-delimited JSON both ways. `sendCommand(cmd)` writes `JSON.stringify(cmd) + '\n'`. Incoming data is decoded with a per-connection `StringDecoder('utf8')` (so multi-byte characters split across chunks survive), split on `\n`, and each line is emitted as a `'message'` event. Non-JSON lines are logged and skipped. A buffer whose string `.length` (UTF-16 code units, not bytes) exceeds 32 Mi (`MAX_LINE_BUFFER_BYTES`) drops the connection.

## Events emitted
`connect`, `disconnect`, `message` (a `ToWebviewMessage`), `commandDropped(cmd, reason)` with reason `'expired' | 'overflow'`.

## Queue while disconnected
- A command sent while the socket is not writable is queued with its timestamp and `connect()` is called.
- On connect, queued frames older than `PENDING_SEND_TTL_MS` (10 s) are dropped with `'expired'`; the rest are flushed **before** the `connect` event fires, so fresh commands issued by connect handlers (e.g. `getModels`) cannot overtake older queued ones (e.g. `selectModel`).
- Above `MAX_PENDING_SENDS` (256) the oldest are dropped with `'overflow'`.
- Why a TTL: after a long outage the daemon that answers is a different process; replaying a `run` into it would start an agent nobody asked for.

`SorcarSidebarView._handleDroppedCommand` undoes the optimistic UI: a dropped `run` removes the tab from `_runningTabs`, posts `status running:false`, and shows a warning; a dropped `generateCommitMessage` fires an error result; a dropped `openTab`/`resumeSession` of a panel's root tab raises the `registrationDropped` panel event.

## Reconnect backoff
`_scheduleReconnect`: delay = `min(base * 2^attempts, max)` with base 500 ms and max 15 s, then jittered to a uniform value in `[capped/2, capped]`. Attempts reset only if the previous connection lasted at least `STABLE_CONNECTION_MS` (5 s), so a crash-looping daemon that accepts and drops is not hammered. Every window runs its own clients against the same socket; a fixed retry used to produce connect storms while the daemon was trying to bind. `AgentClientOptions` (for tests) overrides only reconnect base/max, the pending TTL and the queue size; the 5 s stability threshold is hard-coded.

## Lifecycle rules
- Line-buffer state is reset per connection; events from a stale socket (`this._socket !== sock`) are ignored.
- `dispose()` uses `socket.destroy()`, not `end()` (a wedged daemon never reads), clears the queue and removes all listeners.
- `SorcarSidebarView._getClient` returns an already-disposed client after the controller was terminated, so a late caller cannot resurrect the connection.
- On connect the controller sends `setWorkDir`, and if it has a view: `getModels`, `getInputHistory`, `getConfig`, `getMyModels`. The sidebar controller also sends `getTabsState`.
- `ENOENT`/`ECONNREFUSED` errors are silent (daemon not up yet).

## SorcarApi
`src/SorcarApi.ts` is a typed wrapper (`run`, `stop`, `appendUserMessage`, `userAnswer`, `resumeSession`, `setWorkDir`, `selectModel`, `complete`, `worktreeAction`, `mainTreeAction`, `generateCommitMessage`, `autocommitAction`, `closeTab`, `serverReset`, generic `forward`), all ending in `client.sendCommand`.

## Sources
- `src/kiss/agents/vscode/src/AgentClient.ts` (`AgentClient`, `_scheduleReconnect`, `_handleData`)
- `src/kiss/agents/vscode/src/userAssets.ts` (`sorcarSockPath`, `kissHomeDir`)
- `src/kiss/agents/vscode/src/SorcarSidebarView.ts` (`_getClient`, `_handleDroppedCommand`)
- `src/kiss/agents/vscode/src/SorcarApi.ts`
- `src/kiss/agents/vscode/test/fakeSock.js`
