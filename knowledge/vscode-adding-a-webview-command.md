---
title: Checklist - adding a webview-to-daemon command and reply event
uuid: c8e1798f-93d3-41ad-a3b3-f5252c9d3450
summary: Checklist for a new webview-to-daemon command - sorcar.py catalog, commands.py
  handler, api.js SORCAR_API_COMMANDS, types.ts unions, FORWARDED_COMMANDS, main.js
  case, tests.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Checklist - adding a webview-to-daemon command and reply event

A command sent by the chat webview reaches the daemon over two transports: the Unix socket (VS Code, via `SorcarSidebarView`) and the WebSocket (remote webapp, via the shim in `web_server.py`). Both end in the daemon's `ServerApi.dispatch`, so one command definition serves both surfaces.

## Daemon side (other area, listed for completeness)
1. `src/kiss/server/sorcar.py`: add the command to the API catalog (`ApiCommand("name", required=(...))`).
2. `src/kiss/server/commands.py`: implement `_cmd_<name>` and register it in `_HANDLERS`. Broadcast a reply that every window should repaint; send errors only to the sender.

## Webview/extension side (this area)
3. `media/api.js`: append the name to `SORCAR_API_COMMANDS`. This creates `api.<name>(fields)`; `api.send` throws for names not in the list.
4. `src/types.ts`: add the webview message to `FromWebviewMessage`, the reply to `ToWebviewMessageBody`, and the command to `AgentCommand` (plus any new optional fields).
5. `src/SorcarSidebarView.ts`: add `name: ['field1', 'field2']` to `FORWARDED_COMMANDS` if the host only needs to pass fields through. Only commands that need host logic (file resolution, VS Code APIs, running-state bookkeeping) get a `case` in `_handleMessage`. Replies need no host change: daemon messages are relayed to the webview without an allow-list.
6. `media/main.js`: send with `api.<name>({...})` and handle the reply in the message switch (`case '<event>':`).
7. If the message is host-only (asks VS Code to do something, never reaches the daemon), use `postToHost` in `main.js`, add it to `FromWebviewMessage`, handle it in `_handleMessage`, and do NOT add it to `api.js`.

## Tests
- `test/sorcarApi.test.js` checks the real webview only sends catalog commands through the api object.
- `src/kiss/tests/agents/vscode/test_server_api.py` checks the `api.js` list against the daemon catalog.
- jsdom webview tests can use `test/simplify2_harness.js` (`makeWebview`, `send(win, event)`, `sleep`) to load `chat.html` + `main.js` and capture posted messages. Posted objects come from the jsdom realm, so compare them via `JSON.parse(JSON.stringify(x))` rather than `assert.deepStrictEqual` on the raw object.

## Sources
- `src/kiss/agents/vscode/media/api.js` (`SORCAR_API_COMMANDS`)
- `src/kiss/agents/vscode/src/SorcarSidebarView.ts` (`FORWARDED_COMMANDS`, `_handleMessage`, `_installClientListener`)
- `src/kiss/agents/vscode/src/types.ts` (`FromWebviewMessage`, `ToWebviewMessageBody`, `AgentCommand`)
- `src/kiss/agents/vscode/test/sorcarApi.test.js`, `src/kiss/agents/vscode/test/simplify2_harness.js`
- `src/kiss/tests/agents/vscode/test_server_api.py`
