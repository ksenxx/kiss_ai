# Task

When the user clicks the "Server reset" button in the settings panel
while an agent is running, show a native VS Code modal **dialog**
(with OK and Cancel buttons) asking whether to restart the server.
Only restart on OK. The previous fix used an in-webview toast
notification, which the user reports "didn't work" — so replace the
toast with a real VS Code modal dialog and add an integration test
that drives both the webview and the compiled extension.

Workflow constraint:

- Use `claude-opus-4-7` for coding, bug-fixing, test creation.
- Use `gpt-5.5` (NOT codex) for a thorough review at the end.

## Plan

1. Webview (`media/main.js`) — replace the toast confirmation with a
   single `postMessage({type: 'serverReset', agentRunning})`. The
   webview already owns the per-tab `isRunning` state.
1. Type (`src/types.ts`) — add optional `agentRunning?: boolean` to
   the `serverReset` message variant.
1. Extension (`src/SorcarSidebarView.ts`) — when `agentRunning` is
   true, call
   `vscode.window.showWarningMessage(msg, {modal: true}, 'OK')`
   (modal mode auto-adds the Cancel button). Only forward
   `{type: 'serverReset'}` to the daemon when the user chose OK.
1. Integration test (`test/serverResetDialog.test.js`) — drive the
   real compiled `SorcarSidebarView` against a UDS daemon stub and a
   real JSDOM-rendered `media/main.js`. Verify:
   - With a running agent + user picks OK → daemon receives
     `{type: "serverReset"}`.
   - With a running agent + user picks Cancel → daemon does NOT
     receive any reset command.
   - With no running agent → daemon receives the reset immediately
     and the dialog is never raised.
1. Update the existing `serverResetConfirm.test.js` to match the new
   contract (webview posts the message directly; no toast).
1. Run `npm run test` and `uv run check --full`.
