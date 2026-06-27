# Progress

## Task

Show "Restarting the KISS Sorcar web server…" as a notification (top-right
toast) instead of a chat-output note in the chat webview.

## Implementation (claude-opus-4-7)

### `src/kiss/agents/vscode/web_server.py`

`_handle_server_reset` (and the `_SERVER_RESET_DELAY` docstring) now
broadcasts a `notification` event with a stable id, severity, and
message instead of the old `notice` event:

```py
notification: dict[str, Any] = {
    "type": "notification",
    "id": "server-reset-restarting",
    "severity": "info",
    "message": "Restarting the KISS Sorcar web server…",
}
if conn_id:
    notification["connId"] = conn_id
self._printer.broadcast(notification)
loop.call_later(_SERVER_RESET_DELAY, self._trigger_server_reset)
```

`broadcast()` strips `connId` and routes the notification to ONLY the
requesting connection via `_send_to_conn`, so sibling windows do not
pop a banner.

### `src/kiss/tests/agents/vscode/test_server_reset.py`

- `test_server_reset_acks_and_triggers_restart` now asserts `type == "notification"`, `id == "server-reset-restarting"`, `severity == "info"`, the message contains "restart" / "web server", and `type != "notice"` as an explicit regression guard.
- `test_server_reset_notification_only_to_requesting_window` (renamed
  from `_notice_only_…`) checks the `notification` event does not leak
  to sibling windows.

### `src/kiss/agents/vscode/test/serverResetRestartingNotification.test.js` (new)

JSDOM-driven webview contract test that pins three properties of the
new behaviour:

1. Dispatching `{type:"notification", id:"server-reset-restarting", severity:"info", message:"Restarting the KISS Sorcar web server…"}`
   renders a `.kiss-notification` toast whose text contains the
   message, whose `data-notification-id` is
   `server-reset-restarting`, and whose class set includes
   `kiss-notification-info`.
1. No `div.note` is appended in the chat output.
1. Control case: a legacy `notice` event with the same text DOES still
   render as a chat-output `div.note` and does NOT raise a
   `.kiss-notification` toast — proves the absence of a note in (1) is
   meaningful, not a false-positive from a webview that lost the
   `notice` code path.

### `src/kiss/agents/vscode/package.json`

`serverResetRestartingNotification.test.js` is added to the `test`
script so it runs as part of the extension's regular `npm test` and
`uv run check --full` cycles.

## Verification

- `uv run pytest -q src/kiss/tests/agents/vscode/test_server_reset.py`:
  2 passed.
- `node test/serverResetRestartingNotification.test.js`: passed.
- `node test/serverResetConfirm.test.js`,
  `serverResetDialog.test.js`, `serverResetDialogDoubleClick.test.js`,
  `serverResetDialogGuardBypass.test.js`,
  `webviewNotifications.test.js`, `updateNotification.test.js`: all
  pass.
- `uv run check --full`: all checks pass (ruff, mypy, pyright,
  VS Code TS check + lint, mdformat).

## Review (gpt-5.5)

- Field names match `showNotification(ev)` in `media/main.js` (which
  reads `ev.id`, `ev.severity`, `ev.message`).
- Stable `id="server-reset-restarting"` prevents duplicate stacking if
  the daemon ever re-broadcasts the event.
- Targeted delivery via `connId` is preserved (`broadcast()` already
  routes to a single connection when `connId` is set), so siblings are
  unaffected — pinned by
  `test_server_reset_notification_only_to_requesting_window`.
- The `notice` event handler in `main.js` is intentionally left
  untouched: other features still rely on it; the control case in the
  new JS test exercises it to confirm.
- Auto-dismiss of an `info` toast is acceptable here — the daemon
  SIGTERMs itself after `_SERVER_RESET_DELAY` (0.4 s), and the
  notification flushes to the client before the socket drops.
- Regression guards in the integration test (`assertNotEqual(..., "notice")`) prevent a future refactor from silently restoring the
  buried chat-output banner.
