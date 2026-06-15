# Task: CLI events should stream into an open chat webview

## Root cause (confirmed)

CLI uses `RecordingConsolePrinter` (extends `JsonPrinter`): records + persists events to the chat DB but had NO transport to the running daemon. The daemon's `WebPrinter` is the only thing that fans events out over WSS / UDS to webviews.

## Status — COMPLETE

### Fix

1. NEW `src/kiss/agents/sorcar/cli_daemon_bridge.py` — cached AF_UNIX socket, `send_event(event)` writes newline JSON `{"type":"cliEvent","event":event}`, env override `KISS_SORCAR_SOCK`, `reset_for_tests()`. Silent on connect/write errors with one-shot retry.
1. EDITED `src/kiss/agents/sorcar/cli_printer.py` — `RecordingConsolePrinter.broadcast()` calls `super().broadcast(event)` then forwards a task-id-injected copy to `cli_daemon_bridge.send_event` when a taskId is present.
1. EDITED `src/kiss/agents/vscode/web_server.py`:
   - Added `RemoteAccessServer._relay_cli_event(ev)` that fans the event out via `self._printer._fanout_targets` + `_send_to_ws_clients`, mirroring the tail of `WebPrinter.broadcast`. Does NOT re-record/re-persist — CLI already did both.
   - Added `cliEvent` short-circuit branch in `_dispatch_client_command` before `setWorkDir` handling.

### Tests

NEW `src/kiss/tests/agents/sorcar/test_cli_daemon_live_stream.py` (2 tests, both PASS):

1. `test_cli_event_reaches_subscribed_webview_live` — end-to-end repro: stands up a real `RemoteAccessServer` UDS endpoint on a temp socket, points `KISS_SORCAR_SOCK` at it, subscribes a viewer tab, opens a viewer UDS client, then a `RecordingConsolePrinter` (the same class the CLI installs) broadcasts a `text_delta`. Asserts the viewer receives the event JSON line with `tabId="tab-viewer"` stamped within 3s. This is the bug the user reported.
1. `test_daemon_does_not_double_record_cli_event` — asserts the daemon's `WebPrinter._recordings[task_id]` is unchanged before/after the CLI broadcast (only the CLI process records & persists; the daemon is purely a transport).

### Verification

- `uv run check --full` → all checks pass.
- Related test groups (`test_cli_chat_webview_events`, `test_cli_panel`, `test_replay_event_coalescing`, `test_chat_viewer_live_stream`) re-run together → 22/22 PASS.
