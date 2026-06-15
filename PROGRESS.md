# Task: CLI tasks must be resumable from history with live streaming + blinking green circle

## Status: DONE

Implemented and verified end-to-end. `uv run check --full` is green
and the new integration test
`test_cli_history_click_resumes_live_stream.py` passes.

## What was wrong

The previous fix only relayed CLI events to webview tabs that were
ALREADY subscribed to the task id. In production no tab is
subscribed when a CLI task starts: the user later clicks the task
in the History sidebar, which routes through
`VSCodeServer._replay_session` → `_reattach_running_chat`. The
reattach scans `_RunningAgentState.running_agent_states`, which is
only populated for UI-launched tasks. CLI tasks never have an
entry, so the new viewer tab got no subscription and no
`status:running=true` broadcast — events did not stream live and
the tab title did not blink green.

## Fix

### `src/kiss/agents/sorcar/cli_daemon_bridge.py`

Refactored the UDS write path into `_send_envelope(env)` and added
two lifecycle helpers reused by the CLI printer:

```python
def send_cli_task_start(task_id: int) -> None: ...
def send_cli_task_end(task_id: int) -> None: ...
```

Each writes `{"type": "cliTaskStart"|"cliTaskEnd", "taskId": <int>}`
on the cached UDS connection, with the same silent-on-failure
semantics as `send_event`.

### `src/kiss/agents/sorcar/cli_printer.py`

`RecordingConsolePrinter.__init__` now tracks per-process running
task ids (guarded by a lock) and registers an `atexit` safety net
that closes any task ids still marked running. `broadcast` now:

1. Calls `super().broadcast(event)` to record + persist as before.
1. On the FIRST event seen for a fresh integer `taskId`, calls
   `cli_daemon_bridge.send_cli_task_start(task_id)`.
1. Forwards the event via `cli_daemon_bridge.send_event(...)`.
1. On a terminal `result` event, calls
   `cli_daemon_bridge.send_cli_task_end(task_id)`.

`_cli_atexit_end_all` covers Ctrl+C / crash / uncaught-exception
paths so the daemon's running-set does not leak forever.

### `src/kiss/agents/vscode/web_server.py`

`RemoteAccessServer.__init__` now owns the CLI-running map:

```python
self._cli_running_tasks: set[int] = set()
self._cli_running_lock = threading.Lock()
self._vscode_server.set_cli_running_lookup(self._is_cli_task_running)
```

New methods:

- `_is_cli_task_running(task_id)` — thread-safe lookup hook.
- `_handle_cli_task_start(task_id, conn_state)` — adds the task to
  the running set AND to a per-connection `cli_tasks` set so a
  dropped UDS connection (CLI crash) cleans them up.
- `_handle_cli_task_end(task_id, conn_state)` — drops the task and
  calls `_fanout_cli_status(task_id, running=False)` so every
  webview subscribed to it stops blinking green.
- `_fanout_cli_status(task_id, *, running)` — mirrors the
  per-tab fan-out pattern of `WebPrinter.broadcast`.

`_dispatch_client_command` got two new short-circuits before the
existing `cliEvent` branch:

```python
if cmd_type == "cliTaskStart": ...   # → _handle_cli_task_start
if cmd_type == "cliTaskEnd":   ...   # → _handle_cli_task_end
```

The UDS handler's `finally` block now also drains the
per-connection `cli_tasks` set via `_handle_cli_task_end`, so a
CLI process that drops without sending `cliTaskEnd` (Ctrl+C /
SIGKILL) cannot leak running task ids.

### `src/kiss/agents/vscode/server.py`

`VSCodeServer.__init__` gained `_cli_running_lookup: Callable[[int], bool] | None`
(installed by `RemoteAccessServer` via `set_cli_running_lookup`).
`_replay_session` consults it after `_reattach_running_chat`
returns False:

```python
if (
    not rebound_running
    and rebound_task_id is not None
    and self._cli_running_lookup is not None
    and self._cli_running_lookup(rebound_task_id)
):
    self.printer.subscribe_tab(str(rebound_task_id), tab_id)
    rebound_running = True
```

The existing `if rebound_running:` block then broadcasts
`status:running=true` with the new tab's `tabId`, which is what
makes the tab title show the blinking green circle. Subsequent
`cliEvent` relays now fan out to this subscribed tab.

## Integration test

`src/kiss/tests/agents/sorcar/test_cli_history_click_resumes_live_stream.py`
drives the production code path end-to-end against a real
`RemoteAccessServer` on a temp UDS:

- `test_history_click_subscribes_tab_and_starts_indicator`:
  CLI announces start → fresh viewer opens and `_replay_session`
  is invoked with the running task_id → asserts the viewer is
  subscribed, received `status:running=true` with its tabId
  (green circle), then receives a live relayed `text_delta`
  with its tabId, and finally receives `status:running=false`
  on `result`.
- `test_uds_disconnect_cleans_up_stale_cli_tasks`:
  Connection drop after `cliTaskStart` (no matching End) must
  clear the running set via the UDS handler's `finally` block.

## Verification

```
$ uv run check --full   # ✅ all stages
$ uv run pytest src/kiss/tests/agents/sorcar/test_cli_daemon_live_stream.py \
    src/kiss/tests/agents/sorcar/test_cli_history_click_resumes_live_stream.py \
    src/kiss/tests/agents/sorcar/test_cli_chat_webview_events.py \
    src/kiss/tests/agents/vscode/test_chat_viewer_live_stream.py \
    src/kiss/tests/agents/vscode/test_detach_tab_and_reattach.py \
    src/kiss/tests/agents/vscode/test_multi_viewer_streaming.py
19 passed in 53s
```
