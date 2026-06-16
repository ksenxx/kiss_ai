# PROGRESS — history-resumed running tab follow-up input

## Task

When a task running in tab A is loaded into a fresh tab B from the
task-history panel, the chat webview in tab B does not accept input
messages from the user, the pulsing green circle in tab B's title
stays even after the task finishes, and messages get lost. Reproduce
with an integration test, then fix the root cause.

## Investigation

Frontend (`media/main.js`) flow on history-row click:

1. `createNewTab()` allocates a fresh `tab_id_new`, posts
   `{type: 'newChat', tabId: tab_id_new}`.
2. Posts `{type: 'resumeSession', id: chat_id, taskId, tabId: tab_id_new}`.

Backend (`server.py::_replay_session`) on `resumeSession`:

1. Calls `_reattach_running_chat(chat_id, tab_id_new, task_id=...)`
   which scans `_RunningAgentState.running_agent_states` for a state
   whose `_live_task_id == task_id` (or `chat_id` matches), then
   `printer.subscribe_tab(source_task_id, tab_id_new)` to add tab B as
   a viewer of source tab's task event stream.
2. Broadcasts `{type: 'status', running: True, tabId: tab_id_new, startTs}`.
3. Broadcasts `task_events` replay.

Tab B's `_RunningAgentState[tab_id_new]` was created by `_new_chat`
with `is_task_active=False` — the live agent runs in `tab_id_a`'s
state. Tab B is only a *viewer* (a subscriber in
`printer._subscribers[task_id]`).

### Bug 1 — follow-up input is dropped

User types in tab B's input box while task is running. Frontend
`sendMessage` sees `isRunning=true` and posts
`{type: 'appendUserMessage', prompt, tabId: tab_id_new}`. Backend
`_cmd_append_user_message` (commands.py) looks up
`_running_agent_states[tab_id_new]`, sees `is_task_active=False`, and
silently drops the message with a debug log:

```python
if tab is None or not tab.is_task_active:
    logger.debug("appendUserMessage dropped: tab %s has no live task", tab_id)
    return
```

The live agent's `pending_user_messages` queue, which the agent's
pre-step hook drains, lives on the source tab. The viewer's typed
text never reaches the agent.

### Bug 2 — `status running=False` never reaches the viewer

When the task ends, `_run_task`'s `finally` (task_runner.py) broadcasts:

```python
self.printer.broadcast(
    {"type": "status", "running": False, "tabId": tab_id},  # tab_id == source/launcher tab id
)
```

`WebPrinter.broadcast` sees `tabId` is present and routes the event
verbatim to all WS clients — it does NOT iterate the per-task
subscriber map for tabId-stamped "system" events. Tab B receives the
event but its frontend filters by `ev.tabId === activeTabId`; since
`ev.tabId == source_tab_id ≠ tab_id_new`, `setRunningState(false)` is
never called. Tab B's `isRunning` stays `true`, the tab-title pulse
animation never stops, the input box stays in "queue follow-up"
mode, and subsequent user messages are still routed as
`appendUserMessage` against a now-finished task — getting dropped
again (cascading from bug 1).

## Integration test

`src/kiss/tests/agents/vscode/test_resume_running_followup_input.py`:

End-to-end test driving the real `VSCodeServer` through
`_handle_command`, the real `_replay_session` /
`_reattach_running_chat` path, and the real `_TaskRunnerMixin` worker
thread. Only the innermost LLM-driven `run` is stubbed — it broadcasts
a sentinel `text_delta`, signals a `started` Event so the test knows
the worker is provably inside the run loop, then blocks on a `release`
Event so follow-up actions happen while the task is STILL running.

The captured `broadcast` mirrors `WebPrinter.broadcast` fan-out
exactly: events with explicit `tabId` pass verbatim, task events go
through the per-task subscriber fan-out and are duplicated once per
subscriber with the viewer's tab id stamped.

Three tests:

1. `test_append_user_message_from_viewer_tab_reaches_live_agent` —
   start a task in tab-launcher, open the chat in tab-viewer while
   running, send `appendUserMessage` from tab-viewer, assert the
   prompt lands in `launcher_state.pending_user_messages` (and the
   prompt echo carries `tabId == tab_viewer`).
2. `test_viewer_tab_receives_running_false_when_task_ends` — start
   the task, open the viewer, release the task, assert the viewer
   receives a `status running=False` stamped with the VIEWER's tabId.
3. `test_new_run_from_viewer_tab_after_task_ends` — start, open
   viewer, release, end first task, send a fresh `run` from
   tab-viewer, assert the new task reaches the agent.

Pre-fix: tests 1 and 2 FAIL with the expected symptoms.

## Fix

### `src/kiss/agents/vscode/commands.py::_cmd_append_user_message`

When `tab_id` has no active task, look up the source tab via the
existing `_find_source_tab_for_viewer` helper (scans
`printer._subscribers` for any task `tab_id` is subscribed to, then
finds the peer tab with a live `stop_event` — i.e. the launcher
that owns the running agent). Route the prompt to the source tab's
`pending_user_messages`. The prompt echo still carries the VIEWER's
`tabId` so it appears in the viewer's chat surface (the user typed
it there).

### `src/kiss/agents/vscode/task_runner.py::_run_task`

After broadcasting `status running=False` to the launcher tab, also
broadcast a copy to every viewer tab subscribed to this task's id
via the new `_broadcast_status_end_to_viewers` helper. Mirrors the
start-time per-viewer broadcast in `_subscribe_chat_viewers` and the
per-viewer broadcast in `_replay_session`.

### `src/kiss/agents/vscode/commands.py` TYPE_CHECKING

Added `_find_source_tab_for_viewer` declaration to `_CommandsMixin`'s
`TYPE_CHECKING` block (it's implemented by sibling `_TaskRunnerMixin`).

## Verification

- `uv run pytest src/kiss/tests/agents/vscode/test_resume_running_followup_input.py`
  — all 3 tests pass.
- `uv run pytest src/kiss/tests/agents/vscode/` — 1049 passed, no
  regressions.
- `uv run check --full` — all checks pass (ruff, mypy, pyright,
  mdformat).
