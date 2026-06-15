# Task: Record events+result in DB when ChatSorcarAgent/WorktreeSorcarAgent run outside a chat webview

## Status: COMPLETE (committed 0d2987f6)

## Problem

When ChatSorcarAgent or WorktreeSorcarAgent runs from OUTSIDE a chat webview
(extension CLI, remote webapp, third-party channel agents), it did not record
the agent's EVENTS into sorcar.db. Only `task_history` (task + result column)
was saved. Event replay in the chat webview reads the `events` table, so those
runs loaded as a BLANK session and could not be loaded/replayed.

## Root cause

Events are persisted only via a recording printer (VS Code server's
`JsonPrinter`/`WebPrinter`): `broadcast()` -> `_persist_event()` ->
`_queue_chat_event()` -> `events` table, gated on the agent being registered in
`printer._persist_agents`. A plain console printer (CLI / channel agents /
remote without a recording printer) never persists events, so the `events`
table stayed empty for the task.

## Fix

1. `src/kiss/agents/sorcar/persistence.py`: added `_task_has_events(task_id)`
   helper — flushes the async event queue then checks the `events` table for any
   row for the task.

1. `src/kiss/agents/sorcar/chat_sorcar_agent.py`:

   - Imported `_append_chat_event`, `_task_has_events`, and `parse_result_yaml`.
   - Captured `result_raw` (raw YAML returned by the run) in `run()`.
   - Added `_persist_replay_events_if_missing(task_id, prompt, result_raw, result_summary)`: when the task has NO events yet, synthesizes a minimal
     replayable stream — a `prompt` event (the chat-augmented prompt the agent
     ran, mirroring what a recording printer would persist) and a `result`
     event (summary / success / is_continue / tokens / cost / steps). Guarded by
     `_task_has_events` so a recording printer's full stream is never
     duplicated.
   - Called it from the `run()` `finally` block, inside `if not skip_persistence:` after `_save_task_result` + `_save_task_extra`.
     WorktreeSorcarAgent inherits this via `super().run()`.

   Webview path is unaffected: top-level VS Code runs pass
   `_skip_persistence=True` (server owns persistence); other recording-printer
   runs already populate `events` so the synth is skipped.

## Tests

`src/kiss/tests/agents/sorcar/test_replay_events_outside_webview.py` (4 tests,
real temp-dir SQLite, no mocks — offline agent uses MRO so
`ChatSorcarAgent.run`'s `super().run()` resolves to a canned `SorcarAgent`
subclass):

- no-printer run persists replayable prompt+result events;
- chat_id lookup (`_load_latest_chat_events_by_chat_id`) finds them;
- recording-printer run is NOT duplicated (exactly one result, no synth prompt);
- helper is idempotent on `WorktreeSorcarAgent` (no dup on second call).

## Verification

- `uv run check --full` — all checks pass (ruff, mypy, pyright, mdformat, etc.).
- Full sorcar suite: `1537 passed, 28 deselected` in 320s.
