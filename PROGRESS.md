# Sorcar daemon-client CLI: input box always visible (complete)

## Goal

Show the rectangular bottom-anchored input box during task execution in `sorcar`'s daemon-client
interactive mode, the same way it's shown when sorcar CLI starts. Lines submitted into the box
while a task runs must be queued onto the daemon via `appendUserMessage` — the same wire
command used by the VS Code extension and the remote browser webapp.

## Root cause

Previously `AnchoredRepl` was only integrated into the standalone `cli_repl.run_repl()` path,
but the actual interactive entry point is `cli_client.run_client()` (the daemon client). It
read input via inline `_read_line` and during task execution only rendered streamed events —
no bottom-anchored box.

## Implementation

### `src/kiss/agents/sorcar/cli_steering.py` (+75 lines)

Added `AnchoredRepl.run_steering_loop(on_submit, on_abort, is_done, on_idle=None)`:

- Flips the box title to `STEER_TITLE`, restores on exit.
- `select`-based stdin read loop dispatching keystrokes through `_InputBox.feed`.
- Exits once `is_done()` becomes `True`.
- `on_idle` invoked once per select timeout for callers to drain `askUser` from the dispatcher
  without blocking the loop.
- Catches `InterruptedError`/`OSError`/`KeyboardInterrupt` on `select` and `os.read`; re-draws
  on terminal resize.

### `src/kiss/agents/sorcar/cli_client.py` (+275 lines)

- `_submit_task_anchored(client, prompt, repl, *, use_worktree, use_parallel, auto_commit, timeout_seconds)`: sends `{"type": "run", ...}`, waits for `dispatcher.task_active`, then
  drives `repl.run_steering_loop(...)` with closures that:
  - `on_submit` → sends `{"type": "appendUserMessage", "prompt": text, "tabId": ...}`,
    increments the `queued: N` status, echoes a dim `▸ queued: <text>` line.
  - `on_abort` → sends `{"type": "stop"}`.
  - `on_idle` → drains `dispatcher.ask_user_q`; flips the box title to "answer the question
    above"; the next submitted line is sent as `{"type": "userAnswer", "answer": line}`.
  - `is_done` returns true when `task_active` clears, the client closes, or the wall-clock cap
    is hit.
- `_run_anchored_client(client, work_dir, model_name, active_file, *, use_worktree, use_parallel, auto_commit)`: full REPL loop with anchored box for both idle reads and task
  execution. Persists history via `_load_history_lines`/`_save_history_lines`. Routes slash
  commands through the existing `_handle_client_slash`.
- `run_client()` dispatches to `_run_anchored_client` when `supports_steering() and rows >= _MIN_ROWS`; falls back to the inline path otherwise.

### Tests

- `src/kiss/tests/agents/sorcar/test_cli_client.py` (+283 lines):
  - `_TestRepl`: test stand-in for `AnchoredRepl` with a real `_InputBox` and lock,
    `run_steering_loop` synchronously playing a scripted sequence.
  - `TestSubmitTaskAnchored`:
    - `test_run_command_carries_flags` — `useWorktree`/`useParallel`/`autoCommit` flags
      reach the daemon on the `run` command.
    - `test_submitted_lines_become_append_user_message` — typed lines send
      `appendUserMessage`.
    - `test_abort_sends_stop_to_daemon` — Ctrl+C sends `stop`.
    - `test_ask_user_question_flips_title_and_routes_answer` — `askUser` arrival flips the
      box, next submit is sent as `userAnswer` (NOT `appendUserMessage`).
    - `test_daemon_disconnect_returns_promptly` — closed connection unblocks the loop.
  - All tests use the real `_RecordingRemoteAccessServer` + `_DaemonHarness` to assert on
    inbound commands.
- `src/kiss/tests/agents/sorcar/test_cli_steering.py` (+120 lines):
  - `TestRunSteeringLoop`: title flip to `STEER_TITLE` + restore; `on_idle` is fired;
    `on_idle` exceptions do not crash the loop. `sys.stdin` is replaced with the read end of a
    real OS pipe so `sys.stdin.fileno()` works under pytest's stdin capture.

## Verification

- `uv run check --full` → all green (ruff, mypy, pyright, mdformat, syntax check, API docs).
- `uv run pytest src/kiss/tests/agents/sorcar/test_cli_client.py src/kiss/tests/agents/sorcar/test_cli_steering.py` → **75/75 passed**.
