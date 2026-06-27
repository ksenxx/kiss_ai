# Task

When an agent finishes a task and before autocommit, show a
notification "Generating commit message"; after the commit, show
"Committed {a line of the commit message}".

Reproduce by writing an end-to-end test, then fix.

- Coding/bug-fixing/test creation: `claude-opus-4-7`.
- Final thorough review: `gpt-5.5` (NOT codex).

## Status: complete

## What was changed

1. `src/kiss/agents/sorcar/sorcar_agent.py` — added optional
   `notify_fn: Callable[[str, str], None] | None = None` parameter
   to `auto_commit_changes`. Calls `notify_fn("generating", "")`
   immediately before `message_fn` runs (typically a slow LLM call)
   and `notify_fn("committed", subject)` immediately after a
   successful commit, where `subject` is the first non-empty line
   of the committed message. Added `_commit_subject` and
   `_safe_notify` helpers; the latter swallows exceptions so a
   broken UI hook can never block the commit.
1. `src/kiss/agents/sorcar/worktree_sorcar_agent.py` — added
   `_broadcast_commit_notification(stage, subject)` method on
   `WorktreeSorcarAgent` and wired it into `_auto_commit_worktree`
   via `notify_fn=self._broadcast_commit_notification`. The method
   broadcasts `{type: "notification", id, severity: "info", message, tabId}` through `self.printer.broadcast` (the same
   pipeline the existing warning toasts use), rendering through
   `media/main.js` `case 'notification'`. No-ops silently when no
   printer is attached or the printer lacks `broadcast`.
1. `src/kiss/tests/agents/sorcar/test_autocommit_notifications.py`
   — 10 end-to-end tests against on-disk git repos with a fake
   recording printer. Covers ordering vs HEAD SHA (generating
   fires before HEAD moves; committed fires after), subject
   extraction, no-commit case (only generating fires), missing
   `notify_fn`, notify exceptions, missing printer, printer
   without `broadcast`, and `auto_commit_enabled=False` (zero
   notifications).

## Verification

- `uv run check --full` — all checks pass.
- `uv run pytest -v src/kiss/tests/agents/sorcar/test_autocommit_notifications.py`
  — 10/10 pass.
- Regression: `test_autocommit_user_prompt.py`,
  `test_autocommit_after_merge.py`,
  `test_autocommit_race_new_file.py`,
  `test_autocommit_off_on_failure.py` — 37/37 pass.

## Review (gpt-5.5)

- Implementation is clean and matches the task spec. Two
  separate toasts ("Generating commit message" then "Committed
  <subject>") are appropriate.
- Distinct per-call ids (`f"autocommit-{stage}-{time.time_ns()}"`)
  let each toast auto-dismiss independently via the existing
  `scheduleNotificationDismiss` path in `media/main.js`.
- `_safe_notify` correctly insulates the commit path from any UI
  hook failure (including any exception thrown by a misbehaving
  printer).
- The empty-tree case correctly emits only the "generating" toast
  (no spurious "Committed" when no commit was created).
