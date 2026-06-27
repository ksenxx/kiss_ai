# Task

When an agent finishes a task and before autocommit, show a
notification "Generating commit message"; after the commit, show
"Committed {a line of the commit message}".

Reproduce by writing an end-to-end test, then fix.

- Coding/bug-fixing/test creation: `claude-opus-4-7`.
- Final thorough review: `gpt-5.5` (NOT codex).

## Status: complete (round-2 review applied)

### Round-2 review fixes (this session)

Round-1 gpt-5.5 review of the original implementation
flagged two bugs in addition to the original spec:

- **Bug 1** — `auto_commit_changes` fired the "Generating commit
  message" toast AND invoked the slow `message_fn` LLM call
  *unconditionally*, even when nothing was staged. Result: a
  misleading toast and wasted tokens for a commit that never
  happens.
- **Bug 2** —
  `WorktreeSorcarAgent._broadcast_commit_notification` minted a
  fresh id per stage
  (`f"autocommit-{stage}-{time.time_ns()}"`), so the webview
  stacked two toasts instead of updating the existing one in
  place; the misleading "Generating" toast lingered next to
  "Committed <subject>" until its own auto-dismiss timer fired.

Fixes applied:

- `sorcar_agent.py auto_commit_changes` — short-circuit
  immediately after the first `stage_all` when
  `GitWorktreeOps.staged_diff(commit_dir)` is empty: no
  "generating" toast, no `message_fn` call, no "committed"
  toast. Docstring updated to document the new contract.
- `worktree_sorcar_agent.py` — added
  `self._commit_run_id: str = ""` to `__init__`. Set in
  `_auto_commit_worktree` to
  `f"autocommit-{self._tab_id}-{time.time_ns()}"` before
  calling `auto_commit_changes`, then consumed (with a
  defensive fallback for direct callers) by
  `_broadcast_commit_notification` so both stages share the
  same notification id and `media/main.js`'s `showNotification`
  updates the existing toast in place.

Tests added/updated in
`test_autocommit_notifications.py` (13 cases total, all
passing):

- Inverted `test_no_commit_skips_message_fn_and_notifications`
  and `test_worktree_no_commit_emits_no_notifications` to
  assert ZERO notifications and ZERO `message_fn` calls in the
  empty-tree path (Bug 1).
- New `test_empty_tree_does_not_invoke_message_fn` (Bug 1 unit).
- New `test_worktree_generating_and_committed_share_id`
  (Bug 2).
- New `test_worktree_no_commit_does_not_call_llm` (Bug 1
  worktree integration, spies on
  `sorcar_agent._generate_commit_message`).
- Existing happy-path test now also asserts
  `gen_ev["id"] == committed_ev["id"]` (Bug 2).

### Bug 3 — out of scope

The "Generating" toast still lingers when `message_fn`
succeeds but `commit_staged` returns False (e.g. a pre-commit
hook rejected the commit). Round-1 review flagged this and
explicitly left it for a follow-up because the same-id fix
gives us a natural extension point (emit a third "failed"
event with the same id) — but the current spec only requires
the two-stage flow.

### Round-2 review verdict

Re-reviewed by `gpt-5.5` (non-codex). Both Bug 1 and Bug 2
fixes are correct, minimal, and well-scoped. No new bugs
found. The reduced race window in `auto_commit_changes` is
acceptable because the protection only mattered while
`message_fn` was running.

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
