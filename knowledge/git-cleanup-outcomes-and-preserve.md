---
title: Worktree cleanup outcomes, _pending_review and keep-for-review (never lose
  or silently publish work)
uuid: ae804f54-19a5-4d5c-8a02-6bd3325b503b
summary: _commit_and_clean_worktree returns COMMITTED_AND_REMOVED or a PRESERVED_*
  outcome; failed/stopped tasks set _pending_review and are committed to kiss/wt-*
  only; kiss-preserve marker.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Worktree cleanup outcomes and keep-for-review

## `_commit_and_clean_worktree(wt, force_commit=False)` returns `(_WorktreeCleanupOutcome, leftover)`
Order matters:
1. `reclaim_abandoned_subagents(timeout=_ABANDONED_SUBAGENT_WAIT_SECONDS)` (5 s) FIRST, so files a finishing
   sub-agent writes are committed instead of deleted. Still running: `PRESERVED_SUBAGENT_ACTIVE`.
2. `_auto_commit_worktree`. Changes left and auto-commit off (not forced): `PRESERVED_NO_AUTOCOMMIT`.
3. One retry `commit_all("kiss: auto-commit late-arriving changes")`. Still dirty: `PRESERVED_COMMIT_FAILED`
   with the `git status --porcelain` leftover (usually a pre-commit hook).
4. `rescue_ignored_files`; failure: `PRESERVED_RESCUE_FAILED` (the worktree holds the only copy).
5. `GitWorktreeOps.remove` (it prunes itself): `COMMITTED_AND_REMOVED`.
Only the last outcome deletes the directory. `_finalize_worktree` stores it in `_last_preserve_outcome` so
`merge()` and `_release_worktree` can name the real cause in their warnings.

## `_pending_review`
Raised by the server's task runner (`src/kiss/server/task_runner.py`) for a task that failed or was stopped. While set, retirement
(`_retire_previous_worktree`) and tab teardown call `_preserve_pending_worktree_for_review`: commit onto the
`kiss/wt-*` branch and never merge. The directory is removed only on `COMMITTED_AND_REMOVED`; when auto-commit
is off or the commit fails, it is kept (the only copy) and the user is warned. Session resume does not
preserve: it returns `PRESENT_CLAIMED` and presents Merge/Discard (`merge_flow.py`). The user must click Merge. Explicit
`merge()` clears the flag; `discard()` clears it only after its deferral checks, so a deferred discard keeps
the protection.

## `_keep_for_review(wt)`
When a preserve outcome leaves the worktree on disk, the agent drops its claim only after
`save_preserve_marker` wrote `kiss-preserve`. If the git config write fails, `self._wt` stays set,
`_pending_review` is raised, and a warning asks the user to act. `_try_setup_worktree` then runs the next task
directly rather than overwriting the claim, and `retire_for_disposal()` returns False so the server keeps the
agent reachable.

## Why
The worktree branch or directory is often the only copy of the work. Every path fails closed: preserve rather
than delete, and never publish unfinished work into the user's branch.

## Sources
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` (`_WorktreeCleanupOutcome`, `_commit_and_clean_worktree`, `_finalize_worktree`, `_preserve_pending_worktree_for_review`, `_keep_for_review`, `retire_for_disposal`, `_retire_previous_worktree`)
