---
title: Worktree task lifecycle (.kiss-worktrees, kiss/wt-* branches)
uuid: 04365d34-e62e-4ee4-8560-c3d9150c2536
summary: How WorktreeSorcarAgent creates a kiss/wt-<epoch>-<uuid8> branch and .kiss-worktrees/kiss_wt-*
  dir per task, falls back to direct execution, retires the previous worktree, and
  merges/discards.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Worktree task lifecycle (.kiss-worktrees, kiss/wt-* branches)

## Naming
- Branch: `kiss/wt-<epoch>-<uuid8>` (`worktree_pool.new_task_branch`, prefix `_WORKTREE_BRANCH_PREFIX`),
  numeric suffix on collision.
- Directory: `<repo>/.kiss-worktrees/<branch with / replaced by _>`, i.e. `kiss_wt-<epoch>-<uuid8>`
  (`_WORKTREE_SUBDIR`, `_WORKTREE_SLUG_PREFIX`).
- Worktrees are not tied to `chat_id`; there is no cross-process restore by chat session.

## `WorktreeSorcarAgent.run`
1. Re-read the auto-commit setting (`auto_commit` kwarg overrides `auto_commit_mode` in `~/.kiss/config.json`).
2. Bind the printer early (`set_printer`) so the reclaim pass can see sibling tabs' live branches
   through `printer.live_worktree_branches`. Without it, a fresh agent's first run reclaimed and
   deleted other running tasks' worktrees.
3. Task classifier may demote to direct execution (`is_development=False`); an explicit
   `use_worktree=False` is never promoted back.
4. `GitWorktreeOps.discover_repo` then `_try_setup_worktree`. On success broadcasts
   `worktree_created` (`worktreeDir`, `worktreeWorkDir`, `branch`) and rewrites `work_dir` into the worktree
   (keeping the subdirectory offset of the original `work_dir`).
5. Falls back to running directly when: not a git repo, no commits, detached HEAD, `work_dir`
   outside repo, any setup git failure, or a previous worktree still pending without a durable keep marker.

## `_try_setup_worktree`
Under `repo_lock(repo)` + `_reclaim_process_lock(repo)`: retire the previous worktree
(same repo; a cross-repo retire runs before the lock to avoid ABBA deadlock), read the current branch,
`ensure_excluded`, `ensure_scratch_merge_driver`, `_acquire_task_worktree` (pool spare or inline
`git worktree add`), `save_original_branch`, `copy_dirty_state` + baseline commit, then
`link_node_modules`. See `git-baseline-dirty-state`, `git-worktree-pool`.

## Retiring the previous worktree
`_retire_previous_worktree` (called by `run`, `new_chat`, `retire_for_disposal`):
- `_pending_review` false (task finished): `_release_worktree` = auto-commit, squash-merge into
  original branch, delete branch. Conflicts leave the branch and set a warning with manual steps.
- `_pending_review` true (task failed/stopped): `_preserve_pending_worktree_for_review` commits onto
  the `kiss/wt-*` branch only, never merges.

## User actions
- `merge(conflict_resolver=None)`: force-commits (even with auto-commit off), then `_do_merge`.
- `discard(rescue_ignored=False)`: waits for abandoned sub-agents (5 s), removes the worktree, checks
  out the original branch, `delete_branch`. Automatic discards pass `rescue_ignored=True`.
- `leave_as_is()`: writes the `kiss-preserve` marker and drops the in-memory claim; raises if the
  marker cannot be written (fail closed).
Warnings are queued with `_set_warnings`/`add_warning` and broadcast by `_flush_warnings`, routed to the
tab via `_broadcast_to_watchers` because they are emitted before a task id exists.

## Sources
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` (`run`, `_try_setup_worktree`, `_retire_previous_worktree`, `_release_worktree`, `merge`, `discard`, `leave_as_is`, `_flush_warnings`)
- `src/kiss/agents/sorcar/worktree_pool.py` (`new_task_branch`)
- `src/kiss/agents/sorcar/git_worktree.py` (`_WORKTREE_SUBDIR`, `_WORKTREE_BRANCH_PREFIX`)
