---
title: Orphan worktree reclaim and debris sweep (reclaim_orphaned_worktrees, sweep_orphaned_state)
uuid: 0b29ff91-a0ca-436c-8616-c0d53f8e4756
summary: 'How a new process recovers kiss/wt-* worktrees stranded by a killed Sorcar
  process: commit, squash-merge into kiss-original, rescue, remove; skip rules; sweep
  of leftover branches and config sections.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Orphan worktree reclaim and debris sweep

A Sorcar process killed mid-task (SIGKILL, OOM, reboot, extension reload) leaves a registered worktree with
uncommitted work and no in-memory `GitWorktree`. Two passes clean up, always reclaim first, then sweep.

## When they run
- Inline in `WorktreeSorcarAgent._acquire_task_worktree` when no pool spare was available.
- Otherwise in the pool refill `worktree_pool.prewarm` (off the submit path).
- Both pass an exclusion set: every branch owned by a live agent in this process (`_live_worktree_branches`,
  which includes `printer.live_worktree_branches()` for other tabs and `worktree_pool.spare_branches()`).

## `GitWorktreeOps.reclaim_orphaned_worktrees(repo, exclude_branches)`
Under `repo_lock` + `_reclaim_process_lock`. First `ensure_excluded`, `prune`; returns 0 if HEAD is detached or
the main tree is dirty. For each registered worktree on a `kiss/wt-*` branch, skip (leave untouched) when:
- it is the main working tree itself (a user checkout on a `kiss/wt-*` branch);
- the branch is excluded;
- `kiss-owner-pid` names another process that is still alive (`_owner_alive`, pid + identity);
- it has a `kiss-spare` marker: discard it (`cleanup_partial`), unless `spare_has_content`, then preserve;
- it has the `kiss-preserve` marker;
- the saved original branch is gone, or differs from the main tree's current branch (reclaim never checks out).
Otherwise: `commit_all("kiss: reclaim orphan worktree")` (skip if still dirty), squash-merge (baseline-aware)
with result text "Auto-merged by orphan-worktree reclaim", then `rescue_ignored_files`, then `cleanup_partial`.
A failed merge preserves the worktree; if `merge_has_content_conflict` confirms a real content conflict, it also
writes `kiss-preserve` so later passes stop retrying a multi-second failing merge (operational failures such as a
held `index.lock` stay retryable).

## `GitWorktreeOps.sweep_orphaned_state(repo)`
Removes plumbing debris only: registrations whose directory is gone, `kiss/wt-*` branches nobody has checked out
and that are expendable, and `branch.kiss/wt-*` config sections whose branch no longer exists. A branch checked out
by a live worktree or holding commits no other ref reaches is kept.
`_branch_is_expendable` checks reachability from all other refs with `--single-worktree`; without that flag every
checked-out branch looked reachable from its own worktree HEAD, and spares carrying an external commit were destroyed.

## Sources
- `src/kiss/agents/sorcar/git_worktree.py` (`GitWorktreeOps.reclaim_orphaned_worktrees`, `sweep_orphaned_state`, `_branch_is_expendable`, `registered_worktrees`, `cleanup_partial`)
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` (`_acquire_task_worktree`, `_live_worktree_branches`)
- `src/kiss/agents/sorcar/worktree_pool.py` (`prewarm`)
