---
title: Spare worktree pool (worktree_pool.py) - prewarm, take_spare, discard_all
uuid: 6cba021f-76b2-418c-8a6a-a4bd44a99989
summary: One pre-created spare kiss/wt-* worktree per repo, refilled on a background
  thread so task start skips git worktree add; generation counter and _active_discards
  fence; KISS_DISABLE_WORKTREE_POOL.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Spare worktree pool

`git worktree add` is a full checkout (about a second on a large repo) and dominated the delay between
submit and agent start. `worktree_pool.py` keeps at most ONE ready spare per repository.

## Flow
- `WorktreeSorcarAgent._acquire_task_worktree` calls `take_spare(repo)`. If a spare is returned it is
  consumed with `reset_worktree_to(spare_dir, original_branch)` (hard reset onto the task's base),
  `clean_untracked`, and `clear_spare_marker`. Any failure: `cleanup_partial` and fall back to the
  inline path (reclaim + sweep + `new_task_branch` + `GitWorktreeOps.create`).
- Either way it then calls `prewarm_async(repo, self._live_worktree_branches)` to refill for the next task.
- `prewarm` (under `repo_lock`) runs the orphan maintenance (`reclaim_orphaned_worktrees` +
  `sweep_orphaned_state`, moved off the submit path), `ensure_excluded`, creates the worktree, writes the
  `kiss-spare` marker, and runs `reset_worktree_to(wt_dir, "HEAD")` once to warm the index stat cache
  (otherwise the first consume-time reset re-hashes every file).
- `exclude_branches_fn` is evaluated late, inside the lock, so a task started after scheduling is excluded.
  `None` skips maintenance entirely.

## Validation in `take_spare`
The spare is popped unconditionally, then rejected (returns `None`) when its directory or branch vanished,
when a different branch is checked out in it, or when `GitWorktreeOps.spare_has_content` reports
uncommitted changes, ignored files (or unlistable ones), or commits unique to its branch. A spare is never
written by Sorcar, so content means an external writer; it is left for the reclaim pass.

## Safety
- Spares look exactly like task worktrees on disk. Live spares are excluded from this process's reclaim via
  `spare_branches()` and from other processes' reclaim via the owner pid stamped by `create`.
- `discard_all()` (tests/embedders) joins refill threads (120 s bound), bumps `_generation` at start and
  end, and raises `_active_discards` for its duration. A refill publishes only if its captured generation
  is unchanged AND `_active_discards == 0`; otherwise it removes its spare. `_active_discards` is a counter
  because two overlapping discards broke a boolean.
- `prewarm_async` does the dedup check, thread registration and `start()` under one `_pool_lock` hold, so two
  callers cannot both spawn and the registered thread is the one `discard_all` joins.
- `KISS_DISABLE_WORKTREE_POOL=1` disables background refills; the root `conftest.py` sets it so tests are not
  raced by refill threads (pool tests re-enable it).

## Sources
- `src/kiss/agents/sorcar/worktree_pool.py` (`take_spare`, `prewarm`, `prewarm_async`, `discard_all`, `spare_branches`, `new_task_branch`, `pool_enabled`)
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` (`_acquire_task_worktree`, `_live_worktree_branches`)
- `src/kiss/agents/sorcar/git_worktree.py` (`GitWorktreeOps.spare_has_content`, `reset_worktree_to`)
