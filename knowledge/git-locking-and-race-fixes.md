---
title: Worktree/merge locking and known race conditions (repo_lock, kiss-reclaim.lock
  flock, lock order)
uuid: 74e1c0a2-67c9-411d-b4bc-45ea10d0b6b2
summary: repo_lock (per-repo RLock) then the cross-process flock on <git_common_dir>/kiss-reclaim.lock;
  git timeouts; races between tabs and processes and how they were fixed.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Worktree/merge locking and known race conditions

## Locks
- `repo_lock(repo)`: per resolved repo path `threading.RLock`, serializes multi-step git sequences between
  threads (tabs) of ONE process. Re-entrant so `_try_setup_worktree` can hold it across retire + create.
- `_reclaim_process_lock(repo)`: `flock` on `<git_common_dir>/kiss-reclaim.lock`, serializes across PROCESSES
  (kiss-web daemon vs `kiss` CLI). Held by reclaim, sweep, `create`, `_do_merge`, `discard`, and the baseline
  snapshot in `_try_setup_worktree`. Re-entrant per thread (`_held_reclaim_locks`). The kernel drops it if the
  holder dies. Degrades to in-process only if the common dir cannot be resolved or opening
  `kiss-reclaim.lock` raises `OSError` (unwritable git dir).
- Order: `repo_lock` first, flock inside, everywhere, so there is no ABBA between processes' threads. A
  cross-repo switch retires the previous repo BEFORE taking the new repo's lock.
- `GitWorktreeOps.remove` takes `repo_lock` itself because teardown mutates the shared worktree registry.
- `_git` runs with a 300 s timeout (`_GIT_TIMEOUT_SECONDS`, returncode 124 on timeout) and strips
  repo-scoped env vars (`_REPO_SCOPED_GIT_ENV`: `GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE`, ...).
  `_git_stdout_head` streams bounded output with a watchdog timer.

## Races fixed (what went wrong, what prevents it now)
- Another process's failure-path `git reset --hard` wiped a staged squash merge; `_commit_staged_merge` saw an
  empty index, reported SUCCESS, and the branch was deleted. Fix: whole merge transaction under the flock.
- Peer reclaim found a just-added worktree with no owner and deleted it. Fix: `create` adds and stamps
  `kiss-owner-pid` under the flock; a failed stamp aborts the creation.
- First run of a fresh agent reclaimed sibling tabs' running worktrees. Fix: printer bound before setup.
- PID reuse made a dead owner look alive. Fix: `kiss-owner-identity` (start time + executable).
- Pool: refill published a spare after `discard_all`. Fix: generation counter + `_active_discards` counter and
  one-lock dedup in `prewarm_async` (see `git-worktree-pool`).
- Warnings flushed twice or lost. Fix: take-and-clear under `_warning_lock`; `add_warning` combines atomically.
- Busy check and `is_merging` claim in separate lock sections let `closeTab`/new run dispose a worktree mid-merge.
  Fix: resolve, check and claim in one `_state_lock` section (see `git-server-merge-flow`).
- Auto-commit leftovers from files written during the LLM call. Fix: second `stage_all` + late-arriver retry.
- Huge diffs hung the daemon. Fix: bounded `staged_diff`, `has_staged_changes`.

## Testing races
`KISS_RACE_DELAY` (`_concurrency._race_delay`, capped at 0.1 s, no-op in production) widens windows, e.g. inside
`reclaim_orphaned_worktrees` between guards and mutations.

## Sources
- `src/kiss/agents/sorcar/git_worktree.py` (`repo_lock`, `_reclaim_process_lock`, `_held_reclaim_locks`, `_git`, `_git_stdout_head`, `_REPO_SCOPED_GIT_ENV`, `GitWorktreeOps.create`, `remove`)
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` (`_try_setup_worktree`, `_do_merge`, `discard`, `run`, `_flush_warnings`)
- `src/kiss/agents/sorcar/_concurrency.py` (`_race_delay`)
