---
title: Worktree branch git-config markers (kiss-original, kiss-baseline, kiss-preserve,
  kiss-owner-pid, kiss-spare)
uuid: 9370070a-bd3c-454e-949a-6436d5af20ad
summary: 'Durable per-branch metadata Sorcar stores in repo-local git config under
  branch.kiss/wt-*.<key>: original branch, baseline sha, preserve marker, owner pid/identity,
  spare marker.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Worktree branch git-config markers

All worktree metadata that must survive a process crash lives in repo-local git config under
`branch.<kiss/wt-...>.<key>`, written by `GitWorktreeOps._save_branch_config` and read by
`_load_branch_config`. In-memory state (`GitWorktree` on the agent) is lost on restart.

| Key | Writer | Meaning |
|---|---|---|
| `kiss-original` | `save_original_branch` | Branch to merge back into. Missing = legacy worktree; reclaim then uses the main tree's current branch |
| `kiss-baseline` | `save_baseline_commit` | SHA of the "kiss: baseline from dirty state" commit; merges replay only `baseline..branch` |
| `kiss-preserve` | `save_preserve_marker` | Keep for manual review: reclaim must never merge/delete it. Written by `leave_as_is`, `_keep_for_review`, and reclaim on a real content conflict |
| `kiss-owner-pid` | `save_owner_pid` (called by `create`) | PID of the live owning process |
| `kiss-owner-identity` | `save_owner_pid` | `<pid>:<process_identity>` (start time + executable) to detect PID reuse (`_owner_alive`) |
| `kiss-spare` | `save_spare_marker` / `clear_spare_marker` | Unconsumed pool spare: reclaim discards it instead of squash-merging its snapshot |

## Invariants
- `GitWorktreeOps.create` stamps the owner pid under `_reclaim_process_lock`; a failed stamp removes the
  new worktree and reports failure, because an owner-less `kiss/wt-*` looks like a legacy orphan to
  another process's reclaim pass.
- `_keep_for_review` fails closed: if `git config` cannot write `kiss-preserve` (e.g. `.git/config.lock`
  held), the agent keeps `self._wt`, raises `_pending_review`, and warns. The module-level
  `_volatile_preserve_claims` set makes same-process reclaim treat the branch as preserved until the
  marker lands.
- `delete_branch` removes the `branch.<name>` config section too; `sweep_orphaned_state` purges config
  sections whose branch is gone.

## Sources
- `src/kiss/agents/sorcar/git_worktree.py` (`GitWorktreeOps._save_branch_config`, `save_original_branch`, `save_baseline_commit`, `save_preserve_marker`, `save_owner_pid`, `_owner_alive`, `save_spare_marker`, `create`, `_volatile_preserve_claims`)
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` (`_keep_for_review`, `leave_as_is`)
