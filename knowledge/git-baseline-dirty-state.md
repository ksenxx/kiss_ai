---
title: Baseline commit - carrying the user's uncommitted edits into a worktree
uuid: 9d22200b-e402-4d3e-b97f-3b7d26f58b6c
summary: 'copy_dirty_state mirrors git status of the main tree into the new worktree
  and commits it as "kiss: baseline from dirty state"; merges then replay only baseline..branch
  via cherry-pick.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Baseline commit: carrying uncommitted edits into a worktree

A new worktree checks out the tip of the user's branch, which lacks the user's uncommitted edits.
`_try_setup_worktree` fixes that:

1. `GitWorktreeOps.copy_dirty_state(repo, wt_dir)` reads `git status --porcelain` in the main tree and
   mirrors each dirty file into the worktree (copies existing files, deletes removed ones). Raises
   `OSError` if `git status` fails, and the caller falls back to direct execution rather than running
   without the user's edits.
2. If anything was copied: `stage_all`, then `commit_staged(..., "kiss: baseline from dirty state", no_verify=True)`.
   The SHA is stored in memory (`GitWorktree.baseline_commit`) and in git config `kiss-baseline`.
3. If that commit fails but changes remain, the worktree is cleaned up and the task runs directly.

## Why it matters at merge time
The baseline commit contains the user's edits, which are still uncommitted on main (they are stashed
during the merge). So the merge must not include the baseline itself:
- With a baseline: `squash_merge_from_baseline` cherry-picks `baseline..branch` with `--no-commit`.
- Without one: `squash_merge_branch` (`git merge --squash`).
- `_manual_merge_cmd` gives the user the matching manual command (`git cherry-pick --no-commit <baseline>..<branch>`
  or `git merge --squash <branch>`). `git merge --squash` on a baseline worktree would wrongly include
  the dirty snapshot.
- `merge_flow._check_merge_conflict` uses `<baseline>^` as the original-side fork point when the
  baseline is valid (`_is_valid_baseline`), otherwise `git merge-base`.

See `git-squash-merge` for the `-X theirs` rule that avoids spurious conflicts caused by the baseline.

## Sources
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` (`_try_setup_worktree`, `_manual_merge_cmd`)
- `src/kiss/agents/sorcar/git_worktree.py` (`GitWorktreeOps.copy_dirty_state`, `save_baseline_commit`, `squash_merge_from_baseline`)
- `src/kiss/server/merge_flow.py` (`_check_merge_conflict`, `_is_valid_baseline`)
