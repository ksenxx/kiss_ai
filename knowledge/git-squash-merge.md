---
title: Squash-merging a worktree branch back (_do_merge, cherry-pick baseline..branch,
  -X theirs, MergeResult)
uuid: 575e78e0-6bd0-492b-ae14-b7c384682926
summary: _do_merge stashes the dirty main tree, checks out the original branch, cherry-picks
  baseline..branch --no-commit (or merge --squash), commits, pops the stash, deletes
  the branch; MergeResult values.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Squash-merging a worktree branch back

## `WorktreeSorcarAgent._do_merge(wt, conflict_resolver=None)`
Runs entirely under `repo_lock(wt.repo_root)` + `_reclaim_process_lock(wt.repo_root)`:
1. `ensure_scratch_merge_driver`.
2. `stash_if_dirty` (`git stash push --include-untracked -m "kiss: auto-stash before merge"`). If the tree is
   dirty but nothing was stashed, return `STASH_FAILED` without touching the repo (else staged user edits
   would enter the merge commit and the conflict-path `reset --hard` would destroy them).
3. Check out `wt.original_branch` if needed; failure = `CHECKOUT_FAILED` (stash popped back).
4. With a baseline: `squash_merge_from_baseline`; otherwise `squash_merge_branch`.
5. On `CONFLICT` with a `conflict_resolver`: call it (see `git-merge-conflict-resolution`). If it raises
   (e.g. user stop), pop the stash and re-raise.
6. Stash: popped on SUCCESS; on any failure it is left in `git stash` and the user is told to pop after resolving.
7. On SUCCESS, `delete_branch` (tries `-d`, then `-D`, removes the config section); failure becomes a
   `cleanup_warning`.
Returns `(MergeResult, stash_warning, cleanup_warning)`.

## `squash_merge_from_baseline(repo, branch, baseline, ...)`
- `rev-list --count baseline..branch` == 0: SUCCESS (nothing to merge).
- `git cherry-pick --no-commit baseline..branch`, adding `-X theirs` only when HEAD == `baseline^`
  (`_head_matches_baseline_parent`). In that case the stashed dirty edits make the 3-way merge see
  "revert the dirty edits" on the ours side and raise spurious conflicts; `-X theirs` resolves them for
  the branch. When main moved on, `-X theirs` is NOT used: a conflict is real and must not overwrite the
  user's commits.
- Failure: `_abort_cherry_pick` (verifies the abort; falls back to `--quit` + reset if the tree changed), return CONFLICT.
- Success: `_commit_staged_merge` commits with `_merge_commit_message`; a commit failure (pre-commit hook)
  resets hard and returns `MERGE_FAILED`.

`squash_merge_branch` does the same with `git merge --squash` and resets hard on conflict.

## `MergeResult`
`SUCCESS`, `CONFLICT`, `MERGE_FAILED` (applied but commit refused), `CHECKOUT_FAILED`, `STASH_FAILED`.
Every failed git command reports CONFLICT, including operational failures like a held `index.lock`;
`merge_has_content_conflict` re-checks with `git merge-tree --write-tree` (in memory) to tell a real
content conflict apart.

## `stash_pop`
Tries `git stash pop --index`; falls back to plain `pop` only if the failed attempt left `git status`
unchanged, to avoid double-applying a partially applied stash. A failed pop leaves unmerged paths with no
MERGE_HEAD and the stash entry `kiss: auto-stash before merge` still listed.

## Manual recovery text
`_merge_fix_steps` prints `cd`, `git checkout <original>`, `_manual_merge_cmd`, fix lines, and
`git branch -D <branch>` (force, since a squash merge never makes the branch an ancestor, `-d` always refuses).

## Sources
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` (`_do_merge`, `_merge_fix_steps`, `_manual_merge_cmd`, `merge`)
- `src/kiss/agents/sorcar/git_worktree.py` (`MergeResult`, `stash_if_dirty`, `stash_pop`, `squash_merge_from_baseline`, `squash_merge_branch`, `_commit_staged_merge`, `_abort_cherry_pick`, `merge_has_content_conflict`, `delete_branch`)
