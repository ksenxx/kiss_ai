---
title: Merge conflict resolution with the merge SEA (merge_conflict_resolver.py, merge_sea.py,
  /merge)
uuid: 2a8f43ab-c148-41b8-aee4-fd972bb8ef23
summary: On a CONFLICT in auto-commit mode, resolve_merge_conflict re-applies the
  branch with markers, runs merge_sea as a nested sub-agent ($5 cap), verifies with
  finish_conflicted_merge, else aborts.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Merge conflict resolution with the merge SEA

## Wiring
- `merge_flow._handle_worktree_action(..., resolve_conflicts=True)` (merge in auto-commit mode) passes
  `merge_conflict_resolver.resolve_merge_conflict` as `conflict_resolver` to `WorktreeSorcarAgent.merge`.
- `_do_merge` calls it after the squash merge returned CONFLICT, with the main tree clean, on the original
  branch, still under `repo_lock` + the cross-process flock.
- Layering: the resolver is in `kiss.server` because `kiss.agents.sorcar` must not import `kiss.agents.seas`.

## `resolve_merge_conflict(parent_agent, wt, user_prompt, task_result, run_agent=run_merge_sea)`
1. Capture `status_porcelain` and HEAD (`head_before`).
2. `GitWorktreeOps.begin_conflicted_merge(repo, branch, baseline)`: with a baseline, builds a synthetic squashed
   commit (branch tree, parent = baseline, message `kiss: squashed <branch>`) and cherry-picks it `--no-commit`
   with the same `-X theirs` rule, so the agent sees ONE set of conflicts instead of a stopped multi-commit
   sequencer; without a baseline uses `git merge --squash`. Returns the unmerged paths, `[]` if clean, `None` on
   non-conflict git failure.
3. If there are conflicts: `merge_sea.build_prompt(...)` and `run_merge_sea` (exceptions logged, not raised).
4. `finish_conflicted_merge` accepts only if HEAD is still `head_before`, no path is unmerged, no originally
   conflicted file has markers, and the index is non-empty. A moved HEAD (agent committed itself) or empty index
   is refused, because SUCCESS would delete the only branch holding the work.
5. CONFLICT or any exception: `abort_conflicted_merge` resets to `head_before` and removes untracked leftovers
   only if the tree was clean before.
`merge()` reports "The merge agent resolved the conflicts." when `_last_merge_resolved_by_agent` is set.

## `run_merge_sea`
Runs a `ChatSorcarAgent("Merge conflict resolver")` in-process with `_tab_id = task-<parent>__merge` and
`_subagent_info` so it shows as a nested tab; uses the parent's model (unless the SEA defines `model()`),
`merge_sea.system_prompt()` as base prompt, `max_budget()` = `MAX_BUDGET_USD` 5.0, no parallel, no web tools, no
memory. Spend is added to the parent via `_attribute_sub_usage` in a `finally`; the server persists it
(`_persist_merge_agent_usage`).

## `merge_sea.py` rules
Stay on the current branch; never stash/reset/restore/abort; keep the intent of both sides, prefer the incoming
task branch for alternative edits of the same thing; remove markers and `git add`; do not commit; check
`git diff --name-only --diff-filter=U` is empty; leave a file conflicted and say so if unsure.
The same file is dispatched by the `/merge` slash command through `run_agent` (no worktree, no auto-commit).

## Sources
- `src/kiss/server/merge_conflict_resolver.py` (`resolve_merge_conflict`, `run_merge_sea`)
- `src/kiss/agents/seas/merge_sea.py` (`SYSTEM_PROMPT`, `build_prompt`, `MAX_BUDGET_USD`)
- `src/kiss/agents/sorcar/git_worktree.py` (`begin_conflicted_merge`, `finish_conflicted_merge`, `abort_conflicted_merge`)
- `src/kiss/server/merge_flow.py` (`_handle_worktree_action`, `_persist_merge_agent_usage`)
