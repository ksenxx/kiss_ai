---
title: Git worktrees, auto-commit, merge and task persistence (area overview)
uuid: 13331495-1785-45e0-acc4-ef4154c0689c
summary: Map of Sorcar's worktree isolation, spare-worktree pool, auto-commit, commit-message
  LLM, squash merge, merge SEA conflict resolution, server merge flow and sorcar.db
  persistence files.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Git worktrees, auto-commit, merge and task persistence (area overview)

Sorcar runs a development task on its own git branch in a linked worktree under
`<repo>/.kiss-worktrees/`, so the user's checkout is not edited while the agent works. This
applies only when worktree mode is enabled and setup succeeds: `use_worktree=False`, a
non-development verdict from the task classifier, a non-git directory, or a failed setup runs
the task directly (`WorktreeSorcarAgent.run`, `_try_setup_worktree`). When a task succeeds with
auto-commit on, the server commits on the task branch and squash-merges (or discards an empty
branch) back into the user's branch. With auto-commit off, or when the task failed, was stopped
or returned `success: false`, it presents Merge/Discard instead (`effective_auto_commit` in
`task_runner.py`). Task history and the chat transcript go to SQLite
(`~/.kiss/sorcar.db`).

## Files
| File | Role |
|---|---|
| `src/kiss/agents/sorcar/git_worktree.py` | `GitWorktree` (frozen state), `MergeResult`, `GitWorktreeOps` (all git plumbing), `repo_lock`, `_reclaim_process_lock`, `_git` runner |
| `src/kiss/agents/sorcar/worktree_sorcar_agent.py` | `WorktreeSorcarAgent`: per-task worktree setup, retire, `merge`/`discard`/`leave_as_is` |
| `src/kiss/agents/sorcar/worktree_pool.py` | One pre-created spare worktree per repo, filled on a background thread |
| `src/kiss/agents/sorcar/commit_message.py` | LLM commit-message generation (`generate_commit_message_from_diff`) |
| `src/kiss/agents/sorcar/sorcar_agent.py` | `auto_commit_changes`, `_generate_commit_message` (shared auto-commit engine) |
| `src/kiss/agents/sorcar/persistence.py` | sorcar.db schema, task rows, event writer, orphan recovery |
| `src/kiss/server/merge_flow.py` | `_MergeFlowMixin`: server-side post-task autocommit, merge/discard actions, deferred merges |
| `src/kiss/server/merge_conflict_resolver.py` | Runs the merge SEA on a conflicted merge and commits the result |
| `src/kiss/agents/seas/merge_sea.py` | The merge agent (system prompt, `build_prompt`); also the `/merge` slash command |
| `src/kiss/server/diff_merge.py` | Leftover: file scanner for autocomplete + positional-`cwd` `_git` adapter |

## Pages
- `git-worktree-lifecycle`: setup, run, retire, merge/discard/leave-as-is.
- `git-branch-config-markers`: `branch.kiss/wt-*.kiss-*` git config keys.
- `git-baseline-dirty-state`: how the user's uncommitted edits are carried into a worktree.
- `git-node-modules-and-excludes`: `info/exclude`, `node_modules` symlinks, `kiss-scratch` merge driver.
- `git-worktree-pool`: spare worktree pre-creation and its generation fence.
- `git-auto-commit`: `auto_commit_changes`, the auto-commit toggle, late-arriving files.
- `git-commit-message-generation`: LLM prompt, fallbacks, `User prompt:`/`Result:` blocks, diff cap.
- `git-squash-merge`: `_do_merge`, stash/checkout/cherry-pick, `-X theirs` rule, `MergeResult`.
- `git-merge-conflict-resolution`: merge SEA flow and its fail-closed verification.
- `git-server-merge-flow`: busy guards, deferred merges, `_PendingOutcome`, non-worktree autocommit.
- `git-cleanup-outcomes-and-preserve`: `_WorktreeCleanupOutcome`, `_pending_review`, keep-for-review.
- `git-ignored-file-rescue`: copying git-ignored task output back before removal.
- `git-orphan-reclaim-and-sweep`: recovering worktrees stranded by killed processes.
- `git-locking-and-race-fixes`: `repo_lock`, cross-process flock, known races and fixes.
- `db-overview`, `db-schema`, `db-task-and-chat-ids`, `db-event-writer`, `db-orphan-task-recovery`, `db-concurrency`.

## Sources
- `src/kiss/agents/sorcar/git_worktree.py` (module docstring)
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` (`WorktreeSorcarAgent`)
- `src/kiss/server/merge_flow.py` (module docstring)
