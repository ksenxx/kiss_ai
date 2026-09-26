---
title: Server merge flow (_MergeFlowMixin) - worktree merge/discard actions, busy
  guards, deferred merges
uuid: 6fe6f4ab-6072-48a5-a705-e204729f6809
summary: How merge_flow.py handles worktreeAction merge/discard/nothing, is_merging
  claims, _check_worktree_busy, _main_tree_blocks_merge, deferred auto-merge, _PendingOutcome
  on session resume.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Server merge flow (`_MergeFlowMixin`)

`src/kiss/server/merge_flow.py` is a mixin of the VS Code / web server. It owns the post-task autocommit
of non-worktree tasks (see `git-auto-commit`) and every user or automatic action on a pending worktree.

## `_handle_worktree_action(action, tab_id, *, internal, already_claimed, resolve_conflicts, deferred_branch)`
- `action`: `"merge"`, `"discard"`, or `"nothing"` (Do-nothing button, calls `leave_as_is`, returns `kept: True`).
- Resolves the tab's `AgentState`, checks it, and claims `state.is_merging` in ONE `_state_lock` section; a
  gap there once let `closeTab` or a new run dispose the worktree while `merge()` was running.
- Not `internal`: `_check_worktree_busy(state, verb, repo_root, wt_dir)` refuses while the tab's task runs
  (including the startup window, via `thread_alive()`), or while a task runs inside the worktree itself. A
  non-worktree task on the same repo's main tree blocks a discard outright, but blocks a merge only when
  `_main_tree_blocks_merge` holds (tracked files dirty, status fails, or repo unknown); the merge is then deferred.
- `internal=True` (post-task auto-finalize on the task thread) bypasses only the tab's own flags. It still
  refuses when another tab runs inside the worktree, and refuses a merge when `_main_tree_blocks_merge` holds.
  A discard is exempt from the main-tree guard (it never touches the main tree; refusing leaked worktrees).
- Merge broadcasts `worktree_progress` ("Generating commit message…"), then `wt.merge()` or
  `wt.merge(conflict_resolver=resolve_merge_conflict)` when `resolve_conflicts` (auto-commit mode); merge-agent
  spend is persisted by `_persist_merge_agent_usage` in a `finally`.
- Automatic discards pass `rescue_ignored=internal`. "Partially discarded" or "Discard deferred" results are
  not success; deferred ones are `retryable` so the webview keeps its buttons.
- `finally`: clear `is_merging`/`merge_thread`, then `_dispose_if_closed(tab_id)` so a tab closed mid-merge is
  cleaned up. `merge_thread` lets shutdown wait for an in-flight merge.

## Deferred merges
`_main_tree_blocks_merge(repo_root)` is True only when a non-worktree task runs on that repo AND
`git status --porcelain -uno` shows modified tracked files (or git fails, or repo unknown). A read-only task
next to a clean tree does not block. On refusal, `_defer_worktree_merge` records
`state.wt_merge_deferred_branch`; `_merge_deferred_worktrees(repo)` retries once the tree is clean (after a
non-worktree task finishes and commits, a Git Commit, or a main-tree Discard). A retry whose deferral was
taken over returns the `_DEFERRAL_SUPERSEDED` sentinel.

## Session resume: `_emit_pending_worktree` / `_finalize_pending_worktree`
Returns a `_PendingOutcome`:
- `NOOP`: a merge/discard is in flight or a task is active or submitted (`AgentState.busy`); do nothing.
- `FINALIZED`: auto-commit on, so it attempted a merge (changes) or discard (no changes) and broadcast the
  `worktree_result`; the action itself may have failed or been deferred.
- `PRESENT`: nothing pending, or not worktree mode.
- `PRESENT_CLAIMED`: auto-commit off or `_pending_review` set; presents Merge/Discard (`worktree_done`) under
  an `is_merging` claim released by `_release_present_claim`.
`_present_pending_worktree(discard_if_empty=...)` auto-discards an empty branch unless a task runs inside it.

## Other helpers
`_check_merge_conflict` (pure, file-overlap prediction), `_get_worktree_changed_files`,
`_handle_main_tree_action` (non-worktree bar: discard = `reset --hard` + `clean -fd` without `-x`, under `repo_lock`).

## Sources
- `src/kiss/server/merge_flow.py` (`_MergeFlowMixin._handle_worktree_action`, `_check_worktree_busy`, `_main_tree_blocks_merge`, `_defer_worktree_merge`, `_merge_deferred_worktrees`, `_emit_pending_worktree`, `_finalize_pending_worktree`, `_present_pending_worktree`, `_PendingOutcome`, `_handle_main_tree_action`)
