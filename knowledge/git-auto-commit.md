---
title: Auto-commit of task work (auto_commit_changes, auto_commit_mode toggle)
uuid: 5a13eb06-aa9b-4156-b834-4fc806939083
summary: 'How tasks auto-commit: auto_commit_changes stages twice around the LLM call,
  auto_commit_mode in ~/.kiss/config.json, force_commit on merge, commit toasts, late-arriver
  retry.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Auto-commit of task work

## Engine: `sorcar_agent.auto_commit_changes(commit_dir, user_prompt, message_fn, notify_fn, task_result)`
1. `stage_all`; return False if `has_staged_changes` is false (no toasts then).
2. `notify_fn("generating", "")`, then `message_fn` (normally `_generate_commit_message`: staged diff to
   `generate_commit_message_from_diff`). On exception: fallback `"kiss: auto-commit agent changes"` plus
   the prompt/result blocks.
3. `stage_all` AGAIN, then `commit_staged`. The second stage catches files written during the slow LLM call
   (`PROGRESS.md`, `.DS_Store`, editor swap files); without it `_finalize_worktree` saw leftovers and reported a
   misleading "pre-commit hook may have rejected" warning.
4. `notify_fn("committed", subject)` or `notify_fn("failed", "")` so the sticky toast always ends.

## Worktree path
- `WorktreeSorcarAgent._auto_commit_worktree(force_commit)` wraps the engine, no-op unless
  `auto_commit_enabled or force_commit`. Toast id `autocommit-<ns>` is bound per call with `functools.partial`
  so both stages update one toast (`_broadcast_commit_notification`).
- `_commit_and_clean_worktree` then retries once with `commit_all(..., "kiss: auto-commit late-arriving changes")`
  when changes remain (only if auto-commit is on or forced).
- Setting: `auto_commit_mode` in `~/.kiss/config.json` (default True), read by `_config_auto_commit_enabled`,
  re-read on every `run`; the `auto_commit` run kwarg overrides it. `merge()` always passes `force_commit=True`
  because merging is the user's explicit decision; automatic paths obey the toggle.

## Non-worktree path (server)
`_MergeFlowMixin._autocommit_changes` commits the tab's work_dir repo after a non-worktree task (events
`autocommit_progress` / `autocommit_done`); `manual=True` is the Git Commit button (message from diff only,
no prompt/result blocks, toast notifications). `_autocommit_changed_repos` then commits only the recorded
`Write`/`Edit` paths in OTHER repos the task touched (never `add -A` there, skips `.kiss-worktrees`); files
changed via Bash outside the work_dir repo are not tracked.

## Sources
- `src/kiss/agents/sorcar/sorcar_agent.py` (`auto_commit_changes`, `_generate_commit_message`, `_commit_subject`)
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` (`_config_auto_commit_enabled`, `_auto_commit_worktree`, `_broadcast_commit_notification`, `_commit_and_clean_worktree`)
- `src/kiss/server/merge_flow.py` (`_autocommit_changes`, `_autocommit_changed_repos`)
