---
title: Worktree plumbing - info/exclude, node_modules symlinks, kiss-scratch merge
  driver
uuid: 92734743-0102-472d-98d2-4d6576730ea9
summary: ensure_excluded adds .kiss-worktrees/ to info/exclude, link_node_modules
  symlinks the main checkout's ignored node_modules into a new worktree, ensure_scratch_merge_driver
  auto-resolves PROGRESS.md.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Worktree plumbing: excludes, node_modules, scratch merge driver

All three helpers write only untracked plumbing under `<git_common_dir>/info/` and repo-local
config, never a tracked file of the user's repo.

## `GitWorktreeOps.ensure_excluded(repo)`
Appends `.kiss-worktrees/` to `<git_common_dir>/info/exclude` (not `.gitignore`). Without it the
worktree directory shows as untracked in `git status` and trips the "main tree is dirty" guards
(reclaim calls it itself for that reason).

## `GitWorktreeOps.link_node_modules(repo, wt_dir)`
A fresh worktree has tracked files only, so git-ignored `node_modules` are missing and JS tests,
`npm run compile` and the extension lint fail (an audit of sorcar.db found tasks reinstalling with
`npm install`). This helper:
- lists ignored directories named `node_modules` via `git ls-files --ignored --directory` in the main checkout;
- symlinks each at the same relative path in the worktree if the path is free and its parent exists;
- adds the bare pattern `node_modules` to `info/exclude`, because `node_modules/` patterns match
  directories only, not symlinks, and the link would otherwise be picked up by `git add -A` and the squash merge.
- Deliberately does not link compiled `out/`: it must be rebuilt from the worktree's own sources
  (in this repo: `npm run compile` in `src/kiss/agents/vscode`).
Called at the end of `_try_setup_worktree`; an `OSError` is logged, not fatal.

## `GitWorktreeOps.ensure_scratch_merge_driver(repo)`
Registers a repo-local merge driver `kiss-scratch` and an `info/attributes` line
`<path> merge=kiss-scratch` for agent scratch files (`PROGRESS.md`, `src/kiss/INJECTIONS.md`). On a
content conflict the driver (`cp -f %B %A`) keeps the incoming branch's version: each task rewrites these files,
so the newest task wins. This stops scratch files from blocking whole worktree merges, and also applies
to the manual merge/cherry-pick commands shown to the user. Installed in `_try_setup_worktree` and at
the start of `_do_merge`.

## Sources
- `src/kiss/agents/sorcar/git_worktree.py` (`GitWorktreeOps.ensure_excluded`, `link_node_modules`, `ensure_scratch_merge_driver`, `_append_info_line`)
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` (`_try_setup_worktree`, `_do_merge`)
