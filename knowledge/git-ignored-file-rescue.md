---
title: Rescuing git-ignored task output before worktree removal (rescue_ignored_files,
  .kiss-rescued-*)
uuid: d727b344-7bf7-4597-b0d8-2df23da65cd5
summary: git add -A skips ignored files, so before removing a kept worktree Sorcar
  copies task-created ignored files into the main repo, never overwriting, landing
  conflicts as <stem>.kiss-rescued-<ns><ext>.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Rescuing git-ignored task output

Auto-commit uses `git add -A`, which skips `.gitignore`d files, and teardown ends in
`git worktree remove --force`. A task whose output is ignored (a dataset in `data/`, a generated CSV, a
`.env`) would lose it, although the same task without a worktree would have left it on disk.

## `GitWorktreeOps.rescue_ignored_files(wt_dir, repo)` returns `(landed, ok)`
Called right before removing a worktree whose work is kept: `_commit_and_clean_worktree` (merge,
auto-release, preserve), reclaim, and automatic empty-branch discard (`discard(rescue_ignored=True)`). A
user-clicked Discard does not rescue.

Rules:
- Never overwrite an existing destination (file, dir, symlink). Identical bytes: skip. Different: land next to
  it as `<stem>.kiss-rescued-<ns><ext>`.
- Append `*.kiss-rescued-*` (`_RESCUE_EXCLUDE_PATTERN`) to the destination's `info/exclude` before landing so a
  sibling is never swept into a later `git add -A`; if that fails, proceed and warn about siblings that
  `git check-ignore` does not cover.
- Land atomically with `os.link` (fails on existing path, does not follow symlinks), else an
  `O_CREAT|O_EXCL` copy.
- The nearest existing ancestor of each destination must resolve inside the repo (no escape via a symlinked
  directory). Paths containing a component in `_RESCUE_SKIP_COMPONENTS` are skipped: `.git`, `.kiss-worktrees`, `.venv`, `venv`, `node_modules`, `__pycache__`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `.tox`, `.nox`, `.cache`, `.DS_Store`.
- Fail closed: `ok=False` makes callers preserve the worktree (`PRESERVED_RESCUE_FAILED`, "Discard deferred").

## Sources
- `src/kiss/agents/sorcar/git_worktree.py` (`GitWorktreeOps.rescue_ignored_files`, `list_ignored_files`, `_land_rescued_file`, `_rescue_dst_contained`, `_RESCUE_EXCLUDE_PATTERN`, `_RESCUE_SKIP_COMPONENTS`)
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` (`_commit_and_clean_worktree`, `discard`)
