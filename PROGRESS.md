# Task: Why did the PROGRESS.md merge conflict happen? Fix it in code.

## Root-cause analysis

The Task-3 conflict (`git cherry-pick --no-commit a76d5ff4..kiss/wt-...` →
conflict in PROGRESS.md) is structural, not incidental:

- PROGRESS.md is a **tracked** file that every agent task **wholesale
  rewrites** (convention: "Clear the contents of PROGRESS.md when a new
  task begins").
- The Task-2 worktree branch rewrote it to a 93-line forensic log; by
  merge time `main`'s copy was the divergent 1625-line bughunt log
  (committed by earlier merged iterations).
- Three-way merge: base = worktree fork point / baseline, ours = main's
  log, theirs = branch's log — both sides rewrote the whole file, so
  `git merge --squash` / `git cherry-pick` can never auto-merge it.
- Therefore ANY worktree merge conflicts whenever another task touched
  PROGRESS.md on main in between — i.e. almost always.

## Fix

Register a repo-local **git merge driver** that auto-resolves PROGRESS.md
content conflicts by taking the **incoming (task branch) version** — the
newest task's log wins, matching the clear-on-new-task convention.

- `GitWorktreeOps.ensure_scratch_merge_driver(repo)` in
  `src/kiss/agents/sorcar/git_worktree.py`:
  - appends `PROGRESS.md merge=kiss-scratch` to
    `<git_common_dir>/info/attributes` (untracked, like info/exclude)
  - sets repo-local config `merge.kiss-scratch.driver = cp -f %B %A`
    (%B = other side, %A = result file; exit 0 = clean merge)
  - no tracked file in the user's repo is ever modified
- Wired into `WorktreeSorcarAgent._try_setup_worktree` (next to
  `ensure_excluded`) and `WorktreeSorcarAgent._do_merge` (so pending
  worktrees created before this fix also benefit at merge time).
- Because the driver lives in repo plumbing, it also fixes the **manual**
  `git cherry-pick` / `git merge` commands shown in the conflict message.

## Failing-first test

`src/kiss/tests/agents/sorcar/test_progress_md_merge_conflict.py`:

1. ops-level squash-merge with divergent PROGRESS.md → SUCCESS, branch wins
2. ops-level baseline/cherry-pick path (exact incident replay) → SUCCESS
3. genuine conflict in a real source file → still CONFLICT
4. driver installation is idempotent
5. agent-level: WorktreeSorcarAgent.run + divergent PROGRESS.md + merge()
   → "Successfully merged" (fails pre-fix with "Merge conflict detected")

## Status

- [x] Analysis
- [ ] Test written, confirmed failing pre-fix
- [ ] Fix implemented
- [ ] Tests pass, uv run check --full clean
