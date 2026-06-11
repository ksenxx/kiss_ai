# Task: Bug-hunt iteration 3, group B (sorcar worktree: git_worktree.py / worktree_sorcar_agent.py) — COMPLETE

(Previous session: group E — vscode backend — completed 6 bugs; see git history of
PROGRESS.md @ bec2996e for its full log.)

## Final results (group B)

4 NEW bugs found, reproduced failing-first with real-git integration tests (no mocks),
and fixed. None overlap the 16 bugs from rounds 1-2 or the 18 worktree audit rounds.

### Bugs (file, root cause, fix, test)

1. **BUG-WT-A** `worktree_sorcar_agent.py::merge()`: when `wt_dir` was already deleted
   from disk (crashed cleanup, manual `rm -rf`) but still registered in git's worktree
   bookkeeping, `merge()` skipped `_finalize_worktree()` (the only step that runs
   `git worktree prune`). `git branch -d/-D` then refused deletion ("used by worktree
   at ..."), `_do_merge` ignores `delete_branch`'s result on SUCCESS → every such merge
   silently leaked a permanent `kiss/wt-*` branch. Fix: call `_finalize_worktree()`
   unconditionally (it internally guards on dir existence and always prunes).
   Test: `test_bughunt3_merge_stale_worktree.py` (2 tests: squash path + baseline
   cherry-pick path).
1. **BUG-WT-B** `worktree_sorcar_agent.py::_try_setup_worktree`: unguarded
   `GitWorktreeOps.copy_dirty_state` — a dirty file the process cannot read (mode 000)
   made `shutil.copy2` raise PermissionError which propagated out of `run()` (killing
   the task) and left a half-created worktree dir + branch behind, violating run()'s
   documented fallback contract. Fix: wrap in try/except OSError → log,
   `cleanup_partial`, return None (fall back to direct execution).
   Test: `test_bughunt3_unreadable_dirty_file.py` (skipped when euid==0).
1. **BUG-WT-C** `git_worktree.py::GitWorktreeOps.remove`: single
   `git worktree remove --force` attempt; corrupted worktrees (deleted `.git` link
   file) fail git's validation and LOCKED worktrees (`git worktree lock`) need
   `--force` twice — in both cases the directory survived forever while `discard()`
   reported success (corrupt) or left branch + registration + dir behind (locked).
   Fix: escalate — `--force` → `--force --force` → `rmtree(ignore_errors)` + prune.
   Test: `test_bughunt3_discard_corrupt_locked.py` (2 tests).
1. **BUG-WT-D** `git_worktree.py::cleanup_orphans`: mutated the repo (force-deletes
   branches, rmtrees every dir under `.kiss-worktrees/` missing from a point-in-time
   `git worktree list` snapshot) WITHOUT holding `repo_lock`, while
   `_try_setup_worktree`/`_do_merge`/`discard` all serialize under it (RACE-2
   invariant). A cleanup racing a concurrent task start could rmtree a worktree
   registered after the snapshot. Fix: public `cleanup_orphans` now wraps
   `_cleanup_orphans_locked` in `with repo_lock(repo)`.
   Test: `test_bughunt3_cleanup_repo_lock.py` (real threads: cleanup must block while
   another thread holds the lock, then proceed).

### Not-bugs verified by probes (real git repos under tmp/probe, since deleted)

- cherry-pick `-n` of redundant/already-applied commits: exits 0 (no false CONFLICT).
- cherry-pick `-n` multi-commit conflict: `--abort` restores a fully clean tree.
- Renames of files whose names contain literal `" -> "`: git quotes the side, and
  `_split_rename_tail` + `_unquote_git_path` parse it correctly.
- copy_dirty_state: dirty submodules (skipped cleanly, no crash), unicode + space
  paths, exec-bit-only changes, no-trailing-newline/CRLF files — all mirrored
  correctly. Unborn-branch and detached-HEAD repos fall back to direct execution;
  shallow clones work end-to-end.

### Also fixed (pre-existing, found by `uv run check --full`)

- `vscode/web_server.py::_reject_hunk_in_file`: pyright `reportPossiblyUnboundVariable`
  on `cur_lines` (false positive, but real lint failure blocking CI) — initialize
  `cur_lines: list[str] = []` before the try. Behavior unchanged; 200 impacted
  vscode tests pass.

### Verification

- All 7 new tests failed pre-fix, pass post-fix.
- 387 impacted sorcar worktree/autocommit/workflow tests run in 8 parallel shards: all
  pass. 100 coverage-branch tests pass. 200 vscode merge/web_server tests pass.
- `uv run check --full`: all checks pass.
- tmp/probe, tmp/shards scratch dirs deleted before finish.
