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

# Task: Bug-hunt iteration 3, group D (sorcar_agent.py / chat_sorcar_agent.py / useful_tools.py / web_use_tool.py) — COMPLETE

4 NEW bugs found, reproduced failing-first (10 failing assertions pre-fix), fixed;
12/12 new tests pass. Changes auto-committed in 437303f9; fixes verified intact at
HEAD after sibling-group merges.

### Bugs (file, root cause, fix, test)

1. **D-1** `useful_tools.py::Edit` (~line 450): empty `old_string` —
   `str.count("") == len+1`, so `replace_all=True` interleaved `new_string` between
   EVERY character (silent file corruption); on an empty file it silently acted as
   Write. Fix: explicit `old_string == ""` → Error suggesting the Write tool.
   Test: `test_bughunt3_edit_empty_oldstring.py` (3 tests).
1. **D-2** `sorcar_agent.py::_coerce_tasks` (~line 830): JSON `"[]"` fell through the
   `and parsed` guard → wrapped as `["[]"]` (one garbage sub-agent LLM run); JSON
   lists with non-string elements (`'[1, 2]'`) were treated as ONE task string.
   Fix: any parsed JSON list is accepted — `[]` → zero tasks, non-str elements
   str()-coerced one-per-task. Test: `test_bughunt3_coerce_tasks_json.py` (4 tests).
1. **D-3** `web_use_tool.py::_is_profile_in_use` (~line 113): corrupt/stale
   `SingletonLock` target `host-0` → `os.kill(0, 0)` signals the CALLER'S OWN process
   group and always succeeds → profile permanently "in use" →
   `_resolve_user_data_dir` escalated to `_1`, `_2`, … silently losing the user's
   logged-in browser profile. Fix: `if pid <= 0: return False` before `os.kill`.
   Test: `test_bughunt3_profile_pid0.py` (3 tests).
1. **D-4** `web_use_tool.py::WebUseTool`: `atexit.register(self.close)` per instance,
   never unregistered → one retained tool object + atexit entry leaked per agent run
   (`SorcarAgent.run` builds a fresh WebUseTool every run). Fix:
   `atexit.unregister(self.close)` in `close()`; unregister+register re-arm at top of
   `_ensure_browser` (keeps exactly one entry across relaunches).
   Test: `test_bughunt3_webuse_atexit_leak.py` (2 tests).

### Deliberately skipped (not testable without real LLM calls or mocks/fakes)

- Module-level `run_tasks_parallel` doesn't copy the parent `stop_event` into worker
  thread-locals (the ChatSorcarAgent override does — asymmetry).
- `chat_sorcar_agent.py::run` finally: result parsing as non-dict YAML persists
  summary `""` instead of the `result[:500]` fallback used on parse failure.

### Verification

- 12/12 new group-D tests pass; 196 directly-affected tests pass (useful_tools,
  web_use_tool, run_parallel\*, read_tool robustness, pwd_prefix, read_binary,
  bughunt_agent\*, update_settings\*, sorcar_agent, stateful, bash_worktree_cwd,
  change_model); 155 coverage/parallel tests pass.
- Full sorcar test dir in 4 shards: 1293 passed. Two shard-level failures
  (`test_bughunt_cli` ctrl-c forkpty flake — documented pre-existing — and
  `test_new_chat_hang` ordering under concurrent sibling-session load) both pass in
  isolation → not regressions.
- `uv run check --full`: All checks passed.
- tmp/shard\* scratch files deleted before finish.

# Task: Bug-hunt iteration 3, group F (vscode web_server.py / diff_merge.py / merge_flow.py) — COMPLETE

## Final results (group F)

4 NEW bug classes found, reproduced failing-first with real-git integration tests
(no mocks/patches/fakes), and fixed. Plus 1 cross-test pollution bug fixed in a
sibling group's new test. None overlap the 16 bugs from rounds 1-2.

### Bugs (file, root cause, fix, test)

1. **BUG-F1 — git C-quoted paths bug class** (`diff_merge.py`, `merge_flow.py`,
   `git_worktree.py`): even with `core.quotepath=false`, git C-quotes paths containing
   `"`/backslash/control chars in `diff --git` headers, `--name-only`, `ls-files --others`, and `status --porcelain` output. Effects: `_parse_diff_hunks` matched
   neither header regex so the quoted file's hunks were MISATTRIBUTED to the previous
   file in the diff; `_capture_untracked` returned names that don't exist on disk
   (file invisible in merge view); `_main_dirty_files` stripped quotes without
   unescaping; `_get_worktree_changed_files`/`_check_merge_conflict` overlap sets
   could never intersect real paths (missed conflicts); `GitWorktreeOps._diff_name_only`
   (dirty-set source for the conflict check) had the same hole. Fix: new
   `_diff_header_path()` in diff_merge.py handling the quoted header forms via
   `_unquote_git_path` (reused from git_worktree.py); unquoting applied in
   `_capture_untracked`, `_main_dirty_files` (with `_split_rename_tail`), all 4
   `--name-only` sites in merge_flow.py (new `_unquoted_name_lines` helper), and
   `_diff_name_only`. Test: `test_bughunt3_quoted_paths.py` (5 tests, incl. real
   worktree conflict-overlap scenario).
1. **BUG-F2 — deleted tracked binary invisible in merge view**
   (`diff_merge.py::_prepare_merge_view`): a binary deletion emits `Binary files … differ` with no hunks; the binary branch required `fpath.is_file()` (false when
   deleted) so the deletion fell through every branch — unreviewable and
   unrestorable, while deleted TEXT files get a `.deleted` placeholder. Fix: classify
   empty-hunk + missing-file entries as binary; binary manifest loop now creates a
   `.deleted` placeholder as `current` and keeps the real workspace path as `target`.
   Test: `test_bughunt3_deleted_binary.py` (manifest entry + end-to-end reject
   restores the original bytes).
1. **BUG-F3 — rejecting a binary merge entry crashed / would truncate**
   (`web_server.py::_reject_hunk_in_file` / `_reject_all_hunks_in_file`): binary
   entries point current/target at the real binary file; `read_text()` raised
   `UnicodeDecodeError` (NOT the caught `OSError`), crashing the merge action.
   Fix: new `_restore_base_bytes()`; `binary=True` keyword plumbed from
   `fd.get("binary")` in `_apply_web_merge_action` (via functools.partial) and
   `_reject_all_hunks_in_file`; `UnicodeDecodeError` fallback restores base bytes
   wholesale. Test: `test_bughunt3_binary_reject.py`.
1. **BUG-F4 — rejecting one hunk rewrote entire CRLF file as LF**
   (`web_server.py::_reject_hunk_in_file`): `read_text()` universal-newline
   translation + `write_text()` converted EVERY line ending to LF on any reject —
   inconsistent with `_write_base_copy`'s M5 CRLF-preservation fix. Fix: new
   `diff_merge._read_lines_preserved()`/`_split_lines_keepends()` (newline=''
   reads, \\n-only splitting matching git's line numbering); reject now writes with
   `newline=''`; `_diff_files` and `_file_as_new_hunks` switched to the same
   helpers so hunk numbering is consistent for files containing `\r`/`\f`/`\u2028`.
   Test: `test_bughunt3_crlf_reject.py` (byte-exact restoration).
1. **Cross-test pollution fix** (`test_bughunt3_newchat_phantom_tab.py`, written by
   sibling group E): its `selectModel m-x` persisted `last_model: "m-x"` into
   config.json via `_record_model_usage`, poisoning every later `SorcarAgent` run in
   the same pytest process (`model_name or _load_last_model()`) — deterministically
   broke `test_no_git_error_on_empty_worktree` under shard ordering (bisected to the
   single polluter file). Fix: snapshot/restore `last_model` in setUp/tearDown.

### Explicitly checked, not bugs

- `_check_merge_conflict` `baseline^` on a root commit: unreachable — the baseline is
  always committed on top of the original branch tip in the worktree.
- Mode-only changes (old/new mode, no hunks): excluded from merge view by design,
  worktree flow falls back to `worktree_done` consistently.
- `/media/` static serving path traversal: guarded by `resolve().is_relative_to`;
  absolute-path join trick also rejected. Malformed JSON / non-dict WS + UDS frames:
  skipped safely. `_trajectory_job_response`: job names with `/`, `\`, `..` rejected.

### Verification

- All 8 new group-F tests fail pre-fix (7 directly; conflict-overlap test passed
  pre-fix only because BOTH sides kept quoted names — kept as regression guard) and
  pass post-fix.
- Full vscode+sorcar suite (284 files) in 8 parallel shards: all pass after the
  pollution fix; only `test_bughunt_cli` ctrl-c forkpty flake (documented
  pre-existing) failed once under load, passes in isolation.
- `uv run check --full`: All checks passed.
