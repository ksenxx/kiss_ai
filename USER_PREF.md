# User Preferences

- When auditing code, write tests that CONFIRM bugs exist (assertions pass when buggy behavior is present)
- Worktree-related code spans: `git_worktree.py`, `worktree_sorcar_agent.py`, `stateful_sorcar_agent.py`, `server.py` (VSCode integration), `persistence.py`
- Test helpers pattern: `_redirect_db`, `_restore_db`, `_make_repo`, `_patch_super_run`, `_unpatch_super_run` — reused across worktree test files
- The `_git` function in `git_worktree.py` (keyword `cwd`) differs from `_git` in `diff_merge.py` (positional `cwd`) — be careful with imports
- Use `setup_method`/`teardown_method` pattern (not pytest fixtures) for worktree tests
- Known bugs: BUG-1 through BUG-7 + INC-1/INC-2 are in test_worktree_audit.py / test_worktree_audit2.py
- BUG-8 through BUG-11 (found in audit3) are now FIXED in server.py; test_worktree_audit3.py verifies correct behavior
- When testing `_replay_session`, must persist at least one chat event via `_append_chat_event` — otherwise the function returns early due to `not result.get("events")` check
- `_check_merge_conflict` is now a pure query (no auto-commit side effect); uses fork-point-based file overlap detection
- `_get_worktree_changed_files` now uses `git merge-base` to find fork point, avoiding false positives when original branch advances
- `_replay_session` now restores `tab.use_worktree` from persisted `extra` JSON data
- The canonical `SYSTEM.md` lives at `src/kiss/SYSTEM.md`; root-level `SYSTEM.md` is a duplicate that can be removed once `release.sh` is also updated to point to `src/kiss/SYSTEM.md`
