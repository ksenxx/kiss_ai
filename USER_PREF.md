# User Preferences

- When auditing code, write tests that CONFIRM bugs exist (assertions pass when buggy behavior is present)
- Worktree-related code spans: `git_worktree.py`, `worktree_sorcar_agent.py`, `stateful_sorcar_agent.py`, `server.py` (VSCode integration), `persistence.py`
- Test helpers pattern: `_redirect_db`, `_restore_db`, `_make_repo`, `_patch_super_run`, `_unpatch_super_run` — reused across worktree test files
- The `_git` function in `git_worktree.py` (keyword `cwd`) differs from `_git` in `diff_merge.py` (positional `cwd`) — be careful with imports
- Use `setup_method`/`teardown_method` pattern (not pytest fixtures) for worktree tests
- Known bugs: BUG-1 through BUG-7 + INC-1/INC-2 are in test_worktree_audit.py / test_worktree_audit2.py
- New bugs BUG-8 through BUG-11 are in test_worktree_audit3.py
