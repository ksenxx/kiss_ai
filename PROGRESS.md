# PROGRESS — settings-panel "Git Commit" failed with "Not a git repository" after worktree task

## Task

User report:

> in the last task when I pressed "git commit" it says
> "Not a git repository."

The previous task ran in worktree mode and the worktree directory
under `<repo>/.kiss-worktrees/kiss_wt-*` was removed on cleanup.
When the user then clicked the settings-panel **Git Commit** button,
the autocommit flow rejected with "Not a git repository." even
though the main working tree IS a git repo with uncommitted
changes.

## Investigation

1. Frontend (`media/main.js`) wires the **Git Commit** button to
   post `autocommitAction` with
   `workDir: workDirForTab(activeTabId)`.
1. `workDirForTab` returns `tab.workDir` — set from each task event's
   `extra.work_dir`. For a worktree task `extra.work_dir` points at
   `<repo>/.kiss-worktrees/kiss_wt-<…>/<offset>`.
1. After the worktree task ends (merge or discard),
   `git worktree remove` deletes that directory, but **the tab's
   `workDir` is never reset to the parent repo**.
1. Server-side `_MergeFlowMixin._handle_autocommit_action`
   (`merge_flow.py`) ran
   `GitWorktreeOps.discover_repo(Path(work_dir))` → `git -C <stale>`
   which fails because the directory no longer exists; the handler
   then broadcasts `autocommit_done` with `"Not a git repository."`.

`useful_tools.py` already has `_stale_worktree_fallback(resolved)`
that strips a `.kiss-worktrees/kiss_wt-*` segment from a path. The
read-path in `useful_tools.py` uses it to recover from stale
worktree paths after cleanup. The autocommit flow did not.

## Fix

`src/kiss/agents/vscode/merge_flow.py` —
`_handle_autocommit_action`:

- Before `discover_repo`, if `Path(work_dir)` does not exist, try
  `_stale_worktree_fallback`. When it returns a non-None equivalent
  path inside the parent repo, rewrite `work_dir` (and `work_path`)
  so all subsequent `_git(work_dir, …)` calls — `add -A`,
  `diff --cached`, `commit` via `repo_lock` — operate on the main
  working tree.

```python
work_path = Path(work_dir)
if not work_path.exists():
    fallback = _stale_worktree_fallback(work_path)
    if fallback is not None:
        work_dir = str(fallback)
        work_path = fallback
repo = GitWorktreeOps.discover_repo(work_path)
```

Also added the import:

```python
from kiss.agents.sorcar.useful_tools import _stale_worktree_fallback
```

## Test

New integration test
`src/kiss/tests/agents/vscode/test_git_commit_after_worktree_cleanup.py`
reproduces the bug and verifies the fix:

1. Init a real git repo with a seed commit.
1. Dirty `edited.txt` in the main working tree.
1. Call
   `server._handle_autocommit_action("commit", tab_id="t-stale", work_dir=<repo>/.kiss-worktrees/kiss_wt-…)`
   where the worktree path does NOT exist on disk.
1. Assert (a) no "Not a git repository" message, (b) `autocommit_done`
   reports `success=True, committed=True`, (c) HEAD advanced, and
   (d) `git status --porcelain` is clean afterward.

## Verification

- `uv run pytest src/kiss/tests/agents/vscode/test_git_commit_after_worktree_cleanup.py … test_merge_autocommit_lifecycle.py -v` →
  55 passed (1 new + 54 pre-existing related tests).
- `uv run check --full` → ✅ All checks passed (install, generate
  API docs, compileall, ruff, mypy, pyright, npm vscode check,
  mdformat).
