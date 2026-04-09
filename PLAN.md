# Simplification Plan for `worktree_sorcar_agent.py`

## Current State

`WorktreeSorcarAgent` is 706 lines. It handles five distinct concerns in one class:

1. **Low-level git operations** — `_git()` calls scattered everywhere, each call site checking `.returncode`, parsing `.stdout`, handling errors
1. **Worktree lifecycle** — create worktree, remove worktree, prune, manage branches
1. **State management** — `_repo_root`, `_wt_branch`, `_original_branch`, crash recovery from git config
1. **Merge orchestration** — three strategies (`merge`, `manual_merge`, `discard`) with duplicated preambles
1. **LLM commit message generation** — completely unrelated concern embedded in the class

## Root Causes of Complexity

### 1. `run()` is a 12-step monolith (~80 lines)

The method does repo discovery, state restoration, detached HEAD detection, subdirectory offset calculation, git-exclude setup, branch name generation, collision detection, worktree creation, git config storage, directory creation, task delegation, and error wrapping — all in one linear flow with 6 different fallback-to-`super().run()` exit points.

### 2. Three merge paths duplicate a shared preamble

`merge()`, `manual_merge()`, and (partially) `discard()` all repeat the pattern:

```
auto_commit → remove worktree → prune → checkout original
```

This is ~15 lines of near-identical code in each method.

### 3. Raw `_git()` calls leak abstraction

Every git interaction requires: call `_git(...)` → check `.returncode` → parse `.stdout.strip()` → handle failure. This pattern repeats ~30 times. The caller must know which git subcommands to use and how to interpret their output.

### 4. State is three loosely-coupled attributes

`_repo_root`, `_wt_branch`, and `_original_branch` move together as a unit but are stored as independent attributes with `None` sentinels. `_wt_dir` is derived from two of them. `_wt_pending` is derived from one. The restore-from-git logic reconstructs all three from git queries, but is only valid when all three are consistent.

### 5. Commit message generation is an unrelated concern

`_generate_worktree_commit_message()` imports `KISSAgent`, calls an LLM, and has its own error handling. It's ~30 lines that has nothing to do with worktree management — it's only called from one place (`_auto_commit_worktree`).

## Proposed Simplification

### Extract `GitWorktree` dataclass + `GitWorktreeOps` helper

Create a small module `git_worktree.py` with two things:

**`GitWorktree`** — a frozen dataclass holding the worktree state:

```python
@dataclass(frozen=True)
class GitWorktree:
    repo_root: Path
    branch: str
    original_branch: str
    wt_dir: Path
```

**`GitWorktreeOps`** — a stateless helper class with all git worktree operations:

```python
class GitWorktreeOps:
    @staticmethod
    def discover_repo(path: Path) -> Path | None: ...

    @staticmethod
    def current_branch(repo: Path) -> str | None: ...

    @staticmethod
    def create(repo: Path, branch: str, wt_dir: Path) -> None: ...

    @staticmethod
    def remove(repo: Path, wt_dir: Path) -> None: ...

    @staticmethod
    def commit_all(wt_dir: Path, message: str) -> bool: ...

    @staticmethod
    def merge_branch(repo: Path, branch: str, no_commit: bool = False) -> MergeResult: ...

    @staticmethod
    def delete_branch(repo: Path, branch: str) -> None: ...

    @staticmethod
    def checkout(repo: Path, branch: str) -> bool: ...

    @staticmethod
    def ensure_excluded(repo: Path) -> None: ...

    @staticmethod
    def find_pending_branch(repo: Path, prefix: str) -> str | None: ...

    @staticmethod
    def load_original_branch(repo: Path, branch: str) -> str | None: ...

    @staticmethod
    def save_original_branch(repo: Path, branch: str, original: str) -> None: ...

    @staticmethod
    def cleanup_orphans(repo: Path) -> str: ...
```

The return types become meaningful (e.g., `MergeResult` enum: `SUCCESS`, `CONFLICT`, `CHECKOUT_FAILED`) instead of callers parsing returncode and stderr.

### Simplify `WorktreeSorcarAgent`

With `GitWorktreeOps`, the agent shrinks to pure orchestration:

```python
class WorktreeSorcarAgent(StatefulSorcarAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._wt: GitWorktree | None = None

    def run(self, prompt_template="", **kwargs) -> str:
        repo = GitWorktreeOps.discover_repo(work_dir or Path.cwd())
        if repo is None:
            return super().run(...)

        self._restore_pending(repo)
        if self._wt is not None:
            return self._blocked_message()

        original = GitWorktreeOps.current_branch(repo)
        if original is None:
            return super().run(...)  # detached HEAD

        wt = self._create_worktree(repo, original)
        if wt is None:
            return super().run(...)  # creation failed

        self._wt = wt
        kwargs["work_dir"] = str(wt.wt_dir / offset)
        result = super().run(prompt_template=prompt_template, **kwargs)
        return result + "\n\n---\n" + self.merge_instructions()

    def merge(self) -> str:
        wt = self._require_pending()
        self._finalize_worktree(wt)
        result = GitWorktreeOps.merge_branch(wt.repo_root, wt.branch)
        if result == MergeResult.SUCCESS:
            GitWorktreeOps.delete_branch(wt.repo_root, wt.branch)
            self._wt = None
            return f"Successfully merged '{wt.branch}'."
        ...

    def discard(self) -> str:
        wt = self._require_pending()
        GitWorktreeOps.remove(wt.repo_root, wt.wt_dir)
        GitWorktreeOps.delete_branch(wt.repo_root, wt.branch)
        self._wt = None
        return f"Discarded '{wt.branch}'."

    def _finalize_worktree(self, wt: GitWorktree) -> None:
        """Shared preamble: auto-commit, remove worktree, prune, checkout."""
        self._auto_commit(wt)
        GitWorktreeOps.remove(wt.repo_root, wt.wt_dir)
        GitWorktreeOps.checkout(wt.repo_root, wt.original_branch)
```

### Move commit message generation to a utility

```python
# In a utility module or as a standalone function
def generate_commit_message(diff_text: str) -> str:
    """Ask a fast LLM for a conventional-commit message, with fallback."""
    ...
```

This removes the LLM import and error handling from the worktree agent entirely. `_auto_commit` just calls `generate_commit_message(diff)` and passes the result to `GitWorktreeOps.commit_all()`.

## Expected Result

| Metric | Before | After |
|--------|--------|-------|
| `worktree_sorcar_agent.py` lines | ~706 | ~250 |
| `git_worktree.py` (new) | — | ~200 |
| Distinct concerns per file | 5 | 1–2 |
| `_git()` calls in agent | ~30 | 0 |
| Duplicated merge preamble | 3× | 1× |
| `run()` method lines | ~80 | ~25 |

The agent becomes pure orchestration: "when to create/merge/discard a worktree" — while `GitWorktreeOps` handles "how to interact with git." The two can be tested independently: `GitWorktreeOps` with real git repos, the agent with a simple `GitWorktreeOps` interface.

## Correctness Issues Found in This Plan

### Issue 1: `GitWorktree.original_branch` must be `str | None`, not `str`

The plan proposes `original_branch: str` as a required field in the frozen
dataclass. However, the current code allows `_original_branch = None` when:

- The `branch.{name}.kiss-original` git config is missing (crash between
  worktree creation and config write)
- AND the main worktree has a detached HEAD (so fallback also fails)

This state is exercised by `test_missing_config_detached_head`. In this
state, `merge()` and `manual_merge()` return a graceful error message
("Cannot merge: original branch is unknown") rather than crashing.

**Fix:** Use `original_branch: str | None` in the dataclass, or keep
`_repo_root` and `_wt_branch` as separate attributes alongside the
dataclass (undermining the point of the refactor). The `| None` approach
is simpler.

### Issue 2: `_finalize_worktree()` pseudocode is missing `worktree prune`

The plan shows:

```python
def _finalize_worktree(self, wt):
    self._auto_commit(wt)
    GitWorktreeOps.remove(wt.repo_root, wt.wt_dir)
    GitWorktreeOps.checkout(wt.repo_root, wt.original_branch)
```

But the current code calls `git worktree prune` between remove and
checkout in both `merge()` and `manual_merge()`. The prune step is
necessary for git to properly clean up worktree bookkeeping entries.

**Fix:** Add `GitWorktreeOps.prune(repo)` call after remove.

### Issue 3: `discard()` pseudocode is missing `worktree prune`

Same issue — current code calls prune between remove and branch delete:

```python
_git("worktree", "remove", ...)
_git("worktree", "prune", ...)     # ← missing from plan
self._delete_branch(...)
```

**Fix:** Add prune to the discard pseudocode.

### Issue 4: `merge()` pseudocode omits conflict handling

The plan's `merge()` shows the success path but omits:

- `git merge --abort` call on conflict (current code aborts to leave a
  clean worktree)
- Preserving `_wt_branch` / `_wt` state on conflict (so `discard()`
  still works)
- Returning manual resolution instructions with exact CLI commands

**Fix:** Add conflict branch to the pseudocode showing abort, state
preservation, and instructions.

### Issue 5: `manual_merge()` is not addressed

`manual_merge()` is a public method with complex behavior:

- Uses `--no-commit --no-ff` (two flags, not just `no_commit=True`)
- Detects conflicts by checking "CONFLICT" in stdout+stderr
- On success: calls `git reset HEAD` to unstage changes for user review
- On success: deletes the task branch
- On conflict: does NOT delete the task branch (preserved for reference)
- On conflict: does NOT abort (leaves conflict markers for user)
- Clears agent state (`_wt_branch = None`) in ALL cases

The proposed `merge_branch(repo, branch, no_commit=False)` signature
is insufficient — it doesn't distinguish the conflict vs. non-conflict
failure cases, and doesn't handle the unstage/branch-delete logic.

**Fix:** Either add a dedicated `GitWorktreeOps.manual_merge_branch()`
method, or expand `MergeResult` to include `CONFLICT_NO_COMMIT` and
`NON_CONFLICT_FAILURE` variants, and keep the unstage/branch logic in
the agent's `manual_merge()`.

### Issue 6: `cleanup()` static method is not addressed

`cleanup()` is a public static method that uses `_git()` directly for
branch listing, worktree listing, branch deletion, config removal, and
prune. It should either move to `GitWorktreeOps.cleanup_orphans()` or
be explicitly called out in the migration plan.

### Issue 7: `_cleanup_partial_worktree()` is not addressed

Called in `run()` when worktree creation or config write fails. Uses
`_git("worktree", "remove", ...)`, prune, and `_delete_branch()`. Should
become a `GitWorktreeOps` method.

### Issue 8: `_delete_branch()` has a fallback (`-d` → `-D`)

The plan's `GitWorktreeOps.delete_branch()` doesn't mention this
two-step pattern. The current code tries safe delete (`-d`) first,
falls back to force delete (`-D`) on failure, and also removes the
`branch.{name}.*` config section. All three steps must be preserved.

### Issue 9: `main()` CLI entry point references internal state

`main()` calls `agent._wt_pending`, `agent.merge()`, `agent.discard()`.
The `_wt_pending` property depends on the internal state representation.
If `_wt_branch` becomes `_wt: GitWorktree | None`, the property must be
updated (trivial but must not be forgotten).

### Issue 10: `merge_instructions()` uses all four state attributes

`merge_instructions()` reads `_wt_branch`, `_wt_dir`, `_original_branch`,
and `_repo_root`. With the `_wt: GitWorktree | None` approach, these
become `self._wt.branch`, etc. — straightforward but needs updating.

## Migration Path

1. Create `git_worktree.py` with `GitWorktree`, `MergeResult`, `GitWorktreeOps`
1. Move all `_git()` calls from the agent into `GitWorktreeOps` methods
1. Replace agent's three state attributes with single `_wt: GitWorktree | None`
1. Extract `_finalize_worktree()` shared preamble
1. Move `_generate_worktree_commit_message()` to utility
1. Update tests — `GitWorktreeOps` gets its own unit tests; agent tests simplify
1. Each step is independently testable and mergeable
