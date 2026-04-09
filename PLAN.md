# Simplification Plan for `worktree_sorcar_agent.py`

## Current State

`WorktreeSorcarAgent` is 706 lines. It handles five distinct concerns in one class:

1. **Low-level git operations** — `_git()` calls scattered everywhere, each call site checking `.returncode`, parsing `.stdout`, handling errors
2. **Worktree lifecycle** — create worktree, remove worktree, prune, manage branches
3. **State management** — `_repo_root`, `_wt_branch`, `_original_branch`, crash recovery from git config
4. **Merge orchestration** — three strategies (`merge`, `manual_merge`, `discard`) with duplicated preambles
5. **LLM commit message generation** — completely unrelated concern embedded in the class

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

## Migration Path

1. Create `git_worktree.py` with `GitWorktree`, `MergeResult`, `GitWorktreeOps`
2. Move all `_git()` calls from the agent into `GitWorktreeOps` methods
3. Replace agent's three state attributes with single `_wt: GitWorktree | None`
4. Extract `_finalize_worktree()` shared preamble
5. Move `_generate_worktree_commit_message()` to utility
6. Update tests — `GitWorktreeOps` gets its own unit tests; agent tests simplify
7. Each step is independently testable and mergeable
