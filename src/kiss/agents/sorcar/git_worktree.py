"""Git worktree operations and state.

Provides :class:`GitWorktree` (frozen dataclass for worktree state),
:class:`MergeResult` (outcome enum), and :class:`GitWorktreeOps`
(stateless helper with all git worktree operations).
"""

from __future__ import annotations

import enum
import logging
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_repo_locks: dict[str, threading.Lock] = {}
_repo_locks_guard = threading.Lock()


def repo_lock(repo: Path) -> threading.Lock:
    """Return a per-repo threading lock for multi-step git operations.

    Concurrent tabs operating on the same main repository must
    serialize their checkout → stash → merge → pop sequences to
    prevent interleaving that could corrupt the working tree.

    Args:
        repo: Git repo root path.

    Returns:
        A :class:`threading.Lock` specific to the resolved repo path.
    """
    key = str(repo.resolve())
    with _repo_locks_guard:
        if key not in _repo_locks:
            _repo_locks[key] = threading.Lock()
        return _repo_locks[key]


def _git(
    *args: str,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a git command, returning the CompletedProcess result.

    Args:
        *args: Git sub-command and arguments (without the leading ``git``).
        cwd: Working directory for the git command.

    Returns:
        The completed process with stdout/stderr captured as text.
    """
    cmd = ["git"]
    if cwd is not None:
        cmd += ["-C", str(cwd)]
    cmd += list(args)
    return subprocess.run(cmd, capture_output=True, text=True)


@dataclass(frozen=True)
class GitWorktree:
    """Immutable snapshot of a pending worktree task.

    Attributes:
        repo_root: Git repo root path.
        branch: Branch name of the worktree task.
        original_branch: The branch the user was on when the task started,
            or ``None`` if unknown (crash between creation and config write).
        wt_dir: Worktree directory path.
        baseline_commit: SHA of the initial commit that captured the user's
            dirty state (staged, unstaged, untracked files) at worktree
            creation time.  ``None`` when the main worktree was clean or
            for legacy worktrees created before baseline support.
    """

    repo_root: Path
    branch: str
    original_branch: str | None
    wt_dir: Path
    baseline_commit: str | None = None


class MergeResult(enum.Enum):
    """Outcome of a merge operation."""

    SUCCESS = "success"
    CONFLICT = "conflict"
    CHECKOUT_FAILED = "checkout_failed"
    MERGE_FAILED = "merge_failed"


def _unquote_git_path(path: str) -> str:
    """Unquote a C-style quoted filename from ``git status --porcelain``.

    Git quotes filenames containing non-ASCII bytes (>0x7F), control
    characters, double-quotes, or backslashes.  The quoted form is
    surrounded by double-quotes with C-style escape sequences
    (``\\n``, ``\\t``, ``\\\\``, ``\\"``, ``\\NNN`` octal).

    When the path is not quoted (no surrounding double-quotes), it is
    returned unchanged.

    Args:
        path: Raw filename string from ``git status --porcelain`` output.

    Returns:
        The unquoted filename.
    """
    if not (path.startswith('"') and path.endswith('"')):
        return path
    inner = path[1:-1]
    raw = bytearray()
    i = 0
    _esc = {
        "n": 0x0A,
        "t": 0x09,
        "\\": 0x5C,
        '"': 0x22,
        "a": 0x07,
        "b": 0x08,
        "f": 0x0C,
        "r": 0x0D,
        "v": 0x0B,
    }
    while i < len(inner):
        if inner[i] == "\\" and i + 1 < len(inner):
            nxt = inner[i + 1]
            if nxt in _esc:
                raw.append(_esc[nxt])
                i += 2
            elif (
                nxt.isdigit()
                and i + 3 < len(inner)
                and inner[i + 2].isdigit()
                and inner[i + 3].isdigit()
            ):
                raw.append(int(inner[i + 1 : i + 4], 8))
                i += 4
            else:
                raw.append(ord("\\"))
                i += 1
        else:
            raw.extend(inner[i].encode("utf-8"))
            i += 1
    return raw.decode("utf-8", errors="surrogateescape")


class GitWorktreeOps:
    """Stateless helper class with all git worktree operations.

    Every method is a ``@staticmethod`` — no instance state.  All git
    interactions are encapsulated here so callers never need to parse
    returncode or stderr.
    """

    @staticmethod
    def discover_repo(path: Path) -> Path | None:
        """Find the git repo root containing *path*.

        Args:
            path: Directory to start searching from.

        Returns:
            The repo root path, or ``None`` if *path* is not in a repo.
        """
        result = _git("rev-parse", "--show-toplevel", cwd=path)
        if result.returncode != 0:
            return None
        return Path(result.stdout.strip())

    @staticmethod
    def current_branch(repo: Path) -> str | None:
        """Return the current branch name, or ``None`` for detached HEAD.

        Args:
            repo: Git repo root path.

        Returns:
            Branch name string, or ``None`` if HEAD is detached or empty.
        """
        result = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo)
        branch = result.stdout.strip()
        if not branch or branch == "HEAD":
            return None
        return branch

    @staticmethod
    def create(repo: Path, branch: str, wt_dir: Path) -> bool:
        """Create a new worktree with a new branch.

        Args:
            repo: Git repo root path.
            branch: New branch name to create.
            wt_dir: Directory for the new worktree.

        Returns:
            True if worktree was created successfully, False otherwise.
        """
        result = _git("worktree", "add", "-b", branch, str(wt_dir), cwd=repo)
        if result.returncode != 0:
            logger.warning("Failed to create worktree: %s", result.stderr.strip())
            return False
        return True

    @staticmethod
    def remove(repo: Path, wt_dir: Path) -> None:
        """Remove a worktree directory (best-effort, force).

        Args:
            repo: Git repo root path.
            wt_dir: Worktree directory to remove.
        """
        if wt_dir.exists():
            result = _git("worktree", "remove", str(wt_dir), "--force", cwd=repo)
            if result.returncode != 0:  # pragma: no cover — lock/perm
                logger.warning("worktree remove failed: %s", result.stderr.strip())

    @staticmethod
    def prune(repo: Path) -> None:
        """Prune stale worktree bookkeeping entries.

        Args:
            repo: Git repo root path.
        """
        _git("worktree", "prune", cwd=repo)

    @staticmethod
    def stage_all(wt_dir: Path) -> None:
        """Stage all changes in the worktree (``git add -A``).

        Args:
            wt_dir: Worktree directory.
        """
        _git("add", "-A", cwd=wt_dir)

    @staticmethod
    def commit_all(wt_dir: Path, message: str) -> bool:
        """Stage all changes and commit in the worktree.

        Args:
            wt_dir: Worktree directory.
            message: Commit message.

        Returns:
            True if a commit was created, False if nothing to commit
            or the commit failed (e.g. pre-commit hook rejection).
        """
        _git("add", "-A", cwd=wt_dir)
        diff = _git("diff", "--cached", "--quiet", cwd=wt_dir)
        if diff.returncode == 0:
            return False
        result = _git("commit", "-m", message, cwd=wt_dir)
        if result.returncode != 0:
            logger.warning(
                "git commit failed: %s",
                result.stderr.strip(),
            )
            return False
        return True

    @staticmethod
    def commit_staged(
        wt_dir: Path,
        message: str,
        *,
        no_verify: bool = False,
    ) -> bool:
        """Commit already-staged changes without re-staging.

        Unlike :meth:`commit_all`, this does **not** run ``git add -A``
        first.  Use when the caller has already staged the desired
        changes (e.g. via :meth:`stage_all`).

        Args:
            wt_dir: Worktree directory with pre-staged changes.
            message: Commit message.
            no_verify: If True, pass ``--no-verify`` to skip pre-commit
                and commit-msg hooks.  Use for infrastructure commits
                (e.g. baseline snapshots) that must always succeed.

        Returns:
            True if a commit was created, False if nothing was staged
            or the commit failed (e.g. pre-commit hook rejection).
        """
        diff = _git("diff", "--cached", "--quiet", cwd=wt_dir)
        if diff.returncode == 0:
            return False
        cmd = ["commit", "-m", message]
        if no_verify:
            cmd.append("--no-verify")
        result = _git(*cmd, cwd=wt_dir)
        if result.returncode != 0:
            logger.warning(
                "git commit failed: %s",
                result.stderr.strip(),
            )
            return False
        return True

    @staticmethod
    def has_uncommitted_changes(wt_dir: Path) -> bool:
        """Check if the working tree or index has uncommitted changes.

        Args:
            wt_dir: Git working directory to check.

        Returns:
            True if there are staged, unstaged, or untracked changes.
        """
        status = _git("status", "--porcelain", cwd=wt_dir)
        return bool(status.stdout.strip())

    @staticmethod
    def staged_diff(wt_dir: Path) -> str:
        """Return the staged diff text for the worktree.

        Args:
            wt_dir: Worktree directory (must have staged changes).

        Returns:
            The diff text, or empty string if no staged changes.
        """
        result = _git("diff", "--cached", cwd=wt_dir)
        return result.stdout.strip()

    @staticmethod
    def checkout(repo: Path, branch: str) -> tuple[bool, str]:
        """Checkout a branch in the main worktree.

        Args:
            repo: Git repo root path.
            branch: Branch name to checkout.

        Returns:
            ``(True, "")`` on success, ``(False, stderr)`` on failure.
            The stderr string describes why the checkout failed (e.g.
            dirty working tree, missing branch).
        """
        result = _git("checkout", branch, cwd=repo)
        if result.returncode == 0:
            return (True, "")
        return (False, result.stderr.strip())

    @staticmethod
    def stash_if_dirty(repo: Path) -> bool:
        """Stash uncommitted changes if the working tree or index is dirty.

        Uses ``git stash push --include-untracked`` so both staged and
        unstaged changes (including new files) are saved.

        Args:
            repo: Git repo root path.

        Returns:
            True if a stash entry was created, False if the tree was clean.
        """
        status = _git("status", "--porcelain", cwd=repo)
        if not status.stdout.strip():
            return False
        result = _git(
            "stash",
            "push",
            "--include-untracked",
            "-m",
            "kiss: auto-stash before merge",
            cwd=repo,
        )
        return result.returncode == 0

    @staticmethod
    def stash_pop(repo: Path) -> bool:
        """Pop the latest stash entry, preserving the staging state.

        Tries ``git stash pop --index`` first so that files that were
        staged before the stash stay staged after the pop.  If
        ``--index`` fails (e.g. the merge changed a file that was in
        the index), falls back to plain ``git stash pop`` which
        restores all changes as unstaged.

        Args:
            repo: Git repo root path.

        Returns:
            True if the pop succeeded, False on conflict or error.
        """
        result = _git("stash", "pop", "--index", cwd=repo)
        if result.returncode == 0:
            return True
        # --index can fail when staged changes conflict with the
        # current tree.  Fall back to plain pop (loses staging state
        # but preserves the content).
        result = _git("stash", "pop", cwd=repo)
        return result.returncode == 0

    @staticmethod
    def squash_merge_branch(repo: Path, branch: str) -> MergeResult:
        """Squash-merge a branch and commit the result.

        Uses ``git merge --squash`` to apply all changes from *branch*,
        then commits them.  The commit message is taken from git's
        auto-generated ``SQUASH_MSG``.

        On conflict, resets to a clean state with ``git reset --hard``.

        Args:
            repo: Git repo root path.
            branch: Branch to squash-merge.

        Returns:
            :attr:`MergeResult.SUCCESS` or :attr:`MergeResult.CONFLICT`.
        """
        result = _git("merge", "--squash", branch, cwd=repo)
        if result.returncode != 0:
            logger.warning(
                "squash merge failed: %s",
                result.stderr.strip(),
            )
            _git("reset", "--hard", "HEAD", cwd=repo)
            return MergeResult.CONFLICT
        diff = _git("diff", "--cached", "--quiet", cwd=repo)
        if diff.returncode != 0:
            commit_result = _git("commit", "--no-edit", cwd=repo)
            if commit_result.returncode != 0:
                logger.warning(
                    "squash merge commit failed: %s",
                    commit_result.stderr.strip(),
                )
                _git("reset", "--hard", "HEAD", cwd=repo)
                return MergeResult.MERGE_FAILED
        return MergeResult.SUCCESS

    @staticmethod
    def unstaged_files(repo: Path) -> list[str]:
        """List files with unstaged changes in the working tree.

        Args:
            repo: Git repo root path.

        Returns:
            List of files with uncommitted, unstaged modifications,
            or an empty list if the command fails.
        """
        result = _git("diff", "--name-only", cwd=repo)
        if result.returncode != 0:
            return []
        return [f for f in result.stdout.strip().splitlines() if f]

    @staticmethod
    def staged_files(repo: Path) -> list[str]:
        """List files with staged (cached) changes in the index.

        Args:
            repo: Git repo root path.

        Returns:
            List of files staged for commit, or an empty list if the
            command fails.
        """
        result = _git("diff", "--cached", "--name-only", cwd=repo)
        if result.returncode != 0:
            return []
        return [f for f in result.stdout.strip().splitlines() if f]

    @staticmethod
    def delete_branch(repo: Path, branch: str) -> bool:
        """Delete a branch and its git config section (best-effort).

        Tries ``-d`` first (safe delete), falls back to ``-D`` (force).
        Also removes the ``branch.<name>.*`` config section.

        Args:
            repo: Git repo root path.
            branch: Branch name to delete.

        Returns:
            True if the branch was deleted (or never existed), False
            if git refused both ``-d`` and ``-D`` — typically because
            the branch is the current HEAD of a worktree and cannot
            be deleted without first switching away.
        """
        safe = _git("branch", "-d", branch, cwd=repo)
        if safe.returncode == 0:
            _git("config", "--remove-section", f"branch.{branch}", cwd=repo)
            return True
        force = _git("branch", "-D", branch, cwd=repo)
        if force.returncode == 0:
            _git("config", "--remove-section", f"branch.{branch}", cwd=repo)
            return True
        if GitWorktreeOps.branch_exists(repo, branch):
            logger.warning(
                "Failed to delete branch '%s': %s",
                branch,
                force.stderr.strip(),
            )
            return False
        _git("config", "--remove-section", f"branch.{branch}", cwd=repo)
        return True

    @staticmethod
    def branch_exists(repo: Path, branch: str) -> bool:
        """Check if a branch exists.

        Args:
            repo: Git repo root path.
            branch: Branch name to check.

        Returns:
            True if the branch exists.
        """
        result = _git("rev-parse", "--verify", f"refs/heads/{branch}", cwd=repo)
        return result.returncode == 0

    @staticmethod
    def ensure_excluded(repo: Path) -> None:
        """Add ``.kiss-worktrees/`` to local git exclude (not .gitignore).

        Uses ``<git_common_dir>/info/exclude`` so the agent never modifies
        any tracked file in the user's repo.

        Args:
            repo: Git repo root path.
        """
        result = _git("rev-parse", "--git-common-dir", cwd=repo)
        git_common = Path(result.stdout.strip())
        if not git_common.is_absolute():  # pragma: no branch
            git_common = (repo / git_common).resolve()
        exclude_file = git_common / "info" / "exclude"
        exclude_file.parent.mkdir(parents=True, exist_ok=True)
        entry = ".kiss-worktrees/"
        if exclude_file.exists():
            content = exclude_file.read_text()
            if entry in content.splitlines():
                return
        with open(exclude_file, "a") as f:
            f.write(f"\n{entry}\n")

    @staticmethod
    def find_pending_branch(repo: Path, prefix: str) -> str | None:
        """Find the latest ``kiss/wt-*`` branch matching a prefix.

        Args:
            repo: Git repo root path.
            prefix: Branch name prefix (e.g. ``kiss/wt-<chat_id[:12]>-``).

        Returns:
            The lexicographically last matching branch, or ``None``.
        """
        result = _git(
            "for-each-ref",
            "--format=%(refname:short)",
            f"refs/heads/{prefix}*",
            cwd=repo,
        )
        branches = result.stdout.strip().splitlines()
        if not branches:
            return None
        return sorted(branches)[-1]

    @staticmethod
    def load_original_branch(repo: Path, branch: str) -> str | None:
        """Load the original branch from git config.

        Args:
            repo: Git repo root path.
            branch: The worktree branch name.

        Returns:
            The original branch name, or ``None`` if not stored.
        """
        result = _git("config", f"branch.{branch}.kiss-original", cwd=repo)
        return result.stdout.strip() or None

    @staticmethod
    def save_original_branch(repo: Path, branch: str, original: str) -> bool:
        """Store the original branch in git config.

        Args:
            repo: Git repo root path.
            branch: The worktree branch name.
            original: The original branch to store.

        Returns:
            True if config was saved successfully, False otherwise.
        """
        result = _git("config", f"branch.{branch}.kiss-original", original, cwd=repo)
        if result.returncode != 0:  # pragma: no cover — git config failure
            logger.warning(
                "Failed to store original branch in git config: %s",
                result.stderr.strip(),
            )
            return False
        return True

    @staticmethod
    def save_baseline_commit(
        repo: Path,
        branch: str,
        sha: str,
    ) -> bool:
        """Store the baseline commit SHA in git config.

        The baseline commit captures the user's dirty state (staged,
        unstaged, untracked files) at worktree creation time.  Downstream
        operations diff against this SHA to isolate agent-only changes.

        Args:
            repo: Git repo root path.
            branch: The worktree branch name.
            sha: The baseline commit SHA to store.

        Returns:
            True if config was saved successfully, False otherwise.
        """
        result = _git(
            "config",
            f"branch.{branch}.kiss-baseline",
            sha,
            cwd=repo,
        )
        if result.returncode != 0:  # pragma: no cover — git config failure
            logger.warning(
                "Failed to store baseline commit in git config: %s",
                result.stderr.strip(),
            )
            return False
        return True

    @staticmethod
    def load_baseline_commit(repo: Path, branch: str) -> str | None:
        """Load the baseline commit SHA from git config.

        Args:
            repo: Git repo root path.
            branch: The worktree branch name.

        Returns:
            The baseline commit SHA, or ``None`` if not stored (clean
            worktree or legacy worktree without baseline support).
        """
        result = _git(
            "config",
            f"branch.{branch}.kiss-baseline",
            cwd=repo,
        )
        return result.stdout.strip() or None

    @staticmethod
    def copy_dirty_state(repo: Path, wt_dir: Path) -> bool:
        """Copy uncommitted/staged/untracked files from main worktree.

        Reads ``git status --porcelain`` in *repo* and mirrors every
        dirty file into *wt_dir*.  Files that exist in the main
        worktree are copied; files that were deleted are removed from
        *wt_dir*.  The caller is expected to stage and commit the
        result as a baseline commit.

        Args:
            repo: Git repo root (main worktree).
            wt_dir: Target worktree directory.

        Returns:
            True if any dirty state was copied, False if the main
            worktree was clean.
        """
        status = _git("status", "--porcelain", "-uall", cwd=repo)
        if not status.stdout.strip():
            return False

        copied = False
        for line in status.stdout.splitlines():
            if len(line) < 4:
                continue
            fname = _unquote_git_path(line[3:])
            old_name: str | None = None
            if " -> " in fname:
                old_name, fname = fname.split(" -> ", 1)
                old_name = _unquote_git_path(old_name)
                fname = _unquote_git_path(fname)

            src = repo / fname
            dst = wt_dir / fname

            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
                copied = True
                if old_name is not None:
                    old_dst = wt_dir / old_name
                    if old_dst.exists():
                        old_dst.unlink()
            elif dst.exists():
                dst.unlink()
                copied = True
            elif src.is_dir():
                continue

        return copied

    @staticmethod
    def head_sha(wt_dir: Path) -> str | None:
        """Return the SHA of HEAD in the given directory.

        Args:
            wt_dir: Git working directory (repo root or worktree).

        Returns:
            The full SHA string, or ``None`` on failure.
        """
        result = _git("rev-parse", "HEAD", cwd=wt_dir)
        sha = result.stdout.strip()
        return sha if result.returncode == 0 and sha else None

    @staticmethod
    def squash_merge_from_baseline(
        repo: Path,
        branch: str,
        baseline: str,
    ) -> MergeResult:
        """Squash-merge only the agent's changes (after baseline) into HEAD.

        Uses ``git cherry-pick --no-commit`` to replay each commit
        after *baseline* onto the current HEAD.  Cherry-pick performs
        a proper three-way merge per commit (using the commit's parent
        as the merge base), so it handles cases where the user's dirty
        state (captured in the baseline) diverges from the committed
        HEAD content.

        Falls back to :meth:`squash_merge_branch` when *baseline* is
        ``None`` (legacy worktrees).

        Args:
            repo: Git repo root path.
            branch: The worktree branch to merge from.
            baseline: SHA of the baseline commit to diff against.

        Returns:
            :attr:`MergeResult.SUCCESS` or :attr:`MergeResult.CONFLICT`.
        """
        log_result = _git(
            "rev-list",
            "--count",
            f"{baseline}..{branch}",
            cwd=repo,
        )
        if log_result.returncode != 0:
            logger.warning(
                "rev-list failed for baseline %s..%s: %s",
                baseline,
                branch,
                log_result.stderr.strip(),
            )
            return MergeResult.CONFLICT
        count = log_result.stdout.strip()
        if count == "0":
            return MergeResult.SUCCESS

        result = _git(
            "cherry-pick",
            "--no-commit",
            f"{baseline}..{branch}",
            cwd=repo,
        )
        if result.returncode != 0:
            logger.warning(
                "squash merge from baseline failed: %s",
                result.stderr.strip(),
            )
            _git("cherry-pick", "--abort", cwd=repo)
            return MergeResult.CONFLICT

        diff_check = _git("diff", "--cached", "--quiet", cwd=repo)
        if diff_check.returncode != 0:
            log_msgs = _git(
                "log",
                "--oneline",
                f"{baseline}..{branch}",
                cwd=repo,
            )
            body = log_msgs.stdout.strip()
            msg = f"kiss: merged from {branch}"
            if body:
                msg += f"\n\n{body}"
            commit_result = _git("commit", "-m", msg, cwd=repo)
            if commit_result.returncode != 0:
                logger.warning(
                    "squash merge commit failed: %s",
                    commit_result.stderr.strip(),
                )
                _git("reset", "--hard", "HEAD", cwd=repo)
                return MergeResult.MERGE_FAILED
        return MergeResult.SUCCESS

    @staticmethod
    def cleanup_partial(repo: Path, branch: str, wt_dir: Path) -> None:
        """Remove a partially-created worktree and branch (best-effort).

        Args:
            repo: Git repo root path.
            branch: The branch name to delete.
            wt_dir: The worktree directory to remove.
        """
        if wt_dir.exists():
            _git("worktree", "remove", str(wt_dir), "--force", cwd=repo)
        _git("worktree", "prune", cwd=repo)
        GitWorktreeOps.delete_branch(repo, branch)

    @staticmethod
    def cleanup_orphans(repo: Path) -> str:
        """Scan for orphaned ``kiss/wt-*`` branches and worktrees.

        Cleans up three distinct forms of stale state:

        1.  ``kiss/wt-*`` branches with no active git worktree and no
            ``kiss-original`` config (true orphan branches).
        2.  Registered worktree bookkeeping entries whose directory
            is gone (``git worktree prune``).
        3.  Directories under ``.kiss-worktrees/`` that are not
            registered as git worktrees (orphan directories — e.g.
            leftover files from a crashed agent session or a manually
            unlinked worktree).  Pending-merge branches (those with
            ``kiss-original`` set) are never removed — BUG-58.

        Args:
            repo: Root of the git repository to scan.

        Returns:
            Summary of findings and any cleanup actions taken.
        """
        result = _git(
            "for-each-ref",
            "--format=%(refname:short)",
            "refs/heads/kiss/wt-*",
            cwd=repo,
        )
        branches = result.stdout.strip().splitlines() if result.stdout.strip() else []

        wt_result = _git("worktree", "list", "--porcelain", cwd=repo)
        worktree_branches: set[str] = set()
        registered_dirs: set[Path] = set()
        for line in wt_result.stdout.splitlines():
            if line.startswith("branch refs/heads/kiss/wt-"):
                worktree_branches.add(line.split("refs/heads/")[1])
            elif line.startswith("worktree "):
                registered_dirs.add(Path(line.split(" ", 1)[1]).resolve())

        orphan_branches: list[str] = []
        pending_branches: list[str] = []
        for b in branches:
            if b in worktree_branches:
                continue
            original = GitWorktreeOps.load_original_branch(repo, b)
            if original is not None:
                pending_branches.append(b)
            else:
                orphan_branches.append(b)

        lines = [
            f"Found {len(branches)} kiss/wt-* branch(es), "
            f"{len(worktree_branches)} active worktree(s)."
        ]

        if pending_branches:
            lines.append(f"Pending-merge branches (kept): {pending_branches}")

        if orphan_branches:
            lines.append(f"Orphaned branches (no worktree): {orphan_branches}")
            for b in orphan_branches:
                _git("branch", "-D", b, cwd=repo)
                _git("config", "--remove-section", f"branch.{b}", cwd=repo)
                lines.append(f"  Deleted: {b}")

        _git("worktree", "prune", cwd=repo)
        lines.append("Ran git worktree prune.")

        wt_root = repo / ".kiss-worktrees"
        orphan_dirs: list[str] = []
        if wt_root.is_dir():
            for child in sorted(wt_root.iterdir()):
                if not child.is_dir():
                    continue
                if child.resolve() in registered_dirs:
                    continue
                shutil.rmtree(child, ignore_errors=True)
                orphan_dirs.append(child.name)

        if orphan_dirs:
            lines.append(f"Orphan directories removed: {orphan_dirs}")

        if not orphan_branches and not orphan_dirs:
            lines.append("No orphans found.")

        return "\n".join(lines)
