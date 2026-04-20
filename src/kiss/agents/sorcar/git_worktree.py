"""Git worktree operations and state.

Provides :class:`GitWorktree` (frozen dataclass for worktree state),
:class:`MergeResult` (outcome enum), and :class:`GitWorktreeOps`
(stateless helper with all git worktree operations).
"""

from __future__ import annotations

import enum
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


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
    """

    repo_root: Path
    branch: str
    original_branch: str | None
    wt_dir: Path


class MergeResult(enum.Enum):
    """Outcome of a merge operation."""

    SUCCESS = "success"
    CONFLICT = "conflict"
    CHECKOUT_FAILED = "checkout_failed"
    MERGE_FAILED = "merge_failed"


@dataclass(frozen=True)
class ManualMergeResult:
    """Outcome of a manual (--no-commit) merge operation.

    Attributes:
        status: The merge outcome.
        has_conflicts: True if CONFLICT was detected in merge output.
    """

    status: MergeResult
    has_conflicts: bool


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
            logger.warning(
                "Failed to create worktree: %s", result.stderr.strip()
            )
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
            result = _git(
                "worktree", "remove", str(wt_dir), "--force", cwd=repo
            )
            if result.returncode != 0:  # pragma: no cover — lock/perm
                logger.warning(
                    "worktree remove failed: %s", result.stderr.strip()
                )

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
                "git commit failed: %s", result.stderr.strip(),
            )
            return False
        return True

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
    def checkout(repo: Path, branch: str) -> bool:
        """Checkout a branch in the main worktree.

        Args:
            repo: Git repo root path.
            branch: Branch name to checkout.

        Returns:
            True if checkout succeeded, False otherwise.
        """
        result = _git("checkout", branch, cwd=repo)
        return result.returncode == 0

    @staticmethod
    def checkout_error(repo: Path, branch: str) -> str:
        """Return the stderr from a failed checkout attempt.

        Args:
            repo: Git repo root path.
            branch: Branch name that failed to checkout.

        Returns:
            The stderr text from the failed checkout.
        """
        result = _git("checkout", branch, cwd=repo)
        return result.stderr.strip()

    @staticmethod
    def merge_branch(repo: Path, branch: str) -> MergeResult:
        """Merge a branch into the current HEAD with ``--no-edit``.

        On conflict, the merge is aborted to leave a clean worktree.

        Args:
            repo: Git repo root path.
            branch: Branch to merge.

        Returns:
            :attr:`MergeResult.SUCCESS` or :attr:`MergeResult.CONFLICT`.
        """
        result = _git("merge", branch, "--no-edit", cwd=repo)
        if result.returncode == 0:
            return MergeResult.SUCCESS
        _git("merge", "--abort", cwd=repo)
        return MergeResult.CONFLICT

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
        # Check working tree + index
        status = _git("status", "--porcelain", cwd=repo)
        if not status.stdout.strip():
            return False
        result = _git(
            "stash", "push", "--include-untracked",
            "-m", "kiss: auto-stash before merge",
            cwd=repo,
        )
        return result.returncode == 0

    @staticmethod
    def stash_pop(repo: Path) -> bool:
        """Pop the latest stash entry.

        Args:
            repo: Git repo root path.

        Returns:
            True if the pop succeeded, False on conflict or error.
        """
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
                "squash merge failed: %s", result.stderr.strip(),
            )
            _git("reset", "--hard", "HEAD", cwd=repo)
            return MergeResult.CONFLICT
        diff = _git("diff", "--cached", "--quiet", cwd=repo)
        if diff.returncode != 0:
            _git("commit", "--no-edit", cwd=repo)
        return MergeResult.SUCCESS

    @staticmethod
    def apply_branch_to_working_tree(
        repo: Path, branch: str,
    ) -> MergeResult:
        """Apply a branch's changes to the working tree as unstaged edits.

        Captures the set of files that will change (``git diff
        --name-only HEAD..branch``), runs ``git merge --squash`` to
        stage them, then unstages *only* those files with
        ``git reset HEAD -- <files>`` so any pre-existing staged
        changes the user had are preserved.

        On merge failure (e.g. dirty index that overlaps the branch),
        restores the pre-merge state of the affected files with
        ``git reset HEAD -- <files>`` + ``git checkout -- <files>``
        so unrelated user changes are untouched.

        Args:
            repo: Git repo root path.
            branch: Branch whose changes to apply.

        Returns:
            :attr:`MergeResult.SUCCESS` or :attr:`MergeResult.CONFLICT`.
        """
        diff = _git("diff", "--name-only", f"HEAD..{branch}", cwd=repo)
        merge_files = [
            f for f in diff.stdout.strip().splitlines() if f
        ]
        result = _git("merge", "--squash", branch, cwd=repo)
        if result.returncode != 0:
            # git refused (e.g. dirty index overlaps the merge) —
            # nothing was written, leave the user's state alone.
            return MergeResult.CONFLICT
        if merge_files:  # pragma: no branch — non-empty for real merges
            _git("reset", "HEAD", "--", *merge_files, cwd=repo)
        return MergeResult.SUCCESS

    @staticmethod
    def would_merge_conflict(
        repo: Path, base: str, branch: str,
    ) -> bool:
        """Dry-run check for tree-level merge conflicts.

        Runs ``git merge-tree --write-tree`` which performs a merge
        entirely in memory without touching the working tree or index.
        A non-zero exit code indicates the three-way merge would
        produce textual conflicts.

        Args:
            repo: Git repo root path.
            base: The branch that would receive the merge.
            branch: The branch that would be merged in.

        Returns:
            True if the merge would have tree-level conflicts, False
            if it would apply cleanly.
        """
        result = _git(
            "merge-tree", "--write-tree", base, branch, cwd=repo,
        )
        return result.returncode != 0

    @staticmethod
    def branch_diff_files(
        repo: Path, base: str, branch: str,
    ) -> list[str]:
        """List files that differ between two branches/revisions.

        Args:
            repo: Git repo root path.
            base: First branch or revision.
            branch: Second branch or revision.

        Returns:
            List of file paths that differ between *base* and *branch*,
            or an empty list if the command fails.
        """
        result = _git("diff", "--name-only", base, branch, cwd=repo)
        if result.returncode != 0:
            return []
        return [f for f in result.stdout.strip().splitlines() if f]

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
    def manual_merge_branch(repo: Path, branch: str) -> ManualMergeResult:
        """Merge with ``--no-commit --no-ff`` for interactive review.

        On success (no conflicts), unstages changes via ``git reset HEAD``
        so the user can selectively stage hunks.

        Args:
            repo: Git repo root path.
            branch: Branch to merge.

        Returns:
            A :class:`ManualMergeResult` with status and conflict info.
        """
        result = _git(
            "merge", "--no-commit", "--no-ff", branch, cwd=repo
        )
        has_conflicts = "CONFLICT" in (result.stdout + result.stderr)

        if result.returncode != 0 and not has_conflicts:
            return ManualMergeResult(
                status=MergeResult.MERGE_FAILED, has_conflicts=False
            )

        if not has_conflicts:
            _git("reset", "HEAD", cwd=repo)

        status = MergeResult.CONFLICT if has_conflicts else MergeResult.SUCCESS
        return ManualMergeResult(status=status, has_conflicts=has_conflicts)

    @staticmethod
    def delete_branch(repo: Path, branch: str) -> None:
        """Delete a branch and its git config section (best-effort).

        Tries ``-d`` first (safe delete), falls back to ``-D`` (force).
        Also removes the ``branch.<name>.*`` config section.

        Args:
            repo: Git repo root path.
            branch: Branch name to delete.
        """
        result = _git("branch", "-d", branch, cwd=repo)
        if result.returncode != 0:
            _git("branch", "-D", branch, cwd=repo)
        _git("config", "--remove-section", f"branch.{branch}", cwd=repo)

    @staticmethod
    def branch_exists(repo: Path, branch: str) -> bool:
        """Check if a branch exists.

        Args:
            repo: Git repo root path.
            branch: Branch name to check.

        Returns:
            True if the branch exists.
        """
        result = _git(
            "rev-parse", "--verify", f"refs/heads/{branch}", cwd=repo
        )
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
        result = _git(
            "config", f"branch.{branch}.kiss-original", cwd=repo
        )
        return result.stdout.strip() or None

    @staticmethod
    def save_original_branch(
        repo: Path, branch: str, original: str
    ) -> bool:
        """Store the original branch in git config.

        Args:
            repo: Git repo root path.
            branch: The worktree branch name.
            original: The original branch to store.

        Returns:
            True if config was saved successfully, False otherwise.
        """
        result = _git(
            "config", f"branch.{branch}.kiss-original", original, cwd=repo
        )
        if result.returncode != 0:  # pragma: no cover — git config failure
            logger.warning(
                "Failed to store original branch in git config: %s",
                result.stderr.strip(),
            )
            return False
        return True

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
        branches = (
            result.stdout.strip().splitlines()
            if result.stdout.strip()
            else []
        )

        wt_result = _git("worktree", "list", "--porcelain", cwd=repo)
        worktree_branches: set[str] = set()
        for line in wt_result.stdout.splitlines():
            if line.startswith("branch refs/heads/kiss/wt-"):
                worktree_branches.add(line.split("refs/heads/")[1])

        orphan_branches = [
            b for b in branches if b not in worktree_branches
        ]
        lines = [
            f"Found {len(branches)} kiss/wt-* branch(es), "
            f"{len(worktree_branches)} active worktree(s)."
        ]

        if orphan_branches:
            lines.append(
                f"Orphaned branches (no worktree): {orphan_branches}"
            )
            for b in orphan_branches:
                _git("branch", "-D", b, cwd=repo)
                _git(
                    "config", "--remove-section", f"branch.{b}", cwd=repo
                )
                lines.append(f"  Deleted: {b}")

        _git("worktree", "prune", cwd=repo)
        lines.append("Ran git worktree prune.")

        if not orphan_branches:
            lines.append("No orphans found.")

        return "\n".join(lines)
