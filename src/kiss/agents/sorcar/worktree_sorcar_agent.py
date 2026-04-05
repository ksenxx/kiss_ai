"""Worktree-based agent that runs each task on an isolated git branch.

Creates a ``git worktree`` for every task so the user's main working tree
is never modified.  After the task the user chooses **merge**, **manual
merge**, or **discard**.  The agent refuses further tasks in the same chat
session until the branch is resolved.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.cli_helpers import (
    _apply_chat_args,
    _build_chat_arg_parser,
    _build_run_kwargs,
    _print_recent_chats,
    _print_run_stats,
)
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent

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


class WorktreeSorcarAgent(StatefulSorcarAgent):
    """SorcarAgent that isolates every task in a git worktree.

    State is stored entirely in git (branches and config) — no sidecar
    files.  On process restart, ``_restore_from_git()`` reconstructs all
    instance attributes from git queries.

    Attributes:
        _repo_root: Git repo root path, or ``None`` if not in a repo.
        _wt_branch: Branch name of the current/pending worktree task,
            or ``None`` when idle.
        _original_branch: The branch the user was on when the task started.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._repo_root: Path | None = None
        self._wt_branch: str | None = None
        self._original_branch: str | None = None

    @property
    def _wt_pending(self) -> bool:
        """Whether a worktree task is pending merge/discard."""
        return self._wt_branch is not None

    @property
    def _wt_dir(self) -> Path | None:
        """Worktree directory path, derived from repo root and branch name."""
        if self._repo_root is None or self._wt_branch is None:
            return None
        slug = self._wt_branch.replace("/", "_")
        return self._repo_root / ".kiss-worktrees" / slug

    def _restore_from_git(self) -> None:
        """Restore pending-branch state from git (no sidecar files).

        Queries git for any ``kiss/wt-<chat_id[:12]>-*`` branch.  If
        found, restores ``_wt_branch`` and ``_original_branch`` from
        ``git config``.  If the config entry is missing (crash between
        worktree creation and config write), falls back to the current
        HEAD branch of the main worktree.
        """
        if not self._repo_root:
            return
        if self._wt_branch:
            return
        prefix = f"kiss/wt-{self._chat_id[:12]}-"
        result = _git(
            "for-each-ref", "--format=%(refname:short)",
            f"refs/heads/{prefix}*",
            cwd=self._repo_root,
        )
        branches = result.stdout.strip().splitlines()
        if not branches:
            return
        self._wt_branch = sorted(branches)[-1]
        orig = _git(
            "config", f"branch.{self._wt_branch}.kiss-original",
            cwd=self._repo_root,
        )
        self._original_branch = orig.stdout.strip() or None
        if self._original_branch is None:
            head = _git(
                "rev-parse", "--abbrev-ref", "HEAD",
                cwd=self._repo_root,
            )
            fallback = head.stdout.strip()
            if fallback and fallback != "HEAD":
                self._original_branch = fallback

    def _ensure_worktree_excluded(self) -> None:
        """Add ``.kiss-worktrees/`` to local git exclude (not .gitignore).

        Uses ``<git_common_dir>/info/exclude`` so the agent never modifies
        any tracked file in the user's repo.
        """
        if self._repo_root is None:
            return
        result = _git(
            "rev-parse", "--git-common-dir",
            cwd=self._repo_root,
        )
        git_common = Path(result.stdout.strip())
        if not git_common.is_absolute():  # pragma: no branch — always relative for main worktree
            git_common = (self._repo_root / git_common).resolve()
        exclude_file = git_common / "info" / "exclude"
        exclude_file.parent.mkdir(parents=True, exist_ok=True)
        entry = ".kiss-worktrees/"
        if exclude_file.exists():
            content = exclude_file.read_text()
            if entry in content.splitlines():
                return
        with open(exclude_file, "a") as f:
            f.write(f"\n{entry}\n")

    def _auto_commit_worktree(self) -> bool:
        """Commit any uncommitted changes in the worktree.

        Returns:
            True if a commit was created, False if nothing to commit.
        """
        wt_dir = self._wt_dir
        if wt_dir is None or not wt_dir.exists():
            return False
        _git("add", "-A", cwd=wt_dir)
        diff = _git("diff", "--cached", "--quiet", cwd=wt_dir)
        if diff.returncode == 0:
            return False
        _git("commit", "-m", "kiss: auto-commit agent work", cwd=wt_dir)
        return True

    def _cleanup_partial_worktree(self, branch: str, wt_dir: Path) -> None:
        """Remove a partially-created worktree and branch (best-effort).

        Args:
            branch: The branch name to delete.
            wt_dir: The worktree directory to remove.
        """
        if wt_dir.exists():
            _git("worktree", "remove", str(wt_dir), "--force",
                 cwd=self._repo_root)
        _git("worktree", "prune", cwd=self._repo_root)
        _git("branch", "-D", branch, cwd=self._repo_root)

    def run(  # type: ignore[override]
        self,
        prompt_template: str = "",
        **kwargs: Any,
    ) -> str:
        """Run a task on an isolated git worktree branch.

        Creates a new worktree and branch, redirects ``work_dir`` into
        the worktree, and delegates to ``StatefulSorcarAgent.run()``.
        If a branch from this chat session is already pending, returns
        an error asking the user to merge or discard first.

        Falls back to direct execution (no worktree) when:
        - ``work_dir`` is not inside a git repo
        - The repo has no commits
        - HEAD is detached (no merge target)
        - Any git command fails during setup

        Args:
            prompt_template: The task prompt.
            **kwargs: All other arguments forwarded to
                ``StatefulSorcarAgent.run()``.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        # Step 1: Determine discovery directory
        work_dir_str = kwargs.get("work_dir")
        discovery_dir = Path(work_dir_str) if work_dir_str else Path.cwd()

        # Step 2: Discover repo root
        toplevel = _git("rev-parse", "--show-toplevel", cwd=discovery_dir)
        if toplevel.returncode != 0:
            logger.warning("Not a git repo, running task directly: %s",
                           toplevel.stderr.strip())
            return super().run(prompt_template=prompt_template, **kwargs)
        self._repo_root = Path(toplevel.stdout.strip())

        # Step 3: Restore from git
        self._restore_from_git()

        # Step 4: Check for pending branch
        if self._wt_branch is not None:
            blocked: str = yaml.dump({
                "success": False,
                "summary": (
                    f"Cannot start a new task in this chat session: branch "
                    f"'{self._wt_branch}' is pending merge/discard.\n\n"
                    + self.merge_instructions()
                ),
            })
            return blocked

        # Step 5: Detect original branch
        head_result = _git("rev-parse", "--abbrev-ref", "HEAD",
                           cwd=self._repo_root)
        original_branch = head_result.stdout.strip()
        if not original_branch or original_branch == "HEAD":
            logger.warning("Detached HEAD, running task directly")
            return super().run(prompt_template=prompt_template, **kwargs)

        # Step 6: Compute subdirectory offset
        if work_dir_str:
            try:
                offset = Path(work_dir_str).resolve().relative_to(
                    self._repo_root.resolve())
            except ValueError:  # pragma: no cover — defensive; discovery_dir = work_dir
                logger.warning("work_dir not inside repo, running directly")
                return super().run(prompt_template=prompt_template, **kwargs)
        else:
            offset = Path(".")

        # Step 7: Ensure .kiss-worktrees/ is excluded
        try:
            self._ensure_worktree_excluded()
        except Exception:  # pragma: no cover — filesystem permission error
            logger.warning("Failed to update git exclude", exc_info=True)

        # Step 8: Generate branch name and create worktree
        branch = f"kiss/wt-{self._chat_id[:12]}-{int(time.time())}"
        # Handle branch name collision
        base_branch = branch
        suffix = 1
        while (  # pragma: no branch — timestamp collision extremely unlikely
            _git("rev-parse", "--verify", f"refs/heads/{branch}",
                 cwd=self._repo_root).returncode == 0
        ):
            branch = f"{base_branch}-{suffix}"
            suffix += 1

        slug = branch.replace("/", "_")
        wt_dir = self._repo_root / ".kiss-worktrees" / slug

        wt_result = _git("worktree", "add", "-b", branch, str(wt_dir),
                         cwd=self._repo_root)
        if wt_result.returncode != 0:  # pragma: no cover — git worktree add failure
            logger.warning("Failed to create worktree, running directly: %s",
                           wt_result.stderr.strip())
            self._cleanup_partial_worktree(branch, wt_dir)
            return super().run(prompt_template=prompt_template, **kwargs)

        # Step 9: Store original branch in git config
        config_result = _git("config",
                             f"branch.{branch}.kiss-original", original_branch,
                             cwd=self._repo_root)
        if config_result.returncode != 0:  # pragma: no cover — git config failure
            logger.warning("Failed to store original branch in git config: %s",
                           config_result.stderr.strip())
            self._cleanup_partial_worktree(branch, wt_dir)
            return super().run(prompt_template=prompt_template, **kwargs)

        # Set state
        self._wt_branch = branch
        self._original_branch = original_branch

        # Step 10: Create offset directory and redirect work_dir
        wt_work_dir = wt_dir / offset
        wt_work_dir.mkdir(parents=True, exist_ok=True)
        kwargs["work_dir"] = str(wt_work_dir)

        # Step 11: Run task
        try:
            task_result = super().run(
                prompt_template=prompt_template, **kwargs)
        except Exception as exc:
            task_result = yaml.dump({
                "success": False,
                "summary": f"Task failed with error: {exc}",
            })

        # Step 12: Append merge/discard instructions
        return task_result + "\n\n---\n" + self.merge_instructions()

    def merge(self) -> str:
        """Merge the task branch into the original branch.

        Every step is idempotent — safe to re-run after a crash.
        Auto-commits any uncommitted changes in the worktree before
        merging.

        Returns:
            Success message, or error message if merge fails.

        Raises:
            RuntimeError: If no worktree task is pending.
        """
        if self._wt_branch is None:
            raise RuntimeError("No pending worktree task to merge")

        if self._original_branch is None:
            return (
                "Cannot merge: original branch is unknown (likely due to a "
                "crash during setup).  Please specify the target branch "
                "manually:\n"
                f"    git checkout <branch> && git merge {self._wt_branch}"
            )

        wt_dir = self._wt_dir

        # Step 4: Remove worktree (auto-commit first)
        if wt_dir is not None and wt_dir.exists():
            self._auto_commit_worktree()
            remove_result = _git("worktree", "remove", str(wt_dir),
                                 cwd=self._repo_root)
            if remove_result.returncode != 0:  # pragma: no cover — worktree lock/perm
                logger.warning("worktree remove failed: %s",
                               remove_result.stderr.strip())

        # Step 5: Prune
        _git("worktree", "prune", cwd=self._repo_root)

        # Step 6: Checkout original branch
        checkout = _git("checkout", self._original_branch,
                        cwd=self._repo_root)
        if checkout.returncode != 0:  # pragma: no cover — dirty main worktree
            return (
                f"Cannot checkout '{self._original_branch}': "
                f"{checkout.stderr.strip()}\n"
                "Fix the issue and retry merge(), or call discard()."
            )

        # Step 7: Merge
        merge_result = _git("merge", self._wt_branch, "--no-edit",
                            cwd=self._repo_root)

        if merge_result.returncode == 0:
            # Step 8: Success — delete branch, reset state
            _git("branch", "-d", self._wt_branch, cwd=self._repo_root)
            branch_name = self._wt_branch
            self._wt_branch = None
            self._original_branch = None
            return f"Successfully merged branch '{branch_name}'."

        # Step 9: Merge conflict
        _git("merge", "--abort", cwd=self._repo_root)
        wt_branch = self._wt_branch
        orig_branch = self._original_branch
        repo = self._repo_root
        return (
            "Merge conflict detected.  Resolve manually:\n"
            f"    cd {repo}\n"
            f"    git checkout {orig_branch}\n"
            f"    git merge {wt_branch}\n"
            "    # resolve conflicts in your editor\n"
            "    git add .\n"
            "    git commit\n"
            f"    git branch -d {wt_branch}\n"
            "\nOr discard the branch:\n"
            "    agent.discard()"
        )

    def discard(self) -> str:
        """Throw away the task branch and worktree.

        Every step is idempotent — safe to call multiple times.

        Returns:
            Confirmation message.

        Raises:
            RuntimeError: If no worktree task is pending.
        """
        if self._wt_branch is None:
            raise RuntimeError("No pending worktree task to discard")

        wt_dir = self._wt_dir
        branch_name = self._wt_branch

        # Step 3: Remove worktree
        if wt_dir is not None and wt_dir.exists():
            _git("worktree", "remove", str(wt_dir), "--force",
                 cwd=self._repo_root)

        # Step 4: Prune
        _git("worktree", "prune", cwd=self._repo_root)

        # Step 5: Delete branch
        _git("branch", "-D", self._wt_branch, cwd=self._repo_root)

        # Step 6: Reset state
        self._wt_branch = None
        self._original_branch = None

        return f"Discarded branch '{branch_name}'."

    def merge_instructions(self) -> str:
        """Return human-readable merge/discard instructions.

        Returns:
            Multi-line string with automatic merge, manual merge, and
            discard instructions.
        """
        if self._wt_branch is None:
            return "No pending worktree task."
        wt_dir = self._wt_dir
        orig = self._original_branch or "<branch>"
        repo = self._repo_root or "."
        return (
            f"Task completed on branch: {self._wt_branch}\n"
            "\nTo merge automatically:\n"
            "    agent.merge()\n"
            "\nTo merge manually:\n"
            f"    cd {repo}\n"
            f"    git worktree remove {wt_dir}\n"
            f"    git checkout {orig}\n"
            f"    git merge {self._wt_branch}\n"
            f"    git branch -d {self._wt_branch}\n"
            "\nTo discard:\n"
            "    agent.discard()\n"
            "    # or manually:\n"
            f"    cd {repo}\n"
            f"    git worktree remove {wt_dir} --force\n"
            f"    git branch -D {self._wt_branch}"
        )

    @staticmethod
    def cleanup(repo_root: Path | str) -> str:
        """Scan for orphaned ``kiss/wt-*`` branches and worktrees.

        Args:
            repo_root: Root of the git repository to scan.

        Returns:
            Summary of findings and any cleanup actions taken.
        """
        repo = Path(repo_root)
        # List all kiss/wt-* branches
        result = _git(
            "for-each-ref", "--format=%(refname:short)",
            "refs/heads/kiss/wt-*",
            cwd=repo,
        )
        branches = result.stdout.strip().splitlines() if result.stdout.strip() else []

        # List worktrees
        wt_result = _git("worktree", "list", "--porcelain", cwd=repo)
        worktree_branches: set[str] = set()
        for line in wt_result.stdout.splitlines():
            if line.startswith("branch refs/heads/kiss/wt-"):
                worktree_branches.add(line.split("refs/heads/")[1])

        orphan_branches = [b for b in branches if b not in worktree_branches]
        lines = [f"Found {len(branches)} kiss/wt-* branch(es), "
                 f"{len(worktree_branches)} active worktree(s)."]

        if orphan_branches:
            lines.append(f"Orphaned branches (no worktree): {orphan_branches}")
            for b in orphan_branches:
                _git("branch", "-D", b, cwd=repo)
                lines.append(f"  Deleted: {b}")

        _git("worktree", "prune", cwd=repo)
        lines.append("Ran git worktree prune.")

        if not orphan_branches:
            lines.append("No orphans found.")

        return "\n".join(lines)


def main() -> None:  # pragma: no cover – CLI entry point requires API
    """Run WorktreeSorcarAgent from the command line."""
    import time as time_mod

    if len(sys.argv) <= 1:
        print(
            "Usage: sorcar-wt [-m MODEL] [-e ENDPOINT] [-b BUDGET] "
            "[-w WORK_DIR] [-t TASK] [-f FILE] [-n] [--chat-id ID] "
            "[-l] [--cleanup]"
        )
        sys.exit(1)

    parser = _build_chat_arg_parser()
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Scan for and clean up orphaned worktree branches",
    )
    args = parser.parse_args()

    if args.list_chat_id:
        _print_recent_chats()
        sys.exit(0)

    if args.cleanup:
        work_dir = args.work_dir or str(Path(".").resolve())
        git_result = _git("rev-parse", "--show-toplevel", cwd=work_dir)
        if git_result.returncode != 0:
            print("Not a git repo.")
            sys.exit(1)
        print(WorktreeSorcarAgent.cleanup(git_result.stdout.strip()))
        sys.exit(0)

    agent = WorktreeSorcarAgent("Worktree Sorcar Agent")
    _apply_chat_args(agent, args)

    start_time = time_mod.time()
    result = agent.run(**_build_run_kwargs(args))
    elapsed = time_mod.time() - start_time

    print(result)
    _print_run_stats(agent, elapsed)

    if agent._wt_pending:
        while True:
            choice = input("\n[m]erge / [d]iscard / [s]kip? ").strip().lower()
            if choice == "m":
                print(agent.merge())
                break
            if choice == "d":
                print(agent.discard())
                break
            if choice == "s":
                print("Skipped. Handle manually later.")
                break
            print("Invalid choice.")


if __name__ == "__main__":
    main()
