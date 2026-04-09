"""Worktree-based agent that runs each task on an isolated git branch.

Creates a ``git worktree`` for every task so the user's main working tree
is never modified.  After the task the user chooses **merge**, **manual
merge**, or **discard**.  The agent refuses further tasks in the same chat
session until the branch is resolved.
"""

from __future__ import annotations

import logging
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
from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    MergeResult,
)
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.core.kiss_error import KISSError

logger = logging.getLogger(__name__)


def _generate_commit_message(wt_dir: Path) -> str:
    """Generate a commit message for worktree changes using an LLM.

    Gets the staged diff and asks a fast model to produce a
    conventional-commit-style message.  Returns a fallback message
    on any failure.

    Args:
        wt_dir: The worktree directory containing staged changes.

    Returns:
        A commit message string.
    """
    fallback = "kiss: auto-commit agent work"
    try:
        diff_text = GitWorktreeOps.staged_diff(wt_dir)
        if not diff_text:
            return fallback

        from kiss.agents.vscode.helpers import fast_model_for
        from kiss.core.kiss_agent import KISSAgent

        agent = KISSAgent("Commit Message Generator")
        raw = agent.run(
            model_name=fast_model_for(),
            prompt_template=(
                "Generate a concise git commit message for these "
                "changes. Use conventional commit format with a "
                "clear subject line (type: description) and "
                "optionally a body with bullet points for multiple "
                "changes. Return ONLY the commit message text, no "
                "quotes or markdown fences.\n\n{context}"
            ),
            arguments={"context": f"Diff:\n{diff_text}"},
            is_agentic=False,
            verbose=False,
        )
        msg = raw.strip().strip('"').strip("'")
        return msg if msg else fallback
    except Exception:
        logger.debug("Commit message generation failed", exc_info=True)
        return fallback


class WorktreeSorcarAgent(StatefulSorcarAgent):
    """SorcarAgent that isolates every task in a git worktree.

    State is stored entirely in git (branches and config) — no sidecar
    files.  On process restart, ``_restore_from_git()`` reconstructs all
    instance attributes from git queries.

    Attributes:
        _wt: The current/pending worktree state, or ``None`` when idle.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._wt: GitWorktree | None = None

    # -- Derived properties (preserve public API) --------------------------

    @property
    def _repo_root(self) -> Path | None:
        """Git repo root path, or ``None`` if not in a repo."""
        return self._wt.repo_root if self._wt else None

    @property
    def _wt_branch(self) -> str | None:
        """Branch name of the current/pending worktree task."""
        return self._wt.branch if self._wt else None

    @property
    def _original_branch(self) -> str | None:
        """The branch the user was on when the task started."""
        return self._wt.original_branch if self._wt else None

    @property
    def _wt_pending(self) -> bool:
        """Whether a worktree task is pending merge/discard."""
        return self._wt is not None

    @property
    def _wt_dir(self) -> Path | None:
        """Worktree directory path."""
        return self._wt.wt_dir if self._wt else None

    # -- State management --------------------------------------------------

    def _restore_from_git(self, repo: Path) -> None:
        """Restore pending-branch state from git (no sidecar files).

        Queries git for any ``kiss/wt-<chat_id[:12]>-*`` branch.  If
        found, restores state from ``git config``.  If the config entry
        is missing (crash between worktree creation and config write),
        falls back to the current HEAD branch of the main worktree.

        Args:
            repo: Git repo root path.
        """
        if self._wt is not None:
            return
        prefix = f"kiss/wt-{self._chat_id[:12]}-"
        branch = GitWorktreeOps.find_pending_branch(repo, prefix)
        if branch is None:
            return

        original = GitWorktreeOps.load_original_branch(repo, branch)
        if original is None:
            original = GitWorktreeOps.current_branch(repo)

        slug = branch.replace("/", "_")
        wt_dir = repo / ".kiss-worktrees" / slug
        self._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch=original,
            wt_dir=wt_dir,
        )

    # -- Auto-commit -------------------------------------------------------

    def _auto_commit_worktree(self) -> bool:
        """Commit any uncommitted changes in the worktree.

        Returns:
            True if a commit was created, False if nothing to commit.
        """
        if self._wt is None or not self._wt.wt_dir.exists():
            return False
        GitWorktreeOps.stage_all(self._wt.wt_dir)
        msg = _generate_commit_message(self._wt.wt_dir)
        return GitWorktreeOps.commit_all(self._wt.wt_dir, msg)

    # -- Shared preamble ---------------------------------------------------

    def _finalize_worktree(self) -> None:
        """Auto-commit, remove worktree, prune, checkout original."""
        assert self._wt is not None
        wt = self._wt
        if wt.wt_dir.exists():
            self._auto_commit_worktree()
            GitWorktreeOps.remove(wt.repo_root, wt.wt_dir)
        GitWorktreeOps.prune(wt.repo_root)

    # -- Main entry point --------------------------------------------------

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
        work_dir_str = kwargs.get("work_dir")
        discovery_dir = Path(work_dir_str) if work_dir_str else Path.cwd()

        repo = GitWorktreeOps.discover_repo(discovery_dir)
        if repo is None:
            logger.warning("Not a git repo, running task directly")
            return super().run(prompt_template=prompt_template, **kwargs)

        self._restore_from_git(repo)

        if self._wt is not None:
            blocked: str = yaml.dump({
                "success": False,
                "summary": (
                    f"Cannot start a new task in this chat session: branch "
                    f"'{self._wt.branch}' is pending merge/discard.\n\n"
                    + self.merge_instructions()
                ),
            })
            return blocked

        original_branch = GitWorktreeOps.current_branch(repo)
        if original_branch is None:
            logger.warning("Detached HEAD, running task directly")
            return super().run(prompt_template=prompt_template, **kwargs)

        if work_dir_str:
            try:
                offset = Path(work_dir_str).resolve().relative_to(
                    repo.resolve())
            except ValueError:  # pragma: no cover
                logger.warning("work_dir not inside repo, running directly")
                return super().run(prompt_template=prompt_template, **kwargs)
        else:
            offset = Path(".")

        try:
            GitWorktreeOps.ensure_excluded(repo)
        except Exception:  # pragma: no cover — filesystem permission error
            logger.warning("Failed to update git exclude", exc_info=True)

        # Generate branch name with collision avoidance
        branch = f"kiss/wt-{self._chat_id[:12]}-{int(time.time())}"
        base_branch = branch
        suffix = 1
        while GitWorktreeOps.branch_exists(repo, branch):  # pragma: no branch
            branch = f"{base_branch}-{suffix}"
            suffix += 1

        slug = branch.replace("/", "_")
        wt_dir = repo / ".kiss-worktrees" / slug

        if not GitWorktreeOps.create(repo, branch, wt_dir):
            # pragma: no cover — git worktree add failure
            GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
            return super().run(prompt_template=prompt_template, **kwargs)

        if not GitWorktreeOps.save_original_branch(repo, branch, original_branch):
            # pragma: no cover — git config failure
            GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
            return super().run(prompt_template=prompt_template, **kwargs)

        self._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch=original_branch,
            wt_dir=wt_dir,
        )

        wt_work_dir = wt_dir / offset
        wt_work_dir.mkdir(parents=True, exist_ok=True)
        kwargs["work_dir"] = str(wt_work_dir)

        try:
            task_result = super().run(
                prompt_template=prompt_template, **kwargs)
        except KISSError:
            raise
        except Exception as exc:
            task_result = yaml.dump({
                "success": False,
                "summary": f"Task failed with error: {exc}",
            })

        return task_result + "\n\n---\n" + self.merge_instructions()

    # -- Merge / discard / manual merge ------------------------------------

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
        if self._wt is None:
            raise RuntimeError("No pending worktree task to merge")

        wt = self._wt

        if wt.original_branch is None:
            return (
                "Cannot merge: original branch is unknown (likely due to a "
                "crash during setup).  Please specify the target branch "
                "manually:\n"
                f"    git checkout <branch> && git merge {wt.branch}"
            )

        self._finalize_worktree()

        if not GitWorktreeOps.checkout(wt.repo_root, wt.original_branch):
            # pragma: no cover — dirty main worktree
            return (
                f"Cannot checkout '{wt.original_branch}': "
                f"{GitWorktreeOps.checkout_error(wt.repo_root, wt.original_branch)}\n"
                "Fix the issue and retry merge(), or call discard()."
            )

        result = GitWorktreeOps.merge_branch(wt.repo_root, wt.branch)

        if result == MergeResult.SUCCESS:
            GitWorktreeOps.delete_branch(wt.repo_root, wt.branch)
            self._wt = None
            return f"Successfully merged branch '{wt.branch}'."

        # Conflict — state preserved so discard() still works
        return (
            "Merge conflict detected.  Resolve manually:\n"
            f"    cd {wt.repo_root}\n"
            f"    git checkout {wt.original_branch}\n"
            f"    git merge {wt.branch}\n"
            "    # resolve conflicts in your editor\n"
            "    git add .\n"
            "    git commit\n"
            f"    git branch -d {wt.branch}\n"
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
        if self._wt is None:
            raise RuntimeError("No pending worktree task to discard")

        wt = self._wt
        GitWorktreeOps.remove(wt.repo_root, wt.wt_dir)
        GitWorktreeOps.prune(wt.repo_root)
        GitWorktreeOps.delete_branch(wt.repo_root, wt.branch)
        self._wt = None
        return f"Discarded branch '{wt.branch}'."

    def manual_merge(self) -> str:
        """Merge task branch with ``--no-commit`` for interactive review.

        Prepares changes in the main working tree so the user can
        selectively stage/discard individual hunks using VS Code's
        Source Control UI.  Changes are left unstaged (via
        ``git reset HEAD``) when there are no conflicts.

        Returns:
            Human-readable status message.

        Raises:
            RuntimeError: If no worktree task is pending.
        """
        if self._wt is None:
            raise RuntimeError("No pending worktree task to merge")

        wt = self._wt

        if wt.original_branch is None:
            return (
                "Cannot merge: original branch is unknown.  "
                "Please merge manually:\n"
                f"    git checkout <branch> && git merge {wt.branch}"
            )

        self._finalize_worktree()

        if not GitWorktreeOps.checkout(wt.repo_root, wt.original_branch):
            return (
                f"Cannot checkout '{wt.original_branch}': "
                f"{GitWorktreeOps.checkout_error(wt.repo_root, wt.original_branch)}\n"
                "Fix the issue and retry, or call discard()."
            )

        branch_name = wt.branch
        mr = GitWorktreeOps.manual_merge_branch(wt.repo_root, wt.branch)

        if mr.status == MergeResult.MERGE_FAILED:
            return (
                "Merge failed: fix the issue and retry, or call discard()."
            )

        # Clean up branch and agent state
        if not mr.has_conflicts:
            GitWorktreeOps.delete_branch(wt.repo_root, wt.branch)
        self._wt = None

        if mr.has_conflicts:
            return (
                f"Merge of '{branch_name}' has conflicts. "
                "Resolve them in Source Control, then commit."
            )
        return (
            f"Changes from '{branch_name}' are ready for review. "
            "Use Source Control to stage desired hunks and commit."
        )

    # -- Instructions ------------------------------------------------------

    def merge_instructions(self) -> str:
        """Return human-readable merge/discard instructions.

        Returns:
            Multi-line string with automatic merge, manual merge, and
            discard instructions.
        """
        if self._wt is None:
            return "No pending worktree task."
        wt = self._wt
        orig = wt.original_branch or "<branch>"
        return (
            f"Task completed on branch: {wt.branch}\n"
            "\nTo merge automatically:\n"
            "    agent.merge()\n"
            "\nTo merge manually:\n"
            f"    cd {wt.repo_root}\n"
            f"    git worktree remove {wt.wt_dir}\n"
            f"    git checkout {orig}\n"
            f"    git merge {wt.branch}\n"
            f"    git branch -d {wt.branch}\n"
            "\nTo discard:\n"
            "    agent.discard()\n"
            "    # or manually:\n"
            f"    cd {wt.repo_root}\n"
            f"    git worktree remove {wt.wt_dir} --force\n"
            f"    git branch -D {wt.branch}"
        )

    # -- Cleanup -----------------------------------------------------------

    @staticmethod
    def cleanup(repo_root: Path | str) -> str:
        """Scan for orphaned ``kiss/wt-*`` branches and worktrees.

        Args:
            repo_root: Root of the git repository to scan.

        Returns:
            Summary of findings and any cleanup actions taken.
        """
        return GitWorktreeOps.cleanup_orphans(Path(repo_root))

    # -- Generate commit message (backward compat for tests) ---------------

    def _generate_worktree_commit_message(self, wt_dir: Path) -> str:
        """Generate a commit message for worktree changes using an LLM.

        Args:
            wt_dir: The worktree directory containing staged changes.

        Returns:
            A commit message string.
        """
        return _generate_commit_message(wt_dir)


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
        repo = GitWorktreeOps.discover_repo(Path(work_dir))
        if repo is None:
            print("Not a git repo.")
            sys.exit(1)
        print(WorktreeSorcarAgent.cleanup(repo))
        sys.exit(0)

    agent = WorktreeSorcarAgent("Worktree Sorcar Agent")
    run_kwargs = _build_run_kwargs(args)
    _apply_chat_args(agent, args, task=run_kwargs.get("prompt_template", ""))

    start_time = time_mod.time()
    result = agent.run(**run_kwargs)
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
