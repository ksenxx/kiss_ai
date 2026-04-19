"""Worktree-based agent that runs each task on an isolated git branch.

Creates a ``git worktree`` for every task so the user's main working tree
is never modified.  After the task the user chooses **merge**, **discard**,
or **do nothing**.  The agent refuses further tasks in the same chat
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
    _build_arg_parser,
    _build_run_kwargs,
    _print_recent_chats,
    _print_run_stats,
)
from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    MergeResult,
)
from kiss.agents.sorcar.persistence import _allocate_chat_id
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

        Queries git for any ``kiss/wt-<chat_id>-*`` branch.  If found,
        restores state from ``git config``.  If the config entry is
        missing (crash between worktree creation and config write),
        falls back to the current HEAD branch of the main worktree.

        Args:
            repo: Git repo root path.
        """
        if self._wt is not None:
            return
        prefix = f"kiss/wt-{self._chat_id}-"
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

    # -- Worktree setup ----------------------------------------------------

    def _try_setup_worktree(
        self, repo: Path, work_dir_str: str | None,
    ) -> Path | None:
        """Create a worktree branch for the current task.

        Returns the worktree-relative work directory on success, or
        ``None`` if a worktree cannot be created (caller should fall
        back to direct execution).

        Side effect: sets ``self._wt`` on success.

        Args:
            repo: Git repo root path.
            work_dir_str: Original ``work_dir`` kwarg (may be ``None``).

        Returns:
            Worktree work directory path, or ``None`` on failure.
        """
        original_branch = GitWorktreeOps.current_branch(repo)
        if original_branch is None:
            logger.warning("Detached HEAD, running task directly")
            return None

        if work_dir_str:
            try:
                offset = Path(work_dir_str).resolve().relative_to(
                    repo.resolve())
            except ValueError:  # pragma: no cover
                logger.warning("work_dir not inside repo, running directly")
                return None
        else:
            offset = Path(".")

        try:
            GitWorktreeOps.ensure_excluded(repo)
        except Exception:  # pragma: no cover — filesystem permission error
            logger.warning("Failed to update git exclude", exc_info=True)

        # Generate branch name with collision avoidance
        branch = f"kiss/wt-{self._chat_id}-{int(time.time())}"
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
            return None

        if not GitWorktreeOps.save_original_branch(repo, branch, original_branch):
            # pragma: no cover — git config failure
            GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
            return None

        self._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch=original_branch,
            wt_dir=wt_dir,
        )

        wt_work_dir = wt_dir / offset
        wt_work_dir.mkdir(parents=True, exist_ok=True)
        return wt_work_dir

    # -- Main entry point --------------------------------------------------

    def run(  # type: ignore[override]
        self,
        prompt_template: str = "",
        **kwargs: Any,
    ) -> str:
        """Run a task on an isolated git worktree branch.

        Creates a new worktree and branch, redirects ``work_dir`` into
        the worktree, and delegates to ``StatefulSorcarAgent.run()``.
        Each call starts a fresh worktree; any previously pending
        branch from an earlier run is left as-is in git for the user
        to merge or discard later.

        Falls back to direct execution (no worktree) when:
        - ``use_worktree`` kwarg is explicitly ``False``
        - ``work_dir`` is not inside a git repo
        - The repo has no commits
        - HEAD is detached (no merge target)
        - Any git command fails during setup

        Args:
            prompt_template: The task prompt.
            **kwargs: All other arguments forwarded to
                ``StatefulSorcarAgent.run()``.  The optional
                ``use_worktree`` kwarg (default ``True``) gates the
                worktree behavior — when ``False`` the call is
                equivalent to ``StatefulSorcarAgent.run()``.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        if not kwargs.pop("use_worktree", True):
            return super().run(prompt_template=prompt_template, **kwargs)

        work_dir_str = kwargs.get("work_dir")
        discovery_dir = Path(work_dir_str) if work_dir_str else Path.cwd()

        repo = GitWorktreeOps.discover_repo(discovery_dir)
        if repo is None:
            logger.warning("Not a git repo, running task directly")
            return super().run(prompt_template=prompt_template, **kwargs)

        self._restore_from_git(repo)

        # Pre-allocate a chat_id so the worktree branch name is stable.
        # Without this, _chat_id would still be 0 here and the branch
        # would be named kiss/wt-0-<ts>, but _add_task in super().run()
        # would then assign a different id, breaking _restore_from_git.
        if self._chat_id == "":
            self._chat_id = _allocate_chat_id()

        wt_work_dir = self._try_setup_worktree(repo, work_dir_str)
        if wt_work_dir is None:
            return super().run(prompt_template=prompt_template, **kwargs)

        # Notify VS Code extension so the worktree appears in the SCM panel
        printer = kwargs.get("printer")
        if printer and hasattr(printer, "broadcast"):
            printer.broadcast({
                "type": "worktree_created",
                "worktreeDir": str(self._wt_dir),
                "branch": self._wt_branch,
            })

        kwargs["work_dir"] = str(wt_work_dir)

        try:
            return super().run(prompt_template=prompt_template, **kwargs)
        except KISSError:
            raise
        except Exception as exc:
            return str(yaml.dump({
                "success": False,
                "summary": f"Task failed with error: {exc}",
            }))

    # -- Merge / discard / do nothing --------------------------------------

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

        result = GitWorktreeOps.squash_merge_branch(wt.repo_root, wt.branch)

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
        """Throw away the task branch and worktree, checkout original.

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
        if wt.original_branch:
            GitWorktreeOps.checkout(wt.repo_root, wt.original_branch)
        GitWorktreeOps.delete_branch(wt.repo_root, wt.branch)
        self._wt = None
        return f"Discarded branch '{wt.branch}'."

    def do_nothing(self) -> str:
        """Leave the worktree branch as-is without any git operation.

        Clears the pending worktree state so the user regains control
        and can start new tasks.  The branch and any committed work
        remain in git for the user to merge or discard manually later.

        Returns:
            Informational message with the branch name.

        Raises:
            RuntimeError: If no worktree task is pending.
        """
        if self._wt is None:
            raise RuntimeError("No pending worktree task")

        wt = self._wt
        if wt.wt_dir.exists():
            self._auto_commit_worktree()
            GitWorktreeOps.remove(wt.repo_root, wt.wt_dir)
        GitWorktreeOps.prune(wt.repo_root)
        if wt.original_branch:
            GitWorktreeOps.checkout(wt.repo_root, wt.original_branch)
        self._wt = None
        return (
            f"Left branch '{wt.branch}' as-is.  You can merge or "
            f"delete it later:\n"
            f"    git merge {wt.branch}\n"
            f"    git branch -d {wt.branch}"
        )

    # -- Instructions ------------------------------------------------------

    def merge_instructions(self) -> str:
        """Return human-readable merge/discard/do-nothing instructions.

        Returns:
            Multi-line string with merge, discard, and do-nothing
            instructions.
        """
        if self._wt is None:
            return "No pending worktree task."
        wt = self._wt
        orig = wt.original_branch or "<branch>"
        return (
            f"Task completed on branch: {wt.branch}\n"
            "\nTo commit and merge:\n"
            "    agent.merge()\n"
            "\nTo discard:\n"
            "    agent.discard()\n"
            "\nTo do nothing (keep the branch for later):\n"
            "    agent.do_nothing()\n"
            "\nOr manually:\n"
            f"    cd {wt.repo_root}\n"
            f"    git checkout {orig}\n"
            f"    git merge {wt.branch}\n"
            f"    git branch -d {wt.branch}"
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


def main() -> None:  # pragma: no cover – CLI entry point requires API
    """Run SorcarAgent, StatefulSorcarAgent, or WorktreeSorcarAgent from the CLI.

    Uses ``--use-chat`` or ``--use-worktree`` to select the agent
    type.  Defaults to base SorcarAgent when neither flag is given.
    """
    import time as time_mod

    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.list_chat_id:
        _print_recent_chats()
        sys.exit(0)

    work_dir = args.work_dir or str(Path(".").resolve())

    if args.cleanup:
        repo = GitWorktreeOps.discover_repo(Path(work_dir))
        if repo is None:
            print("Not a git repo.")
            sys.exit(1)
        print(WorktreeSorcarAgent.cleanup(repo))
        sys.exit(0)

    if args.use_worktree:
        agent: SorcarAgent = WorktreeSorcarAgent("Worktree Sorcar Agent")
    elif args.use_chat:
        agent = StatefulSorcarAgent("Stateful Sorcar Agent")
    else:
        agent = SorcarAgent("Sorcar Agent")

    run_kwargs = _build_run_kwargs(args)
    if isinstance(agent, StatefulSorcarAgent):
        _apply_chat_args(agent, args, task=run_kwargs.get("prompt_template", ""))

    start_time = time_mod.time()
    result = agent.run(**run_kwargs)
    elapsed = time_mod.time() - start_time

    print(result)
    if isinstance(agent, StatefulSorcarAgent):
        _print_run_stats(agent, elapsed)
    else:
        print(f"\nTime: {elapsed:.1f}s")
        print(f"Cost: ${agent.budget_used:.4f}")
        print(f"Total tokens: {agent.total_tokens_used}")

    if isinstance(agent, WorktreeSorcarAgent) and agent._wt_pending:
        while True:
            choice = (
                input("\n[c]ommit and merge / [d]iscard / do [n]othing? ")
                .strip().lower()
            )
            if choice == "c":
                print(agent.merge())
                break
            if choice == "d":
                print(agent.discard())
                break
            if choice == "n":
                print(agent.do_nothing())
                break
            print("Invalid choice.")


if __name__ == "__main__":
    main()
