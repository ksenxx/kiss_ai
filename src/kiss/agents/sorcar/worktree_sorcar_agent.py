"""Worktree-based agent that runs each task on an isolated git branch.

Creates a ``git worktree`` for every task so the user's main working tree
is never modified.  After the task the user chooses **merge** or
**discard**.
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
    repo_lock,
)
from kiss.agents.sorcar.persistence import _allocate_chat_id
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.core.kiss_error import KISSError

logger = logging.getLogger(__name__)


def _generate_commit_message(wt_dir: Path) -> str:
    """Generate a commit message for worktree changes using an LLM.

    Gets the staged diff and delegates to
    :func:`~kiss.agents.vscode.helpers.generate_commit_message_from_diff`.

    Args:
        wt_dir: The worktree directory containing staged changes.

    Returns:
        A commit message string.
    """
    from kiss.agents.vscode.helpers import generate_commit_message_from_diff

    diff_text = GitWorktreeOps.staged_diff(wt_dir)
    return generate_commit_message_from_diff(diff_text)


def _manual_merge_cmd(wt: GitWorktree) -> str:
    """Return the correct manual merge command for a worktree.

    When a baseline commit exists, the auto-merge uses
    ``cherry-pick --no-commit baseline..branch`` to replay only agent
    commits.  ``git merge --squash`` would incorrectly include the
    baseline's dirty-state snapshot.

    Args:
        wt: The worktree state.

    Returns:
        A shell command string for manual merge.
    """
    if wt.baseline_commit:
        return f"git cherry-pick --no-commit {wt.baseline_commit}..{wt.branch}"
    return f"git merge --squash {wt.branch}"


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
        self._stash_pop_warning: str | None = None
        self._merge_conflict_warning: str | None = None


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

    @property
    def _baseline_commit(self) -> str | None:
        """SHA of the baseline commit (user's dirty state), or ``None``."""
        return self._wt.baseline_commit if self._wt else None


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

        baseline = GitWorktreeOps.load_baseline_commit(repo, branch)

        slug = branch.replace("/", "_")
        wt_dir = repo / ".kiss-worktrees" / slug
        self._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch=original,
            wt_dir=wt_dir,
            baseline_commit=baseline,
        )


    def _auto_commit_worktree(self) -> bool:
        """Commit any uncommitted changes in the worktree.

        Stages all changes once, generates a commit message from the
        staged diff, then commits the already-staged changes (without
        re-staging).  Falls back to a generic commit message when the
        LLM-based message generator is unavailable.

        Returns:
            True if a commit was created, False if nothing to commit.
        """
        if self._wt is None or not self._wt.wt_dir.exists():
            return False
        GitWorktreeOps.stage_all(self._wt.wt_dir)
        try:
            msg = _generate_commit_message(self._wt.wt_dir)
        except Exception:
            logger.debug("LLM commit message generation failed; using fallback", exc_info=True)
            msg = "kiss: auto-commit agent changes"
        return GitWorktreeOps.commit_staged(self._wt.wt_dir, msg)


    def _finalize_worktree(self) -> bool:
        """Auto-commit, remove worktree, prune.

        Returns:
            True if the worktree was cleaned up successfully.  False if
            uncommitted changes remain after the auto-commit attempt
            (e.g. a pre-commit hook rejected the commit) — the worktree
            directory is preserved so no work is lost.
        """
        assert self._wt is not None
        wt = self._wt
        if wt.wt_dir.exists():
            self._auto_commit_worktree()
            if GitWorktreeOps.has_uncommitted_changes(wt.wt_dir):
                logger.warning(
                    "Worktree has uncommitted changes after auto-commit "
                    "(pre-commit hook may have rejected); preserving: %s",
                    wt.wt_dir,
                )
                return False
            GitWorktreeOps.remove(wt.repo_root, wt.wt_dir)
        GitWorktreeOps.prune(wt.repo_root)
        return True

    def _do_merge(
        self,
        wt: GitWorktree,
    ) -> tuple[MergeResult, str]:
        """Checkout, stash, squash-merge, pop for a worktree branch.

        Serialized under ``repo_lock`` to prevent concurrent tabs from
        interleaving operations on the main repository.

        Args:
            wt: The worktree state to merge.

        Returns:
            ``(result, stash_warning)`` where *result* is the merge
            outcome and *stash_warning* is a non-empty string if
            stash-pop failed.  Checkout failures return
            ``(MergeResult.CHECKOUT_FAILED, "")``.
        """
        stash_warning = ""
        if wt.original_branch is None:
            return (MergeResult.CHECKOUT_FAILED, "")
        with repo_lock(wt.repo_root):
            current = GitWorktreeOps.current_branch(wt.repo_root)
            if current != wt.original_branch:
                ok, err = GitWorktreeOps.checkout(
                    wt.repo_root,
                    wt.original_branch,
                )
                if not ok:
                    logger.warning(
                        "Cannot checkout '%s': %s",
                        wt.original_branch,
                        err,
                    )
                    return (MergeResult.CHECKOUT_FAILED, "")

            did_stash = GitWorktreeOps.stash_if_dirty(wt.repo_root)
            if wt.baseline_commit:
                result = GitWorktreeOps.squash_merge_from_baseline(
                    wt.repo_root,
                    wt.branch,
                    wt.baseline_commit,
                )
            else:
                result = GitWorktreeOps.squash_merge_branch(
                    wt.repo_root,
                    wt.branch,
                )
            if did_stash:
                if result == MergeResult.SUCCESS:
                    if not GitWorktreeOps.stash_pop(wt.repo_root):
                        stash_warning = (
                            "Your uncommitted changes could not be "
                            "auto-restored after merging the previous "
                            f"worktree ('{wt.branch}'). Run "
                            "'git stash pop' to recover them."
                        )
                        logger.warning(
                            "git stash pop failed after merge of '%s'",
                            wt.branch,
                        )
                else:
                    stash_warning = (
                        "Your uncommitted changes were saved before "
                        "the merge attempt and are safe in "
                        "'git stash'. After resolving the merge, "
                        "run 'git stash pop' to restore them."
                    )

            if result == MergeResult.SUCCESS:
                GitWorktreeOps.delete_branch(wt.repo_root, wt.branch)

        return (result, stash_warning)

    def _release_worktree(self) -> str | None:
        """Auto-commit, auto-merge, and clean up a pending worktree.

        Called when the user starts a new chat or a new task without
        explicitly choosing merge/discard/do-nothing for the pending
        worktree.  Generates a detailed LLM commit message, squash-
        merges the task branch into the original branch, and deletes
        the task branch.

        If the merge fails (conflict or checkout failure), the branch
        is kept in git for manual resolution, ``self._wt`` is cleared,
        ``_merge_conflict_warning`` is set, and ``None`` is returned
        so the caller knows the release did not fully succeed.

        Safe for concurrent use: a per-repo lock serializes the
        checkout → stash → merge → pop sequence so concurrent tabs
        cannot interleave operations on the same main repository.

        Returns:
            The branch name that the main worktree ends up on after
            a successful release (i.e. the original branch), or
            ``None`` if no worktree was pending, the release failed,
            or a merge conflict occurred.
        """
        if self._wt is None:
            return None
        wt = self._wt

        if not self._finalize_worktree():
            self._merge_conflict_warning = (
                f"Could not auto-commit worktree changes for "
                f"'{wt.branch}' (a pre-commit hook may have rejected "
                f"the commit). The worktree is preserved at: {wt.wt_dir}"
            )
            self._wt = None
            return None

        if not wt.original_branch:
            self._merge_conflict_warning = (
                f"Could not auto-merge branch '{wt.branch}' because "
                "the original branch is unknown (likely due to a crash "
                "during setup).  The branch is kept for manual resolution."
            )
            self._wt = None
            return None

        result, stash_warning = self._do_merge(wt)
        if stash_warning:
            self._stash_pop_warning = stash_warning

        merge_cmd = _manual_merge_cmd(wt)

        stash_suffix = ""
        if stash_warning:
            stash_suffix = "\n    git stash pop  # restore your uncommitted changes"

        if result == MergeResult.CHECKOUT_FAILED:
            self._merge_conflict_warning = (
                f"Auto-merge of '{wt.branch}' could not checkout "
                f"'{wt.original_branch}'. The branch is kept for "
                "manual resolution."
            )
            self._wt = None
            return None
        if result == MergeResult.MERGE_FAILED:
            self._merge_conflict_warning = (
                f"Auto-merge of '{wt.branch}' into "
                f"'{wt.original_branch}' applied cleanly but "
                "the commit failed (a pre-commit hook may have "
                "rejected it). The branch is kept for manual "
                "resolution. Run:\n"
                f"    cd {wt.repo_root}\n"
                f"    git checkout {wt.original_branch}\n"
                f"    {merge_cmd}\n"
                "    # fix pre-commit issues, then:\n"
                "    git commit --no-verify\n"
                f"    git branch -d {wt.branch}" + stash_suffix
            )
            logger.warning(
                "Auto-merge of '%s' into '%s': commit failed (pre-commit hook?); branch kept",
                wt.branch,
                wt.original_branch,
            )
            self._wt = None
            return None
        if result == MergeResult.CONFLICT:
            self._merge_conflict_warning = (
                f"Auto-merge of '{wt.branch}' into "
                f"'{wt.original_branch}' had conflicts. The "
                "branch is kept for manual resolution. Run:\n"
                f"    cd {wt.repo_root}\n"
                f"    git checkout {wt.original_branch}\n"
                f"    {merge_cmd}\n"
                "    # resolve conflicts, then:\n"
                "    git add . && git commit\n"
                f"    git branch -d {wt.branch}" + stash_suffix
            )
            logger.warning(
                "Auto-merge of '%s' into '%s' had conflicts; branch kept for manual resolution",
                wt.branch,
                wt.original_branch,
            )
            self._wt = None
            return None

        released_branch = wt.original_branch
        self._wt = None
        return released_branch


    def new_chat(self) -> None:
        """Reset to a new chat session, auto-merging any pending worktree.

        If a worktree task is pending from the previous session, it is
        auto-committed with a detailed LLM message and squash-merged
        into the original branch before the chat state is reset.
        """
        self._release_worktree()
        super().new_chat()


    def _try_setup_worktree(
        self,
        repo: Path,
        work_dir_str: str | None,
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
        released_branch = self._release_worktree()

        original_branch: str | None
        if released_branch is not None:
            original_branch = released_branch
        else:
            with repo_lock(repo):
                original_branch = GitWorktreeOps.current_branch(repo)
        if original_branch is None:
            logger.warning("Detached HEAD, running task directly")
            return None

        if work_dir_str:
            try:
                offset = Path(work_dir_str).resolve().relative_to(repo.resolve())
            except ValueError:  # pragma: no cover
                logger.warning("work_dir not inside repo, running directly")
                return None
        else:
            offset = Path(".")

        try:
            GitWorktreeOps.ensure_excluded(repo)
        except Exception:  # pragma: no cover — filesystem permission error
            logger.warning("Failed to update git exclude", exc_info=True)

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

        baseline_commit: str | None = None
        if GitWorktreeOps.copy_dirty_state(repo, wt_dir):
            GitWorktreeOps.stage_all(wt_dir)
            if GitWorktreeOps.commit_staged(
                wt_dir,
                "kiss: baseline from dirty state",
                no_verify=True,
            ):
                baseline_commit = GitWorktreeOps.head_sha(wt_dir)
                if baseline_commit:
                    GitWorktreeOps.save_baseline_commit(
                        repo,
                        branch,
                        baseline_commit,
                    )

        self._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch=original_branch,
            wt_dir=wt_dir,
            baseline_commit=baseline_commit,
        )

        wt_work_dir = wt_dir / offset
        wt_work_dir.mkdir(parents=True, exist_ok=True)
        return wt_work_dir


    def _flush_warnings(self, printer: Any) -> None:
        """Broadcast and clear any pending stash/merge warnings.

        Called on every ``run()`` code path (success and all three
        fallbacks) so that warnings set by ``_release_worktree`` or by
        server-side BUG-B handling are never silently dropped.

        Args:
            printer: An object with a ``broadcast(event)`` method, or
                any other value (ignored when ``broadcast`` is absent).
        """
        if printer is None or not hasattr(printer, "broadcast"):
            return
        if self._stash_pop_warning:
            printer.broadcast(
                {"type": "warning", "message": self._stash_pop_warning}
            )
            self._stash_pop_warning = None
        if self._merge_conflict_warning:
            printer.broadcast(
                {"type": "warning", "message": self._merge_conflict_warning}
            )
            self._merge_conflict_warning = None


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
            self._flush_warnings(kwargs.get("printer"))
            return super().run(prompt_template=prompt_template, **kwargs)

        work_dir_str = kwargs.get("work_dir")
        discovery_dir = Path(work_dir_str) if work_dir_str else Path.cwd()

        repo = GitWorktreeOps.discover_repo(discovery_dir)
        if repo is None:
            logger.warning("Not a git repo, running task directly")
            self._flush_warnings(kwargs.get("printer"))
            return super().run(prompt_template=prompt_template, **kwargs)

        if self._chat_id == "":
            self._chat_id = _allocate_chat_id()

        self._restore_from_git(repo)

        wt_work_dir = self._try_setup_worktree(repo, work_dir_str)
        if wt_work_dir is None:
            self._flush_warnings(kwargs.get("printer"))
            return super().run(prompt_template=prompt_template, **kwargs)

        printer = kwargs.get("printer")
        self._flush_warnings(printer)
        if printer and hasattr(printer, "broadcast"):
            printer.broadcast(
                {
                    "type": "worktree_created",
                    "worktreeDir": str(self._wt_dir),
                    "branch": self._wt_branch,
                }
            )

        kwargs["work_dir"] = str(wt_work_dir)

        try:
            return super().run(prompt_template=prompt_template, **kwargs)
        except KISSError:
            raise
        except Exception as exc:
            return str(
                yaml.dump(
                    {
                        "success": False,
                        "summary": f"Task failed with error: {exc}",
                    }
                )
            )


    def merge(self) -> str:
        """Merge the task branch into the original branch.

        Every step is idempotent — safe to re-run after a crash.
        Auto-commits any uncommitted changes in the worktree before
        merging.  If the main working tree has uncommitted changes,
        they are stashed before the merge and restored afterward so
        user edits don't block the merge.

        Returns:
            Success message, or error message if merge fails.

        Raises:
            RuntimeError: If no worktree task is pending.
        """
        if self._wt is None:
            raise RuntimeError("No pending worktree task to merge")

        wt = self._wt

        if wt.original_branch is None:
            merge_cmd = _manual_merge_cmd(wt)
            return (
                "Cannot merge: original branch is unknown (likely due to a "
                "crash during setup).  Please specify the target branch "
                "manually:\n"
                f"    git checkout <branch> && {merge_cmd}"
            )

        if wt.wt_dir.exists():
            if not self._finalize_worktree():
                return (
                    f"Cannot merge: auto-commit for '{wt.branch}' failed "
                    "(a pre-commit hook may have rejected the commit). "
                    f"The worktree is preserved at: {wt.wt_dir}\n\n"
                    "Fix the issue, then commit manually:\n"
                    f"    cd {wt.wt_dir}\n"
                    "    git add -A && git commit -m 'agent work'\n\n"
                    "Then retry: agent.merge()"
                )

        result, stash_warning = self._do_merge(wt)
        merge_cmd = _manual_merge_cmd(wt)
        stash_suffix = ""
        if stash_warning:
            stash_suffix = "\n\n⚠️  " + stash_warning

        if result == MergeResult.CHECKOUT_FAILED:
            return (
                f"Cannot checkout '{wt.original_branch}'.\n"
                "Fix the issue and retry merge(), or call discard()."
            )

        if result == MergeResult.SUCCESS:
            self._wt = None
            return f"Successfully merged branch '{wt.branch}'." + stash_suffix

        stash_step = ""
        if stash_warning:
            stash_step = "    git stash pop  # restore your uncommitted changes\n"

        if result == MergeResult.MERGE_FAILED:
            return (
                f"Merge of '{wt.branch}' applied cleanly but the commit "
                "failed (a pre-commit hook may have rejected it). "
                "The branch is kept — retry manually:\n"
                f"    cd {wt.repo_root}\n"
                f"    git checkout {wt.original_branch}\n"
                f"    {merge_cmd}\n"
                "    # fix pre-commit issues, then:\n"
                "    git commit --no-verify\n"
                f"    git branch -d {wt.branch}\n" + stash_step + "\nOr discard the branch:\n"
                "    agent.discard()" + stash_suffix
            )

        return (
            "Merge conflict detected.  Resolve manually:\n"
            f"    cd {wt.repo_root}\n"
            f"    git checkout {wt.original_branch}\n"
            f"    {merge_cmd}\n"
            "    # resolve conflicts in your editor\n"
            "    git add .\n"
            "    git commit\n"
            f"    git branch -d {wt.branch}\n" + stash_step + "\nOr discard the branch:\n"
            "    agent.discard()"
        )

    def discard(self) -> str:
        """Throw away the task branch and worktree, checkout original.

        Every step is idempotent — safe to call multiple times.
        Acquires ``repo_lock`` to serialize against concurrent
        merge/release operations on the same repository.

        Returns:
            Confirmation message (includes a warning if checkout
            to the original branch failed).

        Raises:
            RuntimeError: If no worktree task is pending.
        """
        if self._wt is None:
            raise RuntimeError("No pending worktree task to discard")

        wt = self._wt
        checkout_warning = ""
        delete_warning = ""
        with repo_lock(wt.repo_root):
            GitWorktreeOps.remove(wt.repo_root, wt.wt_dir)
            GitWorktreeOps.prune(wt.repo_root)
            if wt.original_branch:
                ok, err = GitWorktreeOps.checkout(
                    wt.repo_root,
                    wt.original_branch,
                )
                if not ok:
                    checkout_warning = f"\n⚠️  Could not checkout '{wt.original_branch}': {err}"
            if not GitWorktreeOps.delete_branch(wt.repo_root, wt.branch):
                delete_warning = (
                    f"\n⚠️  Branch '{wt.branch}' could not be deleted "
                    "and still exists.  Switch to a different branch "
                    f"(e.g. 'git checkout <other>') and run "
                    f"'git branch -D {wt.branch}' to remove it."
                )
        self._wt = None
        if delete_warning:
            return f"Partially discarded branch '{wt.branch}'.{checkout_warning}{delete_warning}"
        return f"Discarded branch '{wt.branch}'.{checkout_warning}"


    def merge_instructions(self) -> str:
        """Return human-readable merge/discard instructions.

        Returns:
            Multi-line string with merge and discard instructions.
        """
        if self._wt is None:
            return "No pending worktree task."
        wt = self._wt
        orig = wt.original_branch or "<branch>"
        merge_cmd = _manual_merge_cmd(wt)
        return (
            f"Task completed on branch: {wt.branch}\n"
            "\nTo commit and merge:\n"
            "    agent.merge()\n"
            "\nTo discard:\n"
            "    agent.discard()\n"
            "\nOr manually:\n"
            f"    cd {wt.repo_root}\n"
            f"    git checkout {orig}\n"
            f"    {merge_cmd}\n"
            "    git commit\n"
            f"    git branch -d {wt.branch}"
        )


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
            choice = input("\n[c]ommit and merge / [d]iscard? ").strip().lower()
            if choice == "c":
                print(agent.merge())
                break
            if choice == "d":
                print(agent.discard())
                break
            print("Invalid choice.")


if __name__ == "__main__":
    main()
