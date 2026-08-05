# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Worktree-based agent that runs each task on an isolated git branch.

Creates a ``git worktree`` for every task so the user's main working tree
is never modified.  After the task the user chooses **merge** or
**discard**.
"""

from __future__ import annotations

import enum
import functools
import logging
import shlex
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    MergeResult,
    repo_lock,
)
from kiss.agents.sorcar.persistence import _allocate_chat_id
from kiss.agents.sorcar.sorcar_agent import (
    _generate_commit_message,
    auto_commit_changes,
)
from kiss.core.kiss_error import KISSError

logger = logging.getLogger(__name__)


class _WorktreeCleanupOutcome(enum.Enum):
    """Outcome of :meth:`WorktreeSorcarAgent._commit_and_clean_worktree`."""

    COMMITTED_AND_REMOVED = "committed_and_removed"
    PRESERVED_NO_AUTOCOMMIT = "preserved_no_autocommit"
    PRESERVED_COMMIT_FAILED = "preserved_commit_failed"


_PRECOMMIT_FIX_LINES = (
    "    # fix pre-commit issues, then:\n"
    "    git commit --no-verify\n"
)


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
        rev_range = shlex.quote(f"{wt.baseline_commit}..{wt.branch}")
        return f"git cherry-pick --no-commit {rev_range}"
    return f"git merge --squash {shlex.quote(wt.branch)}"


def _merge_fix_steps(wt: GitWorktree, fix_lines: str) -> str:
    """Return the shell command block for manually completing a failed merge.

    Shared by :meth:`WorktreeSorcarAgent._release_worktree` and
    :meth:`WorktreeSorcarAgent.merge` so the checkout / merge / delete
    instructions can never drift apart.

    Args:
        wt: The worktree state.
        fix_lines: Result-specific middle lines (conflict resolution or
            pre-commit fix steps), each ending in a newline.

    The final step uses ``git branch -D`` (force): a squash-merge /
    cherry-pick resolution never records the task branch as an
    ancestor of the original branch, so ``git branch -d`` would
    ALWAYS refuse with "the branch ... is not fully merged" after the
    user faithfully completed the steps above it.  (The automatic
    path, :meth:`GitWorktreeOps.delete_branch`, falls back to ``-D``
    for the same reason.)

    Returns:
        The indented multi-line command block (no trailing newline).
    """
    return (
        f"    cd {shlex.quote(str(wt.repo_root))}\n"
        f"    git checkout {shlex.quote(wt.original_branch or '')}\n"
        f"    {_manual_merge_cmd(wt)}\n"
        + fix_lines
        + f"    git branch -D {shlex.quote(wt.branch)}"
    )


class WorktreeSorcarAgent(ChatSorcarAgent):
    """SorcarAgent that isolates every task in a git worktree.

    Each ``run()`` call creates a brand-new worktree on a fresh branch.
    Worktrees are not associated with the agent's ``chat_id``: branch
    names use a unique time + random suffix, and there is no
    cross-process state restoration based on chat session.  Any
    previous worktree owned by this agent instance is retired before
    the new one is created: auto-merged (or kept on conflict) when its
    task finished, and merely committed to its own branch when the
    task failed or was stopped.  See
    :meth:`_retire_previous_worktree`.

    Attributes:
        _wt: The current/pending worktree state, or ``None`` when idle.
    """

    uses_worktree = True

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._wt: GitWorktree | None = None
        self._stash_pop_warning: str | None = None
        self._merge_conflict_warning: str | None = None
        self._warning_lock: threading.Lock = threading.Lock()
        self.auto_commit_enabled: bool = True
        self._tab_id: str = ""
        self._task_start_ms: int = 0
        self._pending_review: bool = False
        self._last_preserve_outcome: _WorktreeCleanupOutcome | None = None


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


    def _auto_commit_worktree(self) -> bool:
        """Commit any uncommitted changes in the worktree.

        Stages all changes once, generates a commit message from the
        staged diff, then commits the already-staged changes (without
        re-staging).  Falls back to a generic commit message when the
        LLM-based message generator is unavailable.

        Emits two best-effort UI notifications via
        :meth:`_broadcast_commit_notification` when a printer with a
        ``broadcast`` method is attached: ``"Generating commit
        message"`` immediately before the (typically slow) LLM call
        that produces the commit message, and ``"Committed
        <subject>"`` once the commit lands in git.  The notifications
        are routed by tab so they appear on the owning chat webview,
        not on every open tab.

        Returns:
            True if a commit was created, False if nothing to commit.
        """
        if self._wt is None or not self._wt.wt_dir.exists():
            return False
        if not self.auto_commit_enabled:
            return False
        commit_run_id = f"autocommit-{self._tab_id}-{time.time_ns()}"
        return auto_commit_changes(
            self._wt.wt_dir,
            self._last_user_prompt or None,
            _generate_commit_message,
            notify_fn=functools.partial(
                self._broadcast_commit_notification, commit_run_id,
            ),
            task_result=getattr(self, "_last_result_summary", "") or None,
        )

    def _broadcast_commit_notification(
        self, notification_id: str, stage: str, subject: str,
    ) -> None:
        """Broadcast an auto-commit life-cycle notification to the webview.

        Hook used by
        :func:`~kiss.agents.sorcar.sorcar_agent.auto_commit_changes`:
        ``stage="generating"`` arrives immediately before the LLM
        call that generates the commit message; ``stage="committed"``
        arrives immediately after a successful commit, with
        *subject* set to the first non-empty line of the commit
        message.  Other stage values are ignored.

        Silently no-ops when no printer with a ``broadcast`` method
        is attached (e.g. the printer-free unit-test path), and never
        raises into the caller — broken UI plumbing must not block
        the commit itself.

        Args:
            notification_id: Toast id shared across both lifecycle
                stages of ONE ``_auto_commit_worktree`` invocation
                (bound via ``functools.partial``) so the webview
                updates the existing toast in place ("Generating
                commit message" → "Committed <subject>") instead of
                stacking two toasts and leaving the "Generating"
                toast lingering until its own auto-dismiss timer
                fires (Bug 2, gpt-5.5 round-1 review).  Binding the
                id per call — rather than stashing it on ``self`` —
                also means concurrent auto-commits on the same agent
                instance can never cross-pair their toasts.
            stage: ``"generating"`` or ``"committed"``.
            subject: First non-empty line of the commit message
                (used as the toast body for the ``"committed"``
                stage; ignored for ``"generating"``).
        """
        printer = getattr(self, "printer", None)
        if printer is None or not hasattr(printer, "broadcast"):
            return
        severity = "info"
        if stage == "generating":
            message = "Generating commit message"
        elif stage == "committed":
            message = f"Committed {subject}" if subject else "Committed"
        elif stage == "failed":
            # Terminal update for the sticky "generating" toast: the
            # commit did not land (e.g. a pre-commit hook rejected it),
            # so replace the toast instead of leaving it forever.
            message = (
                "Auto-commit failed (a pre-commit hook may have "
                "rejected it); the worktree is preserved"
            )
            severity = "warning"
        else:
            return
        event: dict[str, object] = {
            "type": "notification",
            "id": notification_id,
            "severity": severity,
            "message": message,
            "tabId": self._tab_id,
        }
        if stage == "generating":
            event["sticky"] = True
        try:
            printer.broadcast(event)
        except Exception:  # pragma: no cover — best-effort UI hook
            logger.debug("autocommit notification broadcast failed", exc_info=True)


    def _commit_and_clean_worktree(
        self, wt: GitWorktree
    ) -> tuple[_WorktreeCleanupOutcome, str]:
        """Auto-commit *wt*'s changes, then remove the worktree and prune.

        Shared engine of :meth:`_finalize_worktree` and
        :meth:`_preserve_pending_worktree_for_review` — the exact
        auto-commit → late-arriver-retry → preserve-or-remove sequence
        previously duplicated in both.

        After the LLM-driven auto-commit, a single-shot retry runs
        :meth:`GitWorktreeOps.commit_all` with a generic message to
        catch the very narrow remaining race where a file appears
        between :func:`~kiss.agents.sorcar.sorcar_agent.auto_commit_changes`'s
        second ``stage_all`` and its ``commit_staged`` call (e.g.
        ``PROGRESS.md`` being rewritten, ``.DS_Store`` materializing
        after an ``open`` of the report, an editor swap file
        appearing).  ``commit_all`` is a no-op when nothing is
        uncommitted, but skipping the call keeps the happy-path log
        quiet.  Under ``auto_commit_enabled=False`` the retry never
        force-commits: the worktree is preserved for manual review.

        Args:
            wt: The worktree to commit and clean up.

        Returns:
            A ``(outcome, leftover)`` pair.  ``leftover`` is the raw
            ``git status --porcelain`` output when the outcome is
            :attr:`_WorktreeCleanupOutcome.PRESERVED_COMMIT_FAILED`
            (both auto-commit and the retry left uncommitted state —
            e.g. a pre-commit hook rejection), otherwise ``""``.  The
            worktree directory is removed and pruned only on
            :attr:`_WorktreeCleanupOutcome.COMMITTED_AND_REMOVED`; the
            preserved outcomes leave it in place so no work is lost.
        """
        if wt.wt_dir.exists():
            self._auto_commit_worktree()
            if GitWorktreeOps.has_uncommitted_changes(wt.wt_dir):
                if not self.auto_commit_enabled:
                    return _WorktreeCleanupOutcome.PRESERVED_NO_AUTOCOMMIT, ""
                GitWorktreeOps.commit_all(
                    wt.wt_dir,
                    "kiss: auto-commit late-arriving changes",
                )
            if GitWorktreeOps.has_uncommitted_changes(wt.wt_dir):
                leftover = GitWorktreeOps.status_porcelain(wt.wt_dir)
                return _WorktreeCleanupOutcome.PRESERVED_COMMIT_FAILED, leftover
            GitWorktreeOps.remove(wt.repo_root, wt.wt_dir)
        GitWorktreeOps.prune(wt.repo_root)
        return _WorktreeCleanupOutcome.COMMITTED_AND_REMOVED, ""

    def _finalize_worktree(self) -> bool:
        """Auto-commit, remove worktree, prune.

        Delegates the auto-commit → late-arriver-retry →
        preserve-or-remove sequence to
        :meth:`_commit_and_clean_worktree` (see its docstring for the
        race being closed).  Only if the retry STILL leaves
        uncommitted state do we preserve the worktree and log a
        warning that includes the raw ``git status --porcelain``
        leftover, so an operator can distinguish a real pre-commit
        rejection from a race leftover or a corrupt index without
        sshing in.

        Returns:
            True if the worktree was cleaned up successfully.  False if
            uncommitted changes remain after BOTH the auto-commit and
            the late-arriver retry (e.g. a pre-commit hook rejected
            the commit, or a third write landed in the microsecond
            after the retry's own ``stage_all``) — the worktree
            directory is preserved so no work is lost.
        """
        assert self._wt is not None
        wt = self._wt
        outcome, leftover = self._commit_and_clean_worktree(wt)
        if outcome is _WorktreeCleanupOutcome.PRESERVED_NO_AUTOCOMMIT:
            return False
        if outcome is _WorktreeCleanupOutcome.PRESERVED_COMMIT_FAILED:
            logger.warning(
                "Worktree has uncommitted changes after auto-commit "
                "and late-arriver retry (possible causes: a "
                "pre-commit hook rejected the commit, a real "
                "commit failure, or a concurrent write that "
                "outraced both staging passes); preserving %s\n"
                "git status --porcelain:\n%s",
                wt.wt_dir,
                leftover,
            )
            return False
        return True

    def _do_merge(
        self,
        wt: GitWorktree,
    ) -> tuple[MergeResult, str, str]:
        """Stash, checkout, squash-merge, pop for a worktree branch.

        Serialized under ``repo_lock`` to prevent concurrent tabs from
        interleaving operations on the main repository.

        Args:
            wt: The worktree state to merge.

        Returns:
            ``(result, stash_warning, cleanup_warning)`` where *result*
            is the merge outcome, *stash_warning* is a non-empty string
            if stash-pop failed, and *cleanup_warning* is a non-empty
            string when the merge succeeded but the task branch could
            not be deleted (the caller decides how to surface it — the
            interactive ``merge()`` puts it in its immediate response,
            ``_release_worktree`` defers it via ``_set_warnings``).
            Checkout failures return
            ``MergeResult.CHECKOUT_FAILED`` (with a stash warning only
            when the pre-checkout stash could not be restored).  A
            dirty main tree whose ``git stash push`` itself failed
            (e.g. an unreadable untracked file) returns
            ``MergeResult.STASH_FAILED`` without touching the repo:
            proceeding would commit the user's staged changes into
            the squash-merge commit and the conflict-path
            ``git reset --hard`` would destroy their edits.
        """
        stash_warning = ""
        cleanup_warning = ""
        if wt.original_branch is None:
            return (MergeResult.CHECKOUT_FAILED, "", "")
        with repo_lock(wt.repo_root):
            try:
                GitWorktreeOps.ensure_scratch_merge_driver(wt.repo_root)
            except Exception:  # pragma: no cover — filesystem permission error
                logger.warning(
                    "Failed to install scratch merge driver", exc_info=True
                )
            did_stash = GitWorktreeOps.stash_if_dirty(wt.repo_root)
            if not did_stash and GitWorktreeOps.has_uncommitted_changes(
                wt.repo_root
            ):
                return (MergeResult.STASH_FAILED, "", "")
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
                    if did_stash and not GitWorktreeOps.stash_pop(wt.repo_root):
                        stash_warning = (
                            "Your uncommitted changes were stashed "
                            "before the failed checkout and could not "
                            "be auto-restored. Run 'git stash pop' to "
                            "recover them."
                        )
                    return (MergeResult.CHECKOUT_FAILED, stash_warning, "")

            user_prompt = getattr(self, "_last_user_prompt", "") or None
            task_result = getattr(self, "_last_result_summary", "") or None
            if wt.baseline_commit:
                result = GitWorktreeOps.squash_merge_from_baseline(
                    wt.repo_root,
                    wt.branch,
                    wt.baseline_commit,
                    user_prompt=user_prompt,
                    task_result=task_result,
                )
            else:
                result = GitWorktreeOps.squash_merge_branch(
                    wt.repo_root,
                    wt.branch,
                    user_prompt=user_prompt,
                    task_result=task_result,
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

            if result == MergeResult.SUCCESS and not GitWorktreeOps.delete_branch(
                wt.repo_root, wt.branch
            ):
                # The merge itself succeeded, but the task branch could
                # not be deleted (delete_branch returns False when both
                # normal and forced deletion fail).  Surface it instead
                # of silently leaking the branch and its config.
                logger.warning(
                    "Merged branch '%s' could not be deleted; it still "
                    "exists in %s",
                    wt.branch,
                    wt.repo_root,
                )
                cleanup_warning = (
                    f"Merged branch '{wt.branch}' could not be deleted "
                    "and still exists. Run "
                    f"'git branch -D {shlex.quote(wt.branch)}' to "
                    "remove it."
                )

        return (result, stash_warning, cleanup_warning)

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
            # The worktree directory is being left on disk with
            # uncommitted work.  Persist the preserve-for-review
            # marker so a future
            # :meth:`GitWorktreeOps.reclaim_orphaned_worktrees` in a
            # fresh process cannot silently publish this deliberately
            # parked work.
            GitWorktreeOps.save_preserve_marker(wt.repo_root, wt.branch)
            if not self.auto_commit_enabled:
                self._set_warnings(merge=(
                    f"Auto-commit is disabled (--no-auto-commit) and "
                    f"the worktree for '{wt.branch}' has uncommitted "
                    "changes; skipping auto-merge. The worktree is "
                    f"preserved at: {wt.wt_dir}"
                ))
            else:
                self._set_warnings(merge=(
                    f"Could not auto-commit worktree changes for "
                    f"'{wt.branch}' (a pre-commit hook may have rejected "
                    "the commit, the commit itself failed, or a "
                    "concurrent write outraced both staging passes — "
                    "see the kiss-web log for the exact leftover "
                    f"files). The worktree is preserved at: {wt.wt_dir}"
                ))
            self._wt = None
            return None

        if not wt.original_branch:
            self._set_warnings(merge=(
                f"Could not auto-merge branch '{wt.branch}' because "
                "the original branch is unknown (likely due to a crash "
                "during setup).  The branch is kept for manual resolution."
            ))
            self._wt = None
            return None

        result, stash_warning, cleanup_warning = self._do_merge(wt)
        if stash_warning:
            self._set_warnings(stash=stash_warning)
        if cleanup_warning:
            self._set_warnings(merge=cleanup_warning)

        self._wt = None

        if result == MergeResult.SUCCESS:
            return wt.original_branch

        stash_suffix = ""
        if stash_warning:
            stash_suffix = "\n    git stash pop  # restore your uncommitted changes"

        if result == MergeResult.CHECKOUT_FAILED:
            self._set_warnings(merge=(
                f"Auto-merge of '{wt.branch}' could not checkout "
                f"'{wt.original_branch}'. The branch is kept for "
                "manual resolution."
            ))
        elif result == MergeResult.STASH_FAILED:
            self._set_warnings(merge=(
                f"Auto-merge of '{wt.branch}' into "
                f"'{wt.original_branch}' was aborted: your "
                "uncommitted changes in the main repository could "
                "not be stashed (git stash push failed). The branch "
                "is kept for manual resolution. Commit or clean up "
                "your changes, then run:\n"
                + _merge_fix_steps(
                    wt,
                    "    git commit\n",
                )
            ))
            logger.warning(
                "Auto-merge of '%s' into '%s' aborted: stash of dirty main tree failed",
                wt.branch,
                wt.original_branch,
            )
        elif result == MergeResult.MERGE_FAILED:
            self._set_warnings(merge=(
                f"Auto-merge of '{wt.branch}' into "
                f"'{wt.original_branch}' applied cleanly but "
                "the commit failed (a pre-commit hook may have "
                "rejected it). The branch is kept for manual "
                "resolution. Run:\n"
                + _merge_fix_steps(wt, _PRECOMMIT_FIX_LINES) + stash_suffix
            ))
            logger.warning(
                "Auto-merge of '%s' into '%s': commit failed (pre-commit hook?); branch kept",
                wt.branch,
                wt.original_branch,
            )
        else:
            self._set_warnings(merge=(
                f"Auto-merge of '{wt.branch}' into "
                f"'{wt.original_branch}' had conflicts. The "
                "branch is kept for manual resolution. Run:\n"
                + _merge_fix_steps(
                    wt,
                    "    # resolve conflicts, then:\n"
                    "    git add . && git commit\n",
                ) + stash_suffix
            ))
            logger.warning(
                "Auto-merge of '%s' into '%s' had conflicts; branch kept for manual resolution",
                wt.branch,
                wt.original_branch,
            )
        return None


    def _preserve_pending_worktree_for_review(self) -> bool:
        """Commit pending worktree changes onto the branch, no merge.

        Called from :meth:`VSCodeServer._teardown_tab_resources` when
        the agent is in ``_pending_review`` state — that is, the
        current worktree task was stopped or failed and the user has
        not yet explicitly chosen Merge or Discard.

        Behavior:

        * Auto-commits any uncommitted changes inside the worktree so
          the in-flight partial work is captured as a real commit on
          the ``kiss/wt-*`` branch (recoverable via
          ``git checkout <branch>``).
        * Removes the worktree directory and runs ``git worktree
          prune`` so disk state is clean.  Those first two bullets are
          the ``COMMITTED_AND_REMOVED`` outcome, and only there does
          the branch carry the work.  When the commit does not happen
          — ``--no-auto-commit``, or a hook refusing it — the
          directory is deliberately **kept** instead, because it is
          then the only copy of the work; both cases warn the user
          where it is.
        * Does **not** call :meth:`_do_merge`: the partial work is
          NOT squash-merged into the user's original branch.  Closing
          the chat tab (or the WebSocket all-done close path) can
          therefore never silently overwrite the user's main branch
          with incomplete, unverified work — the user must explicitly
          recover the branch with ``git checkout <branch>``.

        Idempotent / safe: no-op when no worktree is pending.

        Records the cleanup outcome in ``_last_preserve_outcome`` so a
        caller can tell whether the work really made it onto the
        branch.  Reading the warning slot instead would be wrong: a
        broadcast that failed puts an *older* warning back
        (:meth:`_flush_warnings`), and that stale text says nothing
        about this worktree.

        Returns:
            True when a pending worktree was preserved (or had no
            uncommitted work to preserve and was just cleaned up).
            False when there was nothing to do.
        """
        self._last_preserve_outcome = None
        if self._wt is None:
            return False
        wt = self._wt
        outcome, leftover = self._commit_and_clean_worktree(wt)
        self._last_preserve_outcome = outcome
        if outcome in (
            _WorktreeCleanupOutcome.PRESERVED_NO_AUTOCOMMIT,
            _WorktreeCleanupOutcome.PRESERVED_COMMIT_FAILED,
        ):
            # Persist the "preserve for manual review" decision so a
            # future :meth:`GitWorktreeOps.reclaim_orphaned_worktrees`
            # (running in a fresh process after the original agent
            # died) does not silently publish this deliberately parked
            # work.
            GitWorktreeOps.save_preserve_marker(wt.repo_root, wt.branch)
        if outcome is _WorktreeCleanupOutcome.PRESERVED_NO_AUTOCOMMIT:
            logger.warning(
                "Auto-commit disabled (--no-auto-commit); "
                "preserving worktree '%s' with uncommitted "
                "changes for manual review at %s",
                wt.branch, wt.wt_dir,
            )
            self._set_warnings(merge=(
                f"Auto-commit is disabled, so the uncommitted changes of "
                f"branch '{wt.branch}' were left in the worktree directory "
                f"{wt.wt_dir}. Recover them there, or commit them to the "
                f"branch yourself; nothing else will clean it up."
            ))
        elif outcome is _WorktreeCleanupOutcome.PRESERVED_COMMIT_FAILED:
            logger.warning(
                "Worktree '%s' has uncommitted changes after "
                "preserve-for-review (likely a pre-commit hook "
                "rejection); preserving worktree directory %s\n"
                "git status --porcelain:\n%s",
                wt.branch, wt.wt_dir, leftover,
            )
            self._set_warnings(merge=(
                f"Branch '{wt.branch}' still has uncommitted changes — the "
                f"commit was refused, most likely by a pre-commit hook. Its "
                f"worktree directory {wt.wt_dir} was kept so the work is not "
                f"lost; recover it there. git status --porcelain:\n{leftover}"
            ))
        self._wt = None
        self._pending_review = False
        return True


    def new_chat(self) -> None:
        """Reset to a new chat session, retiring any pending worktree.

        If a worktree task is pending from the previous session it is
        retired through :meth:`_retire_previous_worktree`, so finished
        work is auto-committed and squash-merged into the original
        branch while work the user has not accepted yet — a failed or
        stopped task, flagged by ``_pending_review`` — is only
        committed to its own ``kiss/wt-*`` branch.  Opening a new chat
        is not a decision about the old task's work.

        When the release fails (merge conflict, checkout failure,
        stash failure, --no-auto-commit with uncommitted changes),
        ``_release_worktree`` sets a warning describing the manual
        recovery steps.  Flush it to the attached printer NOW: if the
        user opens a new chat and never runs another task on this
        agent instance, no later ``run()`` will ever call
        ``_flush_warnings`` and the warning would be silently lost.
        When no printer is attached the warning is retained for the
        next ``run()``'s flush (``_flush_warnings`` no-ops without a
        ``broadcast``-capable printer).
        """
        self._retire_previous_worktree()
        self._flush_warnings(getattr(self, "printer", None))
        super().new_chat()


    def _live_worktree_branches(self) -> set[str]:
        """Return the set of ``kiss/wt-*`` branches owned by live agents.

        Union of *self*'s current worktree branch (if any) and every
        other tab's live agent branch as tracked by
        :attr:`_RunningAgentState.running_agent_states`.  Used by
        :meth:`_try_setup_worktree` to build the ``exclude_branches``
        argument for
        :meth:`GitWorktreeOps.reclaim_orphaned_worktrees` so a
        concurrent tab's active worktree is never adopted, merged, or
        removed by our reclaim pass.
        """
        branches: set[str] = set()
        if self._wt is not None:
            branches.add(self._wt.branch)
        # Local import to avoid a circular import at module load time.
        from kiss.agents.sorcar.running_agent_state import _RunningAgentState
        for state in list(_RunningAgentState.running_agent_states.values()):
            agent = getattr(state, "agent", None)
            wt = getattr(agent, "_wt", None) if agent is not None else None
            if wt is not None:
                branches.add(wt.branch)
        return branches

    def _retire_previous_worktree(self) -> str | None:
        """Clear the way for a new task's worktree, without publishing.

        A new task must not inherit the previous task's branch, so the
        previous worktree has to go somewhere.  *Where* depends on
        whether the user has already been told its work is finished:

        * ``_pending_review`` NOT set — the previous task ran to
          completion, so :meth:`_release_worktree` squash-merges it
          into the original branch as it always has.
        * ``_pending_review`` set — the previous task failed or was
          stopped, and
          :meth:`_preserve_pending_worktree_for_review` states the
          contract: incomplete, unverified work is committed to its
          ``kiss/wt-*`` branch and never merged into the user's branch
          behind their back.  Releasing here would break that contract
          on a technicality — the user typed a *new* prompt, which is
          not a decision about the *old* task's work.

        Returns:
            The branch the main worktree ends up on after a successful
            release, or ``None`` when nothing was pending, the release
            failed, or the work was preserved for review.  The preserve
            path returns ``None`` because it never checks the original
            branch out, so its caller must not treat it as the branch
            the repository is now sitting on.
        """
        if self._pending_review:
            self._preserve_pending_worktree_for_review()
            self._pending_review = False
            return None
        released_branch = self._release_worktree()
        self._pending_review = False
        return released_branch

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
        # Lock-ordering discipline (ABBA deadlock avoidance):
        # _release_worktree -> _do_merge takes the PREVIOUS repo's lock.
        # For a CROSS-repository switch, acquiring the destination lock
        # first would nest the two locks in ABBA order (two agents
        # switching A->B and B->A would deadlock forever), so the release
        # runs BEFORE the destination lock is taken.  For a SAME-repo
        # switch the re-entrant per-repo lock is held continuously
        # across release and creation, keeping the released branch and
        # the new worktree's HEAD atomic against concurrent tabs.
        prev_repo_root = self._wt.repo_root if self._wt is not None else None
        cross_repo = (
            prev_repo_root is not None
            and prev_repo_root.resolve() != repo.resolve()
        )
        released_branch: str | None = None
        if cross_repo:
            released_branch = self._retire_previous_worktree()

        with repo_lock(repo):
            if not cross_repo:
                released_branch = self._retire_previous_worktree()
            original_branch: str | None
            if (
                released_branch is not None
                and prev_repo_root is not None
                and prev_repo_root.resolve() == repo.resolve()
            ):
                original_branch = released_branch
            else:
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
                GitWorktreeOps.ensure_scratch_merge_driver(repo)
                # Reclaim before sweep: reclaim may merge and delete
                # orphan branches, and sweep purges leftover config
                # sections whose branches were just removed.  Exclude
                # every branch owned by a *live* agent in this
                # process — self and every other tab — so a running
                # sibling task's worktree is never adopted or
                # destroyed by our reclaim pass.
                GitWorktreeOps.reclaim_orphaned_worktrees(
                    repo,
                    exclude_branches=self._live_worktree_branches(),
                )
                GitWorktreeOps.sweep_orphaned_state(repo)
            except Exception:  # pragma: no cover — filesystem permission error
                logger.warning("Failed to update git exclude", exc_info=True)

            branch = f"kiss/wt-{int(time.time())}-{uuid.uuid4().hex[:8]}"
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

            try:
                dirty_copied = GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            except OSError:
                logger.warning(
                    "Failed to copy dirty state into worktree; "
                    "falling back to direct execution",
                    exc_info=True,
                )
                GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
                return None

            baseline_commit: str | None = None
            if dirty_copied:
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
                elif GitWorktreeOps.has_uncommitted_changes(wt_dir):
                    logger.warning(
                        "Baseline commit failed in new worktree; "
                        "falling back to direct execution"
                    )
                    GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
                    return None

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


    def _set_warnings(
        self,
        stash: str | None = None,
        merge: str | None = None,
    ) -> None:
        """Set pending warning attribute(s) under ``_warning_lock``.

        All internal writers of ``_stash_pop_warning`` /
        ``_merge_conflict_warning`` go through this helper so a write
        can never land inside :meth:`_flush_warnings`'s atomic
        take-and-clear section and be silently wiped by its clear.

        Args:
            stash: New ``_stash_pop_warning`` value, or ``None`` to
                leave it unchanged.
            merge: New ``_merge_conflict_warning`` value, or ``None``
                to leave it unchanged.
        """
        with self._warning_lock:
            if stash is not None:
                self._stash_pop_warning = stash
            if merge is not None:
                self._merge_conflict_warning = merge

    def _flush_warnings(self, printer: Any) -> None:
        """Broadcast and clear any pending stash/merge warnings.

        Called on every ``run()`` code path (success and all three
        fallbacks) so that warnings set by ``_release_worktree`` or by
        server-side BUG-B handling are never silently dropped.

        The take-and-clear of both warning attributes happens
        atomically under ``_warning_lock`` BEFORE any broadcast, so:

        * two concurrent flushes (e.g. a starting ``run()`` racing a
          server teardown) can never both observe the same warning and
          broadcast it twice, and
        * a warning set concurrently via :meth:`_set_warnings` while a
          flush is broadcasting is never wiped by that flush's clear —
          it survives for the next flush.

        Args:
            printer: An object with a ``broadcast(event)`` method, or
                any other value (ignored when ``broadcast`` is absent).
        """
        if printer is None or not hasattr(printer, "broadcast"):
            return
        with self._warning_lock:
            stash_warning = self._stash_pop_warning
            self._stash_pop_warning = None
            merge_warning = self._merge_conflict_warning
            self._merge_conflict_warning = None
        if stash_warning:
            try:
                printer.broadcast({"type": "warning", "message": stash_warning})
            except Exception:
                # Restore so a broken printer never permanently loses the
                # (already-cleared) warning — but only when the slot is
                # still empty, so a warning set concurrently while this
                # broadcast was failing is never overwritten by the old one.
                logger.debug("stash warning broadcast failed", exc_info=True)
                with self._warning_lock:
                    if self._stash_pop_warning is None:
                        self._stash_pop_warning = stash_warning
        if merge_warning:
            try:
                printer.broadcast({"type": "warning", "message": merge_warning})
            except Exception:
                logger.debug("merge warning broadcast failed", exc_info=True)
                with self._warning_lock:
                    if self._merge_conflict_warning is None:
                        self._merge_conflict_warning = merge_warning


    def run(  # type: ignore[override]
        self,
        prompt_template: str = "",
        **kwargs: Any,
    ) -> str:
        """Run a task on an isolated git worktree branch.

        Creates a new worktree and branch, redirects ``work_dir`` into
        the worktree, and delegates to ``ChatSorcarAgent.run()``.
        Each call starts a fresh worktree; any previously pending
        branch from an earlier run is retired first by
        :meth:`_retire_previous_worktree` — auto-committed and
        squash-merged into its original branch (kept in git for manual
        resolution when that auto-merge fails or conflicts), or, when
        the earlier run failed or was stopped, committed to its own
        branch without ever touching the original one.

        Falls back to direct execution (no worktree) when:
        - ``use_worktree`` kwarg is explicitly ``False``
        - ``work_dir`` is not inside a git repo
        - The repo has no commits
        - HEAD is detached (no merge target)
        - Any git command fails during setup

        Args:
            prompt_template: The task prompt.
            **kwargs: All other arguments forwarded to
                ``ChatSorcarAgent.run()``.  The optional
                ``use_worktree`` kwarg (default ``True``) gates the
                worktree behavior — when ``False`` the call is
                equivalent to ``ChatSorcarAgent.run()``.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        if self._chat_id == "":
            self._chat_id = _allocate_chat_id()
        registered_here = self._register_running_state()

        try:
            wt_work_dir: Path | None = None
            if kwargs.pop("use_worktree", True):
                work_dir_str = kwargs.get("work_dir")
                discovery_dir = Path(work_dir_str) if work_dir_str else Path.cwd()
                repo = GitWorktreeOps.discover_repo(discovery_dir)
                if repo is None:
                    logger.warning("Not a git repo, running task directly")
                else:
                    wt_work_dir = self._try_setup_worktree(repo, work_dir_str)

            printer = kwargs.get("printer")
            self._flush_warnings(printer)
            if wt_work_dir is None:
                try:
                    return super().run(
                        prompt_template=prompt_template, **kwargs
                    )
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
        finally:
            if registered_here:
                self._unregister_running_state()


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
        self._pending_review = False

        if wt.original_branch is None:
            merge_cmd = _manual_merge_cmd(wt)
            return (
                "Cannot merge: original branch is unknown (likely due to a "
                "crash during setup).  Please specify the target branch "
                "manually:\n"
                f"    git checkout <branch> && {merge_cmd}"
            )

        if not self._finalize_worktree():
            if not self.auto_commit_enabled:
                return (
                    f"Cannot merge: auto-commit is disabled "
                    f"(--no-auto-commit) and the worktree for "
                    f"'{wt.branch}' has uncommitted changes. "
                    f"The worktree is preserved at: {wt.wt_dir}\n\n"
                    "Review and commit the changes manually:\n"
                    f"    cd {shlex.quote(str(wt.wt_dir))}\n"
                    "    git add -A && git commit -m 'agent work'\n\n"
                    "Then retry: agent.merge()"
                )
            return (
                f"Cannot merge: auto-commit for '{wt.branch}' failed "
                "(a pre-commit hook may have rejected the commit). "
                f"The worktree is preserved at: {wt.wt_dir}\n\n"
                "Fix the issue, then commit manually:\n"
                f"    cd {shlex.quote(str(wt.wt_dir))}\n"
                "    git add -A && git commit -m 'agent work'\n\n"
                "Then retry: agent.merge()"
            )

        result, stash_warning, cleanup_warning = self._do_merge(wt)
        stash_suffix = ""
        if stash_warning:
            stash_suffix = "\n\n⚠️  " + stash_warning
        if cleanup_warning:
            # Branch cleanup failed after a successful merge: put it in
            # the immediate response so the caller/UI never reports an
            # unqualified success while the task branch leaks.
            stash_suffix += "\n\n⚠️  " + cleanup_warning

        if result == MergeResult.CHECKOUT_FAILED:
            return (
                f"Cannot checkout '{wt.original_branch}'.\n"
                "Fix the issue and retry merge(), or call discard()."
                + stash_suffix
            )

        if result == MergeResult.STASH_FAILED:
            return (
                "Cannot merge: your uncommitted changes in the main "
                "repository could not be stashed (git stash push "
                "failed), so the merge was aborted to avoid mixing "
                "them into the merge commit.\n"
                "Commit or clean up the changes, then retry merge(), "
                "or call discard()." + stash_suffix
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
                + _merge_fix_steps(wt, _PRECOMMIT_FIX_LINES)
                + "\n" + stash_step + "\nOr discard the branch:\n"
                "    agent.discard()" + stash_suffix
            )

        return (
            "Merge conflict detected.  Resolve manually:\n"
            + _merge_fix_steps(
                wt,
                "    # resolve conflicts in your editor\n"
                "    git add .\n"
                "    git commit\n",
            )
            + "\n" + stash_step + "\nOr discard the branch:\n"
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
        self._pending_review = False
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
                    f"'git branch -D {shlex.quote(wt.branch)}' to remove it."
                )
        self._wt = None
        if delete_warning:
            return f"Partially discarded branch '{wt.branch}'.{checkout_warning}{delete_warning}"
        return f"Discarded branch '{wt.branch}'.{checkout_warning}"


_INTERACTIVE_ONLY_FLAGS: frozenset[str] = frozenset({
    "--worktree", "--no-worktree",
    "--auto-commit", "--no-auto-commit",
})


def _reject_interactive_only_flags(argv: list[str]) -> None:
    """Fail fast when a non-interactive run carries interactive-only flags.

    The non-interactive (``-t`` / ``-f``) path now constructs a bare
    :class:`SorcarAgent` and therefore cannot honour
    ``--worktree`` / ``--no-worktree`` / ``--auto-commit`` /
    ``--no-auto-commit``.  Silently accepting them would, in the
    case of ``--worktree`` (the previous default), let edits land
    in the user's working tree instead of the isolated worktree
    branch the flag advertised — a destructive surprise.  This
    helper inspects the user's literal ``argv`` and exits via
    ``sys.exit(2)`` (the argparse convention) with a message
    naming every offending flag.

    Argparse prefix abbreviations (e.g. ``--auto`` for
    ``--auto-commit``) cannot bypass this guard because
    :func:`_build_arg_parser` disables ``allow_abbrev``; the user
    must spell the full flag, and the full spelling is in this set.

    Args:
        argv: The process argument list (typically ``sys.argv``).
    """
    bad = list(dict.fromkeys(
        token for token in argv[1:] if token in _INTERACTIVE_ONLY_FLAGS
    ))
    if not bad:
        return
    flag_list = ", ".join(bad)
    msg = (
        f"sorcar: error: {flag_list} cannot be combined with -t/--task "
        "or -f/--file (non-interactive mode runs a bare SorcarAgent; "
        "drop the flag, or run sorcar without -t/-f for the "
        "interactive daemon-client mode which honours it)"
    )
    print(msg, file=sys.stderr)
    sys.exit(2)
