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
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.cli_helpers import (
    _build_arg_parser,
    _build_run_kwargs,
    _launch_work_dir,
    print_outcome,
)
from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    MergeResult,
    repo_lock,
)
from kiss.agents.sorcar.persistence import _allocate_chat_id

# ``_generate_commit_message`` is re-exported (and looked up from this
# module's globals at call time) so tests can monkeypatch
# ``worktree_sorcar_agent._generate_commit_message``.
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
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


# Result-specific middle lines for the manual-resolution command block
# (see :func:`_merge_fix_steps`).
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
        return f"git cherry-pick --no-commit {wt.baseline_commit}..{wt.branch}"
    return f"git merge --squash {wt.branch}"


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
        f"    cd {wt.repo_root}\n"
        f"    git checkout {wt.original_branch}\n"
        f"    {_manual_merge_cmd(wt)}\n"
        + fix_lines
        + f"    git branch -D {wt.branch}"
    )


class WorktreeSorcarAgent(ChatSorcarAgent):
    """SorcarAgent that isolates every task in a git worktree.

    Each ``run()`` call creates a brand-new worktree on a fresh branch.
    Worktrees are not associated with the agent's ``chat_id``: branch
    names use a unique time + random suffix, and there is no
    cross-process state restoration based on chat session.  Any
    previous worktree owned by this agent instance is auto-merged
    (or kept on conflict) before the new one is created.

    Attributes:
        _wt: The current/pending worktree state, or ``None`` when idle.
    """

    uses_worktree = True

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._wt: GitWorktree | None = None
        self._stash_pop_warning: str | None = None
        self._merge_conflict_warning: str | None = None
        # Guards the two warning attributes above.  The git operations
        # themselves are serialized under ``repo_lock``, but the
        # warning attributes are written by ``_release_worktree``
        # (invoked from ``run()`` / ``new_chat()`` / server teardown
        # threads) and read-then-cleared by ``_flush_warnings`` on the
        # task-runner thread.  Without the lock, two concurrent
        # flushes could both pass the truthiness check and broadcast
        # the same warning twice, and a warning set between a flush's
        # check and its ``= None`` clear would be silently dropped.
        self._warning_lock: threading.Lock = threading.Lock()
        # When False, ``_auto_commit_worktree`` is a no-op so the
        # worktree's uncommitted changes are preserved for manual
        # review (and ``_finalize_worktree`` returns False, keeping
        # the worktree directory in place).  Mirrors the sorcar CLI
        # ``--auto-commit`` / ``--no-auto-commit`` flag, which
        # defaults to True.
        self.auto_commit_enabled: bool = True
        # Frontend tab id, stamped by
        # :meth:`_TaskRunnerMixin._run_task_inner` immediately after
        # constructing the agent.  Consumed by
        # :meth:`SorcarAgent._drain_pending_user_messages` to look up
        # the owning :class:`_RunningAgentState` and pull queued
        # follow-up prompts before each model call.
        self._tab_id: str = ""
        # Wall-clock start of the current task in epoch milliseconds,
        # stamped by the VS Code ``task_runner`` just before the run
        # starts; read (via ``getattr`` with a 0 default) by
        # ``server._live_task_start_ms`` for resume timelines.
        self._task_start_ms: int = 0
        # "Lost slides" bug fix: when a worktree task ends in
        # failure / user-Stop and the partial work is left for
        # review (e.g. the merge view is opened, or the changes
        # are binary-only and even the merge view cannot start),
        # the VS Code ``task_runner`` sets this flag.  At tab
        # teardown (:meth:`VSCodeServer._teardown_tab_resources`)
        # the flag steers the worktree into
        # :meth:`_preserve_pending_worktree_for_review` instead of
        # :meth:`_release_worktree`, so the partial work is
        # committed onto the ``kiss/wt-*`` branch but NOT silently
        # squash-merged into the user's original branch — closing
        # the chat tab can no longer overwrite the main branch
        # with an incomplete, unverified deck.  Cleared whenever
        # the user explicitly merges or discards — both via
        # :meth:`_MergeFlowMixin._handle_worktree_action` and
        # directly in :meth:`merge` / :meth:`discard` — or when
        # the agent boots a fresh worktree via
        # :meth:`_try_setup_worktree` / :meth:`new_chat`.
        self._pending_review: bool = False


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
        # Mint a single notification id for the whole
        # ``auto_commit_changes`` lifecycle so both the "generating"
        # and "committed" toasts share it (Bug 2 from gpt-5.5
        # review).  Includes ``time.time_ns()`` so concurrent
        # auto-commits (e.g. sub-agent sessions on the same tab)
        # don't collide.  The id is a LOCAL bound via
        # ``functools.partial`` — never instance state — so a
        # concurrent ``_auto_commit_worktree`` on the same agent
        # (e.g. teardown-preserve racing an explicit merge) can never
        # overwrite this call's id and pair its "committed" toast
        # with the other call's "generating" toast.
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
        if stage == "generating":
            message = "Generating commit message"
        elif stage == "committed":
            message = f"Committed {subject}" if subject else "Committed"
        else:
            return
        # The ``"generating"`` toast must remain visible for the
        # entire (potentially long) LLM-driven commit-message call —
        # the webview's transient auto-dismiss timer (~5 s) would
        # otherwise hide it mid-flight and mislead the user into
        # thinking the commit had stalled.  Mark it ``sticky`` so
        # ``scheduleNotificationDismiss`` short-circuits, and rely on
        # the subsequent ``"committed"`` event (which reuses the same
        # ``id`` but omits ``sticky``) to replace it with a regular
        # transient toast that fades out normally.
        event: dict[str, object] = {
            "type": "notification",
            "id": notification_id,
            "severity": "info",
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
                # ``auto_commit_enabled=False`` contract (see the
                # attribute docstring in ``__init__``): never
                # force-commit the user's reviewable changes under
                # ``--no-auto-commit`` (``_auto_commit_worktree``
                # no-ops when the flag is off, so EVERYTHING is still
                # uncommitted here).
                if not self.auto_commit_enabled:
                    return _WorktreeCleanupOutcome.PRESERVED_NO_AUTOCOMMIT, ""
                # Single-shot retry: closes the residual race window
                # between ``auto_commit_changes``'s second
                # ``stage_all`` and its ``commit_staged`` call.
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

        After the LLM-driven auto-commit, a single-shot retry runs
        :meth:`GitWorktreeOps.commit_all` with a generic message to
        catch the very narrow remaining race where a file appears
        between :func:`~kiss.agents.sorcar.sorcar_agent.auto_commit_changes`'s
        second ``stage_all`` and its ``commit_staged`` call (e.g.
        ``PROGRESS.md`` being rewritten, ``.DS_Store`` materializing
        after an ``open`` of the report, an editor swap file
        appearing).  Only if that retry STILL leaves uncommitted
        state do we preserve the worktree and log a warning — and
        that warning now includes the raw ``git status --porcelain``
        leftover so an operator can distinguish a real pre-commit
        rejection from a race leftover from a corrupt index without
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
    ) -> tuple[MergeResult, str]:
        """Stash, checkout, squash-merge, pop for a worktree branch.

        Serialized under ``repo_lock`` to prevent concurrent tabs from
        interleaving operations on the main repository.

        Args:
            wt: The worktree state to merge.

        Returns:
            ``(result, stash_warning)`` where *result* is the merge
            outcome and *stash_warning* is a non-empty string if
            stash-pop failed.  Checkout failures return
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
        if wt.original_branch is None:
            return (MergeResult.CHECKOUT_FAILED, "")
        with repo_lock(wt.repo_root):
            # Re-ensure the PROGRESS.md merge driver right before the
            # merge so pending worktrees created by older agent
            # versions (or repos whose local config was wiped) still
            # auto-resolve scratch-file conflicts.
            try:
                GitWorktreeOps.ensure_scratch_merge_driver(wt.repo_root)
            except Exception:  # pragma: no cover — filesystem permission error
                logger.warning(
                    "Failed to install scratch merge driver", exc_info=True
                )
            # Stash BEFORE the checkout: dirty user edits on a
            # different branch would otherwise make the checkout fail
            # ("local changes would be overwritten") even though
            # merge()'s contract promises they are stashed first.
            did_stash = GitWorktreeOps.stash_if_dirty(wt.repo_root)
            if not did_stash and GitWorktreeOps.has_uncommitted_changes(
                wt.repo_root
            ):
                # The tree is dirty but ``git stash push`` FAILED
                # (returncode != 0) — abort before any mutation.
                return (MergeResult.STASH_FAILED, "")
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
                    return (MergeResult.CHECKOUT_FAILED, stash_warning)

            # Thread the recorded task prompt/result into the merge
            # commit message so the final commit on the user's
            # original branch ALWAYS records them — even when the
            # agent hand-committed its own work in the worktree (so
            # the post-task auto-commit was a no-op and the branch
            # HEAD message carries neither block).  See
            # ``GitWorktreeOps._merge_commit_message`` for the dedup
            # contract (production incident: commit dd563a7c).
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
            if not self.auto_commit_enabled:
                # ``--no-auto-commit`` contract: the worktree has
                # uncommitted changes and auto-commit is disabled by
                # user choice — not a pre-commit hook failure.
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

        result, stash_warning = self._do_merge(wt)
        if stash_warning:
            self._set_warnings(stash=stash_warning)

        # From here on the agent no longer tracks the worktree: on
        # success it is fully merged; on failure the branch is kept in
        # git for manual resolution (described by the warning below).
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
        else:  # MergeResult.CONFLICT
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
          prune`` so disk state is clean.
        * Does **not** call :meth:`_do_merge`: the partial work is
          NOT squash-merged into the user's original branch.  Closing
          the chat tab (or the WebSocket all-done close path) can
          therefore never silently overwrite the user's main branch
          with incomplete, unverified work — the user must explicitly
          recover the branch with ``git checkout <branch>``.

        Idempotent / safe: no-op when no worktree is pending.

        Returns:
            True when a pending worktree was preserved (or had no
            uncommitted work to preserve and was just cleaned up).
            False when there was nothing to do.
        """
        if self._wt is None:
            return False
        wt = self._wt
        # Capture any uncommitted partial work as a real commit so it
        # survives the worktree directory removal — the same
        # auto-commit → late-arriver-retry → preserve-or-remove engine
        # ``_finalize_worktree`` uses, for the same reasons
        # (PROGRESS.md being rewritten, pre-commit hook rejections,
        # the ``--no-auto-commit`` contract, etc.).
        outcome, leftover = self._commit_and_clean_worktree(wt)
        if outcome is _WorktreeCleanupOutcome.PRESERVED_NO_AUTOCOMMIT:
            # Preserve the worktree directory intact — the user can
            # review/commit manually in ``wt_dir``.
            logger.warning(
                "Auto-commit disabled (--no-auto-commit); "
                "preserving worktree '%s' with uncommitted "
                "changes for manual review at %s",
                wt.branch, wt.wt_dir,
            )
        elif outcome is _WorktreeCleanupOutcome.PRESERVED_COMMIT_FAILED:
            # A pre-commit hook rejected the commit, or a real commit
            # failure.  The worktree dir is preserved so no work is
            # lost; the user can finish the commit manually with
            # ``cd <wt_dir> && git commit``.
            logger.warning(
                "Worktree '%s' has uncommitted changes after "
                "preserve-for-review (likely a pre-commit hook "
                "rejection); preserving worktree directory %s\n"
                "git status --porcelain:\n%s",
                wt.branch, wt.wt_dir, leftover,
            )
        # Drop the in-memory worktree reference (the branch lives on
        # in git and is recoverable manually) and clear the
        # pending-review flag — future operations on this agent
        # instance should not inherit the stopped-task state.
        self._wt = None
        self._pending_review = False
        return True


    def new_chat(self) -> None:
        """Reset to a new chat session, auto-merging any pending worktree.

        If a worktree task is pending from the previous session, it is
        auto-committed with a detailed LLM message and squash-merged
        into the original branch before the chat state is reset.

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
        self._release_worktree()
        self._flush_warnings(getattr(self, "printer", None))
        # ``_release_worktree`` already cleared ``self._wt``; defensively
        # clear the pending-review flag as well so a brand-new chat
        # session never inherits stop-state from the previous task.
        self._pending_review = False
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
        # RACE-2 fix: hold ``repo_lock`` for the entire release +
        # worktree-create + copy-dirty-state + baseline-commit
        # sequence.  Previously only the inner ``_release_worktree``
        # / ``current_branch`` call held the lock; concurrent
        # ``_handle_worktree_action`` or non-wt task-start handlers
        # could interleave with ``copy_dirty_state`` and
        # ``baseline`` commit, snapshotting an inconsistent main
        # tree.  The lock is re-entrant
        # (:func:`git_worktree.repo_lock` is an ``RLock``) so the
        # ``_do_merge`` invoked by ``_release_worktree`` re-acquires
        # it cleanly on the same thread.
        with repo_lock(repo):
            # The released worktree may live in a DIFFERENT repo (the
            # user changed ``work_dir`` between runs).  Its original
            # branch name must then not leak into *repo*: a same-named
            # branch there would silently become the merge target
            # (wrong merge result) and ``merge()`` would switch the
            # user's checkout to it.
            prev_repo_root = self._wt.repo_root if self._wt is not None else None
            released_branch = self._release_worktree()
            # A brand-new task must never inherit the pending-review
            # state of the previous worktree (e.g. a stopped task
            # followed by an explicit new task on the same agent
            # instance) — otherwise the next ``_teardown_tab_resources``
            # would preserve THIS task's worktree branch even when the
            # new task completes cleanly.
            self._pending_review = False

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
                # E.g. a dirty file the process cannot read (mode 000)
                # makes shutil.copy2 raise PermissionError.  Honor
                # run()'s fallback contract: clean up the half-created
                # worktree/branch and run the task directly instead of
                # crashing the whole task.
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
                    # The baseline commit FAILED (e.g. a
                    # prepare-commit-msg hook — which --no-verify does
                    # NOT skip — or a stale index.lock): the user's
                    # dirty state would sit uncommitted in the
                    # worktree and later be auto-committed as (and
                    # attributed to) agent work, then squash-merged
                    # back — duplicating the user's edits.  Honor
                    # run()'s fallback contract instead: clean up and
                    # run the task directly.
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
            printer.broadcast({"type": "warning", "message": stash_warning})
        if merge_warning:
            printer.broadcast({"type": "warning", "message": merge_warning})


    def run(  # type: ignore[override]
        self,
        prompt_template: str = "",
        **kwargs: Any,
    ) -> str:
        """Run a task on an isolated git worktree branch.

        Creates a new worktree and branch, redirects ``work_dir`` into
        the worktree, and delegates to ``ChatSorcarAgent.run()``.
        Each call starts a fresh worktree; any previously pending
        branch from an earlier run is auto-committed and squash-merged
        into its original branch first (kept in git for manual
        resolution only when that auto-merge fails or conflicts).

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
        # Establish ``chat_id`` BEFORE registering so the entry is
        # keyed by the canonical session identifier.  Identical to
        # the standalone :meth:`ChatSorcarAgent.run` minting (both
        # use a fresh UUID hex via :func:`_allocate_chat_id`).
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
                # Fall back to direct execution (no worktree).  Use the
                # SAME exception contract as the worktree path below:
                # non-``KISSError`` exceptions become a YAML
                # ``success: false`` result rather than propagating, so
                # callers see one failure surface regardless of repo
                # state (git repo vs not, detached HEAD, setup failure).
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
        # The user has explicitly chosen to merge, so the stopped-task
        # ``_pending_review`` state no longer applies (mirrors
        # ``_MergeFlowMixin._handle_worktree_action``, which clears the
        # flag before dispatching; direct API callers must get the
        # same contract).
        self._pending_review = False

        if wt.original_branch is None:
            merge_cmd = _manual_merge_cmd(wt)
            return (
                "Cannot merge: original branch is unknown (likely due to a "
                "crash during setup).  Please specify the target branch "
                "manually:\n"
                f"    git checkout <branch> && {merge_cmd}"
            )

        # Always finalize, even when the worktree directory is already
        # gone: ``_finalize_worktree`` runs ``git worktree prune``,
        # without which a stale registration (deleted dir, bookkeeping
        # kept) makes ``git branch -d/-D`` refuse to delete the task
        # branch after a successful merge.
        if not self._finalize_worktree():
            if not self.auto_commit_enabled:
                # ``--no-auto-commit`` contract: ``_finalize_worktree``
                # deliberately returned False because the worktree has
                # uncommitted changes and auto-commit is disabled —
                # do NOT blame a pre-commit hook for a mode the user
                # explicitly chose.
                return (
                    f"Cannot merge: auto-commit is disabled "
                    f"(--no-auto-commit) and the worktree for "
                    f"'{wt.branch}' has uncommitted changes. "
                    f"The worktree is preserved at: {wt.wt_dir}\n\n"
                    "Review and commit the changes manually:\n"
                    f"    cd {wt.wt_dir}\n"
                    "    git add -A && git commit -m 'agent work'\n\n"
                    "Then retry: agent.merge()"
                )
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
        stash_suffix = ""
        if stash_warning:
            stash_suffix = "\n\n⚠️  " + stash_warning

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
        # Explicit discard clears the stopped-task review state, same
        # contract as ``merge()`` above.
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
                    f"'git branch -D {wt.branch}' to remove it."
                )
        self._wt = None
        if delete_warning:
            return f"Partially discarded branch '{wt.branch}'.{checkout_warning}{delete_warning}"
        return f"Discarded branch '{wt.branch}'.{checkout_warning}"


# Flags that only make sense in interactive (daemon-client) mode:
# they configure features the bare ``SorcarAgent`` used by the
# non-interactive path does not implement.  Listed as the literal CLI
# tokens (matched against ``sys.argv`` so the user's exact spelling
# is reflected in the error).  ``main()`` consults this set after
# parsing to fail fast when one of these flags is combined with
# ``-t`` / ``-f``.  Argparse prefix abbreviations cannot evade this
# table because :func:`_build_arg_parser` sets ``allow_abbrev=False``.
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
    # dict.fromkeys = order-preserving dedup of the offending flags.
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


def main() -> None:
    """Run the ``sorcar`` CLI.

    Two modes:

    * **Interactive** (no ``-t/--task`` / ``-f/--file``): a thin
      terminal client of the local ``sorcar web`` daemon — see
      :mod:`kiss.agents.sorcar.cli_client`.  The ``--worktree`` /
      ``--no-worktree`` / ``--parallel`` / ``--no-parallel`` /
      ``--auto-commit`` / ``--no-auto-commit`` flags are forwarded
      to the daemon so each task can still run on an isolated git
      worktree, with parallel sub-agents and auto-commit.
      Chat-session control (new chat, resume) is driven from the
      interactive client's slash commands rather than CLI flags.
    * **Non-interactive** (``-t`` or ``-f`` supplied): runs a plain
      :class:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent` once on
      the supplied task and exits.  No git worktree isolation, no
      chat-session control — those features were always tied to the
      removed ``-c/--chat-id`` / ``-l/--list-chat-id`` /
      ``--cleanup`` / ``--use-chat`` / ``--use-worktree`` flag set.
      ``--worktree`` / ``--no-worktree`` / ``--auto-commit`` /
      ``--no-auto-commit`` are interactive-only and are rejected
      when combined with ``-t`` / ``-f`` (see
      :func:`_reject_interactive_only_flags`).  Display events from
      the run are still streamed into the local chat DB via
      :class:`RecordingConsolePrinter` so the run is replayable in
      the chat webview; only the *chat session* surface (resume by
      id) is unavailable.

    ``sorcar mcp ...`` is dispatched to the MCP management subcommand
    (:mod:`kiss.agents.sorcar.mcp_cli`) before normal argument parsing.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        from kiss.agents.sorcar.mcp_cli import run_mcp_cli

        sys.exit(run_mcp_cli(sys.argv[2:], str(Path.cwd())))

    parser = _build_arg_parser()
    args = parser.parse_args()
    work_dir = args.work_dir or _launch_work_dir()

    interactive = args.task is None and args.file is None
    if not interactive:
        # Validate AFTER argparse (so ``-t``/``-f`` are decoded
        # correctly) but BEFORE the agent is built / the LLM is
        # contacted (so the user sees the error immediately and no
        # budget is spent).
        _reject_interactive_only_flags(sys.argv)
    run_kwargs = _build_run_kwargs(args)

    if interactive:
        from kiss.agents.sorcar.cli_client import run_client

        sys.exit(
            run_client(
                work_dir=run_kwargs.get("work_dir") or work_dir,
                model_name=run_kwargs.get("model_name", ""),
                active_file=run_kwargs.get("current_editor_file") or "",
                use_worktree=bool(getattr(args, "worktree", True)),
                use_parallel=bool(getattr(args, "parallel", True)),
                # Fallback default matches the argparse default
                # (``--auto-commit`` default=True in cli_helpers) and
                # the sibling ``worktree``/``parallel`` fallbacks.
                auto_commit=bool(getattr(args, "auto_commit", True)),
            ),
        )

    # Non-interactive: plain SorcarAgent, no chat / no worktree.
    from kiss.agents.sorcar.cli_steering import run_with_steering

    agent: SorcarAgent = SorcarAgent("Sorcar Agent")
    start_time = time.time()
    result = run_with_steering(agent, run_kwargs)
    elapsed = time.time() - start_time
    print_outcome(agent, result, elapsed, run_kwargs.get("verbose", True))


if __name__ == "__main__":
    main()
