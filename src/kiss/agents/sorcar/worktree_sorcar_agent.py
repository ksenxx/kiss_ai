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
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar import worktree_pool
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import (
    _WORKTREE_SUBDIR,
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
    PRESERVED_SUBAGENT_ACTIVE = "preserved_subagent_active"
    PRESERVED_RESCUE_FAILED = "preserved_rescue_failed"


_PRECOMMIT_FIX_LINES = (
    "    # fix pre-commit issues, then:\n"
    "    git commit --no-verify\n"
)

# How long a worktree cleanup waits for an abandoned sub-agent thread
# to finish before preserving the directory instead of deleting it.
# Short: the user is watching, and preserving is the safe outcome —
# the next cleanup (or the reclaim pass in a later process) removes it
# once the thread is really gone.
_ABANDONED_SUBAGENT_WAIT_SECONDS = 5.0


def _config_auto_commit_enabled() -> bool:
    """Return the user's persisted "Auto commit" setting.

    The toggle lives in ``~/.kiss/config.json`` as ``auto_commit_mode``
    (see :data:`kiss.core.vscode_config.DEFAULTS`) and reaches a
    server-run agent through ``AgentState`` / the task runner.  Reading
    it here as the DEFAULT means the setting is honoured even for
    callers that construct the agent directly.

    Returns:
        The stored value, defaulting to ``True`` (auto-commit on) when
        the config file is missing or unreadable.
    """
    from kiss.core.vscode_config import load_config

    try:
        return bool(load_config().get("auto_commit_mode", True))
    except Exception:  # pragma: no cover — unreadable config
        logger.debug("Could not read auto_commit_mode", exc_info=True)
        return True


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
        # The user's "Auto commit" setting.  Re-resolved on every
        # ``run`` so a mid-session toggle takes effect, and overridable
        # per run via the ``auto_commit`` kwarg.
        self.auto_commit_enabled: bool = _config_auto_commit_enabled()
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
    def _wt_work_dir(self) -> Path | None:
        """The task's working directory inside the worktree.

        ``wt_dir`` plus the original work dir's offset from the repo
        root; differs from ``wt_dir`` when the task was launched from a
        subdirectory of the repo.  Falls back to ``wt_dir`` for legacy
        snapshots without the field.
        """
        if self._wt is None:
            return None
        return self._wt.work_dir or self._wt.wt_dir

    @property
    def _baseline_commit(self) -> str | None:
        """SHA of the baseline commit (user's dirty state), or ``None``."""
        return self._wt.baseline_commit if self._wt else None


    def _auto_commit_worktree(self, force_commit: bool = False) -> bool:
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
        <subject>"`` once the commit lands in git.  Each
        notification is fanned out to every tab watching this task
        by the printer's transient broadcast primitive
        (``JsonPrinter.broadcast_transient``), while unrelated tabs
        are left alone.

        Args:
            force_commit: Commit even when the user turned Auto-commit
                off.  Set only on paths the user explicitly asked for
                (``merge()``); the automatic paths must obey the
                setting, which is the whole point of the toggle.

        Returns:
            True if a commit was created, False if nothing to commit.
        """
        if self._wt is None or not self._wt.wt_dir.exists():
            return False
        if not (self.auto_commit_enabled or force_commit):
            return False
        commit_run_id = f"autocommit-{time.time_ns()}"
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
        }
        if stage == "generating":
            event["sticky"] = True
        try:
            self._broadcast_to_watchers(printer, event)
        except Exception:  # pragma: no cover — best-effort UI hook
            logger.debug(
                "autocommit notification broadcast failed", exc_info=True,
            )

    def _broadcast_to_watchers(
        self, printer: Any, event: dict[str, object],
    ) -> None:
        """Broadcast *event* so it reaches this agent's tab.

        An event with neither ``tabId`` nor ``taskId`` is addressed to
        nobody: the printer's task fan-out has no targets and every
        client filters it out.  That is fatal for events emitted
        OUTSIDE a run — a worktree warning flushed before
        ``super().run()`` has allocated the task id, an auto-commit
        toast raised by a server-side cleanup — which is exactly when
        this agent's own tab id is the only routing information that
        exists.

        The printer's ``broadcast_transient`` primitive resolves every
        tab currently watching the task and stamps one copy per tab,
        falling back to this agent's tab; printers without it (plain
        ``broadcast``-only stubs) get one copy stamped with the tab id
        directly, which preserves the same transient, never-recorded
        semantics.

        Args:
            printer: A ``broadcast``-capable printer.
            event: The event to route; must not carry ``tabId``.
        """
        transient = getattr(printer, "broadcast_transient", None)
        if callable(transient):
            transient(
                event,
                task_id=self.last_task_id or None,
                tab_id=self._tab_id,
            )
            return
        printer.broadcast({**event, "tabId": self._tab_id})


    def _commit_and_clean_worktree(
        self, wt: GitWorktree, force_commit: bool = False
    ) -> tuple[_WorktreeCleanupOutcome, str]:
        """Auto-commit *wt*'s changes, then remove the worktree and prune.

        Shared engine of :meth:`_finalize_worktree` and
        :meth:`_preserve_pending_worktree_for_review` — the exact
        reclaim → auto-commit → late-arriver-retry → preserve-or-remove
        sequence previously duplicated in both.

        The reclaim comes first: an abandoned sub-agent thread is
        still writing into this worktree, and whatever it produces
        before it finishes has to be visible to the staging passes
        below.  Waiting after them would let a child that finished
        during the wait have its last files deleted with the
        directory.

        After the LLM-driven auto-commit, a single-shot retry runs
        :meth:`GitWorktreeOps.commit_all` with a generic message to
        catch the very narrow remaining race where a file appears
        between :func:`~kiss.agents.sorcar.sorcar_agent.auto_commit_changes`'s
        second ``stage_all`` and its ``commit_staged`` call (e.g.
        ``PROGRESS.md`` being rewritten, ``.DS_Store`` materializing
        after an ``open`` of the report, an editor swap file
        appearing).  ``commit_all`` is a no-op when nothing is
        uncommitted, but skipping the call keeps the happy-path log
        quiet.  With Auto-commit off the retry never
        force-commits: the worktree is preserved for manual review.

        Args:
            wt: The worktree to commit and clean up.
            force_commit: Commit even when Auto-commit is off — used
                only by :meth:`merge`, where committing is exactly what
                the user asked for.

        Returns:
            A ``(outcome, leftover)`` pair.  ``leftover`` is the raw
            ``git status --porcelain`` output when the outcome is
            :attr:`_WorktreeCleanupOutcome.PRESERVED_COMMIT_FAILED`
            (both auto-commit and the retry left uncommitted state —
            e.g. a pre-commit hook rejection), otherwise ``""``.  The
            worktree directory is removed and pruned only on
            :attr:`_WorktreeCleanupOutcome.COMMITTED_AND_REMOVED`; the
            preserved outcomes leave it in place so no work is lost —
            including
            :attr:`_WorktreeCleanupOutcome.PRESERVED_SUBAGENT_ACTIVE`,
            returned when an abandoned sub-agent thread is still
            writing into this worktree.
        """
        if wt.wt_dir.exists():
            # A sub-agent thread this agent abandoned still has its
            # work_dir set to this worktree.  Removing the directory
            # under a live writer loses whatever it produces next and
            # can make `git worktree remove` itself fail.  Waiting
            # BEFORE the commit passes below is what makes the wait
            # worth anything: a child that finishes during it writes
            # its last files first, so they are staged and committed
            # like every other change instead of being deleted with
            # the directory moments later.
            if not self.reclaim_abandoned_subagents(
                timeout=_ABANDONED_SUBAGENT_WAIT_SECONDS,
            ):
                return _WorktreeCleanupOutcome.PRESERVED_SUBAGENT_ACTIVE, ""
            self._auto_commit_worktree(force_commit=force_commit)
            if GitWorktreeOps.has_uncommitted_changes(wt.wt_dir):
                if not (self.auto_commit_enabled or force_commit):
                    return _WorktreeCleanupOutcome.PRESERVED_NO_AUTOCOMMIT, ""
                GitWorktreeOps.commit_all(
                    wt.wt_dir,
                    "kiss: auto-commit late-arriving changes",
                )
            if GitWorktreeOps.has_uncommitted_changes(wt.wt_dir):
                leftover = GitWorktreeOps.status_porcelain(wt.wt_dir)
                return _WorktreeCleanupOutcome.PRESERVED_COMMIT_FAILED, leftover
            # Auto-commit cannot capture git-ignored task output
            # (``git add -A`` skips it); without this rescue the
            # removal below would silently destroy files the same
            # task would have left on disk in non-worktree mode.  The
            # rescue fails closed: when a file could not be landed in
            # the main repo, the worktree is preserved — removing it
            # would destroy the only copy.
            try:
                _, rescue_ok = GitWorktreeOps.rescue_ignored_files(
                    wt.wt_dir, wt.repo_root,
                )
            except Exception:  # pragma: no cover — filesystem failure
                logger.warning(
                    "Ignored-file rescue failed for %s",
                    wt.wt_dir, exc_info=True,
                )
                rescue_ok = False
            if not rescue_ok:
                return _WorktreeCleanupOutcome.PRESERVED_RESCUE_FAILED, ""
        # No separate ``prune`` is issued: :meth:`GitWorktreeOps.remove`
        # prunes on every path that can leave a stale registration —
        # including the one this call covers when the directory has
        # already vanished from disk — and a successful ``git worktree
        # remove`` unregisters the worktree itself.
        GitWorktreeOps.remove(wt.repo_root, wt.wt_dir)
        return _WorktreeCleanupOutcome.COMMITTED_AND_REMOVED, ""

    def _finalize_worktree(self, force_commit: bool = False) -> bool:
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

        Args:
            force_commit: Commit even when Auto-commit is off — set by
                :meth:`merge`, the path where the user explicitly asked
                for the commit.

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
        outcome, leftover = self._commit_and_clean_worktree(
            wt, force_commit=force_commit,
        )
        if outcome is _WorktreeCleanupOutcome.PRESERVED_NO_AUTOCOMMIT:
            return False
        if outcome is _WorktreeCleanupOutcome.PRESERVED_SUBAGENT_ACTIVE:
            logger.warning(
                "A sub-agent thread is still running inside worktree "
                "'%s'; preserving %s rather than deleting a directory "
                "that is being written to",
                wt.branch,
                wt.wt_dir,
            )
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
        if outcome is _WorktreeCleanupOutcome.PRESERVED_RESCUE_FAILED:
            logger.warning(
                "Git-ignored task output in worktree '%s' could not "
                "be rescued into the main repository; preserving %s "
                "so the only copy is not destroyed",
                wt.branch,
                wt.wt_dir,
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
                    f"Auto-commit is turned off and "
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
          — Auto-commit turned off, or a hook refusing it — the
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
            _WorktreeCleanupOutcome.PRESERVED_SUBAGENT_ACTIVE,
            _WorktreeCleanupOutcome.PRESERVED_RESCUE_FAILED,
        ):
            # Persist the "preserve for manual review" decision so a
            # future :meth:`GitWorktreeOps.reclaim_orphaned_worktrees`
            # (running in a fresh process after the original agent
            # died) does not silently publish this deliberately parked
            # work.
            GitWorktreeOps.save_preserve_marker(wt.repo_root, wt.branch)
        if outcome is _WorktreeCleanupOutcome.PRESERVED_NO_AUTOCOMMIT:
            logger.warning(
                "Auto-commit turned off; "
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
        elif outcome is _WorktreeCleanupOutcome.PRESERVED_SUBAGENT_ACTIVE:
            logger.warning(
                "A sub-agent thread is still running inside worktree "
                "'%s'; preserving %s",
                wt.branch, wt.wt_dir,
            )
            self._set_warnings(merge=(
                f"A sub-agent of branch '{wt.branch}' is still running "
                f"and writing into {wt.wt_dir}, so the worktree was kept "
                "instead of being deleted underneath it.  Merge or "
                "discard the branch once the sub-agent has stopped."
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
        elif outcome is _WorktreeCleanupOutcome.PRESERVED_RESCUE_FAILED:
            logger.warning(
                "Git-ignored output of worktree '%s' could not be "
                "rescued; preserving worktree directory %s",
                wt.branch, wt.wt_dir,
            )
            self._set_warnings(merge=(
                f"Git-ignored files created by the task in branch "
                f"'{wt.branch}' could not be copied into the main "
                f"repository, so its worktree directory {wt.wt_dir} was "
                f"kept — it holds the only copy of those files. Recover "
                f"them there."
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
        stash failure, Auto-commit off with uncommitted changes),
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
        other live agent's branch as tracked by the server's
        task-keyed agent-state registry, reached through the printer's
        duck-typed ``live_worktree_branches`` bridge.  Used by
        :meth:`_try_setup_worktree` to build the ``exclude_branches``
        argument for
        :meth:`GitWorktreeOps.reclaim_orphaned_worktrees` so a
        concurrent live agent's active worktree is never adopted,
        merged, or removed by our reclaim pass.
        """
        branches: set[str] = set()
        if self._wt is not None:
            branches.add(self._wt.branch)
        live = getattr(
            getattr(self, "printer", None), "live_worktree_branches", None,
        )
        if live is not None:
            branches.update(live())
        # Pooled spare worktrees are owned by the process too: a
        # reclaim pass that adopted one would merge-and-delete the
        # worktree a concurrent task start is about to consume.
        branches.update(worktree_pool.spare_branches())
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

    def _acquire_task_worktree(
        self,
        repo: Path,
        original_branch: str,
    ) -> tuple[str, Path] | None:
        """Obtain a checked-out ``kiss/wt-*`` worktree for a new task.

        Fast path: consume the spare worktree that
        :mod:`kiss.agents.sorcar.worktree_pool` pre-created on a
        background thread and hard-reset it onto *original_branch*'s
        tip — skipping the full-checkout ``git worktree add`` that
        dominates the delay between task submission and agent start.

        Slow path (empty pool, or a spare that fails validation or the
        reset): run the orphan-maintenance passes and create the
        worktree inline, exactly as before the pool existed.  Reclaim
        before sweep: reclaim may merge and delete orphan branches,
        and sweep purges leftover config sections whose branches were
        just removed.  Excluded from reclaim: every branch owned by a
        *live* agent in this process — self and every other tab — and
        every pooled spare, so neither a running sibling task's
        worktree nor the pool's spare is ever adopted or destroyed.

        Either way, a pool refill for the NEXT task is scheduled on a
        background thread before returning.  Callers must hold
        ``repo_lock(repo)``.

        Args:
            repo: Git repo root path.
            original_branch: The branch the task will merge back into;
                the acquired worktree's branch starts at its tip.

        Returns:
            ``(branch, wt_dir)`` on success, or ``None`` when no
            worktree could be created (caller falls back to direct
            execution).
        """
        acquired: tuple[str, Path] | None = None
        spare = worktree_pool.take_spare(repo)
        if spare is not None:
            spare_branch, spare_dir = spare
            # Three consume steps: move the spare's branch onto the
            # task's base commit; drop untracked files an external
            # writer may have left in the idle directory (reset keeps
            # them and auto-commit would publish them); clear the
            # spare marker so a crash mid-task routes this worktree
            # through the normal orphan reclaim instead of the
            # spare-discard path.
            if (
                GitWorktreeOps.reset_worktree_to(spare_dir, original_branch)
                and GitWorktreeOps.clean_untracked(spare_dir)
                and GitWorktreeOps.clear_spare_marker(repo, spare_branch)
            ):
                acquired = spare
            else:
                GitWorktreeOps.cleanup_partial(repo, spare_branch, spare_dir)
        if acquired is None:
            try:
                GitWorktreeOps.reclaim_orphaned_worktrees(
                    repo,
                    exclude_branches=self._live_worktree_branches(),
                )
                GitWorktreeOps.sweep_orphaned_state(repo)
            except Exception:  # pragma: no cover — unexpected git failure
                logger.warning(
                    "Orphan-worktree maintenance failed", exc_info=True,
                )
            branch = worktree_pool.new_task_branch(repo)
            wt_dir = repo / _WORKTREE_SUBDIR / branch.replace("/", "_")
            if not GitWorktreeOps.create(repo, branch, wt_dir):
                # pragma: no cover — git worktree add failure
                GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
                return None
            acquired = (branch, wt_dir)
        # Refill the pool for the next task while this one runs.  The
        # exclusion callable is evaluated by the refill thread right
        # before its reclaim pass (under ``repo_lock``), so it sees the
        # live-agent set as it stands THEN — including this task's own
        # branch once ``self._wt`` is assigned.
        worktree_pool.prewarm_async(repo, self._live_worktree_branches)
        return acquired

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
            except Exception:  # pragma: no cover — filesystem permission error
                logger.warning("Failed to update git exclude", exc_info=True)

            acquired = self._acquire_task_worktree(repo, original_branch)
            if acquired is None:
                return None
            branch, wt_dir = acquired

            if not GitWorktreeOps.save_original_branch(repo, branch, original_branch):
                # pragma: no cover — git config failure
                GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
                return None
            # Best-effort: a missing owner pid only means another
            # process's reclaim treats this worktree as legacy (no
            # cross-process liveness protection), never a task failure.
            GitWorktreeOps.save_owner_pid(repo, branch)

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

            wt_work_dir = wt_dir / offset
            self._wt = GitWorktree(
                repo_root=repo,
                branch=branch,
                original_branch=original_branch,
                wt_dir=wt_dir,
                baseline_commit=baseline_commit,
                work_dir=wt_work_dir,
            )

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

        Both warnings are routed through :meth:`_broadcast_to_watchers`
        so they carry a tab id.  Every flush happens BEFORE the run's
        task id exists (``run()`` flushes before ``super().run()``) or
        outside a run entirely (:meth:`new_chat`), so an unstamped
        warning has no fan-out target at all and is dropped without
        ever reaching the user — the failure mode that kept "your
        declined work now lives only on ``kiss/wt-*``" silent.

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
                self._broadcast_to_watchers(
                    printer, {"type": "warning", "message": stash_warning},
                )
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
                self._broadcast_to_watchers(
                    printer, {"type": "warning", "message": merge_warning},
                )
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
                equivalent to ``ChatSorcarAgent.run()``.  The optional
                ``auto_commit`` kwarg overrides the user's persisted
                "Auto commit" setting for this run and every automatic
                cleanup that follows it; omit it to use the setting.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        # Before _try_setup_worktree: retiring the PREVIOUS worktree is
        # an automatic path, and it must obey the toggle as it stands
        # now (the user may have switched it off since the last run).
        auto_commit = kwargs.pop("auto_commit", None)
        self.auto_commit_enabled = (
            _config_auto_commit_enabled()
            if auto_commit is None
            else bool(auto_commit)
        )
        if self._chat_id == "":
            self._chat_id = _allocate_chat_id()

        printer = kwargs.get("printer")
        if printer is not None:
            # Bind the caller's printer BEFORE any worktree setup.  The
            # orphan reclaim inside :meth:`_try_setup_worktree` builds
            # its live-branch exclusion set through
            # ``self.printer.live_worktree_branches`` — but on a fresh
            # agent ``self.printer`` is only assigned deep inside
            # ``super().run()`` (via ``_reset``), which runs AFTER the
            # worktree setup.  Without this early bind, the first run
            # of a new agent saw no live siblings and its reclaim pass
            # squash-merged and deleted the worktree of every other
            # RUNNING task in the same repo.
            self.set_printer(printer)

        wt_work_dir: Path | None = None
        if kwargs.pop("use_worktree", True):
            work_dir_str = kwargs.get("work_dir")
            discovery_dir = Path(work_dir_str) if work_dir_str else Path.cwd()
            repo = GitWorktreeOps.discover_repo(discovery_dir)
            if repo is None:
                logger.warning("Not a git repo, running task directly")
            else:
                wt_work_dir = self._try_setup_worktree(repo, work_dir_str)

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
                    "worktreeWorkDir": str(wt_work_dir),
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
        self._pending_review = False

        if wt.original_branch is None:
            merge_cmd = _manual_merge_cmd(wt)
            return (
                "Cannot merge: original branch is unknown (likely due to a "
                "crash during setup).  Please specify the target branch "
                "manually:\n"
                f"    git checkout <branch> && {merge_cmd}"
            )

        # Merging IS the user's explicit "commit this work" decision,
        # so it commits even when the Auto-commit setting is off; that
        # setting only governs the automatic paths.
        if not self._finalize_worktree(force_commit=True):
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

    def discard(self, *, rescue_ignored: bool = False) -> str:
        """Throw away the task branch and worktree, checkout original.

        Every step is idempotent — safe to call multiple times.
        Acquires ``repo_lock`` to serialize against concurrent
        merge/release operations on the same repository.

        Like the commit-and-remove path
        (:meth:`_commit_and_clean_worktree`), the removal first waits
        up to :data:`_ABANDONED_SUBAGENT_WAIT_SECONDS` for abandoned
        sub-agent threads still writing into this worktree.  When one
        is still running after the wait, nothing is discarded and a
        "Discard deferred" message tells the caller to retry: deleting
        a directory under a live writer loses whatever it produces
        next and can leave a half-recreated zombie directory behind.

        Args:
            rescue_ignored: When True, git-ignored files the task
                created in the worktree are copied into the main
                repository (never overwriting existing files) before
                the directory is removed.  The AUTOMATIC discard paths
                pass True — they run because the changed-files probe
                saw "no changes", but that probe cannot see ignored
                files, so a task whose only output was e.g. a dataset
                in an ignored ``data/`` directory would otherwise have
                that output silently destroyed.  A user-explicit
                discard keeps the default False: the user asked for
                the work to be thrown away.

        Returns:
            Confirmation message (includes a warning if checkout
            to the original branch failed), or a "Discard deferred"
            message when a live sub-agent prevented the removal.

        Raises:
            RuntimeError: If no worktree task is pending.
        """
        if self._wt is None:
            raise RuntimeError("No pending worktree task to discard")

        wt = self._wt
        if wt.wt_dir.exists() and not self.reclaim_abandoned_subagents(
            timeout=_ABANDONED_SUBAGENT_WAIT_SECONDS,
        ):
            logger.warning(
                "A sub-agent thread is still running inside worktree "
                "'%s'; deferring discard of %s",
                wt.branch, wt.wt_dir,
            )
            return (
                f"Discard deferred: a sub-agent of branch '{wt.branch}' "
                f"is still running inside {wt.wt_dir}, so the worktree "
                "was kept instead of being deleted underneath it. "
                "Retry the discard once the sub-agent has stopped."
            )
        checkout_warning = ""
        delete_warning = ""
        with repo_lock(wt.repo_root):
            if rescue_ignored and wt.wt_dir.exists():
                try:
                    _, rescue_ok = GitWorktreeOps.rescue_ignored_files(
                        wt.wt_dir, wt.repo_root,
                    )
                except Exception:  # pragma: no cover — filesystem failure
                    logger.warning(
                        "Ignored-file rescue failed for %s",
                        wt.wt_dir, exc_info=True,
                    )
                    rescue_ok = False
                if not rescue_ok:
                    # Fail closed: this automatic discard runs because
                    # the changed-files probe saw nothing, so the
                    # ignored files are the worktree's ONLY content —
                    # deleting the directory now would destroy their
                    # only copy.
                    logger.warning(
                        "Deferring discard of worktree '%s': its "
                        "git-ignored output could not be rescued",
                        wt.branch,
                    )
                    return (
                        f"Discard deferred: git-ignored files created "
                        f"by the task in branch '{wt.branch}' could not "
                        f"be copied into the main repository, so the "
                        f"worktree at {wt.wt_dir} was kept — it holds "
                        "the only copy of those files."
                    )
            # Cleared only past every deferral return above, so a
            # deferred discard keeps the pending-review protection
            # (a tab close must not auto-merge work the user asked
            # to throw away).
            self._pending_review = False
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
