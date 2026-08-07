# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Merge / worktree / autocommit flow mixin for the VS Code server.

Owns:
- Post-task autocommit of non-worktree task changes.
- Worktree lifecycle presentation (emit pending, broadcast done).
- Worktree merge/discard user actions + conflict checking.

Split out of ``server.py`` for organisation.
"""

from __future__ import annotations

import enum
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    _porcelain_entries,
    _unquote_git_path,
    repo_lock,
)
from kiss.agents.sorcar.persistence import _append_chat_event
from kiss.agents.sorcar.useful_tools import _stale_worktree_fallback
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.diff_merge import _capture_untracked, _git
from kiss.server.helpers import generate_commit_message_from_diff

if TYPE_CHECKING:
    from kiss.server.json_printer import JsonPrinter

logger = logging.getLogger(__name__)


def _state_task_key(state: AgentState | None) -> str | None:
    """Return the task id *state* last ran, preferring the live agent's.

    Args:
        state: The agent state to inspect, or ``None``.

    Returns:
        The task id, or ``None`` when the state never ran a task.
    """
    if state is None:
        return None
    agent = state.agent
    task_id = getattr(agent, "_last_task_id", None) if agent is not None else None
    if task_id:
        return str(task_id)
    return state.task_id or None


def _unquoted_name_lines(output: str) -> list[str]:
    """Parse ``git diff --name-only`` output into unquoted paths.

    Even with ``core.quotepath=false``, git C-quotes any path that
    contains a double-quote, backslash, or control character.  Without
    unquoting, changed-file lists show bogus names and the conflict
    file-overlap sets can never intersect the real on-disk paths.

    Args:
        output: Raw stdout from a ``--name-only`` git command.

    Returns:
        List of unquoted relative file paths.
    """
    return [
        _unquote_git_path(line)
        for line in output.split("\n")
        if line
    ]


def _porcelain_paths(
    output: str, *, rename_both_sides: bool = False,
) -> list[str]:
    """Parse ``git status --porcelain`` output into unquoted file paths.

    Shared by :meth:`_MergeFlowMixin._main_dirty_files` and the
    porcelain fallback of
    :meth:`_MergeFlowMixin._get_worktree_changed_files` so the two
    parsers cannot drift apart.  Like :func:`_unquoted_name_lines`,
    the path tail (``line[3:]``) is NOT ``strip()``-ed and the output
    is split on ``\\n`` only: space-adjacent filenames are legal, and
    stripping would mangle any that git leaves unquoted.

    Rename/copy entries (``R  old -> new``) are split on the `` -> ``
    boundary (respecting quoting) instead of being emitted as one
    bogus ``"old -> new"`` path.

    Thin wrapper over the shared
    :func:`kiss.agents.sorcar.git_worktree._porcelain_entries` parser
    (also backing ``GitWorktreeOps.copy_dirty_state``) so the porcelain
    parsers cannot drift apart.

    Args:
        output: Raw stdout from a ``git status --porcelain`` command.
        rename_both_sides: When True, emit BOTH the old and the new
            side of a rename/copy entry (mirroring a
            ``diff --no-renames`` listing); when False, emit only the
            new side.

    Returns:
        De-duplicated list of relative file paths in output order.
    """
    files: list[str] = []

    def _add(path: str) -> None:
        if path and path not in files:
            files.append(path)

    for _code, old_name, new_name in _porcelain_entries(output):
        if rename_both_sides and old_name is not None:
            _add(old_name)
        _add(new_name)
    return files


def _is_valid_baseline(git_dir: str, sha: str) -> bool:
    """Check if *sha* refers to a valid commit object in *git_dir*.

    Args:
        git_dir: Directory to run the git command in.
        sha: Object SHA to validate.

    Returns:
        True if *sha* is a commit that exists in the repo.
    """
    check = _git(git_dir, "cat-file", "-t", sha)
    return check.returncode == 0 and check.stdout.strip() == "commit"


class _PendingOutcome(enum.Enum):
    """What a caller must do after inspecting a pending worktree.

    A boolean cannot express the third case.  ``NOOP`` means the
    pending worktree belongs to somebody else right now — a live task
    writing into it, or a merge/discard in flight — so the caller must
    leave it completely alone rather than fall back to presenting it
    (which could also auto-discard an empty branch out from under a
    running task).
    """

    FINALIZED = "finalized"
    """The worktree's fate was decided here (merged or discarded)."""

    PRESENT = "present"
    """Nothing was done; the caller should present the pending worktree.

    The caller does **not** own the worktree: the tab holds no pending
    branch, or is not in worktree mode at all, so presenting is itself
    harmless (and usually a no-op).
    """

    PRESENT_CLAIMED = "present_claimed"
    """Like :attr:`PRESENT`, but ``is_merging`` was claimed for the caller.

    Returned when the tab really does hold a pending worktree that the
    user must be shown.  The claim is taken in the same locked section
    that observed the worktree free, so a second resume arriving at the
    same moment is turned away instead of racing the presentation (its
    empty-branch auto-discard mutates git state).  The caller owns the
    flag and must release it — see
    :meth:`_MergeFlowMixin._release_present_claim`.
    """

    NOOP = "noop"
    """Another owner holds the worktree; the caller must not touch it."""


class _MergeFlowMixin:
    """Merge-view, worktree-action, and autocommit methods."""

    if TYPE_CHECKING:
        printer: JsonPrinter
        work_dir: str
        _state_lock: threading.RLock

        def _any_non_wt_running(self) -> bool: ...
        def _dispose_if_closed(self, tab_id: str) -> None: ...

    def _main_dirty_files(self, work_dir: str = "") -> list[str]:
        """List modified, staged and untracked files in the main working tree.

        Uses ``git status --porcelain -uall`` so untracked files inside
        new directories are also reported.  Returns an empty list when
        the working tree is clean or ``work_dir`` is not a git repo.

        Args:
            work_dir: The tab's working directory.  Preferred over the
                daemon-wide ``self.work_dir`` because the shared
                ``kiss-web`` daemon may have been launched from (or
                synced to) a different — possibly non-git — folder than
                the window that owns this tab.  Falls back to
                ``self.work_dir`` when empty.

        Returns:
            De-duplicated list of file paths (relative to ``work_dir``).
        """
        work_dir = work_dir or self.work_dir
        repo = GitWorktreeOps.discover_repo(Path(work_dir))
        if repo is None:
            return []
        result = _git(work_dir, "status", "--porcelain", "-uall")
        if result.returncode != 0:
            return []
        return _porcelain_paths(result.stdout)

    def _broadcast_autocommit_done(
        self,
        tab_id: str,
        *,
        success: bool,
        committed: bool,
        message: str,
        commit_message: str | None = None,
    ) -> dict[str, Any]:
        """Broadcast an ``autocommit_done`` event and return it.

        Args:
            tab_id: Frontend tab identifier.
            success: Whether the action succeeded.
            committed: Whether a commit was actually created.
            message: Human-readable status message.
            commit_message: Full commit message (only when committed).

        Returns:
            The event dict (for optional persistence).
        """
        event: dict[str, Any] = {
            "type": "autocommit_done",
            "success": success,
            "committed": committed,
            "message": message,
            "tabId": tab_id,
        }
        if commit_message is not None:
            event["commitMessage"] = commit_message
        self.printer.broadcast(event)
        return event

    def _autocommit_changes(
        self, tab_id: str = "", *, work_dir: str = "",
    ) -> None:
        """Stage-all + generate-message + commit the tab's working tree.

        Called by the post-task path for non-worktree tasks: with the
        interactive diff review gone, task changes are committed
        directly and reported through ``autocommit_progress`` /
        ``autocommit_done`` events.  A clean tree is a cheap no-op
        ("Nothing to commit.").

        Args:
            tab_id: The tab that ran the task (echoed in the
                ``autocommit_done`` event).
            work_dir: The tab's working directory.  Preferred over the
                daemon-wide ``self.work_dir`` because the shared
                ``kiss-web`` daemon may have been launched from (or
                synced to) a different — possibly non-git — folder than
                the window that owns this tab.  Falls back to
                ``self.work_dir`` when empty.
        """
        work_dir = work_dir or self.work_dir
        try:
            work_path = Path(work_dir)
            if not work_path.exists():
                fallback = _stale_worktree_fallback(work_path)
                if fallback is not None:
                    work_dir = str(fallback)
                    work_path = fallback
            repo = GitWorktreeOps.discover_repo(work_path)
            if repo is None:
                self._broadcast_autocommit_done(
                    tab_id, success=False, committed=False,
                    message="Not a git repository.",
                )
                return
            with repo_lock(repo):
                self.printer.broadcast({
                    "type": "autocommit_progress",
                    "message": "Staging changes…",
                    "tabId": tab_id,
                })
                add_result = _git(work_dir, "add", "-A")
                if add_result.returncode != 0:
                    err = (add_result.stderr or "").strip()
                    first_line = err.splitlines()[0] if err else "git add failed"
                    self._broadcast_autocommit_done(
                        tab_id, success=False, committed=False,
                        message=f"Staging failed: {first_line}",
                    )
                    return
                diff = _git(work_dir, "diff", "--cached")
                if not diff.stdout.strip():
                    self._broadcast_autocommit_done(
                        tab_id, success=True, committed=False,
                        message="Nothing to commit.",
                    )
                    return
                self.printer.broadcast({
                    "type": "autocommit_progress",
                    "message": "Generating commit message…",
                    "tabId": tab_id,
                })
                with self._state_lock:
                    prompt_state = agent_state.find_by_tab(tab_id)
                user_prompt = (
                    prompt_state.last_user_prompt if prompt_state else ""
                ) or None
                task_result = (
                    prompt_state.last_result_summary if prompt_state else ""
                ) or None
                msg = (
                    generate_commit_message_from_diff(
                        diff.stdout,
                        user_prompt=user_prompt,
                        task_result=task_result,
                    )
                    or "Auto-commit"
                )
                self.printer.broadcast({
                    "type": "autocommit_progress",
                    "message": "Committing…",
                    "tabId": tab_id,
                })
                ok = GitWorktreeOps.commit_staged(repo, msg)
            if ok:
                msg_lines = msg.splitlines()
                subject = msg_lines[0] if msg_lines else msg
                done_event = self._broadcast_autocommit_done(
                    tab_id, success=True, committed=True,
                    message=f"Committed: {subject}",
                    commit_message=msg,
                )
                if tab_id:
                    with self._state_lock:
                        task_id = _state_task_key(agent_state.find_by_tab(tab_id))
                    if task_id is not None:
                        _append_chat_event(done_event, task_id=task_id)
            else:
                self._broadcast_autocommit_done(
                    tab_id, success=False, committed=False,
                    message="git commit failed (pre-commit hook?).",
                )
        except Exception as e:  # pragma: no cover — unexpected git/LLM error
            logger.debug("Autocommit action failed", exc_info=True)
            self._broadcast_autocommit_done(
                tab_id, success=False, committed=False,
                message=str(e),
            )

    def _emit_pending_worktree(self, tab_id: str = "") -> None:
        """Finalize or present a pending worktree branch on session load.

        Worktrees are no longer associated with chat sessions, so
        there is no cross-process restoration to perform here.  What
        happens to a still-pending worktree depends on the tab's
        auto-commit toggle:

        * **auto-commit ON** — the user asked not to be interrupted, so
          the branch is merged (or discarded when it holds nothing)
          silently via :meth:`_handle_worktree_action`.
        * **auto-commit OFF** — delegate to
          :meth:`_present_pending_worktree`, which re-broadcasts the
          ``worktree_done`` Merge / Discard buttons (or discards an
          empty branch).

        Either way the call no-ops unless the tab has ``use_worktree``
        set and its transient agent still holds a pending worktree.

        Auto-commit ON does *not* always finalize.  Two owners outrank
        the toggle and make this a complete no-op:

        * a merge or discard already in flight on the tab;
        * a task still running on the tab: its agent is writing into
          the worktree right now.

        A third exception — the agent's ``_pending_review`` flag, set
        for a task that failed or was stopped — declines the silent
        finalize but still presents the Merge / Discard buttons,
        because unverified work must never be merged behind the
        user's back.  That case comes back as
        :attr:`_PendingOutcome.PRESENT_CLAIMED`, carrying the
        ownership claim the presentation runs under; plain
        :attr:`_PendingOutcome.PRESENT` means there was nothing to own.

        Args:
            tab_id: The tab to check for pending worktree.
        """
        outcome = self._finalize_pending_worktree(tab_id)
        if outcome is _PendingOutcome.PRESENT:
            self._present_pending_worktree(tab_id)
            return
        if outcome is not _PendingOutcome.PRESENT_CLAIMED:
            return
        # The claim is ours, so it is ours to release.  Presenting no
        # longer starts anything that outlives this call, so the claim
        # is always dropped here; the `finally` matters because an
        # exception must not leave the tab permanently busy.
        try:
            self._present_pending_worktree(tab_id)
        finally:
            self._release_present_claim(tab_id)

    def _release_present_claim(self, tab_id: str) -> None:
        """Drop the ownership claim taken for presenting a pending worktree.

        :meth:`_finalize_pending_worktree` claims ``is_merging`` before
        returning :attr:`_PendingOutcome.PRESENT_CLAIMED` so that only
        one resume presents the worktree.  Once the presentation is
        done the claim must go again, or the tab stays busy forever and
        every later task, merge and discard on it is refused.

        Args:
            tab_id: The tab whose speculative claim to release.
        """
        with self._state_lock:
            state = agent_state.find_by_tab(tab_id)
            if state is None:
                return
            state.is_merging = False
        self._dispose_if_closed(tab_id)

    def _finalize_pending_worktree(self, tab_id: str) -> _PendingOutcome:
        """Merge or discard a pending worktree without asking the user.

        The auto-commit counterpart of :meth:`_present_pending_worktree`,
        and the same decision the post-task fast path in
        ``_run_task_inner`` makes: merge when the branch carries
        changes, discard when it does not.

        Returns :attr:`_PendingOutcome.NOOP` — the worktree already has
        an owner, so the caller must leave it entirely alone — when:

        * a merge or discard is already in flight on the tab;
        * a task is still active on the tab.  Unlike the post-task
          finalize — which runs on the very thread that owns
          ``is_task_active`` — a history click is an unrelated thread,
          and merging or discarding a worktree the agent is still
          writing into would corrupt or delete its work.  Presenting is
          just as unsafe: its ``discard_if_empty`` path would delete a
          branch the running task has not committed to yet;
        * a task has been *submitted* but has not reached its worker
          yet.  Ownership is therefore decided by the one shared
          :meth:`AgentState.busy` predicate rather than by reading the two
          flags directly: during that startup window both of them read
          False, and claiming ``is_merging`` there makes the worker
          refuse the run the user just typed.

        Returns :attr:`_PendingOutcome.PRESENT` — nothing was done, and
        the caller may present the worktree without owning anything —
        when the tab is not in worktree mode, or holds no pending
        worktree.  Presenting is itself a no-op then, so no claim is
        taken and none may be released.

        Returns :attr:`_PendingOutcome.PRESENT_CLAIMED` — the caller
        should present the Merge / Discard buttons and has been handed
        the ``is_merging`` claim to do it under — when a pending
        worktree really is there but must not be finalized silently:

        * auto-commit is off, so the user asked to decide explicitly;
        * the agent is in ``_pending_review`` state.  ``_run_task_inner``
          raises that flag for a task that failed or was stopped, and
          :meth:`WorktreeSorcarAgent._preserve_pending_worktree_for_review`
          documents the contract it encodes: incomplete, unverified work
          stays on its ``kiss/wt-*`` branch and is never merged into the
          user's branch behind their back — the user must click Merge
          explicitly.  Auto-commit means "do not interrupt me", not
          "publish work that never finished".

        A merge that is attempted but still cannot complete — the main
        tree may hold the very conflict that stranded the branch in the
        first place — is reported through the normal ``worktree_result``
        event and the branch is left pending, so work is never lost.

        Args:
            tab_id: The tab whose pending worktree to finalize.

        Returns:
            The :class:`_PendingOutcome` telling the caller what, if
            anything, is left to do.
        """
        with self._state_lock:
            state = agent_state.find_by_tab(tab_id)
            if state is None or not state.use_worktree:
                return _PendingOutcome.PRESENT
            if state.busy():
                return _PendingOutcome.NOOP
            wt_agent = state.agent
            if wt_agent is None or not wt_agent._wt_pending:
                return _PendingOutcome.PRESENT
            if wt_agent._pending_review or not state.auto_commit_mode:
                # The caller will present the pending worktree.  Claim
                # it here too: without a claim taken in the same locked
                # section that observed `is_merging` clear, two
                # simultaneous resumes would race the presentation's
                # empty-branch auto-discard (F4-20).
                state.is_merging = True
                return _PendingOutcome.PRESENT_CLAIMED
            # Claim the worktree before releasing the lock so a
            # concurrent resume (remote commands run on a thread pool)
            # cannot finalize the same branch twice.  The claim is held
            # continuously across BOTH the changed-files probe and the
            # action it selects: dropping it in between would reopen
            # the very race it exists to close, and would also let the
            # probe's answer go stale before it is acted on.
            state.is_merging = True
        try:
            changed = self._get_worktree_changed_files(tab_id)
            action = "merge" if changed else "discard"
            result = self._handle_worktree_action(
                action, tab_id, already_claimed=True,
            )
        finally:
            with self._state_lock:
                state.is_merging = False
            self._dispose_if_closed(tab_id)
        self.printer.broadcast(
            {"type": "worktree_result", "tabId": tab_id, **result},
        )
        return _PendingOutcome.FINALIZED

    def _present_pending_worktree(
        self, tab_id: str, *, discard_if_empty: bool = True,
    ) -> None:
        """Auto-discard an empty pending worktree or emit ``worktree_done``.

        Single source of truth for post-task / session-resume handling
        of a pending worktree (RED-10 fix).

        Behavior:
        - No pending worktree: return.
        - Worktree has changed files: broadcast ``worktree_done`` so
          the user gets the Merge / Discard buttons.
        - Worktree has no changes and *discard_if_empty* is True:
          auto-discard the empty branch (BUG-66 — clean up stale
          resumed sessions).  A concurrent non-worktree task does not
          block this: an empty discard never touches the main working
          tree.
        - Worktree has no changes and *discard_if_empty* is False:
          preserve the branch.  The post-task path passes
          ``discard_if_empty=False`` when the user opted into the
          worktree workflow but has not explicitly chosen to merge or
          discard yet — so the branch must remain visible in
          ``git branch`` for manual inspection / merge / discard.

        Args:
            tab_id: The tab with a pending worktree.
            discard_if_empty: When True (default), auto-discard the
                branch if no files changed.  Post-task callers should
                pass False to preserve the branch for manual action.
        """
        with self._state_lock:
            state = agent_state.find_by_tab(tab_id)
        if state is None or not state.use_worktree:
            return
        wt_agent = state.agent
        if wt_agent is None or not wt_agent._wt_pending:
            return
        changed = self._get_worktree_changed_files(tab_id)
        if not changed and discard_if_empty:
            # Discarding an EMPTY worktree removes its directory and
            # its unmerged branch without touching the main working
            # tree, so a concurrent non-worktree task is no reason to
            # skip it — skipping leaks the worktree forever because
            # nothing ever retries.
            with self._state_lock:
                prev_merging = state.is_merging
                state.is_merging = True
            try:
                wt_agent.discard()
            finally:
                with self._state_lock:
                    state.is_merging = prev_merging
                # A close that arrived during the discard saw the
                # tab busy and deferred disposal; nothing later
                # would dispose it (F4-29).
                self._dispose_if_closed(tab_id)
            return
        if not changed:
            return
        event: dict[str, Any] = {
            "type": "worktree_done",
            "branch": wt_agent._wt_branch,
            "worktreeDir": str(wt_agent._wt_dir),
            "originalBranch": wt_agent._original_branch,
            "changedFiles": changed,
            "hasConflict": self._check_merge_conflict(tab_id),
            "tabId": tab_id,
        }
        self.printer.broadcast(event)

    def _check_merge_conflict(self, tab_id: str = "") -> bool:
        """Check if merging the worktree branch into original would conflict.

        Pure query — does **not** commit or otherwise mutate git state
        (BUG-9 fix).  Uses file-level overlap detection between:

        1. Files changed on the original branch since the fork point.
        2. Files changed in the worktree (committed + uncommitted)
           since the fork point.

        When both sides modify the same file, reports a potential
        conflict.  Also checks for dirty main working-tree files that
        overlap with the worktree changes (which would cause
        ``git merge`` to refuse).

        Args:
            tab_id: The tab whose worktree to check.

        Returns:
            True if the merge would likely fail, False otherwise.
        """
        state = agent_state.find_by_tab(tab_id)
        if state is None or not state.use_worktree:
            return False
        wt_agent = state.agent
        if wt_agent is None:
            return False
        wt = wt_agent._wt
        if wt is None or wt.original_branch is None:
            return False
        wt_dir = wt.wt_dir
        if not wt_dir.exists():
            return False

        baseline_valid = bool(
            wt.baseline_commit
            and _is_valid_baseline(str(wt_dir), wt.baseline_commit)
        )
        if baseline_valid:
            assert wt.baseline_commit is not None
            orig_fork = f"{wt.baseline_commit}^"
            wt_fork: str = wt.baseline_commit
        else:
            mb = _git(str(wt_dir), "merge-base", "HEAD", wt.original_branch)
            if mb.returncode != 0 or not mb.stdout.strip():
                return False
            orig_fork = wt_fork = mb.stdout.strip()

        orig_diff = _git(
            str(wt.repo_root), "diff", "--name-only", "--no-renames",
            orig_fork, wt.original_branch,
        )
        orig_files = (
            set(_unquoted_name_lines(orig_diff.stdout))
            if orig_diff.returncode == 0 else set()
        )

        wt_diff = _git(str(wt_dir), "diff", "--name-only", "--no-renames", wt_fork)
        wt_files = (
            set(_unquoted_name_lines(wt_diff.stdout))
            if wt_diff.returncode == 0 else set()
        )
        wt_files.update(_capture_untracked(str(wt_dir)))

        if orig_files & wt_files:
            return True

        with self._state_lock:
            if self._any_non_wt_running():
                return False
        dirty: set[str] = set()
        for extra_flags in ((), ("--cached",)):
            dirty.update(
                GitWorktreeOps._diff_name_only(
                    wt.repo_root, "--no-renames", *extra_flags,
                )
            )
        dirty.update(_capture_untracked(str(wt.repo_root)))
        return bool(dirty & wt_files)

    @staticmethod
    def _resolve_base_ref(
        git_dir: str, baseline: str | None, original_branch: str,
        tip: str = "HEAD",
    ) -> str:
        """Resolve the base ref for worktree diff operations.

        Uses the baseline commit when available **and valid** (i.e. the
        SHA exists in the repository), otherwise falls back to
        ``git merge-base`` between *tip* and *original_branch*.

        BUG-51 fix: validates baseline SHA with ``git cat-file -t``
        before returning it.  An invalid baseline (e.g. from a
        force-pushed branch or corrupt config) is silently ignored
        so callers get a usable ref instead of a guaranteed-to-fail one.

        Args:
            git_dir: Directory to run git commands in.
            baseline: Baseline commit SHA, or ``None``.
            original_branch: The user's original branch name.
            tip: The tip ref to compute merge-base against (default ``HEAD``).

        Returns:
            A git ref string suitable for ``git diff``.
        """
        if baseline and _is_valid_baseline(git_dir, baseline):
            return baseline
        mb = _git(git_dir, "merge-base", tip, original_branch)
        if mb.returncode == 0 and mb.stdout.strip():
            return mb.stdout.strip()
        return original_branch

    def _get_worktree_changed_files(self, tab_id: str = "") -> list[str]:
        """List files changed in the worktree vs the original branch.

        Detects both committed changes on the worktree branch and
        uncommitted changes in the worktree working tree.  When the
        worktree directory exists, runs ``git diff`` and
        ``git ls-files --others`` inside it so that uncommitted
        edits and new files are included.  Falls back to a branch-
        to-branch diff when the worktree has already been removed.

        Args:
            tab_id: The tab whose worktree to check.

        Returns:
            Sorted deduplicated list of relative file paths.
        """
        state = agent_state.find_by_tab(tab_id)
        if state is None or not state.use_worktree:
            return []
        wt_agent = state.agent
        if wt_agent is None or not wt_agent._original_branch:
            return []
        wt = wt_agent
        original_branch = wt._original_branch
        assert original_branch is not None
        wt_dir = wt._wt_dir
        if wt_dir and wt_dir.exists():
            base_ref = self._resolve_base_ref(
                str(wt_dir), wt._baseline_commit, original_branch,
            )
            tracked = _git(
                str(wt_dir), "diff", "--name-only", "--no-renames", base_ref,
            )
            if tracked.returncode == 0:
                files = _unquoted_name_lines(tracked.stdout)
            else:
                # The diff query failed (e.g. the original branch was
                # renamed/deleted so base_ref no longer resolves).  A
                # clean ``status --porcelain`` alone must NOT be taken
                # as "no changes" (F4-21): the worktree may hold
                # COMMITTED task work, and callers auto-discard the
                # branch when this returns [].  Also list files from
                # commits unique to this worktree (not reachable from
                # any other branch) so committed work is never
                # mistaken for a clean worktree.
                status = _git(str(wt_dir), "status", "--porcelain")
                files = _porcelain_paths(
                    status.stdout, rename_both_sides=True,
                )
                unique_args = ["log", "--pretty=format:", "--name-only",
                               "--no-renames", "HEAD", "--not"]
                if wt._wt_branch:
                    unique_args.append(f"--exclude={wt._wt_branch}")
                unique_args.append("--branches")
                unique = _git(str(wt_dir), *unique_args)
                if unique.returncode == 0:
                    files.extend(_unquoted_name_lines(unique.stdout))
            files.extend(_capture_untracked(str(wt_dir)))
            return sorted(set(files))
        if not wt._wt_branch:
            return []
        repo_root = str(wt._repo_root) if wt._repo_root else self.work_dir
        base_ref = self._resolve_base_ref(
            repo_root, wt._baseline_commit, original_branch,
            tip=wt._wt_branch,
        )
        result = _git(repo_root, "diff", "--name-only", "--no-renames",
                      base_ref,
                      wt._wt_branch)
        return (
            _unquoted_name_lines(result.stdout)
            if result.returncode == 0 else []
        )

    def _check_worktree_busy(self, state: AgentState, verb: str) -> dict[str, Any] | None:
        """Return an error dict if a worktree action should be refused, else None.

        Checks both the tab's own task and any non-worktree task running
        on the main tree (BUG-35, BUG-72 fixes).

        Must be called with ``_state_lock`` already held (RACE-1 fix)
        so the caller can atomically set ``state.is_merging = True``
        before releasing the lock — otherwise a non-wt task on
        another tab could pass its own ``is_merging`` guard in the
        TOCTOU window between this check returning ``None`` and the
        caller acquiring ``_state_lock`` again to set the flag.

        Args:
            state: The agent state to check.
            verb: Human-readable action name (e.g. ``"merging"``).

        Returns:
            Error dict with ``success: False`` when busy, otherwise ``None``.
        """
        if state.is_task_active:
            return {
                "success": False,
                "message": (
                    f"A worktree task is still running on this tab. "
                    f"Wait for it to finish (or stop it) before {verb}."
                ),
            }
        if state.is_merging:
            return {
                "success": False,
                "message": (
                    "A merge or discard is already in progress "
                    f"on this tab. Wait for it to finish before {verb}."
                ),
            }
        if self._any_non_wt_running():
            return {
                "success": False,
                "message": (
                    "Another tab is running a task on the main working "
                    f"tree. Wait for it to finish before {verb}."
                ),
            }
        return None

    def _handle_worktree_action(
        self,
        action: str,
        tab_id: str = "",
        *,
        internal: bool = False,
        already_claimed: bool = False,
    ) -> dict[str, Any]:
        """Execute a worktree merge/discard/manual action.

        Restores agent worktree state from git if needed (e.g. after a
        server process restart where in-memory state was lost).

        Args:
            action: One of ``"merge"`` or ``"discard"``.
            tab_id: The tab whose worktree to act on.
            internal: When True, bypass the ``_check_worktree_busy``
                guard.  Used by ``_run_task_inner``'s post-task
                auto-merge / auto-discard block (RACE-3 fix), which
                runs on the same task thread that owns
                ``state.is_task_active = True`` and therefore would
                otherwise be refused by its own guard.  A concurrent
                non-worktree task on the main tree still blocks a
                ``"merge"`` — but never a ``"discard"``, which does
                not touch the main working tree.
            already_claimed: When True, the caller has already set
                ``state.is_merging`` under ``_state_lock`` after checking
                the busy conditions itself, and will clear it (and call
                ``_dispose_if_closed``) when its own wider critical
                section ends.  Implies *internal*.  This method must
                therefore neither re-claim nor release the flag: doing
                so would punch a hole in the caller's claim exactly
                where :meth:`_finalize_pending_worktree` needs it to be
                continuous.  The main-tree guard still applies.

        Returns:
            Dict with ``success`` bool and ``message`` string.
        """
        internal = internal or already_claimed
        state = agent_state.find_by_tab(tab_id)
        if state is None or not state.use_worktree:
            return {"success": False, "message": "Worktree mode is not enabled"}
        wt_agent = state.agent
        if wt_agent is None or not wt_agent._wt_pending:
            return {
                "success": False,
                "message": "No pending worktree changes to act on",
            }
        wt = wt_agent
        verb = {"merge": "merging", "discard": "discarding"}.get(action)
        if verb is None:
            return {"success": False, "message": f"Unknown action: {action}"}
        repo_root = wt._repo_root
        if repo_root is None:
            return {
                "success": False,
                "message": "No pending worktree changes to act on",
            }
        with self._state_lock:
            if not internal:
                busy = self._check_worktree_busy(state, verb)
                if busy:
                    return busy
            elif action == "merge" and self._any_non_wt_running():
                # internal=True only bypasses this tab's OWN
                # is_task_active/is_merging flags (the post-task
                # auto-finalize runs on the task thread that owns
                # them).  It must NOT bypass the main-tree guard
                # (F4-19): merging stashes/checkouts/merges the
                # main working tree while a direct task on another
                # tab is still writing it.  A DISCARD is exempt: it
                # only removes .kiss-worktrees/<slug> and deletes the
                # unmerged branch, touching neither the main working
                # tree's files nor its HEAD, so refusing it would
                # leak the worktree forever (nothing ever retries).
                return {
                    "success": False,
                    "message": (
                        "Another tab is running a task on the main "
                        "working tree. Wait for it to finish before "
                        f"{verb}."
                    ),
                }
            if not already_claimed:
                state.is_merging = True
        wt._pending_review = False
        try:
            with repo_lock(repo_root):
                if action == "merge":
                    progress_event: dict[str, Any] = {
                        "type": "worktree_progress",
                        "message": "Generating commit message…",
                    }
                    if tab_id:
                        progress_event["tabId"] = tab_id
                    self.printer.broadcast(progress_event)
                    msg = wt.merge()
                    success = "Successfully merged" in msg
                    return {"success": success, "message": msg}
                msg = wt.discard()
                # A partial discard (branch deletion failed) must not
                # report success: the UI would close the workflow
                # while an orphan branch remains (F4-24).
                return {
                    "success": "Partially discarded" not in msg,
                    "message": msg,
                }
        finally:
            if not already_claimed:
                with self._state_lock:
                    state.is_merging = False
                # A close that arrived during the merge/discard saw the
                # tab busy and deferred disposal; without this call the
                # backend tab state would leak indefinitely (F4-23).
                self._dispose_if_closed(tab_id)
