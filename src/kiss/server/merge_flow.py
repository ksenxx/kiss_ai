# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Worktree review flow and manual commit mixin for the VS Code server.

Owns:
- Presentation of a worktree a task left pending (it was stopped, or
  the agent never called ``commit_and_merge``): the ``worktree_done``
  bar with Merge / Discard / Do nothing.
- The worktree merge / discard / detach user actions and the conflict
  probe behind the bar.
- The settings panel's manual "Git Commit" of the tab's working tree.

Split out of ``server.py`` for organisation.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    _git,
    _porcelain_entries,
    _unquote_git_path,
    repo_lock,
)
from kiss.agents.sorcar.merge_conflict_resolver import resolve_merge_conflict
from kiss.agents.sorcar.persistence import _add_task_usage, _append_chat_event
from kiss.agents.sorcar.sorcar_agent import (
    _agent_usage,
    _commit_subject,
    _persisted_task_id,
)
from kiss.agents.sorcar.useful_tools import remap_vanished_worktree
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.helpers import generate_commit_message_from_diff
from kiss.server.json_printer import stamp_event_ts

if TYPE_CHECKING:
    from kiss.server.json_printer import JsonPrinter

logger = logging.getLogger(__name__)


def _git_error_line(result: Any, default: str) -> str:
    """Return the first line of a failed git call's stderr (or stdout).

    Args:
        result: The completed git process.
        default: Returned when git printed nothing at all.

    Returns:
        The first line of the error output, or *default*.
    """
    err = (result.stderr or "").strip() or (result.stdout or "").strip()
    return err.splitlines()[0] if err else default


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
    return [_unquote_git_path(line) for line in output.split("\n") if line]


def _porcelain_paths(output: str, *, rename_both_sides: bool = False) -> list[str]:
    """Parse ``git status --porcelain`` output into unquoted file paths.

    A thin wrapper over the shared
    :func:`kiss.agents.sorcar.git_worktree._porcelain_entries` parser
    (also backing ``GitWorktreeOps.copy_dirty_state``) so the porcelain
    parsers cannot drift apart.  Rename/copy entries (``R  old -> new``)
    are split on the `` -> `` boundary instead of being emitted as one
    bogus ``"old -> new"`` path.

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
    for _code, old_name, new_name in _porcelain_entries(output):
        for path in ((old_name,) if rename_both_sides else ()) + (new_name,):
            if path and path not in files:
                files.append(path)
    return files


def _capture_untracked(work_dir: str) -> set[str]:
    """Return the set of untracked files in the repo.

    Args:
        work_dir: Repository root directory.

    Returns:
        Set of untracked file paths relative to work_dir.
    """
    result = _git("ls-files", "--others", "--exclude-standard", cwd=work_dir)
    return set(_unquoted_name_lines(result.stdout))


def _is_valid_baseline(git_dir: str, sha: str) -> bool:
    """Check if *sha* refers to a valid commit object in *git_dir*.

    Args:
        git_dir: Directory to run the git command in.
        sha: Object SHA to validate.

    Returns:
        True if *sha* is a commit that exists in the repo.
    """
    check = _git("cat-file", "-t", sha, cwd=git_dir)
    return check.returncode == 0 and check.stdout.strip() == "commit"


def _pending_worktrees(agent: Any) -> list[GitWorktree]:
    """Every worktree pending on *agent* (``[]`` for no agent or a non-worktree one)."""
    worktrees = getattr(agent, "worktrees", None)
    return list(worktrees) if isinstance(worktrees, list) else []


class _MergeFlowMixin:
    """Pending-worktree presentation, worktree actions and the manual commit."""

    if TYPE_CHECKING:
        printer: JsonPrinter
        work_dir: str
        _state_lock: threading.RLock

        def _dispose_if_closed(self, tab_id: str) -> None: ...

    # ------------------------------------------------------------------
    # Manual Git Commit
    # ------------------------------------------------------------------

    def _broadcast_autocommit_done(
        self,
        tab_id: str,
        *,
        success: bool,
        committed: bool,
        message: str,
        commit_message: str | None = None,
        work_dir: str = "",
    ) -> dict[str, Any]:
        """Report the outcome of a manual Git Commit and return the event.

        The outcome is a toast ``notification`` — info on success,
        error on failure — plus the ``autocommit_done`` event (flagged
        ``manual``) that re-arms the Git Commit button; the webview and
        the VS Code host skip their own chat rendering of a successful
        manual commit, while a failure also shows its reason in the
        chat webview.

        Args:
            tab_id: Frontend tab identifier.
            success: Whether the commit succeeded.
            committed: Whether a commit was actually created.
            message: Human-readable status message.
            commit_message: Full commit message (only when committed).
            work_dir: The working directory the commit targeted
                (carried as ``workDir``).

        Returns:
            The event dict.
        """
        # Same stable id as the "Auto-generating commit message…"
        # toast, so the outcome replaces it in place (the webview
        # dedups toasts by id) instead of stacking a second one.
        self.printer.broadcast({
            "type": "notification",
            "id": f"manual-commit:{tab_id}",
            "severity": "info" if success else "error",
            "message": message,
            "tabId": tab_id,
        })
        event: dict[str, Any] = {
            "type": "autocommit_done",
            "success": success,
            "committed": committed,
            "message": message,
            "manual": True,
            "tabId": tab_id,
        }
        if commit_message is not None:
            event["commitMessage"] = commit_message
        if work_dir:
            event["workDir"] = work_dir
        self.printer.broadcast(event)
        return event

    def _commit_main_tree(self, tab_id: str, work_dir: str = "") -> None:
        """Stage-all + generate-message + commit the tab's working tree.

        Serves the settings panel's Git Commit button: stages
        everything, generates an LLM commit message from the staged
        diff alone (no ``User prompt:`` / ``Result:`` sections),
        commits, and reports through :meth:`_broadcast_autocommit_done`.
        A clean tree is a cheap no-op ("Nothing to commit.").

        Args:
            tab_id: The tab that pressed the button.
            work_dir: The tab's working directory (preferred over the
                daemon-wide ``self.work_dir``, which may point at a
                different — possibly non-git — folder).  A path under
                a now-deleted ``.kiss-worktrees/kiss_wt-*`` checkout
                is remapped to the parent repository.
        """
        requested_dir = work_dir or self.work_dir
        try:
            work_path = remap_vanished_worktree(Path(requested_dir))
            repo = GitWorktreeOps.discover_repo(work_path)
            if repo is None:
                self._broadcast_autocommit_done(
                    tab_id, success=False, committed=False,
                    message="Not a git repository.", work_dir=requested_dir,
                )
                return
            with repo_lock(repo):
                add_result = _git("add", "-A", cwd=str(work_path))
                if add_result.returncode != 0:
                    self._broadcast_autocommit_done(
                        tab_id, success=False, committed=False,
                        message="Staging failed: " + _git_error_line(add_result, "git add failed"),
                        work_dir=requested_dir,
                    )
                    return
                diff_text = GitWorktreeOps.staged_diff(work_path)
                if not diff_text:
                    self._broadcast_autocommit_done(
                        tab_id, success=True, committed=False,
                        message="Nothing to commit.", work_dir=requested_dir,
                    )
                    return
                self.printer.broadcast({
                    "type": "notification",
                    "id": f"manual-commit:{tab_id}",
                    "severity": "info",
                    "message": "Auto-generating commit message…",
                    "tabId": tab_id,
                })
                msg = generate_commit_message_from_diff(diff_text)
                # Commit directly (the staged diff was verified
                # non-empty above) so a failure's actual git output —
                # pre-commit hook rejection text, identity/config
                # errors, … — can be reported to the user.
                commit_result = _git("commit", "-m", msg, cwd=str(work_path))
            if commit_result.returncode == 0:
                self._broadcast_autocommit_done(
                    tab_id, success=True, committed=True,
                    message=f"Committed: {_commit_subject(msg)}",
                    commit_message=msg, work_dir=requested_dir,
                )
            else:
                self._broadcast_autocommit_done(
                    tab_id, success=False, committed=False,
                    message=(
                        "git commit failed: "
                        + _git_error_line(commit_result, "pre-commit hook?")
                    ),
                    work_dir=requested_dir,
                )
        except Exception as e:  # pragma: no cover — unexpected git/LLM error
            logger.debug("Manual commit failed", exc_info=True)
            self._broadcast_autocommit_done(
                tab_id, success=False, committed=False,
                message=str(e), work_dir=requested_dir,
            )

    # ------------------------------------------------------------------
    # Pending worktree presentation
    # ------------------------------------------------------------------

    def _emit_pending_worktree(self, tab_id: str = "") -> None:
        """Re-present a pending worktree branch on session load.

        A worktree the tab's agent still holds (a task was stopped, or
        finished without ``commit_and_merge``) gets its Merge / Discard
        / Do nothing bar back through :meth:`_present_pending_worktree`.
        A no-op while the tab is busy: a running task may be writing
        into the worktree, and a merge or discard may be in flight.  The
        presentation runs under the tab's ``is_merging`` claim, taken in
        the same locked section that observed the tab free, so two
        simultaneous resumes cannot race its empty-branch auto-discard.

        Args:
            tab_id: The tab to check for a pending worktree.
        """
        with self._state_lock:
            state = agent_state.find_by_tab(tab_id)
            if state is None or state.busy():
                return
            wt_agent = state.agent
            if wt_agent is None or not wt_agent._wt_pending:
                return
            state.is_merging = True
            state.merge_thread = threading.current_thread()
        try:
            self._present_pending_worktree(tab_id)
        finally:
            with self._state_lock:
                state.is_merging = False
                state.merge_thread = None
            self._dispose_if_closed(tab_id)

    def _present_pending_worktree(
        self, tab_id: str, *, discard_if_empty: bool = True,
    ) -> None:
        """Auto-discard an empty pending worktree or emit ``worktree_done``.

        Single source of truth for post-task / session-resume handling
        of a pending worktree.

        - No pending worktree: return.
        - Worktree has changed files: broadcast ``worktree_done`` so
          the user gets the Merge / Discard / Do nothing buttons.
        - Worktree has no changes and *discard_if_empty* is True:
          auto-discard the empty branch (stale resumed sessions).
        - Worktree has no changes and *discard_if_empty* is False:
          preserve the branch.  The post-task path passes False so the
          branch stays visible in ``git branch`` for manual inspection.

        Args:
            tab_id: The tab with a pending worktree.
            discard_if_empty: When True (default), auto-discard the
                branch if no files changed.
        """
        state = agent_state.find_by_tab(tab_id)
        if state is None:
            return
        wt_agent = state.agent
        if wt_agent is None or not wt_agent._wt_pending:
            return
        changed = self._get_worktree_changed_files(tab_id)
        if not changed and discard_if_empty:
            with self._state_lock:
                # Ownership re-check: the reads above ran outside the
                # lock, so a new task may have started on the tab and
                # set up the very worktree the probe just found empty.
                # Refuse unless the tab is still the one we read, still
                # holds this pending worktree, and is either idle or
                # busy solely with OUR claim (the session-resume caller
                # already owns ``is_merging`` on this thread).
                if (
                    agent_state.find_by_tab(tab_id) is not state
                    or state.agent is not wt_agent
                    or not wt_agent._wt_pending
                ):
                    return
                if state.busy() and state.merge_thread is not threading.current_thread():
                    return
                prev_merging = state.is_merging
                prev_thread = state.merge_thread
                state.is_merging = True
                state.merge_thread = threading.current_thread()
            try:
                # Re-probe UNDER the claim: a task could have written
                # output and fully finished between the empty answer
                # above and the claim.  With the claim held, task
                # admission refuses new runs on this tab, so this
                # answer cannot go stale before it is acted on.
                changed = self._get_worktree_changed_files(tab_id)
                if not changed:
                    # Automatic path: rescue git-ignored task output
                    # the changed-files probe cannot see (see
                    # SorcarAgent.discard).
                    wt_agent.discard(rescue_ignored=True)
            finally:
                with self._state_lock:
                    state.is_merging = prev_merging
                    state.merge_thread = prev_thread
                # A close that arrived during the discard saw the
                # tab busy and deferred disposal; nothing later
                # would dispose it.
                self._dispose_if_closed(tab_id)
        if not changed:
            return
        # The singular fields describe the worktree the task's work dir
        # is in (the clients' file-link fallback directory and SCM
        # view); ``worktrees`` lists every pending worktree, one per
        # repository, which Merge / Discard / Do nothing all act on.
        self.printer.broadcast({
            "type": "worktree_done",
            "branch": wt_agent._wt_branch,
            "worktreeDir": str(wt_agent._wt_dir),
            "worktreeWorkDir": str(wt_agent._wt_work_dir),
            "originalBranch": wt_agent._original_branch,
            "changedFiles": changed,
            "hasConflict": self._check_merge_conflict(tab_id),
            "worktrees": [
                {
                    "repoRoot": str(wt.repo_root),
                    "branch": wt.branch,
                    "worktreeDir": str(wt.wt_dir),
                    "originalBranch": wt.original_branch,
                    "changedFiles": self._changed_files_of(wt),
                    "hasConflict": self._conflict_of(wt),
                }
                for wt in _pending_worktrees(wt_agent)
            ],
            "tabId": tab_id,
        })

    def _check_merge_conflict(self, tab_id: str = "") -> bool:
        """Check if merging any pending worktree branch into its original would conflict.

        Pure query — does **not** commit or otherwise mutate git state.

        Args:
            tab_id: The tab whose worktrees to check.

        Returns:
            True if a merge would likely fail, False otherwise.
        """
        state = agent_state.find_by_tab(tab_id)
        wt_agent = state.agent if state is not None else None
        return any(self._conflict_of(wt) for wt in _pending_worktrees(wt_agent))

    def _conflict_of(self, wt: GitWorktree) -> bool:
        """Whether merging *wt*'s branch into its original branch would conflict.

        Uses file-level overlap detection between:

        1. Files changed on the original branch since the fork point.
        2. Files changed in the worktree (committed + uncommitted)
           since the fork point.

        When both sides modify the same file, reports a potential
        conflict.  Also checks for dirty main working-tree files that
        overlap with the worktree changes (which would cause
        ``git merge`` to refuse).

        Args:
            wt: A pending worktree.

        Returns:
            True if the merge would likely fail, False otherwise.
        """
        if wt.original_branch is None or not wt.wt_dir.exists():
            return False
        wt_dir = wt.wt_dir
        wt_fork = self._resolve_base_ref(
            str(wt_dir), wt.baseline_commit, wt.original_branch,
        )
        if wt_fork is None:
            return False
        # The baseline is the worktree's own first commit, so the
        # original branch forked from its parent.
        orig_fork = f"{wt_fork}^" if wt_fork == wt.baseline_commit else wt_fork

        orig_files = set(GitWorktreeOps._diff_name_only(
            wt.repo_root, "--no-renames", orig_fork, wt.original_branch,
        ))
        wt_files = set(GitWorktreeOps._diff_name_only(wt_dir, "--no-renames", wt_fork))
        wt_files.update(_capture_untracked(str(wt_dir)))

        if orig_files & wt_files:
            return True

        dirty: set[str] = set()
        for extra_flags in ((), ("--cached",)):
            dirty.update(
                GitWorktreeOps._diff_name_only(wt.repo_root, "--no-renames", *extra_flags)
            )
        dirty.update(_capture_untracked(str(wt.repo_root)))
        return bool(dirty & wt_files)

    @staticmethod
    def _resolve_base_ref(
        git_dir: str, baseline: str | None, original_branch: str,
        tip: str = "HEAD",
    ) -> str | None:
        """Resolve the base ref for worktree diff operations.

        Uses the baseline commit when available **and valid** (i.e. the
        SHA exists in the repository), otherwise falls back to
        ``git merge-base`` between *tip* and *original_branch*.  An
        invalid baseline (e.g. from a force-pushed branch or corrupt
        config) is silently ignored so callers get a usable ref
        instead of a guaranteed-to-fail one.

        Args:
            git_dir: Directory to run git commands in.
            baseline: Baseline commit SHA, or ``None``.
            original_branch: The user's original branch name.
            tip: The tip ref to compute merge-base against (default ``HEAD``).

        Returns:
            A git ref string suitable for ``git diff``, or ``None`` when
            the baseline is unusable and no merge base exists (e.g. the
            original branch was renamed or deleted).
        """
        if baseline and _is_valid_baseline(git_dir, baseline):
            return baseline
        mb = _git("merge-base", tip, original_branch, cwd=git_dir)
        if mb.returncode == 0 and mb.stdout.strip():
            return mb.stdout.strip()
        return None

    def _get_worktree_changed_files(self, tab_id: str = "") -> list[str]:
        """List files changed in the tab's pending worktrees vs their original branches.

        Files of the worktree the task's work dir is in are relative to
        that worktree; files of any other repository's worktree are
        prefixed with that repository's root, so the list is
        unambiguous for a task that touched several repositories.

        Args:
            tab_id: The tab whose worktrees to check.

        Returns:
            Sorted deduplicated list of file paths.
        """
        state = agent_state.find_by_tab(tab_id)
        if state is None or state.agent is None:
            return []
        primary = state.agent._wt
        files: list[str] = []
        for wt in _pending_worktrees(state.agent):
            prefix = "" if wt is primary else f"{wt.repo_root}/"
            files.extend(prefix + name for name in self._changed_files_of(wt))
        return sorted(set(files))

    def _changed_files_of(self, wt: GitWorktree) -> list[str]:
        """List files changed in *wt* vs its original branch.

        Detects both committed changes on the worktree branch and
        uncommitted changes in the worktree working tree.  When the
        worktree directory exists, runs ``git diff`` and
        ``git ls-files --others`` inside it so that uncommitted
        edits and new files are included.  Falls back to a branch-
        to-branch diff when the worktree has already been removed.

        Args:
            wt: A pending worktree.

        Returns:
            Sorted deduplicated list of relative file paths.
        """
        if not wt.original_branch:
            return []
        if wt.wt_dir.exists():
            base_ref = self._resolve_base_ref(
                str(wt.wt_dir), wt.baseline_commit, wt.original_branch,
            ) or wt.original_branch
            tracked = _git(
                "diff", "--name-only", "--no-renames", base_ref, cwd=str(wt.wt_dir),
            )
            if tracked.returncode == 0:
                files = _unquoted_name_lines(tracked.stdout)
            else:
                # The diff query failed (e.g. the original branch was
                # renamed/deleted so base_ref no longer resolves).  A
                # clean ``status --porcelain`` alone must NOT be taken
                # as "no changes": the worktree may hold COMMITTED task
                # work, and callers auto-discard the branch when this
                # returns [].  Also list files from commits unique to
                # this worktree so committed work is never mistaken for
                # a clean worktree.
                status = _git("status", "--porcelain", cwd=str(wt.wt_dir))
                files = _porcelain_paths(status.stdout, rename_both_sides=True)
                unique = _git(
                    "log", "--pretty=format:", "--name-only", "--no-renames",
                    "HEAD", "--not", f"--exclude={wt.branch}", "--branches",
                    cwd=str(wt.wt_dir),
                )
                if unique.returncode == 0:
                    files.extend(_unquoted_name_lines(unique.stdout))
            files.extend(_capture_untracked(str(wt.wt_dir)))
            return sorted(set(files))
        base_ref = self._resolve_base_ref(
            str(wt.repo_root), wt.baseline_commit, wt.original_branch,
            tip=wt.branch,
        ) or wt.original_branch
        return GitWorktreeOps._diff_name_only(
            wt.repo_root, "--no-renames", base_ref, wt.branch,
        )

    # ------------------------------------------------------------------
    # Worktree actions
    # ------------------------------------------------------------------

    @staticmethod
    def _check_worktree_busy(state: AgentState, verb: str) -> dict[str, Any] | None:
        """Return an error dict if a worktree action should be refused, else None.

        Must be called with ``_state_lock`` already held so the caller
        can atomically set ``state.is_merging = True`` before releasing
        the lock.  ``running()`` covers the run-startup window: the run
        command registers ``state.task_thread`` BEFORE starting it and
        before the worker raises ``is_task_active``, so a merge/discard
        checking only ``is_task_active`` would destroy the pending
        worktree the just-submitted task is about to retire.

        Args:
            state: The agent state to check.
            verb: Human-readable action name (e.g. ``"merging"``).

        Returns:
            Error dict with ``success: False`` when busy, otherwise ``None``.
        """
        if state.running():
            return {
                "success": False,
                "message": (
                    "A task is still running on this tab. "
                    f"Wait for it to finish (or stop it) before {verb}."
                ),
            }
        if state.merge_in_progress():
            return {
                "success": False,
                "message": (
                    "A merge or discard is already in progress "
                    f"on this tab. Wait for it to finish before {verb}."
                ),
            }
        return None

    def _handle_worktree_action(self, action: str, tab_id: str = "") -> dict[str, Any]:
        """Execute a worktree merge/discard/do-nothing action.

        Args:
            action: One of ``"merge"``, ``"discard"`` or ``"nothing"``
                (the Do-nothing button: detach from the worktree,
                leaving its branch, directory, and any uncommitted
                changes untouched on disk).  A merge whose squash
                conflicts is completed by the merge SEA
                (:func:`~kiss.agents.sorcar.merge_conflict_resolver.resolve_merge_conflict`),
                whose spend is added to the task's persisted usage.
            tab_id: The tab whose worktree to act on.

        Returns:
            Dict with ``success`` bool and ``message`` string; a
            failure the user can retry from the bar carries
            ``retryable: True``.
        """
        # The state is resolved, inspected and claimed in ONE locked
        # section: a ``closeTab`` (release the worktree) or a new
        # ``run`` (replace the state) must not slip into a gap between
        # the busy check and the claim.
        with self._state_lock:
            state = agent_state.find_by_tab(tab_id)
            if state is None:
                return {"success": False, "message": "No pending worktree changes to act on"}
            wt = state.agent
            if wt is None or not wt._wt_pending or wt._repo_root is None:
                return {"success": False, "message": "No pending worktree changes to act on"}
            verb = {"merge": "merging", "discard": "discarding", "nothing": "detaching"}.get(
                action,
            )
            if verb is None:
                return {"success": False, "message": f"Unknown action: {action}"}
            busy = self._check_worktree_busy(state, verb)
            if busy:
                return busy
            repo_root = wt._repo_root
            state.is_merging = True
            # This runs in the event loop's default executor, and the
            # task that produced the worktree is long gone, so
            # ``task_thread`` is None and the shutdown sweep of
            # in-flight tasks cannot see this work.  Publishing the
            # thread lets shutdown WAIT for the repository to stop
            # being rewritten instead of returning mid-merge.
            state.merge_thread = threading.current_thread()
        try:
            # One lock per isolated repository, taken in path order so
            # two tabs acting on the same repositories cannot deadlock
            # (the per-worktree merge/discard re-enters its own lock).
            roots = {str(w.repo_root): w.repo_root for w in _pending_worktrees(wt)}
            roots.setdefault(str(repo_root), repo_root)
            with contextlib.ExitStack() as locks:
                for _name, root in sorted(roots.items()):
                    locks.enter_context(repo_lock(root))
                if action == "nothing":
                    # ``kept: True`` tells the clients to KEEP their
                    # pending-worktree fallback directory — the
                    # worktree stays on disk, so transcript file links
                    # into it must keep resolving.  A failed marker
                    # write leaves the worktree pending (fail closed),
                    # and the failure is ``retryable`` so the webview
                    # keeps the bar's buttons.
                    try:
                        msg = wt.leave_as_is()
                    except RuntimeError as e:
                        return {"success": False, "message": str(e), "retryable": True}
                    return {"success": True, "message": msg, "kept": True}
                if action == "merge":
                    self.printer.broadcast({
                        "type": "worktree_progress",
                        "message": "Generating commit message…",
                        "tabId": tab_id,
                    })
                    logger.info(
                        "Worktree merge started: tab=%s worktree=%s thread=%s",
                        tab_id, wt._wt_dir, threading.current_thread().name,
                    )
                    started = time.monotonic()
                    usage_before = _agent_usage(wt)
                    try:
                        msg = wt.merge(conflict_resolver=resolve_merge_conflict)
                    finally:
                        # Also on a user stop that unwinds the merge
                        # agent: its spend was attributed to the live
                        # agent and must reach the row.
                        self._persist_merge_agent_usage(wt, usage_before, tab_id)
                    # Every pending worktree merged (a task touching
                    # several repositories has one per repository).
                    success = not wt._wt_pending
                    logger.info(
                        "Worktree merge finished: tab=%s success=%s elapsed=%.1fs",
                        tab_id, success, time.monotonic() - started,
                    )
                    return {"success": success, "message": msg}
                # A user-explicit Discard throws the work away as
                # asked (no ignored-file rescue).  A partial discard
                # (branch deletion failed) or a deferred one (ignored
                # output could not be rescued) must not report
                # success; a deferred discard is retryable so the
                # webview keeps the bar's buttons.
                msg = wt.discard()
                deferred = "Discard deferred" in msg
                result: dict[str, Any] = {
                    "success": "Partially discarded" not in msg and not deferred,
                    "message": msg,
                }
                if deferred:
                    result["retryable"] = True
                return result
        finally:
            with self._state_lock:
                state.is_merging = False
                state.merge_thread = None
            # A close that arrived during the merge/discard saw the
            # tab busy and deferred disposal; without this call the
            # backend tab state would leak indefinitely.
            self._dispose_if_closed(tab_id)

    def _persist_merge_agent_usage(
        self,
        agent: Any,
        usage_before: tuple[float, int, int],
        tab_id: str,
    ) -> None:
        """Charge the merge agent's spend to the task whose merge it fixed.

        The merge SEA runs after the task's history row (tokens, cost,
        steps) was persisted, as a sub-agent whose spend
        :func:`~kiss.agents.sorcar.sorcar_agent._attribute_sub_usage`
        has already added to *agent*'s live counters.  The difference
        between those counters before and after the merge is added to
        the task's row, and the new totals are pushed to the tab as a
        ``usage_info`` event — persisted on the task so a replay shows
        them, and tab-stamped so the live status row updates now.

        Args:
            agent: The worktree agent that ran the merge.
            usage_before: ``(budget, tokens, steps)`` of *agent* before
                the merge, from :func:`_agent_usage`.
            tab_id: The tab whose status row to refresh.
        """
        budget, tokens, steps = _agent_usage(agent)
        d_budget = budget - usage_before[0]
        d_tokens = tokens - usage_before[1]
        d_steps = steps - usage_before[2]
        if d_budget <= 0 and d_tokens <= 0 and d_steps <= 0:
            return
        task_id = _persisted_task_id(agent)
        if not task_id:
            return
        try:
            totals = _add_task_usage(task_id, d_tokens, d_budget, d_steps)
        except Exception:
            logger.warning(
                "Could not add merge agent usage to task %s", task_id, exc_info=True,
            )
            return
        if totals is None:
            return
        new_tokens, new_cost, new_steps = totals
        event: dict[str, Any] = {
            "type": "usage_info",
            "text": "",
            "total_tokens": new_tokens,
            "cost": f"${new_cost:.4f}",
            "cost_usd": new_cost,
            "total_steps": new_steps,
        }
        stamp_event_ts(event)
        self.printer.broadcast_transient(event, task_id=task_id, tab_id=tab_id)
        _append_chat_event(dict(event), task_id=task_id)
        self.printer.broadcast({"type": "tasks_updated"})
