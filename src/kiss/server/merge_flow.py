# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Merge / worktree / autocommit flow mixin for the VS Code server.

Owns:
- Non-worktree merge view (prepare + start + finish + autocommit).
- Worktree lifecycle presentation (ensure, emit pending, broadcast done).
- Worktree merge/discard user actions + conflict checking.

Split out of ``server.py`` for organisation.
"""

from __future__ import annotations

import enum
import json
import logging
import os
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
from kiss.agents.sorcar.running_agent_state import (
    _RunningAgentState,
    _tab_busy,
)
from kiss.agents.sorcar.useful_tools import _stale_worktree_fallback
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server.diff_merge import (
    _capture_untracked,
    _cleanup_merge_data,
    _git,
    _merge_data_dir,
    _prepare_merge_view,
)
from kiss.server.helpers import (
    generate_commit_message_from_diff,
    tab_busy_with_other_task,
)

if TYPE_CHECKING:
    from kiss.server.json_printer import JsonPrinter

logger = logging.getLogger(__name__)


def _tab_task_key(tab: _RunningAgentState | None) -> str | None:
    """Return the task id *tab* last ran, preferring the live agent's.

    Args:
        tab: The per-tab state to inspect, or ``None``.

    Returns:
        The task id, or ``None`` when the tab never ran a task.
    """
    if tab is None:
        return None
    task_id = tab.task_history_id
    if task_id is None:
        task_id = tab.last_task_id
    if task_id is None and tab.agent is not None:
        task_id = tab.agent._last_task_id
    return task_id


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
    writing into it, or a merge review the user is part-way through —
    so the caller must leave it completely alone rather than fall back
    to presenting a review (which could also auto-discard an empty
    branch out from under a running task).
    """

    FINALIZED = "finalized"
    """The worktree's fate was decided here (merged or discarded)."""

    PRESENT = "present"
    """Nothing was done; the caller should offer the merge review.

    The caller does **not** own the worktree: the tab holds no pending
    branch, or is not in worktree mode at all, so presenting is itself
    harmless (and usually a no-op).
    """

    PRESENT_CLAIMED = "present_claimed"
    """Like :attr:`PRESENT`, but ``is_merging`` was claimed for the caller.

    Returned when the tab really does hold a pending worktree that the
    user must be shown.  The claim is taken in the same locked section
    that observed the worktree free, so a second resume arriving at the
    same moment is turned away instead of opening a duplicate review.
    The caller owns the flag and must release it — see
    :meth:`_MergeFlowMixin._release_present_claim` — unless
    :meth:`_MergeFlowMixin._present_pending_worktree` reports that a
    review took it over.
    """

    NOOP = "noop"
    """Another owner holds the worktree; the caller must not touch it."""


class _MergeFlowMixin:
    """Merge-view, worktree-action, and autocommit methods."""

    if TYPE_CHECKING:
        printer: JsonPrinter
        work_dir: str
        _state_lock: threading.RLock

        def _get_tab(self, tab_id: str) -> _RunningAgentState: ...
        def _any_non_wt_running(
            self, repo_root: Path | None = None,
        ) -> bool: ...
        def _dispose_if_closed(self, tab_id: str) -> None: ...

    def _ensure_wt_agent(
        self, tab: _RunningAgentState,
    ) -> WorktreeSorcarAgent | None:
        """Return the worktree-aware agent for *tab*, or ``None``.

        Worktrees are no longer associated with chat sessions: every
        ``run()`` call mints a fresh branch and there is no
        cross-process restoration.  This method therefore returns
        ``tab.agent`` as-is — it is the only authoritative source of
        worktree state.  When the agent has been disposed (task ended
        and the tab released its reference) there is no worktree to
        operate on and ``None`` is returned.

        The merge-flow call sites read ``tab.agent`` directly; the
        only remaining caller is ``server.py`` (``_replay_session``).

        Returns:
            The agent stored on ``tab``, or ``None``.
        """
        return tab.agent

    def _open_ui_mirror(self, tab_id: str, work_dir: str = "") -> None:
        """Mirror *tab_id*'s interactive UI onto the other clients' tabs.

        The merge review, the auto-commit prompt and the worktree
        merge/discard strip all block the user until they are answered,
        so every tab showing the same chat must show them — a question
        the user cannot see in the window they happen to be looking at
        is a question they cannot answer.  Tab ids are per-client, so
        "the same chat elsewhere" means the tabs subscribed to this
        tab's task.  A co-subscriber that has since started a task of
        its own is skipped: the subscriber set of a finished task is
        deliberately retained, and that tab's UI belongs to its own
        task now.

        Args:
            tab_id: The tab that owns the UI.
            work_dir: The owner's working directory, remembered so an
                action arriving from a viewer is applied to the
                owner's repository rather than the viewer's folder.
        """
        if not tab_id:
            return
        viewers: list[str] = []
        with self._state_lock:
            task_id = _tab_task_key(
                _RunningAgentState.running_agent_states.get(tab_id),
            )
            task_key = self.printer._coerce_task_id(task_id)
            if task_key:
                for viewer_tab_id in self.printer._fanout_targets(task_key):
                    viewer = _RunningAgentState.running_agent_states.get(
                        viewer_tab_id,
                    )
                    if viewer is not None and tab_busy_with_other_task(
                        viewer, task_key,
                    ):
                        continue
                    viewers.append(viewer_tab_id)
        self.printer.open_ui_mirror(
            tab_id, viewers, work_dir or self.work_dir, task_key,
        )

    def _start_merge_session(
        self, merge_json_path: str, tab_id: str = "", work_dir: str = "",
    ) -> bool:
        """Load merge data from disk and broadcast merge_data + merge_started events.

        Args:
            merge_json_path: Path to the pending-merge.json file.
            tab_id: Frontend tab identifier.  Used to set ``is_merging``
                on the correct tab.
            work_dir: The repository (or worktree) directory this merge
                review operates on.  Stamped into the ``merge_data``
                payload as ``work_dir`` so the shared ``kiss-web`` daemon
                can echo it back on the ``all-done`` ``mergeAction`` and
                run the post-merge dirty-file scan against the tab's own
                repository rather than the daemon-wide ``self.work_dir``.
                Falls back to ``self.work_dir`` when empty.

        Returns:
            True if a merge session was started, False otherwise.
        """
        try:
            with open(merge_json_path, encoding="utf-8") as f:
                merge_data = json.load(f)
            merge_data["work_dir"] = work_dir or self.work_dir
            files = merge_data.get("files", [])
            if not files:
                return False
            total_hunks = sum(len(f.get("hunks", [])) for f in files)
            if total_hunks == 0:
                return False
            resolved_tab_id = tab_id or None
            resolved_tab: _RunningAgentState | None = None
            with self._state_lock:
                if resolved_tab_id is not None:
                    resolved_tab = _RunningAgentState.running_agent_states.get(resolved_tab_id)
                    if resolved_tab is not None:
                        resolved_tab.is_merging = True
            try:
                merge_data_event: dict[str, Any] = {
                    "type": "merge_data",
                    "data": merge_data,
                    "hunk_count": total_hunks,
                }
                merge_started_event: dict[str, Any] = {"type": "merge_started"}
                if resolved_tab_id is not None:
                    merge_data_event["tabId"] = resolved_tab_id
                    merge_started_event["tabId"] = resolved_tab_id
                    self._open_ui_mirror(
                        resolved_tab_id, merge_data["work_dir"],
                    )
                self.printer.broadcast_tab_ui(merge_data_event)
                self.printer.broadcast_tab_ui(merge_started_event)
            except BaseException:
                with self._state_lock:
                    if resolved_tab is not None:
                        resolved_tab.is_merging = False
                if resolved_tab_id is not None:
                    self.printer.close_ui_mirror(resolved_tab_id)
                raise
            return True
        except (OSError, json.JSONDecodeError, KeyError):
            logger.debug("Failed to load merge data", exc_info=True)
            return False

    def _prepare_and_start_merge(
        self,
        work_dir: str,
        pre_hunks: dict[str, list[tuple[int, int, int, int]]] | None = None,
        pre_untracked: set[str] | None = None,
        pre_file_hashes: dict[str, str] | None = None,
        base_ref: str = "HEAD",
        tab_id: str = "",
    ) -> bool:
        """Prepare a merge view and start the merge session if changes exist.

        Combines ``_prepare_merge_view`` and ``_start_merge_session``
        into a single call to eliminate the repeated prepare→check→start
        sequence.

        Args:
            work_dir: Repository root (or worktree) directory.
            pre_hunks: Pre-task diff hunks (empty dict when not applicable).
            pre_untracked: Pre-task untracked file set (empty when not applicable).
            pre_file_hashes: Pre-task MD5 hashes for change detection.
            base_ref: Git ref to diff against (default ``"HEAD"``).
                Pass a baseline commit SHA to include committed agent
                changes in the merge review.
            tab_id: Frontend tab identifier for per-tab merge data isolation.

        Returns:
            True if a merge session was started, False otherwise.
        """
        merge_dir = str(_merge_data_dir(tab_id))
        merge_result = _prepare_merge_view(
            work_dir,
            merge_dir,
            pre_hunks or {},
            pre_untracked or set(),
            pre_file_hashes,
            base_ref=base_ref,
        )
        if merge_result.get("status") != "opened":
            return False
        merge_json = os.path.join(merge_dir, "pending-merge.json")
        return self._start_merge_session(
            merge_json, tab_id=tab_id, work_dir=work_dir,
        )

    def _finish_merge(self, tab_id: str = "", *, work_dir: str = "") -> None:
        """End the merge session for a specific tab.

        When a worktree task is pending, emits ``worktree_done`` so the
        user sees merge/discard buttons only after the hunk review is
        complete.

        Uses ``_get_tab`` to obtain the tab so the autocommit-prompt
        check still fires even when the ``mergeAction`` command was
        routed to a process that never ran the original task (e.g. the
        service process after the task process was disposed).

        Args:
            tab_id: The tab whose merge session is finished.  When
                falsy (*None* or empty string), the call is a no-op — a
                missing ``tabId`` at this layer indicates a frontend bug
                that should not silently tear down every tab's merge
                state.
            work_dir: The tab's working directory.  Forwarded to
                :meth:`_broadcast_autocommit_prompt` so the post-merge
                dirty-file scan runs against the tab's own repository
                rather than the daemon-wide ``self.work_dir``.  Falls
                back to ``self.work_dir`` when empty.
        """
        if not tab_id:
            logger.debug("_finish_merge called without tab_id; ignoring")
            return
        tab = self._get_tab(tab_id)
        with self._state_lock:
            tab.is_merging = True
        try:
            _cleanup_merge_data(str(_merge_data_dir(tab_id)))

            self._present_pending_worktree(tab_id, try_merge_review=False)

            if not tab.use_worktree:
                self._broadcast_autocommit_prompt(tab_id, work_dir)
        finally:
            with self._state_lock:
                tab.is_merging = False
            try:
                self.printer.broadcast_tab_ui(
                    {"type": "merge_ended", "tabId": tab_id}
                )
            except Exception:
                logger.debug(
                    "merge_ended broadcast failed for tab %s",
                    tab_id,
                    exc_info=True,
                )
            # Inside the finally (F4-30): an exception from
            # _present_pending_worktree / _broadcast_autocommit_prompt
            # must not skip the deferred disposal of a tab that was
            # closed during the merge — no later lifecycle transition
            # would ever dispose it.
            self._dispose_if_closed(tab_id)

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

    def _broadcast_autocommit_prompt(
        self, tab_id: str, work_dir: str = "",
    ) -> None:
        """Broadcast an ``autocommit_prompt`` if the main tree has dirty files.

        Shared by ``_finish_merge`` (after merge review ends) and
        ``_run_task_inner`` (when no merge view was opened).

        Args:
            tab_id: Frontend tab identifier to include in the event.
            work_dir: The tab's working directory.  Forwarded to
                :meth:`_main_dirty_files` so the dirty-file scan runs
                against the tab's own repository rather than the
                daemon-wide ``self.work_dir``.  Falls back to
                ``self.work_dir`` when empty.
        """
        changed = self._main_dirty_files(work_dir)
        if changed:
            self._open_ui_mirror(tab_id, work_dir)
            self.printer.broadcast_tab_ui({
                "type": "autocommit_prompt",
                "tabId": tab_id,
                "changedFiles": changed,
            })

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
        self.printer.broadcast_tab_ui(event)
        return event

    def _handle_autocommit_action(
        self, action: str, tab_id: str = "", *, work_dir: str = "",
    ) -> None:
        """Process the user's reply to an ``autocommit_prompt``.

        Args:
            action: ``"commit"`` to stage-all + generate-message + commit;
                ``"skip"`` to leave the working tree untouched.
            tab_id: The tab that owns the prompt (echoed in the
                ``autocommit_done`` event).
            work_dir: The tab's working directory.  Preferred over the
                daemon-wide ``self.work_dir`` because the shared
                ``kiss-web`` daemon may have been launched from (or
                synced to) a different — possibly non-git — folder than
                the window that owns this tab.  Falls back to
                ``self.work_dir`` when empty.
        """
        work_dir = work_dir or self.work_dir
        if action == "skip":
            self._broadcast_autocommit_done(
                tab_id, success=True, committed=False,
                message="Left changes uncommitted.",
            )
            return
        if action != "commit":
            self._broadcast_autocommit_done(
                tab_id, success=False, committed=False,
                message=f"Unknown autocommit action: {action}",
            )
            return
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
                self.printer.broadcast_tab_ui({
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
                self.printer.broadcast_tab_ui({
                    "type": "autocommit_progress",
                    "message": "Generating commit message…",
                    "tabId": tab_id,
                })
                with self._state_lock:
                    prompt_tab = _RunningAgentState.running_agent_states.get(
                        tab_id,
                    )
                user_prompt = (
                    prompt_tab.last_user_prompt if prompt_tab else ""
                ) or None
                task_result = (
                    prompt_tab.last_result_summary if prompt_tab else ""
                ) or None
                msg = (
                    generate_commit_message_from_diff(
                        diff.stdout,
                        user_prompt=user_prompt,
                        task_result=task_result,
                    )
                    or "Auto-commit"
                )
                self.printer.broadcast_tab_ui({
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
                        task_id = _tab_task_key(
                            _RunningAgentState.running_agent_states.get(tab_id),
                        )
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
          silently via :meth:`_handle_worktree_action`.  Presenting a
          hunk-by-hunk review here is the defect this branch fixes: a
          post-task auto-merge that could not complete (conflict,
          rejected pre-commit hook, ``git stash`` failure, ...) leaves
          ``_wt_pending`` raised, and the next history click would pop
          the diff/merge UI even though auto-commit was on.
        * **auto-commit OFF** — delegate to
          :meth:`_present_pending_worktree`, which starts the merge
          review the user explicitly opted into.

        Either way the call no-ops unless the tab has ``use_worktree``
        set and its transient agent still holds a pending worktree.

        Auto-commit ON does *not* always finalize.  Two owners outrank
        the toggle and make this a complete no-op — neither finalizing
        nor presenting:

        * a merge review already in flight (F4-20): a session replay
          must not regenerate the merge view and replace the registered
          merge state, which would erase the user's accepted/rejected
          hunk resolutions mid-review;
        * a task still running on the tab: its agent is writing into
          the worktree right now.

        A third exception — the agent's ``_pending_review`` flag, set
        for a task that failed or was stopped — declines the silent
        finalize but still shows the review, because unverified work
        must never be merged behind the user's back.  That case comes
        back as :attr:`_PendingOutcome.PRESENT_CLAIMED`, carrying the
        ownership claim the review is opened under; plain
        :attr:`_PendingOutcome.PRESENT` means there was nothing to own.

        Args:
            tab_id: The tab to check for pending worktree.
        """
        outcome = self._finalize_pending_worktree(tab_id)
        if outcome is _PendingOutcome.PRESENT:
            self._present_pending_worktree(tab_id, try_merge_review=True)
            return
        if outcome is not _PendingOutcome.PRESENT_CLAIMED:
            return
        # The claim is ours, so it is ours to release — unless the
        # review took it over, in which case it stays raised until the
        # user finishes.  The `finally` matters: an exception must not
        # leave the tab permanently busy.
        review_started = False
        try:
            review_started = self._present_pending_worktree(
                tab_id, try_merge_review=True,
            )
        finally:
            if not review_started:
                self._release_present_claim(tab_id)

    def _release_present_claim(self, tab_id: str) -> None:
        """Drop the ownership claim taken for a merge review that never began.

        :meth:`_finalize_pending_worktree` claims ``is_merging`` before
        returning :attr:`_PendingOutcome.PRESENT_CLAIMED` so that only
        one resume can open the review.  When no review started the claim
        must go again, or the tab stays busy forever and every later
        task, merge and discard on it is refused.

        Only the caller that took the claim may drop it, and only when
        :meth:`_present_pending_worktree` reported that nothing took it
        over: a review that really started keeps the flag raised until
        the user finishes it (:meth:`_start_merge_session` sets it,
        :meth:`_finish_merge` clears it), and clearing it here would let
        a task start on top of a live merge view.

        Args:
            tab_id: The tab whose speculative claim to release.
        """
        with self._state_lock:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is None:
                return
            tab.is_merging = False
        self._dispose_if_closed(tab_id)

    def _finalize_pending_worktree(self, tab_id: str) -> _PendingOutcome:
        """Merge or discard a pending worktree without asking the user.

        The auto-commit counterpart of :meth:`_present_pending_worktree`,
        and the same decision the post-task fast path in
        ``_run_task_inner`` makes: merge when the branch carries
        changes, discard when it does not.

        Returns :attr:`_PendingOutcome.NOOP` — the worktree already has
        an owner, so the caller must leave it entirely alone — when:

        * a merge review is already in flight (F4-20): regenerating it
          would erase the user's accepted/rejected hunk resolutions;
        * a task is still active on the tab.  Unlike the post-task
          finalize — which runs on the very thread that owns
          ``is_task_active`` — a history click is an unrelated thread,
          and merging or discarding a worktree the agent is still
          writing into would corrupt or delete its work.  Falling back
          to the review is just as unsafe: it snapshots a half-written
          tree, and its ``discard_if_empty`` path would delete a branch
          the running task has not committed to yet;
        * a task has been *submitted* but has not reached its worker
          yet.  Ownership is therefore decided by the one shared
          :func:`_tab_busy` predicate rather than by reading the two
          flags directly: during that startup window both of them read
          False, and claiming ``is_merging`` there makes the worker
          refuse the run the user just typed ("Cannot run a task while
          merge review is in progress").

        Returns :attr:`_PendingOutcome.PRESENT` — nothing was done, and
        the caller may offer the merge review without owning anything —
        when the tab is not in worktree mode, or holds no pending
        worktree.  Presenting is itself a no-op then, so no claim is
        taken and none may be released.

        Returns :attr:`_PendingOutcome.PRESENT_CLAIMED` — the caller
        should offer the merge review and has been handed the
        ``is_merging`` claim to do it under — when a pending worktree
        really is there but must not be finalized silently:

        * auto-commit is off, so the user asked to be shown the diff;
        * the agent is in ``_pending_review`` state.  ``_run_task_inner``
          raises that flag for a task that failed or was stopped, and
          :meth:`WorktreeSorcarAgent._preserve_pending_worktree_for_review`
          documents the contract it encodes: incomplete, unverified work
          stays on its ``kiss/wt-*`` branch and is never merged into the
          user's branch behind their back.  Auto-commit means "do not
          interrupt me", not "publish work that never finished".

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
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is None or not tab.use_worktree:
                return _PendingOutcome.PRESENT
            if _tab_busy(tab):
                return _PendingOutcome.NOOP
            wt_agent = tab.agent
            if wt_agent is None or not wt_agent._wt_pending:
                return _PendingOutcome.PRESENT
            if wt_agent._pending_review or not tab.auto_commit_mode:
                # The caller will open the merge review.  Claim the
                # worktree here too: `_present_pending_worktree` starts
                # an unguarded review, so without a claim taken in the
                # same locked section that observed `is_merging` clear,
                # two simultaneous resumes both reach
                # `_prepare_and_start_merge` and broadcast a merge view
                # apiece for one branch (F4-20).
                tab.is_merging = True
                return _PendingOutcome.PRESENT_CLAIMED
            # Claim the worktree before releasing the lock so a
            # concurrent resume (remote commands run on a thread pool)
            # cannot finalize the same branch twice.  The claim is held
            # continuously across BOTH the changed-files probe and the
            # action it selects: dropping it in between would reopen
            # the very race it exists to close, and would also let the
            # probe's answer go stale before it is acted on.
            tab.is_merging = True
        try:
            changed = self._get_worktree_changed_files(tab_id)
            action = "merge" if changed else "discard"
            result = self._handle_worktree_action(
                action, tab_id, already_claimed=True,
            )
        finally:
            with self._state_lock:
                tab.is_merging = False
            self._dispose_if_closed(tab_id)
        self.printer.broadcast_tab_ui(
            {"type": "worktree_result", "tabId": tab_id, **result},
        )
        return _PendingOutcome.FINALIZED

    def _present_pending_worktree(
        self, tab_id: str, *, try_merge_review: bool,
        discard_if_empty: bool = True,
    ) -> bool:
        """Auto-discard, start merge review, or emit ``worktree_done``.

        Single source of truth for post-task / post-merge-review /
        session-resume handling of a pending worktree (RED-10 fix).

        Behavior:
        - No pending worktree: return.
        - Worktree has changed files and *try_merge_review* is True:
          attempt to start a merge review; on failure broadcast
          ``worktree_done``.
        - Worktree has changed files and *try_merge_review* is False
          (merge review already finished): broadcast ``worktree_done``.
        - Worktree has no changes and *discard_if_empty* is True:
          auto-discard the empty branch (BUG-66 — clean up stale
          resumed sessions and finished merge reviews).  A
          concurrent non-worktree task does not block this: an
          empty discard never touches the main working tree.
        - Worktree has no changes and *discard_if_empty* is False:
          preserve the branch and broadcast ``worktree_done``.
          The post-task path passes ``discard_if_empty=False``
          when the user opted into the worktree workflow but has
          not explicitly chosen to merge or discard yet — so the
          branch must remain visible in ``git branch`` for manual
          inspection / merge / discard (fixes the user-reported
          "worktree branch is not getting created" symptom in
          ``use_worktree=True`` + ``autoCommit=False`` mode).

        Args:
            tab_id: The tab with a pending worktree.
            try_merge_review: Whether to attempt starting a merge
                review before falling back.  Pass False after a
                merge review has already been completed.
            discard_if_empty: When True (default), auto-discard the
                branch if no files changed.  Post-task callers should
                pass False to preserve the branch for manual action.

        Returns:
            True when a merge review was started and now owns the tab —
            it holds ``is_merging`` until the user finishes it.  False
            in every other case, including the ones that do nothing at
            all.  Callers that claimed the tab before calling must
            release the claim when this is False; see
            :meth:`_emit_pending_worktree`.
        """
        with self._state_lock:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
        if tab is None or not tab.use_worktree:
            return False
        wt_agent = tab.agent
        if wt_agent is None or not wt_agent._wt_pending:
            return False
        changed = self._get_worktree_changed_files(tab_id)
        if changed and try_merge_review:
            wt_dir = wt_agent._wt_dir
            if wt_dir is not None and wt_dir.exists():
                # Resolve the fork point exactly like
                # _get_worktree_changed_files does (F4-22): a plain
                # "HEAD" fallback omits changes the agent already
                # COMMITTED in the worktree from the hunk review.
                base_ref = self._resolve_base_ref(
                    str(wt_dir),
                    wt_agent._baseline_commit,
                    wt_agent._original_branch or "HEAD",
                )
                try:
                    if self._prepare_and_start_merge(
                        str(wt_dir), base_ref=base_ref, tab_id=tab_id,
                    ):
                        return True
                except BaseException:
                    logger.debug("Worktree merge review error", exc_info=True)
        if not changed and discard_if_empty:
            # Discarding an EMPTY worktree removes its directory and
            # its unmerged branch without touching the main working
            # tree, so a concurrent non-worktree task is no reason to
            # skip it — skipping leaks the worktree forever because
            # nothing ever retries.
            with self._state_lock:
                prev_merging = tab.is_merging
                tab.is_merging = True
            try:
                wt_agent.discard()
            finally:
                with self._state_lock:
                    tab.is_merging = prev_merging
                # A close that arrived during the discard saw the
                # tab busy and deferred disposal; nothing later
                # would dispose it (F4-29).
                self._dispose_if_closed(tab_id)
            return False
        if not changed:
            return False
        event: dict[str, Any] = {
            "type": "worktree_done",
            "branch": wt_agent._wt_branch,
            "worktreeDir": str(wt_agent._wt_dir),
            "originalBranch": wt_agent._original_branch,
            "changedFiles": changed,
            "hasConflict": self._check_merge_conflict(tab_id),
            "tabId": tab_id,
        }
        self._open_ui_mirror(tab_id, str(wt_agent._wt_dir))
        self.printer.broadcast_tab_ui(event)
        return False

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
        tab = self._get_tab(tab_id)
        if not tab.use_worktree:
            return False
        wt_agent = tab.agent
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
            if self._any_non_wt_running(wt.repo_root):
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
        tab = self._get_tab(tab_id)
        if not tab.use_worktree:
            return []
        wt_agent = tab.agent
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

    def _check_worktree_busy(
        self,
        tab: _RunningAgentState,
        verb: str,
        repo_root: Path | None = None,
        wt_dir: Path | None = None,
    ) -> dict[str, Any] | None:
        """Return an error dict if a worktree action should be refused, else None.

        Checks both the tab's own task and any non-worktree task running
        on the main tree of *repo_root* (BUG-35, BUG-72 fixes), or
        inside the pending worktree *wt_dir* itself.

        Must be called with ``_state_lock`` already held (RACE-1 fix)
        so the caller can atomically set ``tab.is_merging = True``
        before releasing the lock — otherwise a non-wt task on
        another tab could pass its own ``is_merging`` guard in the
        TOCTOU window between this check returning ``None`` and the
        caller acquiring ``_state_lock`` again to set the flag.

        Args:
            tab: The per-tab state to check.
            verb: Human-readable action name (e.g. ``"merging"``).
            repo_root: The main repository root the action would
                stash/checkout/merge.  Non-worktree tasks running in a
                different repository (or in no repository at all) do
                not occupy this main tree and therefore do not block
                the action.  ``None`` falls back to the conservative
                "any non-worktree task blocks" behavior.
            wt_dir: The pending worktree directory the action would
                remove.  A non-worktree task running *inside* it (its
                ``git rev-parse --show-toplevel`` is the linked
                worktree itself — e.g. a sub-task submitted through
                the daemon API with the parent's worktree as
                ``work_dir``) does not touch the main tree, but both
                merge and discard delete this directory out from under
                that running task, so it must block too.

        Returns:
            Error dict with ``success: False`` when busy, otherwise ``None``.
        """
        if tab.is_task_active:
            return {
                "success": False,
                "message": (
                    f"A worktree task is still running on this tab. "
                    f"Wait for it to finish (or stop it) before {verb}."
                ),
            }
        if tab.is_merging:
            return {
                "success": False,
                "message": (
                    "A merge or merge review is already in progress "
                    f"on this tab. Wait for it to finish before {verb}."
                ),
            }
        if wt_dir is not None and self._any_non_wt_running(wt_dir):
            return {
                "success": False,
                "message": (
                    "Another tab is running a task inside this "
                    f"task's worktree. Wait for it to finish before {verb}."
                ),
            }
        if self._any_non_wt_running(repo_root):
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
                ``tab.is_task_active = True`` and therefore would
                otherwise be refused by its own guard.  A concurrent
                non-worktree task on the main tree still blocks a
                ``"merge"`` — but never a ``"discard"``, which does
                not touch the main working tree.
            already_claimed: When True, the caller has already set
                ``tab.is_merging`` under ``_state_lock`` after checking
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
        tab = self._get_tab(tab_id)
        if not tab.use_worktree:
            return {"success": False, "message": "Worktree mode is not enabled"}
        wt_agent = tab.agent
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
                busy = self._check_worktree_busy(tab, verb, repo_root, wt._wt_dir)
                if busy:
                    return busy
            elif action == "merge" and (
                self._any_non_wt_running(repo_root)
                or (wt._wt_dir is not None and self._any_non_wt_running(wt._wt_dir))
            ):
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
                tab.is_merging = True
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
                    self.printer.broadcast_tab_ui(progress_event)
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
                    tab.is_merging = False
                # A close that arrived during the merge/discard saw the
                # tab busy and deferred disposal; without this call the
                # backend tab state would leak indefinitely (F4-23).
                self._dispose_if_closed(tab_id)
