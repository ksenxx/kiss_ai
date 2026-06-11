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

import json
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    _split_rename_tail,
    _unquote_git_path,
    repo_lock,
)
from kiss.agents.sorcar.persistence import _append_chat_event
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.diff_merge import (
    _capture_untracked,
    _cleanup_merge_data,
    _git,
    _merge_data_dir,
    _prepare_merge_view,
)
from kiss.agents.vscode.helpers import generate_commit_message_from_diff

if TYPE_CHECKING:
    from kiss.agents.vscode.json_printer import JsonPrinter

logger = logging.getLogger(__name__)


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
        for line in output.strip().splitlines()
        if line
    ]


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


class _MergeFlowMixin:
    """Merge-view, worktree-action, and autocommit methods."""

    if TYPE_CHECKING:
        printer: JsonPrinter
        work_dir: str
        _state_lock: threading.RLock

        def _get_tab(self, tab_id: str) -> _RunningAgentState: ...
        def _any_non_wt_running(self) -> bool: ...
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

        Returns:
            The agent stored on ``tab``, or ``None``.
        """
        return tab.agent

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
            with open(merge_json_path) as f:
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
                self.printer.broadcast(merge_data_event)
                self.printer.broadcast(merge_started_event)
            except BaseException:
                with self._state_lock:
                    if resolved_tab is not None:
                        resolved_tab.is_merging = False
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
            tab.is_merging = False
        self.printer.broadcast({"type": "merge_ended", "tabId": tab_id})
        _cleanup_merge_data(str(_merge_data_dir(tab_id)))

        self._present_pending_worktree(tab_id, try_merge_review=False)

        if not tab.use_worktree:
            self._broadcast_autocommit_prompt(tab_id, work_dir)
        # If the user closed the tab while the merge view was open,
        # dispose the now-idle _RunningAgentState.  No-op otherwise.
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
        files: list[str] = []
        for line in result.stdout.splitlines():
            if len(line) < 4:
                continue
            tail = line[3:]
            code = line[:2]
            if ("R" in code or "C" in code) and " -> " in tail:
                # Rename/copy entry: split the RAW tail on the
                # `` -> `` boundary (respecting quoting) first, then
                # unquote the new side exactly once.
                _, new_raw = _split_rename_tail(tail)
                path = _unquote_git_path(new_raw)
            else:
                path = _unquote_git_path(tail.strip())
            if path and path not in files:
                files.append(path)
        return files

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
            self.printer.broadcast({
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
        self.printer.broadcast(event)
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
            repo = GitWorktreeOps.discover_repo(Path(work_dir))
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
                _git(work_dir, "add", "-A")
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
                    prompt_tab = _RunningAgentState.running_agent_states.get(
                        tab_id,
                    )
                user_prompt = (
                    prompt_tab.last_user_prompt if prompt_tab else ""
                ) or None
                msg = (
                    generate_commit_message_from_diff(
                        diff.stdout, user_prompt=user_prompt,
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
                subject = msg.splitlines()[0] if msg.splitlines() else msg
                done_event = self._broadcast_autocommit_done(
                    tab_id, success=True, committed=True,
                    message=f"Committed: {subject}",
                    commit_message=msg,
                )
                if tab_id:
                    with self._state_lock:
                        tab = _RunningAgentState.running_agent_states.get(tab_id)
                    task_id: int | None = None
                    if tab is not None:
                        # Prefer the in-flight task id: when the
                        # "Auto commit" toggle is ON this handler runs
                        # from the task thread's post-task cleanup
                        # (``_run_task_inner``'s finally) BEFORE
                        # ``_run_task``'s outer finally refreshes
                        # ``last_task_id`` — so on a follow-up task in
                        # the same tab, ``last_task_id`` still holds
                        # the PREVIOUS task's id and the commit
                        # confirmation would be persisted into the
                        # prior task's event stream.
                        # ``task_history_id`` is the current task's
                        # row id while a task is in flight and ``None``
                        # otherwise (user-clicked prompt after task
                        # end), in which case ``last_task_id`` is
                        # already up to date.
                        task_id = tab.task_history_id
                        if task_id is None:
                            task_id = tab.last_task_id
                        if task_id is None and tab.agent is not None:
                            # Fallback for legacy callers that wire
                            # the task id onto the agent (e.g. tests
                            # that pre-seed ``tab.agent._last_task_id``
                            # without populating the new
                            # :class:`_RunningAgentState.last_task_id`
                            # field).  Production sets both in
                            # :meth:`_TaskRunnerMixin._run_task`'s
                            # outer ``finally``.
                            task_id = tab.agent._last_task_id
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
        """Emit merge review or ``worktree_done`` for a pending worktree branch.

        Worktrees are no longer associated with chat sessions, so
        there is no cross-process restoration to perform here.  This
        helper now simply delegates to :meth:`_present_pending_worktree`
        (which itself no-ops unless the tab has ``use_worktree`` set
        and its transient agent still holds a pending worktree).

        Args:
            tab_id: The tab to check for pending worktree.
        """
        self._present_pending_worktree(tab_id, try_merge_review=True)

    def _present_pending_worktree(
        self, tab_id: str, *, try_merge_review: bool,
        discard_if_empty: bool = True,
    ) -> None:
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
        - Worktree has no changes and *discard_if_empty* is True
          and no non-wt task is running: auto-discard the empty
          branch (BUG-66 — clean up stale resumed sessions and
          finished merge reviews).
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
        """
        tab = self._get_tab(tab_id)
        if not tab.use_worktree:
            return
        wt_agent = self._ensure_wt_agent(tab)
        if wt_agent is None or not wt_agent._wt_pending:
            return
        changed = self._get_worktree_changed_files(tab_id)
        if changed and try_merge_review:
            wt_dir = wt_agent._wt_dir
            if wt_dir is not None and wt_dir.exists():
                base_ref = wt_agent._baseline_commit or "HEAD"
                try:
                    if self._prepare_and_start_merge(
                        str(wt_dir), base_ref=base_ref, tab_id=tab_id,
                    ):
                        return
                except BaseException:
                    logger.debug("Worktree merge review error", exc_info=True)
        if not changed and discard_if_empty:
            with self._state_lock:
                non_wt_busy = self._any_non_wt_running()
            if not non_wt_busy:
                wt_agent.discard()
                return
        if not changed:
            # Branch is preserved (discard_if_empty=False) but the
            # worktree has no changes — there is nothing to merge,
            # so suppress the "Auto-commit and merge or Discard?"
            # prompt that the ``worktree_done`` frontend handler
            # renders unconditionally.  The branch remains in
            # ``git branch`` for manual inspection / cleanup, but
            # the user is not bothered with a meaningless prompt.
            return
        # ``changed`` is provably non-empty here: both ``not changed``
        # branches above return before reaching this point.
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
        tab = self._get_tab(tab_id)
        if not tab.use_worktree:
            return False
        wt_agent = self._ensure_wt_agent(tab)
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

        # ``--no-renames`` so a rename contributes BOTH paths to the
        # overlap sets: merging the worktree branch deletes the old
        # path from the main tree, so a dirty main-tree edit of the
        # old path must still be detected as a conflict.
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
        dirty = set(GitWorktreeOps.unstaged_files(wt.repo_root))
        dirty.update(GitWorktreeOps.staged_files(wt.repo_root))
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
        wt_agent = self._ensure_wt_agent(tab)
        if wt_agent is None or not wt_agent._original_branch:
            return []
        wt = wt_agent
        original_branch = wt._original_branch
        assert original_branch is not None  # narrowed by the check above
        wt_dir = wt._wt_dir
        if wt_dir and wt_dir.exists():
            base_ref = self._resolve_base_ref(
                str(wt_dir), wt._baseline_commit, original_branch,
            )
            # ``--no-renames`` so a rename lists BOTH the old path
            # (which the merge will delete from the main tree) and the
            # new path, instead of collapsing into the new path only.
            tracked = _git(
                str(wt_dir), "diff", "--name-only", "--no-renames", base_ref,
            )
            if tracked.returncode == 0:
                files = _unquoted_name_lines(tracked.stdout)
            else:
                status = _git(str(wt_dir), "status", "--porcelain")
                files = [
                    _unquote_git_path(line[3:].strip())
                    for line in status.stdout.splitlines()
                    if len(line) >= 4 and line[3:].strip()
                ]
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

    def _check_worktree_busy(self, tab: _RunningAgentState, verb: str) -> dict[str, Any] | None:
        """Return an error dict if a worktree action should be refused, else None.

        Checks both the tab's own task and any non-worktree task running
        on the main tree (BUG-35, BUG-72 fixes).

        Must be called with ``_state_lock`` already held (RACE-1 fix)
        so the caller can atomically set ``tab.is_merging = True``
        before releasing the lock — otherwise a non-wt task on
        another tab could pass its own ``is_merging`` guard in the
        TOCTOU window between this check returning ``None`` and the
        caller acquiring ``_state_lock`` again to set the flag.

        Args:
            tab: The per-tab state to check.
            verb: Human-readable action name (e.g. ``"merging"``).

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
                otherwise be refused by its own guard.

        Returns:
            Dict with ``success`` bool and ``message`` string.
        """
        tab = self._get_tab(tab_id)
        if not tab.use_worktree:
            return {"success": False, "message": "Worktree mode is not enabled"}
        wt_agent = self._ensure_wt_agent(tab)
        if wt_agent is None or not wt_agent._wt_pending:
            return {
                "success": False,
                "message": "No pending worktree changes to act on",
            }
        wt = wt_agent
        verb = {"merge": "merging", "discard": "discarding"}.get(action)
        if verb is None:
            return {"success": False, "message": f"Unknown action: {action}"}
        # RACE-1 / RACE-2 fix: atomically (a) verify nothing else is
        # touching the main tree, (b) claim it for this tab by setting
        # ``is_merging = True``.  Both happen under ``_state_lock`` so
        # a concurrent non-wt task-start (whose guard is also under
        # ``_state_lock``) sees the flag and refuses.  Holding
        # ``repo_lock`` for the slow body additionally serializes
        # with ``_try_setup_worktree``'s release phase and with any
        # concurrent ``_handle_worktree_action`` invocation on a
        # different tab pointed at the same repo.  ``is_merging`` is
        # set BEFORE acquiring ``repo_lock`` so the flag is visible
        # to non-wt task-start guards even when this thread is
        # currently blocked on the lock.
        repo_root = wt._repo_root
        if repo_root is None:
            return {
                "success": False,
                "message": "No pending worktree changes to act on",
            }
        with self._state_lock:
            if not internal:
                busy = self._check_worktree_busy(tab, verb)
                if busy:
                    return busy
            tab.is_merging = True
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
                return {"success": True, "message": msg}
        finally:
            with self._state_lock:
                tab.is_merging = False
