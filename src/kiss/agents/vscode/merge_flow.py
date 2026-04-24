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

from kiss.agents.sorcar.git_worktree import GitWorktreeOps, repo_lock
from kiss.agents.sorcar.persistence import _append_chat_event
from kiss.agents.vscode.diff_merge import (
    _capture_untracked,
    _cleanup_merge_data,
    _git,
    _merge_data_dir,
    _prepare_merge_view,
)
from kiss.agents.vscode.helpers import generate_commit_message_from_diff
from kiss.agents.vscode.tab_state import _TabState

if TYPE_CHECKING:
    from kiss.agents.vscode.printer import VSCodePrinter

logger = logging.getLogger(__name__)


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
        printer: VSCodePrinter
        work_dir: str
        _state_lock: threading.Lock
        _tab_states: dict[str, _TabState]

        def _get_tab(self, tab_id: str) -> _TabState: ...
        def _any_non_wt_running(self) -> bool: ...

    def _start_merge_session(
        self, merge_json_path: str, tab_id: str = "",
    ) -> bool:
        """Load merge data from disk and broadcast merge_data + merge_started events.

        Args:
            merge_json_path: Path to the pending-merge.json file.
            tab_id: Frontend tab identifier.  Used to set ``is_merging``
                on the correct tab.

        Returns:
            True if a merge session was started, False otherwise.
        """
        try:
            with open(merge_json_path) as f:
                merge_data = json.load(f)
            files = merge_data.get("files", [])
            if not files:
                return False
            total_hunks = sum(len(f.get("hunks", [])) for f in files)
            if total_hunks == 0:
                return False
            resolved_tab_id = tab_id or getattr(
                self.printer._thread_local, "tab_id", None,
            )
            resolved_tab: _TabState | None = None
            with self._state_lock:
                if resolved_tab_id is not None:
                    resolved_tab = self._tab_states.get(resolved_tab_id)
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
        return self._start_merge_session(merge_json, tab_id=tab_id)

    def _finish_merge(self, tab_id: str = "") -> None:
        """End the merge session for a specific tab.

        When a worktree task is pending, emits ``worktree_done`` so the
        user sees merge/discard buttons only after the hunk review is
        complete.

        Args:
            tab_id: The tab whose merge session is finished.  When
                falsy (*None* or empty string), the call is a no-op — a
                missing ``tabId`` at this layer indicates a frontend bug
                that should not silently tear down every tab's merge
                state.
        """
        if not tab_id:
            logger.debug("_finish_merge called without tab_id; ignoring")
            return
        with self._state_lock:
            tab = self._tab_states.get(tab_id)
            if tab is not None:
                tab.is_merging = False
        self.printer.broadcast({"type": "merge_ended", "tabId": tab_id})
        _cleanup_merge_data(str(_merge_data_dir(tab_id)))

        self._present_pending_worktree(tab_id, try_merge_review=False)

        if tab is not None and not tab.use_worktree:
            changed = self._main_dirty_files()
            if changed:
                self.printer.broadcast({
                    "type": "autocommit_prompt",
                    "tabId": tab_id,
                    "changedFiles": changed,
                })

    def _main_dirty_files(self) -> list[str]:
        """List modified, staged and untracked files in the main working tree.

        Uses ``git status --porcelain -uall`` so untracked files inside
        new directories are also reported.  Returns an empty list when
        the working tree is clean or ``work_dir`` is not a git repo.

        Returns:
            De-duplicated list of file paths (relative to ``work_dir``).
        """
        repo = GitWorktreeOps.discover_repo(Path(self.work_dir))
        if repo is None:
            return []
        result = _git(self.work_dir, "status", "--porcelain", "-uall")
        if result.returncode != 0:
            return []
        files: list[str] = []
        for line in result.stdout.splitlines():
            if len(line) < 4:
                continue
            path = line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            path = path.strip().strip('"')
            if path and path not in files:
                files.append(path)
        return files

    def _handle_autocommit_action(
        self, action: str, tab_id: str = "",
    ) -> None:
        """Process the user's reply to an ``autocommit_prompt``.

        Args:
            action: ``"commit"`` to stage-all + generate-message + commit;
                ``"skip"`` to leave the working tree untouched.
            tab_id: The tab that owns the prompt (echoed in the
                ``autocommit_done`` event).
        """
        if action == "skip":
            self.printer.broadcast({
                "type": "autocommit_done",
                "success": True,
                "committed": False,
                "message": "Left changes uncommitted.",
                "tabId": tab_id,
            })
            return
        if action != "commit":
            self.printer.broadcast({
                "type": "autocommit_done",
                "success": False,
                "committed": False,
                "message": f"Unknown autocommit action: {action}",
                "tabId": tab_id,
            })
            return
        try:
            repo = GitWorktreeOps.discover_repo(Path(self.work_dir))
            if repo is None:
                self.printer.broadcast({
                    "type": "autocommit_done",
                    "success": False,
                    "committed": False,
                    "message": "Not a git repository.",
                    "tabId": tab_id,
                })
                return
            with repo_lock(repo):
                self.printer.broadcast({
                    "type": "autocommit_progress",
                    "message": "Staging changes…",
                    "tabId": tab_id,
                })
                _git(self.work_dir, "add", "-A")
                diff = _git(self.work_dir, "diff", "--cached")
                if not diff.stdout.strip():
                    self.printer.broadcast({
                        "type": "autocommit_done",
                        "success": True,
                        "committed": False,
                        "message": "Nothing to commit.",
                        "tabId": tab_id,
                    })
                    return
                self.printer.broadcast({
                    "type": "autocommit_progress",
                    "message": "Generating commit message…",
                    "tabId": tab_id,
                })
                msg = generate_commit_message_from_diff(diff.stdout) or "Auto-commit"
                self.printer.broadcast({
                    "type": "autocommit_progress",
                    "message": "Committing…",
                    "tabId": tab_id,
                })
                ok = GitWorktreeOps.commit_staged(repo, msg)
            if ok:
                subject = msg.splitlines()[0] if msg.splitlines() else msg
                done_event: dict[str, Any] = {
                    "type": "autocommit_done",
                    "success": True,
                    "committed": True,
                    "message": f"Committed: {subject}",
                    "commitMessage": msg,
                    "tabId": tab_id,
                }
                self.printer.broadcast(done_event)
                # Persist to task history so the commit shows up in
                # session replay ("the report").
                if tab_id:
                    tab = self._tab_states.get(tab_id)
                    task_id = (
                        tab.agent._last_task_id
                        if tab is not None
                        else None
                    )
                    if task_id is not None:
                        _append_chat_event(done_event, task_id=task_id)
            else:
                self.printer.broadcast({
                    "type": "autocommit_done",
                    "success": False,
                    "committed": False,
                    "message": "git commit failed (pre-commit hook?).",
                    "tabId": tab_id,
                })
        except Exception as e:  # pragma: no cover — unexpected git/LLM error
            logger.debug("Autocommit action failed", exc_info=True)
            self.printer.broadcast({
                "type": "autocommit_done",
                "success": False,
                "committed": False,
                "message": str(e),
                "tabId": tab_id,
            })

    def _emit_pending_worktree(self, tab_id: str = "") -> None:
        """Emit merge review or ``worktree_done`` for a pending worktree branch.

        Called after replaying a session.  Restores worktree state
        from git (for post-restart resume) and delegates to
        :meth:`_present_pending_worktree`.

        Args:
            tab_id: The tab to check for pending worktree.
        """
        tab = self._get_tab(tab_id)
        if not tab.use_worktree:
            return
        self._ensure_worktree_state(tab_id)
        self._present_pending_worktree(tab_id, try_merge_review=True)

    def _present_pending_worktree(
        self, tab_id: str, *, try_merge_review: bool,
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
        - Worktree has no changes and no non-wt task is running:
          auto-discard.
        - Worktree has no changes but a non-wt task is running:
          broadcast ``worktree_done`` so the user is aware of the
          pending branch and can take manual action later
          (BUG-68 fix — previously silent for the post-task and
          post-merge-review paths).

        Args:
            tab_id: The tab with a pending worktree.
            try_merge_review: Whether to attempt starting a merge
                review before falling back.  Pass False after a
                merge review has already been completed.
        """
        tab = self._get_tab(tab_id)
        if not tab.use_worktree or not tab.agent._wt_pending:
            return
        changed = self._get_worktree_changed_files(tab_id)
        if changed and try_merge_review:
            wt_dir = tab.agent._wt_dir
            if wt_dir is not None and wt_dir.exists():
                base_ref = tab.agent._baseline_commit or "HEAD"
                try:
                    if self._prepare_and_start_merge(
                        str(wt_dir), base_ref=base_ref, tab_id=tab_id,
                    ):
                        return
                except BaseException:
                    logger.debug("Worktree merge review error", exc_info=True)
        if not changed:
            with self._state_lock:
                non_wt_busy = self._any_non_wt_running()
            if not non_wt_busy:
                tab.agent.discard()
                return
        wt = tab.agent
        event: dict[str, Any] = {
            "type": "worktree_done",
            "branch": wt._wt_branch,
            "worktreeDir": str(wt._wt_dir),
            "originalBranch": wt._original_branch,
            "changedFiles": changed,
            "hasConflict": self._check_merge_conflict(tab_id) if changed else False,
            "tabId": tab_id,
        }
        self.printer.broadcast(event)

    def _ensure_worktree_state(self, tab_id: str = "") -> None:
        """Restore agent worktree state from git if not already set.

        Discovers the repo root and calls ``_restore_from_git()`` so
        that ``merge()``/``discard()`` work even after a server process
        restart where in-memory state was lost.
        Only applicable when using the worktree agent.

        Args:
            tab_id: The tab whose worktree state to restore.
        """
        tab = self._get_tab(tab_id)
        if not tab.use_worktree:
            return
        wt = tab.agent
        repo_root = wt._repo_root
        if repo_root is None:
            repo_root = GitWorktreeOps.discover_repo(Path(self.work_dir))
            if repo_root is None:
                return
        wt._restore_from_git(repo_root)


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
        wt = tab.agent._wt
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
            str(wt.repo_root), "diff", "--name-only",
            orig_fork, wt.original_branch,
        )
        orig_files = (
            set(orig_diff.stdout.strip().splitlines())
            if orig_diff.returncode == 0 else set()
        )

        wt_diff = _git(str(wt_dir), "diff", "--name-only", wt_fork)
        wt_files = (
            set(wt_diff.stdout.strip().splitlines())
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
        wt = tab.agent
        if not wt._original_branch:
            return []
        wt_dir = wt._wt_dir
        if wt_dir and wt_dir.exists():
            base_ref = self._resolve_base_ref(
                str(wt_dir), wt._baseline_commit, wt._original_branch,
            )
            tracked = _git(str(wt_dir), "diff", "--name-only", base_ref)
            if tracked.returncode == 0:
                files = tracked.stdout.strip().splitlines()
            else:
                status = _git(str(wt_dir), "status", "--porcelain")
                files = [
                    line[3:].strip()
                    for line in status.stdout.splitlines()
                    if len(line) >= 4 and line[3:].strip()
                ]
            files.extend(_capture_untracked(str(wt_dir)))
            return sorted(set(files))
        if not wt._wt_branch:
            return []
        repo_root = str(wt._repo_root) if wt._repo_root else self.work_dir
        base_ref = self._resolve_base_ref(
            repo_root, wt._baseline_commit, wt._original_branch,
            tip=wt._wt_branch,
        )
        result = _git(repo_root, "diff", "--name-only",
                      base_ref,
                      wt._wt_branch)
        return result.stdout.strip().splitlines() if result.returncode == 0 else []

    def _check_worktree_busy(self, tab: _TabState, verb: str) -> dict[str, Any] | None:
        """Return an error dict if a worktree action should be refused, else None.

        Checks both the tab's own task and any non-worktree task running
        on the main tree (BUG-35, BUG-72 fixes).

        Args:
            tab: The per-tab state to check.
            verb: Human-readable action name (e.g. ``"merging"``).

        Returns:
            Error dict with ``success: False`` when busy, otherwise ``None``.
        """
        with self._state_lock:
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

    def _handle_worktree_action(self, action: str, tab_id: str = "") -> dict[str, Any]:
        """Execute a worktree merge/discard/manual action.

        Restores agent worktree state from git if needed (e.g. after a
        server process restart where in-memory state was lost).

        Args:
            action: One of ``"merge"`` or ``"discard"``.
            tab_id: The tab whose worktree to act on.

        Returns:
            Dict with ``success`` bool and ``message`` string.
        """
        tab = self._get_tab(tab_id)
        if not tab.use_worktree:
            return {"success": False, "message": "Worktree mode is not enabled"}
        wt = tab.agent
        if not wt._wt_pending:
            self._ensure_worktree_state(tab_id)
        if action == "merge":
            busy = self._check_worktree_busy(tab, "merging")
            if busy:
                return busy
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
        elif action == "discard":
            busy = self._check_worktree_busy(tab, "discarding")
            if busy:
                return busy
            msg = wt.discard()
            return {"success": True, "message": msg}
        return {"success": False, "message": f"Unknown action: {action}"}
