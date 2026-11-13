# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Worktree review flow and manual commit mixin for the backend server.

Owns:
- Presentation of a worktree a task left pending (it was stopped, or
  the agent never called ``commit_and_merge``): the ``worktree_done``
  bar with Merge / Discard / Do nothing.
- The worktree merge / discard / detach user actions.
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

from kiss.agents.sorcar.commit_message import generate_commit_message_from_diff
from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    _git,
    _porcelain_entries,
    _reclaim_process_lock,
    repo_lock,
    repo_locks,
)
from kiss.agents.sorcar.merge_conflict_resolver import resolve_merge_conflict
from kiss.agents.sorcar.sorcar_agent import (
    _agent_usage,
    _commit_subject,
    _persisted_task_id,
)
from kiss.agents.sorcar.useful_tools import remap_vanished_worktree
from kiss.server import agent_state, sidebar_panels
from kiss.server.agent_state import AgentState
from kiss.server.task_update import charge_side_channel_usage

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


def _git_paths(cwd: str, command: str, *args: str) -> list[str] | None:
    """List the paths git *command* prints with ``-z`` in *cwd* (None when git fails).

    NUL-separated, as :meth:`GitWorktreeOps._diff_name_only` lists:
    git quotes no ``-z`` path, so a name with a double quote, a
    backslash, a control character or an edge space comes back
    byte-exact, equal to the real on-disk path.

    Args:
        cwd: The directory to run git in.
        command: The listing subcommand (``diff``, ``ls-files``, ``log``).
        *args: Its other arguments.

    Returns:
        The relative paths in output order, or None on a git failure.
    """
    result = _git(command, "-z", *args, cwd=cwd)
    if result.returncode != 0:
        return None
    return [path for path in result.stdout.split("\0") if path]


def _stage_all(work_path: Path) -> tuple[str, str]:
    """Run ``git add -A`` in *work_path* and identify the commit it would make.

    Returns:
        ``(snapshot, "")`` where *snapshot* is ``"<HEAD> <index tree>"``:
        the object ids ``git rev-parse HEAD`` (empty on an unborn branch)
        and ``git write-tree`` print.  Two equal snapshots denote the same
        commit, parent and content alike, however large the change; the
        bounded diff text handed to the LLM cannot tell that, as a change
        past its cap leaves it unchanged.  ``("", <reason>)`` when staging
        failed.
    """
    add_result = _git("add", "-A", cwd=str(work_path))
    if add_result.returncode != 0:
        return "", _git_error_line(add_result, "git add failed")
    tree_result = _git("write-tree", cwd=str(work_path))
    if tree_result.returncode != 0:
        return "", _git_error_line(tree_result, "git write-tree failed")
    head = _git("rev-parse", "--verify", "-q", "HEAD", cwd=str(work_path)).stdout.strip()
    return f"{head} {tree_result.stdout.strip()}", ""


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


def _flatten_changed(
    listing: list[tuple[GitWorktree, list[str] | None]], primary: GitWorktree | None,
) -> list[str]:
    """Merge per-worktree file lists into one.

    Files of *primary* (the worktree the task's work dir is in) stay
    relative to it; files of any other worktree are prefixed with that
    repository's root.  A worktree whose files could not be listed
    (``None``, see :meth:`_MergeFlowMixin._changed_files_of`)
    contributes nothing; callers that may discard check for it with
    :func:`_unreadable_worktrees` first.
    """
    files: list[str] = []
    for wt, names in listing:
        prefix = "" if wt is primary else f"{wt.repo_root}/"
        files.extend(prefix + name for name in names or ())
    return sorted(set(files))


def _unreadable_worktrees(
    listing: list[tuple[GitWorktree, list[str] | None]],
) -> list[GitWorktree]:
    """The worktrees of *listing* whose changed files git could not enumerate."""
    return [wt for wt, names in listing if names is None]


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
        reply_to: str = "",
    ) -> None:
        """Report the outcome of a manual Git Commit.

        The outcome is a toast ``notification`` — info on success,
        error on failure — plus the ``autocommit_done`` event (flagged
        ``manual``) that re-arms the Git Commit button; the page skips
        its own chat rendering of a successful manual commit, while a
        failure also shows its reason in the chat.

        Args:
            tab_id: Frontend tab identifier.
            success: Whether the commit succeeded.
            committed: Whether a commit was actually created.
            message: Human-readable status message.
            commit_message: Full commit message (only when committed).
            work_dir: The working directory the commit targeted
                (carried as ``workDir``).
            reply_to: The connection id of a daemon-link peer whose
                forwarded Git Commit this is: the event goes to that
                connection alone, as the reply to its request, and
                nothing is broadcast here.
        """
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
        if reply_to:
            # Another daemon's forwarded commit: the outcome is its
            # reply alone, and that daemon tells its own clients (a
            # tab of its bound to this machine relays this daemon's
            # tab-stamped events, so a broadcast here would reach them
            # a second time).
            self.printer.broadcast({**event, "connId": reply_to})
            return
        self._commit_toast(tab_id, "info" if success else "error", message)
        self.printer.broadcast(event)

    def _commit_toast(self, tab_id: str, severity: str, message: str) -> None:
        """Show *message* as *tab_id*'s Git Commit toast on every surface.

        Every toast of one commit ("Auto-generating commit message…",
        "Committing on <machine>…", the outcome) has the same stable
        id, so each replaces the one before in place (the page dedups
        toasts by id) instead of stacking.

        Args:
            tab_id: The tab that pressed Git Commit.
            severity: ``"info"`` or ``"error"``.
            message: The toast's text.
        """
        self.printer.broadcast({
            "type": "notification",
            "id": f"manual-commit:{tab_id}",
            "severity": severity,
            "message": message,
            "tabId": tab_id,
        })

    def _commit_main_tree(self, tab_id: str, work_dir: str = "", reply_to: str = "") -> None:
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
            reply_to: The connection id of a daemon-link peer that
                forwarded the command (another daemon's tab picked
                onto this machine), or ``""``; see
                :meth:`_broadcast_autocommit_done`.
        """
        requested_dir = work_dir or self.work_dir
        try:
            work_path = remap_vanished_worktree(Path(requested_dir))
            repo = GitWorktreeOps.discover_repo(work_path)
            if repo is None:
                self._broadcast_autocommit_done(
                    tab_id, success=False, committed=False,
                    message="Not a git repository.", work_dir=requested_dir, reply_to=reply_to,
                )
                return
            # The two locks a merge holds for its whole transaction (the
            # repository's ``repo_lock``, then its cross-process flock,
            # the same from any of its checkouts) serialize this with the
            # other git users of the repository, in this daemon and in any
            # other Sorcar process (a new task's worktree setup, Source
            # Control actions, Merge / Discard, an orphan reclaim).  The
            # message generation may take minutes, so they are held only
            # around the git steps: stage + read the diff, then stage
            # again + commit once the staged tree and parent are still
            # the ones the message describes.
            with repo_lock(repo), _reclaim_process_lock(repo):
                snapshot, error = _stage_all(work_path)
                # Only the index says "clean": a patch git cannot print
                # (a missing textconv program) is empty too, and is
                # committed under the generator's fallback subject.
                has_changes = not error and GitWorktreeOps.has_staged_changes(work_path)
                diff_text = GitWorktreeOps.staged_diff(work_path) if has_changes else ""
            if error:
                self._broadcast_autocommit_done(
                    tab_id, success=False, committed=False,
                    message="Staging failed: " + error,
                    work_dir=requested_dir, reply_to=reply_to,
                )
                return
            if not has_changes:
                self._broadcast_autocommit_done(
                    tab_id, success=True, committed=False,
                    message="Nothing to commit.", work_dir=requested_dir, reply_to=reply_to,
                )
                return
            if not reply_to:
                self._commit_toast(tab_id, "info", "Auto-generating commit message…")
            msg = generate_commit_message_from_diff(diff_text)
            with repo_lock(repo), _reclaim_process_lock(repo):
                staged, error = _stage_all(work_path)
                if error:
                    error = "Staging failed: " + error
                elif staged != snapshot:
                    error = (
                        "The working tree changed while the commit message "
                        "was being generated; press Git Commit again."
                    )
                if error:
                    self._broadcast_autocommit_done(
                        tab_id, success=False, committed=False,
                        message=error, work_dir=requested_dir, reply_to=reply_to,
                    )
                    return
                # Commit directly (staged changes were verified
                # above) so a failure's actual git output —
                # pre-commit hook rejection text, identity/config
                # errors, … — can be reported to the user.
                commit_result = _git("commit", "-m", msg, cwd=str(work_path))
            if commit_result.returncode == 0:
                self._broadcast_autocommit_done(
                    tab_id, success=True, committed=True,
                    message=f"Committed: {_commit_subject(msg)}",
                    commit_message=msg, work_dir=requested_dir, reply_to=reply_to,
                )
            else:
                self._broadcast_autocommit_done(
                    tab_id, success=False, committed=False,
                    message=(
                        "git commit failed: "
                        + _git_error_line(commit_result, "pre-commit hook?")
                    ),
                    work_dir=requested_dir, reply_to=reply_to,
                )
        except Exception as e:  # pragma: no cover — unexpected git/LLM error
            logger.debug("Manual commit failed", exc_info=True)
            self._broadcast_autocommit_done(
                tab_id, success=False, committed=False,
                message=str(e), work_dir=requested_dir, reply_to=reply_to,
            )

    def _commit_on_machine(self, tab_id: str, machine: str, work_dir: str) -> None:
        """Run the tab's Git Commit on registered *machine* and report its outcome here.

        The Git Commit button of a tab whose drop-down names another
        machine commits that machine's folder: the command goes to its
        daemon (:func:`~kiss.server.sidebar_panels.machine_autocommit`),
        which stages, writes the message and commits there, and its
        ``autocommit_done`` is re-broadcast to this daemon's clients
        through :meth:`_broadcast_autocommit_done` (toast plus the
        event that re-arms the button).

        Args:
            tab_id: The tab that pressed the button.
            machine: The registered machine's name.
            work_dir: The folder to commit on the machine; empty for
                the machine's own configured folder.
        """
        self._commit_toast(tab_id, "info", f"Committing on {machine}…")
        reply = sidebar_panels.machine_autocommit(
            machine, {"type": "autocommitAction", "tabId": tab_id, "workDir": work_dir},
        )
        # A link error already names the machine ("box (https://…) did
        # not answer: …"); the machine's own outcome is prefixed with it.
        error = str(reply.get("error") or "")
        self._broadcast_autocommit_done(
            tab_id,
            success=bool(reply.get("success")) and not error,
            committed=bool(reply.get("committed")),
            message=error or f"{machine}: {reply.get('message') or ''}",
            commit_message=reply.get("commitMessage"),
            work_dir=str(reply.get("workDir") or work_dir),
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
    ) -> bool:
        """Auto-discard an empty pending worktree or emit ``worktree_done``.

        Single source of truth for post-task / session-resume handling
        of a pending worktree.

        - No pending worktree: return.
        - Worktree has changed files: broadcast ``worktree_done`` so
          the user gets the Merge / Discard / Do nothing buttons.
        - Worktree has no changes and *discard_if_empty* is True:
          auto-discard the empty branch (stale resumed sessions).
        - Worktree has no changes and *discard_if_empty* is False:
          leave the branch to the caller (the post-task path discards
          it itself, announcing the discard).

        Args:
            tab_id: The tab with a pending worktree.
            discard_if_empty: When True (default), auto-discard the
                branch if no files changed.

        Returns:
            ``True`` when ``worktree_done`` was broadcast, ``False``
            otherwise.
        """
        state = agent_state.find_by_tab(tab_id)
        if state is None:
            return False
        wt_agent = state.agent
        if wt_agent is None or not wt_agent._wt_pending:
            return False
        listing = self._list_pending(wt_agent)
        changed = _flatten_changed(listing, wt_agent._wt)
        # A worktree whose changes git could not enumerate is not
        # empty, it is unknown: it is kept and presented, never
        # auto-discarded (its branch and files may hold the only copy of
        # the task's work).
        unreadable = _unreadable_worktrees(listing)
        if not changed and not unreadable and discard_if_empty:
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
                    return False
                if state.busy() and state.merge_thread is not threading.current_thread():
                    return False
                prev_merging = state.is_merging
                prev_thread = state.merge_thread
                state.is_merging = True
                state.merge_thread = threading.current_thread()
            try:
                # Re-probe UNDER the claim: a task could have written
                # output and fully finished between the empty answer
                # above and the claim.  With the claim held, task
                # admission refuses new runs on this tab, so this
                # answer cannot go stale before it is acted on.  A
                # caller that already held the claim (the resume's
                # ``_emit_pending_worktree``) probed under it above.
                if prev_thread is not threading.current_thread():
                    listing = self._list_pending(wt_agent)
                    changed = _flatten_changed(listing, wt_agent._wt)
                    unreadable = _unreadable_worktrees(listing)
                if not changed and not unreadable:
                    wt_agent.discard()
            finally:
                with self._state_lock:
                    state.is_merging = prev_merging
                    state.merge_thread = prev_thread
                # A close that arrived during the discard saw the
                # tab busy and deferred disposal; nothing later
                # would dispose it.
                self._dispose_if_closed(tab_id)
        if not changed and not unreadable:
            return False
        for wt in unreadable:
            logger.warning(
                "Cannot list the changes of pending worktree %s (branch %s); "
                "the worktree is kept for the user to merge or discard",
                wt.wt_dir, wt.branch,
            )
        # The singular fields describe the worktree the task's work dir
        # is in (the clients' file-link fallback directory and SCM
        # view); ``worktrees`` lists every pending worktree, one per
        # repository, which Merge / Discard / Do nothing all act on.
        self.printer.broadcast({
            "type": "worktree_done",
            "branch": wt_agent._wt_branch,
            "worktreeDir": str(wt_agent._wt_dir),
            "worktreeWorkDir": str(wt_agent._wt_work_dir),
            "worktrees": [
                {"branch": wt.branch, "worktreeDir": str(wt.wt_dir)} for wt, _files in listing
            ],
            "tabId": tab_id,
        })
        return True

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

    def _list_pending(self, wt_agent: Any) -> list[tuple[GitWorktree, list[str] | None]]:
        """Pair every pending worktree of *wt_agent* with its changed files, probed once."""
        return [(wt, self._changed_files_of(wt)) for wt in _pending_worktrees(wt_agent)]

    def _changed_files_of(self, wt: GitWorktree) -> list[str] | None:
        """List files changed in *wt* vs its original branch.

        Detects both committed changes on the worktree branch and
        uncommitted changes in the worktree working tree.  When the
        worktree directory exists, runs ``git diff``, ``git diff
        --cached`` and ``git ls-files --others`` inside it so that
        uncommitted edits, staged-only edits and new files are
        included.  Falls back to a branch-to-branch diff when the
        worktree has already been removed.

        An empty list is a VERIFIED clean worktree: callers auto-discard
        on it.  When the comparison cannot be made (no original branch
        on record, a missing object, an unreadable index, a base that
        no longer resolves and no unique commits to fall back on), the
        answer is ``None`` — the changes are unknown, not absent.

        Args:
            wt: A pending worktree.

        Returns:
            Sorted deduplicated list of relative file paths, or ``None``
            when the changes could not be enumerated.
        """
        if not wt.original_branch:
            return None
        if wt.wt_dir.exists():
            base_ref = self._resolve_base_ref(
                str(wt.wt_dir), wt.baseline_commit, wt.original_branch,
            ) or wt.original_branch
            files = _git_paths(str(wt.wt_dir), "diff", "--name-only", "--no-renames", base_ref)
            if files is None:
                # The diff query failed (e.g. the original branch was
                # renamed/deleted so base_ref no longer resolves).  A
                # clean ``status --porcelain`` alone must NOT be taken
                # as "no changes": the worktree may hold COMMITTED task
                # work, and callers auto-discard the branch when this
                # returns [].  Also list files from commits unique to
                # this worktree so committed work is never mistaken for
                # a clean worktree.
                # Both sides of a rename are listed, as by a
                # ``diff --no-renames``.
                status = _git("status", "--porcelain", "-z", cwd=str(wt.wt_dir))
                if status.returncode != 0:
                    return None
                files = [
                    path
                    for _code, old_name, new_name in _porcelain_entries(status.stdout)
                    for path in (old_name, new_name)
                    if path is not None
                ]
                unique = _git_paths(
                    str(wt.wt_dir), "log", "--pretty=format:", "--name-only", "--no-renames",
                    "HEAD", "--not", f"--exclude={wt.branch}", "--branches",
                )
                if unique is None:
                    return None
                files.extend(unique)
            else:
                # ``diff <base>`` compares the base with the files on
                # disk: work that is only in the index (staged, the file
                # then put back on disk) is invisible to it, and the
                # worktree would be discarded as clean.
                staged = _git_paths(
                    str(wt.wt_dir), "diff", "--cached", "--name-only", "--no-renames", base_ref,
                )
                if staged is None:
                    return None
                files.extend(staged)
            # A failed listing (a timeout walking a huge or hung tree)
            # is unknown, not "no new files": an empty answer here
            # discards the worktree.
            untracked = _git_paths(str(wt.wt_dir), "ls-files", "--others", "--exclude-standard")
            if untracked is None:
                return None
            files.extend(untracked)
            return sorted(set(files))
        base_ref = self._resolve_base_ref(
            str(wt.repo_root), wt.baseline_commit, wt.original_branch,
            tip=wt.branch,
        ) or wt.original_branch
        files = _git_paths(
            str(wt.repo_root), "diff", "--name-only", "--no-renames", base_ref, wt.branch,
        )
        if files is None:
            # Same fallback as above for a branch without a directory:
            # the files of the commits unique to the branch.
            files = _git_paths(
                str(wt.repo_root), "log", "--pretty=format:", "--name-only", "--no-renames",
                wt.branch, "--not", f"--exclude={wt.branch}", "--branches",
            )
            if files is None:
                return None
        return sorted(set(files))

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
            Error dict with ``success: False`` when busy, otherwise
            ``None``.  The refusal behind an in-flight merge or discard
            is ``retryable``: it is broadcast to every page, a page
            drops its bar on a result that is not, and the in-flight
            action may still end with the worktree pending.
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
                "retryable": True,
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
            # in-flight tasks cannot see this work.  Shutdown's
            # ``_await_active_merges`` instead waits for this claim
            # (released in the ``finally`` below) so the repository
            # is never left mid-merge; the thread is recorded so an
            # orphaned claim can be detected (``merge_in_progress``).
            state.merge_thread = threading.current_thread()
        try:
            # One lock per isolated repository, in the order every
            # thread takes several in, so two tabs acting on the same
            # repositories cannot deadlock (the per-worktree
            # merge/discard re-enters its own lock).
            roots = [repo_root, *(w.repo_root for w in _pending_worktrees(wt))]
            with contextlib.ExitStack() as locks:
                for lock in repo_locks(roots):
                    locks.enter_context(lock)
                if action == "nothing":
                    # ``kept: True`` tells the clients to KEEP their
                    # pending-worktree fallback directory — the
                    # worktree stays on disk, so transcript file links
                    # into it must keep resolving.  A failed marker
                    # write leaves the worktree pending (fail closed),
                    # and the failure is ``retryable`` so the page
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
                    if success:
                        return {"success": True, "message": msg}
                    # A worktree is still pending: ``retryable`` keeps the
                    # bar's buttons on the page for the retry.
                    return {"success": False, "message": msg, "retryable": True}
                # A partial discard (branch deletion failed) must not
                # report success.
                msg = wt.discard()
                return {"success": "Partially discarded" not in msg, "message": msg}
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
        between those counters before and after the merge is late
        spend on the finished row, charged like any side channel's
        (:func:`~kiss.server.task_update.charge_side_channel_usage`:
        the row and its finished ancestors get it, and their new totals
        reach the tabs as a persisted ``usage_info``, *tab_id*
        included so its live status row updates now).  No live agent
        is passed: the spend is already on *agent*'s counters.

        Args:
            agent: The worktree agent that ran the merge.
            usage_before: ``(budget, tokens, steps)`` of *agent* before
                the merge, from :func:`_agent_usage`.
            tab_id: The tab whose status row to refresh.
        """
        budget, tokens, steps = _agent_usage(agent)
        charge_side_channel_usage(
            self.printer,
            None,
            _persisted_task_id(agent),
            budget - usage_before[0],
            tokens - usage_before[1],
            steps - usage_before[2],
            tab_id=tab_id,
        )
