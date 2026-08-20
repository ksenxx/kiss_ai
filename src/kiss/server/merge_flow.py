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
from kiss.agents.sorcar.sorcar_agent import _commit_subject
from kiss.agents.sorcar.useful_tools import _stale_worktree_fallback
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.diff_merge import _capture_untracked, _git
from kiss.server.helpers import generate_commit_message_from_diff

if TYPE_CHECKING:
    from kiss.server.json_printer import JsonPrinter

logger = logging.getLogger(__name__)


def _repo_of_dir(work_dir: str) -> Path | None:
    """Return the resolved repo root containing directory *work_dir*.

    Args:
        work_dir: Directory path (possibly not in a repo, possibly
            gone — e.g. a reclaimed worktree).

    Returns:
        The resolved repository toplevel, or ``None`` when *work_dir*
        is not inside a git repository.
    """
    try:
        repo = GitWorktreeOps.discover_repo(Path(work_dir))
    except Exception:
        return None
    return repo.resolve() if repo is not None else None


def _existing_dir_of(path_str: str) -> Path | None:
    """Return the nearest existing ancestor DIRECTORY of *path_str*.

    ``git rev-parse --show-toplevel`` needs an existing directory as
    its cwd, but a recorded path may name a file — or a file the task
    later deleted.  Relative paths are refused: the caller resolves
    them against the task's work_dir before coming here.

    Args:
        path_str: A path string recorded from a tool call.

    Returns:
        An existing directory to run repo discovery from, or ``None``.
    """
    try:
        p = Path(path_str)
        if not p.is_absolute():
            return None
        for candidate in [p, *p.parents]:
            if candidate.is_dir():
                return candidate
    except OSError:
        return None
    return None


def _group_paths_by_repo(
    paths: set[str], *, exclude_repo: Path | None, base_dir: str = "",
) -> dict[Path, set[str]]:
    """Group changed *paths* by the git repository containing each.

    Relative paths are resolved against *base_dir* (the task's
    work_dir — the cwd of the file tools that recorded them); without
    a base they cannot be attributed and are dropped.

    Paths outside any repository are dropped (nothing to commit them
    to), as are paths in *exclude_repo* (the work_dir repository —
    already committed whole by the main auto-commit pass) and paths in
    ``.kiss-worktrees`` checkouts (the worktree merge flow owns those;
    committing to a worktree branch behind its agent's back would
    corrupt the pending merge).

    Args:
        paths: Path strings recorded from the task's tool calls.
        exclude_repo: Resolved work_dir repository root, or ``None``.
        base_dir: Directory relative recorded paths are taken against.

    Returns:
        Mapping of resolved repo root -> the recorded paths inside it.
    """
    groups: dict[Path, set[str]] = {}
    for path_str in paths:
        candidate = path_str
        if not Path(path_str).is_absolute():
            if not base_dir:
                continue
            candidate = str(Path(base_dir) / path_str)
        start = _existing_dir_of(candidate)
        if start is None:
            continue
        repo = _repo_of_dir(str(start))
        if repo is None:
            continue
        if exclude_repo is not None and repo == exclude_repo:
            continue
        if ".kiss-worktrees" in repo.parts:
            continue
        groups.setdefault(repo, set()).add(candidate)
    return groups


def _state_task_key(state: AgentState | None) -> str | None:
    """Return the task id *state* last ran, preferring the live agent's.

    The agent's id is read through its ``last_task_id`` property,
    which takes the same lock the publishing assignment in
    ``ChatSorcarAgent.run`` takes, so this cross-thread read is paired
    with its writer.

    Args:
        state: The agent state to inspect, or ``None``.

    Returns:
        The task id, or ``None`` when the state never ran a task.
    """
    if state is None:
        return None
    task_id = getattr(state.agent, "last_task_id", "")
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

        def _any_non_wt_running(
            self, repo_root: Path | None = None,
        ) -> bool: ...
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
        manual: bool = False,
        work_dir: str = "",
    ) -> dict[str, Any]:
        """Broadcast an ``autocommit_done`` event and return it.

        For a *manual* commit (the user pressed the Git Commit button)
        the outcome is reported as a toast ``notification`` — info on
        success, error on failure — instead of transcript text: the
        event carries ``manual: True`` so the webview and the VS Code
        host skip their own chat/toast rendering for the successful
        case (the event itself must still be broadcast so the Git
        Commit button re-arms).  A failed manual commit stays
        non-silent so its reason is also shown in the chat webview.

        Args:
            tab_id: Frontend tab identifier.
            success: Whether the action succeeded.
            committed: Whether a commit was actually created.
            message: Human-readable status message.
            commit_message: Full commit message (only when committed).
            manual: ``True`` when the user pressed the Git Commit
                button (as opposed to the post-task autocommit).
            work_dir: The working directory the commit targeted.
                Carried on the event as ``workDir`` so the webview's
                post-task main-tree bar dismisses itself only for the
                commit flow it started — an ``autocommit_done`` for a
                DIFFERENT repository sharing the tab must not strip
                the bar's controls (gpt-5.6-sol review finding).

        Returns:
            The event dict (for optional persistence).
        """
        if manual:
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
            "tabId": tab_id,
        }
        if commit_message is not None:
            event["commitMessage"] = commit_message
        if manual:
            event["manual"] = True
        if work_dir:
            event["workDir"] = work_dir
        self.printer.broadcast(event)
        return event

    def _autocommit_changes(
        self, tab_id: str = "", *, work_dir: str = "", manual: bool = False,
    ) -> None:
        """Stage-all + generate-message + commit the tab's working tree.

        Called by the post-task path for non-worktree tasks: with the
        interactive diff review gone, task changes are committed
        directly and reported through ``autocommit_progress`` /
        ``autocommit_done`` events.  A clean tree is a cheap no-op
        ("Nothing to commit.").

        Also called with ``manual=True`` when the user presses the Git
        Commit button.  A manual commit differs from the post-task
        path in three ways:

        - The commit message is generated from the staged diff ALONE —
          the tab's last user prompt and task result are NOT appended
          (no ``User prompt:`` / ``Result:`` sections).
        - Progress and outcome are reported as toast ``notification``
          events ("Auto-generating commit message…", then the
          committed subject or the failure reason) instead of
          transcript text, and nothing is appended to the chat
          transcript or its persisted history on success.
        - A failure still shows its reason in the chat webview (the
          ``autocommit_done`` failure event stays non-silent).

        Args:
            tab_id: The tab that ran the task (echoed in the
                ``autocommit_done`` event).
            work_dir: The tab's working directory.  Preferred over the
                daemon-wide ``self.work_dir`` because the shared
                ``kiss-web`` daemon may have been launched from (or
                synced to) a different — possibly non-git — folder than
                the window that owns this tab.  Falls back to
                ``self.work_dir`` when empty.
            manual: ``True`` when the user pressed the Git Commit
                button (as opposed to the post-task autocommit).
        """
        work_dir = work_dir or self.work_dir
        # Echoed on every autocommit_done as `workDir` — the DIR THE
        # CALLER NAMED, not any stale-worktree fallback remap below —
        # so the webview's main-tree bar (which stamped this exact
        # string on its autocommitAction) can recognize its own
        # terminal event.
        requested_dir = work_dir
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
                    message="Not a git repository.", manual=manual, work_dir=requested_dir,
                )
                return
            with repo_lock(repo):
                if not manual:
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
                        manual=manual, work_dir=requested_dir,
                    )
                    return
                diff = _git(work_dir, "diff", "--cached")
                if not diff.stdout.strip():
                    self._broadcast_autocommit_done(
                        tab_id, success=True, committed=False,
                        message="Nothing to commit.", manual=manual, work_dir=requested_dir,
                    )
                    return
                if manual:
                    self.printer.broadcast({
                        "type": "notification",
                        "id": f"manual-commit:{tab_id}",
                        "severity": "info",
                        "message": "Auto-generating commit message…",
                        "tabId": tab_id,
                    })
                else:
                    self.printer.broadcast({
                        "type": "autocommit_progress",
                        "message": "Generating commit message…",
                        "tabId": tab_id,
                    })
                if manual:
                    # A user-invoked commit describes the DIFF, not the
                    # last task: no User prompt: / Result: sections.
                    user_prompt = None
                    task_result = None
                else:
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
                if not manual:
                    self.printer.broadcast({
                        "type": "autocommit_progress",
                        "message": "Committing…",
                        "tabId": tab_id,
                    })
                # Commit directly (the staged diff was verified
                # non-empty above) so a failure's actual git output —
                # pre-commit hook rejection text, identity/config
                # errors, … — can be reported to the user instead of
                # a guess.
                commit_result = _git(work_dir, "commit", "-m", msg)
                ok = commit_result.returncode == 0
            if ok:
                msg_lines = msg.splitlines()
                subject = msg_lines[0] if msg_lines else msg
                done_event = self._broadcast_autocommit_done(
                    tab_id, success=True, committed=True,
                    message=f"Committed: {subject}",
                    commit_message=msg, manual=manual, work_dir=requested_dir,
                )
                if tab_id and not manual:
                    with self._state_lock:
                        task_id = _state_task_key(agent_state.find_by_tab(tab_id))
                    if task_id is not None:
                        _append_chat_event(done_event, task_id=task_id)
            else:
                err = (
                    (commit_result.stderr or "").strip()
                    or (commit_result.stdout or "").strip()
                )
                reason = err.splitlines()[0] if err else "pre-commit hook?"
                self._broadcast_autocommit_done(
                    tab_id, success=False, committed=False,
                    message=f"git commit failed: {reason}",
                    manual=manual, work_dir=requested_dir,
                )
        except Exception as e:  # pragma: no cover — unexpected git/LLM error
            logger.debug("Autocommit action failed", exc_info=True)
            self._broadcast_autocommit_done(
                tab_id, success=False, committed=False,
                message=str(e), manual=manual, work_dir=requested_dir,
            )

    def _autocommit_changed_repos(
        self,
        tab_id: str = "",
        *,
        work_dir: str = "",
        task_id: str | None = None,
        extra_paths: set[str] | None = None,
        extra_task_ids: list[str] | None = None,
    ) -> None:
        """Auto-commit task changes that landed OUTSIDE the work_dir repo.

        :meth:`_autocommit_changes` commits the repository that
        contains the tab's *work_dir* — and nothing else.  A task is
        free to change files anywhere (the standard file tools take
        absolute paths), so with auto-commit on, files it wrote in a
        DIFFERENT repository were silently left uncommitted (observed
        in production: tasks with ``work_dir`` in one project editing
        a sibling project's checkout).

        This companion pass, run right after the work_dir commit at
        the end of a non-worktree auto-commit task, closes that gap:

        1. Collects the file paths the task changed — the printer's
           in-memory per-task record of ``Write`` / ``Edit`` tool
           calls, plus the record of any sub-tasks.
        2. Groups them by containing git repository, skipping the
           work_dir repository (already committed), paths outside any
           repository, and ``.kiss-worktrees`` checkouts (their own
           merge flow owns those).
        3. In each remaining repository, commits ONLY the recorded
           paths — never ``add -A`` — so pre-existing dirty state in a
           repository the user never designated as the task's work_dir
           is not swept into the commit.

        Failures are per-repository: one repo that cannot commit does
        not stop the others, and never propagates to the caller (the
        task's finally block).  A failure while loading the sub-task
        record costs only the sub-task paths — the task's own paths
        (already popped from the printer) are still committed.

        Known limitation, by design: files changed through ``Bash``
        (``sed -i``, build scripts, ...) carry no attributable path
        and are not tracked; in the work_dir repository they are still
        swept up by the main pass's ``git add -A``, elsewhere they
        stay uncommitted.

        Args:
            tab_id: The tab that ran the task (echoed in events).
            work_dir: The tab's working directory (its repository is
                skipped here).  Falls back to ``self.work_dir``.
            task_id: The finished task's history id, used to look up
                the changed paths.  ``None`` does nothing.
            extra_paths: Additional changed paths the caller collected
                itself — the task-runner passes the paths of earlier
                sequential ``<task>`` runs of the same submission,
                whose printer entries its per-subtask persistence
                already popped.
            extra_task_ids: History ids of those earlier sequential
                runs, so THEIR sub-agents' records are collected (and
                their printer entries freed) too.
        """
        if task_id is None:
            return
        paths: set[str] = set(extra_paths or ())
        paths |= self.printer.pop_changed_paths(task_id)
        try:
            from kiss.agents.sorcar.persistence import (
                _changed_paths_of_tasks,
                _descendant_task_ids,
            )

            # Sub-agents record under their own task ids.  Popping
            # each descendant both collects what has not been
            # persisted yet and frees the entry — nothing else ever
            # pops a sub-agent's id, so reading without popping would
            # leak one set per file-changing sub-agent.
            sub_ids: list[str] = []
            for root in [str(task_id), *(extra_task_ids or [])]:
                sub_ids.extend(_descendant_task_ids(root))
            for sub_id in sub_ids:
                paths |= self.printer.pop_changed_paths(sub_id)
            paths |= _changed_paths_of_tasks(sub_ids)
        except Exception:  # pragma: no cover — defensive collection
            logger.debug(
                "Sub-task changed-path collection failed", exc_info=True,
            )
        try:
            repos = _group_paths_by_repo(
                paths,
                exclude_repo=_repo_of_dir(work_dir or self.work_dir),
                base_dir=work_dir or self.work_dir,
            )
        except Exception:  # pragma: no cover — defensive grouping
            logger.debug("Changed-path grouping failed", exc_info=True)
            return
        for repo, repo_paths in sorted(repos.items()):
            try:
                self._autocommit_paths_in_repo(
                    repo, sorted(repo_paths), tab_id,
                )
            except Exception:
                logger.debug(
                    "Auto-commit in %s failed", repo, exc_info=True,
                )

    def _autocommit_paths_in_repo(
        self,
        repo: Path,
        paths: list[str],
        tab_id: str,
    ) -> None:
        """Commit exactly *paths* in *repo* and broadcast the outcome.

        Everything is path-limited: only the files the task itself
        changed enter the commit; unrelated dirty files in *repo* stay
        as they were, and — since the final ``git commit`` carries the
        same pathspec — entries the USER had already staged in *repo*
        for other files stay staged rather than being swept into the
        task's commit.

        The recorded paths are first filtered through ``git status``:
        a path the task rewrote to its original bytes, an ignored
        ``tmp/`` scratch file, or a path that never materialised would
        otherwise make the pathspec commit fail outright ("did not
        match any file(s) known to git").  ``git add -A`` on the
        surviving paths makes the same pathspec cover new files and
        deletions alike.

        A repo whose HEAD is detached is refused with a toast instead
        of committed: a commit no branch points at — e.g. in a
        submodule checkout, which is detached by default — would be
        unreachable the moment the checkout moves, which is data loss
        wearing a success message.

        A repo where nothing survives the filter is skipped silently
        rather than spamming "Nothing to commit" toasts for every
        repository the task touched.

        Args:
            repo: Repository root to commit in.
            paths: The recorded changed paths inside *repo*.
            tab_id: The tab that ran the task (echoed in events).
        """
        # --literal-pathspecs: a recorded filename that happens to
        # contain pathspec magic (``*``, ``?``, ``[``, ``:(...)``)
        # must match itself only — never OTHER paths the task did not
        # record.
        lit = "--literal-pathspecs"
        recorded = {os.path.normpath(p) for p in paths}
        with repo_lock(repo):
            status = _git(
                str(repo), lit, "status", "--porcelain", "--", *paths,
            )
            if status.returncode != 0:
                logger.debug(
                    "git status failed in %s: %s", repo, status.stderr,
                )
                return
            changed: list[str] = []
            for _code, old_name, new_name in _porcelain_entries(
                status.stdout,
            ):
                # Defense in depth: the pathspec-limited status splits
                # a rename whose old side is not among *paths* (git
                # only pairs the sides when both match), but should a
                # pairing ever surface an old path the task never
                # recorded — e.g. the user's own staged ``git mv`` —
                # committing its deletion would break the "only
                # recorded paths" guarantee.
                if old_name and (
                    os.path.normpath(str(repo / old_name)) in recorded
                ):
                    changed.append(old_name)
                changed.append(new_name)
            if not changed:
                return
            if GitWorktreeOps.current_branch(repo) is None:
                self._broadcast_autocommit_done(
                    tab_id, success=False, committed=False,
                    message=(
                        f"{repo.name} is on a detached HEAD; "
                        "left its changes uncommitted."
                    ),
                )
                return
            add_result = _git(str(repo), lit, "add", "-A", "--", *changed)
            if add_result.returncode != 0:
                err = (add_result.stderr or "").strip()
                first_line = err.splitlines()[0] if err else "git add failed"
                self._broadcast_autocommit_done(
                    tab_id, success=False, committed=False,
                    message=f"Staging failed in {repo.name}: {first_line}",
                )
                return
            diff = _git(str(repo), lit, "diff", "--cached", "--", *changed)
            if diff.returncode != 0:
                self._broadcast_autocommit_done(
                    tab_id, success=False, committed=False,
                    message=f"git diff failed in {repo.name}.",
                )
                return
            if not diff.stdout.strip():
                return
            self.printer.broadcast({
                "type": "autocommit_progress",
                "message": f"Committing changes in {repo.name}…",
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
            try:
                msg = (
                    generate_commit_message_from_diff(
                        diff.stdout,
                        user_prompt=user_prompt,
                        task_result=task_result,
                    )
                    or "Auto-commit"
                )
            except Exception:
                logger.debug(
                    "Commit message generation failed; using fallback",
                    exc_info=True,
                )
                msg = "kiss: auto-commit agent changes"
            # Pathspec-limited commit: takes the listed paths from the
            # working tree / index and leaves every OTHER staged entry
            # in the user's index exactly as it was.  A plain
            # ``git commit`` here would sweep the user's own staged
            # work into the task's commit.
            commit = _git(
                str(repo), lit, "commit", "-m", msg, "--", *changed,
            )
            ok = commit.returncode == 0
        if ok:
            subject = _commit_subject(msg)
            done_event = self._broadcast_autocommit_done(
                tab_id, success=True, committed=True,
                message=f"Committed in {repo.name}: {subject}",
                commit_message=msg,
            )
            if tab_id:
                with self._state_lock:
                    task_key = _state_task_key(
                        agent_state.find_by_tab(tab_id),
                    )
                if task_key is not None:
                    _append_chat_event(done_event, task_id=task_key)
        else:
            self._broadcast_autocommit_done(
                tab_id, success=False, committed=False,
                message=(
                    f"git commit failed in {repo.name} "
                    "(pre-commit hook?)."
                ),
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
            # tree, so a concurrent non-worktree task on the MAIN
            # tree is no reason to skip it — skipping leaks the
            # worktree forever because nothing ever retries.  A task
            # running INSIDE this worktree is different: the discard
            # would delete the directory out from under it, so leave
            # the worktree pending (a later resume retries).
            wt_dir = wt_agent._wt_dir
            with self._state_lock:
                if wt_dir is not None and self._any_non_wt_running(
                    wt_dir,
                ):
                    return
            with self._state_lock:
                prev_merging = state.is_merging
                state.is_merging = True
            try:
                # Automatic path: rescue git-ignored task output the
                # changed-files probe cannot see (see
                # WorktreeSorcarAgent.discard).
                wt_agent.discard(rescue_ignored=True)
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
            "worktreeWorkDir": str(wt_agent._wt_work_dir),
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

        The answer describes the worktree the tab's agent owns, so it
        does NOT depend on the mode of the run currently executing on
        that tab.  Returning ``[]`` for a tab whose latest run has
        ``use_worktree`` off would report a worktree full of work as
        empty, and callers destroy an "empty" worktree (R09-1).  The
        agent checks below already cover the case of a tab that owns
        no worktree at all.

        Args:
            tab_id: The tab whose worktree to check.

        Returns:
            Sorted deduplicated list of relative file paths.
        """
        state = agent_state.find_by_tab(tab_id)
        if state is None:
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

    def _check_worktree_busy(
        self,
        state: AgentState,
        verb: str,
        repo_root: Path | None = None,
        wt_dir: Path | None = None,
    ) -> dict[str, Any] | None:
        """Return an error dict if a worktree action should be refused, else None.

        Checks both the tab's own task and any non-worktree task running
        on the main tree of *repo_root* (BUG-35, BUG-72 fixes), or
        inside the pending worktree *wt_dir* itself.

        Must be called with ``_state_lock`` already held (RACE-1 fix)
        so the caller can atomically set ``state.is_merging = True``
        before releasing the lock — otherwise a non-wt task on
        another tab could pass its own ``is_merging`` guard in the
        TOCTOU window between this check returning ``None`` and the
        caller acquiring ``_state_lock`` again to set the flag.

        Args:
            state: The agent state to check.
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
            action: One of ``"merge"``, ``"discard"`` or ``"nothing"``
                (the Do-nothing button: detach from the worktree,
                leaving its branch, directory, and any uncommitted
                changes untouched on disk).
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
        verb = {
            "merge": "merging",
            "discard": "discarding",
            "nothing": "detaching",
        }.get(action)
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
                busy = self._check_worktree_busy(state, verb, repo_root, wt._wt_dir)
                if busy:
                    return busy
            elif wt._wt_dir is not None and self._any_non_wt_running(
                wt._wt_dir,
            ):
                # A task on ANOTHER tab is running INSIDE this pending
                # worktree (its work_dir's toplevel is the linked
                # worktree).  Both merge and discard delete the
                # directory out from under it, so — unlike the
                # main-tree guard below — the discard exemption does
                # NOT apply here (gpt-5.6-sol review finding: the
                # internal auto-discard used to remove an occupied
                # worktree).
                return {
                    "success": False,
                    "message": (
                        "Another tab is running a task inside this "
                        "task's worktree. Wait for it to finish "
                        f"before {verb}."
                    ),
                }
            elif action == "merge" and self._any_non_wt_running(repo_root):
                # internal=True only bypasses this tab's OWN
                # is_task_active/is_merging flags (the post-task
                # auto-finalize runs on the task thread that owns
                # them).  It must NOT bypass the main-tree guard
                # (F4-19): merging stashes/checkouts/merges the
                # main working tree while a direct task on another
                # tab is still writing it.  A DISCARD is exempt from
                # THIS guard: it only removes .kiss-worktrees/<slug>
                # and deletes the unmerged branch, touching neither
                # the main working tree's files nor its HEAD, so
                # refusing it would leak the worktree forever
                # (nothing ever retries).
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
                # This runs in the event loop's default executor, and
                # the task that produced the worktree is long gone, so
                # ``task_thread`` is None and the shutdown sweep of
                # in-flight tasks cannot see this work.  Publishing the
                # thread lets shutdown WAIT for the repository to stop
                # being rewritten instead of returning mid-merge.
                state.merge_thread = threading.current_thread()
        # ``_pending_review`` is NOT cleared here: the agent's own
        # merge()/discard() clear it themselves, and only past their
        # deferral checks.  Clearing it up front would let a tab close
        # during a DEFERRED discard auto-release (merge!) work the
        # user explicitly asked to throw away (gpt-5.6-sol review
        # finding).
        try:
            with repo_lock(repo_root):
                if action == "nothing":
                    # "Do nothing": detach from the worktree without
                    # touching git state beyond the preserve marker.
                    # ``kept: True`` tells the clients to KEEP their
                    # pending-worktree fallback directory — the
                    # worktree stays on disk, so transcript file links
                    # into it must keep resolving.  A failed marker
                    # write leaves the worktree pending (fail closed),
                    # and the failure is ``retryable`` so the webview
                    # keeps the bar's buttons instead of stripping the
                    # only retry controls.
                    try:
                        msg = wt.leave_as_is()
                    except RuntimeError as e:
                        return {
                            "success": False,
                            "message": str(e),
                            "retryable": True,
                        }
                    return {"success": True, "message": msg, "kept": True}
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
                # Only the AUTOMATIC discard (post-task finalize /
                # session-resume, internal=True) rescues git-ignored
                # task output: it runs because the changed-files probe
                # saw nothing, and that probe cannot see ignored
                # files.  A user-explicit Discard click throws the
                # work away as asked.
                msg = wt.discard(rescue_ignored=internal)
                # A partial discard (branch deletion failed) or a
                # deferred one (a sub-agent is still writing into the
                # worktree, or its ignored output could not be
                # rescued) must not report success: the UI would
                # close the workflow while an orphan branch remains
                # (F4-24) or while the worktree is still pending.  A
                # deferred discard is additionally flagged retryable
                # so the webview keeps the Merge / Discard buttons
                # instead of stripping the only retry controls.
                deferred = "Discard deferred" in msg
                result: dict[str, Any] = {
                    "success": (
                        "Partially discarded" not in msg and not deferred
                    ),
                    "message": msg,
                }
                if deferred:
                    result["retryable"] = True
                return result
        finally:
            if not already_claimed:
                with self._state_lock:
                    state.is_merging = False
                    state.merge_thread = None
                # A close that arrived during the merge/discard saw the
                # tab busy and deferred disposal; without this call the
                # backend tab state would leak indefinitely (F4-23).
                self._dispose_if_closed(tab_id)

    def _handle_main_tree_action(
        self, action: str, work_dir: str,
    ) -> dict[str, Any]:
        """Execute a main-working-tree discard or do-nothing action.

        Serves the post-task action bar of non-worktree manual-commit
        runs (its third button, Auto commit, reuses the existing
        ``autocommitAction`` command instead).  The bar's other two
        choices land here:

        * ``"discard"`` — throw away every uncommitted change in the
          repository containing *work_dir*: ``git reset --hard``
          restores tracked files, ``git clean -fd`` removes untracked
          ones.  Ignored files (``.kiss-worktrees/``, build output…)
          are untouched — ``clean`` runs without ``-x``.  Refused
          while a non-worktree task is running in the same repository,
          exactly like a manual commit: the reset would yank
          half-written files out from under the live agent.  The
          per-repo ``repo_lock`` additionally serializes against a
          concurrent worktree merge stashing/rewriting the same main
          tree.
        * ``"nothing"`` — leave the working tree exactly as the task
          ended, changes uncommitted.  Nothing to do daemon-side; the
          success result exists so every connected client dismisses
          the bar together.

        Args:
            action: One of ``"discard"`` or ``"nothing"``.
            work_dir: The tab's working directory (any folder inside
                the target repository).

        Returns:
            Dict with ``success`` bool and ``message`` string.
        """
        if action == "nothing":
            return {
                "success": True,
                "message": (
                    "Left the changes in the working tree, uncommitted."
                ),
            }
        if action != "discard":
            return {"success": False, "message": f"Unknown action: {action}"}
        work_dir = work_dir or self.work_dir
        repo = GitWorktreeOps.discover_repo(Path(work_dir))
        if repo is None:
            return {
                "success": False,
                "message": f"{work_dir} is not inside a git repository.",
            }
        with repo_lock(repo):
            # Re-checked under the repo lock (not only before taking
            # it) to narrow the window in which a direct task starting
            # on another tab could have its half-written files reset
            # out from under it.  The residual TOCTOU — a task starting
            # after this check — is the same window the manual Git
            # Commit already has (`_cmd_autocommit_action` probes the
            # same predicate before its own git mutations); a full fix
            # needs task startup to take a per-repo claim, which
            # neither flow does today.
            with self._state_lock:
                if self._any_non_wt_running(repo):
                    return {
                        "success": False,
                        "message": (
                            "A task is still running in this folder; "
                            "wait for it to finish before discarding."
                        ),
                    }
            dirty = self._main_dirty_files(str(repo))
            if not dirty:
                return {
                    "success": True,
                    "message": "Nothing to discard: the working tree is clean.",
                }
            reset = _git(str(repo), "reset", "--hard")
            if reset.returncode != 0:
                return {
                    "success": False,
                    "message": (
                        "git reset --hard failed: "
                        + (reset.stderr or reset.stdout).strip()
                    ),
                }
            clean = _git(str(repo), "clean", "-fd")
            if clean.returncode != 0:
                return {
                    "success": False,
                    "message": (
                        "Tracked changes were reset, but git clean -fd "
                        "failed: " + (clean.stderr or clean.stdout).strip()
                    ),
                }
            # Both commands can return 0 and still leave dirt behind:
            # `reset --hard` does not enter submodules and `clean -fd`
            # refuses to delete an untracked nested git repository.
            # Claiming success then would dismiss the bar over a tree
            # that `git status` still reports dirty (gpt-5.6-sol
            # review finding).
            leftover = self._main_dirty_files(str(repo))
            if leftover:
                return {
                    "success": False,
                    "message": (
                        f"Discarded {len(dirty) - len(leftover)} of "
                        f"{len(dirty)} uncommitted file(s) in "
                        f"{repo.name}; still dirty (submodule or "
                        "nested git repository?): "
                        + ", ".join(leftover[:10])
                    ),
                }
            return {
                "success": True,
                "message": (
                    f"Discarded {len(dirty)} uncommitted file(s) "
                    f"in {repo.name}."
                ),
            }
