# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Task-runner mixin for the VS Code server.

Implements the background-thread task lifecycle: ``_run_task`` (status
broadcasts) and ``_run_task_inner`` (pre/post snapshots, agent
invocation, merge-view preparation, persistence).  Also hosts the
cooperative-stop machinery and the ``ask_user_question`` callback.

Split out of ``server.py`` for organisation.
"""

from __future__ import annotations

import base64
import ctypes
import logging
import math
import queue
import re
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext, suppress
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from kiss.agents.sorcar.running_agent_state import _RunningAgentState

from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    repo_lock,
    strip_worktree_suffix,
)
from kiss.agents.sorcar.persistence import (
    _append_chat_event,
    _load_last_model,
    _save_task_extra,
    _save_task_result,
)
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import _WorktreeCleanupOutcome
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import get_available_models
from kiss.core.printer import parse_result_yaml
from kiss.server.diff_merge import (
    _capture_untracked,
    _parse_diff_hunks,
    _save_untracked_base,
    _snapshot_files,
)
from kiss.server.helpers import tab_owns_answer_queue
from kiss.server.json_printer import JsonPrinter
from kiss.server.tools_file import load_tools_file

logger = logging.getLogger(__name__)

ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = [
    ctypes.c_ulong,
    ctypes.py_object,
]


def _state_owns_thread(
    tab_id: str,
    state: _RunningAgentState | None,
    thread: threading.Thread,
) -> bool:
    """True while *tab_id* still maps *state* and *state* still runs *thread*.

    Ownership guard for :meth:`_TaskRunnerMixin._force_stop_thread`'s
    asynchronous ``KeyboardInterrupt`` injection.  Callers evaluate it
    under :attr:`_RunningAgentState._registry_lock`; the producers of
    parallel sub-agent states clear ``state.task_thread`` (and
    unregister the state) under the same lock before their pool worker
    thread returns to the executor, so a ``False`` here reliably means
    the stop target already finished and *thread* must not be touched.

    Args:
        tab_id: Registry key of the state that owned the stopped task.
        state: The state object resolved at ``_stop_task`` time.
        thread: The task thread captured at ``_stop_task`` time.

    Returns:
        ``True`` when the registry entry is unchanged and still owns
        *thread*; ``False`` otherwise.
    """
    current = _RunningAgentState.running_agent_states.get(tab_id)
    return current is not None and current is state and current.task_thread is thread


def build_task_extra_payload(
    *,
    model: str,
    work_dir: str,
    version: str,
    tokens: int,
    cost: float,
    steps: int,
    is_parallel: bool,
    is_worktree: bool,
    auto_commit_mode: bool,
    start_ms: int,
    end_ms: int,
) -> dict[str, object]:
    """Build the persisted ``task_history.extra`` payload for a completed task.

    Strips the ``.kiss-worktrees/kiss_wt-<slug>`` suffix from *work_dir*
    so the persisted path is the user-visible workspace folder rather
    than an ephemeral worktree directory that would vanish on merge or
    discard.

    Args:
        model: Model name used for the task.
        work_dir: Working directory the task ran from.  Worktree paths
            are stripped to their parent repo.
        version: KISS version string.
        tokens: Total tokens consumed by the agent.
        cost: Total budget consumed by the agent (USD).
        steps: Total agent steps taken.
        is_parallel: Whether parallel sub-agents were enabled.
        is_worktree: Whether the task ran inside a worktree.
        auto_commit_mode: Auto-commit toggle state at completion.
        start_ms: Agent start timestamp in milliseconds since epoch.
        end_ms: Agent end timestamp in milliseconds since epoch.

    Returns:
        Dict ready to pass to ``_save_task_extra``.
    """
    return {
        "model": model,
        "work_dir": strip_worktree_suffix(work_dir),
        "version": version,
        "tokens": tokens,
        "cost": cost,
        "steps": steps,
        "is_parallel": is_parallel,
        "is_worktree": is_worktree,
        "auto_commit_mode": auto_commit_mode,
        "startTs": start_ms,
        "endTs": end_ms,
    }


def _client_task_id_of(cmd: dict[str, Any]) -> str:
    """Return the client-stamped ``taskId`` of a ``run`` command, or ``""``.

    The CLI client stamps every ``run`` with a per-submission ``taskId``
    (a UUID it minted just before sending) so its dispatcher can filter
    stale ``status`` events from a prior task.  r3-vscode-H1: non-string
    payloads (list, dict, bool, int) are rejected so they never flow
    into the ``status`` envelope echo where they would be compared by
    ``str ==`` against UUID strings on the client.

    Args:
        cmd: The ``run`` command dict.

    Returns:
        The non-empty string ``taskId``, or ``""`` when absent, empty,
        or not a string.
    """
    raw = cmd.get("taskId", "")
    return raw if isinstance(raw, str) else ""


def coerce_budget_override(raw: object) -> float | None:
    """Coerce a wire ``maxBudget`` override to a valid spend cap.

    The per-task budget override arrives straight off the JSON wire.
    Python's ``json`` module parses and emits ``NaN``/``Infinity``, and
    both budget enforcement sites compare spend with ``>= max_budget``
    which is always ``False`` for ``NaN`` — silently disabling the cap.
    Configuration-level code already rejects non-finite budgets, so this
    override path must apply the same guard: only a finite, non-boolean
    number is accepted; anything else returns ``None`` so the caller
    falls back to the configured budget.

    Args:
        raw: The raw ``maxBudget`` value from the ``run`` command.

    Returns:
        The finite budget as a ``float``, or ``None`` when *raw* is
        missing, a bool, non-numeric, or non-finite.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    try:
        value = float(raw)
    except OverflowError:
        # A JSON-valid integer too large for a C double (json accepts
        # thousands of digits) must degrade to the configured budget,
        # not crash the task.
        return None
    if not math.isfinite(value):
        return None
    return value


def decode_attachments(raw: object) -> list[Attachment] | None:
    """Decode the ``attachments`` field of a submit command.

    Each entry carries base64 ``data`` and the ``mimeType`` the browser
    reported.  :meth:`Attachment.from_bytes` transcodes an iPhone camera HEIC
    into JPEG on the way through: the webapp already does that in the browser,
    but a client whose engine cannot decode HEIC still uploads the raw photo,
    and the OpenAI and Anthropic vision APIs reject ``image/heic``.

    Args:
        raw: The command's ``attachments`` value, normally a list of dicts.

    Returns:
        The decoded attachments, or ``None`` when there are none to send.
    """
    if not isinstance(raw, list):
        logger.warning(
            "Ignoring malformed attachments field of type %s",
            type(raw).__name__,
        )
        return None
    if not raw:
        return None
    out: list[Attachment] = []
    for att in raw:
        try:
            data = base64.b64decode(att.get("data", ""))
            mime = att.get("mimeType", "application/octet-stream")
        except Exception:
            logger.warning("Skipping malformed attachment", exc_info=True)
            continue
        out.append(Attachment.from_bytes(data, mime))
    return out


def parse_task_tags(text: str) -> list[str]:
    """Parse ``<task>...</task>`` tags from *text* and return individual tasks.

    When the input contains one or more ``<task>`` blocks with non-empty
    content, each block's content is returned as a separate list element.
    If no valid ``<task>`` blocks are found (or all are empty/whitespace),
    the original *text* is returned as a single-element list so that
    callers can always iterate without special-casing.

    Args:
        text: Input text potentially containing ``<task>...</task>`` tags.

    Returns:
        List of task strings.  Always contains at least one element.
    """
    tasks = [m.strip() for m in re.findall(r"<task>(.*?)</task>", text, re.DOTALL)]
    tasks = [t for t in tasks if t]
    return tasks if tasks else [text]


def _release_worktree_without_merging(
    agent: Any, has_changes: bool,
) -> None:
    """Dispose of *agent*'s pending worktree without touching the main tree.

    Used when a new task starts on a tab that still holds a pending
    worktree while another tab runs a task directly on the main
    working tree.  Auto-merging is unsafe then (it would stash,
    checkout and merge the tree that other task is writing), but
    simply dropping ``agent._wt`` is worse: that handle is the only
    in-memory reference to the worktree, so the directory, the
    ``kiss/wt-*`` branch and its ``branch.<name>.*`` config section
    would leak forever with nothing left to retry the cleanup.

    A change-free worktree is therefore discarded outright — removing
    it touches neither the main working tree's files nor its HEAD.  A
    worktree with real work is preserved as a *branch* the user can
    recover with ``git checkout``, while its on-disk directory is
    committed, removed and pruned.  Either way no artifact is
    orphaned.

    The recovery instructions are only correct when that commit
    actually happened.  When it did not — ``--no-auto-commit``, or a
    pre-commit hook rejecting the commit — the preserve step has
    already recorded exactly where the work was left instead, so that
    message is kept and merely prefixed with the reason no merge was
    attempted.  Telling the user to ``git checkout`` a branch that
    does not carry their work would send them looking in the wrong
    place.

    Which of the two happened is read from the outcome the preserve
    step reports, never from the warning slot: a broadcast that failed
    puts an older warning back there (``_flush_warnings``), and that
    stale text describes a different worktree entirely.  For the same
    reason a preserve that found nothing to do reports nothing at all.

    Args:
        agent: The tab's worktree agent, known to have ``_wt_pending``.
        has_changes: Whether the worktree contains work worth keeping.
    """
    branch = agent._wt_branch
    if not has_changes:
        agent.discard()
        return
    if not agent._preserve_pending_worktree_for_review():
        # No worktree was pending after all, so there is no branch to
        # name and nothing was preserved.  Saying anything here would
        # either invent a branch or recycle a warning left over from an
        # older worktree.
        return
    reason = (
        f"Could not auto-merge branch '{branch}' because another task "
        "is running on the main working tree."
    )
    if agent._last_preserve_outcome is _WorktreeCleanupOutcome.COMMITTED_AND_REMOVED:
        agent._set_warnings(merge=(
            f"{reason} Your work is committed on that branch; recover "
            f"it with: git checkout {branch}"
        ))
        return
    with agent._warning_lock:
        stranded = agent._merge_conflict_warning
    agent._set_warnings(merge=f"{reason} {stranded}" if stranded else reason)


def _wt_merge_on_repo(tab: _RunningAgentState, repo: Path | None) -> bool:
    """True when *tab* is merging a worktree into *repo*'s main tree.

    Used by the non-worktree task admission gate: starting a task
    directly on a main working tree that a concurrent worktree merge
    is stashing/checking-out/merging would interleave writes, so the
    start is refused — but only when the two really share a main tree.
    A merge in a *different* repository never touches *repo*.

    Args:
        tab: The per-tab state of a potentially merging tab.
        repo: The main repository root the starting task will write,
            or ``None`` when the task's ``work_dir`` is not in a git
            repo (then no worktree merge can conflict with it).

    Returns:
        True when *tab* holds a worktree merge whose repository is
        *repo*, whose worktree directory itself is *repo* (the
        starting task would run inside the directory the merge is
        about to remove), or whose repository cannot be determined —
        the conservative pre-repo-aware behavior.
    """
    if not (tab.is_merging and tab.use_worktree):
        return False
    if repo is None:
        return False
    agent = tab.agent
    merge_root = getattr(agent, "_repo_root", None) if agent is not None else None
    if merge_root is None:
        # The merging tab's repository is unknown (e.g. its agent has
        # already been disposed); keep the conservative refusal.
        return True
    try:
        repo_resolved = repo.resolve()
        if Path(merge_root).resolve() == repo_resolved:
            return True
        # A task starting INSIDE the very worktree being merged (its
        # work_dir's toplevel is the linked worktree directory) must
        # also be refused: the merge auto-commits and removes that
        # directory out from under the task.
        merge_wt_dir = getattr(agent, "_wt_dir", None)
        return bool(
            merge_wt_dir is not None
            and Path(merge_wt_dir).resolve() == repo_resolved
        )
    except OSError:  # pragma: no cover — unresolvable path
        return True


_STOP_SENTINEL: object = object()


class _TaskRunnerMixin:
    """Task-lifecycle methods (run, stop, user-question callback)."""

    if TYPE_CHECKING:
        printer: JsonPrinter
        work_dir: str
        _state_lock: threading.RLock
        _default_model: str
        _tab_chat_views: dict[str, str]
        _tab_opened_task_ids: dict[str, str]
        _pending_user_answer_tasks: dict[int, str]

        def _get_tab(self, tab_id: str) -> _RunningAgentState: ...
        def _any_non_wt_running(
            self, repo_root: Path | None = None,
        ) -> bool: ...
        def _dispose_if_closed(self, tab_id: str) -> None: ...
        def _prepare_and_start_merge(
            self,
            work_dir: str,
            pre_hunks: dict[str, list[tuple[int, int, int, int]]] | None = None,
            pre_untracked: set[str] | None = None,
            pre_file_hashes: dict[str, str] | None = None,
            base_ref: str = "HEAD",
            tab_id: str = "",
        ) -> bool: ...
        def _main_dirty_files(self, work_dir: str = "") -> list[str]: ...
        def _broadcast_autocommit_prompt(
            self,
            tab_id: str,
            work_dir: str = "",
        ) -> None: ...
        def _handle_autocommit_action(
            self,
            action: str,
            tab_id: str = "",
            *,
            work_dir: str = "",
        ) -> None: ...
        def _handle_worktree_action(
            self,
            action: str,
            tab_id: str = "",
            *,
            internal: bool = False,
            already_claimed: bool = False,
        ) -> dict[str, Any]: ...
        def _present_pending_worktree(
            self,
            tab_id: str,
            *,
            try_merge_review: bool,
            discard_if_empty: bool = True,
        ) -> bool: ...
        def _get_worktree_changed_files(self, tab_id: str = "") -> list[str]: ...
        def _extract_result_summary(self) -> str: ...
        def _generate_followup_async(
            self,
            task: str,
            result: str,
            task_id: str | None,
        ) -> None: ...
        def _refresh_files_after_task(self, work_dir: str = "") -> None: ...

    def _run_task(self, cmd: dict[str, Any]) -> None:
        """Run the agent with the given task.

        An outer try/finally guarantees that ``status: running: False``
        is **always** broadcast when this method exits, regardless of
        which code-path is taken.
        """
        tab_id = cmd.get("tabId", "")
        start_ms = int(time.time() * 1000)
        cmd["_start_ms"] = start_ms
        client_task_id = _client_task_id_of(cmd)
        try:
            status_start: dict[str, Any] = {
                "type": "status",
                "running": True,
                "tabId": tab_id,
                "startTs": start_ms,
            }
            if client_task_id:
                status_start["taskId"] = client_task_id
            self.printer.broadcast(status_start)
            self._run_task_inner(cmd)
        except BaseException as exc:
            logger.warning(
                "Task setup failed: tab_id=%s error=%s",
                tab_id,
                exc,
                exc_info=True,
            )
            if isinstance(exc, KeyboardInterrupt):
                setup_fail_text = "Task stopped by user"
            else:
                setup_fail_text = f"Task failed: {type(exc).__name__}: {exc}"
            self.printer.broadcast(
                {
                    "type": "result",
                    "text": setup_fail_text,
                    "success": False,
                    "total_tokens": 0,
                    "cost": "$0.0000",
                    "step_count": 0,
                    "tabId": tab_id,
                }
            )
        finally:
            with self._state_lock:
                tab = _RunningAgentState.running_agent_states.get(tab_id)
                owns_tab = tab is None or (
                    tab.task_thread is None or tab.task_thread is threading.current_thread()
                )
                if tab is not None and owns_tab:
                    tab.task_thread = None
                    tab.stop_event = None
                    tab.user_answer_queue = None
                    tab.pending_user_messages.clear()
                    tab.unattributed_prompt_echoes.clear()
                    tab.is_task_active = False
                    tab.is_running_non_wt = False
                    tab.non_wt_repo_root = None
                    tab.interrupted_by_shutdown = False
                    if tab.agent is not None:
                        tab.last_task_id = (
                            getattr(tab.agent, "_last_task_id", None) or tab.last_task_id
                        )
                        if tab.use_worktree and getattr(
                            tab.agent,
                            "_wt_pending",
                            False,
                        ):
                            pass
                        else:
                            tab.agent = None
                            tab.use_worktree = False
                task_id_for_end = tab.last_task_id if tab is not None and owns_tab else None
                if owns_tab:
                    status_end: dict[str, Any] = {
                        "type": "status",
                        "running": False,
                        "tabId": tab_id,
                    }
                    if client_task_id:
                        status_end["taskId"] = client_task_id
                    self.printer.broadcast(status_end)
                    self._restore_user_model_pick(tab_id)
            self._broadcast_status_end_to_viewers(
                task_id_for_end,
                tab_id,
                client_task_id=client_task_id,
            )
            self._dispose_if_closed(tab_id)
            # The binding is per THREAD — that is how a model stream
            # learns about a stop — so it has to end with the run, or
            # anything this thread does next would inherit it.
            self.printer._thread_local.stop_event = None

    def _restore_user_model_pick(self, tab_id: str) -> None:
        """Put the user's own model back in *tab_id*'s picker.

        A finished agent's ``set_model`` override is display-only, so
        the moment the task stops the picker must show the user's own
        choice again — otherwise the next task they launch would
        silently inherit whatever model the agent happened to end on.

        The model is the one selected in *that* tab: the picker is
        per-tab, so a pick made in another window is none of this tab's
        business.  Nothing is emitted unless an agent actually took the
        picker over.

        Args:
            tab_id: The tab whose picker should be restored.
        """
        with self._state_lock:
            state = _RunningAgentState.running_agent_states.get(tab_id)
            model = (state.selected_model if state is not None else "") or (
                _load_last_model() or self._default_model
            )
        self.printer.restore_model_pick(model, tab_id)

    def _broadcast_status_end_to_viewers(
        self,
        task_id: str | None,
        launcher_tab_id: str,
        *,
        client_task_id: str = "",
    ) -> None:
        """Broadcast ``status running=False`` to every viewer subscribed
        to *task_id*, excluding the launcher tab.

        ``status`` events carry an explicit ``tabId`` and are routed
        verbatim by the printer's transport (no per-task subscriber
        fan-out).  Without an explicit per-viewer broadcast, a tab
        that joined the running task via ``_replay_session`` /
        ``_reattach_running_chat`` (history-resume click) or via
        ``_subscribe_chat_viewers`` (idle viewer of the chat) would
        never receive a ``running=False`` event stamped with its own
        tab id — its frontend would keep ``isRunning=true`` forever,
        the pulsing tab-title indicator would not stop, and follow-up
        user input would keep being routed as ``appendUserMessage``
        against a now-finished task (and dropped).

        Args:
            task_id: The finished task's ``task_history`` row id.
                ``None`` when the worker thread unwound before a task
                id was allocated (very early failure path); in that
                case there is nothing to fan out.
            launcher_tab_id: The tab the task was launched in — its
                ``running=False`` broadcast is emitted directly by
                the caller, so skip it here to avoid duplication.
        """
        if task_id is None:
            return
        task_key = JsonPrinter._coerce_task_id(task_id)
        for viewer_tab_id in self.printer._fanout_targets(task_id):
            if viewer_tab_id == launcher_tab_id:
                continue
            with self._state_lock:
                viewer_state = _RunningAgentState.running_agent_states.get(
                    viewer_tab_id,
                )
                if viewer_state is not None and viewer_state.is_task_active:
                    viewer_task = (
                        JsonPrinter._coerce_task_id(
                            getattr(viewer_state.agent, "_last_task_id", None),
                        )
                        if viewer_state.agent is not None
                        else ""
                    )
                    if viewer_task != task_key:
                        continue
            payload: dict[str, Any] = {
                "type": "status",
                "running": False,
                "tabId": viewer_tab_id,
            }
            if client_task_id:
                payload["taskId"] = client_task_id
            self.printer.broadcast(payload)
            self._restore_user_model_pick(viewer_tab_id)

    @staticmethod
    def _capture_pre_snapshot(
        work_dir: str,
        repo: Path | None,
        tab_id: str,
    ) -> tuple[
        str | None,
        dict[str, list[tuple[int, int, int, int]]],
        set[str],
        dict[str, str] | None,
    ]:
        """Capture pre-task git snapshot for non-worktree merge view.

        When *repo* is not None, acquires ``repo_lock`` for atomicity.

        Args:
            work_dir: Repository root directory.
            repo: Repo root Path (None when not in a git repo).
            tab_id: Frontend tab identifier for per-tab isolation.

        Returns:
            ``(head_sha, hunks, untracked, file_hashes)`` tuple.
        """
        with repo_lock(repo) if repo else nullcontext():
            head = GitWorktreeOps.head_sha(repo) if repo else None
            hunks = _parse_diff_hunks(work_dir)
            untracked = _capture_untracked(work_dir)
            hashes = _snapshot_files(
                work_dir,
                set(hunks.keys()) | untracked,
            )
            _save_untracked_base(
                work_dir,
                untracked | set(hunks.keys()),
                tab_id=tab_id,
            )
            return head, hunks, untracked, hashes

    def _broadcast_early_prompts(
        self,
        prompt: str,
        active_file: str | None,
        tab_id: str,
    ) -> None:
        """Broadcast optimistic ``system_prompt``/``prompt`` panels at submit.

        Emitted before the slow pre-run steps (git pre-snapshot,
        worktree creation, chat-context loading, model/tool setup) so
        the chat webview shows the submitted prompt and the system
        prompt right away.  The events are broadcast-only (``taskId:
        ""`` — never recorded or persisted) and flagged ``early`` so
        the frontend replaces them in place once the authoritative
        events from ``KISSAgent.run`` arrive.  The system-prompt text
        mirrors ``SorcarAgent.run``'s ``system_instructions``
        (``SYSTEM_PROMPT`` plus the active-editor-file line); the later
        authoritative event additionally carries the per-run
        ``IMPORTANT_INSTRUCTIONS`` suffix.

        Args:
            prompt: The raw user prompt as submitted.
            active_file: Path of the file open in the editor, if any.
            tab_id: Frontend tab id that owns the run.
        """
        from kiss.core.base import SYSTEM_PROMPT

        system_text = SYSTEM_PROMPT
        if active_file:
            system_text += f"\n\n- The path of the file open in the editor is {active_file}"
        for etype, text in (
            ("system_prompt", system_text),
            ("prompt", prompt),
        ):
            self.printer.broadcast(
                {
                    "type": etype,
                    "text": text,
                    "tabId": tab_id,
                    "taskId": "",
                    "early": True,
                }
            )

    def _run_task_inner(self, cmd: dict[str, Any]) -> None:
        """Inner implementation of _run_task (without the status guarantee)."""
        prompt = cmd.get("prompt", "")
        work_dir = cmd.get("workDir") or self.work_dir
        active_file = cmd.get("activeFile")
        attachments = decode_attachments(cmd.get("attachments", []))
        start_ms = int(cmd.get("_start_ms") or 0)

        tab_id = cmd.get("tabId", "")
        tab = self._get_tab(tab_id)
        model = cmd.get("model") or tab.selected_model

        assert tab.agent is not None
        tab.agent._tab_id = tab_id
        tab.agent._task_start_ms = start_ms
        if tab.chat_id:
            tab.agent._chat_id = tab.chat_id
        tab.chat_id = getattr(tab.agent, "chat_id", "") or tab.chat_id

        available = get_available_models()
        if not available or (model and model not in available):
            no_model_msg = "No model available.  Set at least one API key in the environment."
            self.printer.broadcast(
                {
                    "type": "result",
                    "text": no_model_msg,
                    "success": False,
                    "total_tokens": 0,
                    "cost": "$0.0000",
                    "step_count": 0,
                    "tabId": tab_id,
                }
            )
            return

        with self._state_lock:
            if tab.is_merging:
                self.printer.broadcast(
                    {
                        "type": "error",
                        "text": "Cannot run a task while merge review is in progress."
                        " Accept or reject all changes first.",
                        "tabId": tab_id,
                    }
                )
                return
            tab.use_worktree = bool(cmd.get("useWorktree", False))
            tab.use_parallel = bool(cmd.get("useParallel", False))
            tab.auto_commit_mode = bool(cmd.get("autoCommit", False))
            tab.is_task_active = True
            stop_event = tab.stop_event
            use_worktree = tab.use_worktree
        self.printer._thread_local.stop_event = stop_event

        self._broadcast_early_prompts(prompt, active_file, tab_id)

        pre_hunks: dict[str, list[tuple[int, int, int, int]]] = {}
        pre_untracked: set[str] = set()
        pre_file_hashes: dict[str, str] | None = None
        pre_head_sha: str | None = None
        if not use_worktree:
            repo = GitWorktreeOps.discover_repo(Path(work_dir))
            with repo_lock(repo) if repo else nullcontext(), self._state_lock:
                if any(
                    _wt_merge_on_repo(t, repo)
                    for t in _RunningAgentState.running_agent_states.values()
                ):
                    tab.is_task_active = False
                    self.printer.broadcast(
                        {
                            "type": "error",
                            "text": "A worktree merge is in progress. "
                            "Wait for it to finish before starting a task.",
                            "tabId": tab_id,
                        }
                    )
                    return
                tab.is_running_non_wt = True
                tab.non_wt_repo_root = repo.resolve() if repo else None
            try:
                pre_head_sha, pre_hunks, pre_untracked, pre_file_hashes = (
                    self._capture_pre_snapshot(work_dir, repo, tab_id)
                )
            except BaseException:
                with self._state_lock:
                    tab.is_running_non_wt = False
                    tab.non_wt_repo_root = None
                raise

        if use_worktree and getattr(tab.agent, "_wt_pending", False):
            with self._state_lock:
                main_tree_busy = self._any_non_wt_running(
                    getattr(tab.agent, "_repo_root", None),
                )
            if main_tree_busy:
                _release_worktree_without_merging(
                    tab.agent, bool(self._get_worktree_changed_files(tab_id)),
                )

        with self._state_lock:
            opened_task_id = self._tab_opened_task_ids.pop(tab_id, "")
        if opened_task_id:
            tab.agent.resume_from_task_id(opened_task_id)

        logger.info(
            "Task started: tab_id=%s model=%s use_worktree=%s auto_commit=%s prompt=%r",
            tab_id,
            model,
            use_worktree,
            tab.auto_commit_mode,
            prompt[:200],
        )
        result_summary = "Agent Failed Abruptly"
        task_end_event: dict[str, Any] | None = None
        sub_start_ms = start_ms
        sub_tokens_base = int(getattr(tab.agent, "total_tokens_used", 0) or 0)
        sub_cost_base = float(getattr(tab.agent, "budget_used", 0.0) or 0.0)
        sub_steps_base = int(getattr(tab.agent, "total_steps", 0) or 0)
        agent_returned: str = ""
        try:
            tab.task_history_id = None
            subtasks = parse_task_tags(prompt)
            from kiss.core.vscode_config import (
                build_model_config,
                load_config,
            )

            _vcfg = load_config()
            _cfg_budget = float(_vcfg.get("max_budget", 100))
            _cfg_web = _vcfg.get("use_web_browser", True)
            _model_config = build_model_config(_vcfg)
            _agent_budget = coerce_budget_override(cmd.get("maxBudget"))
            _raw_web = cmd.get("webTools")
            _agent_web = _raw_web if isinstance(_raw_web, bool) else None
            _raw_model_config = cmd.get("modelConfig")
            _agent_model_config = (
                _raw_model_config
                if isinstance(_raw_model_config, dict)
                else None
            )

            on_task_id_allocated = partial(
                self._subscribe_chat_viewers,
                source_tab_id=tab_id,
                start_ms=start_ms,
                client_task_id=_client_task_id_of(cmd),
            )

            client_tools = load_tools_file(cmd.get("toolsFile"))

            for subtask_index, task_prompt in enumerate(subtasks):
                tab.last_user_prompt = task_prompt
                tab.last_result_summary = ""
                if subtask_index > 0:
                    sub_start_ms = int(time.time() * 1000)
                sub_tokens_base = int(
                    getattr(tab.agent, "total_tokens_used", 0) or 0,
                )
                sub_cost_base = float(
                    getattr(tab.agent, "budget_used", 0.0) or 0.0,
                )
                sub_steps_base = int(
                    getattr(tab.agent, "total_steps", 0) or 0,
                )
                subtask_failed = False
                subtask_exc: BaseException | None = None
                try:
                    agent_returned = tab.agent.run(
                        prompt_template=task_prompt,
                        model_name=model,
                        work_dir=work_dir,
                        printer=self.printer,
                        current_editor_file=active_file,
                        attachments=attachments,
                        ask_user_question_callback=self._ask_user_question,
                        is_parallel=tab.use_parallel,
                        use_worktree=use_worktree,
                        max_budget=(_agent_budget if _agent_budget is not None else _cfg_budget),
                        web_tools=(_agent_web if _agent_web is not None else _cfg_web),
                        model_config=(
                            _agent_model_config
                            if _agent_model_config is not None
                            else _model_config
                        ),
                        tools=client_tools,
                        _skip_persistence=True,
                        _subscribe_tab_id=tab_id,
                        _on_task_id_allocated=on_task_id_allocated,
                    )
                    _run_parsed = parse_result_yaml(agent_returned) if agent_returned else None
                    if _run_parsed and _run_parsed.get("summary"):
                        result_summary = str(_run_parsed["summary"])
                    else:
                        result_summary = self._extract_result_summary() or "No summary available"
                    task_end_event = {"type": "task_done"}
                    logger.info(
                        "Agent returned: tab_id=%s task_id=%s summary=%r",
                        tab_id,
                        tab.task_history_id,
                        result_summary[:200],
                    )
                except KeyboardInterrupt as ki:
                    result_summary, task_end_event = self._cancel_outcome(tab)
                    subtask_failed = True
                    subtask_exc = ki
                    logger.info(
                        "%s: tab_id=%s task_id=%s",
                        result_summary,
                        tab_id,
                        tab.task_history_id,
                    )
                except Exception as e:
                    result_summary = f"Task failed: {e}"
                    task_end_event = {"type": "task_error", "text": str(e)}
                    subtask_failed = True
                    subtask_exc = e
                    logger.warning(
                        "Task failed: tab_id=%s task_id=%s error=%s",
                        tab_id,
                        tab.task_history_id,
                        e,
                        exc_info=True,
                    )
                finally:
                    tab.task_history_id = getattr(
                        tab.agent,
                        "_last_task_id",
                        None,
                    )
                    tab.last_result_summary = result_summary
                if subtask_failed:
                    tokens_delta, cost_delta, steps_delta = self._subtask_metric_deltas(
                        tab.agent,
                        sub_tokens_base,
                        sub_cost_base,
                        sub_steps_base,
                    )
                    already_broadcast = bool(
                        getattr(subtask_exc, "terminal_result_broadcast", False)
                    )
                    if not already_broadcast:
                        failure_result: dict[str, Any] = {
                            "type": "result",
                            "text": result_summary,
                            "success": False,
                            "total_tokens": tokens_delta,
                            "cost": f"${cost_delta:.4f}",
                            "step_count": steps_delta,
                        }
                        if tab.task_history_id:
                            failure_result["taskId"] = str(tab.task_history_id)
                        else:
                            failure_result["tabId"] = tab_id
                        self.printer.broadcast(failure_result)
                    break
                if subtask_index < len(subtasks) - 1:
                    self._persist_subtask_row(
                        tab,
                        task_prompt=task_prompt,
                        result_summary=result_summary,
                        model=model,
                        work_dir=work_dir,
                        use_worktree=use_worktree,
                        sub_start_ms=sub_start_ms,
                        sub_tokens_base=sub_tokens_base,
                        sub_cost_base=sub_cost_base,
                        sub_steps_base=sub_steps_base,
                    )
        except BaseException as _outer_exc:
            if result_summary == "Agent Failed Abruptly":
                if isinstance(_outer_exc, KeyboardInterrupt):
                    result_summary, _cancel_event = self._cancel_outcome(tab)
                    task_end_event = task_end_event or _cancel_event
                else:
                    _exc_name = type(_outer_exc).__name__
                    result_summary = f"Task failed: {_exc_name}: {_outer_exc}"
                    task_end_event = task_end_event or {
                        "type": "task_error",
                        "text": f"{_exc_name}: {_outer_exc}",
                    }
            else:
                task_end_event = task_end_event or {"type": "task_stopped"}
            tab.last_result_summary = result_summary
            _agent_for_metrics = getattr(tab, "agent", None)
            tokens_delta, cost_delta, steps_delta = self._subtask_metric_deltas(
                _agent_for_metrics,
                sub_tokens_base,
                sub_cost_base,
                sub_steps_base,
            )
            outer_failure_result: dict[str, Any] = {
                "type": "result",
                "text": result_summary,
                "success": False,
                "total_tokens": tokens_delta,
                "cost": f"${cost_delta:.4f}",
                "step_count": steps_delta,
            }
            if tab.task_history_id:
                outer_failure_result["taskId"] = str(tab.task_history_id)
            else:
                outer_failure_result["tabId"] = tab_id
            self.printer.broadcast(outer_failure_result)
        finally:
            end_event_broadcast = False
            try:
                _agent_parsed = parse_result_yaml(agent_returned) if agent_returned else None
                _agent_reported_failure = bool(
                    _agent_parsed and _agent_parsed.get("success") is False
                )
                task_failed = bool(
                    (
                        task_end_event
                        and task_end_event.get("type")
                        in ("task_error", "task_stopped", "task_interrupted")
                    )
                    or _agent_reported_failure
                )
                effective_auto_commit = tab.auto_commit_mode and not task_failed
                if not use_worktree:
                    try:
                        if effective_auto_commit:
                            self._handle_autocommit_action(
                                "commit",
                                tab_id,
                                work_dir=work_dir,
                            )
                        else:
                            merge_started = self._prepare_and_start_merge(
                                work_dir,
                                pre_hunks,
                                pre_untracked,
                                pre_file_hashes,
                                base_ref=pre_head_sha or "HEAD",
                                tab_id=tab_id,
                            )
                            if not merge_started:
                                self._broadcast_autocommit_prompt(
                                    tab_id,
                                    work_dir,
                                )
                    except BaseException:  # pragma: no cover — merge view error handler
                        logger.debug("Merge view error", exc_info=True)
                    finally:
                        with self._state_lock:
                            tab.is_running_non_wt = False
                            tab.non_wt_repo_root = None
                assert task_end_event is not None
                _append_chat_event(
                    task_end_event,
                    task_id=tab.task_history_id,
                    task=prompt,
                )
                _save_task_result(
                    result=result_summary,
                    task_id=tab.task_history_id,
                    task=prompt,
                )
                logger.info(
                    "Task result persisted: task_id=%s result=%r",
                    tab.task_history_id,
                    result_summary[:200],
                )
                from kiss.core._version import __version__

                end_ms = int(time.time() * 1000)
                tokens_delta, cost_delta, steps_delta = self._subtask_metric_deltas(
                    tab.agent,
                    sub_tokens_base,
                    sub_cost_base,
                    sub_steps_base,
                )
                _save_task_extra(
                    build_task_extra_payload(
                        model=model,
                        work_dir=work_dir,
                        version=__version__,
                        tokens=tokens_delta,
                        cost=round(cost_delta, 6),
                        steps=steps_delta,
                        is_parallel=tab.use_parallel,
                        is_worktree=use_worktree,
                        auto_commit_mode=tab.auto_commit_mode,
                        start_ms=sub_start_ms,
                        end_ms=end_ms,
                    ),
                    task_id=tab.task_history_id,
                )
                self.printer.broadcast({"type": "tasks_updated"})
                if use_worktree and getattr(tab.agent, "_wt_pending", False):
                    if task_failed:
                        tab.agent._pending_review = True
                    try:
                        if effective_auto_commit:
                            if self._get_worktree_changed_files(tab_id):
                                action = "merge"
                            else:
                                action = "discard"
                            result = self._handle_worktree_action(
                                action,
                                tab_id,
                                internal=True,
                            )
                            self.printer.broadcast_tab_ui(
                                {
                                    "type": "worktree_result",
                                    "tabId": tab_id,
                                    **result,
                                }
                            )
                        else:
                            self._present_pending_worktree(
                                tab_id,
                                try_merge_review=True,
                                discard_if_empty=False,
                            )
                    except BaseException:
                        logger.debug("Worktree merge review error", exc_info=True)
                with self._state_lock:
                    tab.is_task_active = False
                self.printer.broadcast(
                    {
                        **task_end_event,
                        "tabId": tab_id,
                        "startTs": start_ms,
                        "endTs": end_ms,
                    }
                )
                end_event_broadcast = True
                self._refresh_files_after_task(work_dir)
                logger.info(
                    "Task lifecycle complete: tab_id=%s task_id=%s elapsed_ms=%d event_type=%s",
                    tab_id,
                    tab.task_history_id,
                    end_ms - start_ms,
                    (task_end_event or {}).get("type", "none"),
                )
                hist_id = tab.task_history_id
                if hist_id is not None:
                    # S3-08: drop the printer's persist-agent BEFORE
                    # starting the follow-up thread, so the follow-up
                    # broadcast is never auto-persisted and the explicit
                    # ``_append_chat_event`` inside the follow-up thread
                    # is the single, scheduling-independent persistence
                    # path.  ``cleanup_task`` keeps the subscriber set
                    # alive for a linger period, so the broadcast still
                    # fans out to the originating tab.
                    self.printer.cleanup_task(hist_id)
                    tab.task_history_id = None
                    self._generate_followup_async(
                        prompt,
                        result_summary,
                        hist_id,
                    )
            except BaseException:  # pragma: no cover — cleanup interrupted
                logger.debug("Cleanup interrupted", exc_info=True)
                # Only emit the terminal event if the normal path did
                # not already broadcast it — a failure AFTER that
                # broadcast (refresh/follow-up/cleanup) must not send
                # the same terminal event twice.
                if task_end_event and not end_event_broadcast:
                    try:
                        self.printer.broadcast(
                            {
                                **task_end_event,
                                "tabId": tab_id,
                                "startTs": start_ms,
                                "endTs": int(time.time() * 1000),
                            }
                        )
                    except BaseException:
                        logger.debug(
                            "End-event broadcast failed",
                            exc_info=True,
                        )
            finally:
                # S3-07: mandatory lifecycle/identity cleanup lives in
                # its own ``finally`` so an exception anywhere in the
                # persistence/merge/broadcast block above can no longer
                # leave the tab flagged active or retain its task id,
                # the printer's persist-agent/recording, or the worker
                # thread-local task id.
                with self._state_lock:
                    tab.is_task_active = False
                    if not use_worktree:
                        tab.is_running_non_wt = False
                        tab.non_wt_repo_root = None
                if tab.task_history_id is not None:
                    try:
                        self.printer.cleanup_task(tab.task_history_id)
                    except BaseException:
                        logger.debug(
                            "cleanup_task failed",
                            exc_info=True,
                        )
                    tab.task_history_id = None
                tl = getattr(self.printer, "_thread_local", None)
                if tl is not None:
                    tl.task_id = ""

    def _persist_subtask_row(
        self,
        tab: _RunningAgentState,
        *,
        task_prompt: str,
        result_summary: str,
        model: str,
        work_dir: str,
        use_worktree: bool,
        sub_start_ms: int,
        sub_tokens_base: int,
        sub_cost_base: float,
        sub_steps_base: int,
    ) -> None:
        """Persist a completed (non-final) subtask's own history row.

        W2-F2: a multi-``<task>`` prompt runs ``tab.agent.run`` once
        per subtask and each run allocates its OWN ``task_history``
        row (with ``_skip_persistence=True`` suppressing the agent's
        internal result save).  The task-level cleanup ``finally`` in
        :meth:`_run_task_inner` persists only the LAST subtask's row,
        so every earlier row must be completed here, right after its
        subtask succeeds: end event, result summary, and the ``extra``
        payload with per-subtask metric deltas and timestamps.

        Failures are logged and swallowed — a persistence hiccup for
        one subtask must not abort the remaining subtasks.

        Args:
            tab: The owning tab state (``tab.task_history_id`` is the
                just-finished subtask's row id).
            task_prompt: The subtask's own prompt text.
            result_summary: The subtask's result summary.
            model: Model name used for the task.
            work_dir: Working directory the task ran from.
            use_worktree: Whether the task ran inside a worktree.
            sub_start_ms: The subtask's start timestamp (ms epoch).
            sub_tokens_base: Agent token counter before the subtask.
            sub_cost_base: Agent budget counter before the subtask.
            sub_steps_base: Agent step counter before the subtask.
        """
        try:
            from kiss.core._version import __version__

            _append_chat_event(
                {"type": "task_done"},
                task_id=tab.task_history_id,
                task=task_prompt,
            )
            _save_task_result(
                result=result_summary,
                task_id=tab.task_history_id,
                task=task_prompt,
            )
            tokens_delta, cost_delta, steps_delta = self._subtask_metric_deltas(
                tab.agent,
                sub_tokens_base,
                sub_cost_base,
                sub_steps_base,
            )
            _save_task_extra(
                build_task_extra_payload(
                    model=model,
                    work_dir=work_dir,
                    version=__version__,
                    tokens=tokens_delta,
                    cost=round(cost_delta, 6),
                    steps=steps_delta,
                    is_parallel=tab.use_parallel,
                    is_worktree=use_worktree,
                    auto_commit_mode=tab.auto_commit_mode,
                    start_ms=sub_start_ms,
                    end_ms=int(time.time() * 1000),
                ),
                task_id=tab.task_history_id,
            )
            self.printer.broadcast({"type": "tasks_updated"})
            if tab.task_history_id is not None:
                self.printer.cleanup_task(tab.task_history_id)
            logger.info(
                "Subtask result persisted: task_id=%s result=%r",
                tab.task_history_id,
                result_summary[:200],
            )
        except Exception:
            logger.warning(
                "Failed to persist subtask row: task_id=%s",
                tab.task_history_id,
                exc_info=True,
            )

    @staticmethod
    def _subtask_metric_deltas(
        agent: object,
        tokens_base: int,
        cost_base: float,
        steps_base: int,
    ) -> tuple[int, float, int]:
        """Return a subtask's ``(tokens, cost, steps)`` consumption deltas.

        The agent's ``total_tokens_used`` / ``budget_used`` /
        ``total_steps`` counters are CUMULATIVE — the agent object is
        reused across tasks on the same tab when a worktree is left
        pending — so per-subtask figures (used by the failure ``result``
        broadcasts, mirroring the W2-F2 delta arithmetic of the
        persisted ``extra`` payload) must subtract the baselines
        captured just before the subtask's ``run``.  All reads are
        defensive (``getattr`` with a zero default) so a nulled agent
        yields zero deltas via the ``max`` clamps.

        RelentlessAgent-derived agents accumulate completed steps into
        ``total_steps`` and leave ``step_count`` at 0; plain agents do
        the opposite.  The steps delta therefore falls back to
        ``step_count`` when the ``total_steps`` delta is 0.

        Args:
            agent: The (possibly ``None``) agent to read counters from.
            tokens_base: ``total_tokens_used`` before the subtask ran.
            cost_base: ``budget_used`` before the subtask ran.
            steps_base: ``total_steps`` before the subtask ran.

        Returns:
            ``(tokens_delta, cost_delta, steps_delta)`` clamped at 0.
        """
        tokens = max(
            0,
            int(getattr(agent, "total_tokens_used", 0) or 0) - tokens_base,
        )
        cost = max(
            0.0,
            float(getattr(agent, "budget_used", 0.0) or 0.0) - cost_base,
        )
        steps = max(
            0,
            int(getattr(agent, "total_steps", 0) or 0) - steps_base,
        ) or int(getattr(agent, "step_count", 0) or 0)
        return tokens, cost, steps

    @staticmethod
    def _cancel_outcome(
        tab: _RunningAgentState,
    ) -> tuple[str, dict[str, Any]]:
        """Resolve the result label + end event for a cancelled task.

        A task is cancelled by injecting a ``KeyboardInterrupt`` into
        the worker thread.  Two unrelated paths do this and are
        otherwise indistinguishable at the ``except KeyboardInterrupt``
        site:

        * the user clicking "Stop" (:meth:`_stop_task`), and
        * a graceful server shutdown on ``SIGTERM`` — e.g. a daemon /
          LaunchAgent restart triggered by a KISS Sorcar extension
          update — which routes through
          :meth:`RemoteAccessServer._stop_active_agent_tasks`.

        The shutdown path sets :attr:`_RunningAgentState.interrupted_by_shutdown`
        on the tab *before* injecting the interrupt, so this flag is the
        single source of truth.  Returning the shutdown-specific label
        and ``task_interrupted`` event prevents the long-standing
        mislabelling where a server restart was reported to the user as
        "Task stopped by user".

        Args:
            tab: The running tab state whose task was cancelled.

        Returns:
            ``(result_summary, task_end_event)`` — the persisted result
            string and the lifecycle end-event dict.
        """
        if tab.interrupted_by_shutdown:
            return (
                "Task interrupted by server restart/shutdown",
                {"type": "task_interrupted"},
            )
        return ("Task stopped by user", {"type": "task_stopped"})

    def _subscribe_chat_viewers(
        self,
        task_id: str,
        chat_id: str,
        *,
        source_tab_id: str,
        start_ms: int,
        client_task_id: str = "",
    ) -> None:
        """Subscribe every tab that has *chat_id* open to a new task's stream.

        Invariant: when a task is running on a chat, every tab — in
        any VS Code window or remote browser window — that has that
        chat open must see the task's events streaming live.  Tabs
        that open the chat WHILE the task is already running are
        handled by ``_replay_session`` → ``_reattach_running_chat``;
        this hook covers the tabs that opened the chat BEFORE the
        task started (e.g. the tab that ran the previous task of the
        chat, or a history viewer in a sibling window).

        Called via ``_on_task_id_allocated`` from
        :meth:`ChatSorcarAgent.run` as soon as the run's
        ``task_history`` row id exists, before any agent event is
        broadcast.  For each viewer tab (excluding the launcher,
        which ``ChatSorcarAgent.run`` already subscribed via
        ``_subscribe_tab_id``) it mirrors the launcher's start
        sequence: ``clear`` (resets the viewer's replayed content and
        per-tab stream state) followed by ``status running=True``
        (flips the viewer's spinner / stop button), after which the
        printer's per-subscriber fan-out delivers every live event.

        Args:
            task_id: The freshly allocated ``task_history`` row id.
            chat_id: The chat id the task runs on.
            source_tab_id: The tab that launched the task (skipped).
            start_ms: The agent's start timestamp (ms since epoch),
                echoed on the ``status`` broadcast so viewer tabs
                anchor their "Running …" timer correctly.
        """
        if not chat_id:
            return
        with self._state_lock:
            viewers = []
            for viewer_tab_id, viewed_chat_id in self._tab_chat_views.items():
                if viewed_chat_id != chat_id or viewer_tab_id == source_tab_id:
                    continue
                viewer_state = _RunningAgentState.running_agent_states.get(
                    viewer_tab_id,
                )
                if viewer_state is not None and viewer_state.is_task_active:
                    continue
                viewers.append(viewer_tab_id)
        for viewer_tab_id in viewers:
            self.printer.subscribe_tab(task_id, viewer_tab_id)
            self.printer.broadcast(
                {
                    "type": "clear",
                    "chat_id": chat_id,
                    "tabId": viewer_tab_id,
                }
            )
            viewer_status: dict[str, Any] = {
                "type": "status",
                "running": True,
                "tabId": viewer_tab_id,
                "startTs": start_ms,
            }
            if client_task_id:
                viewer_status["taskId"] = client_task_id
            self.printer.broadcast(viewer_status)

    def _stop_task(self, tab_id: str = "") -> None:
        """Signal the agent to stop.

        Sets the cooperative stop event and, if the task thread doesn't
        exit promptly, forces a ``KeyboardInterrupt`` in the task thread
        using ``ctypes.pythonapi.PyThreadState_SetAsyncExc``.  This
        handles the case where the agent is blocked in an LLM API call
        or other I/O and never reaches a cooperative ``_check_stop()``
        call.

        When *tab_id* is a subscriber (multi-viewer) tab that has no
        ``stop_event`` of its own, the method resolves through the
        printer's subscriber mapping to locate the source tab that owns
        the running task and stops that instead.  This lets a second
        browser client viewing a running task via history-click stop it.

        Args:
            tab_id: The tab to stop.  When falsy (empty string), the
                call is a no-op — a missing ``tabId`` at this layer
                indicates a frontend bug that should not silently
                stop every tab's task.
        """
        if not tab_id:
            logger.warning("Stop requested without a tabId; ignoring")
            return
        with self._state_lock:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            stop_event = tab.stop_event if tab is not None else None
            task_thread = tab.task_thread if tab is not None else None
            owner_tab_id = tab_id
            owner_state = tab

        if stop_event is None:
            source_tab_id = self._find_source_tab_for_viewer(tab_id)
            if source_tab_id:
                with self._state_lock:
                    source = _RunningAgentState.running_agent_states.get(
                        source_tab_id,
                    )
                    if source is not None:
                        stop_event = source.stop_event
                        task_thread = source.task_thread
                        owner_tab_id = source_tab_id
                        owner_state = source

        thread_alive = task_thread is not None and task_thread.is_alive()
        if stop_event is None and not thread_alive:
            # A stop the server cannot route used to vanish behind a
            # disabled logger.debug, so a mis-targeted click looked
            # exactly like a click on a wedged task.  Say so, in the log
            # and in the UI, instead of dropping it.
            logger.info(
                "Stop requested for tab %s but no running task owns it",
                tab_id,
            )
            self._broadcast_stop_ack(tab_id, accepted=False)
            return

        logger.info(
            "Stop requested for tab %s (task owner %s)", tab_id, owner_tab_id,
        )
        # Acknowledged BEFORE the event is set: the task can die on the
        # very next bytecode, and the user needs to see that the click
        # landed even when it does.
        self._broadcast_stop_ack(tab_id, accepted=True)
        if stop_event is not None:
            stop_event.set()
        if thread_alive and task_thread is not None:
            still_owns = partial(
                _state_owns_thread,
                owner_tab_id,
                owner_state,
                task_thread,
            )
            threading.Thread(
                target=self._force_stop_thread,
                args=(task_thread, still_owns),
                daemon=True,
            ).start()

    def _broadcast_stop_ack(self, tab_id: str, accepted: bool) -> None:
        """Tell *tab_id* that its Stop click was received.

        The button used to give no feedback at all, so a stop that was
        merely *pending* — the agent was inside a quiet model request —
        looked identical to a stop that never arrived, and the only
        sensible reaction was to click again
        (``reports/stop_button_delay_2026-08-05.html``).

        Args:
            tab_id: The tab whose Stop button was pressed.
            accepted: ``True`` when a running task was found and
                signalled, ``False`` when nothing owned the tab.
        """
        self.printer.broadcast(
            {"type": "stop_ack", "accepted": accepted, "tabId": tab_id},
        )

    def _find_source_tab_for_viewer(self, viewer_tab_id: str) -> str | None:
        """Find a peer tab id sharing the same task as *viewer_tab_id*.

        Scans the printer's ``_subscribers`` mapping
        (``task_id -> {tab_ids}``) to locate the tasks that
        *viewer_tab_id* is subscribed to, then returns another tab id
        subscribed to one of those tasks whose
        :class:`_RunningAgentState` carries a live ``stop_event``
        (i.e. the tab that actually started the task).  Returns
        ``None`` when no such tab exists.

        Every subscribed task is considered — not just the first
        match: ``JsonPrinter.cleanup_task`` intentionally preserves
        subscriber sets when a task ends, so a viewer typically holds
        stale subscriptions to FINISHED tasks alongside the one
        RUNNING task.  Stopping the scan at the first (oldest) match
        would resolve the finished task, find no peer with a live
        ``stop_event`` there, and wrongly report "no source tab" for
        a viewer that can in fact stop a running task.

        A peer with a live ``stop_event`` is only accepted when the
        task its agent is actually running (``agent._last_task_id``)
        matches the subscribed task being scanned.  A stale
        finished-task co-subscriber that has since started a brand-new
        UNRELATED task (which the viewer is NOT subscribed to) also
        carries a live ``stop_event`` — returning it would let the
        viewer's Stop kill that unrelated task (cross-task stop
        hijack, symmetric to the answer-queue hijack fixed as
        BUG-TR2-2).  Peers whose running task cannot be identified
        (no agent attached — e.g. bare test states or a task started
        before the agent slot was populated) are kept as a fallback so
        legitimate stops are not silently dropped.

        Args:
            viewer_tab_id: The subscriber/viewer tab id to look up.

        Returns:
            A peer tab id that owns the cooperative stop event for the
            running task, or ``None`` if not found.
        """
        with self.printer._lock:
            peer_lists = [
                (JsonPrinter._coerce_task_id(task_id), list(viewers))
                for task_id, viewers in self.printer._subscribers.items()
                if viewer_tab_id in viewers
            ]
        fallback: str | None = None
        with self._state_lock:
            for task_key, peers in peer_lists:
                for peer in peers:
                    if peer == viewer_tab_id:
                        continue
                    state = _RunningAgentState.running_agent_states.get(peer)
                    if state is None or state.stop_event is None:
                        continue
                    agent_task = (
                        JsonPrinter._coerce_task_id(
                            getattr(state.agent, "_last_task_id", None),
                        )
                        if state.agent is not None
                        else ""
                    )
                    if agent_task and agent_task != task_key:
                        continue
                    if agent_task:
                        return peer
                    if fallback is None:
                        fallback = peer
            for task_key, _peers in peer_lists:
                if not task_key:
                    continue
                for tid, state in _RunningAgentState.running_agent_states.items():
                    if tid == viewer_tab_id or state.stop_event is None:
                        continue
                    agent_task = (
                        JsonPrinter._coerce_task_id(
                            getattr(state.agent, "_last_task_id", None),
                        )
                        if state.agent is not None
                        else ""
                    )
                    if agent_task and agent_task == task_key:
                        return tid
        return fallback

    @staticmethod
    def _force_stop_thread(
        task_thread: threading.Thread,
        still_owns: Callable[[], bool] | None = None,
    ) -> None:
        """Watchdog that forces ``KeyboardInterrupt`` in *task_thread*.

        Waits 1 second for the cooperative stop-event mechanism to work.
        If the thread is still alive, raises ``KeyboardInterrupt``
        asynchronously in it.  Retries once after 5 seconds in case the
        first exception was swallowed or the thread was in C code.

        Args:
            task_thread: The thread running the task being stopped.
            still_owns: Optional ownership guard evaluated — under
                :attr:`_RunningAgentState._registry_lock` — immediately
                before every injection.  When it returns ``False`` the
                watchdog exits without injecting: the task being
                stopped has already finished and *task_thread* (e.g. a
                reusable ``ThreadPoolExecutor`` worker running a
                parallel sub-agent) may now be executing an unrelated
                sibling task that must not be interrupted.  Producers
                of the guarded state clear ``task_thread`` under the
                same lock, making the check+inject pair race-free.
        """
        task_thread.join(timeout=1)
        for _ in range(2):  # pragma: no branch — thread always dies within 2 attempts
            if not task_thread.is_alive():
                return
            tid = task_thread.ident
            if tid is not None:  # pragma: no branch — running thread always has ident
                with _RunningAgentState._registry_lock:
                    if still_owns is not None and not still_owns():
                        return
                    rc = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(tid),
                        ctypes.py_object(KeyboardInterrupt),
                    )
                if rc == 0:
                    return
                if rc > 1:  # pragma: no cover — rare: exception set in multiple states
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)
            task_thread.join(timeout=5)

    def _await_user_response(
        self,
        q: queue.Queue[str] | None = None,
    ) -> str:
        """Block until the user sends a response, checking stop_event periodically.

        Args:
            q: The answer queue to wait on.  When ``None`` it is
                resolved via :meth:`_resolve_task_answer_queue`.
                W2-F9: :meth:`_ask_user_question` passes the queue it
                already resolved (and drained / registered in the
                pending-answer registry) so both steps operate on the
                SAME queue object — re-resolving here opened a TOCTOU
                window in which the owner tab could be disposed and
                re-created (closeTab + reopen), making the agent drain
                and register queue #1 but block on a DIFFERENT queue
                #2 that no ``userAnswer`` routed to the registry entry
                would ever fill.

        Returns:
            The user's answer string.

        Raises:
            KeyboardInterrupt: If the stop event is set before an answer arrives.
        """
        stop = getattr(self.printer._thread_local, "stop_event", None)
        if stop is None:
            raise KeyboardInterrupt("No stop event set")
        if q is None:
            q = self._resolve_task_answer_queue()
        if q is None:
            raise KeyboardInterrupt(
                "User answer queue is missing (tab closed?); aborting wait",
            )
        sentinel = _STOP_SENTINEL
        cancelled = threading.Event()

        def _wake_on_stop() -> None:
            while not cancelled.is_set():
                if stop.wait(0.1):
                    if not cancelled.is_set():
                        with suppress(queue.Full):
                            q.put_nowait(cast(str, sentinel))
                    return

        watcher = threading.Thread(target=_wake_on_stop, daemon=True)
        watcher.start()
        try:
            item = q.get()
        finally:
            cancelled.set()
        if item is sentinel:
            raise KeyboardInterrupt("Stopped while waiting for user")
        return item

    def _resolve_task_answer_queue(self) -> queue.Queue[str] | None:
        """Resolve the current task's user-answer queue.

        Resolves via the printer's task-id → subscriber-tabs mapping:
        picks the first subscribed tab that has a live
        ``user_answer_queue`` owned by THIS task.  Any tab subscribed
        to this task can answer, but the queue lives on the task-owner
        tab.

        A co-subscriber tab that is itself actively running a
        *different* task also carries a live ``user_answer_queue`` —
        owned by that other task.  Returning it would hijack the other
        task's answers: this task's ``ask_user_question`` would
        consume the answer the user submitted for the other task's
        question, and the other agent would never receive it.  Such
        tabs are skipped (their live agent's ``_last_task_id``
        identifies which task their queue belongs to).

        Returns:
            The owner tab's answer queue, or ``None`` when the
            thread-local ``task_id`` is unset or no subscribed tab
            carries a live queue for this task (e.g. the owner tab was
            closed).
        """
        task_key = getattr(self.printer._thread_local, "task_id", None)
        if not task_key:
            return None
        task_key = JsonPrinter._coerce_task_id(task_key)
        with self.printer._lock:
            tab_ids = list(self.printer._subscribers.get(task_key, ()))
        with self._state_lock:
            for tab_id in tab_ids:
                tab = _RunningAgentState.running_agent_states.get(tab_id)
                if tab is None or tab.user_answer_queue is None:
                    continue
                if not tab_owns_answer_queue(tab, task_key):
                    continue
                return tab.user_answer_queue
        return None

    def _ask_user_question(self, question: str) -> str:
        """Callback for agent questions."""
        task_key = JsonPrinter._coerce_task_id(
            getattr(self.printer._thread_local, "task_id", None),
        )
        q = self._resolve_task_answer_queue()
        if q is not None:
            with self._state_lock:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:  # pragma: no cover — race guard
                        break
                if task_key:
                    self._pending_user_answer_tasks[id(q)] = task_key
        try:
            self.printer.broadcast(
                {
                    "type": "askUser",
                    "question": question,
                }
            )
            return self._await_user_response(q)
        finally:
            if q is not None and task_key:
                with self._state_lock:
                    if self._pending_user_answer_tasks.get(id(q)) == task_key:
                        del self._pending_user_answer_tasks[id(q)]
