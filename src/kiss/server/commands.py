# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Command handlers for the VS Code server.

Split out of ``server.py`` for organisation.  ``_CommandsMixin``
provides one ``_cmd_*`` method per frontend command type plus the
class-level ``_HANDLERS`` dispatch table consumed by
``VSCodeServer._handle_command``.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.persistence import (
    _record_file_usage,
    _record_model_usage,
)
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.tab_registry import OpenTabOutcome
from kiss.server.task_runner import _client_task_id_of

if TYPE_CHECKING:
    from kiss.server.json_printer import JsonPrinter
    from kiss.server.tab_registry import TabRegistry

logger = logging.getLogger(__name__)


def _kiss_home_is_default() -> bool:
    """Return True when this process operates on the default ``~/.kiss``.

    ``KISS_HOME`` redirects all KISS state (config.json, sorcar.db) to a
    private directory — the test suite (``src/kiss/tests/conftest.py``)
    and sandboxed runs rely on it for isolation.  Read at call time (not
    import time) so callers see the current environment.
    """
    custom = os.environ.get("KISS_HOME", "")
    if not custom:
        return True
    try:
        from pathlib import Path

        return Path(custom).resolve() == (Path.home() / ".kiss").resolve()
    except OSError:
        return False


def _owner_task_id(state: AgentState) -> str:
    """Return the persisted task id of *state*'s live agent, or ``""``.

    Reads ``state.agent.last_task_id`` — the ``task_history`` row id
    the agent allocated for its current run, through the property that
    takes the same lock the publishing assignment takes.  MUST be
    called while
    holding :data:`agent_state.STATE_LOCK` (the server's
    ``_state_lock``): task teardown replaces/clears ``state.agent``
    under that lock, so capturing the id inside the same critical
    section that queued a pending user message guarantees the id
    belongs to the state the message was queued on (not a successor
    task that re-armed the tab after the lock was released).

    Returns ``""`` when the state has no agent yet or the agent has
    not allocated its task row (the narrow window between
    ``run()`` entry and ``_add_task``); callers then emit a transient
    (unstamped) echo rather than mis-attributing the prompt to a
    previous task.

    Known accepted attribution nuances (by design):

    * In a multi-``<task>`` run, a prompt queued BETWEEN two subtasks
      is stamped with the subtask currently on screen (the one whose
      id ``last_task_id`` still names) even though the NEXT subtask's
      pre-step drain consumes it — the echo lands in the trajectory
      the user was looking at when they typed.
    * If the whole task tears down between queueing and the echo
      broadcast, the stamped echo may find its recording/persistence
      already cleaned up and stay transient — in that interleaving the
      queued message is never consumed by any agent either (teardown
      clears ``pending_user_messages``), matching pre-fix semantics.

    Args:
        state: The running-agent state whose task id to resolve.

    Returns:
        The owning task id, or ``""`` when it cannot be determined.
    """
    agent = state.agent
    return str(getattr(agent, "last_task_id", "") or "")


def _task_accepts_input(state: AgentState | None) -> bool:
    """True when *state* has a live task that can drain queued input.

    The worker thread raises ``is_task_active`` only AFTER
    ``_cmd_run`` installs and starts ``task_thread``, so a follow-up
    typed during that startup window used to be silently dropped
    (S3-05).  Treating an alive worker thread as live closes the
    window; the same predicate is used by the reattachment logic in
    ``server.py``.  MUST be called while holding
    :data:`agent_state.STATE_LOCK`.

    Delegates the thread-liveness half to
    :meth:`AgentState.thread_alive`, which deliberately counts a
    created-but-not-yet-started thread (``ident is None``,
    ``is_alive()`` False) as alive: ``_cmd_run`` installs
    ``task_thread`` and broadcasts before ``thread.start()``, so an
    ``appendUserMessage`` from another connection in that window must
    still be accepted — a raw ``is_alive()`` check here reopened the
    exact S3-05 drop this predicate exists to close.

    Args:
        state: The agent state to inspect (``None`` accepted).

    Returns:
        True when the state's task is active or its worker thread is
        still alive.
    """
    if state is None:
        return False
    return state.is_task_active or state.thread_alive()


def _restart_kiss_web_daemon() -> bool:
    """Restart the ``kiss-web`` daemon so it picks up config changes.

    On macOS, uses ``launchctl kickstart -k`` to restart the
    ``com.kiss.web-server`` LaunchAgent.  On Linux, uses
    ``systemctl --user restart kiss-web``.  Runs asynchronously in
    a background thread so the caller does not block.

    SAFETY: when this process operates on a NON-default ``KISS_HOME``
    (tests, sandboxes), the system LaunchAgent serves a *different*
    home whose config this process never touched — kick-starting it
    could only destroy unrelated in-flight work.  Incident 2026-06-11
    00:37:45: a pytest process exercising ``_cmd_save_config`` with a
    changed ``remote_password`` SIGTERMed the developer's live
    kiss-web daemon (pid 2884), killing the very agent task tree
    (task_history rows 3556, 3618-3624) that had launched the test.
    The guard below makes that impossible.

    Returns:
        True when a restart was dispatched; False when skipped because
        ``KISS_HOME`` points at a non-default location.
    """
    if not _kiss_home_is_default():
        logger.warning(
            "Skipping kiss-web daemon restart: KISS_HOME=%r is not the "
            "default ~/.kiss — the system daemon serves a different home",
            os.environ.get("KISS_HOME", ""),
        )
        return False

    def _do_restart() -> None:
        try:
            if sys.platform == "darwin":
                uid = os.getuid()
                subprocess.run(
                    [
                        "launchctl", "kickstart", "-k",
                        f"gui/{uid}/com.kiss.web-server",
                    ],
                    capture_output=True,
                    timeout=10,
                )
            elif sys.platform == "linux":
                subprocess.run(
                    ["systemctl", "--user", "restart", "kiss-web"],
                    capture_output=True,
                    timeout=10,
                )
        except Exception:
            logger.debug("Failed to restart kiss-web daemon", exc_info=True)

    threading.Thread(target=_do_restart, daemon=True).start()
    return True


def _parse_int(value: Any) -> int | None:
    """Parse a frontend-supplied JSON value as an int.

    Guarded parse for int-typed command fields (e.g. the history
    pager's ``offset`` / ``generation`` / ``limit``) so malformed
    payloads (e.g. ``"offset": "abc"``) never raise out of a command
    handler — an escaping exception terminates the transport's whole
    receive loop and with it the client connection.

    Args:
        value: Arbitrary value taken from a client command dict.

    Returns:
        The parsed int, or ``None`` when the value is missing or not
        int-coercible.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_str(value: Any) -> str | None:
    """Return *value* when it is a non-empty string, else ``None``.

    Used to validate frontend-supplied ids (e.g. ``taskId``) so
    malformed payloads are ignored instead of raising out of a
    command handler.

    Args:
        value: Arbitrary value taken from a client command dict.

    Returns:
        The non-empty string, or ``None`` otherwise.
    """
    return value if isinstance(value, str) and value else None


class _CommandsMixin:
    """Methods that implement frontend command handlers."""

    _save_config_lock = threading.Lock()

    if TYPE_CHECKING:
        printer: JsonPrinter
        work_dir: str
        _state_lock: threading.RLock
        _shutdown_stopping: bool
        _default_model: str
        _complete_seq: int
        _complete_seq_latest: dict[str, int]
        _complete_queue: (
            queue.Queue[tuple[str, int, str, str | None, str, str, str]]
            | None
        )
        _last_active_file: dict[str, str]
        _last_active_content: dict[str, str]
        _file_cache: dict[str, list[str]]
        _tab_chat_views: dict[str, str]
        _tab_models: dict[str, str]
        _commit_msg_tabs: set[str]
        _autocommit_tabs: set[str]
        tab_registry: TabRegistry

        def _broadcast_tabs_state(self) -> None: ...

        def _run_task(self, cmd: dict[str, Any]) -> None: ...
        def _stop_task(
            self, tab_id: str = "", run_token: str = "",
        ) -> None: ...
        def _find_viewer_task_states(
            self, viewer_tab_id: str,
        ) -> list[AgentState]: ...
        def _get_models(self, conn_id: str = "") -> None: ...
        def _get_history(
            self,
            query: str | None,
            offset: int = 0,
            generation: int = 0,
            conn_id: str = "",
        ) -> None: ...
        def _get_frequent_tasks(
            self, limit: int = 50, conn_id: str = "",
        ) -> None: ...
        def _get_files(
            self,
            prefix: str,
            work_dir: str = "",
            conn_id: str = "",
            tab_id: str = "",
        ) -> None: ...
        def _refresh_file_cache(
            self,
            then_emit_for_prefix: str | None = None,
            work_dir: str = "",
            conn_id: str = "",
            tab_id: str = "",
        ) -> None: ...
        def _replay_session(
            self, chat_id: str, tab_id: str = "", task_id: str | None = None,
        ) -> None: ...
        def _new_chat(self, tab_id: str) -> None: ...
        def _close_tab(self, tab_id: str) -> None: ...
        def _registry_update_tab(
            self,
            tab_id: str,
            *,
            chat_id: str | None = None,
            title: str | None = None,
            work_dir: str | None = None,
            scope_work_dir: str | None = None,
            task_id: str | None = None,
            create: bool = False,
        ) -> None: ...
        def _broadcast_to_conn(
            self, event: dict[str, Any], conn_id: str,
        ) -> None: ...
        def _ensure_complete_worker(self) -> None: ...
        def _get_input_history(self, conn_id: str = "") -> None: ...
        def _get_adjacent_task(
            self, chat_id: str, task_id: str | None, direction: str,
            tab_id: str = "",
        ) -> None: ...
        def _generate_commit_message(
            self, tab_id: str = "", *, work_dir: str = "",
        ) -> None: ...
        def _autocommit_changes(
            self, tab_id: str = "", *, work_dir: str = "",
            manual: bool = False,
        ) -> None: ...
        def _any_non_wt_running(
            self, repo_root: Path | None = None,
        ) -> bool: ...
        def _broadcast_autocommit_done(
            self, tab_id: str, *, success: bool, committed: bool,
            message: str, commit_message: str | None = None,
            manual: bool = False, work_dir: str = "",
        ) -> dict[str, Any]: ...
        def _handle_worktree_action(
            self, action: str, tab_id: str = "", *,
            internal: bool = False, already_claimed: bool = False,
        ) -> dict[str, Any]: ...
        def _handle_main_tree_action(
            self, action: str, work_dir: str,
        ) -> dict[str, Any]: ...
        def _handle_delete_frequent_task(self, task: str) -> None: ...
        def _handle_set_favorite(
            self, task_id: str, is_favorite: bool,
        ) -> None: ...


    def _apply_new_work_dir(self, new_dir: str) -> None:
        """Adopt *new_dir* as the daemon-wide fallback working directory.

        Single shared implementation of the work-dir update used by
        both :meth:`_cmd_set_work_dir` and :meth:`_cmd_save_config`
        (D-R1: the latter used to copy-paste the former's block).
        Invalidates the autocomplete file cache only when the
        directory actually changes, and mirrors the value onto the
        printer either way.  Takes ``_state_lock`` itself; the lock is
        re-entrant, so callers already holding it may call this
        directly.

        Args:
            new_dir: The non-empty directory to adopt.
        """
        with self._state_lock:
            if self.work_dir != new_dir:
                self.work_dir = new_dir
                self._file_cache = {}
            if hasattr(self.printer, "work_dir"):
                setattr(self.printer, "work_dir", new_dir)

    def _cmd_run(self, cmd: dict[str, Any]) -> None:
        """Start an agent task in a background thread.

        Initializes the tab's agent chat id (if empty) and broadcasts
        the initial ``clear`` event **synchronously**, before starting
        the worker thread.  Emitting ``clear`` here (rather than from
        inside the worker thread) makes the chat-id → tab-id mapping
        visible to the extension layer immediately after ``_cmd_run``
        returns, so a subsequent ``resumeSession`` for the same chat
        (e.g. a fast history click right after submit) can be routed
        to the correct task process without racing the worker
        thread's first broadcast.
        """
        tab_id = cmd.get("tabId", "")
        # Acknowledge + mirror the task-panel text to EVERY client
        # here, in the common run path, so all run origins (VS Code,
        # remote-web ``submit``, Python clients) behave identically.
        # Broadcast unconditionally — also for a queued follow-up, a
        # merge refusal or a tab-less run — because the echo doubles
        # as the submit acknowledgment: it carries the byte-truncated
        # prompt back to the submitting client even when no task can
        # start (pinned by the prompt-truncation transport tests).
        self.printer.broadcast({
            "type": "setTaskText",
            "text": str(cmd.get("prompt", "") or ""),
            "tabId": tab_id,
        })
        if not tab_id:
            logger.debug("Ignoring run command without tabId")
            return
        inject_prompt: str | None = None
        inject_task = ""
        thread: threading.Thread | None = None
        state: AgentState | None = None
        chat_id = ""
        with self._state_lock:
            prev = agent_state.find_by_tab(tab_id)
            if prev is not None and prev.is_merging:
                # An in-flight merge/discard owns the tab's state (and
                # its worktree agent); replacing it would orphan the
                # operation.  Refuse the run instead.  Both frontends
                # raise the tab's running state optimistically the
                # moment the user hits Enter and only a
                # ``status running:false`` ever lowers it again, so the
                # refusal MUST clear it first or the tab's composer
                # stays disabled forever (F08-1).
                self.printer.broadcast(
                    {"type": "status", "running": False, "tabId": tab_id},
                )
                self.printer.broadcast(
                    {
                        "type": "error",
                        "text": "Cannot run a task while a merge is"
                        " in progress. Wait for it to finish first.",
                        "tabId": tab_id,
                    }
                )
                return
            if prev is not None and prev.task_thread is not None:
                prompt = cmd.get("prompt", "")
                # S3-05: queue the prompt whenever a task thread is
                # installed.  The worker sets ``is_task_active`` only
                # AFTER the thread starts, so gating on the flag (or on
                # thread death) silently dropped a second ``run``
                # submitted during the startup window in which the
                # thread was alive but the flag not yet raised.
                if isinstance(prompt, str) and prompt.strip():
                    prev.pending_user_messages.append(prompt)
                    inject_prompt = prompt
                    inject_task = _owner_task_id(prev)
                    if not inject_task:
                        prev.unattributed_prompt_echoes.append(prompt)
            else:
                requested_chat_id = cmd.get("chatId", "")
                resumed_chat_id = self._tab_chat_views.get(tab_id, "")
                if prev is not None and prev.chat_id:
                    chat_id = prev.chat_id
                elif isinstance(requested_chat_id, str) and requested_chat_id:
                    chat_id = requested_chat_id
                elif resumed_chat_id:
                    chat_id = resumed_chat_id
                else:
                    chat_id = uuid.uuid4().hex
                self._tab_chat_views[tab_id] = chat_id
                state_key = uuid.uuid4().hex
                cmd["_state_key"] = state_key
                state = AgentState(
                    state_key,
                    chat_id=chat_id,
                    tab_id=tab_id,
                    conn_id=str(cmd.get("connId", "") or ""),
                    server_owned=True,
                    stop_event=threading.Event(),
                )
                state.user_answer_queue = queue.Queue(maxsize=1)
                state.client_run_token = _client_task_id_of(cmd)
                # Stamp the submitted prompt NOW: ``_run_task`` only
                # sets ``last_user_prompt`` once its per-subtask loop
                # starts, which is AFTER the worktree/tools/agent-
                # script setup — a client that reconnects during that
                # window replays this run through the pre-history-row
                # branch of ``_replay_session``, whose ``task`` field
                # (the fixed task panel's text) reads this attribute.
                state.last_user_prompt = str(cmd.get("prompt", "") or "")
                if prev is not None:
                    # Carry the previous task's agent (it may hold a
                    # pending worktree) over to the new run's state.
                    state.agent = prev.agent
                    state.frontend_closed = prev.frontend_closed
                    agent_state.unregister(prev.task_id, prev)
                thread = threading.Thread(
                    target=self._run_task, args=(cmd,), daemon=True
                )
                state.task_thread = thread
                agent_state.register(state)
        if thread is None:
            if inject_prompt is not None:
                self._echo_injected_prompt(
                    tab_id, inject_prompt, inject_task,
                )
            return
        # ``thread`` and ``state`` are created together above, so a
        # non-None thread guarantees the state.
        assert state is not None
        try:
            # Register + title + bind the tab in the shared registry
            # BEFORE the ``clear`` broadcast so every client has the
            # tab by the time the run's first event reaches it.  A new
            # run supersedes any historical task the tab was pinned to
            # (``taskId`` cleared: the tab tracks the chat's latest
            # task again — the one this run creates).
            self._registry_update_tab(
                tab_id,
                chat_id=chat_id,
                title=str(cmd.get("prompt", "") or ""),
                work_dir=str(cmd.get("workDir", "") or ""),
                # A ``run_agent`` sub-task (wire field
                # ``tabScopeWorkDir``) executes in a channel/cron
                # scratch directory but must appear in the CALLING
                # workspace's tab bar, so its visibility scope is
                # pinned to that workspace here while ``workDir`` stays
                # the scratch directory.  Empty for ordinary runs,
                # whose scope falls back to ``workDir`` unchanged.
                scope_work_dir=str(cmd.get("tabScopeWorkDir", "") or ""),
                task_id="",
                create=True,
            )
            # The submit-ack ``setTaskText`` at the top of this method
            # raced ahead of the tab's registration when the run
            # CREATES its tab — a Python client's synthetic ``api-…``
            # tab (``sorcar.run`` and the ``run_agent`` dispatch): no
            # client had adopted the tab yet, so every client dropped
            # the task-panel text and the tab showed its transcript
            # WITHOUT the fixed task panel at the top.  Re-echo the
            # text now that the registration's ``tabs_state`` snapshot
            # has handed every client the tab.  Unconditional on
            # purpose: gating it on a pre-registration ``has_tab``
            # probe is a TOCTOU (a concurrent ``closeTab`` between
            # probe and registration recreates the tab yet suppresses
            # the echo), and clients apply a repeated ``setTaskText``
            # idempotently — the daemon already echoes one per queued
            # follow-up as well.
            self.printer.broadcast({
                "type": "setTaskText",
                "text": str(cmd.get("prompt", "") or ""),
                "tabId": tab_id,
            })
            self.printer.broadcast({
                "type": "clear",
                "chat_id": chat_id,
                "tabId": tab_id,
            })
            # Start/cancel handshake (audit0903 F1/F2): a ``stop`` or
            # the graceful-shutdown sweep can land while the registry
            # write and the ``clear`` broadcast above hold the
            # pre-start window open.  The decision to start is taken
            # atomically under ``_state_lock`` — the same lock
            # ``_stop_task`` and ``_stop_active_agent_tasks`` flag
            # under — so a swept or stopped run can never call
            # ``thread.start()`` afterwards.  Before this handshake
            # the shutdown sweep crashed joining the unstarted thread
            # and the run then executed its untrusted setup with no
            # watchdog, and an accepted pre-start stop relied on a
            # watchdog that gives up waiting for the start after 30 s.
            with self._state_lock:
                if self._shutdown_stopping:
                    state.interrupted_by_shutdown = True
                pre_cancelled = state.interrupted_by_shutdown or (
                    state.stop_event is not None
                    and state.stop_event.is_set()
                )
                if not pre_cancelled:
                    thread.start()
        except BaseException:
            with self._state_lock:
                if state is not None and state.task_thread is thread:
                    state.task_thread = None
                    state.stop_event = None
                    state.user_answer_queue = None
            raise
        if pre_cancelled:
            # Route the never-started run through the normal terminal
            # cancellation — ``_cancel_outcome`` labelling, ``result``
            # and ``status`` broadcasts, state cleanup — WITHOUT
            # executing any user setup: ``_run_task`` raises the
            # cancelling ``KeyboardInterrupt`` at its top when it sees
            # the marker, right here on the dispatch thread.
            cmd["_pre_cancelled"] = True
            self._run_task(cmd)

    def _cmd_stop(self, cmd: dict[str, Any]) -> None:
        """Stop a running task.

        An optional ``taskId`` — the client-minted per-submission run
        token, sent by ``daemon_client.run``'s abort-cascade stop —
        restricts the stop to the run it belongs to: a synthetic
        ``api-…`` tab can be reused by a NEWER run after the original
        finishes, and a late tab-only stop would kill that innocent
        run.  UI stops send no ``taskId`` and behave as before.
        """
        self._stop_task(
            cmd.get("tabId", ""),
            run_token=_client_task_id_of(cmd),
        )

    def _cmd_get_models(self, cmd: dict[str, Any]) -> None:
        """Send available models list to the requesting connection only."""
        self._get_models(cmd.get("connId", ""))

    def _cmd_select_model(self, cmd: dict[str, Any]) -> None:
        """Update the selected model for a tab.

        An empty ``tabId`` (malformed payload) updates only the
        daemon-wide default model (when a model was actually
        supplied).
        """
        tab_id = cmd.get("tabId", "")
        model = cmd.get("model", "")
        if not isinstance(model, str):
            model = ""
        with self._state_lock:
            if tab_id:
                if not model:
                    model = self._tab_models.get(tab_id, "") or self._default_model
                self._tab_models[tab_id] = model
            if not model:
                return
            self._default_model = model
            _record_model_usage(model)

    def _cmd_get_history(self, cmd: dict[str, Any]) -> None:
        """Send conversation history to the requesting connection only."""
        query = cmd.get("query")
        if not isinstance(query, str):
            query = None
        offset = _parse_int(cmd.get("offset", 0))
        generation = _parse_int(cmd.get("generation", 0))
        self._get_history(
            query,
            0 if offset is None else offset,
            0 if generation is None else generation,
            cmd.get("connId", ""),
        )

    def _cmd_get_frequent_tasks(self, cmd: dict[str, Any]) -> None:
        """Send the top-N most-frequent tasks (default 50)."""
        limit = _parse_int(cmd.get("limit", 50))
        self._get_frequent_tasks(
            50 if limit is None else limit, cmd.get("connId", ""),
        )

    def _cmd_delete_frequent_task(self, cmd: dict[str, Any]) -> None:
        """Delete a row from the ``frequent_tasks`` table by task text."""
        task = cmd.get("task")
        if isinstance(task, str) and task:
            self._handle_delete_frequent_task(task)

    def _cmd_set_favorite(self, cmd: dict[str, Any]) -> None:
        """Persist the favourite flag on a task history row."""
        task_id = _opt_str(cmd.get("taskId"))
        if task_id is None:
            return
        is_favorite = bool(cmd.get("isFavorite", False))
        self._handle_set_favorite(task_id, is_favorite)

    def _cmd_get_files(self, cmd: dict[str, Any]) -> None:
        """Send file list for autocomplete, scoped to the tab's work_dir.

        The chat webview stamps the active tab's ``workDir`` on every
        ``getFiles`` command so the ``@``-mention picker lists files
        relative to *that* tab's working directory rather than the
        daemon-wide default (which is shared across every tab and
        otherwise reflects whichever directory the daemon was launched
        from or last switched to via ``setWorkDir``).

        The resulting ``files`` events are routed only to the
        requesting connection (via ``connId``) so typing ``@`` in one
        VS Code window never pops the file picker in another window,
        and are stamped with the requesting ``tabId`` so within that
        window they pop only in the chat tab that typed ``@`` — the
        picker element is shared by every tab.
        """
        prefix = cmd.get("prefix", "")
        if not isinstance(prefix, str):
            prefix = ""
        self._get_files(
            prefix,
            cmd.get("workDir", ""),
            cmd.get("connId", ""),
            cmd.get("tabId", ""),
        )

    def _cmd_record_file_usage(self, cmd: dict[str, Any]) -> None:
        """Record a file access for usage-based sorting.

        Usage counts are stored as workspace-relative paths in a
        single shared SQLite table; the ``workDir`` (if any) on the
        command is currently informational — the ranking still applies
        across every tab.  Accepting the field keeps the message shape
        symmetric with ``getFiles`` so the frontend can forward both
        without conditional branching.
        """
        path = cmd.get("path", "")
        if isinstance(path, str) and path:
            _record_file_usage(path)

    def _cmd_user_answer(self, cmd: dict[str, Any]) -> None:
        """Route a user answer to the correct tab's queue.

        The drain-then-put sequence is held under ``_state_lock`` so
        two concurrent ``userAnswer`` commands cannot both observe
        the queue as empty, both call ``q.put`` on the ``maxsize=1``
        queue, and wedge the second handler thread forever.  Using
        ``put_nowait`` after the drain — combined with the lock —
        guarantees the call never blocks: the queue is guaranteed
        empty by the just-completed drain, and any concurrent
        ``userAnswer`` is serialised behind us.
        """
        ans_tab = cmd.get("tabId", "")
        with self._state_lock:
            owner = self._resolve_user_answer_state(ans_tab)
            q = owner.user_answer_queue if owner is not None else None
            if owner is None or q is None:
                logger.debug("userAnswer dropped: no queue for tabId=%s", ans_tab)
                return
            answered_task_id = owner.task_id
            # The question is no longer pending the moment its answer
            # is consumed: clearing under ``_state_lock`` guarantees a
            # concurrent session replay (``_emit_pending_ask``) can
            # never re-show an already-answered modal.
            owner.pending_ask_question = ""
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:  # pragma: no cover — race guard
                    break
            answer = cmd.get("answer", "")
            if not isinstance(answer, str):
                answer = "" if answer is None else str(answer)
            try:
                q.put_nowait(answer)
            except queue.Full:  # pragma: no cover — drained immediately above
                pass
        clear_tabs = self._user_answer_clear_tabs(ans_tab, answered_task_id)
        for tab_id in clear_tabs:
            self.printer.broadcast({"type": "askUserDone", "tabId": tab_id})

    def _user_answer_clear_tabs(
        self, ans_tab: str, answered_task_id: str,
    ) -> list[str]:
        """Return every tab whose ask-user modal should close.

        A submitted answer resolves one pending question for exactly one
        running task/chat, regardless of which subscribed tab supplied it.
        Completed-task subscriber sets are intentionally retained for
        post-task broadcasts, so closing every historic subscriber set
        that contains ``ans_tab`` can dismiss an unrelated tab's current
        question.  The pending-question registry records the task id that
        owns the queue which consumed this answer; only that task's
        subscribers receive ``askUserDone``.

        Args:
            ans_tab: Frontend tab id carried by the ``userAnswer``
                command.
            answered_task_id: Task id associated with the live
                ``ask_user_question`` that consumed the answer.

        Returns:
            Stable list of tab ids to receive ``askUserDone``.
        """
        if not ans_tab:
            return []
        if not answered_task_id:
            return [ans_tab]
        printer_lock = getattr(self.printer, "_lock", None)
        subs_map = getattr(self.printer, "_subscribers", {})
        if printer_lock is None:
            return [ans_tab]
        task_key = self.printer._coerce_task_id(answered_task_id)
        with printer_lock:
            viewers = list(subs_map.get(task_key, ()))
        tabs = {str(v) for v in viewers if v}
        if not tabs:
            tabs.add(ans_tab)
        return sorted(tabs)

    def _resolve_user_answer_state(
        self, ans_tab: str,
    ) -> AgentState | None:
        """Locate the agent state an ``ask_user_question`` is waiting on.

        Routing precedence:

        1. The state launched from the frontend tab ``ans_tab``
           itself, when it holds a non-None ``user_answer_queue``.
           This is the common path: a single-window user answers from
           the same tab that launched the task.

        2. Otherwise, the state of any task that ``ans_tab`` is
           subscribed to.  This covers the multi-viewer case where one
           tab (e.g. a browser viewer of a chat owned by the VS Code
           extension's tab) renders the askUser modal and submits the
           answer: the broadcast was fan-stamped with the viewer's tab
           id, but the live ``user_answer_queue`` lives on the state
           of the task itself.  Resolving through the task id makes a
           cross-task answer hijack structurally impossible.

        Args:
            ans_tab: Frontend tab id carried by the ``userAnswer``
                command.

        Returns:
            The resolved agent state, or ``None`` when no live
            ``ask_user_question`` waiter can be associated with the
            command.  Must be called with ``_state_lock`` held.
        """
        ans_state = agent_state.find_by_tab(ans_tab)
        if ans_state is not None and ans_state.user_answer_queue is not None:
            return ans_state
        printer_lock = getattr(self.printer, "_lock", None)
        subs_map = getattr(self.printer, "_subscribers", {})
        if printer_lock is None:
            return None
        with printer_lock:
            task_keys = [
                self.printer._coerce_task_id(task_id)
                for task_id, viewers in subs_map.items()
                if ans_tab in viewers
            ]
        for task_key in task_keys:
            state = agent_state.get(task_key)
            if state is not None and state.user_answer_queue is not None:
                return state
        return None

    def _echo_injected_prompt(
        self, tab_id: str, prompt: str, owner_task: str,
    ) -> None:
        """Broadcast a queued follow-up prompt back to the tab's viewers.

        Emits a ``prompt`` event stamped with the originating
        ``tabId`` — the tab whose transcript the user is looking at —
        so the queued message appears in the chat surface immediately.

        The echo is ALSO stamped with *owner_task* (the task whose
        ``pending_user_messages`` queue
        received the prompt) so the printer records it into the task's
        in-memory recording and persists it into the task's ``events``
        rows.  Without the stamp the echo is a transient targeted
        broadcast (see ``WebPrinter.broadcast``): it renders once and
        then vanishes from the trajectory on any ``task_events``
        replay — which sub-agent tabs perform on every reopen/history
        click — so an injected prompt would never show up in a
        sub-agent's (or a reloaded main tab's) transcript.

        When *owner_task* is empty (the task row is not allocated yet
        — the narrow window between ``run()`` entry and ``_add_task``)
        the echo is emitted WITHOUT the stamp so the user still sees
        their message immediately; the caller ALSO queued the prompt
        on the state's ``unattributed_prompt_echoes`` list, and the
        drain hook (``SorcarAgent._drain_pending_user_messages``)
        later records + persists a durable copy under the task that
        actually consumed the message (a ``recordOnly`` broadcast — it
        is never re-sent live, so no duplicate panel appears).

        Args:
            tab_id: The frontend tab id the user typed into.
            prompt: The queued follow-up text.
            owner_task: The owning task id captured under
                ``_state_lock`` at queueing time (see
                :func:`_owner_task_id`), or ``""`` when the task row
                is not allocated yet.
        """
        echo: dict[str, Any] = {
            "type": "prompt",
            "text": prompt,
            "tabId": tab_id,
        }
        if owner_task:
            echo["taskId"] = owner_task
        self.printer.broadcast(echo)

    def _cmd_append_user_message(self, cmd: dict[str, Any]) -> None:
        """Queue a user message to be injected into the running agent's context.

        When the user types into the task-input textbox while a task is
        still running, the frontend forwards the prompt here instead of
        silently dropping it.  We append the text to the tab's
        :attr:`AgentState.pending_user_messages` list under
        :data:`agent_state.STATE_LOCK` so the live agent's
        pre-step hook can drain and inject the messages into the model
        conversation before the next model call.

        When the tab itself has no live task (the common case for a
        VIEWER tab opened from the history sidebar while a task runs
        in ANOTHER tab — the viewer is subscribed to the running
        task's event stream but the live agent belongs to that task's
        own state) the prompt is routed to the running task's state
        via the printer's per-task subscriber map.  This
        is what makes a history-resumed viewer tab accept follow-up
        input while the underlying task is still running: without it,
        the typed text would be silently dropped (because the viewer
        tab's own state has ``is_task_active=False``) and the user
        would watch their message disappear from the input box with
        no effect on the running agent.

        The append is silently ignored only when neither the tab nor
        any peer tab the viewer is subscribed to has a live task —
        attempting to queue a follow-up against a truly idle tab
        would be a no-op (no pre-step hook to drain it).  We also
        echo the queued prompt back to every viewer of the tab as a
        ``prompt`` event so the user sees their queued message in
        the chat surface.
        """
        tab_id = cmd.get("tabId", "")
        prompt = cmd.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            return
        with self._state_lock:
            owner = agent_state.find_by_tab(tab_id)
            if not _task_accepts_input(owner):
                owner = None
                for candidate in self._find_viewer_task_states(tab_id):
                    if _task_accepts_input(candidate):
                        owner = candidate
                        break
            if owner is None:
                logger.debug(
                    "appendUserMessage dropped: tab %s has no "
                    "live task and is not a viewer of one",
                    tab_id,
                )
                return
            owner.pending_user_messages.append(prompt)
            owner_task = _owner_task_id(owner)
            if not owner_task:
                owner.unattributed_prompt_echoes.append(prompt)
        self._echo_injected_prompt(tab_id, prompt, owner_task)

    def _cmd_resume_session(self, cmd: dict[str, Any]) -> None:
        """Replay a previous chat session.

        When ``taskId`` is present, load that specific task instead of
        the latest task in the chat session.
        """
        raw_id = cmd.get("chatId")
        chat_id = str(raw_id) if raw_id else ""
        task_id = _opt_str(cmd.get("taskId"))
        if chat_id or task_id is not None:
            self._replay_session(
                chat_id, cmd.get("tabId", ""), task_id=task_id,
            )

    def _cmd_open_tab(self, cmd: dict[str, Any]) -> None:
        """Register a client-opened tab in the shared tab registry.

        Sent by a client the moment it creates a chat tab locally.
        The registry mutation broadcasts a ``tabs_state`` snapshot, so
        every other client opens the same tab.  Idempotent: a tab id
        that is already registered changes nothing (and broadcasts
        nothing).

        A REJECTED open (registry at its hard cap) is answered with an
        ``openTabRejected`` event to the originating client — without
        it the client would keep a permanently local, snapshot-immune
        tab no other client ever sees.  The exists/full distinction is
        made atomically inside :meth:`TabRegistry.open_tab` (its
        :class:`~kiss.server.tab_registry.OpenTabOutcome` return): an
        unlocked ``has_tab`` re-probe here used to let a concurrent
        ``closeTab`` turn a benign re-announce of an existing tab into
        a spurious "Tab limit reached" rejection (D-RC2).
        """
        tab_id = cmd.get("tabId", "")
        if not isinstance(tab_id, str) or not tab_id:
            return
        title = cmd.get("title", "")
        if not isinstance(title, str):
            title = ""
        work_dir = cmd.get("workDir", "")
        if not isinstance(work_dir, str):
            work_dir = ""
        outcome = self.tab_registry.open_tab(tab_id, title, work_dir)
        if outcome is OpenTabOutcome.OPENED:
            self._broadcast_tabs_state()
        elif outcome is OpenTabOutcome.FULL:
            self._broadcast_to_conn(
                {
                    "type": "openTabRejected",
                    "tabId": tab_id,
                    "text": (
                        "Tab limit reached — close some tabs before "
                        "opening new ones."
                    ),
                },
                cmd.get("connId", ""),
            )

    def _cmd_close_tab(self, cmd: dict[str, Any]) -> None:
        """Clean up backend state for a closed frontend tab."""
        tab_id = cmd.get("tabId", "")
        if tab_id:
            self._close_tab(tab_id)

    def _cmd_new_chat(self, cmd: dict[str, Any]) -> None:
        """Start a new chat session."""
        self._new_chat(cmd.get("tabId", ""))

    def _cmd_complete(self, cmd: dict[str, Any]) -> None:
        """Ghost text autocomplete request.

        All mutable autocomplete state is keyed by the command's
        ``connId`` (stamped per client connection by
        :class:`RemoteAccessServer`; ``""`` for direct callers):

        * The active-file snapshot fallback — used when the current
          command carries no ``activeFile`` (e.g. focus is inside the
          webview) — is the *same connection's* last-reported editor
          file, never another window's.
        * Request staleness (``_complete_seq_latest``) is tracked per
          connection so a window typing concurrently with another
          window cannot mark the other window's pending request stale.
        """
        query = cmd.get("query", "")
        if not isinstance(query, str):
            query = ""
        active_file = cmd.get("activeFile")
        active_content = cmd.get("activeFileContent")
        if not isinstance(active_file, str):
            active_file = None
        if not isinstance(active_content, str):
            active_content = None
        conn_id = cmd.get("connId", "")
        tab_id = cmd.get("tabId", "")
        with self._state_lock:
            chat_id = ""
            if tab_id:
                state = agent_state.find_by_tab(tab_id)
                if state is not None:
                    chat_id = state.chat_id
                if not chat_id:
                    chat_id = self._tab_chat_views.get(tab_id, "")
            if active_file:
                if (
                    active_content is None
                    and active_file != self._last_active_file.get(conn_id, "")
                ):
                    # The window reported a DIFFERENT editor file with
                    # no buffer snapshot: the stored content belongs to
                    # the previous file and must not be paired with the
                    # new path (stale cross-file identifiers).
                    self._last_active_content.pop(conn_id, None)
                self._last_active_file[conn_id] = active_file
            if active_content is not None:
                self._last_active_content[conn_id] = active_content
            snapshot_file = self._last_active_file.get(conn_id, "")
            # ``None`` (never reported) must stay ``None`` so
            # ``_active_file_identifier_matches`` falls back to reading
            # ``snapshot_file`` from disk; a ``""`` default would be
            # honoured verbatim as an "open but empty buffer" and
            # dead-code the documented on-disk fallback.  The VS Code
            # client really does send ``activeFile`` without
            # ``activeFileContent`` when the visible editor's document
            # is not among ``vscode.workspace.textDocuments``.
            snapshot_content = self._last_active_content.get(conn_id)
            self._complete_seq += 1
            seq = self._complete_seq
            self._complete_seq_latest[conn_id] = seq
        if query:
            self._ensure_complete_worker()
            self._complete_queue.put(  # type: ignore[union-attr]
                (
                    query, seq, snapshot_file, snapshot_content, chat_id,
                    conn_id, tab_id,
                ),
            )

    def _cmd_get_input_history(self, cmd: dict[str, Any]) -> None:
        """Send deduplicated task texts for arrow-key cycling."""
        self._get_input_history(cmd.get("connId", ""))

    def _cmd_get_adjacent_task(self, cmd: dict[str, Any]) -> None:
        """Send events for the adjacent task in the same chat session.

        Uses only the tab's own agent chat_id.  Previously, when the tab
        had no chat_id the handler fell back to the globally-latest
        chat in history, causing arrow-key navigation in one tab to
        traverse a *different* tab's conversation (C1 fix).

        The current task is identified by its DB row id (``taskId``);
        navigating by id (rather than the task description text)
        unambiguously handles duplicate task texts within a chat.

        Pure-viewer tabs (opened from the history sidebar by
        ``_replay_session``) deliberately have NO registry entry
        (C2/C3 fix) — only a ``_tab_chat_views`` association.  Resolve the chat id from the
        registry entry when one exists, falling back to the
        chat-viewer map, and never CREATE a registry entry here:
        navigation is a read-only view operation.
        """
        tab_id = cmd.get("tabId", "")
        with self._state_lock:
            adj_state = agent_state.find_by_tab(tab_id)
            chat_id = adj_state.chat_id if adj_state is not None else ""
            if not chat_id:
                chat_id = self._tab_chat_views.get(tab_id, "")
        task_id = _opt_str(cmd.get("taskId"))
        self._get_adjacent_task(
            chat_id,
            task_id,
            cmd.get("direction", "prev"),
            tab_id,
        )

    def _cmd_generate_commit_message(self, cmd: dict[str, Any]) -> None:
        """Generate a git commit message in the background.

        Runs the generator in a daemon thread and passes the caller's
        ``tabId`` to :meth:`_generate_commit_message` which stamps it
        on every emitted ``commitMessage`` event so the result reaches
        only the originating tab (B5 fix).

        The command's ``workDir`` (the tab's own folder) is forwarded so
        the generator operates on the tab's repository rather than the
        daemon-wide ``self.work_dir``, which may point at a different —
        possibly non-git — folder and produce a misleading "Not a git
        repository." error.

        At most one generation runs per tab: the generator makes a
        billed LLM call and stamps its answer on the tab, so an
        impatient double click used to pay twice and let the slower
        (not the latest) reply win (R09-8).  Extra clicks are dropped
        while the tab's generation is in flight.
        """
        tab_id = cmd.get("tabId", "")
        work_dir = cmd.get("workDir", "")
        with self._state_lock:
            if tab_id in self._commit_msg_tabs:
                logger.debug(
                    "Commit message generation already in flight for "
                    "tab %r; ignoring duplicate request", tab_id,
                )
                return
            self._commit_msg_tabs.add(tab_id)
        threading.Thread(
            target=self._run_commit_message_job,
            args=(tab_id, work_dir),
            daemon=True,
        ).start()

    def _run_commit_message_job(self, tab_id: str, work_dir: str) -> None:
        """Generate the tab's commit message and re-arm the button.

        Body of the daemon thread spawned by
        :meth:`_cmd_generate_commit_message`; the ``finally`` releases
        the tab's in-flight claim so a failed generation never wedges
        the tab out of ever generating a message again.

        Args:
            tab_id: Frontend tab that requested the message.
            work_dir: The tab's working directory.
        """
        try:
            self._generate_commit_message(tab_id, work_dir=work_dir)
        finally:
            with self._state_lock:
                self._commit_msg_tabs.discard(tab_id)

    def _cmd_autocommit_action(self, cmd: dict[str, Any]) -> None:
        """Stage-all + commit the tab's working tree in the background.

        Serves the settings panel's "Git Commit" button.  Delegates to
        :meth:`_autocommit_changes` with ``manual=True`` (the same
        path the post-task autocommit uses), which stages everything,
        generates an LLM commit message from the staged diff alone
        (no ``User prompt:`` / ``Result:`` sections), commits, and
        reports through toast ``notification`` events — the chat
        transcript stays clean except for a failure, whose reason is
        still rendered there via the non-silent ``autocommit_done``
        event.

        The command's ``workDir`` (the tab's own folder) is forwarded so
        the commit lands in the tab's repository rather than the
        daemon-wide ``self.work_dir``, which may point at a different —
        possibly non-git — folder.

        At most one autocommit runs per tab: the commit-message
        generation is a billed LLM call and ``git add -A``/``git
        commit`` mutate the repository, so an impatient double click
        must not race two commits.  Extra clicks are dropped while the
        tab's autocommit is in flight.

        While a non-worktree task is running in the tab's repository
        the commit is refused: ``git add -A`` would snapshot whatever
        half-written state the agent happens to be in, producing an
        unintended intermediate commit that claims success while later
        task writes stay dirty.  Worktree tasks write inside their own
        linked worktree and never dirty the main tree, so they do not
        block a manual commit.
        """
        tab_id = cmd.get("tabId", "")
        work_dir = cmd.get("workDir", "") or self.work_dir
        repo = GitWorktreeOps.discover_repo(Path(work_dir))
        with self._state_lock:
            if repo is not None and self._any_non_wt_running(repo):
                self._broadcast_autocommit_done(
                    tab_id, success=False, committed=False,
                    message="A task is still running in this folder; "
                            "wait for it to finish before committing.",
                    manual=True, work_dir=work_dir,
                )
                return
            if tab_id in self._autocommit_tabs:
                logger.debug(
                    "Autocommit already in flight for tab %r; "
                    "ignoring duplicate request", tab_id,
                )
                return
            self._autocommit_tabs.add(tab_id)
        threading.Thread(
            target=self._run_autocommit_job,
            args=(tab_id, work_dir),
            daemon=True,
        ).start()

    def _run_autocommit_job(self, tab_id: str, work_dir: str) -> None:
        """Commit the tab's working tree and re-arm the button.

        Body of the daemon thread spawned by
        :meth:`_cmd_autocommit_action`; the ``finally`` releases the
        tab's in-flight claim so a failed commit never wedges the tab
        out of ever committing again.

        Args:
            tab_id: Frontend tab that requested the commit.
            work_dir: The tab's working directory.
        """
        try:
            self._autocommit_changes(tab_id, work_dir=work_dir, manual=True)
        finally:
            with self._state_lock:
                self._autocommit_tabs.discard(tab_id)

    def _cmd_worktree_action(self, cmd: dict[str, Any]) -> None:
        """Execute a worktree merge/discard action."""
        action = cmd.get("action", "")
        wt_tab_id = cmd.get("tabId", "")
        try:
            result = self._handle_worktree_action(action, wt_tab_id)
        except Exception as e:
            logger.debug("Worktree action error", exc_info=True)
            result = {"success": False, "message": str(e)}
        self.printer.broadcast(
            {"type": "worktree_result", "tabId": wt_tab_id, **result},
        )

    def _cmd_main_tree_action(self, cmd: dict[str, Any]) -> None:
        """Execute a main-tree discard/do-nothing action.

        Serves the Discard and Do-nothing buttons of the post-task
        action bar shown after a non-worktree manual-commit run (the
        bar's Auto-commit button sends the existing
        ``autocommitAction`` command instead).  Delegates to
        :meth:`_handle_main_tree_action` and always answers with a
        broadcast ``main_tree_result`` event so every client dismisses
        the bar — mirroring :meth:`_cmd_worktree_action`.
        """
        action = cmd.get("action", "")
        tab_id = cmd.get("tabId", "")
        work_dir = cmd.get("workDir", "")
        try:
            result = self._handle_main_tree_action(action, work_dir)
        except Exception as e:
            logger.debug("Main-tree action error", exc_info=True)
            result = {"success": False, "message": str(e)}
        self.printer.broadcast(
            {"type": "main_tree_result", "tabId": tab_id, **result},
        )

    def _cmd_get_config(self, cmd: dict[str, Any]) -> None:
        """Send the current configuration to the frontend.

        The reported ``work_dir`` is taken from the command's
        ``workDir`` — stamped per connection by
        :class:`RemoteAccessServer` — whenever the connection has one,
        falling back to the globally saved value only for connections
        that never announced a folder.  Each connection (one per
        VS Code window, one per webapp instance) runs its commands in
        its own stamped work_dir (``task_runner`` resolves
        ``cmd["workDir"]`` first), so the settings panel must show the
        directory that will actually be used by *this* instance, not
        whichever folder another instance persisted last.
        """
        from kiss.core.vscode_config import get_current_api_keys, load_config

        cfg = load_config()
        if cmd.get("workDir"):
            cfg["work_dir"] = cmd["workDir"]
        api_keys = get_current_api_keys()
        event: dict[str, Any] = {
            "type": "configData", "config": cfg, "apiKeys": api_keys,
        }
        conn_id = cmd.get("connId", "")
        if conn_id:
            event["connId"] = conn_id
        self.printer.broadcast(event)

    def _cmd_save_config(self, cmd: dict[str, Any]) -> None:
        """Save configuration and API keys from the frontend.

        When the ``remote_password`` actually *changes* to a non-empty
        value, restarts the ``kiss-web`` daemon so it picks up the new
        password and starts (or restarts) its Cloudflare tunnel.

        The change comparison is essential: the webview passively
        flushes the settings form (settings-panel close, blur/change/
        Enter on the password inputs), echoing back the already-saved
        password verbatim.  Restarting on every such echo SIGTERMed the
        daemon mid-task with no user action — the regression that
        persisted ``"Task interrupted by server restart/shutdown"`` for
        in-flight tasks (e.g. task_history row 3515).

        W2-F13: the ``prev_password`` read, the ``save_config`` write,
        and the env re-apply are held under a dedicated lock so two
        concurrent ``saveConfig`` commands (two windows closing their
        settings panels together) cannot both observe the OLD on-disk
        password and both conclude "changed" (dispatching two daemon
        restarts), nor interleave ``apply_config_to_env`` with a
        half-merged config.
        """
        from kiss.core.vscode_config import (
            apply_config_to_env,
            load_config,
            sanitize_config,
            save_api_key,
            save_config,
        )

        cfg = cmd.get("config", {})
        if not isinstance(cfg, dict):
            cfg = {}
        cfg = sanitize_config(cfg)
        with _CommandsMixin._save_config_lock:
            prev_password = load_config().get("remote_password", "")
            if not cfg.get("remote_password") and prev_password:
                cfg.pop("remote_password", None)
            save_config(cfg)
            apply_config_to_env(load_config())
            new_password = cfg.get("remote_password", "")
            password_changed = bool(
                new_password and new_password != prev_password,
            )

            new_work_dir = cfg.get("work_dir", "")
            if new_work_dir:
                self._apply_new_work_dir(new_work_dir)

            # Persist API keys INSIDE ``_save_config_lock``: each
            # ``save_api_key`` edits the canonical key store and the
            # shell RC, and serializing the writes under the same lock
            # that already guards config.json keeps two concurrent
            # ``saveConfig`` calls from interleaving.  An empty value
            # deletes the key from every store (canonical file, legacy
            # systemd mirror, shell RC) — that is the settings panel's
            # delete path.
            api_keys = cmd.get("apiKeys", {})
            if not isinstance(api_keys, dict):
                api_keys = {}
            for key_name, key_value in api_keys.items():
                if (
                    isinstance(key_name, str)
                    and isinstance(key_value, str)
                ):
                    save_api_key(key_name, key_value)

        conn_id = cmd.get("connId", "")
        self._get_models(conn_id)

        new_cfg = load_config()
        event: dict[str, Any] = {"type": "configData", "config": new_cfg}
        if conn_id:
            event["connId"] = conn_id
        self.printer.broadcast(event)

        if password_changed:
            _restart_kiss_web_daemon()

    def _cmd_set_work_dir(self, cmd: dict[str, Any]) -> None:
        """Update the server's *fallback* working directory.

        Sent by the VS Code extension on every (re)connect of its UDS
        client and whenever ``vscode.workspace.workspaceFolders``
        changes (i.e. the user opens a different folder), so a
        freshly-attached extension synchronises the daemon even when
        the daemon was started with a different ``KISS_WORKDIR``.

        Note that ``self.work_dir`` is only the last-resort fallback:
        each connection (one per VS Code window) keeps its own
        work_dir in the server API dispatcher
        (:meth:`kiss.server.sorcar.ServerApi.dispatch`), which stamps
        it onto every command from that connection that lacks an
        explicit ``workDir``.  Two windows sharing this
        daemon therefore never resolve to each other's folder even
        though both of their ``setWorkDir`` commands also land here.

        Clears the calling connection's ``_last_active_file`` snapshot
        (it refers to a file from that window's previous workspace),
        invalidates the connection's in-flight autocomplete generation,
        and, when the daemon-wide fallback actually changes, invalidates
        the autocomplete file cache.
        """
        new_dir = cmd.get("workDir", "")
        if not new_dir:
            return
        conn_id = cmd.get("connId", "")
        with self._state_lock:
            self._last_active_file.pop(conn_id, None)
            self._last_active_content.pop(conn_id, None)
            # Invalidate any in-flight completion for this connection:
            # a request computed against the OLD workspace's active
            # file would otherwise pass the worker's post-computation
            # freshness check (its seq still matches) and emit stale
            # old-workspace identifiers after the switch.  Removing
            # the entry makes both freshness checks in ``_complete``
            # fail (``seq != -1``); the next ``complete`` command
            # re-creates the entry with a fresh sequence number.
            self._complete_seq_latest.pop(conn_id, None)
            self._apply_new_work_dir(new_dir)

    _HANDLERS: dict[str, Any] = {
        "run": _cmd_run,
        "stop": _cmd_stop,
        "getModels": _cmd_get_models,
        "selectModel": _cmd_select_model,
        "getHistory": _cmd_get_history,
        "getFrequentTasks": _cmd_get_frequent_tasks,
        "deleteFrequentTask": _cmd_delete_frequent_task,
        "setFavorite": _cmd_set_favorite,
        "getFiles": _cmd_get_files,
        "recordFileUsage": _cmd_record_file_usage,
        "userAnswer": _cmd_user_answer,
        "appendUserMessage": _cmd_append_user_message,
        "resumeSession": _cmd_resume_session,
        "openTab": _cmd_open_tab,
        "closeTab": _cmd_close_tab,
        "newChat": _cmd_new_chat,
        "complete": _cmd_complete,
        "getInputHistory": _cmd_get_input_history,
        "getAdjacentTask": _cmd_get_adjacent_task,
        "generateCommitMessage": _cmd_generate_commit_message,
        "autocommitAction": _cmd_autocommit_action,
        "worktreeAction": _cmd_worktree_action,
        "mainTreeAction": _cmd_main_tree_action,
        "setWorkDir": _cmd_set_work_dir,
        "getConfig": _cmd_get_config,
        "saveConfig": _cmd_save_config,
    }
