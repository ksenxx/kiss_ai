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
from typing import TYPE_CHECKING, Any

from kiss.agents.sorcar.persistence import (
    _record_file_usage,
    _record_model_usage,
)
from kiss.agents.sorcar.running_agent_state import _RunningAgentState

if TYPE_CHECKING:
    from kiss.agents.vscode.json_printer import JsonPrinter

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

    Mirrors ``_cmd_get_adjacent_task``'s guarded parse so malformed
    payloads (e.g. ``"taskId": "abc"``) never raise out of a command
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


class _CommandsMixin:
    """Methods that implement frontend command handlers."""

    if TYPE_CHECKING:
        printer: JsonPrinter
        work_dir: str
        _state_lock: threading.RLock
        _default_model: str
        _complete_seq: int
        _complete_seq_latest: dict[str, int]
        _complete_queue: (
            queue.Queue[tuple[str, int, str, str, str, str]] | None
        )
        _last_active_file: dict[str, str]
        _last_active_content: dict[str, str]
        _file_cache: dict[str, list[str]]
        _tab_chat_views: dict[str, str]

        def _get_tab(self, tab_id: str) -> _RunningAgentState: ...
        def _run_task(self, cmd: dict[str, Any]) -> None: ...
        def _stop_task(self, tab_id: str = "") -> None: ...
        def _find_source_tab_for_viewer(
            self, viewer_tab_id: str,
        ) -> str | None: ...
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
            self, prefix: str, work_dir: str = "", conn_id: str = "",
        ) -> None: ...
        def _refresh_file_cache(
            self,
            then_emit_for_prefix: str | None = None,
            work_dir: str = "",
            conn_id: str = "",
        ) -> None: ...
        def _replay_session(
            self, chat_id: str, tab_id: str = "", task_id: int | None = None,
        ) -> None: ...
        def _finish_merge(
            self, tab_id: str = "", *, work_dir: str = "",
        ) -> None: ...
        def _new_chat(self, tab_id: str) -> None: ...
        def _close_tab(self, tab_id: str) -> None: ...
        def _ensure_complete_worker(self) -> None: ...
        def _get_input_history(self, conn_id: str = "") -> None: ...
        def _get_adjacent_task(
            self, chat_id: str, task_id: int | None, direction: str,
            tab_id: str = "",
        ) -> None: ...
        def _generate_commit_message(
            self, tab_id: str = "", *, work_dir: str = "",
        ) -> None: ...
        def _handle_worktree_action(
            self, action: str, tab_id: str = "", *, internal: bool = False,
        ) -> dict[str, Any]: ...
        def _handle_autocommit_action(
            self, action: str, tab_id: str = "", *, work_dir: str = "",
        ) -> None: ...
        def _handle_delete_task(self, task_id: int) -> None: ...
        def _handle_delete_frequent_task(self, task: str) -> None: ...
        def _handle_set_favorite(
            self, task_id: int, is_favorite: bool,
        ) -> None: ...


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
        if not tab_id:
            # An empty tab id would mint a phantom registry entry (and
            # start a real task thread) that no other code path can
            # ever address: ``_stop_task``, ``_cmd_close_tab`` and
            # ``_dispose_if_closed`` all treat an empty id as "no tab",
            # so the task would be unstoppable and undisposable.
            logger.debug("Ignoring run command without tabId")
            return
        inject_prompt: str | None = None
        thread: threading.Thread | None = None
        chat_id = ""
        with self._state_lock:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is None:
                tab = _RunningAgentState(tab_id, self._default_model)
                _RunningAgentState.running_agent_states[tab_id] = tab
            if tab.task_thread is not None:
                # A task is already running — or has been created and is
                # about to start — for this tab.  We must NOT start a
                # second task here.  Checking ``is_alive()`` would be
                # racy: the winning submit assigns ``task_thread`` under
                # the lock but calls ``thread.start()`` only after
                # releasing it (and after the ``clear`` broadcast —
                # network I/O), so a concurrent submit could observe a
                # created-but-unstarted thread (``is_alive() == False``),
                # pass the guard, and clobber ``stop_event`` /
                # ``user_answer_queue`` / ``task_thread`` — leaving two
                # tasks running on one tab with the first unstoppable.
                # ``_run_task``'s outer finally always resets
                # ``task_thread`` to None, so non-None ⇔ task in flight.
                #
                # The user's text must NOT be silently dropped, though.
                # A ``run`` that arrives while a task is in flight is
                # routed into the live agent as a follow-up user message
                # — identical to ``appendUserMessage``.  This is the root
                # cause of "input ignored during the task" after a
                # close+reopen: a re-opened webview can momentarily still
                # believe the task is idle (before the resume's
                # ``status running:true`` arrives) and therefore send the
                # typed text as a ``submit`` (→ ``run``) rather than an
                # ``appendUserMessage``.  The daemon is the source of
                # truth for whether a task is live, so it must inject the
                # prompt here instead of discarding it.  A tab that
                # started a task then behaves exactly like a tab that
                # loaded one: typed input never vanishes.
                prompt = cmd.get("prompt", "")
                if (
                    isinstance(prompt, str)
                    and prompt.strip()
                    and tab.is_task_active
                ):
                    tab.pending_user_messages.append(prompt)
                    inject_prompt = prompt
            else:
                tab.stop_event = threading.Event()
                tab.user_answer_queue = queue.Queue(maxsize=1)
                # Establish the canonical chat id for this run.  When
                # :meth:`_replay_session` has already populated
                # ``tab.chat_id`` with the chat id of a resumed history
                # row, preserve it — otherwise the follow-up task would
                # be cut off from the prior chat context
                # (``ChatSorcarAgent.build_chat_prompt`` would query
                # history for the tab id, find nothing, and send the LLM
                # an empty preamble).  Otherwise allocate a fresh chat
                # id.  ``tab_id`` (the frontend routing key) and
                # ``chat_id`` (the persistence key) are kept orthogonal:
                # every run gets its own chat id, regardless of which tab
                # launched it.
                if not tab.chat_id:
                    tab.chat_id = uuid.uuid4().hex
                chat_id = tab.chat_id
                # The launching tab views this chat: record it so a later
                # task started on the same chat from ANOTHER tab (in any
                # window) streams its live events here too (see
                # ``_subscribe_chat_viewers``).
                self._tab_chat_views[tab_id] = chat_id
                thread = threading.Thread(
                    target=self._run_task, args=(cmd,), daemon=True
                )
                tab.task_thread = thread
        if thread is None:
            # Busy tab: the run was routed into the running agent (or
            # ignored as blank).  Echo the injected follow-up so the user
            # sees their queued message in the chat surface, mirroring
            # ``_cmd_append_user_message``.  Broadcast OUTSIDE the lock
            # (network I/O) and never start a second task.
            if inject_prompt is not None:
                self.printer.broadcast({
                    "type": "prompt",
                    "text": inject_prompt,
                    "tabId": tab_id,
                })
            return
        # Emit ``clear`` synchronously (outside the state lock) so the
        # extension layer's chat_id → tab_id index is populated before
        # this command returns.  Without this, a ``resumeSession``
        # racing the worker thread's first broadcast would not find
        # the live task process and live events would never reach the
        # newly-opened tab.
        self.printer.broadcast({
            "type": "clear",
            "chat_id": chat_id,
            "tabId": tab_id,
        })
        thread.start()

    def _cmd_stop(self, cmd: dict[str, Any]) -> None:
        """Stop a running task."""
        self._stop_task(cmd.get("tabId", ""))

    def _cmd_get_models(self, cmd: dict[str, Any]) -> None:
        """Send available models list to the requesting connection only."""
        self._get_models(cmd.get("connId", ""))

    def _cmd_select_model(self, cmd: dict[str, Any]) -> None:
        """Update the selected model for a tab.

        An empty ``tabId`` (malformed payload) must not mint a phantom
        registry entry keyed ``""`` via ``_get_tab`` — such an entry
        could never be disposed because ``_cmd_close_tab`` guards
        against empty ids.  In that case only the daemon-wide default
        model is updated (when a model was actually supplied).
        """
        tab_id = cmd.get("tabId", "")
        model = cmd.get("model", "")
        if not isinstance(model, str):
            # A non-string model (malformed payload) must neither
            # corrupt ``tab.selected_model`` / ``self._default_model``
            # nor raise ``sqlite3.ProgrammingError`` out of
            # ``_record_model_usage`` (which would kill the client
            # connection).  Treat it as "no model supplied".
            model = ""
        with self._state_lock:
            if tab_id:
                tab = self._get_tab(tab_id)
                if not model:
                    model = tab.selected_model
                tab.selected_model = model
            if not model:
                return
            self._default_model = model
        _record_model_usage(model)

    def _cmd_get_history(self, cmd: dict[str, Any]) -> None:
        """Send conversation history to the requesting connection only."""
        query = cmd.get("query")
        if not isinstance(query, str):
            # A non-string query raises AttributeError inside
            # ``_search_history``'s LIKE escaping and kills the
            # connection; treat it as "no filter".
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

    def _cmd_delete_task(self, cmd: dict[str, Any]) -> None:
        """Delete a task from the database and refresh history."""
        task_id = _parse_int(cmd.get("taskId"))
        if task_id is not None:
            self._handle_delete_task(task_id)

    def _cmd_delete_frequent_task(self, cmd: dict[str, Any]) -> None:
        """Delete a row from the ``frequent_tasks`` table by task text."""
        task = cmd.get("task")
        if isinstance(task, str) and task:
            self._handle_delete_frequent_task(task)

    def _cmd_set_favorite(self, cmd: dict[str, Any]) -> None:
        """Persist the favourite flag on a task history row."""
        task_id = _parse_int(cmd.get("taskId"))
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
        VS Code window never pops the file picker in another window.
        """
        prefix = cmd.get("prefix", "")
        if not isinstance(prefix, str):
            # A non-string prefix crashes the background refresh
            # thread (TypeError in ``rank_file_suggestions``) and the
            # file picker never receives its reply.
            prefix = ""
        self._get_files(
            prefix,
            cmd.get("workDir", ""),
            cmd.get("connId", ""),
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
            # A non-string path would raise sqlite3.ProgrammingError
            # from the parameter binding and kill the connection.
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
            q = self._resolve_user_answer_queue(ans_tab)
            if q is None:
                logger.debug("userAnswer dropped: no queue for tabId=%s", ans_tab)
                return
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:  # pragma: no cover — race guard
                    break
            answer = cmd.get("answer", "")
            if not isinstance(answer, str):
                # A non-string answer (e.g. null from a malformed
                # client) must not leak through the ``Queue[str]`` into
                # the agent — ``ask_user_question`` promises ``str``.
                answer = "" if answer is None else str(answer)
            try:
                q.put_nowait(answer)
            except queue.Full:  # pragma: no cover — drained immediately above
                pass

    def _resolve_user_answer_queue(
        self, ans_tab: str,
    ) -> queue.Queue[str] | None:
        """Locate the answer queue an ``ask_user_question`` is waiting on.

        Routing precedence:

        1. The frontend tab id ``ans_tab`` itself, when its
           ``_RunningAgentState`` holds a non-None ``user_answer_queue``.
           This is the common path: a single-window user answers from
           the same tab that launched the task.

        2. Otherwise, the queue of any tab that shares a task
           subscription with ``ans_tab``.  This covers the multi-viewer
           case where one tab (e.g. a browser viewer of a chat owned by
           the VS Code extension's tab) renders the askUser modal and
           submits the answer: the broadcast was fan-stamped with the
           viewer's tab id, but the live ``user_answer_queue`` lives on
           the task-owner tab.  Without this fallback, the answer is
           silently dropped and the agent thread waits forever (or
           until the stop event eventually fires) — surfaced by users
           as "the answer was not delivered immediately".

        Args:
            ans_tab: Frontend tab id carried by the ``userAnswer``
                command.

        Returns:
            The resolved answer queue, or ``None`` when no live
            ``ask_user_question`` waiter can be associated with the
            command.  Must be called with ``_state_lock`` held.
        """
        ans_state = _RunningAgentState.running_agent_states.get(ans_tab)
        if ans_state is not None and ans_state.user_answer_queue is not None:
            return ans_state.user_answer_queue
        # Multi-viewer fallback: find a co-subscriber tab whose
        # ``user_answer_queue`` is live.  ``_subscribers`` is keyed by
        # task id; ``ans_tab`` and the owner tab share at least one
        # task id when the viewer is observing the owner's task.
        printer_lock = getattr(self.printer, "_lock", None)
        subs_map = getattr(self.printer, "_subscribers", {})
        if printer_lock is None:
            return None
        with printer_lock:
            shared_tasks = [
                task_id
                for task_id, viewers in subs_map.items()
                if ans_tab in viewers
            ]
            candidate_tabs: list[str] = []
            for task_id in shared_tasks:
                candidate_tabs.extend(subs_map.get(task_id, ()))
        for tab_id in candidate_tabs:
            if tab_id == ans_tab:
                continue
            state = _RunningAgentState.running_agent_states.get(tab_id)
            if state is not None and state.user_answer_queue is not None:
                return state.user_answer_queue
        return None

    def _cmd_append_user_message(self, cmd: dict[str, Any]) -> None:
        """Queue a user message to be injected into the running agent's context.

        When the user types into the task-input textbox while a task is
        still running, the frontend forwards the prompt here instead of
        silently dropping it.  We append the text to the tab's
        :attr:`_RunningAgentState.pending_user_messages` list under
        :attr:`_RunningAgentState._registry_lock` so the live agent's
        pre-step hook can drain and inject the messages into the model
        conversation before the next model call.

        When the tab itself has no live task (the common case for a
        VIEWER tab opened from the history sidebar while a task runs
        in ANOTHER tab — the viewer is subscribed to the source tab's
        event stream but the live agent runs in the source tab's
        ``_RunningAgentState``) the prompt is routed to the source
        tab's queue via the printer's per-task subscriber map.  This
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
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is not None and tab.is_task_active:
                tab.pending_user_messages.append(prompt)
            else:
                # Viewer-tab fallback: a tab opened from the history
                # sidebar while a task runs in ANOTHER tab carries no
                # live ``_RunningAgentState`` of its own — the live
                # agent (and its ``pending_user_messages`` queue) lives
                # on the source tab the viewer was subscribed to via
                # ``_reattach_running_chat`` /
                # ``_subscribe_chat_viewers``.  Resolve the source tab
                # through the printer's per-task subscriber map and
                # route the prompt there instead of dropping it.
                source_tab_id = self._find_source_tab_for_viewer(tab_id)
                if not source_tab_id:
                    logger.debug(
                        "appendUserMessage dropped: tab %s has no "
                        "live task and is not a viewer of one",
                        tab_id,
                    )
                    return
                source = _RunningAgentState.running_agent_states.get(
                    source_tab_id,
                )
                if source is None or not source.is_task_active:
                    logger.debug(
                        "appendUserMessage dropped: viewer tab %s "
                        "source tab %s has no live task",
                        tab_id, source_tab_id,
                    )
                    return
                source.pending_user_messages.append(prompt)
        # Echo on the originating tab id — that's the tab whose
        # transcript the user is looking at, so the prompt panel must
        # appear there even when the live agent is owned by a peer.
        self.printer.broadcast({
            "type": "prompt",
            "text": prompt,
            "tabId": tab_id,
        })

    def _cmd_resume_session(self, cmd: dict[str, Any]) -> None:
        """Replay a previous chat session.

        When ``taskId`` is present, load that specific task instead of
        the latest task in the chat session.
        """
        raw_id = cmd.get("chatId")
        chat_id = str(raw_id) if raw_id else ""
        task_id = _parse_int(cmd.get("taskId"))
        if chat_id or task_id is not None:
            self._replay_session(
                chat_id, cmd.get("tabId", ""), task_id=task_id,
            )

    def _cmd_merge_action(self, cmd: dict[str, Any]) -> None:
        """Handle merge accept/reject from the extension.

        Only ``all-done`` triggers cleanup. Individual ``accept``/``reject``
        actions are tracked on the TypeScript side; the Python server
        only needs to know when the entire merge session is finished.
        """
        if cmd.get("action", "") == "all-done":
            self._finish_merge(
                cmd.get("tabId", ""), work_dir=cmd.get("workDir", ""),
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
            # A non-string query queued to the singleton autocomplete
            # worker killed the worker thread (AttributeError inside
            # ``_prefix_match_task``); the worker is never restarted,
            # so ghost text would die for the daemon's whole lifetime.
            query = ""
        active_file = cmd.get("activeFile")
        active_content = cmd.get("activeFileContent")
        if active_file is not None and not isinstance(active_file, str):
            active_file = ""
        if active_content is not None and not isinstance(active_content, str):
            active_content = ""
        conn_id = cmd.get("connId", "")
        tab_id = cmd.get("tabId", "")
        chat_id = ""
        if tab_id:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is not None:
                chat_id = tab.chat_id
        with self._state_lock:
            if active_file:
                self._last_active_file[conn_id] = active_file
            if active_content is not None:
                self._last_active_content[conn_id] = active_content
            snapshot_file = self._last_active_file.get(conn_id, "")
            snapshot_content = self._last_active_content.get(conn_id, "")
            self._complete_seq += 1
            seq = self._complete_seq
            self._complete_seq_latest[conn_id] = seq
        if query:
            self._ensure_complete_worker()
            self._complete_queue.put(  # type: ignore[union-attr]
                (query, seq, snapshot_file, snapshot_content, chat_id, conn_id),
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
        ``_replay_session``) deliberately have NO
        ``_RunningAgentState`` registry entry (C2/C3 fix) — only a
        ``_tab_chat_views`` association.  Resolve the chat id from the
        registry entry when one exists, falling back to the
        chat-viewer map, and never CREATE a registry entry here:
        navigation is a read-only view operation.
        """
        tab_id = cmd.get("tabId", "")
        with self._state_lock:
            adj_tab = _RunningAgentState.running_agent_states.get(tab_id)
            chat_id = adj_tab.chat_id if adj_tab is not None else ""
            if not chat_id:
                chat_id = self._tab_chat_views.get(tab_id, "")
        raw_task_id = cmd.get("taskId")
        try:
            task_id = int(raw_task_id) if raw_task_id is not None else None
        except (TypeError, ValueError):
            task_id = None
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
        """
        tab_id = cmd.get("tabId", "")
        work_dir = cmd.get("workDir", "")
        threading.Thread(
            target=self._generate_commit_message,
            args=(tab_id,),
            kwargs={"work_dir": work_dir},
            daemon=True,
        ).start()

    def _cmd_worktree_action(self, cmd: dict[str, Any]) -> None:
        """Execute a worktree merge/discard action."""
        action = cmd.get("action", "")
        wt_tab_id = cmd.get("tabId", "")
        try:
            result = self._handle_worktree_action(action, wt_tab_id)
        except Exception as e:
            logger.debug("Worktree action error", exc_info=True)
            result = {"success": False, "message": str(e)}
        self.printer.broadcast({"type": "worktree_result", "tabId": wt_tab_id, **result})

    def _cmd_autocommit_action(self, cmd: dict[str, Any]) -> None:
        """Process the user's reply to an autocommit prompt."""
        self._handle_autocommit_action(
            cmd.get("action", ""), cmd.get("tabId", ""),
            work_dir=cmd.get("workDir", ""),
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
        from kiss.agents.vscode.vscode_config import get_current_api_keys, load_config

        cfg = load_config()
        if cmd.get("workDir"):
            cfg["work_dir"] = cmd["workDir"]
        api_keys = get_current_api_keys()
        event: dict[str, Any] = {
            "type": "configData", "config": cfg, "apiKeys": api_keys,
        }
        conn_id = cmd.get("connId", "")
        if conn_id:
            # Reply only to the requesting window: another window may
            # have its settings form open with unsaved edits, which an
            # unsolicited configData repaint would clobber.
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
        """
        from kiss.agents.vscode.vscode_config import (
            apply_config_to_env,
            load_config,
            sanitize_config,
            save_api_key_to_shell,
            save_config,
        )

        prev_password = load_config().get("remote_password", "")
        cfg = cmd.get("config", {})
        if not isinstance(cfg, dict):
            # A non-dict config (malformed payload) raises
            # AttributeError below and kills the connection.
            cfg = {}
        # Coerce junk-typed values BEFORE any decision is made on them:
        # a non-string ``work_dir`` must not corrupt ``self.work_dir``
        # and a truthy non-string ``remote_password`` must not count as
        # a genuine password change (which restarts the kiss-web
        # daemon, killing every in-flight task).
        cfg = sanitize_config(cfg)
        # Guard: never overwrite a non-empty remote_password with an
        # empty one from the frontend.  An empty value typically comes
        # from a race condition (config sidebar closed before the async
        # getConfig response populated the form fields).
        if not cfg.get("remote_password") and prev_password:
            cfg.pop("remote_password", None)
        save_config(cfg)
        apply_config_to_env(cfg)

        new_work_dir = cfg.get("work_dir", "")
        if new_work_dir:
            self.work_dir = new_work_dir

        api_keys = cmd.get("apiKeys", {})
        if not isinstance(api_keys, dict):
            api_keys = {}
        for key_name, key_value in api_keys.items():
            if isinstance(key_name, str) and isinstance(key_value, str) and key_value:
                save_api_key_to_shell(key_name, key_value)

        conn_id = cmd.get("connId", "")
        self._get_models(conn_id)

        new_cfg = load_config()
        event: dict[str, Any] = {"type": "configData", "config": new_cfg}
        if conn_id:
            event["connId"] = conn_id
        self.printer.broadcast(event)

        # Restart only on a genuine password change.  A daemon restart
        # kills every in-flight agent task, and the frontend re-posts
        # the unchanged password on passive UI events (panel close,
        # input blur), so an unconditional restart here is destructive.
        new_password = cfg.get("remote_password", "")
        if new_password and new_password != prev_password:
            _restart_kiss_web_daemon()

    def _cmd_cli_info(self, cmd: dict[str, Any]) -> None:
        """Reply to a ``cliInfo`` request from the sorcar CLI client.

        Handles the slash-command surface that was previously rendered
        only inside the standalone REPL (``/help``, ``/commands``,
        ``/skills``, ``/skills <name>``, ``/mcp``, ``/cost``,
        ``/model`` with no argument) by reusing the very same helpers
        from :mod:`kiss.agents.sorcar.custom_commands`,
        :mod:`kiss.agents.sorcar.skills` and
        :mod:`kiss.agents.sorcar.mcp_servers`, plus custom-command
        expansion.

        The reply is emitted as a single ``cliInfo`` event stamped
        with the originating ``connId`` so it is delivered ONLY to the
        requesting CLI client (mirroring the per-connection routing of
        ``models`` / ``files`` / ``configData`` replies).

        Args:
            cmd: The parsed ``cliInfo`` command from the CLI client,
                carrying a ``subtype`` field selecting the reply, plus
                any subtype-specific arguments (``arg`` / ``name`` /
                ``args`` / ``tabId`` / ``workDir``).
        """
        from kiss.agents.sorcar.cli_repl import SLASH_COMMANDS
        from kiss.agents.sorcar.custom_commands import (
            discover_commands,
            expand_command,
            format_command_listing,
        )

        subtype = cmd.get("subtype", "")
        work_dir = cmd.get("workDir", "") or self.work_dir or "."
        conn_id = cmd.get("connId", "")
        tab_id = cmd.get("tabId", "")
        text = ""
        extra: dict[str, Any] = {}

        if subtype == "help":
            lines = ["Commands:"]
            for c, d in SLASH_COMMANDS.items():
                lines.append(f"  {c:<10} {d}")
            try:
                custom = discover_commands(work_dir)
            except Exception:  # pragma: no cover - discovery guard
                logger.debug("custom command discovery failed", exc_info=True)
                custom = {}
            if custom:
                lines.append("")
                lines.append("Custom commands:")
                lines.append(format_command_listing(custom))
            lines.append("")
            lines.append(
                "Input fast-completes (Tab): @path mentions files, "
                "/ completes commands, /model <partial> completes model "
                "names, and typing a prefix of a previous task suggests "
                "its completion.",
            )
            text = "\n".join(lines)
        elif subtype == "commands":
            try:
                custom = discover_commands(work_dir)
            except Exception:  # pragma: no cover - discovery guard
                logger.debug("custom command discovery failed", exc_info=True)
                custom = {}
            text = format_command_listing(custom)
        elif subtype == "skills":
            from kiss.agents.sorcar.skills import (
                discover_skills,
                format_skill_listing,
                load_skill_content,
            )

            try:
                skills = discover_skills(work_dir)
            except Exception:  # pragma: no cover - discovery guard
                logger.debug("skill discovery failed", exc_info=True)
                skills = {}
            name = cmd.get("name", "") or cmd.get("arg", "")
            if name:
                found = skills.get(name)
                if found is None:
                    text = f"Unknown skill: {name}. /skills lists them."
                else:
                    try:
                        text = load_skill_content(found)
                    except Exception as exc:  # pragma: no cover - read guard
                        text = f"Failed to load skill {name}: {exc}"
            else:
                text = format_skill_listing(skills)
        elif subtype == "mcp":
            from kiss.agents.sorcar.mcp_servers import format_mcp_listing

            try:
                text = format_mcp_listing(work_dir, connect=True)
            except Exception as exc:  # pragma: no cover - listing guard
                text = f"Failed to list MCP servers: {exc}"
        elif subtype == "cost":
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            chat_id = tab.chat_id if tab is not None else ""
            budget = float(getattr(tab, "budget_used", 0.0) or 0.0) if tab else 0.0
            tokens = int(getattr(tab, "total_tokens_used", 0) or 0) if tab else 0
            text = (
                f"Chat ID: {chat_id or '(new)'}\n"
                f"Cost: ${budget:.4f}\n"
                f"Total tokens: {tokens}"
            )
            extra["chatId"] = chat_id
            extra["cost"] = budget
            extra["tokens"] = tokens
        elif subtype == "modelCurrent":
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            current = tab.selected_model if tab is not None else self._default_model
            text = f"Current model: {current}"
            extra["model"] = current
        elif subtype == "expandCommand":
            name = cmd.get("name", "") or cmd.get("arg", "")
            args = cmd.get("args", "")
            try:
                custom = discover_commands(work_dir)
            except Exception:  # pragma: no cover - discovery guard
                logger.debug("custom command discovery failed", exc_info=True)
                custom = {}
            found_cmd = custom.get(name)
            if found_cmd is None:
                text = ""
                extra["found"] = False
                extra["error"] = f"Unknown command: /{name}"
            else:
                try:
                    expanded = expand_command(found_cmd, args, work_dir)
                except Exception as exc:  # pragma: no cover - expansion guard
                    text = ""
                    extra["found"] = False
                    extra["error"] = f"Expansion failed: {exc}"
                else:
                    text = expanded
                    extra["found"] = True
                    extra["source"] = found_cmd.source
                    extra["path"] = str(found_cmd.path)
        else:
            text = f"Unknown cliInfo subtype: {subtype}"

        event: dict[str, Any] = {
            "type": "cliInfo",
            "subtype": subtype,
            "text": text,
            "tabId": tab_id,
        }
        event.update(extra)
        if conn_id:
            event["connId"] = conn_id
        self.printer.broadcast(event)

    def _cmd_set_work_dir(self, cmd: dict[str, Any]) -> None:
        """Update the server's *fallback* working directory.

        Sent by the VS Code extension on every (re)connect of its UDS
        client and whenever ``vscode.workspace.workspaceFolders``
        changes (i.e. the user opens a different folder), so a
        freshly-attached extension synchronises the daemon even when
        the daemon was started with a different ``KISS_WORKDIR``.

        Note that ``self.work_dir`` is only the last-resort fallback:
        each connection (one per VS Code window) keeps its own
        work_dir in ``RemoteAccessServer._dispatch_client_command``,
        which stamps it onto every command from that connection that
        lacks an explicit ``workDir``.  Two windows sharing this
        daemon therefore never resolve to each other's folder even
        though both of their ``setWorkDir`` commands also land here.

        Clears the calling connection's ``_last_active_file`` snapshot
        (it refers to a file from that window's previous workspace) and,
        when the daemon-wide fallback actually changes, invalidates the
        autocomplete file cache.
        """
        new_dir = cmd.get("workDir", "")
        if not new_dir:
            return
        conn_id = cmd.get("connId", "")
        with self._state_lock:
            # The calling window switched folders: its last-reported
            # active editor file belongs to the previous workspace and
            # must not feed that window's autocomplete any more.  Only
            # the caller's own snapshot is dropped — other windows'
            # snapshots stay valid (their folders did not change).
            self._last_active_file.pop(conn_id, None)
            self._last_active_content.pop(conn_id, None)
            if self.work_dir == new_dir:
                return
            self.work_dir = new_dir
            # Stale cache from the previous folder must not bleed
            # into the new folder's autocomplete results.  The cache
            # is keyed per work_dir so distinct keys would not actually
            # cross-contaminate, but switching the daemon-wide folder
            # is a clear signal that any in-memory file lists are
            # potentially stale (files may have been added/removed
            # while the daemon was pointed elsewhere), so wipe them
            # all and let subsequent ``getFiles`` rebuild lazily.
            self._file_cache = {}
        # Keep the printer's work_dir in sync so global ``configData``
        # events report the active folder (the ``WebPrinter`` used by
        # the remote server fills ``cfg["work_dir"]`` from its own
        # ``work_dir`` attribute; without this it would keep reporting
        # the folder the daemon was launched with).
        if hasattr(self.printer, "work_dir"):
            setattr(self.printer, "work_dir", new_dir)

    _HANDLERS: dict[str, Any] = {
        "run": _cmd_run,
        "stop": _cmd_stop,
        "getModels": _cmd_get_models,
        "selectModel": _cmd_select_model,
        "getHistory": _cmd_get_history,
        "getFrequentTasks": _cmd_get_frequent_tasks,
        "deleteTask": _cmd_delete_task,
        "deleteFrequentTask": _cmd_delete_frequent_task,
        "setFavorite": _cmd_set_favorite,
        "getFiles": _cmd_get_files,
        "recordFileUsage": _cmd_record_file_usage,
        "userAnswer": _cmd_user_answer,
        "appendUserMessage": _cmd_append_user_message,
        "resumeSession": _cmd_resume_session,
        "mergeAction": _cmd_merge_action,
        "closeTab": _cmd_close_tab,
        "newChat": _cmd_new_chat,
        "complete": _cmd_complete,
        "getInputHistory": _cmd_get_input_history,
        "getAdjacentTask": _cmd_get_adjacent_task,
        "generateCommitMessage": _cmd_generate_commit_message,
        "worktreeAction": _cmd_worktree_action,
        "autocommitAction": _cmd_autocommit_action,
        "setWorkDir": _cmd_set_work_dir,
        "getConfig": _cmd_get_config,
        "saveConfig": _cmd_save_config,
        "cliInfo": _cmd_cli_info,
    }
