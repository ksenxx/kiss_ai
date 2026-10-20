# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Command handlers for the daemon's backend server.

Split out of ``server.py`` for organisation.  ``_CommandsMixin``
provides one ``_cmd_*`` method per frontend command type plus the
class-level ``_HANDLERS`` dispatch table consumed by
``AgentServer._handle_command``.
"""

from __future__ import annotations

import functools
import logging
import queue
import sqlite3
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kiss.agents.sorcar._concurrency import _race_delay
from kiss.agents.sorcar.machines import local_machine_name
from kiss.agents.sorcar.persistence import (
    _chat_task_tree,
    _delete_chat,
    _delete_task,
    _load_slash_command_usage,
    _record_file_usage,
    _record_model_usage,
    _record_steer_input,
    _save_last_model,
    _set_chat_summary,
    _set_task_favorite,
    _task_tree,
)
from kiss.agents.sorcar.sea_commands import (
    RESERVED_SUBCOMMANDS,
    SeaError,
    help_text_if_command,
    list_commands,
    run_picked_hook,
)
from kiss.agents.sorcar.sorcar_agent import _persisted_task_id
from kiss.core.brand import HOME_DIR
from kiss.core.browser_handoff import open_in_default_browser
from kiss.core.models.model import Attachment
from kiss.core.utils import is_root_dir
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.tab_registry import OpenTabOutcome
from kiss.server.task_runner import (
    _client_task_id_of,
    _result_event,
    contains_task_tags,
    parse_task_tags,
    pending_attachments,
)

if TYPE_CHECKING:
    from kiss.server.browser_tab import BrowserTabService
    from kiss.server.file_index import FileIndexRegistry
    from kiss.server.json_printer import JsonPrinter
    from kiss.server.tab_registry import TabRegistry

logger = logging.getLogger(__name__)


def _reruns_after_teardown(owner: AgentState) -> bool:
    """True when a prompt queued on *owner* now can only run as the tab's next run.

    ``followup_queue_closed`` is raised once the run's subtask loop is
    over — after the agent's last step, so no pre-step drain will ever
    consume ``pending_user_messages`` again.  Whatever is queued from
    then on is picked up by ``TaskRunner._run_task``'s cleanup and
    re-submitted through ``_cmd_run`` as a fresh run on the same tab,
    which echoes the text as its own prompt; the caller therefore
    skips the steering echo (the message would otherwise appear
    twice).  Sub-agent runs (``parentTaskId``) and tabs the user has
    closed are not re-run, so their late prompts keep the plain
    steering echo.

    MUST be called while holding :data:`agent_state.STATE_LOCK`.

    Args:
        owner: The running-task state that just queued the prompt.
    """
    return (
        owner.server_owned
        and owner.followup_queue_closed
        and not owner.is_subagent
        and not owner.frontend_closed
    )


# How many of the most recently run commands the ``/`` popup lists
# first, and the run count from which a command counts as frequent.
_RECENT_COMMANDS = 3
_FREQUENT_COMMAND_MIN_RUNS = 2


def sea_commands_event(commands: list[str]) -> dict[str, Any]:
    """Build the ``seaCommands`` event the composer's ``/`` popup renders.

    The popup lists the commands the user ran most recently first,
    then the ones run often, then the rest; both ranked lists come
    from the task history (see
    :func:`kiss.agents.sorcar.persistence._load_slash_command_usage`)
    so every surface shows the same order.

    Args:
        commands: The registered command names (``list_commands()``).

    Returns:
        ``{"type": "seaCommands", "commands": [...], "recent": [...],
        "frequent": [...]}`` -- ``recent`` is the :data:`_RECENT_COMMANDS`
        most recently run registered commands, newest first;
        ``frequent`` the other registered commands run at least
        :data:`_FREQUENT_COMMAND_MIN_RUNS` times, most-run first (ties
        newest first).  Both are empty when the history cannot be read.
    """
    try:
        usage = _load_slash_command_usage()
    except sqlite3.Error:
        logger.warning("slash command usage unavailable", exc_info=True)
        usage = {}
    used = sorted(
        (name for name in commands if name in usage),
        key=lambda name: usage[name][1],
        reverse=True,
    )
    recent = used[:_RECENT_COMMANDS]
    frequent = sorted(
        (name for name in used[_RECENT_COMMANDS:] if usage[name][0] >= _FREQUENT_COMMAND_MIN_RUNS),
        key=lambda name: usage[name][0],
        reverse=True,
    )
    return {
        "type": "seaCommands",
        "commands": commands,
        "recent": recent,
        "frequent": frequent,
    }


# Prefix that flags an ``appendUserMessage`` as a live-side-channel
# /ask query.  Kept identical to the SEA registry stem so a
# ``list_commands`` reader and the interceptor agree on the name.
_ASK_COMMAND_PREFIX = "/ask"
# Seconds an /ask answering session may run before it is stopped.
_ASK_TIMEOUT_SECONDS = 600.0


def _split_ask_command(prompt: str) -> str | None:
    """Return the question text of a ``/ask <question>`` prompt.

    Returns the question with surrounding whitespace stripped when
    *prompt* begins with ``/ask`` followed by whitespace and at least
    one non-whitespace character; otherwise returns ``None``.  A bare
    ``/ask`` or ``/ask   `` returns ``None`` (no question to answer),
    and a prompt whose ``/ask`` prefix is glued to more text
    (``/askme``) does not match — the same word-boundary rule the
    general SEA slash-command parser uses.

    Args:
        prompt: The raw user message.

    Returns:
        The question text, or ``None``.
    """
    if not prompt.startswith(_ASK_COMMAND_PREFIX):
        return None
    tail = prompt[len(_ASK_COMMAND_PREFIX):]
    if not tail or not tail[:1].isspace():
        return None
    question = tail.strip()
    return question or None


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


def broadcast_to_conn(
    printer: Any,
    event: dict[str, Any],
    conn_id: str,
) -> None:
    """Broadcast *event* on *printer*, stamped with *conn_id* when non-empty.

    Stamping ``connId`` makes the printer deliver the event ONLY to the
    requesting connection (the browser tab whose user triggered the
    command), so one tab's request never repaints — or pops a banner
    in — another tab's UI; ``""`` broadcasts to all.
    Shared by :meth:`_CommandsMixin._broadcast_to_conn` and
    ``RemoteAccessServer._broadcast_to_conn`` (web_server.py).

    Args:
        printer: Any printer exposing ``broadcast(event)``.
        event: The event payload to broadcast (mutated in place).
        conn_id: Requesting connection id (``""`` reaches all).
    """
    if conn_id:
        event["connId"] = conn_id
    printer.broadcast(event)


def _any_running(task_ids: list[str], chat_id: str = "") -> bool:
    """Whether a live agent state runs one of *task_ids* (or a task of *chat_id*).

    A sub-task that has not allocated its ``task_history`` row yet is
    known only by its parent (``state.parent_task_id``): it counts when
    its parent is among *task_ids*.

    Args:
        task_ids: The persisted task ids a delete would remove.
        chat_id: A chat whose running tasks count too (a run that has
            no ``task_history`` row yet is known by its chat alone).
    """
    wanted = set(task_ids)
    for state in agent_state.snapshot():
        if not state.running():
            continue
        if (
            state.task_id in wanted
            or (state.parent_task_id or "") in wanted
            or (chat_id and state.chat_id == chat_id)
        ):
            return True
    return False


class _CommandsMixin:
    """Methods that implement frontend command handlers."""

    _save_config_lock = threading.Lock()
    # Serializes a whole working-directory adoption (live value,
    # persistence, broadcast — see ``_apply_new_work_dir``): commands run
    # on a thread pool, so two concurrent adoptions would otherwise
    # interleave and leave the disk or the clients on the loser.
    # Ordering: taken BEFORE ``_state_lock`` and ``_save_config_lock``,
    # never while holding either.
    _work_dir_lock = threading.Lock()

    # Admission barrier for the self-update: ``True`` from the moment
    # the web server decides to spawn ``install.sh`` (the idle poller's
    # "no active tasks" verdict, or an explicit Update click) until the
    # installer exits.  ``_cmd_run`` refuses to start a NEW task while
    # it is up; it is read and written only under ``_state_lock``
    # (:data:`agent_state.STATE_LOCK`).  A class-level default keeps
    # it valid on every ``AgentServer`` instance.
    _update_installing: bool = False

    if TYPE_CHECKING:
        printer: JsonPrinter
        work_dir: str
        _state_lock: threading.RLock
        _shutdown_stopping: bool
        _default_model: str
        _model_pick: tuple[int, str]
        _model_picks_pending: int
        _complete_seq: int
        _complete_seq_latest: dict[str, int]
        _complete_queue: (
            queue.Queue[tuple[str, int, str, str | None, str, str, str]]
            | None
        )
        _last_active_file: dict[str, str]
        _last_active_content: dict[str, str]
        _file_index: FileIndexRegistry
        _tab_chat_views: dict[str, str]
        _tab_models: dict[str, str]
        _autocommit_tabs: set[str]
        tab_registry: TabRegistry
        browser_tabs: BrowserTabService

        def _broadcast_tabs_state(self) -> None: ...
        def _publish_tabs_state(self) -> None: ...

        def _run_task(self, cmd: dict[str, Any]) -> None: ...
        def _stop_task(
            self, tab_id: str = "", run_token: str = "",
        ) -> None: ...
        def _interrupt_tool_call(
            self, tab_id: str, tool_name: str = "", call_id: int | None = None,
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
            tag: str = "",
            machine: str = "",
            work_dir: str = "",
        ) -> None: ...
        def _get_files(
            self,
            prefix: str,
            work_dir: str = "",
            conn_id: str = "",
            tab_id: str = "",
        ) -> None: ...
        def _replay_session(
            self,
            chat_id: str,
            tab_id: str = "",
            task_id: str | None = None,
            conn_id: str = "",
            replay_id: str = "",
        ) -> None: ...
        def _defer_replay(
            self,
            chat_id: str,
            tab_id: str,
            task_id: str | None,
            conn_id: str,
        ) -> None: ...
        def _new_chat(self, tab_id: str) -> None: ...
        def _close_tab(self, tab_id: str) -> None: ...
        def _dispose_if_closed(self, tab_id: str) -> None: ...
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
        ) -> int: ...
        def _ensure_complete_worker(self) -> None: ...
        def _get_input_history(self, conn_id: str = "") -> None: ...
        def _get_adjacent_task(
            self, chat_id: str, task_id: str | None, direction: str,
            tab_id: str = "",
        ) -> None: ...
        def _commit_main_tree(
            self, tab_id: str, work_dir: str = "", reply_to: str = "",
        ) -> None: ...
        def _commit_on_machine(self, tab_id: str, machine: str, work_dir: str) -> None: ...
        def _broadcast_autocommit_done(
            self, tab_id: str, *, success: bool, committed: bool, message: str,
            commit_message: str | None = None, work_dir: str = "", reply_to: str = "",
        ) -> dict[str, Any]: ...
        def _handle_worktree_action(
            self, action: str, tab_id: str = "",
        ) -> dict[str, Any]: ...

    def _broadcast_to_conn(self, event: dict[str, Any], conn_id: str) -> None:
        """Broadcast *event*, stamped with *conn_id* when non-empty.

        Args:
            event: The event payload to broadcast (mutated in place).
            conn_id: Requesting connection id (``""`` reaches all).
        """
        broadcast_to_conn(self.printer, event, conn_id)

    def _apply_new_work_dir(self, new_dir: str) -> None:
        """Adopt *new_dir* as the one global working directory.

        The working directory is a single daemon-wide value that every
        task from every client (the remote webapp, the Python API's
        ``sorcar.run`` without ``work_dir=``) runs in.
        Shared by :meth:`_cmd_set_work_dir` and :meth:`_cmd_save_config`;
        when the directory actually changes it starts indexing the new
        directory for the ``@``-mention picker, persists it as
        ``config.json`` ``work_dir`` so a daemon restart keeps it, and
        broadcasts ``workDirChanged`` so every connected client
        re-scopes its Explorer, history and chat-header work-dir drop-down.
        Mirrors the value onto the printer either way.  Takes
        ``_work_dir_lock`` and then ``_state_lock`` itself, so it must
        not be called while holding ``_state_lock`` (the reverse order
        would deadlock against a concurrent adoption).

        Refuses a filesystem root (``/``, ``C:\\`` — see
        :func:`kiss.core.utils.is_root_dir`): adopting one (from a
        persisted ``config.work_dir`` via ``saveConfig``, the one root
        source ``ServerApi.dispatch``'s top-level ``workDir``
        normalization cannot see) would root every task — and the
        ``@``-mention file scan — at the whole disk.

        Args:
            new_dir: The non-empty directory to adopt.
        """
        if not isinstance(new_dir, str) or not new_dir or is_root_dir(new_dir):
            logger.warning(
                "Refusing %r as the daemon work dir; keeping %r",
                new_dir, self.work_dir,
            )
            return
        from kiss.core.user_config import (
            load_config,
            record_recent_work_dir,
            save_config,
        )

        # One adoption at a time: two interleaved adoptions (a pick
        # racing a settings save) must end with the live value, the
        # disk and every client on the same directory — the last one
        # in.
        with _CommandsMixin._work_dir_lock:
            with self._state_lock:
                changed = self.work_dir != new_dir
                if changed:
                    self.work_dir = new_dir
                    self._file_index.ensure(new_dir)
                if hasattr(self.printer, "work_dir"):
                    setattr(self.printer, "work_dir", new_dir)
            # Every surface's chat-header work-dir drop-down lists the
            # directories selected so far (most recent first); this is the
            # one place every adopted directory passes through.
            record_recent_work_dir(new_dir)
            with _CommandsMixin._save_config_lock:
                if load_config().get("work_dir") != new_dir:
                    save_config({"work_dir": new_dir})
            if changed:
                self.printer.broadcast(
                    {"type": "workDirChanged", "workDir": new_dir},
                )

    def _refuse_run(self, tab_id: str, text: str) -> None:
        """Turn down a ``run`` on *tab_id* with an ``error`` broadcast.

        Both frontends raise the tab's running state optimistically
        the moment the user hits Enter and only a ``status
        running:false`` ever lowers it again, so the refusal MUST
        clear it first or the tab's composer stays disabled forever
        (F08-1).  The ``error`` carries ``code: "prompt_refused"`` so
        the page can put the refused draft (text and attachments)
        back into the composer instead of losing what the user typed.

        Args:
            tab_id: The tab whose submit is refused.
            text: The user-facing reason.
        """
        self.printer.broadcast(
            {"type": "status", "running": False, "tabId": tab_id},
        )
        self.printer.broadcast({
            "type": "error",
            "code": "prompt_refused",
            "text": text,
            "tabId": tab_id,
        })

    def _broadcast_run_notice(self, cmd: dict[str, Any], tab_id: str) -> None:
        """Deliver the submit-time notice a ``run`` carries in ``_notice``.

        The remote webapp's ``submit`` handler attaches a notice (e.g.
        "only the first 32 attachments were sent") to the run it
        builds instead of broadcasting it itself: a notice sent before
        the run's ``clear`` is wiped from the transcript by that very
        reset, so it goes out here, after the new task has cleared the
        tab (or, for a prompt steered into a running task, at once).

        Args:
            cmd: The ``run`` command, possibly carrying ``_notice``.
            tab_id: The tab the notice is addressed to.
        """
        text = cmd.get("_notice", "")
        if isinstance(text, str) and text:
            self.printer.broadcast(
                {"type": "notice", "text": text, "tabId": tab_id},
            )

    def _cmd_run(self, cmd: dict[str, Any]) -> None:
        """Start an agent task in a background thread.

        Initializes the tab's agent chat id (if empty) and broadcasts
        the initial ``clear`` event **synchronously**, before starting
        the worker thread.  Emitting ``clear`` here (rather than from
        inside the worker thread) makes the chat-id → tab-id mapping
        visible to the transport layer immediately after ``_cmd_run``
        returns, so a subsequent ``resumeSession`` for the same chat
        (e.g. a fast history click right after submit) can be routed
        to the correct task process without racing the worker
        thread's first broadcast.
        """
        tab_id = cmd.get("tabId", "")
        # Acknowledge + mirror the task-panel text to EVERY client
        # here, in the common run path, so all run origins (a
        # ``submit``, Python clients) behave identically.
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
        remember = True
        thread: threading.Thread | None = None
        state: AgentState | None = None
        chat_id = ""
        prompt = cmd.get("prompt", "")
        typed = isinstance(prompt, str) and bool(prompt.strip())
        # Nothing is decoded here: a running owner takes the files as
        # lazily decoded ``PendingAttachment``s (so queueing them under
        # the lock costs nothing), and a fresh run's worker decodes its
        # own from the command, after the run is registered.
        attachments = pending_attachments(cmd.get("attachments", []))
        with self._state_lock:
            prev = agent_state.find_by_tab(tab_id)
            # A plain message typed into a tab that only VIEWS a task
            # whose agent is blocked in ``ask_user_question`` (a
            # client-local sub-agent tab, or a stale viewer that sent
            # ``run`` instead of ``appendUserMessage``) is that
            # question's answer — not a new task.  A ``<task>`` message
            # still starts a new task in the tab, as before.
            asked: AgentState | None = None
            if (
                typed
                and (prev is None or prev.task_thread is None)
                and not contains_task_tags(prompt)
            ):
                asked = self._viewer_awaiting_answer(tab_id)
            if prev is not None and prev.merge_in_progress():
                # An in-flight merge/discard owns the tab's state (and
                # its worktree agent); replacing it would orphan the
                # operation.  Refuse the run instead.
                self._refuse_run(
                    tab_id,
                    "Cannot run a task while a merge is in progress."
                    " Wait for it to finish first.",
                )
                return
            if prev is not None and prev.task_thread is not None:
                # S3-05: queue the prompt whenever a task thread is
                # installed.  The worker sets ``is_task_active`` only
                # AFTER the thread starts, so gating on the flag (or on
                # thread death) silently dropped a second ``run``
                # submitted during the startup window in which the
                # thread was alive but the flag not yet raised.
                if typed:
                    queued = self._queue_prompt_on_owner(prev, prompt, tab_id, attachments)
                    if queued is None:
                        # Queued after the agent loop ended: no drain
                        # will consume it, so ``_run_task``'s cleanup
                        # re-submits it as the tab's next run, whose
                        # own prompt echo replaces the steering echo.
                        return
                    remember, inject_task = queued
                    inject_prompt = prompt
            elif asked is not None:
                queued = self._queue_prompt_on_owner(asked, prompt, tab_id, attachments)
                if queued is None:
                    return
                remember, inject_task = queued
                inject_prompt = prompt
            elif self._update_installing:
                # The self-update installer is (about to be) running
                # and will restart this daemon: a task started now
                # would be killed mid-flight.  Checked under the same
                # lock the idle poller arms the flag under, so a
                # submit can never slip between its "no active tasks"
                # observation and the installer spawn.  Already-running
                # tasks are untouched (steering above still works).
                self._refuse_run(
                    tab_id,
                    "An update is being installed; wait for the daemon"
                    " to restart.",
                )
                return
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
                # (the task panel's text) reads this attribute.
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
                    tab_id, inject_prompt, inject_task, remember,
                )
                self._broadcast_run_notice(cmd, tab_id)
            return
        # ``thread`` and ``state`` are created together above, so a
        # non-None thread guarantees the state.
        assert state is not None
        try:
            # A sub-agent dispatch (wire field ``parentTaskId``, set by
            # ``run_agent``'s daemon client on behalf of a calling
            # task) gets NO top-level registry tab: like a
            # ``run_parallel`` sub-task, its tab is client-local,
            # created on every client viewing the parent by the run's
            # own ``new_tab`` broadcast (see ``ChatSorcarAgent.run``)
            # and nested under the parent's tab.  Registering it here
            # would ALSO show it as an ordinary top-level tab.
            is_subagent_run = bool(str(cmd.get("parentTaskId", "") or ""))
            if not is_subagent_run:
                # Register + title + bind the tab in the shared registry
                # BEFORE the ``clear`` broadcast so every client has the
                # tab by the time the run's first event reaches it.  A new
                # run supersedes any historical task the tab was pinned to
                # (``taskId`` cleared: the tab tracks the chat's latest
                # task again — the one this run creates).
                publish_generation = self._registry_update_tab(
                    tab_id,
                    chat_id=chat_id,
                    title=str(cmd.get("prompt", "") or ""),
                    work_dir=str(cmd.get("workDir", "") or ""),
                    # A standalone ``sorcar.run`` sub-task (wire field
                    # ``tabScopeWorkDir``) may execute in a channel/cron
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
                # tab (``sorcar.run``): no
                # client had adopted the tab yet, so every client dropped
                # the task text and the tab's transcript opened WITHOUT
                # its task panel (and the tab had no title).  Re-echo the
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
                # A ``closeTab`` dispatched on another connection can
                # land between this run's state registration and the
                # ``_registry_update_tab`` above: the close removes
                # the canonical tab and defers state disposal (the
                # pre-start worker counts as busy), then the
                # publication resurrects the tab — which nothing ever
                # removes again, while ``_dispose_if_closed`` retires
                # the run's backend state at task end.  ``_close_tab``
                # sets ``frontend_closed`` BEFORE its registry
                # removal, so re-checking the flag AFTER the
                # publication is race-free: whenever the close's
                # removal misses the recreated tab, the flag is
                # already visible here and the recreate is undone —
                # every interleaving converges on the run→closeTab
                # serial outcome (tab closed; the task still runs to
                # completion, exactly as a close during a started run
                # behaves).  (gpt-5.6-sol review, finding 4.)
                #
                # The undo must only ever delete THIS run's own
                # publication: between the flag read below and the
                # removal, a legitimate reopen (``resumeSession``
                # clears the flag, then republishes the tab) can
                # recreate the row, and the stale unconditional
                # ``close_tab`` used here before deleted the reopen's
                # tab.  ``close_tab_if_generation`` compares the
                # captured publication token under the registry lock,
                # so a row republished by anyone else survives the
                # stale undo (gpt-5.6-sol review 2, introduced bug 1).
                with self._state_lock:
                    closed_while_publishing = state.frontend_closed
                if closed_while_publishing:
                    if self.tab_registry.close_tab_if_generation(
                        tab_id, publish_generation,
                    ):
                        self._broadcast_tabs_state()
            self.printer.broadcast({
                "type": "clear",
                "chat_id": chat_id,
                "tabId": tab_id,
            })
            self._broadcast_run_notice(cmd, tab_id)
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
        except BaseException as exc:
            with self._state_lock:
                never_started = state.task_thread is thread
                if never_started:
                    state.task_thread = None
                    state.stop_event = None
                    state.user_answer_queue = None
            if never_started:
                # Both frontends raised the tab's running state the
                # moment the user hit Enter, and only a terminal
                # ``result`` + ``status running:false`` ever lower it
                # — normally ``_run_task``'s own failure path, which
                # never ran here (``RuntimeError: can't start new
                # thread`` under thread exhaustion, or a failed
                # registry publication).  Emit them, or the tab stays
                # "running" until a daemon restart.
                self._report_run_start_failure(tab_id, cmd, exc)
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

    def _report_run_start_failure(
        self, tab_id: str, cmd: dict[str, Any], exc: BaseException,
    ) -> None:
        """Broadcast the terminal events of a run whose thread never started.

        Mirrors the ``result`` / ``status running:false`` pair that
        ``_run_task`` emits when setup fails, for the case where the
        worker thread itself could not be started, and retires the
        tab's state if the frontend closed it meanwhile.

        Args:
            tab_id: The tab whose run failed to start.
            cmd: The run command (its client run token is stamped on
                the ``status`` event like a normal run end).
            exc: The error raised before the worker thread ran.
        """
        logger.warning(
            "Task could not be started: tab_id=%s error=%s", tab_id, exc,
            exc_info=True,
        )
        self.printer.broadcast({
            **_result_event(f"Task failed: {type(exc).__name__}: {exc}", success=False),
            "tabId": tab_id,
        })
        status_end: dict[str, Any] = {
            "type": "status", "running": False, "tabId": tab_id,
        }
        client_task_id = _client_task_id_of(cmd)
        if client_task_id:
            status_end["taskId"] = client_task_id
        self.printer.broadcast(status_end)
        self._dispose_if_closed(tab_id)

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

    def _cmd_interrupt_tool(self, cmd: dict[str, Any]) -> None:
        """Interrupt the tool call running on a tab's task (per-panel Stop).

        The task keeps running; only the tool call in progress returns
        early with ``"User interrupted the tool call."``.  ``callId``
        (the clicked panel's ``tool_call`` event id) and ``toolName``
        name the call the user clicked, so a click that arrives after
        that call finished cannot hit the next one.
        """
        tool_name = cmd.get("toolName", "")
        if not isinstance(tool_name, str):
            tool_name = ""
        call_id = _parse_int(cmd.get("callId"))
        self._interrupt_tool_call(cmd.get("tabId", ""), tool_name, call_id)

    def _cmd_get_models(self, cmd: dict[str, Any]) -> None:
        """Send available models list to the requesting connection only."""
        self._get_models(cmd.get("connId", ""))

    def _cmd_select_model(self, cmd: dict[str, Any]) -> None:
        """Update the selected model for a tab.

        An empty ``tabId`` (malformed payload) updates only the
        daemon-wide default model (when a model was actually
        supplied).  Picking a model-routing SEA (``autorouter``) also
        fires its ``on_picked_as_model(work_dir)`` hook with the tab's
        work directory, without waiting for it.
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
            self._model_pick = (self._model_pick[0] + 1, model)
            self._model_picks_pending += 1
        # Persisted after the lock is released: the write waits for the
        # history database's writer turn (behind a long ``getHistory``,
        # say), and ``_state_lock`` is on the path of every other tab's
        # commands.  The pending count and the pick generation keep a
        # concurrent ``_refresh_default_model`` from reverting the pick
        # with the still-stale file.  Overlapping picks write the file
        # in no particular order: once its own write is done, a pick
        # that a later one has superseded writes that newest pick
        # again, so the last write of every pick names the newest and
        # the file cannot end up with an older pick than memory.
        try:
            _record_model_usage(model)
            with self._state_lock:
                newest = self._model_pick[1]
            if newest != model:
                _save_last_model(newest)
        finally:
            with self._state_lock:
                self._model_picks_pending -= 1
        # A real model name runs no hook; the registry look-up that tells
        # happens on the hook's thread, so a changed SEA's re-import never
        # stalls the command loop.
        run_picked_hook(model, self._tab_work_dir(tab_id), wait=False)

    def _tab_work_dir(self, tab_id: str) -> str:
        """Return the tab's pinned work directory, else the daemon's."""
        for entry in self.tab_registry.snapshot():
            if entry["tabId"] == tab_id:
                return entry["workDir"] or self.work_dir
        return self.work_dir

    def _cmd_get_history(self, cmd: dict[str, Any]) -> None:
        """Send conversation history to the requesting connection only."""
        query = cmd.get("query")
        if not isinstance(query, str):
            query = None
        offset = _parse_int(cmd.get("offset", 0))
        generation = _parse_int(cmd.get("generation", 0))
        tag = cmd.get("tag", "")
        machine = cmd.get("machine", "")
        work_dir = cmd.get("workDir", "")
        self._get_history(
            query,
            0 if offset is None else offset,
            0 if generation is None else generation,
            cmd.get("connId", ""),
            tag if isinstance(tag, str) else "",
            machine if isinstance(machine, str) else "",
            work_dir if isinstance(work_dir, str) else "",
        )

    def _cmd_set_favorite(self, cmd: dict[str, Any]) -> None:
        """Persist the favourite flag on a task history row."""
        task_id = _opt_str(cmd.get("taskId"))
        if task_id is None:
            return
        # Merges ``{"is_favorite": <bool>}`` into the row's ``extra``
        # JSON, preserving its other keys.  No broadcast: the
        # originating page updates its star optimistically and the
        # next ``getHistory`` refresh shows the persisted flag to
        # every other client.
        _set_task_favorite(task_id, bool(cmd.get("isFavorite", False)))

    def _cmd_set_chat_summary(self, cmd: dict[str, Any]) -> None:
        """Store the title the user typed over a chat's summary in the CHATS panel.

        An empty ``summary`` drops the user's title, so the computed
        summary shows again.  Every page is nudged to refresh its
        history so the other surfaces pick the new title up.
        """
        chat_id = _opt_str(cmd.get("chatId"))
        if not chat_id:
            return
        summary = cmd.get("summary")
        _set_chat_summary(chat_id, summary.strip() if isinstance(summary, str) else "")
        self.printer.broadcast({"type": "tasks_updated"})

    def _cmd_open_external(self, cmd: dict[str, Any]) -> None:
        """Open an ``http(s)`` link of the chat in the user's default browser.

        The standalone app window (``kiss-app``, a Chromium ``--app``
        window) has no browser UI, so the page sends the links it
        shows here instead of following them in the window.  The
        opener can wait a few seconds for a launcher that fails at
        once, so it runs off the command thread; the page gets an
        ``openExternalResult`` either way and falls back to its own
        ``window.open`` when nothing opened (a headless daemon).
        """
        url = _opt_str(cmd.get("url"))
        conn_id = str(cmd.get("connId", "") or "")
        if not url or not url.lower().startswith(("http://", "https://")):
            return

        def _open() -> None:
            opened = open_in_default_browser(url, once=False)
            self._broadcast_to_conn(
                {"type": "openExternalResult", "url": url, "opened": opened},
                conn_id,
            )

        threading.Thread(target=_open, name="open-external", daemon=True).start()

    def _refuse_delete_while_running(self, cmd: dict[str, Any], what: str) -> None:
        """Tell the requesting page that *what* cannot be deleted while its task runs."""
        self._broadcast_to_conn(
            {
                "type": "notice",
                "text": f"Cannot delete {what}: its task is still running. Stop it first.",
                "tabId": str(cmd.get("tabId", "") or ""),
            },
            cmd.get("connId", ""),
        )

    def _cmd_delete_task(self, cmd: dict[str, Any]) -> None:
        """Delete one task (and its sub-tasks) from history.db at the CHATS panel's request.

        A live task is never deleted — nor a finished task whose
        sub-task (deleted with it) still runs: the worker still appends
        events to the row, and the page shows a notice instead.  Every
        page is nudged to refresh its history once the rows are gone.
        """
        task_id = _opt_str(cmd.get("taskId"))
        if not task_id:
            return
        # The liveness check and the delete form one critical section
        # under ``_state_lock``: ``_cmd_run`` registers a new task's
        # state under the same lock, so a run that starts after the
        # check cannot lose its freshly inserted row.
        with self._state_lock:
            if _any_running(_task_tree(task_id)):
                self._refuse_delete_while_running(cmd, "the task")
                return
            _race_delay()  # test hook: widens the check-to-delete window
            _delete_task(task_id)
        self.printer.broadcast({"type": "tasks_updated"})

    def _cmd_delete_chat(self, cmd: dict[str, Any]) -> None:
        """Delete a chat — all its tasks, their events and its title — from history.db.

        Refused with a notice while any task the delete would remove
        (a task of the chat, or a sub-task of one) is running.
        """
        chat_id = _opt_str(cmd.get("chatId"))
        if not chat_id:
            return
        # Same critical section as ``_cmd_delete_task``.
        with self._state_lock:
            if _any_running(_chat_task_tree(chat_id), chat_id):
                self._refuse_delete_while_running(cmd, "the chat")
                return
            _race_delay()  # test hook: widens the check-to-delete window
            _delete_chat(chat_id)
        self.printer.broadcast({"type": "tasks_updated"})

    def _cmd_get_files(self, cmd: dict[str, Any]) -> None:
        """Send file list for autocomplete, scoped to the tab's work_dir.

        The chat page stamps the active tab's ``workDir`` on every
        ``getFiles`` command so the ``@``-mention picker lists files
        relative to *that* tab's working directory rather than the
        daemon-wide default (which is shared across every tab and
        otherwise reflects whichever directory the daemon was launched
        from or last switched to via ``setWorkDir``).

        The resulting ``files`` events are routed only to the
        requesting connection (via ``connId``) so typing ``@`` in one
        browser tab never pops the file picker in another, and are
        stamped with the requesting ``tabId`` so within that page they
        pop only in the chat tab that typed ``@`` — the
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

        Usage counts are stored by mention text (``./path`` relative to
        the tab's work dir, ``~/path`` below the home directory) in a
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

        Resolving the waiting state and delivering the answer (see
        :meth:`_deliver_user_answer`) form one critical section under
        ``_state_lock``.
        """
        ans_tab = cmd.get("tabId", "")
        with self._state_lock:
            owner = self._resolve_user_answer_state(ans_tab)
            q = owner.user_answer_queue if owner is not None else None
            if owner is None or q is None:
                logger.debug("userAnswer dropped: no queue for tabId=%s", ans_tab)
                return
            answer = cmd.get("answer", "")
            if not isinstance(answer, str):
                answer = "" if answer is None else str(answer)
            self._deliver_user_answer(owner, q, answer, ans_tab)

    def _deliver_user_answer(
        self,
        owner: AgentState,
        q: queue.Queue[str],
        answer: str,
        ans_tab: str,
    ) -> None:
        """Hand *answer* to the ``ask_user_question`` blocked on *q* and close it.

        The question is no longer pending the moment its answer is
        consumed: clearing ``pending_ask_question`` under the lock
        guarantees a concurrent session replay (``_emit_pending_ask``)
        can never re-show an already-answered modal.  The drain-then-put
        sequence under the same lock means two concurrent deliveries
        cannot both observe the ``maxsize=1`` queue as empty, both
        ``put`` and wedge the second handler thread forever; with the
        queue guaranteed empty by the drain, ``put_nowait`` never blocks.

        The ``askUserDone`` broadcast belongs to the SAME critical
        section: the ``put`` wakes the agent thread, which may return
        from the tool and publish its NEXT question at once —
        ``_ask_user_question`` takes this lock to do so — and a close
        sent after the lock is released could reach the clients after
        that second ``askUser`` and dismiss it (clients clear whatever
        question is showing; a close carries no question identity).
        Broadcasting before the lock is released orders every
        ``askUserDone`` ahead of the next ``askUser``.  ``STATE_LOCK`` →
        printer-lock is the established nesting order (the
        ``askUser`` broadcast itself is made under this lock).

        MUST be called while holding :data:`agent_state.STATE_LOCK`.

        Args:
            owner: The state whose agent thread is waiting on *q*.
            q: The owner's live ``user_answer_queue``.
            answer: The user's answer text.
            ans_tab: Frontend tab id the answer was typed into; the
                question closes there and on every tab subscribed to
                the answered task (not on every historic subscriber
                set containing *ans_tab*, which could dismiss an
                unrelated tab's current question).
        """
        owner.pending_ask_question = ""
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:  # pragma: no cover — race guard
                break
        try:
            q.put_nowait(answer)
        except queue.Full:  # pragma: no cover — drained immediately above
            pass
        self.printer.broadcast_transient({"type": "askUserDone"}, owner.task_id, ans_tab)

    def _route_prompt_to_owner(
        self,
        owner: AgentState,
        prompt: str,
        tab_id: str,
        attachments: list[Attachment] | None = None,
    ) -> bool:
        """Queue a mid-run user *prompt* on the right list of *owner*.

        A plain message is appended to ``pending_user_messages`` — the
        live agent's pre-step hook drains it into the model conversation
        as a steering instruction.  A message wrapped in
        ``<task>...</task>`` tags is instead split into its task blocks
        and appended to ``queued_followup_tasks``: the task runner's
        per-subtask loop drains that list once the CURRENT task finishes
        and runs each block one-by-one as further sequential subtasks, so
        a task list typed mid-run never steers the running task.

        Exception — a pending ``ask_user_question``: while the agent
        thread is blocked inside that tool (``owner.pending_ask_question``
        is set and the state holds a live ``user_answer_queue``) it never
        reaches the pre-step hook, so a steering message would sit
        undrained and the agent would hang until the task is stopped
        (history.db task ``e8a8407967d645c28c87750eda7a6cc0``: the user
        typed the reply into the chat box instead of the answer box).  A
        plain message typed then IS the answer and is delivered through
        :meth:`_deliver_user_answer`, exactly like a ``userAnswer``
        command.  A ``<task>`` message keeps its follow-up semantics even
        then — it is an explicit "run this afterwards", not a reply.

        Only server-owned states take the queued-tasks path: the drain
        lives in ``TaskRunner._run_task_inner``, which executes only
        UI-launched runs.  A sub-agent or standalone state has no such
        loop, so a ``<task>`` message sent to one would sit undrained
        forever — it falls back to live steering injection instead.  The
        same fallback applies once the run's ``followup_queue_closed``
        flag is up (the loop passed its final drain, a subtask failed, or
        the run is finalizing): a task queued then would never be
        drained.  The flag is raised under the same
        :data:`agent_state.STATE_LOCK` this helper runs under, so a
        message either lands in the queue before the final drain (and
        runs) or takes the steering path — never the
        accepted-then-dropped middle ground.  A steering message queued
        after the flag is not lost either: ``TaskRunner._run_task``'s
        cleanup re-submits the leftover ``pending_user_messages`` as the
        tab's next run (see :func:`_reruns_after_teardown`).

        MUST be called while holding :data:`agent_state.STATE_LOCK`.

        Files attached to the message (*attachments*) follow the text:
        a steering message's go to ``pending_user_attachments`` for the
        live agent, a ``<task>`` follow-up's to
        ``queued_followup_files`` (one entry per block, so each queued
        task runs with the message's files), and an answer to a
        pending question carries none (the composer never attaches
        files to an answer).

        Args:
            owner: The running-task state that accepted the prompt.
            prompt: The user's message (non-empty).
            tab_id: Frontend tab id the message was typed into.
            attachments: The message's decoded attachments, if any.

        Returns:
            ``True`` when the text was queued as a prompt (steering
            message or follow-up tasks) and so belongs in the composer's
            autocomplete history; ``False`` when it was consumed as the
            answer to a pending ``ask_user_question`` — an answer is not
            prompt history (the frontend's ``userAnswer`` path never
            records one either, and it may be a secret).
        """
        if (
            owner.server_owned
            and not owner.followup_queue_closed
            and contains_task_tags(prompt)
        ):
            blocks = parse_task_tags(prompt)
            owner.queued_followup_tasks.extend(blocks)
            owner.queued_followup_files.extend(list(attachments or []) for _ in blocks)
            return True
        if owner.pending_ask_question and owner.user_answer_queue is not None:
            self._deliver_user_answer(
                owner, owner.user_answer_queue, prompt, tab_id,
            )
            return False
        owner.pending_user_messages.append(prompt)
        owner.pending_user_attachments.extend(attachments or [])
        return True

    def _queue_prompt_on_owner(
        self,
        owner: AgentState,
        prompt: str,
        tab_id: str,
        attachments: list[Attachment] | None = None,
    ) -> tuple[bool, str] | None:
        """Route *prompt* to *owner* and resolve how to echo it.

        Wraps :meth:`_route_prompt_to_owner` with the echo bookkeeping
        every mid-run message needs: when the prompt will instead be
        re-run as the tab's next run (:func:`_reruns_after_teardown`),
        that run echoes it itself and the caller must emit no steering
        echo; when the owner has no task row yet
        (``_persisted_task_id`` is ``""``), the prompt is remembered on
        ``unattributed_prompt_echoes`` so the row can claim it later.

        MUST be called while holding :data:`agent_state.STATE_LOCK`:
        task teardown replaces/clears ``owner.agent`` under that lock,
        so reading the id inside the critical section that queued the
        prompt guarantees it belongs to the state the prompt was
        queued on, not to a successor task that re-armed the tab.
        ``""`` (no row yet — the narrow window between ``run()`` entry
        and ``_add_task``) makes the caller emit a transient echo
        rather than mis-attribute the prompt to a previous task.

        Args:
            owner: The running-task state that accepted the prompt.
            prompt: The user's message (non-empty).
            tab_id: Frontend tab id the message was typed into.
            attachments: The message's decoded attachments, if any.

        Returns:
            ``(remember, owner_task)`` — the *remember* flag of
            :meth:`_route_prompt_to_owner` and the task id to stamp the
            echo with (``""`` when unknown) — or ``None`` when no echo
            must be emitted.
        """
        remember = self._route_prompt_to_owner(owner, prompt, tab_id, attachments)
        if _reruns_after_teardown(owner):
            return None
        owner_task = _persisted_task_id(owner.agent)
        if not owner_task:
            owner.unattributed_prompt_echoes.append(prompt)
        return remember, owner_task

    def _viewer_awaiting_answer(self, tab_id: str) -> AgentState | None:
        """Return the task *tab_id* views whose agent is blocked in ``ask_user_question``.

        A daemon-dispatched sub-agent (``run_agent``) keeps its
        server-side ``api-…`` source tab on its state, while every
        client renders it in a client-local ``<parent>__sub_<task>``
        tab that is merely SUBSCRIBED to the task — so
        ``agent_state.find_by_tab`` never resolves that tab to the
        running task.  A ``run`` typed there while the sub-agent waits
        for an answer must reach that waiter (see
        :meth:`_route_prompt_to_owner`) rather than start an unrelated
        new task in the tab while the sub-agent hangs; the same holds
        for a stale history viewer of a running task.

        MUST be called while holding :data:`agent_state.STATE_LOCK`.

        Args:
            tab_id: The frontend tab id the message was typed into.

        Returns:
            The first viewed state with a pending question and a live
            answer queue, or ``None``.
        """
        for state in self._find_viewer_task_states(tab_id):
            if state.pending_ask_question and state.user_answer_queue is not None:
                return state
        return None

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
           tab (e.g. a second browser viewing a chat owned by another
           tab) renders the askUser question and submits the
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
        for state in self._find_viewer_task_states(ans_tab):
            if state.user_answer_queue is not None:
                return state
        return None

    def _echo_injected_prompt(
        self, tab_id: str, prompt: str, owner_task: str, remember: bool = True,
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

        This is also where the text is remembered for autocomplete
        (:func:`_record_steer_input`): every accepted mid-run message
        — ``appendUserMessage``, a ``run`` sent to a busy tab, or a
        ``/ask`` line — passes through here, and unlike a fresh submit
        it never gets a ``task_history`` row, so without this the
        composer's prefix completions and ArrowUp history would forget
        it on reload.  The caller passes ``remember=False`` when
        :meth:`_route_prompt_to_owner` consumed the text as the answer
        to a pending ``ask_user_question``: it is still echoed, but an
        answer is not prompt history.

        Args:
            tab_id: The frontend tab id the user typed into.
            prompt: The queued follow-up text.
            owner_task: The owning task id captured under
                ``_state_lock`` at queueing time (see
                :meth:`_queue_prompt_on_owner`), or ``""`` when the task row
                is not allocated yet.
            remember: Whether to save *prompt* for autocomplete.
        """
        if remember:
            try:
                _record_steer_input(prompt)
            except Exception:
                logger.warning("steer input not recorded", exc_info=True)
        # ``steer`` tells the chat page this is a message the user
        # typed into a RUNNING task (not the task's own prompt), which
        # it shows in a panel like the task's.
        echo: dict[str, Any] = {
            "type": "prompt",
            "text": prompt,
            "tabId": tab_id,
            "steer": True,
        }
        if owner_task:
            echo["taskId"] = owner_task
        self.printer.broadcast(echo)

    def _dispatch_ask_side_channel(
        self,
        *,
        tab_id: str,
        owner_task_id: str,
        owner_agent: Any,
        chat_id: str,
        question: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        """Fire a background ``ask_sea`` dispatch for a live ``/ask`` query.

        Spawns a daemon thread that calls
        :func:`daemon_client.run` with the resolved ``ask_sea`` script
        as ``sea_path``: the daemon accepts the run over
        its own local endpoint and runs it as a sub-agent of
        *owner_task_id*.  The frontend then renders the answering
        session as a nested sub-agent tab under the running task's
        tab — same page, no interaction with the outer agent's
        (possibly blocked) tool call.

        The nested tab is closed by the frontend the moment the
        answering session ends (``subagentDone``), and the child is
        dispatched as a *side channel* so replays of the parent (or of
        the child itself) re-issue that close instead of re-opening a
        finished tab — see :meth:`_announce_subagent_rows`.  The
        answer itself is therefore delivered separately: when
        :func:`daemon_client.run` returns (or
        raises), the worker broadcasts a persisted ``ask_answer``
        event into the OWNER task's transcript via
        :meth:`_broadcast_ask_answer`, and the running task's tab
        renders it as a distinct "Answer" panel that survives replays.
        The answering session's spend (including a timed-out, stopped
        session's) is charged to the owner task with
        :func:`~kiss.server.task_update.charge_side_channel_usage`.
        When the worker thread itself cannot be started (thread
        exhaustion) the same failed ``ask_answer`` panel is broadcast
        synchronously instead.

        The answering session is dispatched with the owner as its
        ``parent_task_id``: the daemon applies the ask SEA's
        ``prompt(task)`` (which names that id as the task the question
        is about) and ``system_prompt``.

        Args:
            tab_id: The frontend tab whose ``/ask`` produced this
                dispatch.  Threaded as ``parent_tab_id`` so the
                nested sub-agent tab renders inside it.
            owner_task_id: The persisted task id of the running task
                the user is asking about.  Empty means the running
                task has not allocated its row yet (rare — the
                narrow window between ``run()`` entry and
                ``_add_task``); the dispatch still fires but the
                answering agent will see an empty task id and report
                that no events exist.
            owner_agent: The running task's live agent, which the
                answering session's spend is charged to while the task
                still runs.
            chat_id: The chat the running task belongs to; passed so
                the answering sub-agent joins the same chat's
                history.
            question: The user's question, already stripped of the
                ``/ask`` prefix and surrounding whitespace.
            attachments: The files sent with the question, in the
                ``submit`` wire shape; the answering run gets them.
        """
        from kiss.agents.seas.ask import ask_sea
        from kiss.agents.sorcar import daemon_client
        from kiss.agents.sorcar.agent_dispatch import _daemon_endpoint_file
        from kiss.server.task_update import charge_side_channel_usage

        # Always the bundled script: ``seas/`` has the lowest registry
        # precedence, so a ``SEAS.md`` folder may shadow the ``/ask``
        # chat command, but the side channel is this answerer's.
        sea_path = Path(ask_sea.__file__)
        endpoint_file = _daemon_endpoint_file()
        wire_attachments = [a for a in attachments or [] if isinstance(a, dict)]
        epoch_getter = getattr(owner_agent, "_usage_epoch", None)
        epoch = epoch_getter() if callable(epoch_getter) else None

        def _run() -> None:
            spent: daemon_client.TaskResult | None = None
            try:
                result = daemon_client.run(
                    question,
                    sea_path=str(sea_path),
                    parent_task_id=owner_task_id,
                    parent_tab_id=tab_id,
                    side_channel=True,
                    chat_id=chat_id,
                    endpoint_file=endpoint_file,
                    timeout=_ASK_TIMEOUT_SECONDS,
                    stop_on_timeout=True,
                    attachments=wire_attachments,
                )
                spent = result
                text, success = result.text, result.success
            except Exception as exc:
                if isinstance(exc, daemon_client.StoppedOnTimeoutError):
                    spent = exc.result
                else:
                    spent = getattr(exc, "task_result", None)
                # A crashed side-channel MUST NOT bring down the
                # daemon: the interactive tab keeps running.  The
                # exception goes to the daemon log for triage, and
                # the user gets a failed answer panel instead of
                # waiting on a reply that will never come.
                logger.exception(
                    "/ask side-channel dispatch failed for tab %s", tab_id,
                )
                text, success = f"The /ask agent failed: {exc}", False
            if spent is not None:
                charge_side_channel_usage(
                    self.printer,
                    owner_agent,
                    owner_task_id,
                    spent.cost,
                    spent.tokens,
                    spent.steps,
                    epoch=epoch,
                )
            self._broadcast_ask_answer(
                tab_id=tab_id,
                owner_task_id=owner_task_id,
                question=question,
                text=text,
                success=success,
            )

        try:
            threading.Thread(
                target=_run, daemon=True, name="kiss-ask-sidechannel",
            ).start()
        except RuntimeError as exc:
            # Thread exhaustion (``can't start new thread``).  The
            # ``/ask`` line is already echoed into the transcript, so
            # the user gets a failed answer panel right under it
            # instead of the error escaping to the command reader and
            # the question hanging there unanswered forever.
            logger.exception(
                "/ask side-channel thread could not start for tab %s", tab_id,
            )
            self._broadcast_ask_answer(
                tab_id=tab_id,
                owner_task_id=owner_task_id,
                question=question,
                text=f"The /ask agent could not be started: {exc}",
                success=False,
            )

    def _broadcast_ask_answer(
        self,
        *,
        tab_id: str,
        owner_task_id: str,
        question: str,
        text: str,
        success: bool,
    ) -> None:
        """Deliver a finished ``/ask`` answer into the running task's transcript.

        Broadcasts an ``ask_answer`` event stamped with the asking
        *tab_id* and, when known, the OWNER task's id.  The stamp makes
        :meth:`WebPrinter.broadcast` treat it like the ``/ask`` prompt
        echo (see :meth:`_echo_injected_prompt`): it is rendered live
        in the tab, appended to the owner task's in-memory recording
        (so a viewer attaching to the still-running task replays it)
        and persisted into the task's ``events`` rows (so it survives
        a history reopen).  Without *owner_task_id* the event is a
        transient targeted broadcast — shown once, not replayed.

        Args:
            tab_id: The frontend tab the ``/ask`` was typed into.
            owner_task_id: The running task's persisted id, or ``""``
                when its row was not allocated at dispatch time.
            question: The user's question, ``/ask`` prefix stripped.
            text: The answering agent's final summary (HTML from
                ``finish(summary_in_html=...)``), or the failure text
                when the dispatch itself failed.
            success: Whether the answering agent reported success.
        """
        event: dict[str, Any] = {
            "type": "ask_answer",
            "question": question,
            "text": text,
            "success": success,
            "tabId": tab_id,
        }
        if owner_task_id:
            event["taskId"] = owner_task_id
        self.printer.broadcast(event)

    def _cmd_append_user_message(self, cmd: dict[str, Any]) -> None:
        """Queue a user message to be injected into the running agent's context.

        When the user types into the task-input textbox while a task is
        still running, the frontend forwards the prompt here instead of
        silently dropping it.  We append the text to the tab's
        :attr:`AgentState.pending_user_messages` list under
        :data:`agent_state.STATE_LOCK` so the live agent's
        pre-step hook can drain and inject the messages into the model
        conversation before the next model call.

        Exception (see :meth:`_route_prompt_to_owner`): a message
        wrapped in ``<task>...</task>`` tags is NOT injected into the
        running task.  Its task blocks are queued on
        :attr:`AgentState.queued_followup_tasks` instead, and the task
        runner executes them one-by-one as further sequential subtasks
        once the current task finishes.  And while the running agent
        is blocked inside ``ask_user_question``, a plain message is
        delivered as that question's answer (the agent cannot drain
        steering input until the tool returns) and every viewer tab
        receives ``askUserDone``, exactly as for a ``userAnswer``.

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
        # Lazily decoded (see ``PendingAttachment``): the owner is
        # chosen and the message queued in ONE locked step, so a task
        # finishing meanwhile cannot strand the follow-up.
        attachments = pending_attachments(cmd.get("attachments", []))
        # ``running()`` (not a raw ``is_alive()``) so a follow-up typed
        # while ``_cmd_run`` has installed but not yet started the
        # worker thread is queued instead of silently dropped.
        with self._state_lock:
            owner = agent_state.find_by_tab(tab_id)
            if owner is not None and not owner.running():
                owner = None
            if owner is None:
                for candidate in self._find_viewer_task_states(tab_id):
                    if candidate.running():
                        owner = candidate
                        break
            if owner is None:
                logger.debug(
                    "appendUserMessage dropped: tab %s has no "
                    "live task and is not a viewer of one",
                    tab_id,
                )
                return
            # ``/ask <question>`` is a live side-channel Q&A over the
            # running task's persisted events: routed to a background
            # ``ask_sea`` dispatch instead of ``pending_user_messages``
            # so the answering session runs independently of the outer
            # agent (which may be blocked inside a long tool call).
            # The captured id is the SAME task id the message was
            # queued against — the state lock is still held here, so
            # a successor task cannot have re-armed the tab yet.  The
            # SEA parser (``sea_commands._split_slash_command``) is
            # word-boundary strict at character zero (no leading
            # whitespace), and this helper matches it: passing
            # ``prompt`` without pre-stripping keeps ``  /ask q``
            # (leading spaces) OUT of the side channel so it flows
            # through the normal steering queue like any other typed
            # message.
            question = _split_ask_command(prompt)
            owner_task = _persisted_task_id(owner.agent)
            # An empty ``owner_task`` means the running task has not
            # allocated its ``task_history`` row yet (the narrow
            # window between ``run()`` entry and ``_add_task``): the
            # answering session would receive an empty task id and
            # be unable to read any events, so bail out of the side
            # channel and let the normal queue path handle the
            # ``/ask …`` line as a steering message instead — that
            # path already handles the pre-allocation window through
            # ``unattributed_prompt_echoes``.
            ask_question: str | None = None
            owner_chat_id = owner.chat_id
            remember = True
            if question is not None and owner_task:
                ask_question = question
            else:
                queued = self._queue_prompt_on_owner(owner, prompt, tab_id, attachments)
                if queued is None:
                    # The agent loop is over: nothing drains the queue
                    # any more and ``_run_task``'s cleanup re-submits
                    # the text as the tab's next run, which echoes it
                    # as its own prompt — a steering echo here would
                    # show the message twice.
                    return
                remember, owner_task = queued
        if ask_question is not None:
            # Echo the raw ``/ask …`` line the user typed, then hand
            # off to the side-channel worker.  The echo carries the
            # OWNER's task id (not a fresh one) so the message appears
            # in the running task's history stream, right where the
            # answer will land.
            self._echo_injected_prompt(tab_id, prompt, owner_task)
            if ask_question.lower() in RESERVED_SUBCOMMANDS:
                # ``/ask help`` is the SEA's ``description()`` and
                # ``/ask check`` its dry-run report, like every
                # ``/xxx help`` / ``/xxx check`` — answered here, no
                # dispatch.
                try:
                    help_text, help_ok = help_text_if_command(prompt), True
                except SeaError as exc:
                    help_text, help_ok = str(exc), False
                if help_text is not None:
                    self._broadcast_ask_answer(
                        tab_id=tab_id,
                        owner_task_id=owner_task,
                        question=ask_question,
                        text=help_text,
                        success=help_ok,
                    )
                    return
            raw_attachments = cmd.get("attachments")
            self._dispatch_ask_side_channel(
                tab_id=tab_id,
                owner_task_id=owner_task,
                owner_agent=owner.agent,
                chat_id=owner_chat_id,
                question=ask_question,
                attachments=raw_attachments if isinstance(raw_attachments, list) else None,
            )
            return
        self._echo_injected_prompt(tab_id, prompt, owner_task, remember)

    def _cmd_resume_session(self, cmd: dict[str, Any]) -> None:
        """Replay a previous chat session.

        When ``taskId`` is present, load that specific task instead of
        the latest task in the chat session.

        ``replayConnId`` is set only on the resumes a client's ``ready``
        fans out (``_handle_ready``): the transcript then goes to that
        connection alone.  ``scoped: true`` is the client's own way to
        ask for the same thing (a deferred tab it just activated, see
        ``replay_deferred``); the stamped ``connId`` is used then.  A
        user's history click carries neither and its transcript is
        broadcast, since every window mirroring the tab must show the
        newly bound chat.  ``deferred: true`` (also from
        ``_handle_ready``) sends no transcript at all: the connection
        gets the tab's state and a ``replay_deferred`` offer instead
        (:meth:`_defer_replay`).

        ``replayId`` is set by a daemon forwarding the resume for one
        of its clients (:mod:`kiss.server.remote_tabs`): every frame
        of the answer echoes it, and a ``replayDone`` to the asking
        connection ends the answer, so that daemon can tell the frames
        of this replay from the tab's other frames.
        """
        raw_id = cmd.get("chatId")
        chat_id = str(raw_id) if raw_id else ""
        task_id = _opt_str(cmd.get("taskId"))
        if not chat_id and task_id is None:
            return
        conn_id = str(cmd.get("replayConnId") or "")
        if not conn_id and cmd.get("scoped"):
            conn_id = str(cmd.get("connId") or "")
        if cmd.get("deferred") and conn_id:
            self._defer_replay(chat_id, cmd.get("tabId", ""), task_id, conn_id)
            return
        replay_id = str(cmd.get("replayId") or "")
        self._replay_session(
            chat_id,
            cmd.get("tabId", ""),
            task_id=task_id,
            conn_id=conn_id,
            replay_id=replay_id,
        )
        if replay_id:
            self._broadcast_to_conn(
                {"type": "replayDone", "replayId": replay_id}, str(cmd.get("connId") or ""),
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
            self._publish_tabs_state()
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

    # -- Browser tab: the daemon machine's browser streamed to every surface.
    # Browser page ids travel as ``tab_id`` (never ``tabId``, which names
    # chat tabs and is canonicalised by the API layer).

    def _cmd_browser_open(self, cmd: dict[str, Any]) -> None:
        """Open a page in the machine's browser and announce it on every surface."""
        self.browser_tabs.open(str(cmd.get("url") or ""), cmd.get("connId", ""))

    def _cmd_browser_close(self, cmd: dict[str, Any]) -> None:
        """Close a browser page; every surface drops its tab."""
        self.browser_tabs.close(str(cmd.get("tab_id") or ""))

    def _cmd_browser_navigate(self, cmd: dict[str, Any]) -> None:
        """Address-bar action on a browser page: ``go``, ``back``, ``forward``, ``reload``."""
        self.browser_tabs.navigate(
            str(cmd.get("tab_id") or ""),
            str(cmd.get("action") or "go"),
            str(cmd.get("url") or ""),
        )

    def _cmd_browser_input(self, cmd: dict[str, Any]) -> None:
        """Replay a client's mouse, wheel, key or pasted-text event on a browser page."""
        event = cmd.get("event")
        if isinstance(event, dict):
            self.browser_tabs.input(str(cmd.get("tab_id") or ""), event)

    def _cmd_browser_viewport(self, cmd: dict[str, Any]) -> None:
        """A client shows (or hid) a browser tab at the given CSS-pixel size."""
        self.browser_tabs.viewport(
            str(cmd.get("tab_id") or ""),
            cmd.get("connId", ""),
            max(0, _parse_int(cmd.get("width")) or 0),
            max(0, _parse_int(cmd.get("height")) or 0),
            bool(cmd.get("visible", True)),
        )

    def _cmd_browser_list(self, cmd: dict[str, Any]) -> None:
        """Tell the sender every live browser page (the ``browserTabs`` snapshot a ``ready`` gets).

        Sent by another daemon's :class:`~kiss.server.remote_views.RemoteViewStream`
        when it opens, so the pages open here before the stream are
        shown on that daemon's surfaces too.
        """
        self._broadcast_to_conn(self.browser_tabs.snapshot_event(), cmd.get("connId", ""))

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
          chat) — is the *same connection's* last-reported editor
          file, never another tab's.
        * Request staleness (``_complete_seq_latest``) is tracked per
          connection so a tab typing concurrently with another tab
          cannot mark the other tab's pending request stale.
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
            if active_file == "":
                # An explicit empty path means "no file open" (the
                # browser client closed its last file tab): forget the
                # snapshot rather than completing against a file the
                # user no longer has open.  An *absent* ``activeFile``
                # still keeps it (focus inside the chat).
                self._last_active_file.pop(conn_id, None)
                self._last_active_content.pop(conn_id, None)
                active_content = None
            elif active_file:
                if (
                    active_content is None
                    and active_file != self._last_active_file.get(conn_id, "")
                ):
                    # The client reported a DIFFERENT editor file with
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
            # dead-code the documented on-disk fallback: a client may
            # send ``activeFile`` without ``activeFileContent``.
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

    def _cmd_get_sea_commands(self, cmd: dict[str, Any]) -> None:
        """Send the slash-command list built from every SEA folder.

        Sent as a ``seaCommands`` event scoped to the requesting
        connection (see :func:`sea_commands_event` for its shape).
        The chat page uses it to render the autocomplete popup when
        the user types ``/`` in the composer, and asks again each
        time that popup opens so the recent / frequent ranking is
        current.
        """
        self._broadcast_to_conn(sea_commands_event(list_commands()), cmd.get("connId", ""))

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

    def _run_autocommit_job(self, tab_id: str, job: Callable[[], None]) -> None:
        """Run *job*, then release *tab_id*'s ``_autocommit_tabs`` claim.

        Thread body spawned by :meth:`_cmd_autocommit_action`; the
        ``finally`` makes a failed commit re-arm the tab.
        """
        try:
            job()
        finally:
            with self._state_lock:
                self._autocommit_tabs.discard(tab_id)

    def _cmd_autocommit_action(self, cmd: dict[str, Any]) -> None:
        """Stage-all + commit the tab's working tree in the background.

        Serves the settings panel's "Git Commit" button.  Delegates to
        :meth:`_commit_main_tree`, which stages everything, generates
        an LLM commit message from the staged diff alone, commits, and
        reports through toast ``notification`` events — the chat
        transcript stays clean except for a failure, whose reason is
        still rendered there via the non-silent ``autocommit_done``
        event.

        The command's ``workDir`` (the tab's own folder) is forwarded so
        the commit lands in the tab's repository rather than the
        daemon-wide ``self.work_dir``, which may point at a different —
        possibly non-git — folder.  A command stamped ``machine`` (the
        tab's drop-down names a registered machine) commits that
        machine's folder instead, through :meth:`_commit_on_machine`.

        At most one commit runs per tab: the commit-message generation
        is a billed LLM call and ``git add -A``/``git commit`` mutate
        the repository, so an impatient double click must not race two
        commits.  While the tab's commit is in flight an extra click is
        answered, on its own connection alone, with a failed
        ``autocommit_done`` saying so: a link waiting for its answer
        would otherwise wait out the whole request timeout, and a
        browser of this daemon whose click lost to a forwarded commit
        (whose outcome goes to its link alone) would never hear back.
        """
        tab_id = cmd.get("tabId", "")
        machine = str(cmd.get("machine") or "")
        work_dir = str(cmd.get("workDir") or "")
        # ``replyTo``: the command came over a daemon link (see
        # ``ServerApi.autocommit_action``); the outcome goes to that
        # connection as the reply to its request.
        reply_to = str(cmd.get("replyTo") or "")
        if machine:
            job = functools.partial(self._commit_on_machine, tab_id, machine, work_dir)
        else:
            job = functools.partial(
                self._commit_main_tree, tab_id, work_dir or self.work_dir, reply_to,
            )
        with self._state_lock:
            in_flight = tab_id in self._autocommit_tabs
            if not in_flight:
                self._autocommit_tabs.add(tab_id)
        if in_flight:
            logger.debug("Commit already in flight for tab %r; refusing the duplicate", tab_id)
            requester = reply_to or str(cmd.get("connId") or "")
            if requester:
                self._broadcast_autocommit_done(
                    tab_id, success=False, committed=False,
                    message="A Git Commit of this tab is already running.", reply_to=requester,
                )
            return
        # The claim is released when the job returns, or right here
        # when the thread cannot be started, so a failed spawn never
        # leaves the tab refusing every later click as a duplicate.
        try:
            threading.Thread(
                target=self._run_autocommit_job, args=(tab_id, job), daemon=True,
            ).start()
        except BaseException:
            with self._state_lock:
                self._autocommit_tabs.discard(tab_id)
            raise

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

    def _cmd_get_config(self, cmd: dict[str, Any]) -> None:
        """Send the current configuration to the frontend.

        The reported ``work_dir`` is the daemon's live global working
        directory — the one every task from every surface runs in
        (``task_runner`` falls back to it whenever a ``run`` carries no
        explicit ``workDir``).  It equals the persisted ``config.json``
        value once anything has been adopted, and the daemon's startup
        fallback (``--workdir``, ``$KISS_WORKDIR``, cwd) before that.
        """
        self._broadcast_to_conn(self._config_data_event(), cmd.get("connId", ""))

    def _config_data_event(self) -> dict[str, Any]:
        """Build the ``configData`` reply both ``getConfig`` and ``saveConfig`` send.

        One shape for both: the settings sheet repaints from either
        reply (``populateConfigForm`` writes every API-key field from
        ``apiKeys``, and the status bar takes ``machine``), so a save
        echo without them blanked the key fields (conc1018 H2-F3).
        """
        from kiss.core.user_config import (
            get_current_api_keys,
            load_config,
            recent_work_dirs,
        )

        cfg = load_config()
        # ``remote_password`` travels along: the Settings panel edits it
        # (``saveConfig`` stores a new one, in force at once; see
        # ``_cmd_save_config``).  The reported ``work_dir`` is the
        # daemon's live global directory, not the stored one: a save
        # that carries no ``work_dir`` must not report the startup
        # fallback as "".
        cfg["work_dir"] = self.work_dir
        # Only directories that still exist, most recently opened first
        # (the raw stored list may hold deleted or malformed entries,
        # and the chat header's drop-down shows whatever this carries).
        cfg["recent_work_dirs"] = recent_work_dirs()
        return {
            "type": "configData", "config": cfg, "apiKeys": get_current_api_keys(),
            # The server machine's short host name, shown centered in the
            # page's status bar so the user always sees which machine the
            # agent runs on.  The same name labels the local row of the
            # `machines` reply (the chat header's machine drop-down):
            # `platform.node()` would add the domain on cloud hosts.
            "machine": local_machine_name(),
        }

    def _cmd_save_config(self, cmd: dict[str, Any]) -> None:
        """Save configuration and API keys from the frontend.

        A ``remote_password`` in the payload replaces the stored one
        (``sanitize_config`` drops the retired hash key of an earlier
        release).  The daemon reads the stored password on every frame
        of a password-authenticated browser, so the new one is in force
        at once, without a restart; an empty one disables remote
        access.  When it changed, the requesting connection gets a
        ``remotePasswordChanged`` event carrying the new password
        BEFORE its ``configData``, so its transport adopts it before it
        sends another frame.  Every other password-authenticated client
        is shut out at its next command and asked for the new password;
        local-token and loopback clients are unaffected.

        The ``save_config`` write and the env re-apply are held under a
        dedicated lock so two concurrent ``saveConfig`` commands (two
        windows closing their settings panels together) cannot
        interleave ``apply_config_to_env`` with a half-merged config.
        """
        from kiss.core.user_config import (
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
        # ``work_dir`` is adopted (and persisted) by ``_apply_new_work_dir``
        # alone, below: saving it here first would put a directory it
        # refuses (a filesystem root) on disk, where it would shadow the
        # previously persisted directory on the next daemon start.
        new_work_dir = cfg.pop("work_dir", "")
        with _CommandsMixin._save_config_lock:
            prev_password = load_config().get("remote_password", "")
            save_config(cfg)
            apply_config_to_env(load_config())

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

        # Outside ``_save_config_lock``: ``_apply_new_work_dir`` takes
        # that (non-reentrant) lock itself to persist the directory.
        if new_work_dir:
            self._apply_new_work_dir(new_work_dir)

        conn_id = cmd.get("connId", "")
        self._get_models(conn_id)

        event = self._config_data_event()
        new_password = event["config"].get("remote_password", "")
        if new_password != prev_password:
            self._broadcast_to_conn(
                {"type": "remotePasswordChanged", "password": new_password},
                conn_id,
            )
        self._broadcast_to_conn(event, conn_id)

    def _cmd_get_my_models(self, cmd: dict[str, Any]) -> None:
        """Send the custom models from ``~/.kiss/MY_MODELS.json``.

        Answers the settings panel's Custom Models subpanel with a
        ``myModelsData`` event, stamped with the sender's ``connId`` so
        one window opening its settings panel never repaints another
        window's list mid-edit.
        """
        from kiss.core.models.model_info import list_custom_models

        self._broadcast_to_conn(
            {"type": "myModelsData", "models": list_custom_models()},
            cmd.get("connId", ""),
        )

    def _cmd_get_claude_code_status(self, cmd: dict[str, Any]) -> None:
        """Answer the settings panel's Claude Code subpanel.

        Replies with a ``claudeCodeStatus`` event (the dict of
        :func:`~kiss.core.models.claude_code_model.claude_auth_status`),
        stamped with the sender's ``connId``.
        """
        from kiss.core.models.claude_code_model import claude_auth_status

        self._reply_from_thread(
            "claudeCodeStatus", claude_auth_status, cmd.get("connId", ""), "claude-code-status",
        )

    def _cmd_get_codex_status(self, cmd: dict[str, Any]) -> None:
        """Answer the settings panel's Codex subpanel.

        Replies with a ``codexStatus`` event (the dict of
        :func:`~kiss.core.models.codex_model.codex_auth_status`), stamped
        with the sender's ``connId``.
        """
        from kiss.core.models.codex_model import codex_auth_status

        self._reply_from_thread(
            "codexStatus", codex_auth_status, cmd.get("connId", ""), "codex-status",
        )

    def _cmd_start_codex_login(self, cmd: dict[str, Any]) -> None:
        """Start a ChatGPT device-code login for the Codex subpanel.

        Runs :func:`~kiss.core.models.codex_model.codex_device_login` on
        a daemon thread and replies with a ``codexLogin`` event carrying
        the verification ``url`` and one-time ``code`` the user must
        enter in a browser (or ``error``).  The panel then polls
        ``getCodexStatus`` until the login shows up (a push would not
        survive the machine link, which answers one reply per request).

        One login at a time (:func:`codex_device_login`): while the CLI
        of an earlier request is still alive, waiting for its code to
        be entered, a request (another surface's click, a retry) is
        answered with that same code and starts no second CLI; a
        request carrying ``fresh: true`` ends that CLI and starts a
        new one.
        """
        from kiss.core.models.codex_model import codex_device_login

        self._reply_from_thread(
            "codexLogin", functools.partial(codex_device_login, fresh=bool(cmd.get("fresh"))),
            cmd.get("connId", ""), "codex-login",
        )

    def _reply_from_thread(
        self, reply_type: str, report: Callable[[], dict[str, Any]], conn_id: str, name: str,
    ) -> None:
        """Answer connection *conn_id* with a *reply_type* event built by *report*.

        For the settings sheet's CLI probes (``claude auth status``,
        ``codex login status``, ``codex login --device-auth``): each a
        bounded process start-up of seconds, run on a daemon thread
        *name* rather than in the command loop.  When the thread cannot
        start (``RuntimeError`` under thread pressure) the event carries
        the reason as ``error`` instead, so the sheet is never left
        waiting for an answer that is not coming.

        Args:
            reply_type: The event type of the reply.
            report: Builds the reply's fields (never raises).
            conn_id: The ``connId`` of the sender to answer.
            name: The thread's name.
        """
        try:
            threading.Thread(
                target=self._report_from_thread, args=(reply_type, report, conn_id),
                name=name, daemon=True,
            ).start()
        except RuntimeError as exc:
            logger.exception("%s thread could not start", name)
            self._broadcast_to_conn(
                {"type": reply_type, "error": f"{name} could not start: {exc}"}, conn_id,
            )

    def _report_from_thread(
        self, reply_type: str, report: Callable[[], dict[str, Any]], conn_id: str,
    ) -> None:
        """Thread body of :meth:`_reply_from_thread`."""
        self._broadcast_to_conn({"type": reply_type, **report()}, conn_id)

    def _broadcast_my_models(self) -> None:
        """Broadcast the current custom-model list to every client.

        Mutations are broadcast UNstamped (no ``connId``): the file is
        shared by every window, so all open settings panels must repaint
        after an add / edit / delete, not only the one that clicked.
        """
        from kiss.core.models.model_info import list_custom_models

        self.printer.broadcast({
            "type": "myModelsData", "models": list_custom_models(),
        })

    def _cmd_save_my_model(self, cmd: dict[str, Any]) -> None:
        """Add or update one custom model in ``~/.kiss/MY_MODELS.json``.

        Services the settings panel's Add and Save (edit) buttons.  The
        payload carries the model ``name`` plus optional ``endpoint`` /
        ``apiKey`` / ``headers`` strings and, for an edit that renamed
        the model, ``originalName``.  A rejected name (empty, or
        ``_``-prefixed — reserved for documentation keys) answers the
        sender with an ``error`` event; success rebroadcasts the list
        to every client.
        """
        from kiss.core.models.model_info import save_custom_model

        def field(key: str) -> str:
            value = cmd.get(key, "")
            return value if isinstance(value, str) else ""

        try:
            error = save_custom_model(
                name=field("name"),
                endpoint=field("endpoint"),
                api_key=field("apiKey"),
                headers=field("headers"),
                original_name=field("originalName"),
            )
        except OSError as e:
            # A failed write (permissions, disk full) must answer the
            # client instead of killing its connection's dispatch.
            logger.warning("saveMyModel failed", exc_info=True)
            error = f"Could not write ~/{HOME_DIR}/MY_MODELS.json: {e}"
        if error:
            self._send_error_to_sender(error, cmd)
            return
        self._broadcast_my_models()

    def _send_error_to_sender(self, error: str, cmd: dict[str, Any]) -> None:
        """Answer a failed ``~/.kiss`` file mutation with an ``error`` event.

        Stamped with the sender's ``connId`` when present so the banner
        pops only in the window that clicked.
        """
        self._broadcast_to_conn(
            {"type": "error", "text": error}, cmd.get("connId", ""),
        )

    def _cmd_delete_my_model(self, cmd: dict[str, Any]) -> None:
        """Delete one custom model from ``~/.kiss/MY_MODELS.json``.

        Services the settings panel's per-model Delete button, then
        rebroadcasts the list to every client.
        """
        from kiss.core.models.model_info import delete_custom_model

        name = cmd.get("name", "")
        error = None
        if isinstance(name, str) and name:
            try:
                error = delete_custom_model(name)
            except OSError as e:
                logger.warning("deleteMyModel failed", exc_info=True)
                error = f"Could not write ~/{HOME_DIR}/MY_MODELS.json: {e}"
        if error:
            self._send_error_to_sender(error, cmd)
            return
        self._broadcast_my_models()

    def _cmd_add_trick(self, cmd: dict[str, Any]) -> None:
        """Append a promptlet to ``~/.kiss/MY_INJECTION.md``.

        Services the Inject promptlet panel's Add button.  ``text`` is
        the promptlet body; a rejected body (empty, duplicate, or one
        that would start a new ``##`` section) or a failed write
        answers the sender with an ``error`` event.  Success
        rebroadcasts the full list as an UNstamped ``tricksData`` event
        — the file is shared by every window, so every open panel
        repaints, not only the one that clicked.
        """
        from kiss.server.tricks import append_my_injection_trick

        self._edit_my_injection(cmd, append_my_injection_trick, "addTrick")

    def _cmd_delete_trick(self, cmd: dict[str, Any]) -> None:
        """Remove a promptlet from ``~/.kiss/MY_INJECTION.md``.

        Services the delete button that the Inject promptlet panel shows
        on user-added rows.  ``text`` is the promptlet body as listed;
        a body that is not in the file (a bundled promptlet, or one
        another window already deleted) or a failed write answers the
        sender with an ``error`` event.  Success rebroadcasts the full
        list as an UNstamped ``tricksData`` event, like ``addTrick``.
        """
        from kiss.server.tricks import delete_my_injection_trick

        self._edit_my_injection(cmd, delete_my_injection_trick, "deleteTrick")

    def _cmd_edit_trick(self, cmd: dict[str, Any]) -> None:
        """Rewrite a promptlet of ``~/.kiss/MY_INJECTION.md`` in place.

        Services the edit (pencil) button that the Inject promptlet panel
        shows on user-added rows.  ``text`` is the promptlet body as
        listed and ``newText`` its replacement; the section keeps its
        position in the file.  A body that is not in the file, an
        empty or ``##``-starting replacement, a replacement that
        duplicates another promptlet, or a failed write answers the
        sender with an ``error`` event.  Success rebroadcasts the full
        list as an UNstamped ``tricksData`` event, like ``addTrick``.
        """
        from kiss.server.tricks import edit_my_injection_trick

        new_text = cmd.get("newText", "")
        self._edit_my_injection(
            cmd,
            functools.partial(
                edit_my_injection_trick,
                new_text=new_text if isinstance(new_text, str) else "",
            ),
            "editTrick",
        )

    def _edit_my_injection(
        self,
        cmd: dict[str, Any],
        edit: Callable[[str], str | None],
        name: str,
    ) -> None:
        """Run *edit* on ``cmd["text"]`` and answer the panel.

        Shared tail of ``addTrick`` / ``deleteTrick`` / ``editTrick``.
        Success broadcasts the fresh ``tricksData`` list (``tricks`` plus
        ``userCount``, the number of leading user-owned rows) to every
        window, since the file is shared by all of them.  A rejection or
        ``OSError`` goes to the sender as an ``error`` event followed by
        a ``tricksData`` stamped for the sender alone: the panel drops a
        deleted row (or shows an edited one) before the answer arrives,
        so the list on disk is re-sent to put the row back.
        """
        from kiss.server.tricks import read_tricks_data

        text = cmd.get("text", "")
        try:
            error = edit(text if isinstance(text, str) else "")
        except OSError as e:
            logger.warning("%s failed", name, exc_info=True)
            error = f"Could not write ~/{HOME_DIR}/MY_INJECTION.md: {e}"
        event: dict[str, Any] = {"type": "tricksData", **read_tricks_data()}
        if error:
            self._send_error_to_sender(error, cmd)
        # A failed write repaints only the sender; a successful one
        # repaints every window (the file is shared).
        self._broadcast_to_conn(event, cmd.get("connId", "") if error else "")

    def _cmd_set_work_dir(self, cmd: dict[str, Any]) -> None:
        """Make ``workDir`` the global working directory of every task.

        Sent when the user picks a folder in any surface's chat-header
        work-dir drop-down or ticks a top-level folder in the
        remote webapp's Explorer.

        Clears the calling connection's ``_last_active_file`` snapshot
        (it refers to a file from the previous workspace), invalidates
        the connection's in-flight autocomplete generation, and, when
        the directory actually changes, invalidates the autocomplete
        file cache and broadcasts ``workDirChanged`` (see
        :meth:`_apply_new_work_dir`).

        With ``reply: true`` the caller also gets the ``configData`` of
        a ``getConfig`` once the directory is adopted: another daemon
        proxying a pick made for this machine in its chat header
        (``setWorkDir`` with ``machine``) reads the adopted ``work_dir``
        and ``recent_work_dirs`` from it.
        """
        new_dir = cmd.get("workDir", "")
        if not isinstance(new_dir, str) or not new_dir:
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
        # Outside ``_state_lock``: the adoption takes ``_work_dir_lock``
        # first (see its ordering note).
        self._apply_new_work_dir(new_dir)
        if cmd.get("reply"):
            self._cmd_get_config(cmd)

    def _cmd_set_machine(self, cmd: dict[str, Any]) -> None:
        """Make ``machine`` the global machine new chats run on.

        Sent when the user picks a machine in any surface's chat-header
        drop-down: the pick is ONE global setting (a new chat on every
        surface runs there; a chat that already shows a task keeps that
        task's machine).  Persisted as ``config.json`` ``last_machine``
        (``configData`` carries it on load) and broadcast as
        ``machineChanged`` to every client, so the surfaces agree.
        ``""`` is this machine.  A pick that is not a string, or the
        machine already picked, changes nothing.  The broadcast goes
        out under the config lock, so two picks racing on the thread
        pool reach every client in the order they were persisted (the
        last broadcast names the persisted pick).
        """
        machine = cmd.get("machine", "")
        if not isinstance(machine, str):
            return
        from kiss.core.user_config import load_config, save_config

        with _CommandsMixin._save_config_lock:
            if load_config().get("last_machine", "") == machine:
                return
            save_config({"last_machine": machine})
            self.printer.broadcast({"type": "machineChanged", "machine": machine})

    _HANDLERS: dict[str, Any] = {
        "run": _cmd_run,
        "stop": _cmd_stop,
        "interruptTool": _cmd_interrupt_tool,
        "getModels": _cmd_get_models,
        "selectModel": _cmd_select_model,
        "getHistory": _cmd_get_history,
        "setFavorite": _cmd_set_favorite,
        "setChatSummary": _cmd_set_chat_summary,
        "deleteTask": _cmd_delete_task,
        "deleteChat": _cmd_delete_chat,
        "openExternal": _cmd_open_external,
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
        "getSeaCommands": _cmd_get_sea_commands,
        "getAdjacentTask": _cmd_get_adjacent_task,
        "autocommitAction": _cmd_autocommit_action,
        "worktreeAction": _cmd_worktree_action,
        "setWorkDir": _cmd_set_work_dir,
        "setMachine": _cmd_set_machine,
        "getConfig": _cmd_get_config,
        "saveConfig": _cmd_save_config,
        "getMyModels": _cmd_get_my_models,
        "getClaudeCodeStatus": _cmd_get_claude_code_status,
        "getCodexStatus": _cmd_get_codex_status,
        "startCodexLogin": _cmd_start_codex_login,
        "saveMyModel": _cmd_save_my_model,
        "deleteMyModel": _cmd_delete_my_model,
        "addTrick": _cmd_add_trick,
        "deleteTrick": _cmd_delete_trick,
        "editTrick": _cmd_edit_trick,
        "browserOpen": _cmd_browser_open,
        "browserClose": _cmd_browser_close,
        "browserNavigate": _cmd_browser_navigate,
        "browserInput": _cmd_browser_input,
        "browserViewport": _cmd_browser_viewport,
        "browserList": _cmd_browser_list,
    }
