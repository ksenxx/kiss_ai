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
import platform
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kiss.agents.sorcar.persistence import (
    _record_file_usage,
    _record_model_usage,
    _record_steer_input,
)
from kiss.agents.sorcar.sea_commands import (
    list_commands as list_sea_commands,
)
from kiss.core.utils import is_root_dir
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.merge_flow import _effective_commit_repo
from kiss.server.tab_registry import OpenTabOutcome
from kiss.server.task_runner import (
    _client_task_id_of,
    contains_task_tags,
    parse_task_tags,
)

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


# Prefix that flags an ``appendUserMessage`` as a live-side-channel
# /ask query.  Kept identical to the SEA registry stem so a
# ``list_commands`` reader and the interceptor agree on the name.
_ASK_COMMAND_PREFIX = "/ask"


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
        ``KISS_HOME`` points at a non-default location or because the
        platform has no managed kiss-web daemon (Windows: the VS Code
        extension installs no service there, so there is nothing to
        kick).
    """
    if not _kiss_home_is_default():
        logger.warning(
            "Skipping kiss-web daemon restart: KISS_HOME=%r is not the "
            "default ~/.kiss — the system daemon serves a different home",
            os.environ.get("KISS_HOME", ""),
        )
        return False
    if sys.platform not in ("darwin", "linux"):
        logger.info(
            "Skipping kiss-web daemon restart: no launchd/systemd-managed "
            "kiss-web daemon on %s",
            sys.platform,
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
            manual: bool = False, claimed_repo: Path | None = None,
            own_state: AgentState | None = None,
        ) -> None: ...
        def _any_non_wt_running(
            self,
            repo_root: Path | None = None,
            *,
            exclude: AgentState | None = None,
        ) -> bool: ...
        def _claim_main_tree(
            self, repo_root: Path, reason: str,
            holder: list[Any] | None = None,
        ) -> bool: ...
        def _release_main_tree_claim(self, claim: Any) -> None: ...
        def _broadcast_autocommit_done(
            self, tab_id: str, *, success: bool, committed: bool,
            message: str, commit_message: str | None = None,
            manual: bool = False, work_dir: str = "",
        ) -> dict[str, Any]: ...
        def _handle_worktree_action(
            self, action: str, tab_id: str = "", *,
            internal: bool = False, already_claimed: bool = False,
            resolve_conflicts: bool = False,
        ) -> dict[str, Any]: ...
        def _handle_main_tree_action(
            self, action: str, work_dir: str,
        ) -> dict[str, Any]: ...
        def _merge_deferred_worktrees(self, repo: Path | None) -> None: ...
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

        Refuses a filesystem root (``/``, ``C:\\`` — see
        :func:`kiss.core.utils.is_root_dir`): the fallback is what an
        unstamped command resolves to, so adopting a root here (from a
        persisted ``config.work_dir`` via ``saveConfig``, the one root
        source ``ServerApi.dispatch``'s top-level ``workDir``
        normalization cannot see) would root those commands — and the
        ``@``-mention file scan — at the whole disk.

        Args:
            new_dir: The non-empty directory to adopt.
        """
        if is_root_dir(new_dir):
            logger.warning(
                "Refusing filesystem root %r as the daemon work dir; "
                "keeping %r", new_dir, self.work_dir,
            )
            return
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
        remember = True
        thread: threading.Thread | None = None
        state: AgentState | None = None
        chat_id = ""
        prompt = cmd.get("prompt", "")
        typed = isinstance(prompt, str) and bool(prompt.strip())
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
                # S3-05: queue the prompt whenever a task thread is
                # installed.  The worker sets ``is_task_active`` only
                # AFTER the thread starts, so gating on the flag (or on
                # thread death) silently dropped a second ``run``
                # submitted during the startup window in which the
                # thread was alive but the flag not yet raised.
                if typed:
                    remember = self._route_prompt_to_owner(prev, prompt, tab_id)
                    inject_prompt = prompt
                    inject_task = _owner_task_id(prev)
                    if not inject_task:
                        prev.unattributed_prompt_echoes.append(prompt)
            elif asked is not None:
                remember = self._route_prompt_to_owner(asked, prompt, tab_id)
                inject_prompt = prompt
                inject_task = _owner_task_id(asked)
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
                    tab_id, inject_prompt, inject_task, remember,
                )
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
            "type": "result",
            "text": f"Task failed: {type(exc).__name__}: {exc}",
            "success": False,
            "total_tokens": 0,
            "cost": "$0.0000",
            "step_count": 0,
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
            ans_tab: Frontend tab id the answer was typed into (see
                :meth:`_user_answer_clear_tabs`).
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
        for tab_id in self._user_answer_clear_tabs(ans_tab, owner.task_id):
            self.printer.broadcast({"type": "askUserDone", "tabId": tab_id})

    def _route_prompt_to_owner(
        self, owner: AgentState, prompt: str, tab_id: str,
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
        (sorcar.db task ``e8a8407967d645c28c87750eda7a6cc0``: the user
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
        the run is finalizing): a task queued then would be echoed to the
        user and silently discarded by the end-of-run cleanup.  The flag
        is raised under the same :data:`agent_state.STATE_LOCK` this
        helper runs under, so a message either lands in the queue before
        the final drain (and runs) or takes the steering path — never the
        accepted-then-dropped middle ground.

        MUST be called while holding :data:`agent_state.STATE_LOCK`.

        Args:
            owner: The running-task state that accepted the prompt.
            prompt: The user's message (non-empty).
            tab_id: Frontend tab id the message was typed into.

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
            owner.queued_followup_tasks.extend(parse_task_tags(prompt))
            return True
        if owner.pending_ask_question and owner.user_answer_queue is not None:
            self._deliver_user_answer(
                owner, owner.user_answer_queue, prompt, tab_id,
            )
            return False
        owner.pending_user_messages.append(prompt)
        return True

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

    def _user_answer_clear_tabs(
        self, ans_tab: str, answered_task_id: str,
    ) -> list[str]:
        """Return every tab whose pending ask-user question should close.

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
           extension's tab) renders the askUser question and submits the
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
                :func:`_owner_task_id`), or ``""`` when the task row
                is not allocated yet.
            remember: Whether to save *prompt* for autocomplete.
        """
        if remember:
            try:
                _record_steer_input(prompt)
            except Exception:
                logger.warning("steer input not recorded", exc_info=True)
        echo: dict[str, Any] = {
            "type": "prompt",
            "text": prompt,
            "tabId": tab_id,
        }
        if owner_task:
            echo["taskId"] = owner_task
        self.printer.broadcast(echo)

    def _dispatch_ask_side_channel(
        self, *, tab_id: str, owner_task_id: str, chat_id: str, question: str,
    ) -> None:
        """Fire a background ``ask_sea`` dispatch for a live ``/ask`` query.

        Spawns a daemon thread that calls
        :func:`daemon_client.run` with the resolved ``ask_sea`` script
        as ``extension_agent_path``: the daemon accepts the run over
        its own Unix socket and runs it as a sub-agent of
        *owner_task_id*.  The frontend then renders the answering
        session as a nested sub-agent tab under the running task's
        tab — same webview, no interaction with the outer agent's
        (possibly blocked) tool call.

        The nested tab is closed by the frontend the moment the
        answering session ends, so the answer itself is delivered
        separately: when :func:`daemon_client.run` returns (or
        raises), the worker broadcasts a persisted ``ask_answer``
        event into the OWNER task's transcript via
        :meth:`_broadcast_ask_answer`, and the running task's tab
        renders it as a distinct "Answer" panel that survives replays.

        The ``<task_id>`` placeholder is substituted HERE so the
        answering session receives the OWNER's task id even when it
        would end up as the answering task's own parent id (which is
        the same value; kept explicit for clarity).

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
            chat_id: The chat the running task belongs to; passed so
                the answering sub-agent joins the same chat's
                history.
            question: The user's question, already stripped of the
                ``/ask`` prefix and surrounding whitespace.
        """
        from kiss.agents.sorcar import daemon_client, sea_commands
        from kiss.agents.sorcar.agent_dispatch import _daemon_sock_path
        from kiss.agents.third_party_agents import ask_sea

        sea_path = sea_commands.get_command("ask")
        if sea_path is None:
            logger.warning(
                "/ask received on tab %s but ask_sea is not registered",
                tab_id,
            )
            return
        append_to_prompt = (
            f"Read the events of the task {owner_task_id} from "
            f"~/.kiss/sorcar.db and answer the user question above."
        )
        append_to_system_prompt = ask_sea.append_to_system_prompt()
        sock_path = _daemon_sock_path()

        def _run() -> None:
            try:
                result = daemon_client.run(
                    question,
                    extension_agent_path=str(sea_path),
                    append_to_prompt=append_to_prompt,
                    append_to_system_prompt=append_to_system_prompt,
                    parent_task_id=owner_task_id,
                    parent_tab_id=tab_id,
                    chat_id=chat_id,
                    use_worktree=False,
                    auto_commit=False,
                    sock_path=sock_path,
                    timeout=600.0,
                    stop_on_timeout=True,
                )
                text, success = result.text, result.success
            except Exception as exc:
                # A crashed side-channel MUST NOT bring down the
                # daemon: the interactive tab keeps running.  The
                # exception goes to the daemon log for triage, and
                # the user gets a failed answer panel instead of
                # waiting on a reply that will never come.
                logger.exception(
                    "/ask side-channel dispatch failed for tab %s", tab_id,
                )
                text, success = f"The /ask agent failed: {exc}", False
            self._broadcast_ask_answer(
                tab_id=tab_id,
                owner_task_id=owner_task_id,
                question=question,
                text=text,
                success=success,
            )

        threading.Thread(
            target=_run, daemon=True, name="kiss-ask-sidechannel",
        ).start()

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
            owner_task = _owner_task_id(owner)
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
                remember = self._route_prompt_to_owner(owner, prompt, tab_id)
                if not owner_task:
                    owner.unattributed_prompt_echoes.append(prompt)
        if ask_question is not None:
            # Echo the raw ``/ask …`` line the user typed, then hand
            # off to the side-channel worker.  The echo carries the
            # OWNER's task id (not a fresh one) so the message appears
            # in the running task's history stream, right where the
            # answer will land.
            self._echo_injected_prompt(tab_id, prompt, owner_task)
            self._dispatch_ask_side_channel(
                tab_id=tab_id,
                owner_task_id=owner_task,
                chat_id=owner_chat_id,
                question=ask_question,
            )
            return
        self._echo_injected_prompt(tab_id, prompt, owner_task, remember)

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

    def _cmd_get_tabs_state(self, cmd: dict[str, Any]) -> None:
        """Broadcast the canonical ``tabs_state`` snapshot on request.

        Sent by the VS Code extension host's long-lived controller on
        every daemon (re)connect.  The daemon otherwise emits snapshots
        only after registry mutations and webview ``ready`` syncs, so a
        host with no open chat webview would have no baseline — the
        next remote-client mutation would be the FIRST snapshot it ever
        sees, and a reconnecting host would never learn of tabs created
        during the outage.  The reply is a normal broadcast: snapshots
        are idempotent and every client reconciles against the full
        list, so answering all clients is as cheap as answering one.
        """
        del cmd
        self._broadcast_tabs_state()

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

    def _cmd_get_sea_commands(self, cmd: dict[str, Any]) -> None:
        """Send the slash-command list built from every SEA folder.

        Sent as a ``seaCommands`` event scoped to the requesting
        connection: ``{type: "seaCommands", commands: [...]}``.  The
        chat webview uses the list to render the autocomplete popup
        when the user types ``/`` at the start of the composer.
        """
        commands = list_sea_commands()
        event: dict[str, Any] = {
            "type": "seaCommands",
            "commands": commands,
        }
        self._broadcast_to_conn(event, cmd.get("connId", ""))

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
        try:
            threading.Thread(
                target=self._run_commit_message_job,
                args=(tab_id, work_dir),
                daemon=True,
            ).start()
        except BaseException:
            # The worker never ran, so its ``finally`` cannot release
            # the claim published above — release it here or the tab
            # drops every later request as a duplicate.
            with self._state_lock:
                self._commit_msg_tabs.discard(tab_id)
            raise

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
        # Resolve the repository the worker will ACTUALLY stage — the
        # worker applies a stale-worktree fallback to a vanished
        # ``.kiss-worktrees/kiss_wt-*`` path, so discovering from the
        # raw path here claimed the wrong (or no) repository while the
        # parent repository was mutated unprotected (gpt-5.6-sol
        # review 2, missed wiring 1).  Should the path's target change
        # again between this dispatch and the worker's own resolution,
        # the worker re-runs the busy-check + claim on the repository
        # it resolved (see ``_autocommit_changes``).
        repo = _effective_commit_repo(work_dir)
        # Release-armed before publication (``_claim_main_tree``
        # appends to this list before it publishes) with the releasing
        # handler installed first, so no injected-stop boundary can
        # strand the claim (gpt-5.6-sol review 3, introduced bug 2).
        dispatch_claims: list[Any] = []
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
            # Publish the per-repo main-tree claim in the SAME locked
            # section as the busy check above: without it, a direct
            # non-worktree task could start (its admission saw no
            # worktree merge and no claim) between this check and the
            # worker's ``git add -A``, which would then stage the
            # task's half-written intermediate state (gpt-5.6-sol
            # review, finding 3).  Non-worktree task admission refuses
            # to start while the claim is held; the worker's
            # ``finally`` releases it.
            if repo is not None and not self._claim_main_tree(
                repo, "manual commit", holder=dispatch_claims,
            ):
                self._broadcast_autocommit_done(
                    tab_id, success=False, committed=False,
                    message="Another operation is modifying this "
                            "repository; wait for it to finish before "
                            "committing.",
                    manual=True, work_dir=work_dir,
                )
                return
            self._autocommit_tabs.add(tab_id)
        try:
            threading.Thread(
                target=self._run_autocommit_job,
                args=(tab_id, work_dir, repo, dispatch_claims),
                daemon=True,
            ).start()
        except BaseException:
            # The worker never ran, so its ``finally`` cannot release
            # the claims published above — release them here or the
            # repo (and the tab's commit button) stay wedged.
            with self._state_lock:
                self._autocommit_tabs.discard(tab_id)
                for claim in dispatch_claims:
                    self._release_main_tree_claim(claim)
            raise

    def _run_autocommit_job(
        self,
        tab_id: str,
        work_dir: str,
        repo: Path | None,
        dispatch_claims: list[Any] | None = None,
    ) -> None:
        """Commit the tab's working tree and re-arm the button.

        Body of the daemon thread spawned by
        :meth:`_cmd_autocommit_action`; the ``finally`` releases the
        tab's in-flight claim (so a failed commit never wedges the tab
        out of ever committing again) and the repository's main-tree
        claim (so tasks can start again).

        Args:
            tab_id: Frontend tab that requested the commit.
            work_dir: The tab's working directory.
            repo: The repository whose main-tree claim the dispatcher
                published (``None`` when *work_dir* is not in a repo).
            dispatch_claims: The claim objects the dispatcher
                published; released here conditionally (identity
                check), so a stale release can never pop a successor's
                claim.
        """
        try:
            self._autocommit_changes(
                tab_id, work_dir=work_dir, manual=True, claimed_repo=repo,
            )
        finally:
            with self._state_lock:
                self._autocommit_tabs.discard(tab_id)
                for claim in dispatch_claims or []:
                    self._release_main_tree_claim(claim)
        # The main tree is committed (and its claim released): merge
        # the worktrees whose merge waited for exactly this commit.
        self._merge_deferred_worktrees(repo)

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
        if action == "discard" and result.get("success"):
            # The task's uncommitted main-tree changes are gone, so the
            # tree is back at its committed state: merge the worktrees
            # whose merge waited for that.  ("Do nothing" leaves the
            # tree dirty; a later Git Commit triggers them instead.)
            self._merge_deferred_worktrees(_effective_commit_repo(work_dir))

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
            # The server machine's hostname, shown centered in the
            # webview's status bar (both the VS Code webview and the
            # remote webapp) so the user always sees which machine the
            # agent runs on.
            "machine": platform.node(),
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

    def _cmd_get_my_models(self, cmd: dict[str, Any]) -> None:
        """Send the custom models from ``~/.kiss/MY_MODELS.json``.

        Answers the settings panel's Custom Models subpanel with a
        ``myModelsData`` event, stamped with the sender's ``connId`` so
        one window opening its settings panel never repaints another
        window's list mid-edit.
        """
        from kiss.core.models.model_info import list_custom_models

        event: dict[str, Any] = {
            "type": "myModelsData", "models": list_custom_models(),
        }
        conn_id = cmd.get("connId", "")
        if conn_id:
            event["connId"] = conn_id
        self.printer.broadcast(event)

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
            error = f"Could not write ~/.kiss/MY_MODELS.json: {e}"
        if error:
            self._send_error_to_sender(error, cmd)
            return
        self._broadcast_my_models()

    def _send_error_to_sender(self, error: str, cmd: dict[str, Any]) -> None:
        """Answer a failed ``~/.kiss`` file mutation with an ``error`` event.

        Stamped with the sender's ``connId`` when present so the banner
        pops only in the window that clicked.
        """
        event: dict[str, Any] = {"type": "error", "text": error}
        conn_id = cmd.get("connId", "")
        if conn_id:
            event["connId"] = conn_id
        self.printer.broadcast(event)

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
                error = f"Could not write ~/.kiss/MY_MODELS.json: {e}"
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
        from kiss.server.tricks import append_my_injection_trick, read_tricks

        text = cmd.get("text", "")
        try:
            error = append_my_injection_trick(
                text if isinstance(text, str) else ""
            )
        except OSError as e:
            logger.warning("addTrick failed", exc_info=True)
            error = f"Could not write ~/.kiss/MY_INJECTION.md: {e}"
        if error:
            self._send_error_to_sender(error, cmd)
            return
        self.printer.broadcast({"type": "tricksData", "tricks": read_tricks()})

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
        "interruptTool": _cmd_interrupt_tool,
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
        "getTabsState": _cmd_get_tabs_state,
        "closeTab": _cmd_close_tab,
        "newChat": _cmd_new_chat,
        "complete": _cmd_complete,
        "getInputHistory": _cmd_get_input_history,
        "getSeaCommands": _cmd_get_sea_commands,
        "getAdjacentTask": _cmd_get_adjacent_task,
        "generateCommitMessage": _cmd_generate_commit_message,
        "autocommitAction": _cmd_autocommit_action,
        "worktreeAction": _cmd_worktree_action,
        "mainTreeAction": _cmd_main_tree_action,
        "setWorkDir": _cmd_set_work_dir,
        "getConfig": _cmd_get_config,
        "saveConfig": _cmd_save_config,
        "getMyModels": _cmd_get_my_models,
        "saveMyModel": _cmd_save_my_model,
        "deleteMyModel": _cmd_delete_my_model,
        "addTrick": _cmd_add_trick,
    }
