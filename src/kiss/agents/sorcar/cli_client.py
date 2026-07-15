# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Client-mode interactive CLI for ``sorcar``.

This module replaces the standalone in-process REPL
(:mod:`kiss.agents.sorcar.cli_repl`) with a thin WebSocket / UDS client
that drives an already-running ``sorcar web`` daemon — the same
:class:`~kiss.agents.vscode.web_server.RemoteAccessServer` that backs
the VS Code extension and the remote browser webapp.  The CLI input
panel, fast-completes (``@``-mentions, ``/`` slash commands, ``/model``
names, predictive ghost text), readline history, and Rich-rendered
output are preserved by re-using
:class:`kiss.agents.sorcar.cli_repl.CliCompleter` for the completer and
:class:`kiss.core.print_to_console.ConsolePrinter` for terminal
rendering of streamed events received from the server.

Transport: a local Unix-domain-socket connection to
``$KISS_SORCAR_SOCK`` (default ``$KISS_HOME/sorcar.sock``, falling
back to ``~/.kiss/sorcar.sock``).  The daemon's
:meth:`RemoteAccessServer._uds_handler` skips password authentication
because POSIX mode 0o600 on the socket file gates access to the owning
user.  The CLI client therefore needs no password.

Protocol: newline-delimited JSON commands and events identical to what
the VS Code extension speaks (see
:meth:`RemoteAccessServer._dispatch_client_command`).  The CLI client
announces itself with ``setWorkDir`` then ``ready`` (the daemon
fans the latter out into ``getModels`` / ``getInputHistory`` /
``getConfig`` replies), routes slash commands to the matching server
command (``getModels`` / ``selectModel`` / ``newChat`` /
``resumeSession`` / ``autocommitAction`` / ``cliInfo``), and submits
task prompts as ``run`` commands.  Incoming events (``text_delta``,
``tool_call``, ``tool_result``, ``result``, ``askUser``,
``commitMessage``, ``status``, …) are rendered to the terminal by
:meth:`_render_event`.

Server-side commands required by this client beyond the existing
webview surface are listed in
:meth:`kiss.agents.vscode.commands._CommandsMixin._cmd_cli_info` —
``cliInfo`` with a ``subtype`` selector handles ``/help``,
``/commands``, ``/skills``, ``/mcp``, ``/cost`` and custom-command
expansion.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import queue
import socket
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import cli_talk, cli_voice
from kiss.agents.sorcar.cli_daemon_bridge import _sock_path
from kiss.agents.sorcar.cli_helpers import (
    _parse_resume_arg,
    _print_recent_chats,
)
from kiss.agents.sorcar.cli_panel import (
    ASK_TITLE,
    QUESTION_FMT,
    QUEUED_FMT,
    QUEUED_STATUS_FMT,
    RESET,
    STEER_TITLE,
    YELLOW,
    _term_size,
)
from kiss.agents.sorcar.cli_repl import (
    _EXIT_WORDS,
    _PROMPT,
    CliCompleter,
    _history_path,
    _load_history_lines,
    _make_ptk_reader,
    _print_help,
    _print_model_list,
    _print_welcome,
    _read_line,
    _save_history,
    _save_history_lines,
    _setup_readline,
)
from kiss.agents.sorcar.cli_steering import (
    _MIN_ROWS,
    AnchoredRepl,
    supports_steering,
)
from kiss.agents.sorcar.persistence import _load_chat_events_by_task_id
from kiss.core.print_to_console import ConsolePrinter

logger = logging.getLogger(__name__)


def _clear_terminal() -> None:
    """Clear the terminal screen and scrollback when stdout is a TTY.

    Emits the ANSI sequences ``ESC[H`` (cursor home), ``ESC[2J``
    (erase visible screen) and ``ESC[3J`` (erase scrollback) so the
    interactive CLI starts on a clean canvas — matching the behaviour
    of ``clear`` / Cmd-K in modern terminals.  Skipped when stdout is
    not a TTY (pytest capture, piped output) so test runs and log
    redirection are not polluted with escape codes.
    """
    try:
        if not sys.stdout.isatty():
            return
    except Exception:  # pragma: no cover - defensive isatty guard
        return
    try:
        sys.stdout.write("\033[H\033[2J\033[3J")
        sys.stdout.flush()
    except Exception:  # pragma: no cover - defensive write guard
        return


def _wait_for_socket(path: Path, timeout: float = 5.0) -> bool:
    """Block up to *timeout* seconds for *path* to be a connectable UDS.

    The CLI client requires an already-running ``sorcar web`` daemon
    (decision 1.b from the project plan); polling the socket lets us
    accommodate a daemon that is still in the middle of startup —
    e.g. during the e2e tests which spin a fresh
    :class:`RemoteAccessServer` and immediately start the client.

    Args:
        path: The UDS path to probe.
        timeout: How long to wait, in seconds.

    Returns:
        ``True`` on success, ``False`` when the deadline expires.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect(str(path))
                s.close()
                return True
            except OSError:
                pass
        time.sleep(0.1)
    return False


class _EventDispatcher:
    """Render incoming server events to the terminal.

    Routes each newline-delimited JSON event from the daemon to one of:

    * Specific synchronous waiters (:class:`queue.Queue` instances) for
      request/reply events the slash-command dispatcher must block on
      (``cliInfo``, ``models``, ``commitMessage``).
    * :class:`ConsolePrinter` for streamed display events.
    * A side-channel callback for ``askUser`` so the main REPL can
      prompt the user and reply with ``userAnswer``.
    * A small set of housekeeping events tracked on the client
      (``clear`` records the chat id, ``status`` toggles the
      "task running" flag, ``configData`` captures the daemon's
      reported model).

    All terminal output goes through the single
    :class:`ConsolePrinter` instance so the Rich panels and streamed
    token output match what the standalone REPL used to render via
    :class:`kiss.agents.sorcar.cli_printer.RecordingConsolePrinter`.
    """

    def __init__(
        self, printer: ConsolePrinter, tab_id: str = "",
    ) -> None:
        self.printer = printer
        # The CLI client's own tab id.  Every task event the daemon
        # fans out is stamped with the *recipient* tab's id (the
        # subscribed tab id in ``WebPrinter.broadcast``) and then
        # broadcast verbatim to ALL connected clients — both WSS
        # clients and UDS clients in lockstep.  The chat webview
        # filters incoming events client-side by ``tabId`` so one
        # window only renders panels for its own tab; pre-fix the
        # CLI did NOT filter and therefore rendered every other
        # client's task panels (VS Code extension webview, remote
        # browser tab, parallel sorcar CLI instance) on the local
        # terminal.  Set the recipient tab id here so ``dispatch``
        # can drop foreign-tab events before they reach the printer
        # or mutate per-task state (``task_active``, ``chat_id``,
        # waiting queues).  Empty string preserves backwards-compat
        # behaviour (no filtering) so tests / callers that construct
        # a dispatcher without a tab id keep working.
        self.tab_id = tab_id
        # Synchronous waiters used by ``CliClient`` slash commands.
        self.cli_info_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self.models_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self.commit_q: queue.Queue[dict[str, Any]] = queue.Queue()
        # Latest in-flight task / chat metadata mirrored from server.
        self.chat_id: str = ""
        self.current_model: str = ""
        self.task_active = threading.Event()
        # Latched (never cleared by the loop thread) companion to
        # ``task_active``: set as soon as ANY ``status`` event for the
        # armed task is observed.  ``task_active`` is level-triggered —
        # a task that starts AND finishes between two of the
        # submitter's 50 ms polls flips it set→clear invisibly, which
        # used to wedge :func:`_start_task` until its full
        # ``timeout_seconds`` (an hour by default).  Submitters clear
        # this latch before sending ``run`` and wait on it instead.
        self.task_started = threading.Event()
        # Per-submission task id used to filter ``status`` events.
        # Stale ``status:false`` events from a prior task (or from a
        # peer subscriber on the same tab) carry a different
        # ``taskId`` and are ignored here — review #3 / #4.
        self.current_task_id: str = ""
        # Guards the ``current_task_id`` handoff between the REPL
        # thread (which arms/resets it around each submission) and
        # the loop thread (which reads it in the ``status`` filter).
        # CPython's GIL makes the bare str swap atomic today, but the
        # invariant was fragile and undocumented without an explicit
        # lock (w2 F17).
        self.task_id_lock = threading.Lock()
        # Queue of pending askUser questions forwarded from the loop
        # thread to the REPL thread.  Bouncing the prompt off the loop
        # thread is mandatory: ``input()`` on the asyncio loop thread
        # would block every other event (streamed tokens, ``status``,
        # ``result``) for the duration of the user's typing.
        self.ask_user_q: queue.Queue[str] = queue.Queue()

    def dispatch(self, event: dict[str, Any]) -> None:
        """Route one event to the appropriate handler."""
        et = event.get("type", "")
        # Drop events that target a different client's tab BEFORE any
        # rendering or per-task state mutation happens.  The daemon
        # fans out task events to every connected client (WSS +
        # UDS) — see :meth:`WebPrinter.broadcast` — and stamps each
        # copy with the *recipient* tab id.  Without this filter the
        # CLI silently rendered another window's ``text_delta`` /
        # ``tool_call`` / ``tool_result`` / ``result`` panels as soon
        # as that other client started a task, and corrupted its own
        # cached ``chat_id`` / ``task_active`` / waiter queues from
        # broadcasts targeted at other tabs.  An empty ``self.tab_id``
        # (back-compat construction without a tab id) disables the
        # filter so existing callers keep working.  Events with no
        # ``tabId`` are global (``configData`` / ``models`` from
        # ``ready`` fanout, server-reset notifications, etc.) and
        # always pass through.
        ev_tab = event.get("tabId", "")
        if self.tab_id and ev_tab and ev_tab != self.tab_id:
            return
        # Capture ``taskId`` BEFORE we pop it so status filtering can
        # match against the dispatcher's currently armed task id
        # (review #3).  Routing metadata is then stripped so consumers
        # downstream do not see server-internal fields.
        task_id = event.get("taskId", "")
        event.pop("taskId", None)
        if et == "cliInfo":
            self.cli_info_q.put(event)
            return
        if et == "models":
            # The daemon stamps every ``models`` reply with
            # ``selected`` — its canonical current model (see
            # ``server._get_models``).  Mirror it into
            # ``current_model`` so ``/model list`` (and any other
            # ``getModels`` round-trip) refreshes the client's view of
            # the model that will be sent with the next ``run``; the
            # old code dropped the field and the comment in
            # ``/model list`` claimed a refresh that never happened.
            selected = event.get("selected")
            if isinstance(selected, str) and selected:
                self.current_model = selected
            self.models_q.put(event)
            return
        if et == "commitMessage":
            self.commit_q.put(event)
            return
        if et == "clear":
            self.chat_id = event.get("chat_id", "") or self.chat_id
            return
        if et == "configData":
            cfg = event.get("config") or {}
            if isinstance(cfg, dict):
                self.current_model = cfg.get("model", "") or self.current_model
            return
        if et == "status":
            # Filter on ``taskId``: stale ``status:false`` events from
            # a prior task that finished after the new task was sent
            # would otherwise clear ``task_active`` immediately and
            # silently terminate the wait (review #3).  When the
            # dispatcher is not armed (``current_task_id`` empty) we
            # accept every status — preserves the existing behaviour
            # for non-task-driven status broadcasts.
            with self.task_id_lock:
                current = self.current_task_id
            if current and task_id != current:
                # Armed: only status events tagged with the EXACT
                # armed task id may touch ``task_active`` or the
                # ``task_started`` latch.  The daemon symmetrically
                # echoes the CLI-supplied ``taskId`` on both the
                # ``status:true`` and ``status:false`` broadcasts of a
                # CLI-run task (task_runner.py ``status_start`` /
                # ``status_end``), so an exact match is always observed
                # for our own submission.  UNTAGGED status events do
                # exist (webview-launched tasks on the same chat
                # ending, viewer-fanout ``status`` broadcasts) and must
                # be dropped while armed: a stray untagged
                # ``running:false`` landing between the submitter's
                # ``task_started.clear()`` and the daemon's first
                # echoed status would otherwise set the latch with
                # ``task_active`` clear — ``_start_task`` would report
                # the task acknowledged, ``_submit_task`` would skip
                # its wait loop, and the freshly submitted task would
                # keep running in the daemon silently orphaned (w3 F1).
                return
            running = bool(event.get("running", False))
            if running:
                self.task_active.set()
            else:
                self.task_active.clear()
            if current:
                # Latch that a status for the (possibly already
                # finished) armed task was observed, AFTER the level
                # toggle above so a waiter woken by the latch sees the
                # final level.  This closes the race where a
                # fast-finishing task's ``status:true`` →
                # ``status:false`` pair lands entirely inside one of
                # the submitter's 50 ms poll gaps.  Only exact-match
                # events reach this point while armed (see above), so
                # a stray untagged status can never satisfy the latch.
                self.task_started.set()
            return
        if et == "askUser":
            # Forward the question to the REPL thread; the loop
            # thread MUST NOT call ``input()`` itself because that
            # would block every other inbound event for the duration
            # of the user's typing.
            self.ask_user_q.put(str(event.get("question", "")))
            return
        if et == "error":
            # ``event.get("text", "")`` only falls back when the key
            # is absent — ``text: null`` still yields ``None`` and the
            # terminal printed the literal "None" (the ``_render``
            # branches were hardened by review #36; this branch was
            # missed).  Coerce explicitly.
            self.printer.print(
                f"[red]✗ {event.get('text') or ''}[/red]", type="text",
            )
            return
        # Streamed display events: route to ConsolePrinter.
        self._render(event)

    def _render(self, event: dict[str, Any]) -> None:
        # ``event.get(key, default)`` only falls back when the key is
        # absent — ``text: null`` still returns ``None``.  Defensively
        # coerce ``None`` to the empty string / empty dict so a daemon
        # version drift cannot crash the loop thread (review #36).
        et = event.get("type", "")
        if et in ("text_delta", "thinking_delta"):
            self.printer.token_callback(event.get("text") or "")
            return
        if et in ("thinking_start", "thinking_end"):
            self.printer.thinking_callback(et == "thinking_start")
            return
        if et == "text_end":
            # Force a newline so the next panel starts on its own row.
            self.printer.flush_newline()
            return
        if et in ("prompt", "system_prompt") and event.get("early"):
            # Optimistic submit-time panels meant for the chat WEBVIEW
            # (see ``task_runner._broadcast_early_prompts``): the
            # authoritative events follow once the agent starts, so
            # printing the early copies here would show every prompt
            # twice in the CLI.
            return
        if et == "prompt":
            self.printer.print(event.get("text") or "", type="prompt")
            return
        if et == "system_prompt":
            self.printer.print(event.get("text") or "", type="system_prompt")
            return
        if et == "tool_call":
            # The daemon (``JsonPrinter._format_tool_call``) broadcasts
            # a *flat* event whose argument fields sit at the top level
            # (``path`` / ``lang`` / ``command`` / ``content`` /
            # ``description`` / ``old_string`` / ``new_string`` /
            # ``extras``).  It never emits an ``input`` key, so a naive
            # ``event.get("input")`` lookup yields ``None`` and the
            # console panel rendered ``(no arguments)`` for every tool
            # call.  Rebuild the ``tool_input`` dict the shared
            # :class:`ConsolePrinter` formatter expects.
            tool_input: dict[str, Any] = {}
            if path := event.get("path"):
                # Use ``file_path`` so :func:`extract_path_and_lang`
                # picks it up exactly like the in-process printer.
                # The daemon's ``lang`` field is intentionally dropped —
                # ``ConsolePrinter._format_tool_call`` recomputes it from
                # the same path via ``lang_for_path``, so forwarding it
                # would be redundant.
                tool_input["file_path"] = str(path)
            for key in (
                "description", "command", "content",
                "old_string", "new_string",
            ):
                if (val := event.get(key)) is not None:
                    tool_input[key] = str(val)
            extras = event.get("extras") or {}
            if isinstance(extras, dict):
                # ``extras`` keys are by definition not in ``KNOWN_KEYS``
                # so merging them in straight is safe — they will be
                # picked up by :func:`extract_extras` for display.
                # EXCEPT the synthesized ``talk`` clip persisted for
                # demo replays (see ``attach_talk_audio``): the base64
                # blob is audio data, not a tool argument — printing
                # it would flood the terminal panel.
                for k, v in extras.items():
                    if k in ("audioB64", "audioMime"):
                        continue
                    tool_input[str(k)] = v
            self.printer.print(
                event.get("name") or "",
                type="tool_call",
                tool_input=tool_input,
            )
            return
        if et == "tool_result":
            # Reconstruct the minimal ``tool_input`` slice the
            # ConsolePrinter needs to syntax-highlight the body of a
            # ``Read`` result (the only tool whose return value is a
            # raw source-file body).  When the daemon-side broadcast
            # carries a ``path``/``start_line`` we forward it so the
            # local Rich ``Syntax`` widget picks the right lexer and
            # gutter offset; otherwise we omit ``tool_input`` and
            # fall back to the plain-write panel.
            result_tool_input: dict[str, Any] | None = None
            path = event.get("path")
            if path:
                result_tool_input = {"file_path": str(path)}
                start_line = event.get("start_line")
                if isinstance(start_line, int) and start_line >= 1:
                    result_tool_input["start_line"] = start_line
            self.printer.print(
                event.get("content") or "",
                type="tool_result",
                is_error=bool(event.get("is_error", False)),
                tool_name=event.get("tool_name") or "",
                tool_input=result_tool_input,
            )
            return
        if et == "system_output":
            self.printer.print(event.get("text") or "", type="bash_stream")
            return
        if et == "usage_info":
            self.printer.print(
                event.get("text") or "",
                type="usage_info",
                total_tokens=event.get("total_tokens", 0),
                cost=event.get("cost", "N/A"),
                total_steps=event.get("total_steps", 0),
            )
            return
        if et == "result":
            self.printer.print(
                event.get("text") or "(no result)",
                type="result",
                total_tokens=event.get("total_tokens", 0),
                cost=event.get("cost", "N/A"),
                step_count=event.get("step_count", 0),
            )
            return
        if et == "notification":
            # Webview-style toast notifications (auto-commit
            # life-cycle, server-reset, etc.).  Pre-fix these were
            # dropped into the "frontend-only" silent-ignore branch
            # below, so a sorcar CLI user saw none of the toasts a
            # chat webview user saw.  Route to the printer so the
            # operator at the terminal sees the same information.
            self.printer.print(
                event.get("message") or "",
                type="notification",
                severity=event.get("severity") or "info",
                progress_message=event.get("progressMessage") or "",
            )
            return
        if et == "talk":
            # Agent-initiated text-to-speech (the ``talk`` tool).  In
            # the interactive REPL the agent runs in the ``sorcar web``
            # daemon, so the ``talk`` event arrives here over the UDS
            # connection like any other display event; the terminal
            # machine plays it on its default speakers exactly as a
            # chat webview would (``media/main.js`` case 'talk').
            # Foreign-tab copies were already dropped by ``dispatch``'s
            # tab filter, and the shared player's talkId dedupe keeps
            # any duplicate fan-out copies to one playback per device.
            cli_talk.shared_player().play(event)
            return
        # Silently ignore frontend-only / merge / setTaskText / focus
        # events that have no useful CLI rendering.


class CliClient:
    """Persistent UDS connection to the local ``sorcar web`` daemon.

    Owns a background asyncio event loop running in a thread; the main
    (REPL) thread submits commands through
    :meth:`send` (synchronous) which schedules a JSON write on the
    loop.  Incoming events are forwarded to :class:`_EventDispatcher`
    on the loop thread, which routes synchronous replies into queues
    the REPL thread can ``get(timeout=...)`` on.

    Args:
        sock_path: UDS path of the daemon.
        work_dir: Initial working directory announced via ``setWorkDir``.
        tab_id: Frontend tab id used by the daemon for routing.
        printer: Renderer for streamed events.
    """

    def __init__(
        self,
        sock_path: Path,
        work_dir: str,
        tab_id: str,
        printer: ConsolePrinter,
    ) -> None:
        self.sock_path = sock_path
        self.work_dir = work_dir
        self.tab_id = tab_id
        self.dispatcher = _EventDispatcher(printer, tab_id=tab_id)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._thread: threading.Thread | None = None
        self._connected = threading.Event()
        self._closed = threading.Event()
        self._connect_error: BaseException | None = None

    def start(self, timeout: float = 5.0) -> None:
        """Connect to the daemon and start the event-pump thread.

        Raises:
            ConnectionError: When the connection cannot be established
                within ``timeout`` seconds.  The CLI client requires a
                running ``sorcar web`` daemon; without one we surface a
                clear error pointing the user at the right command.
        """
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="sorcar-cli-client",
        )
        self._thread.start()
        if not self._connected.wait(timeout):
            if self._connect_error is not None:
                raise ConnectionError(
                    f"Cannot connect to sorcar daemon at "
                    f"{self.sock_path}: {self._connect_error}",
                )
            raise ConnectionError(
                f"Cannot connect to sorcar daemon at {self.sock_path} "
                f"within {timeout}s — start it with `sorcar web`.",
            )

    def _run_loop(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._main())
        except BaseException as exc:  # noqa: BLE001 - record startup error
            self._connect_error = exc
            self._connected.set()  # unblock start()
        finally:
            self._closed.set()

    async def _main(self) -> None:
        # The daemon emits very large single-line JSON events at the
        # start of every task — most notably ``system_prompt``, which
        # carries the full ``SYSTEM.md`` plus injections and easily
        # exceeds 64 KiB.  ``asyncio.open_unix_connection`` defaults
        # ``StreamReader``'s buffer to 64 KiB, so an oversize line
        # would raise :class:`asyncio.LimitOverrunError` from
        # :meth:`StreamReader.readline` and tear down the connection,
        # which the user sees as ``Daemon connection lost``.  Use a
        # 16 MiB buffer to accommodate any realistic single event.
        try:
            self._reader, self._writer = await asyncio.open_unix_connection(
                str(self.sock_path),
                limit=16 * 1024 * 1024,
            )
        except OSError as exc:
            self._connect_error = exc
            self._connected.set()
            return
        await self._send_async({"type": "setWorkDir", "workDir": self.work_dir})
        await self._send_async({"type": "ready", "tabId": self.tab_id,
                                "workDir": self.work_dir})
        # Identify this tab as a CLI terminal player.  The daemon
        # arbitrates ``talk`` playback per device: CLI REPL tabs share
        # the daemon machine's speakers with any local webview, so the
        # daemon mutes this tab's talk copies whenever a local webview
        # tab is also subscribed (the webview plays), and lets exactly
        # one CLI tab play otherwise.  Without this hello the daemon cannot
        # tell a CLI tab from a webview tab (both send ``ready``).
        await self._send_async({"type": "cliTabHello", "tabId": self.tab_id})
        self._connected.set()
        try:
            while True:
                try:
                    line = await self._reader.readline()
                except asyncio.LimitOverrunError:
                    # Defence in depth: an event larger than even the
                    # 16 MiB buffer would normally tear down the UDS
                    # silently.  Log loudly and return so the caller's
                    # ``finally`` marks the client as closed — the
                    # symptom (``Daemon connection lost``) is now
                    # accompanied by an actionable log line.
                    logger.error(
                        "daemon emitted oversize event "
                        "(exceeds StreamReader buffer); UDS closed",
                    )
                    return
                except asyncio.IncompleteReadError:
                    return
                if not line:
                    return
                try:
                    event = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                try:
                    self.dispatcher.dispatch(event)
                except Exception:
                    logger.debug("event dispatch failed", exc_info=True)
        except (asyncio.CancelledError, ConnectionError):
            return

    async def _send_async(self, cmd: dict[str, Any]) -> None:
        if self._writer is None:
            return
        cmd.setdefault("tabId", self.tab_id)
        self._writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        try:
            await self._writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            logger.debug("daemon connection lost on send", exc_info=True)

    def send(self, cmd: dict[str, Any]) -> None:
        """Schedule *cmd* on the loop thread (synchronous, non-blocking).

        Args:
            cmd: The JSON command to send.  ``tabId`` is auto-stamped
                with the client's own tab id when missing.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        fut = asyncio.run_coroutine_threadsafe(
            self._send_async(cmd), loop,
        )
        try:
            fut.result(timeout=5)
        except Exception:
            logger.debug("send failed", exc_info=True)

    def close(self) -> None:
        """Tell the daemon the tab is done and stop the loop thread.

        Cleanly closes the UDS writer (so ``_main``'s ``readline``
        unblocks via EOF and ``run_until_complete`` returns), joins
        the loop thread, then closes the asyncio event loop itself.

        Closing the loop is essential for test hygiene: every
        :class:`asyncio.AbstractEventLoop` owns a self-pipe + selector
        kqueue/epoll FD, so leaking them across hundreds of unit
        tests trips the per-process FD soft-limit (256 on macOS) and
        every subsequent ``asyncio.new_event_loop()`` /
        ``socket.socket(AF_UNIX, ...)`` / ``os.pipe()`` call fails
        with :class:`OSError` ``[Errno 24] Too many open files``.
        """
        try:
            self.send({"type": "closeTab", "tabId": self.tab_id})
        except Exception:
            logger.debug("closeTab on shutdown failed", exc_info=True)
        loop = self._loop
        if loop is not None and loop.is_running():
            # Close the writer from the loop thread; this unblocks
            # ``readline()`` in ``_main`` with EOF and causes
            # ``run_until_complete`` to return cleanly.
            async def _close_writer() -> None:
                writer = self._writer
                if writer is None:
                    return
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    logger.debug("writer close failed", exc_info=True)

            try:
                fut = asyncio.run_coroutine_threadsafe(_close_writer(), loop)
                fut.result(timeout=2)
            except Exception:
                logger.debug("writer-close future failed", exc_info=True)
            # Fall-back kick in case ``_main`` is still parked.
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                logger.debug("loop stop failed", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=2)
        if loop is not None and not loop.is_closed():
            try:
                loop.close()
            except Exception:
                logger.debug("loop close failed", exc_info=True)
        self._writer = None
        self._reader = None


def _drain_queue(q: queue.Queue[Any]) -> None:
    """Drain *q* without blocking — used between requests to reset state."""
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            return


def _print_daemon_lost(printer: ConsolePrinter) -> None:
    """Print the standard daemon-disconnect error line."""
    printer.print(
        "[red]✗ Daemon connection lost — type /exit to quit[/red]",
        type="text",
    )


def _start_task(
    client: CliClient,
    prompt: str,
    task_id: str,
    *,
    use_worktree: bool,
    use_parallel: bool,
    auto_commit: bool,
    timeout_seconds: float,
) -> bool:
    """Send the ``run`` command and wait for the daemon to acknowledge.

    Shared by :func:`_submit_task` and :func:`_submit_task_anchored`.
    Blocks until the daemon's first ``status:running=true`` for
    *task_id* arms ``task_active``, the connection drops, or
    *timeout_seconds* expires — printing the matching error line on
    either failure.  The full ``timeout_seconds`` budget is used so
    slow daemon startup (worktree creation, MCP probe, cold model)
    does not silently abandon the task — review #4.

    Args:
        client: The live :class:`CliClient`.
        prompt: The user's instruction for this turn.
        task_id: Freshly minted per-submission task id.
        use_worktree: Forwarded as ``useWorktree`` to the daemon.
        use_parallel: Forwarded as ``useParallel`` to the daemon.
        auto_commit: Forwarded as ``autoCommit`` to the daemon.
        timeout_seconds: Hard cap on the acknowledgement wait.

    Returns:
        ``True`` when the task was acknowledged and is running.
    """
    client.send({
        "type": "run",
        "prompt": prompt,
        "model": client.dispatcher.current_model,
        "workDir": client.work_dir,
        "useWorktree": use_worktree,
        "useParallel": use_parallel,
        "autoCommit": auto_commit,
        "taskId": task_id,
    })
    # Wait on the LATCHED ``task_started`` event, not the
    # level-triggered ``task_active``: a task that starts and finishes
    # between two polls flips ``task_active`` set→clear invisibly and
    # would wedge this loop until ``armed_deadline`` (an hour by
    # default) even though the task ran to completion.  The latch is
    # set by the dispatcher on every status event for this task, so a
    # fast set→clear transition is never missed.
    armed_deadline = time.monotonic() + timeout_seconds
    while (
        not client.dispatcher.task_started.is_set()
        and time.monotonic() < armed_deadline
        and not client._closed.is_set()
    ):
        time.sleep(0.05)
    if client._closed.is_set():
        _print_daemon_lost(client.dispatcher.printer)
        return False
    if not client.dispatcher.task_started.is_set():
        # The daemon never acknowledged the task within the user's
        # timeout budget; surface a clear error instead of pretending
        # the task ran.
        client.dispatcher.printer.print(
            f"[yellow]⏹  Daemon did not acknowledge the task within "
            f"{int(timeout_seconds)}s.[/yellow]",
            type="text",
        )
        return False
    return True


def _request_cli_info(
    client: CliClient, subtype: str, *, timeout: float = 10.0, **extra: Any,
) -> dict[str, Any]:
    """Issue a ``cliInfo`` request and block for the matching reply.

    Each request carries a unique ``requestId`` that the server echoes
    back so the client can filter stale replies racing with newer
    requests (review #14).  The waiter also bails early when the
    daemon connection drops (``client._closed``) so a single
    disconnect does not stall every subsequent slash command for the
    full timeout (review #8 / #25).
    """
    _drain_queue(client.dispatcher.cli_info_q)
    request_id = uuid.uuid4().hex
    cmd: dict[str, Any] = {
        "type": "cliInfo",
        "subtype": subtype,
        "workDir": client.work_dir,
        "tabId": client.tab_id,
        "requestId": request_id,
    }
    cmd.update(extra)
    # Disconnect / timeout sentinels use disjoint fields for the
    # bool flag (``error``) and the human-readable string
    # (``errorMessage``) — review A5/B4 round 2.  The old code put a
    # bool into the same ``error`` field the server uses for an error
    # string, which made the custom-command branch literally print
    # "True" on disconnect.
    disc_msg = "Daemon connection lost — type /exit to quit"
    disc_reply: dict[str, Any] = {
        "type": "cliInfo", "subtype": subtype, "text": f"✗ {disc_msg}",
        "error": True, "errorMessage": disc_msg,
    }
    # Early-fail when the daemon is already gone so the caller does
    # not block on a queue that no producer can write to.
    if client._closed.is_set():
        return disc_reply
    client.send(cmd)
    deadline = time.monotonic() + timeout
    while True:
        if client._closed.is_set():
            return disc_reply
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timeout_msg = f"Daemon timed out waiting for {subtype} reply"
            return {"type": "cliInfo", "subtype": subtype,
                    "text": f"(no reply for {subtype})",
                    "error": True, "errorMessage": timeout_msg,
                    "timedOut": True}
        try:
            ev = client.dispatcher.cli_info_q.get(
                timeout=min(0.25, remaining),
            )
        except queue.Empty:
            continue
        # Filter on requestId so stale replies from prior requests do
        # not get routed to the current waiter.  Replies without an id
        # are accepted as wildcard matches for backwards compat (the
        # server may be older than the client during a rolling
        # upgrade) — but if the subtype also mismatches we keep
        # looking.
        ev_rid = ev.get("requestId", "")
        if ev_rid:
            if ev_rid == request_id:
                return ev
            # Stale reply for a previous request — discard and keep
            # waiting for our matching reply.
            continue
        # No id on the reply: accept only when the subtype matches so
        # at least we cannot confuse a /mcp reply with a /help reply.
        if ev.get("subtype") == subtype:
            return ev


def _request_models(client: CliClient) -> list[dict[str, Any]]:
    """Issue ``getModels`` and block for the ``models`` reply.

    Bails out early when the daemon connection drops
    (``client._closed``) so a disconnect does not stall ``/model
    list`` for the full 10-second timeout — the same early-bail
    contract :func:`_request_cli_info` (review #8 / #25) and the
    ``/autocommit`` wait loop (review A3) already honour.
    """
    _drain_queue(client.dispatcher.models_q)
    if client._closed.is_set():
        return []
    client.send({"type": "getModels"})
    deadline = time.monotonic() + 10.0
    while True:
        if client._closed.is_set():
            return []
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return []
        try:
            ev = client.dispatcher.models_q.get(
                timeout=min(0.25, remaining),
            )
        except queue.Empty:
            continue
        raw = ev.get("models", [])
        return list(raw) if isinstance(raw, list) else []


def _handle_client_slash(  # noqa: PLR0911,PLR0912 - branchy by design
    client: CliClient,
    line: str,
    submit: Callable[[str], None] | None = None,
) -> bool:
    """Handle a ``/`` slash command in client mode.

    Returns ``True`` when the caller should exit the REPL (``/exit``
    or ``/quit``), ``False`` otherwise.

    Every action flows through the daemon: information panels
    (``/help`` / ``/commands`` / ``/skills`` / ``/mcp`` / ``/cost``)
    are answered by the new server-side ``cliInfo`` command (see
    :meth:`_CommandsMixin._cmd_cli_info`), state-changing commands
    re-use the existing webview commands (``newChat`` / ``selectModel``
    / ``getModels`` / ``autocommitAction``), and custom slash commands
    are expanded server-side then submitted back as a normal ``run``.

    Args:
        client: The live :class:`CliClient`.
        line: The raw input line beginning with ``/``.
        submit: Callable used to run the expansion of a custom slash
            command — the SAME one the REPL loop uses for plain task
            lines, so it carries the operator's ``--no-worktree`` /
            ``--no-parallel`` / ``--auto-commit`` flags and, in
            anchored mode, drives the anchored steering box.  The old
            code hard-coded ``_submit_task(client, prompt)`` here,
            silently dropping those flags and bypassing the anchored
            path.  ``None`` falls back to a default
            :func:`_submit_task` for direct callers.
    """
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit"):
        # If a task is in flight, send ``stop`` so the daemon does
        # not keep running it after the CLI client disconnects — the
        # old code only sent ``closeTab`` from the outer ``finally``,
        # leaking a long-running task whenever the user typed
        # ``/exit`` mid-task (review #20 round 1, still present in
        # round 2).
        if client.dispatcher.task_active.is_set():
            try:
                client.send({"type": "stop", "tabId": client.tab_id})
            except Exception:
                logger.debug("/exit stop send failed", exc_info=True)
        return True
    if cmd == "/help":
        reply = _request_cli_info(client, "help")
        text = reply.get("text", "")
        if text:
            print(f"\n{text}\n")
        else:
            _print_help(client.work_dir)
        return False
    if cmd == "/commands":
        reply = _request_cli_info(client, "commands")
        print(f"\n{reply.get('text', '')}\n")
        return False
    if cmd == "/skills":
        reply = _request_cli_info(client, "skills", name=arg)
        print(f"\n{reply.get('text', '')}\n")
        return False
    if cmd == "/mcp":
        reply = _request_cli_info(client, "mcp")
        print(f"\n{reply.get('text', '')}\n")
        return False
    if cmd in ("/clear", "/new"):
        client.send({"type": "newChat"})
        client.dispatcher.chat_id = ""
        print("Started a new chat — context cleared.\n")
        return False
    if cmd == "/resume":
        try:
            chat_id, task_id, limit = _parse_resume_arg(arg)
        except ValueError as exc:
            print(f"Invalid /resume argument: {exc}\n")
            return False
        if task_id:
            # Resolve the task's owning chat from the shared kiss DB
            # the daemon also writes to (the daemon's resumeSession
            # path resolves it the same way server-side, but the CLI
            # needs the chat id locally so the next ``run`` continues
            # the right chat).
            row = _load_chat_events_by_task_id(task_id)
            if row is None:
                print(f"No task found with id {task_id}.\n")
                return False
            chat_id = str(row.get("chat_id", "") or "")
            client.send({
                "type": "resumeSession",
                "chatId": chat_id,
                "taskId": task_id,
            })
            client.dispatcher.chat_id = chat_id
            print(f"Resumed task {task_id} (chat {chat_id}).\n")
        elif chat_id:
            client.send({"type": "resumeSession", "chatId": chat_id})
            client.dispatcher.chat_id = chat_id
            print(f"Resumed chat {chat_id}.\n")
        else:
            # List recent chats from the shared kiss DB the daemon
            # also writes to (there is no in-process agent in client
            # mode), via :func:`_print_recent_chats` directly.
            _print_recent_chats(limit=limit)
            print(
                "\nResume one with: /resume <chat-id>  "
                "or  /resume --task <task-id>\n"
            )
        return False
    if cmd == "/model":
        if arg == "list":
            # The original REPL's local listing reads
            # ``get_generation_model_listing()`` which returns
            # ``(name, provider, configured)`` triples — including the
            # ``configured`` (API-key-present) flag the daemon does
            # NOT emit on its ``models`` event (which only carries
            # ``{name, vendor, inp, out, uses}``).  Falling through to
            # the local helper preserves the green check / red cross
            # column from the standalone REPL.  The daemon's models
            # event is still requested (and consumed) so the side
            # effect of refreshing ``current_model`` is preserved.
            _request_models(client)
            _print_model_list(client.dispatcher.current_model)
            return False
        if not arg:
            reply = _request_cli_info(client, "modelCurrent")
            text = reply.get("text", "")
            if text:
                print(f"\n{text}\n")
            else:
                print(f"\nCurrent model: {client.dispatcher.current_model}\n")
            return False
        client.send({"type": "selectModel", "model": arg})
        client.dispatcher.current_model = arg
        print(f"Model switched to {arg} for subsequent tasks.\n")
        return False
    if cmd in ("/cost", "/usage", "/context"):
        reply = _request_cli_info(client, "cost")
        print(f"\n{reply.get('text', '')}\n")
        return False
    if cmd == "/autocommit":
        # Drive the same flow the webview kicks off: request a
        # generated commit message, then dispatch the autocommit
        # action so the daemon stages + commits.
        _drain_queue(client.dispatcher.commit_q)
        client.send({"type": "generateCommitMessage",
                     "workDir": client.work_dir})
        # Poll for the reply but honour ``client._closed`` so a
        # daemon disconnect mid-commit does NOT freeze the REPL for
        # the full 30 s budget (review A3 round 2).
        ev: dict[str, Any] | None = None
        deadline = time.monotonic() + 30.0
        while True:
            if client._closed.is_set():
                print(
                    "\n✗ Auto-commit aborted: daemon connection lost.\n",
                )
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(
                    "\n✗ Auto-commit timed out waiting for a "
                    "commit message.\n",
                )
                return False
            try:
                ev = client.dispatcher.commit_q.get(
                    timeout=min(0.25, remaining),
                )
                break
            except queue.Empty:
                continue
        msg = ev.get("message", "")
        if not msg:
            # Prefer ``errorMessage`` (string) over the bool ``error``
            # field so a daemon that sets ``error=True`` does not make
            # the CLI print the literal "True" (review M1 round 3).
            err = ev.get("errorMessage") or ev.get("error") or "no message"
            print(f"\n✗ Auto-commit: {err}\n")
            return False
        client.send({
            "type": "autocommitAction",
            "action": "commit",
            "message": msg,
            "workDir": client.work_dir,
        })
        print(f"\n✓ Auto-commit dispatched: {msg.splitlines()[0]}\n")
        return False
    # Try custom command via server expansion.
    name = cmd.lstrip("/")
    reply = _request_cli_info(
        client, "expandCommand", name=name, args=arg,
    )
    if reply.get("found"):
        prompt = reply.get("text", "")
        src = reply.get("source", "")
        path = reply.get("path", "")
        print(f"⚡ Running custom command {cmd} ({src}:{path})")
        if submit is not None:
            submit(prompt)
        else:
            _submit_task(client, prompt)
        return False
    # ``errorMessage`` (string) carries the human text; ``error``
    # is a bool flag.  Old code conflated them and printed the
    # literal "True" on disconnect (review A5/B4 round 2).
    err = reply.get("errorMessage") or (
        f"Unknown command: {cmd}. Type /help for the list of commands."
    )
    print(f"{err}\n")
    return False


def _submit_task(
    client: CliClient,
    prompt: str,
    *,
    use_worktree: bool = True,
    use_parallel: bool = True,
    auto_commit: bool = False,
    timeout_seconds: float = 60 * 60,
) -> None:
    """Send a ``run`` command for *prompt* and block until task ends.

    Uses the dispatcher's ``task_active`` event (toggled by the
    server's ``status`` broadcasts) as the completion signal and
    additionally polls :class:`queue.Queue` for ``askUser`` questions
    on the REPL thread so the user's ``input()`` reply does NOT block
    the loop thread (which would freeze every other inbound event).

    Args:
        client: The live :class:`CliClient`.
        prompt: The user's instruction for this turn.
        use_worktree: Forward the ``useWorktree`` flag from the CLI's
            argparsed ``run_kwargs`` so ``--no-worktree`` is honoured
            in client mode.
        use_parallel: Same as above for ``--no-parallel``.
        auto_commit: Same as above for ``--auto-commit``.
        timeout_seconds: Hard cap on the wait loop so a wedged daemon
            does not pin the REPL forever.
    """
    # Mint a per-submission task id BEFORE clearing ``task_active``
    # so any inbound status event observed mid-transition matches
    # the new task id (review B1 round 2 ordering fix).
    new_task_id = uuid.uuid4().hex
    with client.dispatcher.task_id_lock:
        client.dispatcher.current_task_id = new_task_id
    # Drain any stale status / askUser queue entries left over from a
    # prior task; this closes issue #46 from the review (a stale
    # ``status:false`` between two tasks would otherwise clear
    # ``task_active`` immediately and silently drop the new task).
    client.dispatcher.task_active.clear()
    client.dispatcher.task_started.clear()
    _drain_queue(client.dispatcher.ask_user_q)
    start = time.time()
    try:
        if not _start_task(
            client, prompt, new_task_id,
            use_worktree=use_worktree, use_parallel=use_parallel,
            auto_commit=auto_commit, timeout_seconds=timeout_seconds,
        ):
            return
        deadline = time.monotonic() + timeout_seconds
        while client.dispatcher.task_active.is_set():
            if client._closed.is_set():
                _print_daemon_lost(client.dispatcher.printer)
                return
            if time.monotonic() > deadline:
                client.dispatcher.printer.print(
                    f"[yellow]⏹  Task wait timed out after "
                    f"{int(timeout_seconds)}s.[/yellow]",
                    type="text",
                )
                return
            # Drain pending askUser questions on the REPL thread so the
            # user's input() blocks here, never on the asyncio loop.
            try:
                question = client.dispatcher.ask_user_q.get(timeout=0.1)
            except queue.Empty:
                continue
            client.dispatcher.printer.print(
                f"[bold yellow]Agent asks:[/bold yellow] {question}",
                type="text",
            )
            try:
                ans = input("> ")
            except (EOFError, KeyboardInterrupt):
                # Ctrl+C (KeyboardInterrupt) and Ctrl+D / closed
                # stdin (EOFError) at the question prompt are both
                # cancellation gestures, not answers.  The old code
                # fabricated the literal reply "done" on EOF, which
                # the agent treated as a genuine user answer —
                # inconsistent with the anchored path and with the
                # surrounding loop's Ctrl+C semantics.  There is
                # also a signal-vs-EOF race under load: when a
                # parent process delivers SIGINT and closes the
                # child's stdin nearly simultaneously (as
                # ``subprocess.communicate()`` does), the EOF may
                # win the race and ``input()`` raises EOFError
                # instead of KeyboardInterrupt.  Treat both
                # identically: forward ``stop`` to the daemon and
                # keep waiting for the task to wind down.
                client.send({"type": "stop"})
                client.dispatcher.printer.print(
                    "[yellow]⏹  Sent stop to daemon.[/yellow]",
                    type="text",
                )
                continue
            client.send({"type": "userAnswer", "answer": ans})
    finally:
        # Always reset the per-task dispatcher state, regardless of
        # which exit path was taken (early returns, timeout, disconnect,
        # exceptions) — review B1 round 2.  Without this, a stale
        # ``current_task_id`` survives the call and the next task's
        # status events get filtered out under the OLD id.
        with client.dispatcher.task_id_lock:
            client.dispatcher.current_task_id = ""
        client.dispatcher.task_active.clear()
        client.dispatcher.task_started.clear()
        _print_elapsed(client, start)


def _submit_task_anchored(
    client: CliClient,
    prompt: str,
    repl: AnchoredRepl,
    *,
    use_worktree: bool = True,
    use_parallel: bool = True,
    auto_commit: bool = False,
    timeout_seconds: float = 60 * 60,
) -> None:
    """Run a daemon task while keeping the anchored input box pinned.

    Mirrors :func:`_submit_task` but uses the already-drawn bottom
    box (owned by ``repl``) for the duration of the task: the box's
    title flips to :data:`STEER_TITLE`, every line the user submits
    while the task is running is sent to the daemon as an
    ``appendUserMessage`` command (exactly the way the VS Code
    frontend / remote browser webapp queue follow-ups), and the
    box's status shows a running ``queued: N`` count.  Ctrl+C
    forwards a ``stop`` command to the daemon.  ``askUser`` events
    coming back from the daemon flip the box title to "answer the
    question above" so the next submitted line is sent as a
    ``userAnswer`` reply.

    Args:
        client: The live :class:`CliClient`.
        prompt: The user's instruction for this turn.
        repl: The anchored REPL whose box drives the steering loop.
        use_worktree: Forwarded as ``useWorktree`` to the daemon.
        use_parallel: Forwarded as ``useParallel`` to the daemon.
        auto_commit: Forwarded as ``autoCommit`` to the daemon.
        timeout_seconds: Hard wall-clock cap on the wait so a wedged
            daemon never pins the REPL forever.
    """
    new_task_id = uuid.uuid4().hex
    with client.dispatcher.task_id_lock:
        client.dispatcher.current_task_id = new_task_id
    client.dispatcher.task_active.clear()
    client.dispatcher.task_started.clear()
    _drain_queue(client.dispatcher.ask_user_q)
    queued = [0]
    pending_question = [False]
    # Wall-clock ``start`` is only for the elapsed-time display; the task
    # timeout must use a monotonic deadline (comparing ``time.monotonic()``
    # against a ``time.time()`` anchor never fires).
    start = time.time()
    deadline = time.monotonic() + timeout_seconds
    try:
        if not _start_task(
            client, prompt, new_task_id,
            use_worktree=use_worktree, use_parallel=use_parallel,
            auto_commit=auto_commit, timeout_seconds=timeout_seconds,
        ):
            return

        def on_submit(line: str) -> None:
            text = line.strip()
            if not text:
                return
            if pending_question[0]:
                client.send({"type": "userAnswer", "answer": line})
                pending_question[0] = False
                with repl.lock:
                    repl.box.title = STEER_TITLE
                    repl.box.status = (
                        QUEUED_STATUS_FMT.format(n=queued[0])
                        if queued[0] else ""
                    )
                    repl.box.redraw()
                return
            client.send({
                "type": "appendUserMessage",
                "prompt": text,
                "tabId": client.tab_id,
            })
            queued[0] += 1
            with repl.lock:
                repl.box.status = QUEUED_STATUS_FMT.format(n=queued[0])
                repl.box.redraw()
                sys.stdout.write(QUEUED_FMT.format(text=text))
                sys.stdout.flush()

        def on_abort() -> None:
            try:
                client.send({"type": "stop"})
            except Exception:  # noqa: BLE001 - defensive
                logger.debug("stop send failed", exc_info=True)
            client.dispatcher.printer.print(
                "[yellow]⏹  Sent stop to daemon.[/yellow]",
                type="text",
            )

        def on_idle() -> None:
            try:
                question = client.dispatcher.ask_user_q.get_nowait()
            except queue.Empty:
                return
            with repl.lock:
                sys.stdout.write(QUESTION_FMT.format(question=question))
                sys.stdout.flush()
                repl.box.title = ASK_TITLE
                repl.box.redraw()
            pending_question[0] = True

        def is_done() -> bool:
            return (
                not client.dispatcher.task_active.is_set()
                or client._closed.is_set()
                or time.monotonic() > deadline
            )

        repl.run_steering_loop(on_submit, on_abort, is_done, on_idle)
        if client._closed.is_set():
            _print_daemon_lost(client.dispatcher.printer)
    finally:
        with client.dispatcher.task_id_lock:
            client.dispatcher.current_task_id = ""
        client.dispatcher.task_active.clear()
        client.dispatcher.task_started.clear()
        _print_elapsed(client, start)


def _run_repl_loop(
    client: CliClient,
    read_line: Callable[[], str | None],
    submit: Callable[[str], None],
    voice_start: Callable[[], cli_voice.VoiceSession | None] | None = None,
) -> None:
    """Drive the shared interactive client loop until the user exits.

    Both interactive client modes — the anchored bottom-box REPL and
    the inline readline / prompt_toolkit fallback — share the exact
    same loop semantics: double-Ctrl+C arming (the first Ctrl+C stops
    a running task or warns, the second exits), EOF / Ctrl+D exits,
    blank lines are skipped, exit words break, ``/`` lines dispatch
    to :func:`_handle_client_slash` resiliently, and any other line
    is submitted as a task with Ctrl+C during the wait forwarding a
    ``stop`` to the daemon.  The loop used to be copy-pasted in both
    modes, so a behavioural fix applied to one could silently miss
    the other (w2 F10/F20) — it now lives here once.

    ``/voice`` is handled here rather than in
    :func:`_handle_client_slash` because captured speech must flow
    through this very loop and be treated exactly like a typed line
    (blank → skip, exit words, slash commands, otherwise submitted as
    a task).  In the anchored REPL voice runs in the *background*
    (:func:`~kiss.agents.sorcar.cli_voice.start_voice_anchored`):
    *read_line* keeps reading the keyboard as usual — typing stays
    fully usable — while the session's pump injects recognised speech
    into the box so it arrives through *read_line* like typed input,
    both at the idle prompt and (as queued steering follow-ups) while
    a task runs.  The plain fallback keeps the modal per-utterance
    capture, where the next line comes from the wake-word listener
    instead of *read_line*.  Voice chat is continuous until ``/voice``
    toggles it off (typed or spoken), the modal capture reports
    ``None`` (cancel key), or the listener fails.  The listener child
    is always terminated on exit, including exceptions.

    Args:
        client: The connected :class:`CliClient`.
        read_line: Blocking reader returning the next input line, or
            ``None`` on EOF; may raise :class:`KeyboardInterrupt`.
        submit: Runs one task for the given line; may raise
            :class:`KeyboardInterrupt` when the user aborts the wait.
        voice_start: Zero-arg factory starting a voice session for
            ``/voice`` (the anchored REPL binds
            :func:`~kiss.agents.sorcar.cli_voice.start_voice_anchored`
            to its box); ``None`` falls back to the modal
            :func:`~kiss.agents.sorcar.cli_voice.read_voice_line_plain`
            capture.
    """
    interrupt_armed = False
    voice: cli_voice.VoiceSession | None = None
    try:
        while True:
            if voice is not None and voice.background and not voice.active:
                # The background listener died; its pump already told
                # the user.  Drop the session so /voice can restart it.
                voice.close()
                voice = None
            if voice is not None and not voice.background:
                line = voice.read()
                if line is None:  # cancelled or listener failed
                    voice.close()
                    voice = None
                    continue
            else:
                try:
                    line = read_line()
                except KeyboardInterrupt:
                    if interrupt_armed:
                        print("\nGoodbye.")
                        break
                    interrupt_armed = True
                    # If a task is running, stop it; otherwise just arm
                    # the second Ctrl-C to exit (matches REPL behaviour).
                    if client.dispatcher.task_active.is_set():
                        client.send({"type": "stop"})
                        print("\n⏹  Sent stop to daemon.")
                    else:
                        print("\n(Press Ctrl+C again or type /exit to quit)")
                    continue
                if line is None:  # EOF / Ctrl+D
                    print("\nGoodbye.")
                    break
            interrupt_armed = False
            text = line.strip()
            if not text:
                continue
            if text in _EXIT_WORDS:
                break
            if text == "/voice":
                if voice is not None:
                    # /voice toggles: typed or spoken while voice is on,
                    # it turns voice mode off and reaps the listener.
                    voice.close()
                    voice = None
                    print(f"{YELLOW}🎤 Voice mode off.{RESET}")
                    continue
                if voice_start is not None:
                    voice = voice_start()
                else:
                    voice = cli_voice.start_voice(
                        cli_voice.read_voice_line_plain
                    )
                continue
            if text.startswith("/"):
                try:
                    if _handle_client_slash(client, line, submit):
                        break
                except Exception as exc:  # noqa: BLE001 - resilient REPL
                    logger.debug("slash command failed", exc_info=True)
                    print(f"\n✗ Command failed: {exc}\n")
                continue
            try:
                submit(line)
            except KeyboardInterrupt:
                client.send({"type": "stop"})
                print("\n⏹  Task interrupted.\n")
    finally:
        # Never leak the listener child — REPL exit, exit words and
        # exceptions all land here while a voice session is active.
        if voice is not None:
            voice.close()


def _run_anchored_client(
    client: CliClient,
    work_dir: str,
    model_name: str,
    active_file: str,
    *,
    use_worktree: bool,
    use_parallel: bool,
    auto_commit: bool,
) -> int:
    """Run the daemon-client REPL with the input bar pinned to the bottom.

    Used when the terminal supports the steering box (POSIX TTY with
    termios) and is at least :data:`_MIN_ROWS` tall.  The bottom box
    is owned by :class:`AnchoredRepl` for the entire session: idle
    reads (next instruction) flow through :meth:`read_idle_line`
    under the :data:`IDLE_TITLE` preset, and task execution
    (``run`` → ``status:false``) flows through :func:`_submit_task_anchored`
    under the :data:`STEER_TITLE` preset, with submitted lines sent
    to the daemon as ``appendUserMessage`` exactly as the VS Code
    frontend / remote browser webapp queue follow-ups.

    Args:
        client: The connected :class:`CliClient`.
        work_dir: Project directory (only used for completion / history).
        model_name: Initial model name for the welcome banner.
        active_file: Active editor file used by the completer for
            identifier-suffix predictive completion.
        use_worktree: Forwarded to every ``run`` command.
        use_parallel: Forwarded to every ``run`` command.
        auto_commit: Forwarded to every ``run`` command.

    Returns:
        ``0`` on a clean exit.
    """
    completer = CliCompleter(work_dir, active_file)
    history_path = _history_path(work_dir)
    history = _load_history_lines(history_path)

    def completer_fn(buf: str) -> list[tuple[str, str]]:
        try:
            return completer.build_menu(buf)
        except Exception:  # pragma: no cover - defensive
            logger.debug("anchored completion failed", exc_info=True)
            return []

    with AnchoredRepl(completer_fn=completer_fn, history=history) as repl:
        _print_welcome(work_dir, model_name or "(daemon)")
        _run_repl_loop(
            client,
            repl.read_idle_line,
            functools.partial(
                _submit_task_anchored,
                client,
                repl=repl,
                use_worktree=use_worktree,
                use_parallel=use_parallel,
                auto_commit=auto_commit,
            ),
            voice_start=functools.partial(
                cli_voice.start_voice_anchored, repl.box,
            ),
        )
        _save_history_lines(history_path, repl.box.history)
    return 0


def _print_elapsed(client: CliClient, start: float) -> None:
    """Print the wall-clock task duration after a task ends.

    The standalone REPL printed this via ``print_outcome`` /
    ``_print_run_stats`` in :mod:`cli_helpers` — only the in-process
    agent's chat object knew when ``run()`` returned.  In client mode
    the daemon's ``Result`` panel already shows tokens + cost + steps,
    so we only emit the elapsed-time line here for parity with the
    old REPL (review item #25).  Routed through ``ConsolePrinter`` so
    a partially-written streamed line gets terminated first.
    """
    elapsed = time.time() - start
    client.dispatcher.printer.flush_newline()
    # Route through the same ``ConsolePrinter`` the dispatcher uses so
    # tests that capture the printer's configured ``file=`` see the
    # elapsed line, and so downstream redirection (file logging,
    # captured-output panels) does not split it from the streamed
    # output (review #34).
    client.dispatcher.printer.print(f"Time: {elapsed:.1f}s", type="text")


def run_client(
    work_dir: str,
    model_name: str = "",
    sock_path: Path | None = None,
    active_file: str = "",
    *,
    use_worktree: bool = True,
    use_parallel: bool = True,
    auto_commit: bool = False,
) -> int:
    """Run the interactive sorcar CLI as a client of ``sorcar web``.

    This is the entry point invoked by
    :func:`worktree_sorcar_agent.main` in interactive mode (no
    ``-t/--task`` / ``-f/--file``).  The on-screen behaviour — prompt
    panel, fast-completes, slash commands, streamed agent events,
    rich Result panel — runs locally while the actual execution
    happens in the local ``sorcar web`` daemon process.

    Args:
        work_dir: Project directory; sent to the daemon via
            ``setWorkDir`` and used by the local completer.
        model_name: Initial model to display; the daemon's reply to
            ``getConfig`` (fanned out from ``ready``) supplies the
            canonical value.
        sock_path: UDS path of the daemon (override for tests via the
            ``KISS_SORCAR_SOCK`` env var).
        active_file: Active editor file used by the completer for
            identifier-suffix predictive ghost completion.

    Returns:
        ``0`` on a clean exit (``/exit`` / Ctrl+D), ``1`` when the
        daemon cannot be reached so the operator sees a non-zero
        shell exit code.
    """
    path = sock_path or _sock_path()
    if not _wait_for_socket(path, timeout=2.0):
        print(
            f"✗ No sorcar daemon found at {path}.\n"
            f"   Start one in another terminal with:  sorcar web",
            file=sys.stderr,
        )
        return 1
    tab_id = uuid.uuid4().hex
    printer = ConsolePrinter()
    client = CliClient(path, work_dir, tab_id, printer)
    try:
        client.start(timeout=5.0)
    except ConnectionError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1

    client.dispatcher.current_model = model_name

    # Wipe the terminal so the interactive session starts on a clean
    # canvas (no leftover shell prompt / prior command output above
    # the welcome banner).  No-op when stdout is not a TTY so pytest
    # capture and piped output stay clean.
    _clear_terminal()

    # When the terminal supports the steering box (POSIX TTY with
    # termios) and is tall enough, run the REPL with the input bar
    # pinned to the bottom of the screen for both idle reads and
    # task execution — matching Claude Code's fullscreen TUI
    # behaviour where the rectangular input box stays visible while
    # the agent works.  Lines submitted into the box during a task
    # are sent to the daemon as ``appendUserMessage`` (the same
    # command the VS Code extension and the remote browser webapp
    # use to queue follow-ups).  Off-TTY (pytest, pipes), Windows,
    # or tiny terminals fall back to the inline readline /
    # prompt_toolkit path below.
    rows, _ = _term_size()
    if supports_steering() and rows >= _MIN_ROWS:
        try:
            return _run_anchored_client(
                client,
                work_dir,
                model_name,
                active_file,
                use_worktree=use_worktree,
                use_parallel=use_parallel,
                auto_commit=auto_commit,
            )
        finally:
            client.close()

    completer = CliCompleter(work_dir, active_file)
    history_path = _history_path(work_dir)
    reader = _make_ptk_reader(completer, history_path)
    using_readline = reader is None
    if using_readline:
        _setup_readline(completer, history_path)

    _print_welcome(work_dir, model_name or "(daemon)")

    try:
        _run_repl_loop(
            client,
            functools.partial(_read_line, _PROMPT, reader),
            functools.partial(
                _submit_task,
                client,
                use_worktree=use_worktree,
                use_parallel=use_parallel,
                auto_commit=auto_commit,
            ),
        )
    finally:
        if using_readline:
            _save_history(history_path)
        client.close()
    return 0


__all__ = [
    "CliClient",
    "_EventDispatcher",
    "_handle_client_slash",
    "_run_anchored_client",
    "_sock_path",
    "_submit_task",
    "_submit_task_anchored",
    "_wait_for_socket",
    "run_client",
]
