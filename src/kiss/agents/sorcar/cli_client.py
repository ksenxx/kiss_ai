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
``$KISS_SORCAR_SOCK`` (default ``~/.kiss/sorcar.sock``).  The daemon's
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
import json
import logging
import os
import queue
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.cli_panel import (
    CYAN,
    PROMPT_MARKER,
    RESET,
)
from kiss.agents.sorcar.cli_repl import (
    _EXIT_WORDS,
    CliCompleter,
    _handle_resume,
    _history_path,
    _make_ptk_reader,
    _print_help,
    _print_model_list,
    _print_welcome,
    _read_line,
    _record_mentions,
    _save_history,
    _setup_readline,
)
from kiss.core.print_to_console import ConsolePrinter

logger = logging.getLogger(__name__)

_DEFAULT_SOCK_PATH = Path.home() / ".kiss" / "sorcar.sock"
_PROMPT = f"{CYAN}{PROMPT_MARKER}{RESET}"


def _sock_path() -> Path:
    """Return the daemon UDS socket path the client should connect to."""
    env = os.environ.get("KISS_SORCAR_SOCK")
    return Path(env) if env else _DEFAULT_SOCK_PATH


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

    def __init__(self, printer: ConsolePrinter) -> None:
        self.printer = printer
        # Synchronous waiters used by ``CliClient`` slash commands.
        self.cli_info_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self.models_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self.commit_q: queue.Queue[dict[str, Any]] = queue.Queue()
        # Latest in-flight task / chat metadata mirrored from server.
        self.chat_id: str = ""
        self.current_model: str = ""
        self.task_active = threading.Event()
        # Hook called when the server emits ``askUser`` so the main
        # REPL thread can prompt and reply.
        self.ask_user_cb: Any | None = None

    def dispatch(self, event: dict[str, Any]) -> None:
        """Route one event to the appropriate handler."""
        et = event.get("type", "")
        # Strip server-side routing metadata that has no UI meaning.
        event.pop("taskId", None)
        if et == "cliInfo":
            self.cli_info_q.put(event)
            return
        if et == "models":
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
            running = bool(event.get("running", False))
            if running:
                self.task_active.set()
            else:
                self.task_active.clear()
            return
        if et == "askUser":
            cb = self.ask_user_cb
            question = str(event.get("question", ""))
            if cb is not None:
                cb(question)
            return
        if et == "error":
            self.printer.print(
                f"[red]✗ {event.get('text', '')}[/red]", type="text",
            )
            return
        # Streamed display events: route to ConsolePrinter.
        self._render(event)

    def _render(self, event: dict[str, Any]) -> None:
        et = event.get("type", "")
        if et == "text_delta":
            self.printer.token_callback(event.get("text", ""))
            return
        if et == "thinking_delta":
            self.printer.token_callback(event.get("text", ""))
            return
        if et == "thinking_start":
            self.printer.thinking_callback(True)
            return
        if et == "thinking_end":
            self.printer.thinking_callback(False)
            return
        if et == "text_end":
            # Force a newline so the next panel starts on its own row.
            if getattr(self.printer, "_mid_line", False):
                self.printer._flush_newline()  # type: ignore[attr-defined]
            return
        if et == "prompt":
            self.printer.print(event.get("text", ""), type="prompt")
            return
        if et == "system_prompt":
            self.printer.print(event.get("text", ""), type="system_prompt")
            return
        if et == "tool_call":
            self.printer.print(
                event.get("name", ""),
                type="tool_call",
                tool_input=event.get("input", {}) or {},
            )
            return
        if et == "tool_result":
            self.printer.print(
                event.get("content", ""),
                type="tool_result",
                is_error=bool(event.get("is_error", False)),
                tool_name=event.get("tool_name", ""),
            )
            return
        if et == "system_output":
            self.printer.print(event.get("text", ""), type="bash_stream")
            return
        if et == "usage_info":
            self.printer.print(
                event.get("text", ""),
                type="usage_info",
                total_tokens=event.get("total_tokens", 0),
                cost=event.get("cost", "N/A"),
                total_steps=event.get("total_steps", 0),
            )
            return
        if et == "result":
            self.printer.print(
                event.get("text", "") or "(no result)",
                type="result",
                total_tokens=event.get("total_tokens", 0),
                cost=event.get("cost", "N/A"),
                step_count=event.get("step_count", 0),
            )
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
        self.dispatcher = _EventDispatcher(printer)
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
        try:
            self._reader, self._writer = await asyncio.open_unix_connection(
                str(self.sock_path),
            )
        except OSError as exc:
            self._connect_error = exc
            self._connected.set()
            return
        await self._send_async({"type": "setWorkDir", "workDir": self.work_dir})
        await self._send_async({"type": "ready", "tabId": self.tab_id,
                                "workDir": self.work_dir})
        self._connected.set()
        try:
            while True:
                line = await self._reader.readline()
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
        """Tell the daemon the tab is done and stop the loop thread."""
        try:
            self.send({"type": "closeTab", "tabId": self.tab_id})
        except Exception:
            logger.debug("closeTab on shutdown failed", exc_info=True)
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                logger.debug("loop stop failed", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=2)


def _drain_queue(q: queue.Queue[Any]) -> None:
    """Drain *q* without blocking — used between requests to reset state."""
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            return


def _request_cli_info(
    client: CliClient, subtype: str, **extra: Any,
) -> dict[str, Any]:
    """Issue a ``cliInfo`` request and block for the matching reply."""
    _drain_queue(client.dispatcher.cli_info_q)
    cmd: dict[str, Any] = {
        "type": "cliInfo",
        "subtype": subtype,
        "workDir": client.work_dir,
        "tabId": client.tab_id,
    }
    cmd.update(extra)
    client.send(cmd)
    try:
        return client.dispatcher.cli_info_q.get(timeout=10)
    except queue.Empty:
        return {"type": "cliInfo", "subtype": subtype,
                "text": f"(no reply for {subtype})"}


def _request_models(client: CliClient) -> list[dict[str, Any]]:
    """Issue ``getModels`` and block for the ``models`` reply."""
    _drain_queue(client.dispatcher.models_q)
    client.send({"type": "getModels"})
    try:
        ev = client.dispatcher.models_q.get(timeout=10)
    except queue.Empty:
        return []
    raw = ev.get("models", [])
    return list(raw) if isinstance(raw, list) else []


def _handle_client_slash(  # noqa: PLR0911,PLR0912 - branchy by design
    client: CliClient, line: str,
) -> bool:
    """Handle a ``/`` slash command in client mode.

    Returns ``True`` when the caller should exit the REPL (``/exit``
    or ``/quit``), ``False`` otherwise.

    The command surface mirrors :func:`cli_repl._handle_slash` but
    every action now flows through the daemon: information panels
    (``/help`` / ``/commands`` / ``/skills`` / ``/mcp`` / ``/cost``)
    are answered by the new server-side ``cliInfo`` command (see
    :meth:`_CommandsMixin._cmd_cli_info`), state-changing commands
    re-use the existing webview commands (``newChat`` / ``selectModel``
    / ``getModels`` / ``autocommitAction``), and custom slash commands
    are expanded server-side then submitted back as a normal ``run``.

    Args:
        client: The live :class:`CliClient`.
        line: The raw input line beginning with ``/``.
    """
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit"):
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
        if arg:
            client.send({"type": "resumeSession", "chatId": arg})
            client.dispatcher.chat_id = arg
            print(f"Resumed chat {arg}.\n")
        else:
            # Listing recent chats reads the local kiss DB the daemon
            # also writes to; reuse the local helper unchanged.  The
            # stub deliberately fails the ``isinstance(..., Chat
            # SorcarAgent)`` check inside :func:`_handle_resume` so
            # only the "list recent chats" branch executes.
            _handle_resume(
                _StubAgent(client.dispatcher.chat_id),  # type: ignore[arg-type]
                arg,
            )
        return False
    if cmd == "/model":
        if arg == "list":
            models = _request_models(client)
            if models:
                _print_model_listing_from_server(
                    models, client.dispatcher.current_model,
                )
            else:
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
        try:
            ev = client.dispatcher.commit_q.get(timeout=30)
        except queue.Empty:
            print("\n✗ Auto-commit timed out waiting for a commit message.\n")
            return False
        msg = ev.get("message", "")
        if not msg:
            print(f"\n✗ Auto-commit: {ev.get('error', 'no message')}\n")
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
        _submit_task(client, prompt)
        return False
    err = reply.get("error", "") or (
        f"Unknown command: {cmd}. Type /help for the list of commands."
    )
    print(f"{err}\n")
    return False


def _print_model_listing_from_server(
    models: list[dict[str, Any]], current: str,
) -> None:
    """Render the ``models`` event in the same columnar layout as REPL.

    The daemon emits one entry per generation model; each entry has
    at least ``name`` and may carry ``provider`` / ``configured``
    flags depending on the server version.  Missing fields degrade
    gracefully to a single-column listing.
    """
    if not models:
        print("No generation models are available.\n")
        return
    name_w = max(len(m.get("name", "")) for m in models)
    prov_w = max(len(m.get("provider", "")) for m in models) or 1
    configured = sum(1 for m in models if m.get("configured", True))
    print(
        f"\nGeneration models ({configured}/{len(models)} with credentials "
        f"configured):"
    )
    for m in models:
        name = m.get("name", "")
        provider = m.get("provider", "")
        ok = bool(m.get("configured", True))
        mark = "✓" if ok else "✗"
        status = "configured" if ok else "no API key"
        here = "  ← current" if name == current else ""
        print(
            f"  {mark} {name:<{name_w}}  {provider:<{prov_w}}  "
            f"{status}{here}"
        )
    print()


class _StubAgent:
    """Minimal agent stub passed to local-only helpers reused from REPL.

    :func:`cli_repl._handle_resume` checks ``isinstance(agent,
    ChatSorcarAgent)`` before letting the user resume; in client mode
    no agent runs in this process, so the stub deliberately fails
    that check and only the "list recent chats" branch executes —
    which reads from the local kiss DB the daemon also writes to.
    """

    def __init__(self, chat_id: str = "") -> None:
        self.chat_id = chat_id


def _submit_task(client: CliClient, prompt: str) -> None:
    """Send a ``run`` command for *prompt* and block until task ends.

    Uses the dispatcher's ``task_active`` event (toggled by the
    server's ``status`` broadcasts) as the completion signal, with a
    last-event fallback so a missed final ``status:false`` does not
    wedge the REPL forever.

    During the wait an in-band ``askUser`` event triggers a
    synchronous prompt on this thread and the reply is shipped back
    as ``userAnswer`` — exactly the same protocol the webview uses.

    Args:
        client: The live :class:`CliClient`.
        prompt: The user's instruction for this turn.
    """
    answer_pending = threading.Event()
    answer_text: list[str] = []

    def _on_ask(question: str) -> None:
        print(f"\n[bold yellow]Agent asks:[/bold yellow] {question}")
        try:
            ans = input("> ")
        except (EOFError, KeyboardInterrupt):
            ans = "done"
        answer_text.append(ans)
        answer_pending.set()
        client.send({"type": "userAnswer", "answer": ans})

    client.dispatcher.ask_user_cb = _on_ask
    client.dispatcher.task_active.set()
    client.send({
        "type": "run",
        "prompt": prompt,
        "model": client.dispatcher.current_model,
        "workDir": client.work_dir,
        "useWorktree": True,
        "useParallel": True,
        "autoCommit": False,
    })
    try:
        # Wait for the server to emit the final ``status:false``.
        # The dispatcher clears ``task_active`` on receipt.  An
        # outer timeout caps total wait so a stuck server cannot
        # wedge the REPL forever.
        deadline = time.monotonic() + 60 * 60
        while client.dispatcher.task_active.is_set():
            if time.monotonic() > deadline:
                print("\n⏹  Task wait timed out after 1h.\n")
                break
            time.sleep(0.1)
    finally:
        client.dispatcher.ask_user_cb = None


def run_client(
    work_dir: str,
    model_name: str = "",
    sock_path: Path | None = None,
    active_file: str = "",
) -> int:
    """Run the interactive sorcar CLI as a client of ``sorcar web``.

    This is the entry point invoked by
    :func:`worktree_sorcar_agent.main` in interactive mode (no
    ``-t/--task`` / ``-f/--file``).  Mirrors the surface of the old
    :func:`cli_repl.run_repl` so the on-screen behaviour — prompt
    panel, fast-completes, slash commands, streamed agent events,
    rich Result panel — is preserved while the actual execution
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

    completer = CliCompleter(work_dir, active_file)
    history_path = _history_path(work_dir)
    reader = _make_ptk_reader(completer, history_path)
    using_readline = reader is None
    if using_readline:
        _setup_readline(completer, history_path)

    client.dispatcher.current_model = model_name
    _print_welcome(work_dir, model_name or "(daemon)")

    interrupt_armed = False
    try:
        while True:
            try:
                line = _read_line(_PROMPT, reader)
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
            if text.startswith("/"):
                try:
                    if _handle_client_slash(client, line):
                        break
                except Exception as exc:
                    logger.debug("slash command failed", exc_info=True)
                    print(f"\n✗ Command failed: {exc}\n")
                continue
            _record_mentions(line)
            try:
                _submit_task(client, line)
            except KeyboardInterrupt:
                client.send({"type": "stop"})
                print("\n⏹  Task interrupted.\n")
    finally:
        if using_readline:
            _save_history(history_path)
        client.close()
    return 0


__all__ = [
    "CliClient",
    "_EventDispatcher",
    "_handle_client_slash",
    "_sock_path",
    "_submit_task",
    "_wait_for_socket",
    "run_client",
]
