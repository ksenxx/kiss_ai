# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Expose KISS tools to a ``codex exec`` run as an MCP stdio server.

Codex only takes external tools as MCP servers that it spawns itself, so
the KISS tools (which live in the KISS process, bound to the running
agent) reach it through two halves:

* :class:`ToolBridge` — the in-process half.  It listens on an ephemeral
  ``127.0.0.1`` port for one-line JSON requests and answers them from the
  agent's tool schema and tool executor.  :meth:`ToolBridge.codex_args`
  are the ``-c mcp_servers.kiss.*`` overrides that tell Codex how to
  start the proxy.
* :func:`main` — the proxy half, run by Codex as
  ``python -m kiss.core.models.codex_mcp_bridge <port>``.  It is a plain
  MCP stdio server (``mcp.server.lowlevel.Server``, which takes the JSON
  schemas KISS already builds; ``FastMCP`` only derives schemas from
  Python signatures) whose ``list_tools`` and ``call_tool`` forward to
  the bridge.

Wire protocol (one request per TCP connection, newline-terminated JSON):

* ``{"op": "tools"}`` → ``{"tools": [{"name", "description",
  "parameters"}, ...]}``
* ``{"op": "call", "name": ..., "arguments": {...}}`` → ``{"text": ...}``
  or ``{"error": ...}`` when the executor itself raised.  A tool that
  failed in the ordinary way is not an ``error``: the executor returns
  the failure text as the tool's result, which is what every KISS model
  sees (the agent loop never hands a model an error flag either).

Every request also carries ``"token"``, the bridge's secret: any local
user can connect to a loopback port, so a request without it is refused.
The proxy gets the secret from its environment (:data:`BRIDGE_TOKEN_ENV`,
which Codex forwards to it, see :meth:`ToolBridge.codex_env`), never
from its command line, which other users can read.

A loopback port is used instead of a Unix socket because the bridge must
also run on Windows and macOS caps Unix socket paths at 104 bytes.  Tool
executions are serialized with a lock: the KISS tool loop runs one tool
at a time, and the agent's tool executor assumes that.  Leaving the
``with`` block waits for calls still executing, so no tool keeps working
unobserved after the run has returned; under a Stop the calls are
interrupted and the wait is bounded (:meth:`ToolBridge._wait_for_calls`).
"""

import asyncio
import hmac
import json
import logging
import os
import secrets
import socket
import socketserver
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, NamedTuple, cast

from kiss.core import stop_signal, tool_interrupt
from kiss.core.locked import locked_call

logger = logging.getLogger(__name__)

KISS_MCP_SERVER_NAME = "kiss"

# The proxy's environment variable holding the bridge's secret.
BRIDGE_TOKEN_ENV = "KISS_MCP_BRIDGE_TOKEN"

# How long a Stop waits for the interrupted calls before abandoning them:
# the interrupt's grace period, the injection's delivery, and some slack.
STOP_WAIT_SECONDS = 5.0

ToolExecutor = Callable[[dict[str, Any]], tuple[str, str]]

# What a Stop raises in the task thread: the daemon's forced Stop (or a
# Ctrl+C), and the per-panel Stop of the tool call an inline agent runs as.
_STOPS = (KeyboardInterrupt, tool_interrupt.ToolCallInterrupted)

# The reply of a call the bridge refused because it is closing.
_REFUSED = {"error": "KISSError: the KISS tool bridge is closed"}

# The reply to a request without the bridge's secret.
_UNAUTHENTICATED = {"error": "KISSError: unauthenticated KISS tool bridge request"}

#: Longest request line a handler reads (a tool call's arguments, as one
#: JSON line); a longer one is dropped.
_MAX_REQUEST_BYTES = 64 * 1024 * 1024


def _read_request_line(sock: socket.socket, deadline: float) -> bytes:
    """Read one line from *sock* by ``time.monotonic()`` *deadline*; ``b""`` without one.

    A plain ``readline`` is bounded per read, which a client sending a
    byte within the socket's timeout every time resets for good; this
    read is bounded as a whole and in size (:data:`_MAX_REQUEST_BYTES`).

    Args:
        sock: The accepted connection.
        deadline: When to give up.

    Returns:
        The line with its newline, or ``b""`` when the client hung up or
        went past the size cap or, between two receives, the deadline.

    Raises:
        TimeoutError: A receive got no byte by the deadline: the handler
            fails and its socket is closed (:meth:`_BridgeServer.handle_error`).
    """
    timeout = sock.gettimeout()
    line = bytearray()
    try:
        while not line.endswith(b"\n"):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or len(line) >= _MAX_REQUEST_BYTES:
                return b""
            sock.settimeout(remaining)
            # Never past the cap: a line crossing it inside a receive
            # would otherwise be accepted whole.
            chunk = sock.recv(min(65536, _MAX_REQUEST_BYTES - len(line)))
            if not chunk:
                return b""
            line += chunk
    finally:
        sock.settimeout(timeout)  # the reply's write keeps its own bound
    return bytes(line)


class ServedCall(NamedTuple):
    """One tool call the bridge answered: what Codex asked and what it got back."""

    name: str
    arguments: dict[str, Any]
    result: str


def toml_value(value: str | list[str]) -> str:
    """Return *value* as a TOML basic string or array of them, for ``codex -c key=value``.

    JSON string syntax is TOML string syntax except for two details:
    JSON's default ``\\uXXXX`` escaping of non-ASCII text writes non-BMP
    characters (emoji) as surrogate pairs, which TOML rejects, and TOML
    forbids a raw DEL character.  So non-ASCII is written raw (TOML is
    UTF-8) and DEL is escaped.

    Args:
        value: A string, or a list of strings.

    Returns:
        The TOML literal.
    """
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007F")


def _send_line(sock: socket.socket, payload: dict[str, Any]) -> None:
    """Write *payload* as one newline-terminated JSON line to *sock*."""
    sock.sendall(json.dumps(payload).encode("utf-8") + b"\n")


class _BridgeServer(socketserver.ThreadingTCPServer):
    """Loopback TCP server that answers tool-list and tool-call requests.

    Handler threads are daemon threads the server does not join on
    close: :class:`ToolBridge` tracks the ones executing a call itself
    (``running``) so it can interrupt them on a Stop and bound the wait.
    """

    daemon_threads = True
    block_on_close = False

    def __init__(self, tools_schema: list[dict[str, Any]], execute: ToolExecutor):
        super().__init__(("127.0.0.1", 0), _BridgeHandler)
        self.token = secrets.token_hex(32)
        # The ``{"name", "description", "parameters"}`` of each tool, out of
        # its OpenAI ``{"type": "function", "function": {...}}`` wrapper.
        self.tools = [tool["function"] for tool in tools_schema]
        self.tool_names = frozenset(tool["name"] for tool in self.tools)
        self.execute = execute
        self.execute_lock = threading.Lock()
        # The handler threads executing a call (by ident), every call
        # answered so far and the first executor exception, all guarded
        # by ``state_lock``: handler threads update them while the task
        # thread reads them.  An RLock taken through ``locked_call`` on
        # the task thread, which a Stop can interrupt at any check
        # point; the handler threads, interrupted only inside the
        # executor (``tool_interrupt``), take it plainly.
        self.state_lock = threading.RLock()
        self.running: dict[int, threading.Thread] = {}
        self.served: list[ServedCall] = []
        self.error: Exception | None = None
        self.closing = False

    def serve_call(self, call: dict[str, Any]) -> dict[str, Any]:
        """Execute *call* and return the reply line, recording the call.

        Args:
            call: ``{"name": ..., "arguments": {...}}``.

        Returns:
            ``{"text": result}`` or ``{"error": message}`` when the
            executor itself raised, or when the bridge is closing: a
            call that has not started executing when the block ends
            (its request still being read, or queued behind another
            tool) is refused rather than run after the turn.
        """
        ident = threading.get_ident()
        with self.state_lock:
            if self.closing:
                return _REFUSED
            self.running[ident] = threading.current_thread()
        try:
            # Recorded under the execution lock, so the record's order is
            # the execution order.
            with self.execute_lock:
                # Re-checked once the call is first in line: a Stop that
                # ended the block while this call waited behind another
                # tool interrupted only that tool (this one had nothing
                # to interrupt yet) and must not be followed by a tool
                # starting after the bridge has returned.
                with self.state_lock:
                    closing = self.closing
                if closing:
                    return _REFUSED
                error: Exception | None = None
                try:
                    _, result = self.execute(call)
                    reply = {"text": result}
                except Exception as e:
                    error = e
                    result = f"{type(e).__name__}: {e}"
                    reply = {"error": result}
                with self.state_lock:
                    self.served.append(ServedCall(call["name"], call["arguments"], result))
                    if error is not None and self.error is None:
                        self.error = error
        finally:
            with self.state_lock:
                del self.running[ident]
        return reply

    def refuse_new_calls(self) -> None:
        """Refuse every call not yet executing its tool; call under ``state_lock``."""
        self.closing = True

    def running_handlers(self) -> list[tuple[int, threading.Thread]]:
        """The ``(ident, thread)`` of every handler executing a call; call under ``state_lock``."""
        return list(self.running.items())

    def take_served(self) -> list[ServedCall]:
        """Return the calls answered so far and forget them; call under ``state_lock``."""
        served, self.served = self.served, []
        return served

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Log a handler failure instead of printing a traceback to stderr.

        The usual cause is a proxy that went away (Codex exited) before
        the reply was written, or a connection that never sent its
        request line within :data:`REQUEST_TIMEOUT_SECONDS`.

        Args:
            request: The connection the handler was serving.
            client_address: The proxy's address.
        """
        logger.debug("KISS MCP bridge request failed", exc_info=True)


# Bound on the whole read of a request line and on each write
# (``StreamRequestHandler`` sets ``timeout`` on the accepted socket).  The
# proxy sends its request line the moment it connects and reads the reply
# at once, so a connection that has not finished its line this long is a
# client that never will (silent, or sending a byte now and then): the
# handler fails and its socket is closed instead of waiting for good, since
# ``ToolBridge._close`` joins only the handlers executing a call.  The
# tool's execution itself is no socket I/O and is not bounded by this.
REQUEST_TIMEOUT_SECONDS = 5.0


class _BridgeHandler(socketserver.StreamRequestHandler):
    """Handle one request: read a JSON line, answer with a JSON line."""

    timeout = REQUEST_TIMEOUT_SECONDS

    def handle(self) -> None:
        """Dispatch the request to the tool list or the tool executor."""
        server = cast(_BridgeServer, self.server)
        line = _read_request_line(self.connection, time.monotonic() + REQUEST_TIMEOUT_SECONDS)
        if not line:  # the client connected and hung up without asking
            return
        request = json.loads(line.decode("utf-8"))
        token = str(request.get("token", "")).encode("utf-8")
        if not hmac.compare_digest(token, server.token.encode("utf-8")):
            _send_line(self.connection, _UNAUTHENTICATED)
            return
        if request.get("op") == "tools":
            _send_line(self.connection, {"tools": server.tools})
            return
        name = request["name"]
        if name not in server.tool_names:
            # Only the listed tools are served: ``finish`` and the
            # between-turns tools are kept off the bridge on purpose.
            _send_line(self.connection, {"error": f"KISSError: unknown tool {name!r}"})
            return
        call = {"name": name, "arguments": request.get("arguments") or {}}
        _send_line(self.connection, server.serve_call(call))


class ToolBridge:
    """Serve the agent's tools to the MCP proxy for the duration of a ``with`` block.

    Args:
        tools_schema: The tools in OpenAI function-calling layout (what
            ``Model._build_openai_tools_schema`` returns); ``name``,
            ``description`` and ``parameters`` become the MCP tool's
            name, description and input schema.
        execute: Called with ``{"name": ..., "arguments": {...}}`` for
            every tool call and returns ``(name, result_text)`` —
            ``KISSAgent._execute_tool``, so each call gets the usual
            printer events and interrupt handling.  An exception it
            raises is fatal to the run (a limit hit, a sub-agent's
            budget exceeded): it is reported to Codex as an ``error``
            and kept as :attr:`error`, so the model can end the turn
            instead of waiting for Codex to talk past it.
    """

    def __init__(self, tools_schema: list[dict[str, Any]], execute: ToolExecutor):
        self._server = _BridgeServer(tools_schema, execute)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="kiss-mcp-bridge", daemon=True
        )
        # A Stop that ended the block or landed in :meth:`_close` (re-raised
        # once it is done), and the bounded wait's deadline: on the instance
        # so that they survive a step's retry (see :meth:`_close`).
        self._forced_stop: BaseException | None = None
        self._stop_deadline: float | None = None

    @property
    def port(self) -> int:
        """The loopback port the bridge listens on."""
        return int(self._server.server_address[1])

    @property
    def tool_names(self) -> frozenset[str]:
        """The names of the tools the bridge serves."""
        return self._server.tool_names

    @property
    def calls_in_flight(self) -> int:
        """How many tool calls the bridge has received but not yet answered."""
        return len(locked_call(self._server.state_lock, self._server.running_handlers))

    @property
    def error(self) -> Exception | None:
        """The first exception the executor raised, fatal to the run; ``None`` so far."""
        return self._server.error

    def take_served(self) -> list[ServedCall]:
        """Return the calls answered so far, in order, and forget them."""
        return locked_call(self._server.state_lock, self._server.take_served)

    def codex_args(self, tool_timeout_seconds: int) -> list[str]:
        """Return the ``codex exec`` overrides that register the proxy as an MCP server.

        Args:
            tool_timeout_seconds: How long Codex waits for one tool call
                (its default of 60 s is far too short for KISS tools
                such as ``Bash`` or ``run_parallel``).

        Returns:
            ``-c key=value`` pairs for ``mcp_servers.kiss``; the values
            are TOML literals (:func:`toml_value`).
            ``default_tools_approval_mode="approve"`` pre-approves every
            call: ``codex exec`` runs with ``approval_policy=never`` and
            would otherwise refuse the call instead of asking.
            ``env_vars`` forwards :data:`BRIDGE_TOKEN_ENV` from Codex's
            environment (:meth:`codex_env`) to the proxy, whose
            environment Codex otherwise clears.
        """
        prefix = f"mcp_servers.{KISS_MCP_SERVER_NAME}"
        return [
            "-c",
            f"{prefix}.command={toml_value(sys.executable)}",
            "-c",
            f"{prefix}.args={toml_value(['-m', __name__, str(self.port)])}",
            "-c",
            f"{prefix}.env_vars={toml_value([BRIDGE_TOKEN_ENV])}",
            "-c",
            f"{prefix}.tool_timeout_sec={int(tool_timeout_seconds)}",
            "-c",
            f"{prefix}.default_tools_approval_mode={toml_value('approve')}",
        ]

    def codex_env(self) -> dict[str, str]:
        """Return the variables Codex's environment must hold for the proxy.

        Returns:
            :data:`BRIDGE_TOKEN_ENV` set to the bridge's secret, which
            every request must carry.
        """
        return {BRIDGE_TOKEN_ENV: self._server.token}

    def __enter__(self) -> "ToolBridge":
        try:
            self._thread.start()
        except BaseException as e:
            # A task-thread Stop (``KeyboardInterrupt`` injected with
            # ``PyThreadState_SetAsyncExc``) lands at the next check
            # point, which can be inside ``start()`` after the OS
            # thread was spawned: ``__exit__`` never runs for an
            # ``__enter__`` that raised, so the listener would serve
            # for good.  Close it on the way out.
            self._close(e)
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._close(exc_info[1])

    def _close(self, leaving: object) -> None:
        """Refuse late calls, stop the listener, then wait for the calls still executing.

        *leaving* is the exception ending the block, if any.  A Stop among
        them (a per-panel Stop of the inline agent, a Ctrl+C: neither
        sets the task's stop event) bounds the wait like one landing
        here; Python re-raises it itself once this returns.

        The refusal comes first: the listener takes up to its poll
        interval to stop, and a call queued behind a tool that returns
        meanwhile must not start its own tool after the block.  A
        handler accepted before the listener stopped may still be
        reading its request line; it is not in ``running`` yet, so the
        refusal, set under the same lock the registration takes, is
        what keeps it from running a tool after the block.

        A Stop (an injected ``KeyboardInterrupt`` or
        ``ToolCallInterrupted``) landing while the
        refusal is set, the listener shuts down or the wait begins must
        still interrupt and await the calls already executing, or a
        long tool would keep working on its daemon thread after the run
        returned.  A ``finally`` is not enough: the exception lands at
        any bytecode, inside the ``finally`` too, and skips the rest of
        it.  So each step is retried until it completes (all four are
        idempotent: the wait keeps its deadline on the instance) and the
        exception, which makes the wait a bounded one, is re-raised once
        every step is done.
        """
        steps = [
            self._refuse_new_calls,
            self._stop_listener,
            self._server.server_close,
            self._wait_for_calls,
        ]
        while steps:
            try:
                if self._forced_stop is None and isinstance(leaving, _STOPS):
                    self._forced_stop = leaving
                steps[0]()
                del steps[0]
            except _STOPS as e:
                self._forced_stop = e
        if self._forced_stop is not None and self._forced_stop is not leaving:
            raise self._forced_stop

    def _stop_listener(self) -> None:
        if _spawned(self._thread):
            self._server.shutdown()
            self._thread.join()

    def _refuse_new_calls(self) -> None:
        locked_call(self._server.state_lock, self._server.refuse_new_calls)

    def _wait_for_calls(self) -> None:
        """Wait for the calls still executing; a Stop interrupts them and bounds the wait.

        Without a Stop the wait is open-ended: the tool's work (a file
        write, a sub-agent) is better finished than cut, and nothing
        keeps working unobserved after the run has returned.  Once the
        task's stop is requested, before or during the wait, each call
        is interrupted the way the per-panel Stop interrupts one
        (:func:`~kiss.core.tool_interrupt.interrupt_tool_call`: the
        cooperative event at once, an injected exception after the
        grace period) and the wait ends after :data:`STOP_WAIT_SECONDS`
        at the latest; a tool that survives even the injection is left
        on its daemon thread.  A call not inside its tool yet (queued
        behind another, or about to start) has nothing to interrupt, so
        the interrupt is retried on every poll; and since :meth:`_close`
        marked the bridge closing first, a queued call is refused the
        moment it is first in line (:meth:`_BridgeServer.serve_call`).

        A ``KeyboardInterrupt`` or ``ToolCallInterrupted`` raised in this
        thread meanwhile, or ending the block (the daemon's forced Stop,
        injected a second after the stop event; a Ctrl+C; the per-panel
        Stop of an inline agent) is a Stop too: :meth:`_close` holds it as
        ``_forced_stop`` and retries the wait, now a bounded one, so it
        cannot cut the wait short and leave an interrupted tool working
        after the run has returned.
        """
        while True:
            running = locked_call(self._server.state_lock, self._server.running_handlers)
            if not running:
                break
            if self._stop_deadline is None and (
                self._forced_stop is not None or stop_signal.stop_requested()
            ):
                self._stop_deadline = time.monotonic() + STOP_WAIT_SECONDS
            if self._stop_deadline is not None:
                for ident, _thread in running:  # idempotent on an interrupted call
                    tool_interrupt.interrupt_tool_call(ident)
                if time.monotonic() >= self._stop_deadline:
                    logger.warning(
                        "KISS MCP bridge: %d tool call(s) still running after the Stop; abandoned",
                        len(running),
                    )
                    break
            running[0][1].join(0.1)


def _spawned(thread: threading.Thread) -> bool:
    """Whether *thread*'s OS thread exists.

    A ``Thread.start()`` that raised may have spawned it (an injected
    Stop landing after the spawn) or not (thread exhaustion, or the Stop
    landing before); a spawned thread sets its ident within moments.
    ``BaseServer.shutdown()`` would wait for good on a ``serve_forever``
    that never ran, so the distinction matters.
    """
    deadline = time.monotonic() + 1.0
    while thread.ident is None and time.monotonic() < deadline:
        time.sleep(0.001)
    return thread.ident is not None


# ---------------------------------------------------------------------------
# Proxy half: the MCP stdio server Codex spawns.
# ---------------------------------------------------------------------------


def _bridge_request(port: int, token: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send one request with the secret *token* to the bridge on *port*; return its reply."""
    with socket.create_connection(("127.0.0.1", port)) as sock:
        _send_line(sock, {**payload, "token": token})
        with sock.makefile("rb") as reply:
            return dict(json.loads(reply.readline().decode("utf-8")))


async def _serve_stdio(port: int, token: str) -> None:
    """Run the MCP stdio server, forwarding every request to the bridge on *port*."""
    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    server: Server[Any, Any] = Server(KISS_MCP_SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        reply = await anyio.to_thread.run_sync(_bridge_request, port, token, {"op": "tools"})
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return [
            types.Tool(
                name=tool["name"],
                description=tool.get("description", ""),
                inputSchema=tool["parameters"],
            )
            for tool in reply["tools"]
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        payload = {"op": "call", "name": name, "arguments": arguments}
        reply = await anyio.to_thread.run_sync(_bridge_request, port, token, payload)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return [types.TextContent(type="text", text=reply["text"])]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: list[str] | None = None) -> None:
    """Entry point of the proxy: ``python -m kiss.core.models.codex_mcp_bridge <port>``.

    The bridge's secret is read from :data:`BRIDGE_TOKEN_ENV`.

    Args:
        argv: Command-line arguments without the program name; defaults
            to ``sys.argv[1:]``.  The only argument is the bridge port.
    """
    args = sys.argv[1:] if argv is None else argv
    asyncio.run(_serve_stdio(int(args[0]), os.environ.get(BRIDGE_TOKEN_ENV, "")))


if __name__ == "__main__":
    main()
