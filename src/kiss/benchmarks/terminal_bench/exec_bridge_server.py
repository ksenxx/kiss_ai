# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Local HTTP server that runs commands on a Harbor environment for the exec bridge.

The daemon-side client is :mod:`kiss.agents.sorcar.exec_bridge`; this is
the server the Harbor agent starts for a trial whose environment is not a
local Docker container (Harbor's ``modal`` environment for the GPU tasks).
Each ``POST /exec`` runs ``environment.exec(command, timeout_sec)`` on the
agent's event loop and answers with the exit code and the combined output.
The command's process tree is tagged with the request's token
(:func:`kiss.agents.sorcar.docker_manager.tagged_command`) so a command
still running at the deadline is killed on the environment before the
``null`` exit code is answered, and so the client can kill a command it
no longer waits for (a task Stop) with ``POST /kill``.

The server listens on the loopback interface only; nothing but the local
daemon can reach the sandbox through it.

When a command fails because the environment itself is gone (the provider
shut the sandbox down) and a probe fails too, the server records the error
in :attr:`ExecBridgeServer.lost`; the Harbor agent polls it, confirms it
with a later probe (:meth:`ExecBridgeServer.confirm_lost`, so a passing
provider error does not end a healthy trial) and ends the trial with an
error (re-runnable with ``harbor jobs resume -f``) instead of leaving the
agent to work around a dead sandbox.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from kiss.agents.sorcar.docker_manager import kill_tagged_tree_script, tagged_command

#: Name of the thread serving a bridge; a closed bridge leaves no thread of this name.
SERVER_THREAD_NAME = "exec-bridge-server"
#: Seconds between the reaper's sweeps of a cancelled command that is still pending.
REAP_INTERVAL_S = 0.5
#: Reply to an ``/exec`` whose token was killed before the command started.
STOPPED_REPLY: dict[str, Any] = {"exit_code": -1, "output": ""}
#: Seconds the ``true`` probe of :meth:`ExecBridgeServer.alive` may take.
PROBE_TIMEOUT_S = 15
#: Seconds :meth:`ExecBridgeServer.confirm_lost` waits before it probes a sandbox again.
LOST_CONFIRM_S = 10.0


async def kill_tagged(environment: Any, token: str) -> None:
    """Kill the process tree tagged with *token* on *environment*; never raises."""
    try:
        await asyncio.wait_for(
            environment.exec(kill_tagged_tree_script(token), timeout_sec=30), timeout=60,
        )
    except Exception:  # the sandbox is gone: nothing left to kill
        return None


async def exec_once(
    environment: Any, command: str, timeout: float, token: str = "",
) -> dict[str, Any]:
    """Run *command* on *environment* and return the bridge's reply dict.

    *environment* needs one method, Harbor's ``async exec(command,
    timeout_sec=...)`` returning an object with ``stdout``, ``stderr`` and
    ``return_code``.

    The command runs tagged with *token*; a timeout (``asyncio.TimeoutError``
    or a provider error naming the deadline) kills the tagged tree and is
    reported as a ``null`` exit code with whatever output is available
    (none, for most providers).

    Args:
        environment: Harbor's running environment.
        command: The shell command (``sh -c`` semantics on the environment).
        timeout: Seconds the command may run.
        token: The tag for the command's process tree (a fresh one when empty).
    """
    token = token or uuid.uuid4().hex
    try:
        result = await asyncio.wait_for(
            environment.exec(tagged_command(command, token), timeout_sec=max(1, int(timeout))),
            timeout=timeout + 30,
        )
    except TimeoutError:
        await kill_tagged(environment, token)
        return {"exit_code": None, "output": ""}
    except Exception as error:  # the sandbox is gone or the provider failed
        return {"exit_code": 255, "output": f"exec failed: {type(error).__name__}: {error}"}
    output = (result.stdout or "") + (result.stderr or "")
    return {"exit_code": int(result.return_code), "output": output}


class ExecBridgeServer:
    """Serves one environment's ``exec`` over ``http://127.0.0.1:<port>``.

    Built on the running event loop that owns the environment (Harbor's
    agent loop); the HTTP handlers run on server threads and hand each
    command to that loop with ``run_coroutine_threadsafe``.

    A ``/kill`` may arrive while the provider is still launching the
    command (or before its ``/exec`` request has even been read), when a
    single sweep of the tagged tree finds nothing.  The token is therefore
    remembered in :attr:`cancelled` (an ``/exec`` that arrives later for it
    never starts) and, while its exec is still pending, a reaper thread
    sweeps again every :data:`REAP_INTERVAL_S` until the exec ends, as
    :class:`kiss.agents.sorcar.docker_manager.DockerManager` does for a
    start the Docker daemon delayed past the deadline.
    """

    def __init__(self, environment: Any, loop: asyncio.AbstractEventLoop) -> None:
        """Start serving *environment* on a free loopback port.

        Args:
            environment: The environment whose ``exec`` the bridge exposes.
            loop: The event loop ``environment.exec`` must run on.
        """
        self.environment = environment
        self.loop = loop
        self.lock = threading.Lock()
        self.pending: dict[str, Future[dict[str, Any]]] = {}
        self.cancelled: set[str] = set()
        #: The error of the exec that found the environment gone (a sandbox the
        #: provider shut down; :meth:`alive` failed right after), ``""`` while
        #: it answers; cleared by a later command or probe that succeeds.
        self.lost = ""
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                self._reply({"ok": True} if self.path == "/ping" else {"error": "not found"},
                            200 if self.path == "/ping" else 404)

            def do_POST(self) -> None:  # noqa: N802
                if self.path not in ("/exec", "/kill"):
                    self._reply({"error": "not found"}, 404)
                    return
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8", errors="replace"))
                    token = str(body.get("token") or uuid.uuid4().hex)
                    if self.path == "/kill":
                        bridge.kill(token)
                        self._reply({"ok": True}, 200)
                        return
                    command, timeout = str(body["command"]), float(body.get("timeout", 30))
                except (ValueError, KeyError, TypeError):
                    self._reply({"error": "bad request"}, 400)
                    return
                self._reply(bridge.run(command, timeout, token), 200)

            def _reply(self, payload: dict[str, Any], status: int) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever, name=SERVER_THREAD_NAME, daemon=True,
        )
        self.thread.start()

    @property
    def url(self) -> str:
        """The bridge's base URL (``http://127.0.0.1:<port>``)."""
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def run(self, command: str, timeout: float, token: str) -> dict[str, Any]:
        """Run *command* tagged with *token* from a server thread and return the reply dict.

        A token already killed gets :data:`STOPPED_REPLY` without running.
        """
        with self.lock:
            if token in self.cancelled:
                self.cancelled.discard(token)
                return STOPPED_REPLY
            future = asyncio.run_coroutine_threadsafe(
                exec_once(self.environment, command, timeout, token), self.loop)
            self.pending[token] = future
        try:
            reply = future.result(timeout=timeout + 120)
        except TimeoutError:
            future.cancel()
            self.sweep(token)
            return {"exit_code": None, "output": ""}
        finally:
            with self.lock:
                self.pending.pop(token, None)
                self.cancelled.discard(token)
        if reply["exit_code"] == 255:
            # The environment itself is gone (the provider shut the sandbox
            # down): every later command would fail the same way, so the trial
            # runner ends the trial instead of letting the agent retry for hours.
            if not self.lost and not self.alive():
                self.lost = str(reply["output"])
        elif isinstance(reply["exit_code"], int) and reply["exit_code"] >= 0:
            self.lost = ""  # the command ran: an earlier failure was a passing one
        return reply

    async def confirm_lost(self) -> bool:
        """Whether the environment recorded as lost is still gone after :data:`LOST_CONFIRM_S`.

        Runs on the environment's loop (the trial runner's side, not the
        server thread).  A probe that succeeds clears :attr:`lost`: the earlier
        failures were a passing provider error, and the trial goes on.
        """
        await asyncio.sleep(LOST_CONFIRM_S)
        if not self.lost:
            return False
        if (await exec_once(self.environment, "true", PROBE_TIMEOUT_S))["exit_code"] == 0:
            self.lost = ""
            return False
        return True

    def alive(self) -> bool:
        """Whether the environment still runs commands (a ``true`` exec succeeds).

        The probe runs at most :data:`PROBE_TIMEOUT_S` seconds (plus the
        provider's grace in :func:`exec_once`), well within the slack the
        daemon-side client allows a reply.
        """
        future = asyncio.run_coroutine_threadsafe(
            exec_once(self.environment, "true", PROBE_TIMEOUT_S), self.loop)
        try:
            return bool(future.result(timeout=PROBE_TIMEOUT_S + 60)["exit_code"] == 0)
        except TimeoutError:
            future.cancel()
            return False

    def sweep(self, token: str) -> None:
        """Kill the process tree tagged with *token* on the environment, once."""
        future = asyncio.run_coroutine_threadsafe(kill_tagged(self.environment, token), self.loop)
        try:
            future.result(timeout=90)
        except TimeoutError:
            future.cancel()

    def kill(self, token: str) -> None:
        """Cancel the command tagged with *token* (``POST /kill``).

        Marks the token cancelled, sweeps its tree now and, while its exec is
        still pending, keeps sweeping on a reaper thread so a launch the
        provider completes after the first sweep is killed as well.
        """
        with self.lock:
            self.cancelled.add(token)
            future = self.pending.get(token)
        self.sweep(token)
        if future is not None and not future.done():
            threading.Thread(target=self.reap, args=(token, future), daemon=True,
                             name=f"exec-bridge-reaper-{token[:8]}").start()

    def reap(self, token: str, future: Future[dict[str, Any]]) -> None:
        """Reaper thread body: sweep *token*'s tree every half second until *future* ends."""
        while not future.done():
            time.sleep(REAP_INTERVAL_S)
            self.sweep(token)

    def close(self) -> None:
        """Stop serving; commands still running on the environment are not interrupted."""
        self.server.shutdown()
        self.server.server_close()
