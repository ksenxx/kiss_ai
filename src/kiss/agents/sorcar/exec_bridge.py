# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Exec bridge: the agent's shell and file tools on an environment that is not a local container.

A :class:`DockerManager` runs the tools of a Docker-mode task with ``docker
exec``.  A trial on a remote sandbox (a Modal GPU box behind Harbor's
``modal`` environment) has no local container, only a ``run this command``
primitive held by the process that created the sandbox.  That process
serves the primitive over a local HTTP bridge (see
:mod:`kiss.benchmarks.terminal_bench.exec_bridge_server`) and the task's
``docker_image`` names it as ``exec:http://127.0.0.1:<port>``; this
module is the client, a drop-in for the manager's ``Bash`` and
``run_commands_parallel`` (the file tools are built on ``Bash``, see
:class:`kiss.agents.sorcar.docker_tools.DockerTools`).

Wire protocol, one JSON object per request and response::

    POST /exec  {"command": "<sh -c command>", "timeout": <seconds>, "token": "<hex>"}
             -> {"exit_code": <int or null when the deadline passed>, "output": "<stdout+stderr>"}
    POST /kill  {"token": "<hex>"}  -> {"ok": true}   (kills that command's process tree)
    GET  /ping  -> {"ok": true}
"""

from __future__ import annotations

import functools
import http.client
import json
import shlex
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from typing import Any

from kiss.agents.sorcar.docker_manager import (
    _FETCH_SCRIPT,
    _KILL_WAIT_S,
    MAX_OUTPUT_CHARS,
    STOPPED_MESSAGE,
    PendingAppend,
    _refuse_or_lift,
    _timeout_message,
    _with_exit_code,
    fetched_bytes,
)
from kiss.agents.sorcar.useful_tools import (
    _MAX_BINARY_READ_BYTES,
    _output_budget,
    _stop_requested,
    _truncate_output,
    run_commands_pool,
)
from kiss.core import tool_interrupt
from kiss.core.kiss_error import KISSError

#: ``docker_image`` prefix that selects the bridge instead of a Docker container.
BRIDGE_PREFIX = "exec:"

#: What ``urlopen`` + ``read`` + ``json.loads`` raise for an unreachable bridge
#: or a malformed reply.  ``http.client.HTTPException`` (``BadStatusLine``,
#: ``IncompleteRead`` when the bridge dies mid-reply) derives from
#: ``Exception``, not ``OSError``, and ``urlopen`` does not wrap it.
_BRIDGE_ERRORS = (urllib.error.URLError, OSError, ValueError, http.client.HTTPException)


def _post(url: str, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST *body* as JSON to *url* + *path* and return the decoded reply dict."""
    request = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            reply = json.loads(response.read().decode("utf-8", errors="replace"))
    except _BRIDGE_ERRORS as error:
        raise KISSError(f"Exec bridge {url} unreachable: {error}") from error
    if not isinstance(reply, dict):
        raise KISSError(f"Exec bridge {url} returned an invalid reply: {reply!r}")
    return reply


def _kill_quietly(url: str, token: str) -> None:
    """POST ``/kill`` for *token*, swallowing an unreachable bridge (helper-thread body)."""
    try:
        _post(url, "/kill", {"token": token}, 120)
    except KISSError:
        pass


def _kill_bounded(url: str, token: str) -> None:
    """Post ``/kill`` for *token* from a helper thread; wait at most :data:`_KILL_WAIT_S` for it.

    The bridge server sweeps the remote sandbox before it replies, and a
    stalled sandbox must not hold the task thread for the request's
    whole socket timeout: the kill goes on with the helper thread.
    """
    killer = threading.Thread(
        target=_kill_quietly, args=(url, token),
        name=f"exec-bridge-kill-{token[:8]}", daemon=True,
    )
    killer.start()
    killer.join(_KILL_WAIT_S)


def bridge_exec(
    url: str, command: str, timeout: float, cancel: threading.Event | None = None,
) -> tuple[int | None, str]:
    """Run *command* through the bridge at *url*.

    The request is made on a helper thread while this thread polls
    *cancel* and the calling tool call's interrupt event in 0.2 s slices
    (as the Docker tools do, see
    :func:`kiss.agents.sorcar.useful_tools._stop_requested`); a Stop
    kills the command's process tree on the environment through
    ``POST /kill`` and returns at once.  The kill is posted from a helper
    thread and waited for at most :data:`_KILL_WAIT_S`, as
    ``DockerManager`` does (:func:`_kill_bounded`).  An exception thrown
    into this thread's wait (a ``KeyboardInterrupt``, the runner's Stop
    injected as one) posts the same kill before it propagates, so the
    command never outlives the call.

    Args:
        url: The bridge's base URL (``http://127.0.0.1:<port>``).
        command: The shell command (run with ``sh -c``).
        timeout: Seconds after which the bridge gives up on the command.
        cancel: Optional event that ends the wait early (the task's stop
            event or a ``run_commands_parallel`` fan-out's).

    Returns:
        ``(exit code, combined output)``; the exit code is ``None`` when the
        command was still running at the deadline and ``-1`` when it was
        stopped (the output then ends with :data:`STOPPED_MESSAGE`).

    Raises:
        KISSError: When the bridge cannot be reached or answers garbage.
    """
    token = uuid.uuid4().hex
    body = {"command": command, "timeout": timeout, "token": token}
    outcome: list[dict[str, Any] | KISSError] = []

    def fetch() -> None:
        try:
            outcome.append(_post(url, "/exec", body, timeout + 60))
        except KISSError as error:
            outcome.append(error)

    worker = threading.Thread(target=fetch, name=f"exec-bridge-{token[:8]}", daemon=True)
    try:
        worker.start()
        while worker.is_alive():
            if _stop_requested(cancel):
                _kill_bounded(url, token)
                return -1, STOPPED_MESSAGE
            worker.join(0.2)
    except BaseException:
        # The wait was unwound from outside (an injected Stop, a
        # KeyboardInterrupt): the request thread and the command on the
        # environment must not go on without anyone waiting for them.
        _kill_bounded(url, token)
        raise
    reply = outcome[0]
    if isinstance(reply, KISSError):
        raise reply
    if "output" not in reply:
        raise KISSError(f"Exec bridge {url} returned an invalid reply: {reply!r}")
    code = reply.get("exit_code")
    return (int(code) if code is not None else None), str(reply["output"])


def bridge_alive(url: str) -> bool:
    """Whether the bridge at *url* answers ``GET /ping``."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/ping", timeout=10) as response:
            return bool(json.loads(response.read()).get("ok"))
    except _BRIDGE_ERRORS:
        return False


class ExecBridge:
    """The agent's shell tools on a remote environment, through a local HTTP bridge.

    Mirrors the parts of :class:`kiss.agents.sorcar.docker_manager.DockerManager`
    the agent uses: ``Bash``, ``run_commands_parallel`` (both with the
    Docker tools' shell guards), ``stop_event``, ``stream_callback``
    (output is not streamed: the bridge returns it whole), ``volumes``
    (none), ``workdir`` (the environment's working directory, asked for
    on entry) and the context-manager protocol (nothing to open or close:
    the sandbox belongs to the server).  It has no ``container``: a child
    of a bridged task inherits the bridge by its ``url``
    (``agent_dispatch``), never a container.
    """

    def __init__(self, docker_image: str, **_: Any) -> None:
        """Bind to the bridge named by *docker_image* (``exec:<url>``).

        Args:
            docker_image: ``exec:http://127.0.0.1:<port>``.
            _: The :class:`DockerManager` keyword arguments (``workdir``,
                ``volumes``), accepted and ignored so the two are built alike.
        """
        self.url = docker_image[len(BRIDGE_PREFIX):]
        self.stop_event: threading.Event | None = None
        self.stream_callback: Callable[[str], None] | None = None
        self.volumes: dict[str, str] = {}
        self.workdir: str | None = None
        # The append a ``>>`` copy ``Bash`` redirected stands for, until
        # ``DockerTools.Write`` settles it against the file's current bytes.
        self.pending_append: PendingAppend | None = None

    def __enter__(self) -> ExecBridge:
        if not bridge_alive(self.url):
            raise KISSError(f"Exec bridge {self.url} is not answering.")
        # Where the environment starts every command: the here-document
        # guard resolves relative targets against it (unknown: refused).
        cwd = bridge_exec(self.url, "pwd", 30)[1].strip()
        self.workdir = cwd if cwd.startswith("/") else None
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def Bash(  # noqa: N802
        self,
        command: str,
        description: str,
        timeout_seconds: int = 30,
        max_output_chars: int = MAX_OUTPUT_CHARS,
    ) -> str:
        """Run one shell command on the remote environment and return its output.

        Args:
            command: The bash command to execute.
            description: A short description of the command in natural language.
            timeout_seconds: Maximum time to wait before treating the command as hung.
            max_output_chars: Characters of output kept (head and tail).

        Returns:
            The output of the command, with ``[exit code: N]`` appended when
            it failed, or the timeout error with the output seen so far.
        """
        print(description)
        probe = functools.partial(self._path_test, timeout_seconds, self.stop_event)
        guard, deadline = _refuse_or_lift(
            command, self.workdir, timeout_seconds, probe, self.read_bytes,
            description, max_output_chars, functools.partial(probe, test="-d"),
            functools.partial(setattr, self, "pending_append"),
        )
        if guard is not None:
            return guard
        exit_code, output = bridge_exec(self.url, command, deadline, self.stop_event)
        # A tool-panel Stop ended the wait (the command is already
        # killed): surface it as the interrupted result.
        tool_interrupt.raise_if_interrupted()
        budget = _output_budget(max_output_chars)
        if exit_code is None:
            return _truncate_output(_timeout_message(deadline, output), budget)
        return _truncate_output(_with_exit_code(output, exit_code), budget)

    def run_commands_parallel(
        self,
        commands: str,
        max_workers: int = 0,
        timeout_seconds: float = 1800,
        max_output_chars: int = 5000,
        description: str = "",
    ) -> str:
        """Run several independent shell commands concurrently on the remote environment.

        Args:
            commands: A JSON-encoded list of shell command strings.
            max_workers: Maximum number of commands running at once (``0`` = all).
            timeout_seconds: Timeout in seconds applied to EACH command.
            max_output_chars: Maximum characters of each command's output kept.
            description: A brief description of the commands.

        Returns:
            The tally line and one section per command, as the Docker tool returns.
        """
        del description
        run_one = functools.partial(self._timed_exec, timeout_seconds=timeout_seconds)
        return run_commands_pool(commands, run_one, max_workers, self.stop_event, max_output_chars)

    def _timed_exec(
        self, command: str, cancel: threading.Event, timeout_seconds: float,
    ) -> tuple[int | None, str, float]:
        """Run one ``run_commands_parallel`` command under :meth:`Bash`'s guards; never raises."""
        started = time.monotonic()
        try:  # the guard probes the environment too
            exists = functools.partial(self._path_test, timeout_seconds, cancel)
            guard, timeout_seconds = _refuse_or_lift(
                command, self.workdir, timeout_seconds, exists,
            )
            if guard is not None:
                return -1, guard, 0.0
            exit_code, output = bridge_exec(self.url, command, timeout_seconds, cancel)
        except KISSError as error:
            exit_code, output = 1, str(error)
        return exit_code, output, time.monotonic() - started

    def _path_test(
        self, timeout_seconds: float, cancel: threading.Event | None, path: str, test: str = "-e",
    ) -> bool:
        """Whether *path* may pass ``test <test>`` on the environment (the guards' probe).

        ``-e`` (the default) asks whether the path exists, the
        here-document guard's question; ``-d`` whether it is a
        directory, the question about a ``cd``.  Bounded like the
        command (its deadline and a Stop).  A probe that gave no answer
        (timed out or stopped) fails closed either way, as the Docker
        manager's does: it counts as existing (a copy is refused rather
        than run on a target nobody checked) and as no directory (a
        ``cd`` nobody checked authorizes no rewrite after it).
        """
        exit_code, _ = bridge_exec(
            self.url, f"test {test} {shlex.quote(path)}", timeout_seconds, cancel,
        )
        return exit_code == 0 if test == "-d" else exit_code != 1

    def read_bytes(
        self,
        path: str,
        max_bytes: int = _MAX_BINARY_READ_BYTES,
        timeout_seconds: float = 60,
    ) -> bytes:
        """Return the bytes of the file *path* on the environment.

        One command prints the file's size and its base64
        (:data:`~kiss.agents.sorcar.docker_manager._FETCH_SCRIPT`), as
        ``DockerManager.read_bytes`` does; a file over *max_bytes* is
        not fetched.

        Args:
            path: The file's path on the environment.
            max_bytes: The largest file fetched.
            timeout_seconds: Deadline for the command.

        Returns:
            The file's bytes.

        Raises:
            OSError: When the file is missing, unreadable or over
                *max_bytes*, or the fetch timed out or was stopped.
        """
        script = _FETCH_SCRIPT.format(path=shlex.quote(path), max=max_bytes)
        exit_code, output = bridge_exec(self.url, script, timeout_seconds, self.stop_event)
        tool_interrupt.raise_if_interrupted()
        return fetched_bytes(path, max_bytes, exit_code, output)
