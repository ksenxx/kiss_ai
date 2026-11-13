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

import contextlib
import functools
import http.client
import json
import shlex
import socket
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable
from typing import Any

from kiss.agents.sorcar.docker_manager import (
    _FETCH_SCRIPT,
    _KILL_WAIT_S,
    MAX_OUTPUT_CHARS,
    STOPPED_MESSAGE,
    PendingRewrite,
    _collapse_progress,
    _refuse_or_lift,
    _run_until,
    _timeout_message,
    _with_exit_code,
    fetched_bytes,
    probe_command,
    realpath_command,
    resolved_path,
    until_refusal,
)
from kiss.agents.sorcar.edit_guards import shadow_command
from kiss.agents.sorcar.shell_exit import Ran, exit_word, unrun
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

#: What ``urlsplit`` + ``http.client`` + ``read`` + ``json.loads`` raise for
#: a URL that names no endpoint, an unreachable bridge or a malformed reply.
#: ``http.client.HTTPException`` (``BadStatusLine``, ``IncompleteRead`` when
#: the bridge dies mid-reply) derives from ``Exception``, not ``OSError``;
#: ``RuntimeError`` is a thread that cannot be started (out of threads) and,
#: as ``RecursionError``, ``json.loads`` on a reply nested too deep.
_BRIDGE_ERRORS = (OSError, ValueError, http.client.HTTPException, RuntimeError)

#: Seconds a ``GET /ping`` gets, as a whole.
_PING_TIMEOUT_S = 10.0

#: Seconds past a command's deadline the bridge gets to deliver its reply
#: (the server reports an overrun after ``timeout + 30`` at the latest, see
#: :func:`~kiss.benchmarks.terminal_bench.exec_bridge_server.exec_once`, then
#: sweeps the command).  It is the request's socket timeout, which bounds an
#: idle read, and the margin of :func:`bridge_exec`'s overall deadline, which
#: bounds the whole wait: a reply that trickles in without ever completing
#: keeps the idle timeout from firing.
_REPLY_MARGIN_S = 60.0


class _Connection(http.client.HTTPConnection):
    """An HTTP connection that keeps its socket where its deadline's timer finds it.

    ``getresponse`` sets ``sock`` to ``None`` once it has handed the
    socket to a reply that ends the connection (every reply of the
    bridge: the request says ``Connection: close``), so
    :func:`_shut_down` reads ``watched``, set at the connect and kept
    until the close.
    """

    watched: socket.socket | None = None

    def connect(self) -> None:
        """Connect, then remember the socket for :func:`_shut_down`."""
        super().connect()
        self.watched = self.sock


def _shut_down(connection: _Connection, expired: threading.Event) -> None:
    """Mark the request on *connection* *expired* and shut its socket down (its deadline's timer).

    ``shutdown`` makes a read blocked on the socket return end of stream
    at once, wherever the request is (the status line, the headers, a
    chunk's size line, the body); a ``close`` would not wake it.  The
    socket is not there yet while the connect is in progress: that one
    is bounded by the socket's own timeout, and :func:`_request` checks
    the flag once the request is sent.
    """
    expired.set()
    sock = connection.watched
    if sock is not None:
        with contextlib.suppress(OSError):  # closed already
            sock.shutdown(socket.SHUT_RDWR)


def _request(url: str, path: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    """Request *url* + *path* (POST *body* as JSON, GET without) and return the reply dict.

    *timeout* bounds the request as a whole.  The socket's timeout
    bounds the connect and one idle read, which a peer that trickles a
    byte at a time never lets fire, so a timer shuts the socket down at
    the deadline (:func:`_shut_down`) wherever the request is then, and
    the failure past the deadline is that timeout.  The connection is
    closed before this returns, whatever happened.

    Raises:
        KISSError: When the bridge cannot be reached (also for want of
            a thread for the timer), the reply did not complete in
            time, or the reply is no JSON object.
    """
    try:
        target = urllib.parse.urlsplit(url.rstrip("/") + path)
        # The port is always passed: given none, ``http.client`` looks for one
        # after the host's last colon, which an IPv6 address has too.
        port = http.client.HTTP_PORT if target.port is None else target.port
        connection = _Connection(target.hostname or "", port, timeout=timeout)
    except _BRIDGE_ERRORS as error:  # a port that is no number, an unclosed IPv6 bracket
        raise KISSError(f"Exec bridge {url} unreachable: {error}") from error
    expired = threading.Event()
    watchdog = threading.Timer(timeout, _shut_down, (connection, expired))
    watchdog.daemon = True
    try:
        watchdog.start()
        connection.request(
            "GET" if body is None else "POST", target.path or "/",
            body=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Connection": "close"},
        )
        if expired.is_set():  # the deadline passed during the connect
            raise TimeoutError
        with connection.getresponse() as response:
            reply = json.loads(response.read().decode("utf-8", errors="replace"))
        if expired.is_set():  # a body read to its end as the deadline passed
            raise TimeoutError
    except _BRIDGE_ERRORS as error:
        reason = "the reply did not complete within the deadline" if expired.is_set() else error
        raise KISSError(f"Exec bridge {url} unreachable: {reason}") from error
    finally:
        watchdog.cancel()
        connection.close()
    if not isinstance(reply, dict):
        raise KISSError(
            f"Exec bridge {url} unreachable: returned an invalid reply of type "
            f"{type(reply).__name__}")
    return reply


def _kill_quietly(url: str, token: str) -> None:
    """POST ``/kill`` for *token*, swallowing an unreachable bridge (helper-thread body)."""
    try:
        _request(url, "/kill", {"token": token}, 120)
    except KISSError:
        pass


def _kill_bounded(url: str, token: str) -> None:
    """Post ``/kill`` for *token* from a helper thread; wait at most :data:`_KILL_WAIT_S` for it.

    The bridge server sweeps the remote sandbox before it replies, and a
    stalled sandbox must not hold the task thread for the request's
    whole socket timeout: the kill goes on with the helper thread.
    Out of threads nothing is posted (a request needs one for its
    timer too), and the error being handled is not replaced by that
    one: the bridge sweeps the command itself once its deadline passes.
    """
    killer = threading.Thread(
        target=_kill_quietly, args=(url, token),
        name=f"exec-bridge-kill-{token[:8]}", daemon=True,
    )
    try:
        killer.start()
    except RuntimeError:
        return
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
    command never outlives the call; so does a request the bridge failed
    after it may have started the command (the reply connection dropped,
    an invalid reply), before the error propagates.  The wait as a whole
    ends :data:`_REPLY_MARGIN_S` past the command's deadline: a bridge
    that has not delivered the reply by then (stalled, or trickling a
    body that never completes) gets the same kill and the command is
    reported as timed out; the request thread's request ends at the same
    deadline (:func:`_request`'s timer), so it does not outlive the call, and
    its failure past the deadline is that timeout, whichever of the two
    notices first.  A Stop set before the call posts nothing; one
    landing as the reply completes wins over it, as it does for a
    Docker exec.

    Args:
        url: The bridge's base URL (``http://127.0.0.1:<port>``).
        command: The shell command (run with ``sh -c``).
        timeout: Seconds after which the bridge gives up on the command.
        cancel: Optional event that ends the wait early (the task's stop
            event or a ``run_commands_parallel`` fan-out's).

    Returns:
        ``(exit code, combined output)``; the exit code is ``None`` when the
        command was still running at the deadline (or the bridge delivered
        no reply within the margin past it) and ``-1`` when it was stopped
        (the output then ends with :data:`STOPPED_MESSAGE`).

    Raises:
        KISSError: When the bridge cannot be reached (also for want of
            a thread to make the request on) or answers garbage.
    """
    token = uuid.uuid4().hex
    body = {"command": command, "timeout": timeout, "token": token}
    outcome: list[dict[str, Any] | KISSError] = []

    def fetch() -> None:
        try:
            outcome.append(_request(url, "/exec", body, timeout + _REPLY_MARGIN_S))
        except KISSError as error:
            outcome.append(error)

    if _stop_requested(cancel):
        return -1, STOPPED_MESSAGE
    worker = threading.Thread(target=fetch, name=f"exec-bridge-{token[:8]}", daemon=True)
    deadline = time.monotonic() + timeout + _REPLY_MARGIN_S
    try:
        try:
            worker.start()
        except RuntimeError as error:  # out of threads
            raise KISSError(f"Exec bridge {url} unreachable: {error}") from error
        # ``while True``: on CPython 3.13 the back edge of
        # ``while worker.is_alive():`` is outside this ``try``'s
        # exception table, and a Stop raised there would skip the kill
        # below.
        while True:
            if not worker.is_alive():
                break
            if _stop_requested(cancel):
                _kill_bounded(url, token)
                return -1, STOPPED_MESSAGE
            if time.monotonic() >= deadline:
                _kill_bounded(url, token)
                return None, ""
            worker.join(0.2)
        if _stop_requested(cancel):
            _kill_bounded(url, token)
            return -1, STOPPED_MESSAGE
        reply = outcome[0]
        if isinstance(reply, KISSError):
            if time.monotonic() >= deadline:
                _kill_bounded(url, token)
                return None, ""
            raise reply
        code = reply.get("exit_code")
        if "output" not in reply or not (code is None or isinstance(code, int)):
            raise KISSError(f"Exec bridge {url} returned an invalid reply: {reply!r}")
    except BaseException:
        # The wait was unwound from outside (an injected Stop, a
        # KeyboardInterrupt) or the request failed after the bridge may
        # have started the command: the request thread and the command
        # on the environment must not go on without anyone waiting for
        # them.
        _kill_bounded(url, token)
        raise
    return code, str(reply["output"])


def bridge_alive(url: str) -> bool:
    """Whether the bridge at *url* answers ``GET /ping`` with ``{"ok": true}``.

    The ping as a whole is bounded by :data:`_PING_TIMEOUT_S`: a reply
    that never completes, like a bridge that does not answer or answers
    with no JSON object, means it is not alive.
    """
    try:
        return bool(_request(url, "/ping", None, _PING_TIMEOUT_S).get("ok"))
    except KISSError:
        return False


class ExecBridge:
    """The agent's shell tools on a remote environment, through a local HTTP bridge.

    Mirrors the parts of :class:`kiss.agents.sorcar.docker_manager.DockerManager`
    the agent uses: ``Bash``, ``run_commands_parallel`` (both with the
    Docker tools' shell guards), ``run``, ``stop_event``, ``stream_callback``
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
        # The append a ``>>`` copy ``Bash`` redirected stands for, or the
        # edit a ``sed -i`` or python rewrite does, until the ``Write`` or
        # ``Edit`` of ``DockerTools`` settles it against the file's current
        # bytes.
        self.pending_append: PendingRewrite | None = None

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
        until: str = "",
        poll_seconds: float = 5.0,
    ) -> str:
        """Run one shell command on the remote environment and return its output.

        Args:
            command: The bash command to execute.
            description: A short description of the command in natural language.
            timeout_seconds: Maximum time to wait before treating the command as hung.
            max_output_chars: Characters of output kept (head and tail).
            until: A regular expression: when given, the command (a cheap
                read-only check such as ``grep DONE build.log; tail -3
                build.log``) is re-run every ``poll_seconds`` until its
                output matches or ``timeout_seconds`` elapse, in this one
                call, so a wait for a file, a port or a log line costs one
                turn however long it takes.  The result is the last run's
                output plus one line saying whether it matched and after
                how many runs and seconds.
            poll_seconds: Seconds between two runs with ``until`` (at least 1).

        Returns:
            The output of the command, with ``[exit code: N]`` appended when
            it failed, or the timeout error with the output seen so far.
        """
        return self.run_bash(
            command, description, timeout_seconds, max_output_chars, until, poll_seconds).text

    def run_bash(
        self,
        command: str,
        description: str,
        timeout_seconds: int = 30,
        max_output_chars: int = MAX_OUTPUT_CHARS,
        until: str = "",
        poll_seconds: float = 5.0,
    ) -> Ran:
        """Make a :meth:`Bash` call: its result and how the command ended.

        As ``DockerManager.run_bash``: the exit is data for the agent's
        ``Bash`` tool to hand to the coding gate, which cannot tell the
        tool's ``[exit code: N]`` line from one the command printed.

        Args:
            command: As for :meth:`Bash`, like every argument.
            description: The call's description.
            timeout_seconds: The call's deadline.
            max_output_chars: The result's output budget.
            until: The regular expression of a poll.
            poll_seconds: Seconds between two runs of a poll.

        Returns:
            The result with the command's exit status, ``"timeout"``, or
            ``"refused"`` / ``"error"`` for a command that did not run.

        Raises:
            ToolRedirect: The ``Write``/``Edit`` to make instead of *command*.
        """
        print(description)
        asked = (timeout_seconds, self.stop_event)
        probe = functools.partial(self._path_test, *asked)
        guard, deadline = _refuse_or_lift(
            command, self.workdir, timeout_seconds, probe, functools.partial(probe, test="-d"),
            functools.partial(self._realpath, *asked), self.read_bytes,
            description, max_output_chars,
            functools.partial(setattr, self, "pending_append"),
            functools.partial(probe, test="link"),
            functools.partial(self._shadows, *asked),
            functools.partial(probe, test="run"),
        )
        if guard is not None:
            return unrun(guard)
        run = functools.partial(bridge_exec, self.url, command, cancel=self.stop_event)
        verdict = ""
        if until:
            refusal = until_refusal(until, poll_seconds)
            if refusal is not None:
                return unrun(refusal)
            (exit_code, output), verdict = _run_until(
                run, until, poll_seconds, deadline, self.stop_event)
            verdict = "\n" + verdict
        else:
            exit_code, output = run(deadline)
        # A tool-panel Stop ended the wait (the command is already
        # killed): surface it as the interrupted result.
        tool_interrupt.raise_if_interrupted()
        # The status goes after the cut output, whole: cut with it, a small
        # budget dropped the status and a failed command read as one that passed.
        output = _truncate_output(_collapse_progress(output), _output_budget(max_output_chars))
        text = (_timeout_message(deadline, output) if exit_code is None
                else _with_exit_code(output, exit_code))
        return Ran(text + verdict, exit_word(exit_code))

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
        # The pending append or edit of a redirected rewrite belongs to the
        # Write or Edit made right after it: like any other call, this one
        # drops it.
        self.pending_append = None
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
                functools.partial(exists, test="-d"),
                functools.partial(self._realpath, timeout_seconds, cancel),
            )
            if guard is not None:
                return -1, guard, 0.0
            exit_code, output = bridge_exec(self.url, command, timeout_seconds, cancel)
        except KISSError as error:
            # Not run (the bridge unreachable, its reply invalid): ``-1``
            # and an ``Error:`` output, as the Docker and host tools report it.
            exit_code, output = -1, f"Error: {error}"
        return exit_code, output, time.monotonic() - started

    def _path_test(
        self, timeout_seconds: float, cancel: threading.Event | None, path: str, test: str = "-e",
    ) -> bool:
        """Whether *path* may pass ``test <test>`` on the environment (the guards' probe).

        ``-e`` (the default) asks whether the path exists, the
        here-document guard's question; ``-d`` whether it is a
        directory, the question about a ``cd``; ``link`` whether a
        ``sed -i`` on it would cut a link or replace a read-only file;
        ``run`` whether the environment's shell finds the program
        *path* names, the question about the program of a rewrite
        (:func:`probe_command`).  Bounded like the command (its
        deadline and a Stop).  A probe that gave no answer (timed out
        or stopped) fails closed either way, as the Docker manager's
        does: it counts as existing (a copy is refused rather than run
        on a target nobody checked), as no directory (a ``cd`` nobody
        checked authorizes no rewrite after it), as a link (no Edit
        stands for a sed nobody checked) and as a program the shell
        does not find (none stands for a command nobody knows to run).
        """
        exit_code, _ = bridge_exec(self.url, probe_command(path, test), timeout_seconds, cancel)
        return exit_code == 0 if test in ("-d", "run") else exit_code != 1

    def _realpath(self, timeout_seconds: float, cancel: threading.Event | None, path: str) -> str:
        """Return the path the environment's filesystem gives *path* (``os.path.realpath`` there).

        The destructive guard's oracle, as the Docker manager's
        (:func:`realpath_command`).

        Raises:
            KISSError: When the environment gave no answer.
        """
        exit_code, output = bridge_exec(self.url, realpath_command(path), timeout_seconds, cancel)
        return resolved_path(path, exit_code, output)

    def _shadows(
        self, timeout_seconds: float, cancel: threading.Event | None,
        program: str, path: str, modules: set[str],
    ) -> bool:
        """Whether the python *program*, run in *path* on the environment, shadows *modules*.

        It loads one of them from elsewhere than its own library, that
        is (:func:`shadow_command`): the guards' question about a python
        rewrite's imports, asked like :meth:`_path_test`'s.  A probe
        that gave no answer fails closed, as the Docker manager's does:
        no Edit stands for a program that imports a module nobody checked.
        """
        asked = shadow_command(program, path, modules)
        exit_code, _ = bridge_exec(self.url, asked, timeout_seconds, cancel)
        return exit_code != 1

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

    def run(self, command: str, timeout_seconds: float) -> tuple[int | None, str]:
        """Run *command* on the environment: its exit code and output, uncut.

        The finish gate's primitive, as ``DockerManager.run`` is for a
        container: no guard, no echo of a description and none of
        :meth:`Bash`'s output cap.

        Args:
            command: The shell command to execute.
            timeout_seconds: Seconds it may run; past them the code is ``None``.

        Returns:
            ``(exit code, output)``; the code is ``None`` at the deadline
            and ``-1`` when the task was stopped meanwhile.

        Raises:
            KISSError: When the bridge cannot be reached or answers garbage.
        """
        exit_code, output = bridge_exec(self.url, command, timeout_seconds, self.stop_event)
        tool_interrupt.raise_if_interrupted()
        return exit_code, output
