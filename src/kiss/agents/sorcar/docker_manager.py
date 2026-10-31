# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Docker library for managing Docker containers and executing commands."""

import codecs
import functools
import logging
import os
import queue
import shlex
import shutil
import socket
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

import docker
from docker.models.containers import Container  # type: ignore[assignment]

from kiss.agents.sorcar._concurrency import _race_delay, locked_call
from kiss.agents.sorcar.edit_guards import heredoc_guard, rewrite_guard
from kiss.agents.sorcar.shell_guards import refuse_or_lift
from kiss.agents.sorcar.useful_tools import (
    _output_budget,
    _stop_requested,
    _truncate_output,
    run_commands_pool,
)
from kiss.core import tool_interrupt
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_error import KISSError

logger = logging.getLogger(__name__)

#: Default cap on the characters a single command may return, matching
#: ``UsefulTools.Bash``.  Without it an unbounded ``pip install`` log goes
#: straight into the conversation and blows the model's context window.
MAX_OUTPUT_CHARS = DEFAULT_CONFIG.tool_output_default_chars

#: Line of a ``Bash`` result (before its ``[exit code: -1]`` line) whose
#: command was killed by a task Stop or a tool-panel Stop; ``DockerTools``
#: reads it to know that a follow-up command would be killed the same way.
STOPPED_MESSAGE = "Killed: the task was stopped."

#: Environment variable used to tag an exec — and every process
#: it spawns — so a timed-out command can be killed by matching
#: ``/proc/<pid>/environ`` inside the container's own pid namespace.
_EXEC_TOKEN_VAR = "KISS_EXEC_TOKEN"

#: Poll cadence of the reaper that waits for a timed-out exec whose start
#: the docker daemon has delayed.  The reaper runs on a daemon thread and
#: never gives up while the container lives (giving up would leave a
#: sufficiently delayed command running for the rest of the container's
#: life); instead the interval backs off exponentially to this cap so an
#: indefinitely delayed start costs one inspect every few seconds.
_REAP_POLL_INTERVAL_S = 0.2
_REAP_POLL_MAX_INTERVAL_S = 5.0

#: How long a timed-out or stopped command waits for the reaper's first
#: kill sweep before returning: long enough that the command is normally
#: dead when the caller gets its result (one ``exec_run``, under a
#: hundred milliseconds on an idle daemon, over a second on a busy one),
#: short enough that a daemon withholding the sweep's stream cannot hold
#: the task thread.  The sweep itself continues on the reaper thread.
_KILL_WAIT_S = 2.0

#: ``docker_image`` spelling that attaches to an already running container
#: (``container:<name-or-id>``) instead of starting one from an image.
ATTACH_PREFIX = "container:"

#: Docker label stamped on every container ``open()`` starts, valued
#: :func:`owner_label_value` of the starting process.  The daemon is shared
#: with other work (benchmarks, concurrent test processes), so the image
#: alone cannot tell a container of this process from a stranger's; the
#: label lets bookkeeping and cleanup filter with
#: ``containers.list(filters={"label": f"{OWNER_LABEL}={owner_label_value()}"})``.
OWNER_LABEL = "kiss.owner"


def owner_label_value() -> str:
    """Return the :data:`OWNER_LABEL` value of the current process (``host:pid``)."""
    return f"{socket.gethostname()}:{os.getpid()}"


def _new_utf8_decoder() -> Any:
    """Return an incremental UTF-8 decoder that never raises.

    Docker delivers exec output as byte frames split at arbitrary
    boundaries, so a single multi-byte character (an accented letter, an
    emoji, a progress-bar glyph) is routinely delivered as two frames.
    Decoding each frame on its own would raise ``UnicodeDecodeError``
    nondeterministically, and strict decoding would also lose the entire
    output of a command that merely printed a stray binary byte.  An
    incremental decoder in ``replace`` mode carries the partial sequence
    over to the next frame and substitutes U+FFFD for genuinely invalid
    bytes — the same guarantee ``UsefulTools._spawn`` gives with
    ``errors="replace"``.

    Returns:
        A fresh ``codecs`` incremental decoder; feed it with
        ``decoder.decode(chunk)`` and flush with ``decoder.decode(b"", True)``.
    """
    return codecs.getincrementaldecoder("utf-8")("replace")


def _drain_exec_stream(
    output_gen: Any,
    out_queue: "queue.Queue[tuple[bool, str] | None]",
) -> None:
    """Decode a docker exec stream onto *out_queue* until it ends.

    Runs on the exec's worker thread so the caller can enforce a timeout
    on a generator that otherwise blocks forever.  stdout and stderr each
    get their own incremental decoder because their frames interleave.
    A read failure propagates to the worker (it is not a clean end of
    stream); note that the SDK's ``CancellableStream`` already turns a
    dropped socket into a plain end of stream, which the worker then
    detects through ``exec_inspect`` (:meth:`_ExecRun._wait_for_exit_code`).

    Args:
        output_gen: The demuxed ``CancellableStream`` from ``exec_start``
            (typed ``Any`` because the SDK exposes no stub for it).
        out_queue: Receives ``(is_stderr, text)`` items.
    """
    decoders = {False: _new_utf8_decoder(), True: _new_utf8_decoder()}
    try:
        for chunk in output_gen:
            if isinstance(chunk, tuple):  # pragma: no branch
                stdout_chunk, stderr_chunk = chunk
            else:
                stdout_chunk, stderr_chunk = chunk, None
            for is_stderr, raw in ((False, stdout_chunk), (True, stderr_chunk)):
                if not raw:
                    continue
                text = decoders[is_stderr].decode(raw)
                if text:
                    out_queue.put((is_stderr, text))
    finally:
        # The SDK leaves closing a streamed exec to the caller; without
        # it the socket lingers until a GC cycle.  The HTTP response is
        # closed rather than the stream: ``CancellableStream.close()``
        # only shuts the raw socket, keeps the descriptor until GC and
        # makes the response's own close raise ``ValueError`` later
        # (measured against a live daemon: 15 execs left 16 / 11 / 0
        # descriptors open with no close / stream close / response
        # close).  Guarded so the trailing bytes are always flushed.
        try:
            getattr(output_gen, "_response", output_gen).close()
        except Exception:
            logger.debug("closing docker exec response failed", exc_info=True)
        for is_stderr, decoder in decoders.items():
            trailing = decoder.decode(b"", True)
            if trailing:
                out_queue.put((is_stderr, trailing))


def _take_output(
    item: tuple[bool, str],
    stdout_parts: list[str],
    stderr_parts: list[str],
    on_text: Callable[[str], None] | None,
) -> None:
    """Keep one decoded ``(is_stderr, text)`` piece and hand it to *on_text* when given."""
    is_stderr, text = item
    (stderr_parts if is_stderr else stdout_parts).append(text)
    if on_text is not None:
        on_text(text)


def _collapse_progress(output: str) -> str:
    """Render carriage returns the way a terminal does: later text overwrites earlier text.

    Progress bars (pip, tqdm, fastText, wget) redraw one line thousands of
    times with ``\\r``; in a captured stream those redraws arrive as one huge
    line whose only useful part is its final state.  Each ``\\r`` moves the
    cursor to the start of the line, so every segment is written over the
    previous ones from column 0; characters beyond the new segment's end stay
    (as on a terminal without erase-to-end-of-line).  ``\\r\\n`` line endings
    therefore become plain ``\\n``.

    Args:
        output: The command's combined output.

    Returns:
        *output* as it would look on a terminal after the carriage returns.
    """
    if "\r" not in output:
        return output
    lines = []
    for line in output.split("\n"):
        if "\r" in line:
            rendered = ""
            for segment in line.split("\r"):
                rendered = segment + rendered[len(segment):]
            line = rendered
        lines.append(line)
    return "\n".join(lines)


def kill_tagged_tree_script(token: str) -> str:
    """Return a ``sh`` script that kills every process whose environment carries *token*.

    A command the agent runs is tagged with ``KISS_EXEC_TOKEN=<token>``
    (see :func:`tagged_command`); every child inherits the variable, so
    matching on ``/proc/<pid>/environ`` finds the whole tree.  The tree
    is collected first, frozen with SIGSTOP, then killed: killing one
    process per scan step let ``bash -c "sleep 20; touch f"`` continue
    to ``touch f`` when the glob visited ``sleep`` before its parent
    shell (``/proc/[0-9]*`` sorts lexically, not by pid).

    Args:
        token: The unique tag planted in the command's environment.
    """
    return (
        "pids=\n"
        "for d in /proc/[0-9]*; do\n"
        '  env=$(tr "\\0" "\\n" < "$d/environ" 2>/dev/null)\n'
        f'  case "$env" in *"{_EXEC_TOKEN_VAR}={token}"*)\n'
        '    pids="$pids ${d#/proc/}";;\n'
        "  esac\n"
        "done\n"
        '[ -n "$pids" ] || exit 0\n'
        "kill -STOP $pids 2>/dev/null\n"
        "kill -9 $pids 2>/dev/null"
    )


def tagged_command(command: str, token: str) -> str:
    """Return a shell line that runs *command* in a bash whose process tree carries *token*.

    For environments whose ``exec`` takes a shell string and no
    environment dict (the exec bridge); :meth:`DockerManager._tagged_exec_create`
    plants the same variable through the Docker API instead.  The tag must
    be in the environment block the shell is *exec'd* with (that is what
    ``/proc/<pid>/environ`` shows, for the shell and every subshell it
    forks); an ``export`` inside the running shell would tag only the
    programs it later execs, and a background subshell would survive the
    sweep.  ``sh`` is used when the environment has no bash.

    Args:
        command: The shell command to run.
        token: The unique tag (hex) :func:`kill_tagged_tree_script` later matches on.
    """
    return (
        "kiss_sh=bash; command -v bash >/dev/null 2>&1 || kiss_sh=sh; "
        f'exec env {_EXEC_TOKEN_VAR}={token} "$kiss_sh" -c {shlex.quote(command)}'
    )


def _timeout_message(timeout_seconds: float, output: str) -> str:
    """Return the error text for a command killed at *timeout_seconds*.

    The output produced before the deadline is what tells the model
    whether the command was hung or merely slow, so it is returned
    with the error (progress lines collapsed) instead of discarded.

    Args:
        timeout_seconds: The deadline that expired.
        output: Everything the command printed before it was killed.
    """
    message = f"Error: command timed out after {timeout_seconds}s"
    if output:
        message += " and was killed. Output before the timeout:\n" + _collapse_progress(output)
    return message


def _refuse_or_lift(
    command: str, workdir: str | None, timeout_seconds: float, exists: Callable[[str], bool],
) -> tuple[str | None, float]:
    """Apply the guards of the container ``Bash`` tools to *command*.

    The destructive and install-timeout guards of the host ``Bash``
    (:func:`refuse_or_lift`), and its refusals of shell rewrites that
    belong in ``Edit``: :func:`rewrite_guard`, and :func:`heredoc_guard`,
    which calls *exists* to learn whether a here-document's target exists
    in the container (or the exec bridge's environment, whose *workdir*
    is ``None`` when unknown).

    Returns:
        ``(refusal, timeout_seconds)``: the guard's message (``None``
        when the command may run) and the possibly lifted deadline.
    """
    guard, timeout_seconds = refuse_or_lift(command, workdir, timeout_seconds)
    refusal = guard or rewrite_guard(command) or heredoc_guard(command, workdir, exists)
    return refusal, timeout_seconds


def _with_exit_code(output: str, exit_code: int) -> str:
    """Append the ``[exit code: N]`` marker for a failed command.

    Carriage-return progress output is collapsed first (see :func:`_collapse_progress`).

    Args:
        output: The command's combined output.
        exit_code: The command's exit status.

    Returns:
        *output* unchanged on success, else *output* plus the marker.
    """
    output = _collapse_progress(output)
    if exit_code == 0:
        return output
    suffix = f"[exit code: {exit_code}]"
    return f"{output}\n{suffix}" if output else suffix


class _ExecRun:
    """One docker exec, shared by the task thread, its worker thread and its reaper.

    The task thread (:meth:`DockerManager._exec`) only ever waits on
    :attr:`out_queue` in 0.2 s slices, where an injected Stop lands.
    Every daemon call that may block — ``exec_create``, ``exec_start``,
    reading the stream, ``exec_inspect`` for the exit code — runs on the
    worker (:meth:`run_worker`), and the kill sweep on the reaper
    (:meth:`reap`), so neither a slow daemon nor a withheld stream can
    hold the task thread past its deadline.

    The exec is tagged with a unique environment token so a timed-out
    command can be killed inside the container instead of running (and
    consuming container resources) for the rest of the container's life.
    Timeout and exec startup are coordinated through :attr:`lock`: a
    single kill scan at the deadline is not enough, because the daemon
    may delay ``exec_create``/``exec_start`` past the deadline, in which
    case the scan sees no tagged process and the command starts — and
    runs forever — *after* ``Bash`` has returned the timeout error.  The
    worker therefore commits to starting only while not cancelled
    (checked under the lock after ``exec_create``); a timed-out caller
    sets ``cancelled`` under the same lock, so either the worker never
    calls ``exec_start`` at all, or the caller sees the commitment and
    hands the exec to a reaper that kills it once it has started.  A
    start the daemon definitively refused (``exec_start`` answered with
    an error status, e.g. ``409`` for a paused container) is recorded as
    well, because such an exec never gets a pid and a reaper waiting for
    one would poll until the container closes.
    """

    def __init__(self, manager: "DockerManager", container_id: str | None, command: str) -> None:
        self.manager = manager
        self.container_id = container_id
        self.command = command
        self.token = uuid.uuid4().hex
        self.exec_id = ""  # set by the worker before it commits to starting
        # Output arrives piece by piece (not returned whole by
        # ``exec_start``) so a timed-out command can still report what
        # it printed before the kill.  A ``None`` item marks the end of
        # the worker, whether the exec finished or the worker failed.
        self.out_queue: queue.Queue[tuple[bool, str] | None] = queue.Queue()
        self.error: BaseException | None = None
        self.exit_code: int | None = None
        self.lock = threading.RLock()
        self.cancelled = False
        self.start_committed = False
        self.start_refused = False

    def run_worker(self) -> None:
        """Worker thread body: create, start and drain the exec, then fetch its exit code."""
        try:
            self.exec_id = self.manager._tagged_exec_create(
                self.container_id, self.command, self.token,
            )
            with self.lock:
                if self.cancelled:
                    return
                self.start_committed = True
            try:
                stream = self.manager.client.api.exec_start(
                    self.exec_id, stream=True, demux=True,
                )
            except docker.errors.APIError:  # type: ignore[attr-defined]
                # The daemon answered the start with an error status:
                # the exec never ran.  Any other failure (a dropped
                # connection) is ambiguous and keeps the reaper.
                with self.lock:
                    self.start_refused = True
                raise
            _drain_exec_stream(stream, self.out_queue)
            self._wait_for_exit_code()
        except BaseException as exc:
            self.error = exc
        finally:
            self.out_queue.put(None)

    def _wait_for_exit_code(self) -> None:
        """Poll ``exec_inspect`` until the exit code is known or the exec is cancelled.

        The stream can end before the process does: it closed its own
        stdio, or the daemon connection dropped (the SDK reports that as
        a plain end of stream).  Each inspect may block for as long as
        the daemon waits for the exec's streams to close, which is why
        this runs on the worker and not on the task thread.
        """
        while not self.cancelled:
            self.exit_code = self.manager.client.api.exec_inspect(self.exec_id).get("ExitCode")
            if self.exit_code is not None:
                return
            time.sleep(0.2)

    def _mark_cancelled(self) -> bool:
        """Cancel the exec (caller holds :attr:`lock`); return whether it needs a reaper."""
        self.cancelled = True
        return self.start_committed and not self.start_refused

    def cancel(self) -> None:
        """Kill the exec from the task thread (timeout or injected Stop).

        The reaper is installed first, so a Stop forced during the wait
        below still leaves someone to kill a start the daemon delayed
        past the deadline; the wait for its first sweep is bounded by
        :data:`_KILL_WAIT_S` and sliced so an injected Stop lands.  The
        state is taken through ``locked_call`` so a second injection
        cannot leak the lock under the worker.
        """
        if not locked_call(self.lock, self._mark_cancelled):
            return
        swept = threading.Event()
        threading.Thread(
            target=self.reap, args=(swept,),
            name=f"docker-exec-reaper-{self.token[:8]}", daemon=True,
        ).start()
        deadline = time.monotonic() + _KILL_WAIT_S
        while not swept.is_set() and time.monotonic() < deadline:
            swept.wait(0.2)

    def reap(self, swept: threading.Event) -> None:
        """Reaper thread body: sweep the token-tagged tree now and again once the exec has started.

        The first sweep runs at once (``exec_inspect`` can block for up
        to 2 s while the daemon waits for the exec's streams to close,
        which is exactly the window in which a background child of an
        already-exited shell — ``(sleep 2; touch m) &`` — would finish
        its work) and *swept* is set when it is done.  By then the
        daemon may still not have started the container-side process,
        so the thread polls ``exec_inspect`` as well: once the exec has
        started (``Pid`` is non-zero only then) it sweeps the tree, and
        it stops after the sweep that finds the exec itself finished.
        That last sweep matters: the exec's own shell may already have
        exited while background children it spawned (``sleep 71 &``)
        live on, holding the output stream open; they carry the token
        too and die here.  A never-started exec is watched for as long
        as the container lives — a fixed window would let a start
        delayed past it run forever, the exact bug this reaper exists
        to prevent — unless the daemon refused the start.  The poll
        interval backs off so an indefinitely delayed start costs one
        inspect every few seconds.

        Args:
            swept: Set once the first sweep has run.
        """
        try:
            self.manager._kill_exec(self.token)
        finally:
            swept.set()
        interval = _REAP_POLL_INTERVAL_S
        while self.manager.container is not None and not self.start_refused:
            try:
                info = self.manager.client.api.exec_inspect(self.exec_id)
            except Exception:  # container/exec already gone
                logger.debug("could not inspect timed-out exec", exc_info=True)
                return
            if info.get("Pid"):
                self.manager._kill_exec(self.token)
                if not info.get("Running"):
                    return  # the exec itself is finished; its tree is swept
            time.sleep(interval)
            interval = min(interval * 2, _REAP_POLL_MAX_INTERVAL_S)


class DockerManager:
    """Manages Docker container lifecycle and command execution."""

    def __init__(
        self,
        image_name: str,
        tag: str = "latest",
        workdir: str = "/",
        mount_shared_volume: bool = True,
        ports: dict[int, int] | None = None,
        volumes: dict[str, str] | None = None,
    ) -> None:
        """Initialize the Docker client.

        Args:
            image_name: The name of the Docker image (e.g., 'ubuntu', 'python'),
                or ``container:<name-or-id>`` to attach to a container that
                is already running.  An attached container is owned by the
                caller: ``open()`` looks it up instead of starting one,
                commands run in the container's own ``WorkingDir``, no shared
                volume and no *volumes* are mounted, and ``close()`` leaves
                it running.
            tag: The tag/version of the image (default: 'latest')
            workdir: The working directory inside the container.  The
                container is started with it as its ``WorkingDir`` too, so
                a later ``container:<id>`` attach runs commands there.
            mount_shared_volume: Whether to mount a shared volume. Set to False
                for images that already have content in the workdir (e.g., SWE-bench).
            ports: Port mapping from container port to host port.
                Example: {8080: 8080} maps container port 8080 to host port 8080.
                Example: {80: 8000, 443: 8443} maps multiple ports.
            volumes: Extra read-write bind mounts, host path to container
                path.  Example: ``{"/home/me/repo": "/home/me/repo"}``
                makes the host directory visible at the same path inside
                the container.  Unlike the shared volume, these directories
                belong to the caller and are never deleted by ``close()``.
                After ``open()``, ``self.volumes`` holds the container's
                bind mounts: this mapping for a started container, the
                attached container's own ones in attach mode.
        """
        self.client = docker.from_env()
        self.container: Container | None = None
        # The unique name ``open()`` gives the container it creates, chosen
        # before the create call: the container is recoverable by it from
        # the moment the daemon makes it, even when a Stop lands inside
        # ``containers.create()`` (between its create API call and the GET
        # that returns the handle) and leaves ``container`` None.  Cleared
        # once the container is removed.
        self.container_name: str | None = None
        # Serialises open()/close(): guard, shared-dir creation, the
        # (slow) container start, publication and teardown.  Without it
        # two concurrent open() calls both passed the "already open"
        # check while ``container`` was still None and each started a
        # container and a temp dir; the manager kept only the last one.
        # RLock, taken through ``locked_call``: ``open``/``close`` run
        # on the task thread (``with DockerManager(...)`` in the agent),
        # where the runner's Stop injection would leak a plain ``with``
        # (CPython gh-74174) and make the same thread's ``close()`` in
        # ``__exit__`` hang on it for good.
        self._lifecycle_lock = threading.RLock()

        self.workdir = workdir
        self.mount_shared_volume = mount_shared_volume
        self.ports = ports
        self.volumes = dict(volumes or {})
        self.client_shared_path = "/testbed"
        self.host_shared_path: str | None = None
        self.stream_callback: Callable[[str], None] | None = None
        # The task's stop event, when the owning agent has one:
        # ``run_commands_parallel`` kills its running execs on it.
        self.stop_event: threading.Event | None = None

        self.attached_container: str | None = None
        if image_name.startswith(ATTACH_PREFIX):
            self.attached_container = image_name[len(ATTACH_PREFIX):]
            self.image, self.tag = "", ""
        elif ":" in image_name:
            self.image, self.tag = image_name.rsplit(":", 1)
        else:
            self.image = image_name
            self.tag = tag

    def open(self) -> None:
        """Pull and load a Docker image, then create and start a container.

        Raises:
            KISSError: If a container is already open on this manager
                (starting a second one would orphan the first — ``close``
                only knows the newest container — and leak its shared
                volume directory), or if the previous container's shared
                volume directory still cannot be removed (starting anyway
                would replace the only reference to it).
        """
        locked_call(self._lifecycle_lock, self._open_locked)

    def _open_locked(self) -> None:
        """Body of :meth:`open`; the caller holds ``_lifecycle_lock``."""
        if self.container is not None:
            raise KISSError(
                "A container is already open on this DockerManager; "
                "call close() before open() again."
            )
        if self.host_shared_path is not None:
            # A previous close() could not delete the shared volume
            # dir (see _remove_shared_volume_dir); retry before a new
            # one is created so the leftover never becomes untraceable.
            self._remove_shared_volume_dir()
            if self.host_shared_path is not None:
                raise KISSError(
                    "The previous container's shared volume directory "
                    f"{self.host_shared_path} could not be removed; "
                    "delete it and call open() again."
                )
        if self.attached_container is not None:
            self.container = self.client.containers.get(self.attached_container)
            self.workdir = self.container.attrs["Config"].get("WorkingDir") or "/"
            # Report the container's real bind mounts so callers can tell
            # which host directories are visible inside (e.g. a work dir
            # mounted by the manager that started this container).
            self.volumes = {
                mount["Source"]: mount["Destination"]
                for mount in self.container.attrs.get("Mounts", [])
                if mount.get("Type") == "bind"
            }
            print(f"Attached to running container {(self.container.id or '')[:12]}")
            return
        image = self.image
        tag = self.tag
        full_image_name = f"{image}:{tag}"
        print(f"Pulling Docker image: {full_image_name}")
        try:
            self.client.images.get(full_image_name)
        except docker.errors.ImageNotFound:  # type: ignore[attr-defined]
            logger.debug("Exception caught", exc_info=True)
            self.client.images.pull(image, tag=tag)
        print(f"Creating and starting container from {full_image_name}")
        container_kwargs: dict[str, Any] = {
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "command": "/bin/bash",
            "working_dir": self.workdir,
            "labels": {OWNER_LABEL: owner_label_value()},
        }
        mounts = {
            host: {"bind": target, "mode": "rw"} for host, target in self.volumes.items()
        }
        if self.mount_shared_volume:
            self.host_shared_path = tempfile.mkdtemp()
            mounts[self.host_shared_path] = {"bind": self.client_shared_path, "mode": "rw"}
        if mounts:
            container_kwargs["volumes"] = mounts
        if self.ports:
            container_kwargs["ports"] = {f"{cp}/tcp": hp for cp, hp in self.ports.items()}
        # create() and start() are separate calls, under a name chosen
        # here, so the manager owns the container from the moment it
        # exists: docker-py's ``run()`` never removes a container whose
        # start() failed, and a Stop injected into this thread while
        # create() or run() was still executing (after the daemon made
        # the container, before the handle came back) would orphan a
        # container nothing references.
        self.container_name = f"kiss-{uuid.uuid4().hex[:12]}"
        container_kwargs["name"] = self.container_name
        try:
            self.container = self.client.containers.create(
                full_image_name, **container_kwargs,
            )
            self.container.start()
        except BaseException:
            # The daemon rejected or failed the container, or a Stop
            # landed: whatever it created is removed by handle or by
            # name, and the shared volume dir created above with it
            # (``close()`` returns early with no container).
            self._discard_container()
            self._remove_shared_volume_dir()
            raise
        assert self.container is not None
        container_id = self.container.id[:12] if self.container.id else "unknown"
        print(f"Container {container_id} is now running")

    def _open_container(self) -> Container:
        """Return ONE snapshot of the open container, or raise :class:`KISSError`.

        ``close()`` (under the lifecycle lock, which command paths
        deliberately do not take: a running command must not block a
        teardown) nulls ``self.container``; the caller uses the snapshot
        only, since a second read after this guard would turn the
        orderly refusal into an AttributeError.
        """
        container = self.container
        if container is None:
            raise KISSError("No container is open. Please call open() first.")
        return container

    def Bash(  # noqa: N802
        self,
        command: str,
        description: str,
        timeout_seconds: int = 30,
        max_output_chars: int = MAX_OUTPUT_CHARS,
    ) -> str:
        """
        Execute a bash command in the running Docker container.

        Args:
            command: The bash command to execute
            description: A short description of the command in natural language
            timeout_seconds: Maximum time to wait before treating the command as hung.
            max_output_chars: Characters of output kept (head and tail),
                up to the configured ceiling.

        Returns:
            The output of the command, including stdout, stderr, and exit code
        """
        max_output_chars = _output_budget(max_output_chars)
        container = self._open_container()
        # Test hook (no-op in production): widens the window between
        # the guard above and the uses of the snapshot below so the
        # close() race is deterministic under KISS_RACE_DELAY.
        _race_delay()

        print(description)

        exists = functools.partial(self._path_exists, container, timeout_seconds, self.stop_event)
        guard, deadline = _refuse_or_lift(command, self.workdir, timeout_seconds, exists)
        if guard is not None:
            return guard
        exit_code, output = self._exec(
            container, command, deadline, self.stop_event, self.stream_callback,
        )
        # A tool-panel Stop ended the wait cooperatively (the exec is
        # already killed): surface it as the interrupted result.
        tool_interrupt.raise_if_interrupted()
        if exit_code is None:
            return _truncate_output(_timeout_message(deadline, output), max_output_chars)
        return _truncate_output(
            _with_exit_code(output, exit_code), max_output_chars,
        )

    def run_commands_parallel(
        self,
        commands: str,
        max_workers: int = 0,
        timeout_seconds: float = 1800,
        max_output_chars: int = 5000,
        description: str = "",
    ) -> str:
        """Run several independent shell commands concurrently in the container, with no LLM.

        Use this instead of ``run_parallel`` whenever the parallel work is
        just shell commands whose output you read once they finish: test
        splits (one ``pytest`` invocation per split), builds, linters,
        benchmarks.  Each command runs as its own ``docker exec`` in the
        task's container; output is not streamed live, the combined
        report is returned when the last command ends.

        Args:
            commands: A JSON-encoded list of shell command strings, e.g.
                ``'["pytest tests/a.py", "pytest tests/b.py"]'``.  Shell
                substitutions such as ``"$(cat cmds.json)"`` are not
                expanded and are rejected.
            max_workers: Maximum number of commands running at once.  ``0``
                (default) runs all of them at once.
            timeout_seconds: Timeout in seconds applied to EACH command.
            max_output_chars: Maximum characters of each command's output
                kept in the report (head and tail are kept when truncating).
            description: A brief description of the commands, shown as
                the title of this call in the chat.

        Returns:
            A report starting with a tally line
            (``N commands: K succeeded, F failed, T timed out``) followed by
            one section per command, in input order, giving its exit code,
            wall time and (truncated) output.  A string starting with
            ``Error:`` when the arguments were invalid.

        Raises:
            KISSError: If no container is open.
        """
        del description
        container = self._open_container()
        run_one = functools.partial(
            self._timed_exec, container, timeout_seconds=timeout_seconds,
        )
        return run_commands_pool(commands, run_one, max_workers, self.stop_event, max_output_chars)

    def _timed_exec(
        self,
        container: Container,
        command: str,
        cancel: threading.Event,
        timeout_seconds: float,
    ) -> tuple[int | None, str, float]:
        """Run one ``run_commands_parallel`` command; never raises.

        Args:
            container: The caller's snapshot of the open container.
            command: The bash command to execute.
            cancel: The fan-out's stop event; setting it kills the exec.
            timeout_seconds: Deadline for the command.

        Returns:
            ``(exit_code, output, seconds)``: the exit code (``None`` on
            timeout, ``-1`` when the exec failed to run or was stopped),
            the output and the wall time spent.
        """
        started = time.monotonic()
        try:  # the guard probes the container too
            exists = functools.partial(self._path_exists, container, timeout_seconds, cancel)
            guard, timeout_seconds = _refuse_or_lift(
                command, self.workdir, timeout_seconds, exists,
            )
            if guard is not None:
                return -1, guard, 0.0
            exit_code, output = self._exec(container, command, timeout_seconds, cancel)
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            exit_code, output = -1, f"Error: {e}"
        return exit_code, output, time.monotonic() - started

    def _path_exists(
        self,
        container: Container,
        timeout_seconds: float,
        cancel: threading.Event | None,
        path: str,
    ) -> bool:
        """Whether *path* may exist in *container* (the here-document guard's probe).

        The probe is an exec like any command (:meth:`_exec`), so the
        command's deadline and a Stop bound it.  A probe that gave no
        answer (timed out or stopped) counts as existing: the copy is
        refused, never run on a target nobody checked.
        """
        exit_code, _ = self._exec(
            container, f"test -e {shlex.quote(path)}", timeout_seconds, cancel,
        )
        return exit_code != 1

    def _exec(
        self,
        container: Container,
        command: str,
        timeout_seconds: float,
        cancel: threading.Event | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> tuple[int | None, str]:
        """Run *command* in *container*, bounded by *timeout_seconds*.

        The exec is created, started, drained and inspected for its
        exit code on a worker thread (see :class:`_ExecRun`) so this
        thread can enforce the deadline on every step, including the
        daemon's own ``exec_create``/``exec_start``/``exec_inspect``
        latency; this thread consumes the decoded output in 0.2 s
        slices (where an injected Stop lands) and hands each piece to
        *on_text* when given — here and not on the worker, because
        printers attribute output to a task via thread-local state.  At
        the deadline, on a Stop or on a worker failure the exec is
        cancelled: a reaper thread kills its token-tagged process tree
        and this thread waits a bounded time for that first sweep.

        Args:
            container: The caller's snapshot of the open container —
                passed in (not re-read from ``self.container``) so a
                concurrent ``close()`` cannot null the attribute
                between the caller's guard and this method's use.
            command: The bash command to execute.
            timeout_seconds: Maximum time to wait before treating the
                command as hung; the container-side process is killed.
            cancel: Optional event that ends the wait early; the
                container-side process is killed the same way.  An
                interrupt of the calling thread's tool call (see
                :func:`_stop_requested`) ends it the same way.
            on_text: Optional callback receiving each piece of output as
                it arrives (the ``Bash`` stream callback).

        Returns:
            ``(exit_code, output)`` — the exit code is ``None`` on
            timeout and ``-1`` when *cancel* or the interrupt fired (the
            output then ends with ``Killed: the task was stopped.``); the
            output is stdout followed by stderr, untruncated.  The
            verdict is taken from the stop event, the clock and the exit
            code alone, not from how the wait ended: a Stop or a deadline
            that lands while the worker's final item is being taken
            still wins over a known exit code.
        """
        run = _ExecRun(self, container.id, command)
        threading.Thread(
            target=run.run_worker, name=f"docker-exec-worker-{run.token[:8]}", daemon=True,
        ).start()
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        deadline = time.monotonic() + timeout_seconds
        try:
            while not _stop_requested(cancel):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = run.out_queue.get(timeout=min(remaining, 0.2))
                except queue.Empty:
                    continue
                if item is None:
                    if run.error is not None:
                        raise run.error
                    break
                _take_output(item, stdout_parts, stderr_parts, on_text)
            # Output the worker had decoded by the time the wait ended
            # still belongs to the command: a Stop or a deadline that
            # lands between two slices (after an empty slice, before the
            # next check) must not drop the pieces already sitting in the
            # queue.  Only those pieces are taken — the worker keeps
            # producing until it is cancelled below — and this thread is
            # the queue's only consumer, so each of them is there.
            for _ in range(run.out_queue.qsize()):
                item = run.out_queue.get_nowait()
                if item is None:
                    break
                _take_output(item, stdout_parts, stderr_parts, on_text)
        except BaseException:
            # An injected Stop (task stop watchdog, tool-panel interrupt)
            # landed between slices or in the final drain, or the worker
            # failed mid-stream: the exec must not outlive the call.
            run.cancel()
            raise
        stopped = _stop_requested(cancel)
        timed_out = time.monotonic() >= deadline
        output = "\n".join(
            part for part in ("".join(stdout_parts), "".join(stderr_parts)) if part
        )
        if stopped or timed_out or run.exit_code is None:
            run.cancel()
            if stopped:
                return -1, "\n".join(filter(None, (output, STOPPED_MESSAGE)))
            return None, output
        return run.exit_code, output

    def _tagged_exec_create(
        self, container_id: str | None, command: str, token: str,
    ) -> str:
        """Create a token-tagged bash exec and return its id.

        Builds the exec options (shell wrapping, captured streams,
        ``workdir``, the :data:`_EXEC_TOKEN_VAR` tag that
        :meth:`_kill_exec` matches on) in one place.

        Args:
            container_id: Id of the container to run the exec in
                (``Container.id`` is typed optional; docker-py resolves
                it exactly as before this helper existed).
            command: The bash command to execute.
            token: The unique tag to plant in the exec's environment.

        Returns:
            The docker exec id.
        """
        resp = self.client.api.exec_create(
            container_id,
            f"/bin/bash -c {shlex.quote(command)}",
            stdout=True,
            stderr=True,
            workdir=self.workdir,
            environment={_EXEC_TOKEN_VAR: token},
        )
        return str(resp["Id"])

    def _kill_exec(self, token: str) -> None:
        """Kill the container-side processes of a timed-out exec.

        Without this the hung command keeps running (and holding the
        stream open) for the rest of the container's life.  The exec is
        tagged with a unique environment variable, which every child
        inherits, so matching on ``/proc/<pid>/environ`` kills the whole
        tree.  ``exec_inspect``'s ``Pid`` is deliberately not used: it is
        a *host*-namespace pid and means nothing inside the container.

        Args:
            token: The unique tag given to the exec's environment.
        """
        # Snapshot: the reaper's poll loop calls this without the
        # lifecycle lock, so a concurrent close() can null
        # ``self.container`` after its own liveness check.  A closed container took the tagged
        # process tree with it — nothing is left to kill.
        container = self.container
        if container is None:
            return
        try:
            container.exec_run(["/bin/sh", "-c", kill_tagged_tree_script(token)])
        except Exception:  # pragma: no cover — container already gone
            logger.debug("could not kill timed-out exec", exc_info=True)

    def get_host_port(self, container_port: int) -> int | None:
        """Get the host port mapped to a container port.

        Args:
            container_port: The container port to look up.

        Returns:
            The host port mapped to the container port, or None if not mapped.
        """
        container = self._open_container()
        _race_delay()  # test hook: widens the guard-to-use window

        container.reload()
        port_bindings = container.attrs.get("NetworkSettings", {}).get("Ports", {})
        port_key = f"{container_port}/tcp"
        if port_key in port_bindings and port_bindings[port_key]:
            return int(port_bindings[port_key][0]["HostPort"])
        return None

    def close(self) -> None:
        """Stop and remove the Docker container.

        Handles cleanup of both the container and any temporary directories
        created for shared volumes — including a directory an earlier
        ``close()`` failed to delete, which is retried here — and
        releases the Docker client's pooled connections to the daemon
        (a later ``open()`` or a reaper still polling an exec gets a
        fresh connection on demand).
        """
        try:
            locked_call(self._lifecycle_lock, self._close_locked)
        finally:
            self.client.close()

    def _close_locked(self) -> None:
        """Body of :meth:`close`; the caller holds ``_lifecycle_lock``."""
        if self.container is None:
            # A create() a Stop interrupted (and whose cleanup a second
            # Stop interrupted) may still have made the named container.
            self._discard_container()
            print("No container to close.")
            self._remove_shared_volume_dir()
            return
        if self.attached_container is not None:
            # The caller owns an attached container: only drop the handle.
            self.container = None
            return

        container_id = self.container.id[:12] if self.container.id else "unknown"
        try:
            print(f"Stopping container {container_id}")
            self.container.stop()
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            print(f"Failed to stop container {container_id}: {e}")

        try:
            print(f"Removing container {container_id}")
            self.container.remove()
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            print(f"Failed to remove container {container_id}: {e}")

        self.container = None
        self.container_name = None
        self._remove_shared_volume_dir()
        print("Container closed successfully")

    def _discard_container(self) -> None:
        """Force-remove the container :meth:`open` created, by handle or by its name.

        The failure path of :meth:`open` and the no-handle path of
        :meth:`close`: a container the daemon made but whose handle
        never reached ``self.container`` (a Stop inside
        ``containers.create()``) is found by ``container_name``.  A
        name the daemon never created (the create call itself failed)
        is a no-op; any other failure to remove is reported and leaves
        the name on the manager, so a later :meth:`close` retries.
        """
        if self.container_name is None:
            return
        try:
            container = self.container or self.client.containers.get(self.container_name)
            container.remove(force=True)
        except docker.errors.NotFound:  # type: ignore[attr-defined]
            logger.debug("Exception caught", exc_info=True)
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            print(f"Failed to remove container {self.container_name}: {e}")
            self.container = None
            return
        self.container = None
        self.container_name = None

    def _remove_shared_volume_dir(self) -> None:
        """Delete the host side of the shared volume, if one was created.

        Shared by :meth:`close` and the failure path of :meth:`open`.
        ``host_shared_path`` is cleared only once the directory is gone
        (deleted here, or already absent): a failed ``rmtree`` keeps the
        path on the manager so the leak stays traceable and the next
        :meth:`close` or :meth:`open` retries the removal.  A repeated
        call with nothing tracked is a no-op.
        """
        path = self.host_shared_path
        if path is None:
            return
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except Exception as e:
                logger.debug("Exception caught", exc_info=True)
                print(f"Failed to clean up temp directory {path}: {e}")
                return
        self.host_shared_path = None

    def __enter__(self) -> "DockerManager":
        """Context manager entry point.

        Returns:
            DockerManager: The initialized DockerManager instance with running container.
        """
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit point.

        Args:
            exc_type: The exception type if an exception was raised.
            exc_val: The exception value if an exception was raised.
            exc_tb: The traceback if an exception was raised.
        """
        self.close()
