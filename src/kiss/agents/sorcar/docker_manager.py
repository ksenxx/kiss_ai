# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Docker library for managing Docker containers and executing commands."""

import base64
import binascii
import codecs
import functools
import glob
import json
import logging
import os
import posixpath
import queue
import re
import shlex
import shutil
import socket
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any, NamedTuple

import docker
from docker.models.containers import Container  # type: ignore[assignment]
from docker.utils import parse_repository_tag

from kiss.agents.sorcar.edit_guards import (
    MechanicalEdit,
    edit_call,
    heredoc_copy,
    heredoc_guard,
    mechanical_edit,
    preview,
    rewrite_guard,
    shadow_command,
)
from kiss.agents.sorcar.fanout_guard import SUBAGENT_DIR_PREFIX
from kiss.agents.sorcar.shell_exit import Ran, exit_word, unrun
from kiss.agents.sorcar.shell_guards import refuse_or_lift
from kiss.agents.sorcar.useful_tools import (
    _MAX_BINARY_READ_BYTES,
    _QUEUED_PIECES,
    SUBAGENT_DIR_ENV,
    _Capture,
    _output_budget,
    _searched,
    _stop_requested,
    _truncate_output,
    _until_pattern,
    cd_refusal,
    edit_args,
    followup_call,
    rewrite_redirect,
    run_commands_pool,
)
from kiss.core import tool_interrupt
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_error import KISSError
from kiss.core.locked import _race_delay, locked_call
from kiss.core.secret_redaction import StreamRedactor

logger = logging.getLogger(__name__)

#: Default cap on the characters a single command may return, matching
#: ``UsefulTools.Bash``.  Without it an unbounded ``pip install`` log goes
#: straight into the conversation and blows the model's context window.
MAX_OUTPUT_CHARS = DEFAULT_CONFIG.tool_output_default_chars

#: Line of a ``Bash`` result (before its ``[exit code: -1]`` line) whose
#: command was killed by a task Stop or a tool-panel Stop; ``DockerTools``
#: reads it to know that a follow-up command would be killed the same way.
STOPPED_MESSAGE = "Killed: the task was stopped."


class PendingAppend(NamedTuple):
    """The append a redirected ``>>`` copy stands for, until its ``Write`` lands.

    :func:`_rewrite_instead` redirects ``cat >> path`` to a ``Write`` of
    *snapshot* (the file's bytes when the redirect was raised, decoded)
    followed by *body*; the manager keeps it as ``pending_append`` so
    ``DockerTools.Write`` appends *body* to the file as it is when the
    Write lands instead (the agent's hook runs in between, and another
    writer's append must not be overwritten by the stale snapshot).  It
    belongs to that Write alone: the next call, whichever it is, takes
    or drops it.
    """

    path: str
    snapshot: bytes
    body: str


class PendingEdit(NamedTuple):
    """The edit a redirected ``sed -i`` or python rewrite stands for, until its ``Edit`` lands.

    :func:`_rewrite_instead` redirects the *rewrite* to an ``Edit`` of
    *edits*, the pairs exact for the text *path* held when the redirect
    was raised; the manager keeps it, as it keeps a
    :class:`PendingAppend`, so ``DockerTools.Edit`` makes them only
    while they are the ones exact for the text the file holds when the
    Edit lands (the agent's hook runs in between, and another writer
    may leave a text the command changes otherwise: a line holding a
    ``sed`` pattern twice).  It belongs to that Edit alone.
    """

    path: str
    edits: list[tuple[str, str, bool]]
    rewrite: MechanicalEdit


#: What a ``Bash`` call redirected to a ``Write`` or an ``Edit`` leaves for that call.
type PendingRewrite = PendingAppend | PendingEdit


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
#: Consecutive ``exec_inspect`` failures (a dropped connection, a read
#: timeout: not the exec being gone) after which the reaper stops watching a
#: start that never came: a daemon failing for this long (minutes at the
#: interval cap) is not about to start the exec.
_REAP_MAX_INSPECT_FAILURES = 60

#: How long a timed-out or stopped command waits for the reaper's first
#: kill sweep before returning: long enough that the command is normally
#: dead when the caller gets its result (one ``exec_run``, under a
#: hundred milliseconds on an idle daemon, over a second on a busy one),
#: short enough that a daemon withholding the sweep's stream cannot hold
#: the task thread.  The sweep itself continues on the reaper thread.
_KILL_WAIT_S = 2.0

#: Longest silence :meth:`DockerManager.close` accepts from its sweep of
#: the shared volume, which prints a line every 5 s while its ``find``
#: gets somewhere (:func:`_sweep_command`): a sweep at work is waited for
#: however long it takes, since what it would leave is root-owned on the
#: host, out of ``rmtree``'s reach.  Finite, because docker-py reads an
#: exec's stream without a timeout: a wedged daemon, a frozen container
#: or a ``find`` stuck in one call would hold ``close()``, and the
#: lifecycle lock, for good.  The sweep of a sub-agent's private folder
#: (:meth:`DockerManager.remove_private_dir`), which prints nothing,
#: gets this long in all.
_SWEEP_TIMEOUT_S = 30.0

#: How long :meth:`DockerManager._remove_shared_volume_dir` retries a failed
#: ``rmtree``: Docker Desktop on macOS releases the mount points it made
#: in the shared directory for nested bind mounts a moment after the
#: container's removal returns, and until then deleting them fails with
#: ``EACCES``.
_RMTREE_RETRY_S = 2.0

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
    put: Callable[[tuple[bool, str]], None],
) -> None:
    """Decode a docker exec stream into *put* until it ends.

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
        put: Receives ``(is_stderr, text)`` items (:meth:`_ExecRun.put`).
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
                    put((is_stderr, text))
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
                put((is_stderr, trailing))


def _take_output(
    item: tuple[bool, str],
    stdout: list[str] | _Capture,
    stderr: list[str] | _Capture,
    shown: tuple[StreamRedactor, StreamRedactor] | None,
) -> None:
    """Keep one decoded ``(is_stderr, text)`` piece and show it through its stream's redactor.

    *shown* holds the redactor of stdout and that of stderr; ``None`` shows nothing.
    """
    is_stderr, text = item
    (stderr if is_stderr else stdout).append(text)
    if shown is not None:
        shown[is_stderr].feed(text)


def _kept(parts: list[str] | _Capture) -> tuple[str, int]:
    """Return the text *parts* holds and the characters it dropped (a list drops none)."""
    if isinstance(parts, _Capture):
        return parts.text(), parts.dropped
    return "".join(parts), 0


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


#: Defines two ``sh`` functions over the environments in ``/proc``.
#:
#: ``marked <signal> <suffix> <entry>...`` sends *signal* to every process
#: whose environment has one of the entries (``NAME=value``; with a
#: *suffix* that is not empty, an entry ``NAME=value<suffix>...`` too), as
#: soon as its environment was read, and prints its pid: one reading of
#: the process table, however many entries.  An entry is matched whole in
#: ``/proc/<pid>/environ``: another variable that ends in its name, a
#: value that only starts with its value and a line of another variable's
#: value (a newline in a value is read as a space) are no match.
#:
#: ``kill_marked <suffix> <entry>...`` kills them all.  A process is frozen
#: (SIGSTOP) as it is found and killed once the scan has frozen them all:
#: killing one process per scan step let ``bash -c "sleep 20; touch f"``
#: continue to ``touch f`` when the glob visited ``sleep`` before its
#: parent shell (``/proc/[0-9]*`` sorts lexically, not by pid).  A shell
#: has no handle on a process beyond its pid; a frozen process starts no
#: other and does not end by itself, so its number does not pass to a
#: stranger before the kill.  The scan is repeated until it finds none (at
#: most 100 times), so a process forked while one scan ran does not survive.
MARKED_FN = (
    "nl='\n'\n"
    "marked() {\n"
    "  sig=$1 sep=${2:-$nl}; shift 2\n"
    "  for d in /proc/[0-9]*; do\n"
    '    env=$(tr "\\n\\0" " \\n" 2>/dev/null < "$d/environ")\n'
    '    for m in "$@"; do\n'
    '      case "$nl$env$nl" in *"$nl$m$nl"*|*"$nl$m$sep"*)\n'
    '        kill -$sig "${d#/proc/}" 2>/dev/null; echo "${d#/proc/}"; break;;\n'
    "      esac\n"
    "    done\n"
    "  done\n"
    "}\n"
    "kill_marked() {\n"
    "  n=0\n"
    "  while [ $n -lt 100 ]; do\n"
    '    pids=$(marked STOP "$@")\n'
    '    [ -n "$pids" ] || break\n'
    "    kill -9 $pids 2>/dev/null\n"
    "    sleep 0.05; n=$((n+1))\n"
    "  done\n"
    "}\n"
)


def kill_tagged_tree_script(token: str) -> str:
    """Return a ``sh`` script that kills every process whose environment carries *token*.

    A command the agent runs is tagged with ``KISS_EXEC_TOKEN=<token>``
    (see :func:`tagged_command`); every child inherits the variable, so
    matching on ``/proc/<pid>/environ`` finds the whole tree
    (:data:`MARKED_FN`).

    Args:
        token: The unique tag planted in the command's environment.
    """
    return MARKED_FN + f"kill_marked '' {shlex.quote(f'{_EXEC_TOKEN_VAR}={token}')}\n"


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


def remove_private_dir_script(folder: str) -> str:
    """Return a ``sh`` script that ends a sub-agent's processes in a container and deletes *folder*.

    The container's side of :func:`~kiss.core.processes.kill_env_marked`:
    every process whose environment has the entry
    :data:`SUBAGENT_DIR_ENV` ``=folder``, or that of a folder nested in
    *folder* (:meth:`DockerManager.make_private_dir` puts the entry in
    each exec, and every child inherits it), is killed
    (:data:`MARKED_FN`).  Then the folder goes, read-only directories
    in it included.

    Args:
        folder: The private folder's path in the container.
    """
    quoted = shlex.quote(folder)
    return (
        MARKED_FN
        + f"kill_marked / {shlex.quote(f'{SUBAGENT_DIR_ENV}={folder}')}\n"
        + f"chmod -R u+rwx {quoted} 2>/dev/null; rm -rf {quoted}"
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


#: Shortest interval between two runs of a ``Bash(until=...)`` command.
MIN_POLL_SECONDS = 1.0

#: Seconds the search of a run's output gets when less is left of the call's deadline:
#: the output of a run that ended at the deadline, or was killed there, still counts.
_MIN_SEARCH_SECONDS = 2.0


def _run_until[RunResult: tuple[Any, ...]](
    run: Callable[[float], RunResult],
    until: str,
    poll_seconds: float,
    timeout_seconds: float,
    cancel: threading.Event | None,
) -> tuple[RunResult, str]:
    """Re-run a command until its output matches *until* or *timeout_seconds* pass.

    The ``until`` loop of the container ``Bash`` tools: a wait for a file,
    a port or a log line is one tool call however long it takes, instead
    of a ``sleep N; grep`` turn per poll.  Each run gets the time left of
    the deadline and no run starts after it; a run killed at the deadline
    ends the loop with its timeout result.  Without a match the call
    lasts the whole *timeout_seconds*, whatever the interval.  The output
    is searched in a child process, as the host ``Bash`` searches its own
    (:func:`~kiss.agents.sorcar.useful_tools._searched`), that gets the time
    left of the deadline, at least :data:`_MIN_SEARCH_SECONDS`; a search
    that takes longer counts as no match.  An output too long to be kept
    whole is not searched and ends the loop, as the host ``Bash`` ends
    its poll: a match across or inside the gap would be wrong.

    Args:
        run: Runs the command once, bounded by the seconds it is given, and
            returns a tuple starting with ``(exit code, output)`` (``None``
            for a kill at the deadline), as the exec helpers do; a third
            element, when there is one, counts the characters dropped
            from the middle of that output (the container exec's).
        until: The regular expression (Python ``re``, searched in the
            output with ``^`` and ``$`` matching at line ends as in
            ``grep``), valid (:func:`until_refusal` checked it).
        poll_seconds: Seconds between two runs, at least :data:`MIN_POLL_SECONDS`.
        timeout_seconds: The call's deadline.
        cancel: The task's stop event (``None`` for none); a Stop ends the
            loop after the current run.

    Returns:
        ``(result, verdict)``: the last run's result as *run* returned it,
        and one line saying whether the pattern matched and after how many
        runs and seconds.
    """
    # The search child is given the pattern's text alone: the flag travels in it.
    pattern = re.compile("(?m)" + until)
    interval = max(MIN_POLL_SECONDS, float(poll_seconds))
    started = time.monotonic()
    end = started + timeout_seconds
    runs = 0
    while True:
        runs += 1
        result = run(timeout_seconds if runs == 1 else end - time.monotonic())
        exit_code, output = result[0], result[1]
        dropped = result[2] if len(result) > 2 else 0
        if dropped:
            return result, (
                f"[until {until!r}: not searched on run {runs}: the output is too long to keep "
                f"whole ({dropped:,} characters dropped); poll with a command that prints less "
                "(grep, tail)]")
        search_end = max(end, time.monotonic() + _MIN_SEARCH_SECONDS)
        if _searched(pattern, output, search_end, cancel):
            return result, (
                f"[until {until!r}: matched on run {runs} after "
                f"{time.monotonic() - started:.1f} s]")
        remaining = end - time.monotonic()
        if exit_code is None or remaining <= 0 or _stop_requested(cancel):
            break
        # Sleep one interval, or what is left of the deadline (then no further run).
        stopped = tool_interrupt.wait_interruptible(cancel or threading.Event(),
                                                    min(interval, remaining))
        if stopped or remaining <= interval:
            break
    return result, (
        f"[until {until!r}: not matched after {runs} runs and {time.monotonic() - started:.1f} s "
        f"of the {timeout_seconds:g} s timeout_seconds]")


def until_refusal(until: str, poll_seconds: float) -> str | None:
    """The one-line refusal of a ``Bash(until=...)`` call, or ``None`` when it may run.

    The pattern is refused as the host ``Bash`` refuses it
    (:func:`_until_pattern`): one that does not compile, and one the
    search child cannot be given as an argument.

    Args:
        until: The ``until`` argument (``""`` for none).
        poll_seconds: The ``poll_seconds`` argument.
    """
    pattern = _until_pattern(until)
    if isinstance(pattern, str):
        return pattern
    try:
        float(poll_seconds)
    except (TypeError, ValueError, OverflowError):
        return f"Error: poll_seconds must be a number of seconds, got {poll_seconds!r}"
    return None


def _sweep_command(shared: str, mounts: list[str]) -> str:
    """Return the shell command emptying the shared volume at *shared* but no mount below it.

    ``find -delete`` descends into every bind mount nested under the
    shared path, and a mount of the caller's (*mounts*, the container
    paths of :attr:`DockerManager.volumes`) is the caller's directory,
    not the manager's to empty; ``-xdev`` would not stop there, since a
    bind mount of the same host file system has the same device.  Each
    nested mount and everything below it is excluded by path (a glob
    pattern, so its own glob characters are escaped, and so is a
    backslash, which ``find -path`` reads as an escape and
    :func:`glob.escape` leaves as it is).  Until ``find`` ends, a
    background loop prints an empty line every 5 s in which ``find`` got
    somewhere: its CPU time or its count of context switches moved
    (``/proc``).  That tells a sweep at work from one that is frozen or
    sits in one call (:data:`_SWEEP_TIMEOUT_S`); where ``/proc`` cannot
    be read nothing tells them apart, so no line is printed and the
    sweep gets that long in all.  The loop's ``sleep`` writes nowhere,
    since the one left behind would hold the stream open.
    """
    inside = shared.rstrip("/") + "/"
    spared = ""
    for mount in mounts:
        path = posixpath.normpath(mount)
        if path.startswith(inside):
            pattern = glob.escape(path).replace("\\", "\\\\")
            spared += f" -not -path {shlex.quote(pattern)} -not -path {shlex.quote(pattern + '/*')}"
    find = f"find {shlex.quote(shared)} -mindepth 1{spared} -delete"
    # Its CPU ticks (fields 14 and 15 of ``stat``) and its context switches, nothing else:
    # a line such as the resident set size moves under a process that gets nowhere.
    ticks = "read -r x x x x x x x x x x x x x u k x < /proc/$f/stat && echo $u $k"
    seen = f"s=$( {{ {ticks}; grep ctxt /proc/$f/status; }} 2>/dev/null )"
    beat = f'{seen}; [ -z "$s" ] || [ "$s" = "$p" ] || echo; p=$s'
    return f"{find} & f=$! p=; while sleep 5 >/dev/null 2>&1; do {beat}; done & wait $f; kill $!"


def probe_command(path: str, test: str) -> str:
    """Return the shell command whose status answers the guards' question *test* about *path*.

    *test* is a ``test`` primary (``-e``, ``-d``, ``-L``), ``"run"``:
    whether bash finds the program *path* names (a path, or a bare
    name on the ``PATH`` of the command's environment, unless that
    holds a relative entry, which names another directory after the
    command's ``cd``), or ``"link"``:
    whether a ``sed -i`` on the path does what no ``Edit`` of it does,
    as the host asks before it makes one as an ``Edit``
    (:meth:`~kiss.agents.sorcar.useful_tools.UsefulTools._edit_instead`).
    It would cut a link (the path is a symlink, or a file with other
    hard links; POSIX ``find -links``), or replace a file this user
    may not write, which an ``Edit``, writing in place, is refused.
    The status is 1 only when it would do neither.
    """
    quoted = shlex.quote(path)
    if test == "run":
        found = f"command -v -- {quoted} >/dev/null"
        return found if "/" in path else f"{found} && case :$PATH: in *:[!/]*) false;; esac"
    if test == "link":
        return (
            f'[ -L {quoted} ] || [ -n "$(find {quoted} -prune -links +1 2>/dev/null)" ]'
            f" || {{ [ -f {quoted} ] && [ ! -w {quoted} ]; }}"
        )
    return f"test {test} {quoted}"


def realpath_command(path: str) -> str:
    """Return the shell command printing the path the filesystem gives the absolute *path*.

    As ``os.path.realpath`` names it on a host, for the destructive
    guard, which must ask the environment that holds the work
    directory: where a ``cd`` to the longest part of *path* that is a
    directory lands (``cd -P``: every link resolved), then the rest as
    written (a link that leads nowhere included: nothing is behind it).
    POSIX ``sh`` alone: an image need not hold ``realpath``.  The ``x``
    keeps the newlines a directory's name ends with, which a command
    substitution drops: the guard would take that directory for another.
    """
    return (
        f"p={shlex.quote(path)}; case $p in /*) ;; *) exit 2;; esac; rest=; "
        'while ! cd -P -- "$p" 2>/dev/null; do [ "$p" = / ] && exit 3; '
        "rest=/${p##*/}$rest; p=${p%/*}; p=${p:-/}; done; "
        'd=$(pwd -P && printf x) && d=${d%x} && d=${d%?} && printf \'%s\\n\' "${d%/}$rest"'
    )


class OracleStoppedError(KISSError):
    """A guard's oracle went unanswered because the task was stopped."""


def resolved_path(path: str, exit_code: int | None, output: str) -> str:
    """Return the answer of a :func:`realpath_command` run for *path*, as ``os.path.realpath``'s.

    Raises:
        KISSError: When the environment gave none (the run timed out,
            was stopped or failed): the command the guard asks for is
            then not run on a guess (:func:`_refuse_or_lift`).
    """
    if exit_code != 0:
        why = "it timed out" if exit_code is None else output.strip() or f"status {exit_code}"
        raise KISSError(f"the environment did not say where {path} leads ({why})")
    # One leading slash: bash names what a link to ``/`` leads to ``//work``.
    return posixpath.normpath("/" + output.removesuffix("\n").lstrip("/"))


def _refuse_or_lift(
    command: str,
    workdir: str | None,
    timeout_seconds: float,
    exists: Callable[[str], bool],
    isdir: Callable[[str], bool],
    realpath: Callable[[str], str],
    read_bytes: Callable[[str], bytes] | None = None,
    description: str = "",
    max_output_chars: int = MAX_OUTPUT_CHARS,
    remember: Callable[[PendingRewrite | None], None] | None = None,
    islink: Callable[[str], bool] | None = None,
    shadowed: Callable[[str, str, set[str]], bool] | None = None,
    runs: Callable[[str], bool] | None = None,
) -> tuple[str | None, float]:
    """Apply the guards of the container ``Bash`` tools to *command*.

    The destructive and install-timeout guards of the host ``Bash``
    (:func:`refuse_or_lift`; *isdir* and *realpath* tell it where the
    ``cd`` of a delete lands in the container, whose work directory is
    no path of this host), and its refusals of shell rewrites that
    belong in ``Edit``: :func:`rewrite_guard`, and :func:`heredoc_guard`,
    which calls *exists* to learn whether a here-document's target exists
    in the container (or the exec bridge's environment, whose *workdir*
    is ``None`` when unknown) and *isdir* to learn where a ``cd`` before
    it leads.  With *read_bytes*, *remember* and *islink* too (the
    ``Bash`` tools pass them; ``run_commands_parallel`` cannot redirect
    a command and passes none of the three) a literal here-document copy
    or mechanical edit is redirected to the ``Write``/``Edit`` it stands
    for (:func:`_rewrite_instead`) instead of being refused; the rest
    of a compound command then runs as a ``Bash`` call with the
    original call's *description*, *timeout_seconds* and
    *max_output_chars*, *isdir* tells whether its leading ``cd``
    names a directory there, *islink* whether a ``sed -i`` on its
    target would cut a link or replace a read-only file (then no
    Edit is exact), and *remember* keeps the :class:`PendingAppend`
    of a redirected ``>>`` copy, or the :class:`PendingEdit` of a
    rewrite whose pairs depend on the file's text, on the manager,
    after dropping (``None``) the one an earlier call left there: its
    Write or Edit was refused, and it belonged to that call alone.
    *shadowed* tells whether a python there, run in a directory there,
    loads one of the modules named from elsewhere than its own library
    (:func:`shadow_command`): a python rewrite that imports such a
    module is no Edit; without it every import is taken for the
    library's.  *runs* tells whether bash finds a program there
    (:func:`probe_command`'s ``run``): a rewrite whose program it does
    not find fails as written and is no Edit; without it every program
    is taken to be there.

    Returns:
        ``(refusal, timeout_seconds)``: the guard's message (``None``
        when the command may run; :data:`STOPPED_MESSAGE` when a Stop
        left *realpath* unanswered) and the possibly lifted deadline.

    Raises:
        ToolRedirect: The ``Write``/``Edit`` to make instead of *command*.
    """
    if remember is not None:
        remember(None)
    try:
        guard, timeout_seconds = refuse_or_lift(command, workdir, timeout_seconds, isdir, realpath)
    except OracleStoppedError:  # the task was stopped: nothing runs
        return STOPPED_MESSAGE, timeout_seconds
    except KISSError as unanswered:  # nobody knows what the delete would take
        return f"Error: the command did not run: {unanswered.args[0]}.", timeout_seconds
    if guard is None and read_bytes is not None and remember is not None and islink is not None:
        guard = _rewrite_instead(
            command, workdir, exists, isdir, islink, read_bytes, description, timeout_seconds,
            max_output_chars, remember, shadowed, runs,
        )
    # Each answer of *isdir* is an exec in the container: it is asked about
    # the ``cd`` of a command that holds a here-document only.
    refusal = guard or rewrite_guard(command) or heredoc_guard(
        command, workdir, exists, isdir if "<<" in command else None,
    )
    return refusal, timeout_seconds


def _rewrite_instead(
    command: str,
    workdir: str | None,
    exists: Callable[[str], bool],
    isdir: Callable[[str], bool],
    islink: Callable[[str], bool],
    read_bytes: Callable[[str], bytes],
    description: str,
    timeout_seconds: float,
    max_output_chars: int,
    remember: Callable[[PendingRewrite], None],
    shadowed: Callable[[str, str, set[str]], bool] | None = None,
    runs: Callable[[str], bool] | None = None,
) -> str | None:
    """Redirect a shell rewrite of a container file to the ``Write``/``Edit`` it stands for.

    A ``cat``/``tee`` here-document copy into one literal path
    (:func:`heredoc_copy`) stands for a ``Write`` of its body (after
    the current text for an append, which *remember* keeps as the
    :class:`PendingAppend` the ``Write`` settles against the file's
    bytes when it lands, whether or not the file exists yet); a python
    ``.replace`` rewrite or a one-substitution ``sed -i``
    (:func:`mechanical_edit`) for the ``Edit`` it spells out, with the
    pairs exact for the file's text (:meth:`MechanicalEdit.edits_for`,
    read from the container when they depend on it, and then kept by
    *remember* as the :class:`PendingEdit` the ``Edit`` settles against
    the text the file holds when it lands; when none is, the other
    guards decide).  The call is raised as the redirect the agent
    makes through its hook, guard and events, with the rest of a
    compound command as the follow-up ``Bash`` call
    (:func:`followup_call`); a ``cd`` to anything but a directory is
    refused up front, as bash would run nothing after it.  The
    container tracks no shown files, so the target need not have been
    read; a copy creating a new file is redirected too, and a copy into
    ``/dev/`` runs.

    Args:
        command: The shell command text.
        workdir: The directory it starts in (``None`` when unknown).
        exists: Whether a path exists in the container.
        isdir: Whether a path is a directory in the container.
        islink: Whether a ``sed -i`` on a path would cut a link there
            (a symlink, a file with other hard links: sed replaces the
            name, an Edit writes through it) or replace a file its
            user may not write (an Edit, writing in place, is
            refused): no Edit is exact and the guards decide, as on
            the host.
        read_bytes: Fetches a file's bytes from the container.
        description: The ``Bash`` call's description, for the follow-up.
        timeout_seconds: The call's deadline, for the follow-up.
        max_output_chars: The call's output budget, for the follow-up.
        remember: Keeps the pending append or edit on the manager.
        shadowed: Whether a python of the container, run in a directory
            there, loads one of the modules named from elsewhere than
            its own library (``None``: every import is taken for the
            library's).  A python rewrite that imports such a module is
            no Edit: in bash the import may fail or act.
        runs: Whether bash finds a program in the container (``None``:
            every program is taken to be there).  A rewrite whose
            program it does not find is no Edit: in bash it fails.

    Returns:
        ``None`` when *command* is no such rewrite (the other guards
        decide), else the refusal of an append whose current text could
        not be read or of a ``cd`` to a missing directory.

    Raises:
        ToolRedirect: The call to make instead.
    """
    # The container's (or the bridge environment's) home is not this
    # process's: a ``~`` target stays unresolved and heredoc_guard
    # refuses the copy, asking for the literal path.
    copy = heredoc_copy(command, workdir, home=None)
    if copy is not None and copy.path is not None and not copy.path.startswith("/dev/"):
        refusal = cd_refusal(copy.cwd, isdir)
        if refusal is not None:
            return refusal
        content = copy.body
        if copy.append:
            snapshot = b""
            if exists(copy.path):
                try:
                    snapshot = read_bytes(copy.path)
                    content = snapshot.decode("utf-8") + content
                except (OSError, UnicodeDecodeError) as e:
                    return f"Error: cannot append to {copy.path}: {e}"
            # Remembered for an absent target too: another writer may
            # create the file before the Write lands, and ``>>`` would
            # have kept its bytes.
            remember(PendingAppend(copy.path, snapshot, copy.body))
        call = f"Write(file_path={json.dumps(copy.path)}, content={preview(content)})"
        args = {"file_path": copy.path, "content": content}
        followup = followup_call(
            copy.cwd, copy.remainder, description, timeout_seconds, max_output_chars,
        )
        raise rewrite_redirect("Write", args, call, copy.append, followup)
    edit = mechanical_edit(command, workdir, shadowed, runs)
    if edit is not None and edit.path is not None:
        refusal = cd_refusal(edit.cwd, isdir)
        if refusal is not None:
            return refusal
        if edit.what != "python heredoc" and islink(edit.path):
            return None
        edits: list[tuple[str, str, bool]] | None = edit.edits
        if edit.text_dependent and exists(edit.path):
            try:
                edits = edit.edits_for(read_bytes(edit.path).decode("utf-8"))
            except (OSError, UnicodeDecodeError):
                # A text nobody could read (too large, not UTF-8) settles no
                # exact Edit: the other guards decide.
                logger.debug("could not read %s for its exact edits", edit.path, exc_info=True)
                edits = None
            if edits is not None:
                remember(PendingEdit(edit.path, edits, edit))
        if edits is None:
            return None
        followup = followup_call(
            edit.cwd, edit.remainder, description, timeout_seconds, max_output_chars,
        )
        call = edit_call(edit.path, edits)
        raise rewrite_redirect("Edit", edit_args(edit.path, edits), call, followup=followup)
    return None


#: Prints the byte size of ``{path}`` and, when it is at most ``{max}``
#: bytes, its base64 on a second line (:func:`fetched_bytes` decodes both).
#: POSIX tools only: ``base64`` reads its standard input everywhere, while
#: GNU's ``-w0`` (no line wrapping) is rejected by the BSD one of macOS,
#: so the wrapping is taken out with ``tr``.
_FETCH_SCRIPT = (
    "size=$(wc -c < {path}) && echo \"$size\" && [ \"$size\" -le {max} ]"
    " && base64 < {path} | tr -d '\\n'"
)
#: A line of the fetch's output that is (part of) the base64 payload.
_BASE64_LINE = re.compile(r"[A-Za-z0-9+/=]+")


def fetched_bytes(path: str, max_bytes: int, exit_code: int | None, output: str) -> bytes:
    """Decode what :data:`_FETCH_SCRIPT` printed for *path* into the file's bytes.

    Args:
        path: The file, for the error.
        max_bytes: The size the script was given.
        exit_code: The script's exit code (``None`` when it timed out).
        output: What it printed.

    Returns:
        The file's bytes.

    Raises:
        OSError: When the file is missing, unreadable or over *max_bytes*,
            the fetch gave no answer (timed out or stopped), or the base64
            does not decode to the announced size (the pipeline's exit
            code is ``tr``'s, so an encoder that died mid-file is caught
            here).
    """
    size, _, encoded = output.partition("\n")
    if not size.strip().isdigit():
        reason = output.strip() or ("timed out" if exit_code is None else f"exit code {exit_code}")
        raise OSError(f"cannot read {path}: {reason}")
    if int(size) > max_bytes:
        raise OSError(
            f"{path} is {size.strip()} bytes, over the {max_bytes} byte limit; use Bash tools "
            "(ffmpeg, ImageMagick) to produce a smaller artifact first"
        )
    if exit_code != 0:
        # The base64 that arrived before the failure (megabytes of it,
        # possibly) is no reason; what the pipeline printed besides it
        # (its stderr, the stop notice, after the payload) is.
        detail = " ".join(
            line.strip() for line in encoded.splitlines()
            if line.strip() and not _BASE64_LINE.fullmatch(line.strip())
        )
        what = "timed out" if exit_code is None else "did not finish"
        raise OSError(f"cannot read {path}: {detail or f'the fetch {what}'}")
    try:
        data = base64.b64decode(encoded.strip(), validate=True)
    except binascii.Error as e:
        raise OSError(f"cannot read {path}: the fetched base64 is damaged ({e})") from e
    if len(data) != int(size):
        raise OSError(f"cannot read {path}: fetched {len(data)} of its {int(size)} bytes")
    return data


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

    def __init__(self, manager: "DockerManager", container: Container, command: str) -> None:
        self.manager = manager
        # The container the exec is created in, kept by the run itself:
        # the reaper sweeps it even after a ``close()`` of an attached
        # manager nulled ``manager.container`` while the container, owned
        # by someone else, lives on with the delayed command inside.
        self.container = container
        self.container_id = container.id
        self.command = command
        self.token = uuid.uuid4().hex
        self.exec_id = ""  # set by the worker before it commits to starting
        # Output arrives piece by piece (not returned whole by
        # ``exec_start``) so a timed-out command can still report what
        # it printed before the kill.  A ``None`` item marks the end of
        # the worker, whether the exec finished or the worker failed.
        # Bounded (:meth:`put`): output read ahead of a slow taker waits
        # on the stream, not in this process.
        self.out_queue: queue.Queue[tuple[bool, str] | None] = queue.Queue(_QUEUED_PIECES)
        self.error: BaseException | None = None
        self.exit_code: int | None = None
        self.lock = threading.RLock()
        self.cancelled = False
        self.start_committed = False
        self.start_refused = False
        # Taken by the one reaper thread that sweeps; a repeat of
        # ``cancel()`` may start a second thread, which finds it taken.
        self.reaper_claim = threading.Lock()

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
            _drain_exec_stream(stream, self.put)
            self._wait_for_exit_code()
        except BaseException as exc:
            self.error = exc
        finally:
            self.put(None)

    def put(self, item: tuple[bool, str] | None) -> None:
        """Queue *item* for the task thread, waiting for room until the exec is cancelled.

        A command that prints faster than its output is taken therefore
        waits on its stream instead of filling memory.  Once the exec is
        cancelled nobody takes from the queue any more: the worker reads
        the rest of the stream, which the kill ends, and drops it.
        """
        while not self.cancelled:
            try:
                self.out_queue.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

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
        cannot leak the lock under the worker.  Safe to repeat: a call
        cut short by an injection that landed after the mark but before
        the reaper's start (:meth:`DockerManager._exec` retries it once)
        installs the reaper the first call did not; one cut short later
        starts a second thread, which exits at once because the first
        holds :attr:`reaper_claim` (no bookkeeping on this thread can
        tell the two apart: an injection can land between any two of
        its statements, so the claim is taken by the reaper itself).
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
        as its container lives (the inspect fails once the container
        is removed) — a fixed window would let a start delayed past it
        run forever, the exact bug this reaper exists to prevent — unless
        the daemon refused the start.  Only the exec being gone
        (``NotFound``) ends the watch; a failed inspect (a dropped
        connection, a read timeout) is retried, since the start may
        still come, up to :data:`_REAP_MAX_INSPECT_FAILURES` times in a
        row.  The watch is on the exec's own container, not on
        ``manager.container``: a ``close()`` of an attached manager
        drops that handle while the container lives on.  The poll
        interval backs off so an indefinitely delayed start costs one
        inspect every few seconds.

        Args:
            swept: Set once the first sweep has run (at once when another
                reaper of this run holds the claim and sweeps).
        """
        if not self.reaper_claim.acquire(blocking=False):
            swept.set()
            return
        try:
            self.manager._kill_exec(self.token, self.container)
        finally:
            swept.set()
        interval = _REAP_POLL_INTERVAL_S
        failures = 0
        while not self.start_refused:
            try:
                info = self.manager.client.api.exec_inspect(self.exec_id)
                failures = 0
            except docker.errors.NotFound:  # type: ignore[attr-defined]
                logger.debug("timed-out exec is gone", exc_info=True)
                return
            except Exception:
                logger.debug("could not inspect timed-out exec", exc_info=True)
                failures += 1
                if failures >= _REAP_MAX_INSPECT_FAILURES:
                    return
                info = {}
            if info.get("Pid"):
                self.manager._kill_exec(self.token, self.container)
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
        self.volumes = dict(volumes or {})
        self.client_shared_path = "/testbed"
        self.host_shared_path: str | None = None
        self.stream_callback: Callable[[str], None] | None = None
        # The task's stop event, when the owning agent has one:
        # ``run_commands_parallel`` kills its running execs on it.
        self.stop_event: threading.Event | None = None
        # The append a ``>>`` copy ``Bash`` redirected stands for, or the
        # edit a ``sed -i`` or python rewrite does, until the ``Write`` or
        # ``Edit`` of ``DockerTools`` settles it against the file's current
        # bytes.
        self.pending_append: PendingRewrite | None = None
        # Extra environment of every exec: a sub-agent's private folder,
        # once :meth:`make_private_dir` made it.
        self.exec_env: dict[str, str] = {}

        self.attached_container: str | None = None
        if image_name.startswith(ATTACH_PREFIX):
            self.attached_container = image_name[len(ATTACH_PREFIX):]
            self.image, self.tag = "", ""
        else:
            # docker's reference grammar, as docker-py's pull() reads it:
            # the text after the last colon is a tag only when no "/"
            # follows it (``localhost:5000/name`` is a registry port, no
            # tag), and ``name@sha256:...`` is a digest.
            self.image, parsed_tag = parse_repository_tag(image_name)
            self.tag = parsed_tag or tag

    def open(self) -> None:
        """Pull and load a Docker image, then create and start a container.

        Raises:
            KISSError: If a container is already open on this manager
                (starting a second one would orphan the first — ``close``
                only knows the newest container — and leak its shared
                volume directory), or if the previous container, or its
                shared volume directory, still cannot be removed (starting
                anyway would replace the only reference to it).
        """
        locked_call(self._lifecycle_lock, self._open_locked)

    def _open_locked(self) -> None:
        """Body of :meth:`open`; the caller holds ``_lifecycle_lock``."""
        if self.container is not None:
            raise KISSError(
                "A container is already open on this DockerManager; "
                "call close() before open() again."
            )
        if self.container_name is not None:
            # A previous close() could not remove the container (see
            # _discard_container); retry before a new one takes its
            # name on the manager and the shared volume dir it mounts.
            self._discard_container()
            if self.container_name is not None:
                raise KISSError(
                    f"The previous container {self.container_name} could not be "
                    "removed; remove it and call open() again."
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
        full_image_name = f"{image}@{tag}" if tag.startswith("sha256:") else f"{image}:{tag}"
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
        until: str = "",
        poll_seconds: float = 5.0,
    ) -> str:
        """
        Execute a bash command in the running Docker container.

        Args:
            command: The bash command to execute
            description: A short description of the command in natural language
            timeout_seconds: Maximum time to wait before treating the command as hung.
            max_output_chars: Characters of output kept (head and tail),
                up to the configured ceiling.
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
            The output of the command, including stdout, stderr, and exit code
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

        The exit is data for the agent's ``Bash`` tool to hand to the
        coding gate (:mod:`~kiss.agents.sorcar.shell_exit`): in the
        result's text the ``[exit code: N]`` line of the tool cannot be
        told from one the command printed.

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
            KISSError: If no container is open.
            ToolRedirect: The ``Write``/``Edit`` to make instead of *command*.
        """
        max_output_chars = _output_budget(max_output_chars)
        container = self._open_container()
        # Test hook (no-op in production): widens the window between
        # the guard above and the uses of the snapshot below so the
        # close() race is deterministic under KISS_RACE_DELAY.
        _race_delay()

        print(description)

        asked = (container, timeout_seconds, self.stop_event)
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
        if guard == STOPPED_MESSAGE:  # reported as a stopped exec reports it
            return Ran(_with_exit_code(guard, -1), exit_word(-1))
        if guard is not None:
            return unrun(guard)
        run = functools.partial(self._exec, container, command, cancel=self.stop_event,
                                on_text=self.stream_callback)
        verdict = ""
        if until:
            refusal = until_refusal(until, poll_seconds)
            if refusal is not None:
                return unrun(refusal)
            (exit_code, output, dropped), verdict = _run_until(
                run, until, poll_seconds, deadline, self.stop_event)
            verdict = "\n" + verdict
        else:
            exit_code, output, dropped = run(deadline)
        # A tool-panel Stop ended the wait cooperatively (the exec is
        # already killed): surface it as the interrupted result.
        tool_interrupt.raise_if_interrupted()
        # The status goes after the cut output, whole: cut with it, a small
        # budget dropped the status and a failed command read as one that passed.
        output = _truncate_output(_collapse_progress(output), max_output_chars, dropped=dropped)
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
        # The pending append or edit of a redirected rewrite belongs to the
        # Write or Edit made right after it: like any other call, this one
        # drops it.
        self.pending_append = None
        container = self._open_container()
        run_one = functools.partial(
            self._timed_exec, container, timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        )
        return run_commands_pool(commands, run_one, max_workers, self.stop_event, max_output_chars)

    def _timed_exec(
        self,
        container: Container,
        command: str,
        cancel: threading.Event,
        timeout_seconds: float,
        max_output_chars: int,
    ) -> tuple[int | None, str, float]:
        """Run one ``run_commands_parallel`` command; never raises.

        Args:
            container: The caller's snapshot of the open container.
            command: The bash command to execute.
            cancel: The fan-out's stop event; setting it kills the exec.
            timeout_seconds: Deadline for the command.
            max_output_chars: Characters kept of an output too long to be
                captured whole (head and tail).

        Returns:
            ``(exit_code, output, seconds)``: the exit code (``None`` on
            timeout, ``-1`` when the exec failed to run or was stopped),
            the output and the wall time spent.
        """
        started = time.monotonic()
        try:  # the guard probes the container too
            asked = (container, timeout_seconds, cancel)
            exists = functools.partial(self._path_test, *asked)
            guard, timeout_seconds = _refuse_or_lift(
                command, self.workdir, timeout_seconds, exists,
                functools.partial(exists, test="-d"), functools.partial(self._realpath, *asked),
            )
            if guard is not None:
                return -1, guard, 0.0
            exit_code, output, dropped = self._exec(container, command, timeout_seconds, cancel)
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return -1, f"Error: {e}", time.monotonic() - started
        if dropped:
            # Cut here, as the host runner does, so the marker counts the
            # whole output and not just the head and tail that were kept.
            output = _truncate_output(output, max_output_chars, dropped=dropped)
        return exit_code, output, time.monotonic() - started

    def _path_test(
        self,
        container: Container,
        timeout_seconds: float,
        cancel: threading.Event | None,
        path: str,
        test: str = "-e",
    ) -> bool:
        """Whether *path* may pass ``test <test>`` in *container* (the guards' probe).

        ``-e`` (the default) asks whether the path exists, the
        here-document guard's question; ``-d`` whether it is a
        directory, the question about a ``cd``; ``link`` whether a
        ``sed -i`` on it would cut a link or replace a read-only file;
        ``run`` whether bash finds the program *path* names, the
        question about the program of a rewrite
        (:func:`probe_command`).  The probe is an exec like any
        command (:meth:`_exec`), so the command's deadline and a Stop
        bound it.  A probe that gave no answer (timed out or stopped)
        fails closed either way: it counts as existing, so a copy is
        refused rather than run on a target nobody checked, as no
        directory, so a ``cd`` nobody checked authorizes no rewrite
        after it, as a link, so no Edit stands for a sed nobody
        checked, and as a program bash does not find, so none stands
        for a command nobody knows to run.
        """
        exit_code, _, _ = self._exec(
            container, probe_command(path, test), timeout_seconds, cancel,
        )
        return exit_code == 0 if test in ("-d", "run") else exit_code != 1

    def _realpath(
        self,
        container: Container,
        timeout_seconds: float,
        cancel: threading.Event | None,
        path: str,
    ) -> str:
        """Return the path the container's filesystem gives *path* (``os.path.realpath`` there).

        The destructive guard's oracle (:func:`realpath_command`): the
        work directory and the directory a delete's ``cd`` enters are
        the container's, where a link may lead back into the work
        directory.  An exec like :meth:`_path_test`'s.

        Raises:
            OracleStoppedError: When a Stop left it unanswered.
            KISSError: When the container gave no answer otherwise.
        """
        exit_code, output, _ = self._exec(
            container, realpath_command(path), timeout_seconds, cancel,
        )
        if exit_code != 0 and _stop_requested(cancel):
            raise OracleStoppedError(f"the task was stopped before it said where {path} leads")
        return resolved_path(path, exit_code, output)

    def _shadows(
        self,
        container: Container,
        timeout_seconds: float,
        cancel: threading.Event | None,
        program: str,
        path: str,
        modules: set[str],
    ) -> bool:
        """Whether the python *program*, run in *path*, may load one of *modules* from elsewhere.

        Than its own library, that is.  The ``shadowed`` oracle of
        :func:`mechanical_edit`: an exec like :meth:`_path_test`'s, in
        the environment a command gets (:func:`shadow_command`).  A
        probe that gave no answer (timed out or stopped) fails closed
        here too: the directory may hold any module, so no Edit stands
        for a program that imports one.
        """
        exit_code, _, _ = self._exec(
            container, shadow_command(program, path, modules), timeout_seconds, cancel,
        )
        return exit_code != 1

    def read_bytes(
        self,
        path: str,
        max_bytes: int = _MAX_BINARY_READ_BYTES,
        timeout_seconds: float = 60,
    ) -> bytes:
        """Return the bytes of the file *path* in the container.

        One exec prints the file's size and its base64
        (:data:`_FETCH_SCRIPT`), so neither the text decoding nor the
        output cap of :meth:`Bash` touches the bytes; a file over
        *max_bytes* is not fetched.  What ``Read`` attaches an image or
        a PDF with, and an appending here-document reads the current
        text with.

        Args:
            path: The file's path in the container (a relative one
                resolves against the working directory).
            max_bytes: The largest file fetched.
            timeout_seconds: Deadline for the exec.

        Returns:
            The file's bytes.

        Raises:
            OSError: When the file is missing, unreadable or over
                *max_bytes*, or the fetch timed out or was stopped.
            KISSError: If no container is open.
        """
        container = self._open_container()
        script = _FETCH_SCRIPT.format(path=shlex.quote(path), max=max_bytes)
        exit_code, output, _ = self._exec(
            container, script, timeout_seconds, self.stop_event, whole=True,
        )
        tool_interrupt.raise_if_interrupted()
        return fetched_bytes(path, max_bytes, exit_code, output)

    def run(self, command: str, timeout_seconds: float) -> tuple[int | None, str]:
        """Run *command* in the container: its exit code and output, uncut.

        The finish gate's primitive (the host side is
        :func:`~kiss.agents.sorcar.coding_gate.run_on_host`): no guard,
        no echo of a description and none of :meth:`Bash`'s output cap,
        since the gate parses ``-z`` file lists and hashes whole diffs
        from the output and bounds its own reads.

        Args:
            command: The bash command to execute.
            timeout_seconds: Seconds it may run; past them the code is ``None``.

        Returns:
            ``(exit code, output)``; the code is ``None`` at the deadline
            and ``-1`` when the task was stopped meanwhile.

        Raises:
            KISSError: If no container is open.
        """
        container = self._open_container()
        exit_code, output, _ = self._exec(
            container, command, timeout_seconds, self.stop_event, whole=True,
        )
        tool_interrupt.raise_if_interrupted()
        return exit_code, output

    def _exec(
        self,
        container: Container,
        command: str,
        timeout_seconds: float,
        cancel: threading.Event | None = None,
        on_text: Callable[[str], None] | None = None,
        whole: bool = False,
        sweep: bool = False,
    ) -> tuple[int | None, str, int]:
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
        and this thread waits a bounded time for that first sweep; the
        output still queued then (a bounded number of pieces,
        :meth:`_ExecRun.put`) is kept, and streamed unless the task was
        stopped.  A Stop already requested when the call is made creates
        no exec at all: left to the worker, the start would race this
        thread's cancellation, and a quick command (the store of a
        ``Write``) would be done before the sweep found it.

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
            on_text: Optional callback receiving the output as it arrives,
                with the key store's secrets redacted (the ``Bash``
                stream callback).
            whole: Keep every piece of output (``read_bytes``, whose
                fetch is bounded by its own size limit).  Otherwise each
                stream is kept as the host ``Bash`` keeps its output
                (:class:`_Capture`): whole until it passes twice
                ``_CAPTURE_CHARS``, its head and tail from then on, so a
                flood of output never sits whole in this process.
            sweep: The sweep of :meth:`close` or of
                :meth:`remove_private_dir`.  The deadline restarts at
                every piece of output, so that *timeout_seconds* bounds
                the command's silence and not its run, and no requested
                Stop ends it: the container is removed next, and with it
                the only way to delete what its root user wrote, or the
                sub-agent ends, and with it the last chance to kill what
                it left running.

        Returns:
            ``(exit_code, output, dropped)`` — the exit code is ``None``
            on timeout and ``-1`` when *cancel* or the interrupt fired
            (the output then ends with ``Killed: the task was stopped.``);
            the output is stdout followed by stderr, untruncated except
            for the *dropped* characters left out between the head and
            the tail of a stream that was kept bounded.  The verdict is
            taken from the stop event, the clock and the exit code alone,
            not from how the wait ended: a Stop or a deadline that lands
            while the worker's final item is being taken still wins over
            a known exit code.
        """
        if not sweep and _stop_requested(cancel):
            return -1, STOPPED_MESSAGE, 0
        run = _ExecRun(self, container, command)
        stdout: list[str] | _Capture = [] if whole else _Capture()
        stderr: list[str] | _Capture = [] if whole else _Capture()
        # One redactor per stream: the pieces of stdout and stderr arrive
        # interleaved, and a secret cut in two is whole within its stream only.
        shown = (StreamRedactor(on_text), StreamRedactor(on_text))
        deadline = time.monotonic() + timeout_seconds
        try:
            # Started inside the protected region: a Stop landing as
            # ``start()`` returns must still cancel the worker it started.
            threading.Thread(
                target=run.run_worker, name=f"docker-exec-worker-{run.token[:8]}", daemon=True,
            ).start()
            while sweep or not _stop_requested(cancel):
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
                _take_output(item, stdout, stderr, shown)
                if sweep:
                    deadline = time.monotonic() + timeout_seconds
            stopped = not sweep and _stop_requested(cancel)
            exit_code = None if stopped or time.monotonic() >= deadline else run.exit_code
            if exit_code is None:
                # The command is killed first; the output the worker had
                # queued is taken after that.  It still belongs to the
                # command: a Stop or a deadline that lands between two
                # slices (after an empty slice, before the next check)
                # must not drop it.  It is at most the queue's bound of
                # pieces, and after a Stop none of them is streamed, so a
                # slow callback delays neither the kill nor the Stop.
                # This thread is the queue's only consumer, so each piece
                # counted is there.
                run.cancel()
                for _ in range(run.out_queue.qsize()):
                    item = run.out_queue.get_nowait()
                    if item is None:
                        break
                    _take_output(item, stdout, stderr, None if stopped else shown)
            if not stopped:
                # The output has ended: what was held back as the possible
                # beginning of a secret is shown.
                for redactor in shown:
                    redactor.flush()
        except BaseException:
            # An injected Stop (task stop watchdog, tool-panel interrupt)
            # landed anywhere from the worker's start to the verdict's
            # own cancellation, or the worker failed mid-stream: the exec
            # must not outlive the call.  A second injection (the
            # watchdog repeats its own after 5 s; a tool-panel Stop can
            # follow a task Stop) may land inside the cancellation in
            # turn, after the run is marked cancelled but before its
            # reaper exists, and the mark alone kills nothing: the
            # cancellation is retried once, ``_mark_cancelled`` reporting
            # again whether a reaper is needed.
            try:
                run.cancel()
            except BaseException:
                run.cancel()
            raise
        (out, out_dropped), (err, err_dropped) = _kept(stdout), _kept(stderr)
        output = "\n".join(part for part in (out, err) if part)
        if stopped:
            output = "\n".join(filter(None, (output, STOPPED_MESSAGE)))
        return (-1 if stopped else exit_code), output, out_dropped + err_dropped

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
            environment={**self.exec_env, _EXEC_TOKEN_VAR: token},
        )
        return str(resp["Id"])

    @property
    def private_dir(self) -> str | None:
        """The private folder :meth:`make_private_dir` made in the container, if any."""
        return self.exec_env.get(SUBAGENT_DIR_ENV)

    def make_private_dir(self) -> str | None:
        """Make a sub-agent's private folder in the container and run every later exec in its name.

        The folder is the sub-agent's scratch dir.  Every exec from now
        on has it as its temp directory (``TMPDIR``, ``TMP``, ``TEMP``)
        and names it in :data:`SUBAGENT_DIR_ENV`, the mark
        :meth:`remove_private_dir` finds the sub-agent's processes by.

        Returns:
            The folder's path in the container, or ``None`` when it could
            not be made (the sub-agent then scratches in the work tree).
        """
        folder = f"/tmp/{SUBAGENT_DIR_PREFIX}{uuid.uuid4().hex[:8]}"
        if self.run(f"mkdir -m 700 {folder}", 30)[0] != 0:
            return None
        self.exec_env = dict.fromkeys(("TMPDIR", "TMP", "TEMP", SUBAGENT_DIR_ENV), folder)
        return folder

    def remove_private_dir(self) -> None:
        """Kill what the sub-agent's execs left running in the container and delete its folder.

        Run when the sub-agent ends, on a Stop too: the sweep is run as
        :meth:`close` runs its own (:meth:`_exec` with ``sweep``), which
        starts although the task is stopped and is given up after
        :data:`_SWEEP_TIMEOUT_S` (a container frozen under it would
        otherwise hold the sub-agent's end for good: docker-py reads an
        exec's stream without a timeout).  A Stop injected into the wait
        does not cut the sweep short: it is run again, whole, and the
        Stop is raised once it is done (the task's stop watchdog injects
        twice at most for one Stop, so the sweep is not run again
        without end).  Nothing to do when :meth:`make_private_dir` made
        no folder.
        """
        folder, container = self.private_dir, self.container
        self.exec_env = {}
        if folder is None or container is None:
            return
        stop: KeyboardInterrupt | None = None
        while True:
            try:
                self._exec(
                    container, remove_private_dir_script(folder), _SWEEP_TIMEOUT_S, sweep=True,
                )
            except KeyboardInterrupt as exc:
                stop = stop or exc
                continue
            except Exception:  # the container is already gone, and the folder with it
                logger.debug("could not remove the private folder %s", folder, exc_info=True)
            break
        if stop is not None:
            raise stop

    def _kill_exec(self, token: str, container: Container) -> None:
        """Kill the container-side processes of a timed-out exec.

        Without this the hung command keeps running (and holding the
        stream open) for the rest of the container's life.  The exec is
        tagged with a unique environment variable, which every child
        inherits, so matching on ``/proc/<pid>/environ`` kills the whole
        tree.  ``exec_inspect``'s ``Pid`` is deliberately not used: it is
        a *host*-namespace pid and means nothing inside the container.

        Args:
            token: The unique tag given to the exec's environment.
            container: The container the exec was created in (the
                reaper passes its run's own, which outlives a ``close()``
                of an attached manager).
        """
        try:
            container.exec_run(["/bin/sh", "-c", kill_tagged_tree_script(token)])
        except Exception:  # the container is already gone, and the tagged tree with it
            logger.debug("could not kill timed-out exec", exc_info=True)

    def close(self) -> None:
        """Remove the Docker container.

        Handles cleanup of both the container and any temporary directories
        created for shared volumes — including a directory an earlier
        ``close()`` failed to delete, which is retried here — and
        releases the Docker client's pooled connections to the daemon
        (a later ``open()`` or a reaper still polling an exec gets a
        fresh connection on demand).  The container is force-removed
        (one daemon call, no stop grace: its PID 1 is an interactive
        bash, which ignores the graceful stop's SIGTERM, and nothing in
        it is preserved), after it emptied the shared volume itself,
        since what its commands made there is root-owned on the host:
        that sweep is waited for as long as its stream speaks, and
        given up after :data:`_SWEEP_TIMEOUT_S` of silence.
        A Stop injected into this thread while a daemon call is in
        flight still removes the container and the directory before it
        propagates; one that lands in the sweep ends it, and what the
        sweep had not deleted yet stays on the host.
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
            if self.host_shared_path is not None:
                # Best effort: a container already gone, or a sweep ended
                # by silence, leaves the directory to rmtree.
                try:
                    self._exec(
                        self.container,
                        _sweep_command(self.client_shared_path, list(self.volumes.values())),
                        _SWEEP_TIMEOUT_S,
                        sweep=True,
                    )
                except Exception:
                    logger.debug("could not empty the shared volume", exc_info=True)
            print(f"Removing container {container_id}")
            # A container not removed keeps its name on the manager (the
            # handle is dropped) so a later close() retries by name
            # instead of forgetting a running container and deleting the
            # shared directory it still mounts.
            self._discard_container()
            self._remove_shared_volume_dir()
        except BaseException:
            # An injected Stop landed inside a daemon call or the rmtree:
            # both are idempotent and keep the name / path on failure.
            self._discard_container()
            self._remove_shared_volume_dir()
            raise
        if self.container_name is None and self.host_shared_path is None:
            print("Container closed successfully")

    def _discard_container(self) -> None:
        """Force-remove the container :meth:`open` created, by handle or by its name.

        The failure path of :meth:`open` and the removal of
        :meth:`close`: a container the daemon made but whose handle
        never reached ``self.container`` (a Stop inside
        ``containers.create()``) is found by ``container_name``.
        A name the daemon never created (the create call itself failed)
        is a no-op; any other failure to remove is reported and leaves
        the name on the manager, so a later :meth:`close` retries.
        The volumes the daemon made for the image's ``VOLUME`` entries
        are removed with the container.
        """
        if self.container_name is None:
            return
        try:
            container = self.container or self.client.containers.get(self.container_name)
            container.remove(force=True, v=True)
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
        :meth:`close` or :meth:`open` retries the removal.  While
        ``container_name`` is set the container is not known to be
        removed (:meth:`_discard_container` failed) and may still mount
        the directory, so it is kept for the retry that removes the
        container.  A repeated call with nothing tracked is a no-op.
        A failing ``rmtree`` is retried for :data:`_RMTREE_RETRY_S`.
        """
        path = self.host_shared_path
        if path is None or self.container_name is not None:
            return
        deadline = time.monotonic() + _RMTREE_RETRY_S
        while os.path.exists(path):
            try:
                shutil.rmtree(path)
            except Exception as e:
                if time.monotonic() < deadline:
                    time.sleep(0.1)
                    continue
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
