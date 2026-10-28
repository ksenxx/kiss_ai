# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Process probes and process-group control that behave the same on POSIX and Windows.

The POSIX idioms these wrap are actively harmful on Windows rather than
merely missing: ``os.kill(pid, 0)`` there means ``CTRL_C_EVENT`` and
delivers Ctrl+C to every process on the console (the caller included),
and ``os.killpg`` / ``start_new_session`` do not exist.  Every liveness
probe and every "stop this child and its descendants" call in the
project goes through this module.
"""

from __future__ import annotations

import contextlib
import logging
import os
import selectors
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import IO, Any

from kiss.core.locked import locked_call

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

_WINDOWS_BASH_CANDIDATES = (
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files\Git\usr\bin\bash.exe",
    r"C:\Program Files (x86)\Git\bin\bash.exe",
)


def suppress_error_dialogs() -> None:
    """Keep this process, and the children it starts, from blocking on a Windows error dialog.

    Unless a process has opted out, Windows answers some failed system
    calls with a modal dialog instead of an error code: ``CreateProcess``
    of a file that is not a valid executable, for one, shows "This
    version of %1 is not compatible with the version of Windows you're
    running" and returns only once somebody clicks it.  A process that
    was started with a hidden window or by a service has nobody to
    click, so the call blocks forever: a daemon starting a corrupt
    converter or CLI would hang instead of logging the ``OSError``.
    The opt-out is inherited by child processes, so one call at daemon
    start covers every agent and CLI it launches.  A no-op elsewhere.
    """
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        import msvcrt

        msvcrt.SetErrorMode(  # type: ignore[attr-defined]
            msvcrt.GetErrorMode()  # type: ignore[attr-defined]
            | msvcrt.SEM_FAILCRITICALERRORS  # type: ignore[attr-defined]
            | msvcrt.SEM_NOOPENFILEERRORBOX  # type: ignore[attr-defined]
        )


def find_bash() -> str | None:
    """Return the ``bash`` executable to run shell scripts with, or ``None``.

    ``shutil.which("bash")`` on every platform; on Windows, where the Git
    for Windows installer adds only ``Git\\cmd`` to ``PATH`` and leaves
    ``bash.exe`` in ``Git\\bin``, the usual install locations are probed
    as well.  ``subprocess.Popen(["bash", ...])`` there fails with
    ``WinError 2`` unless the caller passes the path this returns.

    Returns:
        Absolute path of the bash executable, or ``None`` when none is
        installed.
    """
    found = shutil.which("bash")
    if found or not IS_WINDOWS:
        return found
    for candidate in _WINDOWS_BASH_CANDIDATES:  # pragma: no cover — Windows only
        if os.path.exists(candidate):
            return candidate
    return None  # pragma: no cover — Windows without Git bash


# ``signal.SIGKILL`` does not exist on Windows.  Callers that want an
# unconditional kill pass this; :func:`kill_process_group` ignores the
# signal there anyway (``taskkill /F`` is the only option).
SIGKILL: int = getattr(signal, "SIGKILL", signal.SIGTERM)

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5


def _windows_pid_alive(pid: int) -> bool:  # pragma: no cover — Windows-only branch
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # Access denied means the process exists but belongs to someone else.
        last_error: int = ctypes.get_last_error()  # type: ignore[attr-defined]
        return last_error == _ERROR_ACCESS_DENIED
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def pid_alive(pid: int) -> bool:
    """Return whether the OS process *pid* currently exists.

    On POSIX this probes with ``os.kill(pid, 0)``, which sends no signal
    but performs the kernel's existence and permission checks; on
    Windows it opens the process handle and checks its exit status.

    Args:
        pid: The process id to probe.  ``pid <= 0`` never names a single
            process (``0`` is the caller's own process group, negative
            values address whole groups), so it is reported dead without
            signalling anything.

    Returns:
        True when the process exists (even if it belongs to another
        user), False when it does not.
    """
    if pid <= 0:
        return False
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:  # pragma: no cover — not reachable without doubles on Linux
        return False
    return True


# Windows: pid -> Job Object handle of a child started by
# :func:`popen_process_group`.  Windows only tracks parent pids, so a
# ``taskkill /T`` tree walk loses any descendant whose parent already
# exited (Git bash's fork+exec leaves exactly such orphans); a Job Object
# keeps every descendant, however re-parented, and terminates them all.
_WINDOWS_JOBS: dict[int, int] = {}
# Reentrant for ``locked_call``: both takers run on task threads (the
# Bash tool's spawn and its cleanup kill), where a Stop injection
# landing right after a plain ``with``'s ``__enter__`` would leak the
# lock for every later child of the daemon.
_WINDOWS_JOBS_LOCK = threading.RLock()
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1


def _windows_job_active_processes(job: int) -> int:  # pragma: no cover — Windows-only branch
    """Return how many processes are still alive in Job Object *job* (0 when unknown)."""
    import ctypes
    from ctypes import wintypes

    class _Accounting(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", wintypes.LARGE_INTEGER),
            ("TotalKernelTime", wintypes.LARGE_INTEGER),
            ("ThisPeriodTotalUserTime", wintypes.LARGE_INTEGER),
            ("ThisPeriodTotalKernelTime", wintypes.LARGE_INTEGER),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    info = _Accounting()
    ok = kernel32.QueryInformationJobObject(
        ctypes.c_void_p(job),
        _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
        ctypes.byref(info),
        ctypes.sizeof(info),
        None,
    )
    return int(info.ActiveProcesses) if ok else 0


def _windows_enrol_in_job(proc: subprocess.Popen[Any]) -> None:  # pragma: no cover — Windows-only
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    handle = getattr(proc, "_handle", None)
    if handle is None:
        return
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return
    process_handle = ctypes.c_void_p(int(handle))
    if not kernel32.AssignProcessToJobObject(ctypes.c_void_p(job), process_handle):
        kernel32.CloseHandle(ctypes.c_void_p(job))
        return
    locked_call(_WINDOWS_JOBS_LOCK, _windows_record_job, kernel32, proc.pid, job)


def _windows_record_job(kernel32: Any, pid: int, job: int) -> None:  # pragma: no cover — Windows
    """Register *job* for *pid*, dropping dead jobs; the caller holds ``_WINDOWS_JOBS_LOCK``."""
    import ctypes

    # A job stays reachable while ANY of its processes lives: the
    # leader may exit and leave grandchildren behind that a later
    # :func:`kill_process_group` must still be able to terminate.
    stale = [
        (p, h) for p, h in list(_WINDOWS_JOBS.items()) if _windows_job_active_processes(h) == 0
    ]
    for stale_pid, handle in stale:
        if _WINDOWS_JOBS.pop(stale_pid, None) is not None:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    # Windows reuses the pid of an exited leader whose job still
    # holds grandchildren: that job's entry is replaced here, so its
    # handle must be closed rather than leaked (closing a job handle
    # does not terminate the job without KILL_ON_JOB_CLOSE).
    reused = _WINDOWS_JOBS.pop(pid, None)
    if reused is not None:
        kernel32.CloseHandle(ctypes.c_void_p(reused))
    _WINDOWS_JOBS[pid] = job


def popen_process_group(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
    """Start a child that :func:`kill_process_group` can stop with all its descendants.

    ``subprocess.Popen`` with the child in its own group
    (``start_new_session=True`` on POSIX, ``CREATE_NEW_PROCESS_GROUP`` on
    Windows); on Windows the child is additionally placed in its own Job
    Object right after it starts (descendants inherit the job
    automatically).

    Args:
        *args: Positional arguments for :class:`subprocess.Popen`.
        **kwargs: Keyword arguments for :class:`subprocess.Popen`.

    Returns:
        The started process.
    """
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    proc: subprocess.Popen[Any] = subprocess.Popen(*args, **kwargs)
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        _windows_enrol_in_job(proc)
    return proc


def kill_process_group(pid: int, sig: int = signal.SIGTERM) -> None:
    """Signal the whole process group led by *pid*.

    POSIX sends *sig* with ``os.killpg``.  Windows has no signals: a
    child started by :func:`popen_process_group` is terminated with its
    whole Job Object (every descendant, even re-parented ones), any other
    pid with ``taskkill /F /T``, whatever *sig* is.

    Call this before the leader is reaped, or only while a member of its
    group is known to be alive: once the leader has been ``wait()``ed
    for and the group is empty, the kernel may hand *pid* to an
    unrelated session leader (every :func:`popen_process_group` child
    is one) and the signal would hit that stranger's group.  The
    helpers here do not check ``returncode`` themselves, because a
    reaped leader's grandchildren are exactly what a caller may still
    need to take down.

    Args:
        pid: The group leader (a child started with
            :func:`popen_process_group`).
        sig: The signal to send on POSIX; ignored on Windows.

    Raises:
        ProcessLookupError: When the group no longer exists, or *pid* is
            not a positive pid (``os.killpg(0, ...)`` would signal the
            caller's own group, i.e. the daemon and every test runner).
        OSError: When Windows could not terminate the tree (``taskkill``
            missing, timed out, or refused) and *pid* is still alive, so
            callers fall back to ``proc.kill()`` exactly as after a
            failed ``os.killpg``.
    """
    if pid <= 0:
        raise ProcessLookupError(pid)
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        # The job comes first, whatever the leader's state: a leader
        # that exited on its own (or on the first termination) may have
        # left grandchildren in the job, and those are exactly what the
        # caller's SIGKILL escalation must take down.
        # The handle is used under the lock: ``_windows_enrol_in_job``
        # closes (and may let the kernel reuse) the handle of a reused
        # pid under the same lock, so a snapshot taken here and used
        # after the release could terminate nothing, or another job.
        if locked_call(_WINDOWS_JOBS_LOCK, _windows_terminate_job, pid):
            return
        if not pid_alive(pid):
            raise ProcessLookupError(pid)
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OSError(f"taskkill failed for pid {pid}: {exc}") from exc
        if result.returncode != 0:
            if not pid_alive(pid):
                raise ProcessLookupError(pid)
            raise OSError(
                f"taskkill exited {result.returncode} for pid {pid}: "
                f"{result.stderr.decode(errors='replace').strip()}",
            )
        return
    os.killpg(pid, sig)


def kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    """Kill *proc* and every process in its group, falling back to *proc* alone.

    ``SIGKILL`` to the group :func:`popen_process_group` gave the child
    (its Job Object on Windows), so a launcher's grandchild (the native
    binary behind an npm wrapper, the CLI behind a ``.cmd`` shim) dies
    with it instead of living on with the pipes.  When the group is
    already gone, or Windows could not end it, the child itself is
    killed.  Same precondition as :func:`kill_process_group`: call it
    before *proc* is reaped, or only while a group member is known to
    be alive.

    Args:
        proc: A child started with :func:`popen_process_group`.
    """
    try:
        kill_process_group(proc.pid, SIGKILL)
    except OSError:
        proc.kill()


def terminate_process_group(proc: subprocess.Popen[Any], grace: float = 5.0) -> None:
    """``SIGTERM`` *proc*'s group, escalate to ``SIGKILL`` after *grace* seconds, reap.

    The graceful first signal lets a child clean up (git removes its
    ``.lock`` files, Xvfb its socket); a group with any member still
    alive *grace* seconds later (the leader, or a descendant that
    ignored the signal after the leader honoured it) is killed with
    :func:`kill_process_tree` and given another *grace* seconds to be
    reaped.  Windows has no ``SIGTERM``: both rounds terminate the Job
    Object.  Same precondition as :func:`kill_process_group`: a caller
    that has already ``poll()``ed *proc* as exited must skip this call
    (``if proc.poll() is None: terminate_process_group(proc)``) unless
    it knows a descendant is still running.

    Args:
        proc: A child started with :func:`popen_process_group`.
        grace: Seconds to wait after each signal.
    """
    try:
        kill_process_group(proc.pid, signal.SIGTERM)
    except OSError:
        proc.terminate()
    if _group_exited(proc, grace):
        return
    kill_process_tree(proc)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=grace)


def _group_exited(proc: subprocess.Popen[Any], grace: float) -> bool:
    """Reap *proc* and wait for its whole group to be gone, at most *grace* seconds.

    The leader is reaped first: an unreaped leader is a zombie that
    still counts as a group member, so ``killpg(pgid, 0)`` could never
    report the group empty.  Once it is reaped, POSIX keeps its pid
    reserved as the group id for as long as any member lives, so the
    probe names this group and nothing else; a ``ProcessLookupError``
    means every member is gone.  On Windows the first round terminated
    the whole Job Object, so the leader's exit is the group's.
    """
    deadline = time.monotonic() + grace
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        return False
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        return True
    while True:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:  # pragma: no cover — needs a setuid descendant
            pass  # a member we may not signal still exists
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


# Seconds a killed child is given to exit before the caller stops waiting
# for it.
_REAP_GRACE_SECONDS = 2.0


def run_process_group(
    args: list[str], timeout: float, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run *args* to completion in its own process group, bounded by *timeout*.

    Like ``subprocess.run(args, capture_output=True, text=True,
    timeout=timeout)`` with stdin closed, but a child that outlives
    *timeout* is killed with its whole group (:func:`kill_process_tree`)
    rather than alone, and so is a child whose wait any other exception
    cuts short (a Stop landing as ``KeyboardInterrupt`` or an injected
    exception), which ``subprocess.run`` would leave running.  The reap
    after the kill waits for the child itself, at most
    :data:`_REAP_GRACE_SECONDS`, never for its pipes: a grandchild that
    outlived the kill in its own session (or on Windows without
    ``taskkill``) is left behind instead of parking the caller until it
    exits.

    Args:
        args: The command line.
        timeout: Seconds the child may run.
        env: The child environment, or ``None`` to inherit.

    Returns:
        The completed process with its decoded output.

    Raises:
        OSError: When the command could not be started.
        subprocess.TimeoutExpired: When the child did not exit within
            *timeout*; it and its group have been killed and reaped.
    """
    proc = popen_process_group(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except BaseException:
        kill_process_tree(proc)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_REAP_GRACE_SECONDS)
        raise
    return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)


#: How long :func:`run_capped` keeps reading a command's pipes after the
#: timeout watchdog killed its process group.
PIPE_GIVE_UP_SECONDS = 5.0


class CappedOutput:
    """Result of :func:`run_capped`.

    Attributes:
        returncode: The exit status, ``-1`` when the command never exited.
        stdout: At most ``limit`` bytes of stdout (cut back to the last
            complete line when ``last_full_line`` was asked and the
            output is incomplete).
        stderr: The first ``keep_stderr`` bytes of stderr, decoded.
        truncated: More stdout existed than was kept.
        timed_out: The command outlived ``timeout`` and was killed.
            Never set together with *truncated*.
    """

    def __init__(self) -> None:
        self.returncode = -1
        self.stdout = b""
        self.stderr = ""
        self.truncated = False
        self.timed_out = False
        # Set (under ``_guard``) once the result is decided: the cap was
        # hit or the process exited.  Stdout EOF alone does not decide
        # it (a process may close stdout and run on), so the timer can
        # tell a slow command from a finished one.  No kill of the
        # process group happens after it is set, and the leader is
        # reaped only after it (see :func:`run_capped`).
        self._finished = False
        self._guard = threading.Lock()


def _capped_timeout(proc: subprocess.Popen[bytes], out: CappedOutput) -> None:
    """Flag the timeout and kill the tree unless the result is already decided.

    Decided on ``out._finished`` rather than ``proc.poll()``: the leader
    may have exited while a descendant still holds the stdout pipe, and
    that read would otherwise block for as long as the descendant lives.
    Runs from the timer (the reader is blocked on the pipe) and from
    the reader itself when the process outlives stdout EOF.  The kill
    happens under the guard, and the reader sets ``_finished`` under it
    before reaping the leader, so the kill can never hit a reaped pid
    (see :func:`kill_process_group`); a leader found reaped anyway
    (``returncode`` set) is left alone for the same reason.
    """
    with out._guard:
        if out._finished or proc.returncode is not None:
            return
        out.timed_out = True
        kill_process_tree(proc)


def _exited(proc: subprocess.Popen[bytes]) -> bool:
    """Whether *proc* has exited, without reaping it.

    The leader stays a zombie, so its pid (and process group) cannot be
    handed to a stranger before :func:`run_capped`'s last group kill;
    POSIX asks with ``waitid(WNOWAIT)``.  On Windows the ``Popen``
    handle keeps the pid reserved, so ``poll`` is as good.
    """
    if proc.returncode is not None:
        return True
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        return proc.poll() is not None
    return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _wait_exited(proc: subprocess.Popen[bytes], deadline: float) -> bool:
    """Poll :func:`_exited` until it holds or ``time.monotonic()`` passes *deadline*."""
    while not _exited(proc):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def _drain_stderr(stream: IO[bytes], out: CappedOutput, keep: int, give_up: float) -> None:
    """Read *stream* (stderr) to the end or *give_up*, publishing its first *keep* bytes.

    ``out.stderr`` is updated after every chunk (an atomic reference
    store), so a caller that stops waiting for this thread while a
    stray group member still holds the pipe gets the stderr read so far.
    """
    kept = b""
    while True:
        chunk = _read_some(stream, 65536, give_up)
        if not chunk:
            break
        if len(kept) < keep:
            kept = (kept + chunk)[:keep]
            out.stderr = kept.decode("utf-8", "replace")


def _read_some(stream: IO[bytes], limit: int, give_up: float) -> bytes:
    """Read up to *limit* bytes of a child's pipe; ``b""`` at EOF or past *give_up*.

    A plain ``read`` would stay blocked for as long as ANY process
    holds the pipe's write end: a hook git ran that started a daemon
    in its own session (``setsid``) survives the watchdog's
    process-group kill with the pipe inherited, and the caller (the
    merge thread, which holds the tab's ``is_merging`` claim) would
    sit in ``read`` until that daemon exits — or, if that daemon keeps
    writing, keep reading its output for as long as it runs.  The read
    therefore waits for the pipe only until *give_up*, one deadline
    shared by every read of the command, and returns nothing once it
    has passed even when output is ready.  The wait uses the
    platform's readiness API (``epoll``/``kqueue``), not ``select``,
    whose ``FD_SETSIZE`` limit rejects a pipe numbered 1024 or higher
    in a daemon with that many sockets and files open.  On Windows the
    watchdog's Job Object kills every descendant, so a plain read
    cannot be held open there (and readiness APIs do not work on pipes).

    Args:
        stream: The child's stdout or stderr pipe.
        limit: The most bytes to return.
        give_up: ``time.monotonic()`` deadline, :data:`PIPE_GIVE_UP_SECONDS`
            past the watchdog's timeout.

    Returns:
        The bytes read, possibly fewer than *limit*; ``b""`` at EOF or
        once *give_up* has passed.
    """
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        return stream.read(limit)
    remaining = give_up - time.monotonic()
    if remaining > 0:
        with selectors.DefaultSelector() as ready:
            ready.register(stream, selectors.EVENT_READ)
            if ready.select(remaining):
                return os.read(stream.fileno(), limit)
    logger.warning(
        "child pipe still open %ss after the kill; giving up", PIPE_GIVE_UP_SECONDS,
    )
    return b""


def run_capped(
    cmd: list[str],
    *,
    limit: int,
    timeout: float,
    cwd: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    keep_stderr: int = 0,
    last_full_line: bool = False,
) -> CappedOutput:
    """Run *cmd*, keeping at most *limit* bytes of its stdout.

    Unlike ``subprocess.run(capture_output=True)`` the whole output is
    never held in memory: stdout is read in chunks and the process
    group is killed as soon as the cap is exceeded (``truncated``) or
    *timeout* seconds pass (``timed_out``), so a search over a huge
    tree or a diff of a huge worktree costs the daemon at most *limit*
    bytes; an exception that cuts the wait short (``KeyboardInterrupt``)
    kills the group and reaps the child before propagating.  The
    command runs in its own process group with stdin
    closed: ``read`` returns EOF only once every writer of the pipe is
    gone, so a grandchild that inherited stdout (an external diff
    tool, a pager) must die with the command or the timeout could
    never be enforced; one that escaped into its own session is given
    up on :data:`PIPE_GIVE_UP_SECONDS` after the kill.

    Args:
        cmd: The command line.
        limit: Most stdout bytes to keep.
        timeout: Seconds the command may run before it is killed.
        cwd: Working directory, or ``None`` for the caller's.
        env: The child environment, or ``None`` to inherit.
        keep_stderr: Stderr bytes to keep in ``stderr``; ``0`` discards
            stderr without reading it.
        last_full_line: Cut an incomplete ``stdout`` (capped or timed
            out) back to its last complete line, dropping that newline;
            a head without any newline is kept as is.

    Returns:
        The :class:`CappedOutput`.

    Raises:
        OSError: When the command could not be started.
    """
    out = CappedOutput()
    proc = popen_process_group(
        cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if keep_stderr else subprocess.DEVNULL,
    )
    stdout = proc.stdout
    assert stdout is not None
    deadline = time.monotonic() + timeout
    give_up = deadline + PIPE_GIVE_UP_SECONDS
    err_thread = None
    if proc.stderr is not None:
        err_thread = threading.Thread(
            target=_drain_stderr, args=(proc.stderr, out, keep_stderr, give_up), daemon=True,
        )
        err_thread.start()
    timer = threading.Timer(timeout, _capped_timeout, args=(proc, out))
    timer.daemon = True
    timer.start()
    chunks: list[bytes] = []
    size = 0
    try:
        while size < limit:
            chunk = _read_some(stdout, min(65536, limit - size), give_up)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        else:
            # Exactly *limit* bytes read: one more byte tells a longer
            # output from one that ends here.  Decided under the guard,
            # before the kill: a timer firing meanwhile must not relabel
            # the capped result as timed out, and a timer that already
            # did wins.
            if _read_some(stdout, 1, give_up):
                with out._guard:
                    if not out.timed_out:
                        out.truncated = out._finished = True
        if out.truncated:
            kill_process_tree(proc)
        stdout.close()
        # Stdout EOF is not exit: the deadline stays armed until the
        # process is gone (the timer may fire meanwhile and kill it).
        # The leader is only watched, not reaped, until every group
        # kill is behind us: a reaped leader's pid may already be a
        # stranger's (see :func:`kill_process_group`).
        if not _wait_exited(proc, deadline):
            _capped_timeout(proc, out)
            # A SIGKILLed process that still does not exit is stuck in
            # the kernel; report it (returncode -1) rather than wait.
            _wait_exited(proc, time.monotonic() + 5)
        with out._guard:
            out._finished = True
        if err_thread is not None:
            err_thread.join(timeout=1)
            if err_thread.is_alive():
                # The command is done but a stray group member still
                # holds the stderr pipe: it has nothing left to say.
                kill_process_tree(proc)
                err_thread.join(timeout=PIPE_GIVE_UP_SECONDS)
    except BaseException:
        # A wait cut short (a Stop landing as ``KeyboardInterrupt``, an
        # injected exception) must not leave the child running with
        # its stdout open; the reap is bounded as in
        # :func:`run_process_group`.  Stderr is left to its drain
        # thread, which sees EOF once the group is dead (closing the
        # pipe under its read would only turn that into an error).  The
        # leader is still unreaped here, so the group kill is safe; the
        # timer must not kill again after the reap below.
        with out._guard:
            out._finished = True
        kill_process_tree(proc)
        stdout.close()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_REAP_GRACE_SECONDS)
        raise
    finally:
        timer.cancel()
    if _exited(proc):
        out.returncode = proc.wait()
    data = b"".join(chunks)
    if last_full_line and (out.truncated or out.timed_out) and b"\n" in data:
        data = data[: data.rfind(b"\n")]
    out.stdout = data
    return out


def _windows_terminate_job(pid: int) -> bool:  # pragma: no cover — Windows-only branch
    """Terminate the Job Object of *pid*, if any; the caller holds ``_WINDOWS_JOBS_LOCK``."""
    job = _WINDOWS_JOBS.get(pid)
    if job is None:
        return False
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    return bool(kernel32.TerminateJobObject(ctypes.c_void_p(job), 1))


def _windows_process_identity(pid: int) -> str | None:  # pragma: no cover — Windows-only branch
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        size = wintypes.DWORD(32768)
        image = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
            return None
        return f"{created} {image.value}"
    finally:
        kernel32.CloseHandle(handle)


def process_identity(pid: int) -> str | None:
    """Return a fingerprint of *pid* (start time + executable) that a recycled pid cannot share.

    Callers record it when they capture a pid and compare before every
    signal, so a process that exited and had its pid reassigned to a
    stranger is never killed.  POSIX reads ``ps -o lstart= -o command=``;
    Windows reads the process creation time and full image path through
    ``kernel32`` (there is no ``ps``).

    Args:
        pid: Process id to fingerprint.

    Returns:
        The identity string, or ``None`` when the process is gone or
        could not be inspected.
    """
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        return _windows_process_identity(pid)
    try:
        result = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "lstart=", "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover — ps missing/unresponsive
        return None
    return result.stdout.strip() or None
