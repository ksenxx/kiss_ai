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
import errno
import logging
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from functools import cache, partial
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


@cache
def bash_version(executable: str | None = None) -> tuple[int, int]:
    """Return the major and minor version of *executable*, or ``(0, 0)`` if unknown.

    The probe uses the same Bash path :func:`find_bash` gives shell-tool
    callers, including Git Bash outside Windows ``PATH``.  Its result is
    cached because shell scanning asks for it on every command.  An absent,
    unstartable or unparseable Bash is treated as the oldest version so
    safety checks take their conservative branch.

    Args:
        executable: Bash to probe; ``None`` uses :func:`find_bash`.

    Returns:
        ``(major, minor)``, or ``(0, 0)`` when no reliable answer is available.
    """
    path = executable or find_bash()
    if path is None:
        return (0, 0)
    try:
        result = subprocess.run(
            [path, "--noprofile", "--norc", "-c",
             'printf "%s.%s" "${BASH_VERSINFO[0]}" "${BASH_VERSINFO[1]}"'],
            capture_output=True, text=True, timeout=5, check=False,
        )
        major, minor = result.stdout.split(".", 1)
        return int(major), int(minor)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return (0, 0)


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

    A caller whose cleanup must also cover an exception raised as this
    call itself returns to it (a Stop injected into its thread) makes
    the ``Popen`` first and has :func:`start_process_group` initialise
    it inside its own ``try``, as :func:`run_process_group` does.

    Args:
        *args: Positional arguments for :class:`subprocess.Popen`.
        **kwargs: Keyword arguments for :class:`subprocess.Popen`.

    Returns:
        The started process.
    """
    proc: subprocess.Popen[Any] = object.__new__(subprocess.Popen)
    start_process_group(proc, *args, **kwargs)
    return proc


def start_process_group(proc: subprocess.Popen[Any], *args: Any, **kwargs: Any) -> None:
    """Start *proc*, a ``Popen`` made with ``object.__new__``, as :func:`popen_process_group` would.

    The ``Popen`` exists before the spawn: a Stop injected into this
    thread is raised at the check that ends a call, before its result
    is bound to any name (or inside ``Popen.__init__`` once the child is
    forked), so a ``proc = Popen(...)`` lost the child it started.  Held
    from the start, the child this call started is killed and reaped on
    any exception raised here, with its pipes closed (a child whose exec
    failed, ``__init__`` reaped itself); and a caller holding *proc*
    across the call still has the child when the exception is raised as
    this call returns to it, so its own handler can act on it.

    ``Popen.__init__`` has the same gap inside, where no handler
    reaches: it stores the pid in the bytecode after the call that
    forks, and that call blocks until the child execs.  An exception
    set meanwhile was raised before the pid existed anywhere.  This
    thread is therefore marked in ``_spawning`` for the length of
    ``Popen.__init__``: the Stop's injector leaves the exception for a
    marked thread with the mark (:func:`inject_outside_spawn`), and
    this thread sets it itself once ``Popen.__init__`` has returned, so
    it is raised here, where the child is killed.

    Args:
        proc: ``object.__new__(subprocess.Popen)``, not yet initialised.
        *args: Positional arguments for :class:`subprocess.Popen`.
        **kwargs: Keyword arguments for :class:`subprocess.Popen`.
    """
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    ident = threading.get_ident()
    held: list[Callable[[], int] | None] = [None]
    try:
        try:
            locked_call(_spawning_lock, _spawning.__setitem__, ident, held)
            subprocess.Popen.__init__(proc, *args, **kwargs)
        finally:
            # A C call, first in the block: no check point precedes it,
            # so the mark never outlives the spawn.  An injector that
            # found the mark fills ``held`` before it releases the
            # lock, so the take under that lock misses nothing.
            _spawning.pop(ident, None)
            inject = locked_call(_spawning_lock, held.pop)
            if inject is not None:
                inject()  # raised as this call returns, the child in hand
        if IS_WINDOWS:  # pragma: no cover — Windows-only branch
            _windows_enrol_in_job(proc)
    except BaseException:
        if getattr(proc, "pid", None) is not None and proc.returncode is None:
            kill_process_tree(proc)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=_REAP_GRACE_SECONDS)
            # Nothing reads the pipes of a child that never reached its
            # caller; closed here, they do not live on with the error.
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is not None:
                    pipe.close()
        raise


#: Ident of each thread inside ``Popen.__init__`` (:func:`start_process_group`) ->
#: its one-item list: the injection held back for that spawn, or None.
_spawning: dict[int, list[Callable[[], int] | None]] = {}
_spawning_lock = threading.RLock()


def inject_outside_spawn(tid: int, inject: Callable[[], int]) -> int:
    """Call *inject*, which sets an asynchronous exception for thread *tid*, outside a spawn.

    An exception set while *tid* is inside ``Popen.__init__`` is raised
    there, possibly between the fork and the statement that stores the
    child's pid, and then nobody can kill or reap that child (see
    :func:`start_process_group`).  For a thread that is spawning,
    *inject* is therefore not called here: it is left to that thread,
    which calls it as its spawn ends, however long the spawn takes (a
    blocked fork could not be interrupted anyway).  This call does not
    wait for the spawn.  One left for a spawn replaces one left earlier,
    as an exception set for a thread replaces one it has not raised yet.

    Args:
        tid: The target thread's ``ident``.
        inject: Sets the exception and returns the number of threads it was set in.

    Returns:
        What *inject* returned, or ``1`` when it was left to the spawning thread.
    """
    return locked_call(_spawning_lock, _inject_or_hold, tid, inject)


def _inject_or_hold(tid: int, inject: Callable[[], int]) -> int:
    """Under ``_spawning_lock``: leave *inject* to the spawn thread *tid* is in, else call it."""
    held = _spawning.get(tid)
    if held is None:
        return inject()
    held[0] = inject
    return 1


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
    reaped.  The leader is reaped last, after the group's last signal:
    reaped earlier, its pid (the group id) could be a stranger's by the
    time that signal is sent (see :func:`kill_process_group`), so an
    exited leader waits as a zombie while its descendants run out their
    grace.  Windows has no ``SIGTERM``: both rounds terminate the Job
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
    if not _group_exited(proc, grace):
        kill_process_tree(proc)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=grace)


def _group_exited(proc: subprocess.Popen[Any], grace: float) -> bool:
    """Wait for *proc*'s whole group to be gone, at most *grace* seconds, without reaping *proc*.

    The leader is only watched (:func:`exited_unreaped`), never reaped
    here: a zombie leader keeps its pid, and with it the group id,
    reserved, so the ``SIGKILL`` the caller sends to the group when
    this returns False still names this group and not a stranger's (a
    later session leader's, which the kernel may hand the number to
    the moment the group's last member is gone).  The zombie is itself
    a member for ``killpg(pgid, 0)``, which could therefore never
    report the group empty: the other members are counted in the
    process table instead (:func:`_other_group_members`).  A leader
    the caller reaped already (against the precondition of
    :func:`terminate_process_group`) whose pid a stranger now holds
    has no group of ours left to wait for.  On Windows the first round
    terminated the whole Job Object, so the leader's exit is the
    group's.
    """
    deadline = time.monotonic() + grace
    if not _wait_exited(proc, deadline):
        return False
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        return True
    if proc.returncode is not None and pid_alive(proc.pid):  # pragma: no cover — pid recycled
        return True
    while _other_group_members(proc.pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _other_group_members(pid: int) -> bool:
    """Whether the process table lists a member of process group *pid* other than *pid* itself.

    Read with ``ps`` (POSIX): the group's leader, kept as a zombie so
    that the group id stays reserved, is a member too, which is why
    ``killpg(pgid, 0)`` cannot answer this.  A ``ps`` that is missing
    or fails counts as members present: the caller then waits out its
    grace and kills the group it still owns, the harmless side.
    """
    try:
        result = subprocess.run(
            ["ps", "-A", "-o", "pid=,pgid="], capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover — ps missing/unresponsive
        return True
    if result.returncode != 0:  # pragma: no cover — ps failing
        return True
    group = str(pid)
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == group and fields[0] != group:
            return True
    return False


#: Seconds :func:`kill_env_marked` keeps sweeping the process table.
_SWEEP_SECONDS = 5.0


def kill_env_marked(mark: str) -> None:
    """``SIGKILL`` every process started with the environment entry *mark*, or ``mark/...``.

    *mark* is ``NAME=folder``, an entry the caller put in the environment
    of every command it started (a sub-agent's private folder; a folder
    nested in it marks the commands of a daemon one of those commands
    started).  A process keeps the entry through ``fork``, ``setsid`` and
    ``nohup``, so the ones that left their process group are found too;
    one started with a scrubbed environment (``env -i``) is not.  The
    sweep is repeated until it finds nothing, at most
    :data:`_SWEEP_SECONDS`, so a process forked while a sweep ran does
    not survive.  On Linux a process is held by a pidfd from before its
    environment is read and signalled through it: the signal reaches
    that process or none, whatever becomes of its pid.  Where there is
    no pidfd (an older kernel, a sandbox that refuses the call, macOS,
    BSD) the pid is signalled right after its environment was read, which
    shortens the time in which the number could pass to a stranger and
    does not end it.  Windows has no such process table and is skipped.

    Args:
        mark: The environment entry, ``NAME=folder``.
    """
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        return
    killed: set[int] = set()
    deadline = time.monotonic() + _SWEEP_SECONDS
    while pids := _env_marked_pids(mark):
        killed.update(pids)
        if time.monotonic() >= deadline:  # pragma: no cover — processes that outlive SIGKILL
            logger.warning("Processes carrying %s are still listed: %s", mark, pids)
            break
        time.sleep(0.05)
    if killed:
        logger.info("Killed %d processes carrying %s", len(killed), mark)


def _env_marked_pids(mark: str) -> list[int]:
    """``SIGKILL`` the processes, this one aside, whose environment holds *mark* or ``mark/...``.

    The environment is read from ``/proc/<pid>/environ``, which holds the
    one the process was started with (this user's processes only; a
    zombie's is empty), or from ``ps eww`` where there is no ``/proc``
    (macOS, BSD).  A process is signalled as soon as its environment was
    read: through a pidfd opened before the reading, or by its pid where
    none can be opened (no ``os.pidfd_open``, a kernel or seccomp profile
    without the call, no descriptor left); on the ``ps`` path by its pid,
    after its environment was read once more, alone (the listing of the
    whole table took its time).

    Returns:
        The pids found, one whose process ended before the signal included.
    """
    own = os.getpid()
    pids = []
    if os.path.exists("/proc/self/environ"):
        entry = os.fsencode(mark)
        pidfd_open = getattr(os, "pidfd_open", None)
        for name in os.listdir("/proc"):
            if not name.isdigit() or int(name) == own:
                continue
            pid, pidfd = int(name), -1
            if pidfd_open is not None:
                try:
                    pidfd = pidfd_open(pid)
                except ProcessLookupError:  # it ended meanwhile
                    continue
                except OSError:  # no pidfd: the pid is signalled below
                    pass
            try:
                with open(f"/proc/{name}/environ", "rb") as environ:
                    entries = environ.read().split(b"\0")
                if any(e == entry or e.startswith(entry + b"/") for e in entries):
                    pids.append(pid)
                    if pidfd >= 0 and sys.platform == "linux":
                        signal.pidfd_send_signal(pidfd, SIGKILL)
                    else:
                        os.kill(pid, SIGKILL)
            except OSError:  # another user's process, or one that ended meanwhile
                continue
            finally:
                if pidfd >= 0:
                    os.close(pidfd)
        return pids
    for pid in _env_marked_pids_ps(mark, own):  # pragma: no cover — no /proc: macOS, BSD
        # Read once more, alone: the listing of the whole table took its time.
        if _env_marked_pids_ps(mark, own, pid):
            pids.append(pid)
            with contextlib.suppress(OSError):  # it ended meanwhile
                os.kill(pid, SIGKILL)
    return pids


#: Times :func:`_env_marked_pids_ps` reads the processes while one replaces its command.
_PS_READINGS = 3


def _env_marked_pids_ps(mark: str, own: int, only: int | None = None) -> list[int]:
    """The pids, *own* aside, whose environment holds *mark* or ``mark/...``, from ``ps``.

    ``ps eww`` prints the environment on the line of the command, after
    it (FreeBSD's ``ps``: before it): of every process, or of the process
    with the pid *only* alone.  The command as ``ps ww`` prints it is
    taken off that line: a mark among the arguments of a command (``grep
    NAME=folder``) is no entry of its environment.  The two listings are
    not one instant.  A line its command cannot be taken off, of a
    process that replaced the command between them, names no process:
    the listings are read again, :data:`_PS_READINGS` times at most.
    """
    which = ["-A"] if only is None else ["-p", str(only)]
    marked = re.compile(rf"(?:^|\s){re.escape(mark)}(?:[/\s]|$)")
    environs: dict[int, str | None] = {}
    for _ in range(_PS_READINGS):
        lines = {
            pid: line
            for pid, line in _ps_commands(["eww", *which]).items()
            if pid != own and marked.search(line)
        }
        commands = _ps_commands(["ww", *which]) if lines else {}
        environs = {pid: _without_command(line, commands.get(pid)) for pid, line in lines.items()}
        if None not in environs.values():
            break
    return [pid for pid, environ in environs.items() if environ and marked.search(environ)]


def _without_command(line: str, command: str | None) -> str | None:
    """The environment on a process's ``ps eww`` *line*: the line without its ``ps ww`` *command*.

    Returns:
        ``""`` for a process that ended between the two listings (no
        *command*); ``None`` for one that replaced its command between them.
    """
    if not command:
        return ""
    if line.startswith(command):
        return line[len(command):]
    if line.endswith(command):
        return line[: len(line) - len(command)]
    return None


def _ps_commands(options: list[str]) -> dict[int, str]:
    """The command column of ``ps`` run with *options*, by pid."""
    try:
        listing = subprocess.run(
            ["ps", *options, "-o", "pid=,command="], capture_output=True, timeout=15,
        ).stdout.decode(errors="replace")
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover — ps missing/unresponsive
        return {}
    commands = {}
    for line in listing.splitlines():
        pid, _, command = line.strip().partition(" ")
        if pid.isdigit():
            commands[int(pid)] = command
    return commands


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
    exception), which ``subprocess.run`` would leave running.  The wait
    is taken in slices of :data:`_STOP_POLL_SECONDS`, between which an
    injected Stop is raised (never inside a blocking wait).  The reap
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
    # The Popen is held from before the spawn, which runs inside the
    # ``try``: a Stop raised as the spawning call returns reaches the
    # handler with the child in hand (see :func:`start_process_group`).
    proc: subprocess.Popen[Any] = object.__new__(subprocess.Popen)
    # Pipe -> who closes it, whichever claims it first: the thread that
    # reads it for ``communicate`` (Windows) or the handler below.
    owners: dict[Any, str] = {}
    try:
        start_process_group(
            proc,
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        if IS_WINDOWS:  # pragma: no cover — Windows-only branch
            proc._readerthread = partial(_read_claimed_pipe, owners)  # type: ignore[attr-defined]
        deadline = time.monotonic() + timeout
        while True:
            try:
                stdout, stderr = proc.communicate(
                    timeout=min(deadline - time.monotonic(), _STOP_POLL_SECONDS)
                )
                break
            except subprocess.TimeoutExpired as exc:
                if time.monotonic() >= deadline:
                    exc.timeout = timeout  # the caller's, not the last slice's
                    raise
    except BaseException:
        if getattr(proc, "pid", None) is None:
            raise  # never spawned; ``Popen.__init__`` closed what it had opened
        # ``communicate`` may have reaped the child before the exception
        # landed: its group id could be a stranger's by then (see
        # :func:`kill_process_group`), so only a live child is killed.
        if proc.returncode is None:
            kill_process_tree(proc)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=_REAP_GRACE_SECONDS)
        # The pipes go with the child: left open, they lived as long as
        # the Popen, which the raised error's traceback keeps alive for
        # as long as the caller keeps the error.  On Windows
        # ``communicate`` reads each pipe on a thread of its own, which
        # closes it at EOF; closing under its read would park this
        # thread instead.  Only a pipe no reader has claimed is closed
        # here: a reader never made or never started, and one starting
        # only now, which finds the claim taken and leaves the pipe.
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None and owners.setdefault(pipe, "handler") == "handler":
                pipe.close()
        raise
    return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)


def _read_claimed_pipe(  # pragma: no cover — Windows-only
    owners: dict[Any, str], pipe: IO[str], buffer: list[str]
) -> None:
    """Read *pipe* to EOF into *buffer* and close it, unless its claim in *owners* is taken.

    The body of a Windows ``communicate`` reader thread, in place of
    ``Popen._readerthread``, which reads unasked.  ``dict.setdefault``
    is atomic, so of the thread and the handler of
    :func:`run_process_group` exactly one gets each pipe, in any order.
    """
    if owners.setdefault(pipe, "reader") == "reader":
        try:
            buffer.append(pipe.read())
        finally:
            pipe.close()


#: How long :func:`run_capped` keeps reading a command's pipes after the
#: timeout watchdog killed its process group.
PIPE_GIVE_UP_SECONDS = 5.0

# Longest single wait on a child's pipe, so that a Stop injected into
# the waiting thread is raised within this long (see _read_some).
_STOP_POLL_SECONDS = 0.1


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
        # Reentrant and taken only through ``locked_call``: the reader
        # takes it on a task thread, where a Stop injected right after a
        # plain ``with``'s ``__enter__`` would leave it held and the
        # ``except`` handler of :func:`run_capped` would wait on it forever.
        self._guard = threading.RLock()


def _mark_capped(out: CappedOutput) -> None:
    """Label *out* capped and final unless the timer labelled it timed out; guard held."""
    if not out.timed_out:
        out.truncated = out._finished = True


def _mark_finished(out: CappedOutput) -> None:
    """Mark *out*'s result final, so the timer kills nothing more; guard held."""
    out._finished = True


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
    locked_call(out._guard, _time_out_unless_finished, proc, out)


def _time_out_unless_finished(proc: subprocess.Popen[bytes], out: CappedOutput) -> None:
    """Body of :func:`_capped_timeout`; the caller holds ``out._guard``."""
    if out._finished or proc.returncode is not None:
        return
    out.timed_out = True
    kill_process_tree(proc)


def waitid_exited(pid: int) -> os.waitid_result | None:
    """Return the exit of this process's child *pid* without reaping it, or None.

    ``waitid(WEXITED | WNOHANG | WNOWAIT)``, except that a report of a
    child that has not exited is None too: macOS answers that call for a
    STOPPED child as well (``si_code`` ``CLD_STOPPED``), as if it had
    exited.  POSIX only.

    Args:
        pid: A child of this process.

    Returns:
        The ``waitid`` result once *pid* has exited (``si_code``
        ``CLD_EXITED``, ``CLD_KILLED`` or ``CLD_DUMPED``), else None.

    Raises:
        ChildProcessError: *pid* is no unreaped child of this process.
    """
    info = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if info is None or info.si_code not in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED):
        return None
    return info


def exited_unreaped(proc: subprocess.Popen[Any]) -> bool:
    """Whether *proc* has exited, without reaping it.

    The leader stays a zombie, so its pid (and process group) cannot be
    handed to a stranger before the caller's last group kill (that of
    :func:`run_capped`, or :func:`terminate_process_group` after a
    service exited); POSIX asks with ``waitid(WNOWAIT)``.  On Windows
    the ``Popen`` handle keeps the pid reserved, so ``poll`` is as good.

    Args:
        proc: A child started with :func:`popen_process_group`.

    Returns:
        True once *proc* has exited, reaped or not.
    """
    return exit_status_unreaped(proc) is not None


def exit_status_unreaped(proc: subprocess.Popen[Any]) -> int | None:
    """Return *proc*'s exit status without reaping it, or ``None`` while it runs.

    :func:`exited_unreaped` with the status: what ``Popen.poll`` would
    report (negative for a death by signal), for a caller that must act
    on the exit but keep the leader a zombie through its last group
    signal (the CLI supervisor of :mod:`kiss.core.models.model`).  A
    child already reaped reports its ``returncode``.

    Args:
        proc: A child started with :func:`popen_process_group`.

    Returns:
        The exit status, or ``None`` while *proc* is running.
    """
    if proc.returncode is not None:
        return proc.returncode
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        return proc.poll()
    try:
        info = waitid_exited(proc.pid)
    except ChildProcessError:  # pragma: no cover — reaped by another thread meanwhile
        return proc.returncode
    if info is None:
        return None
    return info.si_status if info.si_code == os.CLD_EXITED else -info.si_status


def pid_handed_on(proc: subprocess.Popen[Any]) -> bool:
    """Whether *proc*'s pid, and so its group's id, may already belong to a stranger.

    A POSIX child's pid is free once it is reaped (``returncode`` set by
    a ``wait``): every :func:`popen_process_group` child is a session
    leader, so the kernel may hand the id to another one, whose group a
    signal would then hit (see :func:`kill_process_group`).  Windows
    never reaps: the ``Popen`` handle keeps the pid reserved, and the
    status :func:`exit_status_unreaped` reads there with ``poll`` says
    nothing about the descendants a leader left in its Job Object,
    which the group kill must still end.  A handler deciding whether to
    kill the group of a child it may no longer own asks this rather
    than ``returncode``.

    Args:
        proc: A child started with :func:`popen_process_group`.

    Returns:
        True when a group signal to *proc*'s pid could hit a stranger.
    """
    return proc.returncode is not None and not IS_WINDOWS


def signal_unreaped_child(pid: int, sig: int) -> bool:
    """Send *sig* to this process's child *pid* unless it has exited or been reaped.

    A pid is signalled only while it is this process's unreaped child
    (``waitid`` without reaping, as :func:`exit_status_unreaped` asks),
    so a pid that another thread's wait has reaped and the kernel has
    handed to a stranger is left alone.  Where the kernel has pidfds
    (Linux) the signal goes through one opened before the check: it
    names that process and no later holder of the pid, which closes the
    window between the check and the signal that ``os.kill`` leaves
    open (and ``Popen.send_signal`` accepts).  The bare pid is signalled
    only where pidfds do not exist (no ``os.pidfd_open``, a kernel or
    seccomp profile without the call); a pidfd that cannot be opened
    for another reason (no descriptor left) raises instead.  POSIX only.

    Args:
        pid: A child of this process.
        sig: The signal to send.

    Returns:
        True while *pid* is this process's unreaped child (signalled
        now, or exited and waiting to be reaped), False once reaped.

    Raises:
        OSError: The pidfd could not be opened although the kernel has
            pidfds (``EMFILE`` and the like); nothing was signalled.
    """
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd = -1
    if pidfd_open is not None:
        try:
            pidfd = pidfd_open(pid)
        except ProcessLookupError:
            return False
        except OSError as exc:
            # Without pidfds the pid is signalled below.  Not logged: callers signal under a
            # lock of their own (the browser watchdog's), which a blocked log sink would hold.
            if exc.errno not in (errno.ENOSYS, errno.EPERM):
                raise
    try:
        try:
            exited = waitid_exited(pid)
        except ChildProcessError:
            return False
        if exited is not None:
            return True
        try:
            if pidfd >= 0 and sys.platform == "linux":
                signal.pidfd_send_signal(pidfd, sig)
            else:
                os.kill(pid, sig)
        except ProcessLookupError:  # pragma: no cover — reaped between the check and the signal
            return False
        return True
    finally:
        if pidfd >= 0:
            os.close(pidfd)


def _wait_exited(proc: subprocess.Popen[bytes], deadline: float) -> bool:
    """Poll :func:`exited_unreaped` until it holds or ``time.monotonic()`` passes *deadline*."""
    while not exited_unreaped(proc):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def _drain_stderr(
    stream: IO[bytes], out: CappedOutput, keep: int, give_up: float, stop: threading.Event,
) -> None:
    """Read *stream* (stderr) to the end, *give_up* or *stop*, publishing its first *keep* bytes.

    ``out.stderr`` is updated after every chunk (an atomic reference
    store), so a caller that stops waiting for this thread while a
    stray group member still holds the pipe gets the stderr read so far.
    *stop* is that caller giving the pipe up: the read ends within a
    poll slice, so the thread neither keeps the pipe open nor rewrites
    ``out.stderr`` after the result was returned.  The pipe is closed
    here, by its only reader; found set before the first read, *stop*
    says the caller closed the pipe itself, so it is not read (see
    :func:`run_capped`; closing it once more does nothing).
    """
    kept = b""
    try:
        while not stop.is_set():
            chunk = _read_some(stream, 65536, give_up, stop)
            if not chunk:
                break
            if len(kept) < keep:
                kept = (kept + chunk)[:keep]
                out.stderr = kept.decode("utf-8", "replace")
    finally:
        stream.close()


def _windows_pipe_readable(stream: IO[bytes], seconds: float) -> bool:  # pragma: no cover — Windows
    """:func:`_pipe_readable` for Windows, where no readiness API works on a pipe.

    ``PeekNamedPipe`` is polled instead, 1 ms apart at first and twice
    as long after every miss: it reports the bytes waiting, and fails
    (a broken pipe) once the pipe is empty and every writer is gone.
    """
    import _winapi
    import msvcrt

    handle = msvcrt.get_osfhandle(stream.fileno())  # type: ignore[attr-defined]
    deadline = time.monotonic() + seconds
    pause = 0.001
    while True:
        try:
            if _winapi.PeekNamedPipe(handle)[0]:  # type: ignore[attr-defined]
                return True
        except OSError:
            return True  # EOF or a dead handle: the read reports which (b"" or the error)
        left = deadline - time.monotonic()
        if left <= 0:
            return False
        time.sleep(min(pause, left))
        pause *= 2


def _pipe_readable(stream: IO[bytes], seconds: float) -> bool:
    """Whether *stream* has bytes to read or is at EOF, waiting at most *seconds* for either."""
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        return _windows_pipe_readable(stream, seconds)
    with selectors.DefaultSelector() as ready:
        ready.register(stream, selectors.EVENT_READ)
        return bool(ready.select(seconds))


def _read_some(
    stream: IO[bytes], limit: int, give_up: float, stop: threading.Event | None = None,
) -> bytes:
    """Read up to *limit* bytes of a child's pipe; ``b""`` at EOF, past *give_up* or on *stop*.

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
    in a daemon with that many sockets and files open.  It is taken in
    slices of :data:`_STOP_POLL_SECONDS`: a Stop is an exception
    injected into the task thread (see :func:`run_capped`), raised
    between bytecodes and never inside the blocking wait, so one wait
    for the whole remaining time would hold it, and the child, back
    that long.  Windows polls the pipe for the same slices
    (:func:`_windows_pipe_readable`): a plain read there was deaf to
    the Stop, and held open by a descendant outside the Job Object.

    Args:
        stream: The child's stdout or stderr pipe.
        limit: The most bytes to return.
        give_up: ``time.monotonic()`` deadline, :data:`PIPE_GIVE_UP_SECONDS`
            past the watchdog's timeout.
        stop: Set by the caller that no longer wants the pipe read (the
            stderr drain, once the command's result was returned);
            checked once per slice.

    Returns:
        The bytes read, possibly fewer than *limit*; ``b""`` at EOF,
        once *give_up* has passed, or once *stop* is set.
    """
    remaining = give_up - time.monotonic()
    while remaining > 0:
        if stop is not None and stop.is_set():
            return b""
        if _pipe_readable(stream, min(remaining, _STOP_POLL_SECONDS)):
            return os.read(stream.fileno(), limit)
        remaining = give_up - time.monotonic()
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
    drain_stop = threading.Event()
    proc: subprocess.Popen[Any] = object.__new__(subprocess.Popen)
    err_thread = None
    timer = None
    chunks: list[bytes] = []
    size = 0
    try:
        # Every call from the spawn on is inside the ``try``, the spawn
        # included (the Popen is held from before it, see
        # :func:`start_process_group`): an injected Stop is raised at the
        # check point that ends a call (and the thread starts block on
        # the new thread's bootstrap, so one is likely delivered right
        # there), and the handler kills and reaps the child, which
        # nothing else would once the timer is dead.
        start_process_group(
            proc, cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if keep_stderr else subprocess.DEVNULL,
        )
        stdout = proc.stdout
        assert stdout is not None
        deadline = time.monotonic() + timeout
        give_up = deadline + PIPE_GIVE_UP_SECONDS
        timer = threading.Timer(timeout, _capped_timeout, args=(proc, out))
        timer.daemon = True
        if proc.stderr is not None:
            err_thread = threading.Thread(
                target=_drain_stderr,
                args=(proc.stderr, out, keep_stderr, give_up, drain_stop),
                daemon=True,
            )
            err_thread.start()
        timer.start()
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
                locked_call(out._guard, _mark_capped, out)
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
        locked_call(out._guard, _mark_finished, out)
        if err_thread is not None:
            err_thread.join(timeout=1)
            if err_thread.is_alive():
                # The command is done but a stray group member still
                # holds the stderr pipe: it has nothing left to say.
                kill_process_tree(proc)
                err_thread.join(timeout=PIPE_GIVE_UP_SECONDS)
                # A holder outside the group (its own session) survived
                # the kill: stop reading rather than leave the thread,
                # the pipe and ``out.stderr`` to it until ``give_up``.
                drain_stop.set()
                err_thread.join(timeout=1)
    except BaseException:
        if getattr(proc, "pid", None) is None:
            raise  # never spawned; ``Popen.__init__`` closed what it had opened
        # A wait cut short (a Stop landing as ``KeyboardInterrupt``, an
        # injected exception) must not leave the child running with
        # its stdout open; the reap is bounded as in
        # :func:`run_process_group`.  Stderr is its drain thread's once
        # that runs: told to stop, it closes the pipe within a poll
        # slice (closing it under its read would only turn that into an
        # error).  Until the drain runs (the exception landed in the
        # setup, or the thread could not start) the pipe is still this
        # thread's to close; a drain that starts after the check below
        # finds ``drain_stop`` set and leaves the pipe alone.  The
        # leader is unreaped here unless :func:`start_process_group`
        # already killed and reaped it (a Stop raised as the spawn
        # returned) or the exec failed (``Popen.__init__`` reaped it):
        # a reaped one's group id could be a stranger's, so the group is
        # signalled only while the pid is still ours (:func:`pid_handed_on`:
        # on Windows the ``poll`` behind :func:`_wait_exited` set
        # ``returncode`` while the Job Object, descendants included, is
        # still ours to end) and only a live child is waited for; the
        # timer must not kill again after the reap below either.
        locked_call(out._guard, _mark_finished, out)
        if not pid_handed_on(proc):
            kill_process_tree(proc)
        drain_stop.set()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None and (err_thread is None or not err_thread.is_alive()):
            proc.stderr.close()
        if proc.returncode is None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=_REAP_GRACE_SECONDS)
        raise
    finally:
        if timer is not None:
            timer.cancel()
    if exited_unreaped(proc):
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
