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
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from kiss.core.locked import locked_call

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
    killed.

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
    Object.

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
