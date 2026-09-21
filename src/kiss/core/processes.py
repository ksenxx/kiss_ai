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

import os
import signal
import subprocess
import sys
import threading
from typing import Any

IS_WINDOWS = sys.platform == "win32"

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


def new_process_group_kwargs() -> dict[str, Any]:
    """Return the :class:`subprocess.Popen` keyword arguments that put a child in its own group.

    ``start_new_session=True`` on POSIX, ``CREATE_NEW_PROCESS_GROUP`` on
    Windows.  Prefer :func:`popen_process_group`, which also enrols the
    Windows child in a Job Object so :func:`kill_process_group` reaches
    every descendant.

    Returns:
        A dict to splat into ``subprocess.Popen(...)``.
    """
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
    return {"start_new_session": True}


# Windows: pid -> Job Object handle of a child started by
# :func:`popen_process_group`.  Windows only tracks parent pids, so a
# ``taskkill /T`` tree walk loses any descendant whose parent already
# exited (Git bash's fork+exec leaves exactly such orphans); a Job Object
# keeps every descendant, however re-parented, and terminates them all.
_WINDOWS_JOBS: dict[int, int] = {}
_WINDOWS_JOBS_LOCK = threading.Lock()


def _windows_enrol_in_job(proc: subprocess.Popen[Any]) -> None:  # pragma: no cover — Windows-only
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return
    process_handle = ctypes.c_void_p(int(proc._handle))  # type: ignore[attr-defined]
    if not kernel32.AssignProcessToJobObject(ctypes.c_void_p(job), process_handle):
        kernel32.CloseHandle(ctypes.c_void_p(job))
        return
    with _WINDOWS_JOBS_LOCK:
        stale = [(p, h) for p, h in list(_WINDOWS_JOBS.items()) if not pid_alive(p)]
        for pid, handle in stale:
            if _WINDOWS_JOBS.pop(pid, None) is not None:
                kernel32.CloseHandle(ctypes.c_void_p(handle))
        _WINDOWS_JOBS[proc.pid] = job


def popen_process_group(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
    """Start a child that :func:`kill_process_group` can stop with all its descendants.

    ``subprocess.Popen`` plus :func:`new_process_group_kwargs`; on Windows
    the child is additionally placed in its own Job Object right after it
    starts (descendants inherit the job automatically).

    Args:
        *args: Positional arguments for :class:`subprocess.Popen`.
        **kwargs: Keyword arguments for :class:`subprocess.Popen`.

    Returns:
        The started process.
    """
    kwargs.update(new_process_group_kwargs())
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
            :func:`new_process_group_kwargs`).
        sig: The signal to send on POSIX; ignored on Windows.

    Raises:
        ProcessLookupError: When the group no longer exists.
        OSError: When Windows could not terminate the tree (``taskkill``
            missing, timed out, or refused) and *pid* is still alive, so
            callers fall back to ``proc.kill()`` exactly as after a
            failed ``os.killpg``.
    """
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        if not pid_alive(pid):
            raise ProcessLookupError(pid)
        with _WINDOWS_JOBS_LOCK:
            job = _WINDOWS_JOBS.get(pid)
        if job is not None:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            if kernel32.TerminateJobObject(ctypes.c_void_p(job), 1):
                return
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, check=False, timeout=15,
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
