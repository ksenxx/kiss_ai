# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test helper: spawn a child on a fresh PTY without ``pty.fork()``.

``pty.fork()`` calls ``os.forkpty()`` which raises a
``DeprecationWarning`` in CPython 3.14+ when the calling process is
multi-threaded (pytest's session is multi-threaded due to capture,
logging and IO plugins). The warning is not cosmetic — it points at a
real deadlock hazard: any thread that holds a mutex at the moment of
fork carries the locked mutex into the child, where no thread exists
to release it.

This helper avoids the warning *and* the underlying hazard by

1. allocating the PTY pair in the parent with :func:`pty.openpty`, and
2. spawning the child via :class:`subprocess.Popen`.

``subprocess.Popen`` invokes the C-level ``_posixsubprocess.fork_exec``
which is engineered to be safe in multi-threaded parents (it forks and
immediately ``execve``s without re-entering the Python runtime), so no
``DeprecationWarning`` is emitted.

A small ``preexec_fn`` runs after ``setsid`` in the child to acquire
the slave PTY as the controlling terminal, so the parent can deliver
job-control signals (e.g. ``Ctrl+C`` via writing ``\\x03`` to the
master fd) through the line discipline exactly as ``pty.fork()`` did.
"""

from __future__ import annotations

import fcntl
import os
import pty
import signal
import subprocess
import termios


def pty_spawn(argv: list[str], env: dict[str, str] | None = None) -> tuple[int, int]:
    """Spawn *argv* on a fresh PTY and return ``(pid, master_fd)``.

    Drop-in replacement for ``pty.fork()`` followed by ``os.execvp`` in
    the child branch, but without invoking ``os.forkpty()`` (which is
    deprecated in multi-threaded processes from Python 3.14).

    Args:
        argv: Command line to ``execve`` in the child (for example
            ``[sys.executable, "-c", code]``).
        env: Optional environment mapping for the child. When ``None``
            the parent's environment is inherited unchanged, matching
            the behaviour of the old ``pty.fork()`` + ``os.execvp``
            pattern that callers previously used.

    Returns:
        A ``(pid, master_fd)`` tuple equivalent to the result of
        ``pty.fork()`` in the parent process. Write bytes to
        ``master_fd`` to feed the child's stdin, read from it to receive
        stdout/stderr, and reap the child with ``os.waitpid(pid, 0)``.
        The caller owns ``master_fd`` and is responsible for closing it.
    """
    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            argv,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            start_new_session=True,
            preexec_fn=_acquire_ctty,
            close_fds=True,
            env=env,
        )
    except BaseException:
        # If Popen fails (e.g. invalid argv / env), the parent still
        # owns both ends of the PTY and must release them so the test
        # process does not leak fds. ``slave_fd`` is closed in the
        # second ``finally`` below; close ``master_fd`` here.
        os.close(master_fd)
        os.close(slave_fd)
        raise
    else:
        # The child has inherited (and dup2'd) the slave fd into its
        # standard streams; the parent only needs the master end.
        os.close(slave_fd)
    return proc.pid, master_fd


def _acquire_ctty() -> None:
    """Make the inherited stdin fd (the PTY slave) the controlling terminal.

    Runs in the freshly forked child after subprocess has dup2'd the
    slave PTY onto fds 0/1/2 and after ``setsid`` has placed the child
    in a new session (because we passed ``start_new_session=True``). At
    this point fd 0 is a tty with no controlling session, so a
    ``TIOCSCTTY`` ioctl claims it. Without this the parent cannot
    deliver SIGINT through ``\\x03`` because the child has no foreground
    process group on the PTY.

    It also resets terminal-signal dispositions (SIGINT, SIGQUIT,
    SIGTERM, SIGHUP) to ``SIG_DFL``.  POSIX requires a non-interactive
    shell to start asynchronous (``cmd &``) children with SIGINT/SIGQUIT
    *ignored*, and ignored dispositions survive fork+exec — so when the
    test session itself is launched as a background job (CI runners,
    ``nohup pytest &``, parallel test drivers), every child would
    inherit ``SIG_IGN`` for SIGINT.  CPython only installs its
    ``KeyboardInterrupt`` handler when SIGINT is *not* ignored at
    startup, so a ``\\x03`` written to the PTY master would silently do
    nothing and every Ctrl+C test would fail.  Resetting to ``SIG_DFL``
    before ``exec`` restores the disposition the tests (and real
    interactive use) assume.

    Returns:
        None. ``TIOCSCTTY`` failures are silenced because on some
        platforms (or when re-entered) the fd is already the
        controlling terminal — that is the desired end state anyway.
    """
    for sig in (signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, signal.SIG_DFL)
    try:
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    except OSError:
        # Already the controlling terminal, or platform/kernel quirk —
        # nothing actionable here; the test will surface a real problem
        # if signal delivery is broken downstream.
        pass
