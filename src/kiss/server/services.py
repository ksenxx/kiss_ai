# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A brand's background services: the ``$KISS_HOME/services.d/`` programs run with the daemon.

A brand drops an executable into ``$KISS_HOME/services.d/`` for a helper that
must live exactly as long as the daemon, such as Seamless Loop's hub relay,
which lets the user's browser reach this daemon's web app through the
brand's hub from any device, without a tunnel.  The daemon starts every
executable file there (dot files skipped) once its local endpoint file is
written (:meth:`Services.start`), restarts one that exits after
:data:`RESTART_MIN_SECONDS`, doubling up to :data:`RESTART_MAX_SECONDS` while
it keeps failing, and stops them all at shutdown (:meth:`Services.stop`:
``SIGTERM`` to the process group, ``SIGKILL`` when ignored).  Each runs in its
own process group with ``KISS_HOME`` set and its working directory there; its
output is appended to ``$KISS_HOME/logs/services/<name>.log``.  A service
stays in the foreground: when it exits, whatever it left running in its
process group is stopped the same way before it is started again.  A daemon
that dies without stopping its services (``SIGKILL``, a crash) still takes
them along (:data:`TIED`; not on Windows, which has no ``sh``).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from kiss.core.config import kiss_home
from kiss.core.processes import exited_unreaped, popen_process_group, terminate_process_group

logger = logging.getLogger(__name__)

SERVICES_DIR = "services.d"
RESTART_MIN_SECONDS = 1.0
RESTART_MAX_SECONDS = 60.0
STABLE_SECONDS = 60.0
"""A run at least this long resets the restart delay to :data:`RESTART_MIN_SECONDS`."""
STOP_GRACE_SECONDS = 5.0
EXIT_POLL_SECONDS = 0.1
"""How often a supervising thread checks whether its service exited."""
TIED = (
    'exec 3<&0; (read -r eof <&3; trap "" TERM; kill -TERM 0; '
    f'sleep {STOP_GRACE_SECONDS:g}; kill -KILL 0) & exec "$1" </dev/null 3<&-'
)
"""The ``sh`` script that becomes service ``$1``, tied to the life of the daemon.

Its standard input is a pipe whose write end only the daemon holds.  A
subshell left in the service's process group reads it: the end of the file
means the daemon is gone, however it died, and the subshell stops the group
as :meth:`Services.stop` would.  The ``exec`` keeps the service the leader
of that group, reading ``/dev/null``.
"""


def service_paths(home: Path) -> list[Path]:
    """The executables in ``<home>/services.d``, by name (none when the directory is absent)."""
    directory = home / SERVICES_DIR
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if not path.name.startswith(".") and path.is_file() and os.access(path, os.X_OK)
    )


class Services:
    """Supervises the ``services.d`` executables of one daemon."""

    def __init__(self) -> None:
        """Supervise the services of :func:`kiss_home` (resolved when started)."""
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        """Start a supervising thread for every executable in ``services.d``.

        Idempotent once services run, and a no-op after :meth:`stop`.
        """
        home = kiss_home()
        with self.lock:
            if self.threads or self.stopping.is_set():
                return
            for path in service_paths(home):
                thread = threading.Thread(
                    target=self._supervise,
                    args=(path, home),
                    name=f"service-{path.name}",
                    daemon=True,
                )
                thread.start()  # one the OS refuses is not kept: ``stop`` joins started ones
                self.threads.append(thread)
        for thread in self.threads:
            logger.info("Started service %s", thread.name.removeprefix("service-"))

    def stop(self) -> None:
        """Stop every service and its supervising thread; later starts do nothing.

        Each supervising thread stops its own service: only the thread that
        reaps a service signals its process group, so a group id freed by a
        reaped service is never signalled.
        """
        with self.lock:
            self.stopping.set()
            threads = list(self.threads)
        for thread in threads:
            thread.join(timeout=3 * STOP_GRACE_SECONDS)

    def _supervise(self, path: Path, home: Path) -> None:
        """Run *path* until :meth:`stop`, restarting it with a growing delay while it fails."""
        delay = RESTART_MIN_SECONDS
        while not self.stopping.is_set():
            started = time.monotonic()
            proc = self._spawn(path, home)
            if proc is not None:
                while not exited_unreaped(proc):
                    if self.stopping.wait(EXIT_POLL_SECONDS):
                        break
                # The leader is not reaped yet, so its group id is still this
                # service's: stop the service (when stopping) and whatever it
                # left running, then reap it.
                terminate_process_group(proc, STOP_GRACE_SECONDS)
                if proc.stdin is not None:  # the tie, once nothing is left for it to stop
                    proc.stdin.close()
                if not self.stopping.is_set():
                    logger.warning(
                        "Service %s exited with status %s", path.name, proc.returncode
                    )
            if time.monotonic() - started >= STABLE_SECONDS:
                delay = RESTART_MIN_SECONDS
            if self.stopping.wait(delay):
                return
            delay = min(delay * 2, RESTART_MAX_SECONDS)

    def _spawn(self, path: Path, home: Path) -> subprocess.Popen[bytes] | None:
        """Start *path* in its own process group; ``None`` when stopping or it cannot start."""
        log_dir = home / "logs" / "services"
        # Windows has no ``sh``.  PID 1 needs no tie (its death ends its PID namespace)
        # and reaps no orphan: each watcher would stay a zombie in its service's group.
        tied = sys.platform != "win32" and os.getpid() != 1
        with self.lock:
            if self.stopping.is_set():
                return None
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                with open(log_dir / f"{path.name}.log", "ab") as log:
                    proc: subprocess.Popen[bytes] = popen_process_group(
                        ["sh", "-c", TIED, "service", str(path)] if tied else [str(path)],
                        stdin=subprocess.PIPE if tied else subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=str(home),
                        env={**os.environ, "KISS_HOME": str(home)},
                    )
            except OSError as exc:
                logger.warning("Service %s could not start: %s", path.name, exc)
                return None
            return proc
