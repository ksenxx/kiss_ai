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
process group is stopped the same way before it is started again.
"""

from __future__ import annotations

import logging
import os
import subprocess
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
                self.threads.append(thread)
                thread.start()
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
        with self.lock:
            if self.stopping.is_set():
                return None
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                with open(log_dir / f"{path.name}.log", "ab") as log:
                    proc: subprocess.Popen[bytes] = popen_process_group(
                        [str(path)],
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=str(home),
                        env={**os.environ, "KISS_HOME": str(home)},
                    )
            except OSError as exc:
                logger.warning("Service %s could not start: %s", path.name, exc)
                return None
            return proc
