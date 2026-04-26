"""Cron Manager Daemon — thin Unix-socket proxy over the system ``crontab``.

Runs as a daemon process listening on ``~/.kiss/cron_manager.sock`` for
add / list / remove commands from any KISS agent.  Scheduling, execution
and persistence are delegated to the host's cron service; this daemon
only marshals JSON commands into ``crontab -l`` / ``crontab -`` calls.

KISS-owned crontab entries are framed by a marker comment so user-authored
lines are never disturbed::

    # KISS-JOB <job_id> <created_at>
    <schedule> <command>

Daemon control::

    python -m kiss.agents.third_party_agents.cron_manager_daemon start [--foreground]
    python -m kiss.agents.third_party_agents.cron_manager_daemon stop
    python -m kiss.agents.third_party_agents.cron_manager_daemon status

Client usage from any KISS agent::

    from kiss.agents.third_party_agents.cron_manager_daemon import CronClient

    client = CronClient()
    job_id = client.add_job("*/5 * * * * echo hello")
    jobs = client.list_jobs()
    client.remove_job(job_id)
    client.stop_daemon()
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

KISS_DIR = Path.home() / ".kiss"
SOCK_PATH = KISS_DIR / "cron_manager.sock"
PID_PATH = KISS_DIR / "cron_manager.pid"
LOG_PATH = KISS_DIR / "cron_manager.log"

_MAX_MSG = 1_048_576

_MARKER = "# KISS-JOB"


def _read_crontab(crontab_cmd: str) -> str:
    """Return the current user's crontab contents.

    Args:
        crontab_cmd: Path or name of the ``crontab`` executable.

    Returns:
        The crontab text, or ``""`` when the user has no crontab installed.

    Raises:
        RuntimeError: If ``crontab -l`` fails for a reason other than an
            empty crontab.
    """
    result = subprocess.run(
        [crontab_cmd, "-l"], capture_output=True, text=True, check=False
    )
    if result.returncode == 0:
        return result.stdout
    if "no crontab" in result.stderr.lower():
        return ""
    raise RuntimeError(
        f"`{crontab_cmd} -l` failed (rc={result.returncode}): {result.stderr.strip()}"
    )


def _write_crontab(content: str, crontab_cmd: str) -> None:
    """Replace the current user's crontab with ``content``.

    A trailing newline is appended if missing.

    Args:
        content: Full crontab text to install.
        crontab_cmd: Path or name of the ``crontab`` executable.

    Raises:
        RuntimeError: If ``crontab -`` rejects the input.
    """
    if content and not content.endswith("\n"):
        content += "\n"
    result = subprocess.run(
        [crontab_cmd, "-"],
        input=content,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"`{crontab_cmd} -` failed (rc={result.returncode}): {result.stderr.strip()}"
        )


@dataclass(frozen=True)
class CronJob:
    """A KISS-owned cron entry parsed from the system crontab.

    Attributes:
        id: Short unique identifier used to remove the job later.
        schedule: Five-field cron expression (minute hour dom month dow).
        command: The shell command scheduled to run.
        created_at: ISO-8601 timestamp recorded when the job was added.
    """

    id: str
    schedule: str
    command: str
    created_at: str


def _parse_kiss_jobs(content: str) -> list[CronJob]:
    """Extract KISS-owned jobs from raw crontab ``content``, in order.

    Entries whose marker line is malformed, orphaned (no schedule line
    follows), or whose schedule line does not have five fields plus a
    command are silently skipped.

    Args:
        content: Full crontab text.

    Returns:
        The list of parsed :class:`CronJob` instances in file order.
    """
    jobs: list[CronJob] = []
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(_MARKER):
            parts = line.split(None, 3)
            if len(parts) == 4 and i + 1 < len(lines):
                job_id, created_at = parts[2], parts[3]
                fields = lines[i + 1].split(None, 5)
                if len(fields) == 6:
                    jobs.append(
                        CronJob(job_id, " ".join(fields[:5]), fields[5], created_at)
                    )
                    i += 2
                    continue
        i += 1
    return jobs


def _remove_kiss_block(content: str, job_id: str) -> tuple[str, bool]:
    """Strip the two-line KISS block for ``job_id`` from ``content``.

    Non-KISS lines and other KISS blocks are left untouched.

    Args:
        content: Full crontab text.
        job_id: Job identifier to remove.

    Returns:
        ``(new_content, removed)`` — ``new_content`` equals ``content``
        untouched when ``removed`` is ``False``.
    """
    lines = content.splitlines()
    kept: list[str] = []
    removed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(_MARKER):
            parts = line.split(None, 3)
            if len(parts) == 4 and parts[2] == job_id and i + 1 < len(lines):
                i += 2
                removed = True
                continue
        kept.append(line)
        i += 1
    if not removed:
        return content, False
    new_content = "\n".join(kept)
    if kept:
        new_content += "\n"
    return new_content, True


def _validate_entry(entry: str) -> None:
    """Raise :class:`ValueError` unless ``entry`` is a non-empty single line.

    Args:
        entry: A full crontab line, e.g. ``"*/5 * * * * echo hello"``.

    Raises:
        ValueError: If the entry is blank or contains newlines.
    """
    if not entry.strip():
        raise ValueError("Crontab entry must not be empty")
    if "\n" in entry or "\r" in entry:
        raise ValueError("Crontab entry must be a single line")


class CronDaemon:
    """Unix-domain-socket daemon proxying job commands to system ``crontab``.

    The daemon owns no scheduling state — each incoming request is
    translated into one or two ``crontab`` invocations under a mutex so
    concurrent add/remove operations are race-free.  System cron performs
    the actual scheduling, execution, and persistence.

    Args:
        sock_path: Path for the Unix domain socket the daemon listens on.
        pid_path: Path for the daemon PID file.
        crontab_cmd: Name or absolute path of the ``crontab`` executable.
            Tests may point this at a fake binary; production uses ``"crontab"``.
    """

    def __init__(
        self,
        sock_path: Path = SOCK_PATH,
        pid_path: Path = PID_PATH,
        crontab_cmd: str = "crontab",
    ) -> None:
        self.sock_path = sock_path
        self.pid_path = pid_path
        self.crontab_cmd = crontab_cmd
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._server_socket: socket.socket | None = None


    def _add_job(self, entry: str) -> dict[str, Any]:
        """Add a new job to the system crontab and return a response dict.

        Args:
            entry: The exact crontab line to install,
                e.g. ``"*/5 * * * * echo hello"``.
        """
        try:
            _validate_entry(entry)
        except ValueError as e:
            return {"status": "error", "message": str(e)}
        job_id = uuid.uuid4().hex[:12]
        created_at = datetime.now().isoformat()
        block = f"{_MARKER} {job_id} {created_at}\n{entry}\n"
        with self._lock:
            try:
                current = _read_crontab(self.crontab_cmd)
                if current and not current.endswith("\n"):
                    current += "\n"
                _write_crontab(current + block, self.crontab_cmd)
            except RuntimeError as e:
                return {"status": "error", "message": str(e)}
        logger.info("Added job %s: %s", job_id, entry)
        return {"status": "ok", "job_id": job_id}

    def _remove_job(self, job_id: str) -> dict[str, Any]:
        """Remove a job from the system crontab and return a response dict."""
        with self._lock:
            try:
                current = _read_crontab(self.crontab_cmd)
                new_content, removed = _remove_kiss_block(current, job_id)
                if not removed:
                    return {"status": "error", "message": f"Job {job_id!r} not found"}
                _write_crontab(new_content, self.crontab_cmd)
            except RuntimeError as e:
                return {"status": "error", "message": str(e)}
        logger.info("Removed job %s", job_id)
        return {"status": "ok", "message": f"Job {job_id!r} removed"}

    def _list_jobs(self) -> dict[str, Any]:
        """Return a response dict listing every KISS-owned crontab entry."""
        with self._lock:
            try:
                current = _read_crontab(self.crontab_cmd)
            except RuntimeError as e:
                return {"status": "error", "message": str(e)}
        jobs = [
            {
                "id": j.id,
                "schedule": j.schedule,
                "command": j.command,
                "created_at": j.created_at,
            }
            for j in _parse_kiss_jobs(current)
        ]
        return {"status": "ok", "jobs": jobs}

    def _get_status(self) -> dict[str, Any]:
        """Return a response dict with the daemon PID and KISS job count."""
        try:
            with self._lock:
                count = len(_parse_kiss_jobs(_read_crontab(self.crontab_cmd)))
        except RuntimeError as e:
            return {
                "status": "ok",
                "pid": os.getpid(),
                "job_count": -1,
                "warning": str(e),
            }
        return {"status": "ok", "pid": os.getpid(), "job_count": count}


    def _handle_command(self, data: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a parsed JSON command and return the response payload."""
        action = data.get("action")
        if action == "add":
            entry = data.get("entry", "")
            if not entry:
                return {"status": "error", "message": "Missing 'entry'"}
            return self._add_job(entry)
        if action == "remove":
            job_id = data.get("job_id", "")
            if not job_id:
                return {"status": "error", "message": "Missing 'job_id'"}
            return self._remove_job(job_id)
        if action == "list":
            return self._list_jobs()
        if action == "status":
            return self._get_status()
        if action == "stop":
            self._stop_event.set()
            return {"status": "ok", "message": "Daemon stopping"}
        return {"status": "error", "message": f"Unknown action: {action!r}"}


    def _handle_client(self, conn: socket.socket) -> None:
        """Read one JSON request from ``conn``, dispatch, and send the reply."""
        try:
            conn.settimeout(10.0)
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > _MAX_MSG:
                    conn.sendall(
                        json.dumps(
                            {"status": "error", "message": "Message too large"}
                        ).encode()
                    )
                    return
            if not chunks:
                return
            raw = b"".join(chunks)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                conn.sendall(
                    json.dumps({"status": "error", "message": "Invalid JSON"}).encode()
                )
                return
            response = self._handle_command(data)
            conn.sendall(json.dumps(response).encode())
        except (OSError, TimeoutError):
            logger.debug("Client connection error", exc_info=True)
        finally:
            conn.close()

    def _serve(self) -> None:
        """Accept incoming connections until the stop event is set."""
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.sock_path.exists():
            self.sock_path.unlink()

        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.bind(str(self.sock_path))
        self._server_socket.listen(16)
        self._server_socket.settimeout(1.0)
        logger.info("Listening on %s (pid=%d)", self.sock_path, os.getpid())

        while not self._stop_event.is_set():
            try:
                conn, _ = self._server_socket.accept()
                threading.Thread(
                    target=self._handle_client, args=(conn,), daemon=True
                ).start()
            except TimeoutError:
                continue
            except OSError:
                if not self._stop_event.is_set():
                    logger.error("Socket accept error", exc_info=True)
                break

    def _cleanup(self) -> None:
        """Close the server socket and remove the socket / PID files."""
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
        for path in (self.sock_path, self.pid_path):
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    def run(self) -> None:
        """Run the daemon in the current process.

        Writes the PID file, installs signal handlers (when invoked from the
        main thread), enters the socket-accept loop, and cleans up on exit.
        """
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(os.getpid()))

        def _signal_handler(signum: int, frame: Any) -> None:
            logger.info("Received signal %d, shutting down", signum)
            self._stop_event.set()

        try:
            signal.signal(signal.SIGTERM, _signal_handler)
            signal.signal(signal.SIGINT, _signal_handler)
        except ValueError:
            pass

        try:
            self._serve()
        finally:
            self._stop_event.set()
            self._cleanup()
            logger.info("Daemon stopped")


def _daemonize() -> None:
    """Double-fork to detach from the controlling terminal.

    Redirects stdio to ``/dev/null``; logs are written through the
    ``logging`` module to a file configured by :func:`start_daemon`.
    """
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)


def _read_pid(pid_path: Path = PID_PATH) -> int | None:
    """Return the PID recorded in ``pid_path`` or ``None`` if unreadable."""
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return None


def _is_running(pid: int) -> bool:
    """Return ``True`` if a process with ``pid`` exists and is reachable."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_daemon(
    sock_path: Path = SOCK_PATH,
    pid_path: Path = PID_PATH,
    foreground: bool = False,
) -> str:
    """Start the cron manager daemon.

    Does nothing if an instance is already running.  When ``foreground``
    is ``False`` (the default) the process double-forks first.  The call
    blocks for the lifetime of the daemon when run in the foreground.

    Args:
        sock_path: Path for the Unix domain socket.
        pid_path: Path for the PID file.
        foreground: If ``True``, run in the foreground without daemonizing.

    Returns:
        A human-readable status message (returned only after the daemon
        exits; normally a long-running call).
    """
    pid = _read_pid(pid_path)
    if pid is not None and _is_running(pid):
        return f"Daemon already running (pid={pid})"

    for p in (sock_path, pid_path):
        if p.exists():
            p.unlink()

    if not foreground:
        _daemonize()

    sock_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = sock_path.parent / "cron_manager.log"
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    CronDaemon(sock_path=sock_path, pid_path=pid_path).run()
    return "Daemon stopped"


def stop_daemon(sock_path: Path = SOCK_PATH, pid_path: Path = PID_PATH) -> str:
    """Ask a running daemon to exit.

    Tries a graceful stop through the socket first, then falls back to
    ``SIGTERM`` against the PID in ``pid_path``.

    Args:
        sock_path: Path of the daemon's Unix domain socket.
        pid_path: Path of the daemon's PID file.

    Returns:
        Human-readable status string.
    """
    pid = _read_pid(pid_path)
    if pid is None:
        return "Daemon not running (no PID file)"
    if not _is_running(pid):
        pid_path.unlink(missing_ok=True)
        return "Daemon not running (stale PID file cleaned)"

    def _stopped() -> bool:
        return not pid_path.exists() or not _is_running(pid)

    CronClient(sock_path=sock_path).stop_daemon()
    for _ in range(50):
        if _stopped():
            return "Daemon stopped"
        time.sleep(0.1)

    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if _stopped():
            break
        time.sleep(0.1)
    return "Daemon stopped"


def daemon_status(
    pid_path: Path = PID_PATH, sock_path: Path = SOCK_PATH
) -> str:
    """Return a human-readable description of the daemon's running state."""
    pid = _read_pid(pid_path)
    if pid is None:
        return "Daemon not running (no PID file)"
    if not _is_running(pid):
        return f"Daemon not running (stale PID file for pid={pid})"
    try:
        resp = CronClient(sock_path=sock_path).status()
        return f"Daemon running (pid={pid}, jobs={resp.get('job_count', '?')})"
    except (ConnectionError, OSError):
        return f"Daemon running (pid={pid}) but socket not responding"


class CronClient:
    """Client for the Cron Manager Daemon.

    Sends a single JSON command per connection over the Unix domain
    socket and returns the parsed JSON reply.

    Args:
        sock_path: Path of the daemon's Unix domain socket.
            Defaults to ``~/.kiss/cron_manager.sock``.
    """

    def __init__(self, sock_path: Path = SOCK_PATH) -> None:
        self.sock_path = sock_path

    def _send(self, data: dict[str, Any]) -> dict[str, Any]:
        """Send ``data`` to the daemon and return the decoded response.

        Raises:
            ConnectionError: If the socket cannot be reached or the
                response is not valid JSON.
        """
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(10.0)
            sock.connect(str(self.sock_path))
            sock.sendall(json.dumps(data).encode())
            sock.shutdown(socket.SHUT_WR)
            chunks: list[bytes] = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
            if not raw:
                return {"status": "error", "message": "Empty response"}
            result: dict[str, Any] = json.loads(raw)
            return result
        except (OSError, json.JSONDecodeError) as e:
            raise ConnectionError(f"Cannot reach daemon at {self.sock_path}: {e}") from e
        finally:
            sock.close()

    def add_job(self, entry: str) -> str:
        """Add a cron job and return its identifier.

        The caller supplies the exact crontab line — no parsing is
        performed by the client or the daemon.

        Args:
            entry: Full crontab line, e.g. ``"*/5 * * * * echo hello"``.

        Returns:
            The 12-character job identifier.

        Raises:
            ConnectionError: If the daemon is not reachable.
            RuntimeError: If the daemon rejects the request.
        """
        resp = self._send({"action": "add", "entry": entry})
        if resp.get("status") != "ok":
            raise RuntimeError(resp.get("message", "Unknown error"))
        job_id: str = resp["job_id"]
        return job_id

    def remove_job(self, job_id: str) -> None:
        """Remove the cron job with identifier ``job_id``.

        Raises:
            ConnectionError: If the daemon is not reachable.
            RuntimeError: If the daemon rejects the request (e.g. unknown id).
        """
        resp = self._send({"action": "remove", "job_id": job_id})
        if resp.get("status") != "ok":
            raise RuntimeError(resp.get("message", "Unknown error"))

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return every KISS-owned cron job currently installed.

        Raises:
            ConnectionError: If the daemon is not reachable.
        """
        resp = self._send({"action": "list"})
        jobs: list[dict[str, Any]] = resp.get("jobs", [])
        return jobs

    def status(self) -> dict[str, Any]:
        """Return the daemon's status payload (``pid``, ``job_count``, ...).

        Raises:
            ConnectionError: If the daemon is not reachable.
        """
        return self._send({"action": "status"})

    def stop_daemon(self) -> None:
        """Ask the daemon to shut down. Silent if the daemon is already gone."""
        try:
            self._send({"action": "stop"})
        except ConnectionError:
            pass


def run_cron_job_lifecycle(
    cron_job: str, sock_path: Path | None = None
) -> dict[str, Any]:
    """Add a cron job, list jobs, remove it, and stop the daemon.

    The ``cron_job`` string is the exact crontab entry to install — it is
    sent to the daemon verbatim with no client-side parsing.

    Lifecycle steps executed in order:

    1. Create a :class:`CronClient`.
    2. Add the job via :meth:`CronClient.add_job`.
    3. List all jobs via :meth:`CronClient.list_jobs`.
    4. Remove the job via :meth:`CronClient.remove_job`.
    5. Stop the daemon via :meth:`CronClient.stop_daemon`.

    Args:
        cron_job: A single-line crontab entry, e.g.
            ``"*/5 * * * * echo hello"``.
        sock_path: Optional path to the daemon's Unix domain socket.
            Defaults to ``~/.kiss/cron_manager.sock``.

    Returns:
        A dict with keys ``job_id`` (str) and ``jobs`` (list of dicts as
        returned by :meth:`CronClient.list_jobs` after adding).

    Raises:
        ValueError: If ``cron_job`` is empty.
        ConnectionError: If the daemon is not reachable.
        RuntimeError: If the daemon rejects the add or remove request.
    """
    entry = cron_job.strip()
    if not entry:
        raise ValueError(f"Cron job entry must not be empty: {cron_job!r}")

    client = CronClient(sock_path=sock_path or SOCK_PATH)
    job_id = client.add_job(entry)
    jobs = client.list_jobs()
    client.remove_job(job_id)
    client.stop_daemon()

    return {
        "job_id": job_id,
        "jobs": jobs,
    }


def main() -> None:
    """CLI entry point for daemon management.

    Usage::

        python -m kiss.agents.third_party_agents.cron_manager_daemon start [--foreground]
        python -m kiss.agents.third_party_agents.cron_manager_daemon stop
        python -m kiss.agents.third_party_agents.cron_manager_daemon status
    """
    if len(sys.argv) < 2:
        print("Usage: cron_manager_daemon {start|stop|status} [--foreground]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "start":
        foreground = "--foreground" in sys.argv
        print(start_daemon(foreground=foreground))
    elif cmd == "stop":
        print(stop_daemon())
    elif cmd == "status":
        print(daemon_status())
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
