"""Cron Manager Daemon — scheduled task manager with Unix domain socket interface.

Runs as a daemon process listening on ``~/.kiss/cron_manager.sock`` for
cron job management commands from any KISS agent.  Jobs are stored in a
JSON file at ``~/.kiss/cron_jobs.json`` so they survive daemon restarts.

Daemon control::

    python -m kiss.channels.cron_manager_daemon start
    python -m kiss.channels.cron_manager_daemon stop
    python -m kiss.channels.cron_manager_daemon status

Client usage from any KISS agent::

    from kiss.channels.cron_manager_daemon import CronClient

    client = CronClient()
    job_id = client.add_job("*/5 * * * *", "echo hello")
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
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

KISS_DIR = Path.home() / ".kiss"
SOCK_PATH = KISS_DIR / "cron_manager.sock"
PID_PATH = KISS_DIR / "cron_manager.pid"
LOG_PATH = KISS_DIR / "cron_manager.log"
JOBS_PATH = KISS_DIR / "cron_jobs.json"

# Maximum message size: 1 MB
_MAX_MSG = 1_048_576


# ---------------------------------------------------------------------------
# Cron expression parsing
# ---------------------------------------------------------------------------


def _parse_cron_field(field: str, lo: int, hi: int) -> set[int]:
    """Parse a single cron field into a set of matching integer values.

    Supports: ``*``, ``*/N``, ``N``, ``N-M``, ``N-M/S``, and comma-separated
    combinations thereof.

    Args:
        field: The cron field string (e.g. ``"*/5"``, ``"1,15"``, ``"0-23/2"``).
        lo: Minimum valid value for this field (inclusive).
        hi: Maximum valid value for this field (inclusive).

    Returns:
        Set of integers that match the field specification.

    Raises:
        ValueError: If the field cannot be parsed.
    """
    values: set[int] = set()
    for part in field.split(","):
        part = part.strip()
        if part == "*":
            values.update(range(lo, hi + 1))
        elif part.startswith("*/"):
            step = int(part[2:])
            if step <= 0:
                raise ValueError(f"Invalid step: {step}")
            values.update(range(lo, hi + 1, step))
        elif "-" in part and "/" in part:
            range_part, step_str = part.split("/", 1)
            start_str, end_str = range_part.split("-", 1)
            start, end, step = int(start_str), int(end_str), int(step_str)
            if step <= 0:
                raise ValueError(f"Invalid step: {step}")
            values.update(range(start, end + 1, step))
        elif "-" in part:
            start_str, end_str = part.split("-", 1)
            values.update(range(int(start_str), int(end_str) + 1))
        else:
            values.add(int(part))
    # Validate
    for v in values:
        if v < lo or v > hi:
            raise ValueError(f"Value {v} out of range [{lo}, {hi}]")
    return values


def parse_cron_expression(expr: str) -> dict[str, set[int]]:
    """Parse a 5-field cron expression into matched value sets.

    Fields: minute hour day-of-month month day-of-week

    Args:
        expr: Cron expression string, e.g. ``"*/5 * * * *"``.

    Returns:
        Dictionary with keys ``minute``, ``hour``, ``dom``, ``month``, ``dow``
        each mapping to a set of matching integer values.

    Raises:
        ValueError: If the expression does not have exactly 5 fields or
            any field is invalid.
    """
    fields = expr.strip().split()
    if len(fields) != 5:
        raise ValueError(f"Cron expression must have 5 fields, got {len(fields)}: {expr!r}")
    return {
        "minute": _parse_cron_field(fields[0], 0, 59),
        "hour": _parse_cron_field(fields[1], 0, 23),
        "dom": _parse_cron_field(fields[2], 1, 31),
        "month": _parse_cron_field(fields[3], 1, 12),
        "dow": _parse_cron_field(fields[4], 0, 6),
    }


def cron_matches_time(parsed: dict[str, set[int]], dt: datetime) -> bool:
    """Check whether a parsed cron schedule matches a specific datetime.

    Args:
        parsed: Output of :func:`parse_cron_expression`.
        dt: The datetime to check against.

    Returns:
        True if all five fields match the datetime.
    """
    return (
        dt.minute in parsed["minute"]
        and dt.hour in parsed["hour"]
        and dt.day in parsed["dom"]
        and dt.month in parsed["month"]
        and dt.weekday() in parsed["dow"]  # Monday=0, Sunday=6
    )


# ---------------------------------------------------------------------------
# Job persistence
# ---------------------------------------------------------------------------


def _load_jobs(path: Path) -> dict[str, dict[str, Any]]:
    """Load jobs from the JSON file.

    Args:
        path: Path to the jobs JSON file.

    Returns:
        Dictionary mapping job ID to job data.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return {}
        return data
    except (json.JSONDecodeError, OSError):
        return {}


def _save_jobs(path: Path, jobs: dict[str, dict[str, Any]]) -> None:
    """Save jobs to the JSON file atomically.

    Args:
        path: Path to the jobs JSON file.
        jobs: Dictionary mapping job ID to job data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------


class CronDaemon:
    """Daemon process that manages cron jobs via a Unix domain socket.

    The daemon listens for JSON commands on the socket at
    ``~/.kiss/cron_manager.sock`` and runs a scheduler thread that
    checks every 30 seconds for jobs that need to execute.

    Attributes:
        sock_path: Path to the Unix domain socket.
        pid_path: Path to the PID file.
        jobs_path: Path to the persistent jobs file.
    """

    def __init__(
        self,
        sock_path: Path = SOCK_PATH,
        pid_path: Path = PID_PATH,
        jobs_path: Path = JOBS_PATH,
    ) -> None:
        self.sock_path = sock_path
        self.pid_path = pid_path
        self.jobs_path = jobs_path
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._server_socket: socket.socket | None = None
        # Track which minute we last ran jobs for to avoid double-execution
        self._last_run_minute: str = ""

    # -- Job management -----------------------------------------------------

    def _add_job(self, schedule: str, command: str) -> dict[str, Any]:
        """Add a new cron job.

        Args:
            schedule: Cron expression (5 fields).
            command: Shell command to execute.

        Returns:
            Response dict with status and job_id.
        """
        # Validate cron expression
        try:
            parse_cron_expression(schedule)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "schedule": schedule,
            "command": command,
            "created_at": datetime.now().isoformat(),
            "last_run": None,
            "run_count": 0,
        }
        with self._lock:
            self._jobs[job_id] = job
            _save_jobs(self.jobs_path, self._jobs)
        logger.info("Added job %s: %s -> %s", job_id, schedule, command)
        return {"status": "ok", "job_id": job_id}

    def _remove_job(self, job_id: str) -> dict[str, Any]:
        """Remove a cron job by ID.

        Args:
            job_id: The job identifier.

        Returns:
            Response dict with status.
        """
        with self._lock:
            if job_id not in self._jobs:
                return {"status": "error", "message": f"Job {job_id!r} not found"}
            del self._jobs[job_id]
            _save_jobs(self.jobs_path, self._jobs)
        logger.info("Removed job %s", job_id)
        return {"status": "ok", "message": f"Job {job_id!r} removed"}

    def _list_jobs(self) -> dict[str, Any]:
        """List all cron jobs.

        Returns:
            Response dict with job list.
        """
        with self._lock:
            return {"status": "ok", "jobs": list(self._jobs.values())}

    def _get_status(self) -> dict[str, Any]:
        """Get daemon status information.

        Returns:
            Response dict with daemon status.
        """
        with self._lock:
            job_count = len(self._jobs)
        return {
            "status": "ok",
            "pid": os.getpid(),
            "job_count": job_count,
            "uptime_info": "running",
        }

    # -- Command dispatch ---------------------------------------------------

    def _handle_command(self, data: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a command from a client.

        Args:
            data: Parsed JSON command with at least an ``"action"`` key.

        Returns:
            Response dictionary.
        """
        action = data.get("action")
        if action == "add":
            schedule = data.get("schedule", "")
            command = data.get("command", "")
            if not schedule or not command:
                return {"status": "error", "message": "Missing 'schedule' or 'command'"}
            return self._add_job(schedule, command)
        elif action == "remove":
            job_id = data.get("job_id", "")
            if not job_id:
                return {"status": "error", "message": "Missing 'job_id'"}
            return self._remove_job(job_id)
        elif action == "list":
            return self._list_jobs()
        elif action == "status":
            return self._get_status()
        elif action == "stop":
            self._stop_event.set()
            return {"status": "ok", "message": "Daemon stopping"}
        else:
            return {"status": "error", "message": f"Unknown action: {action!r}"}

    # -- Client connection handler ------------------------------------------

    def _handle_client(self, conn: socket.socket) -> None:
        """Handle a single client connection.

        Reads a JSON message, dispatches the command, and sends back the
        JSON response.

        Args:
            conn: Connected client socket.
        """
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
                        json.dumps({"status": "error", "message": "Message too large"}).encode()
                    )
                    return
                # Check if we have a complete JSON message
                try:
                    data = json.loads(b"".join(chunks))
                    break
                except json.JSONDecodeError:
                    continue

            if not chunks:
                return

            raw = b"".join(chunks)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                conn.sendall(json.dumps({"status": "error", "message": "Invalid JSON"}).encode())
                return

            response = self._handle_command(data)
            conn.sendall(json.dumps(response).encode())
        except (OSError, TimeoutError):
            logger.debug("Client connection error", exc_info=True)
        finally:
            conn.close()

    # -- Scheduler ----------------------------------------------------------

    def _scheduler_loop(self) -> None:
        """Run the cron scheduler loop.

        Checks every 30 seconds for jobs whose schedule matches the
        current minute. Each job runs at most once per matching minute.
        """
        while not self._stop_event.is_set():
            now = datetime.now()
            minute_key = now.strftime("%Y-%m-%d %H:%M")
            if minute_key != self._last_run_minute:
                self._last_run_minute = minute_key
                with self._lock:
                    jobs_snapshot = list(self._jobs.items())
                for job_id, job in jobs_snapshot:
                    try:
                        parsed = parse_cron_expression(job["schedule"])
                        if cron_matches_time(parsed, now):
                            logger.info("Running job %s: %s", job_id, job["command"])
                            threading.Thread(
                                target=self._run_job, args=(job_id, job["command"]), daemon=True
                            ).start()
                    except (ValueError, KeyError):
                        logger.warning("Invalid job %s, skipping", job_id)
            self._stop_event.wait(timeout=30)

    def _run_job(self, job_id: str, command: str) -> None:
        """Execute a cron job command in a subprocess.

        Args:
            job_id: The job identifier (for logging and updating metadata).
            command: Shell command to execute.
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            logger.info(
                "Job %s completed (rc=%d): stdout=%r stderr=%r",
                job_id,
                result.returncode,
                result.stdout[:200],
                result.stderr[:200],
            )
        except subprocess.TimeoutExpired:
            logger.warning("Job %s timed out after 3600s", job_id)
        except OSError as e:
            logger.error("Job %s failed: %s", job_id, e)
        # Update run metadata
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["last_run"] = datetime.now().isoformat()
                self._jobs[job_id]["run_count"] = self._jobs[job_id].get("run_count", 0) + 1
                _save_jobs(self.jobs_path, self._jobs)

    # -- Socket server ------------------------------------------------------

    def _serve(self) -> None:
        """Main server loop: accept connections and dispatch to handlers."""
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        # Clean stale socket
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
                threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()
            except TimeoutError:
                continue
            except OSError:
                if not self._stop_event.is_set():
                    logger.error("Socket accept error", exc_info=True)
                break

    def _cleanup(self) -> None:
        """Clean up socket and PID files on shutdown."""
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self.sock_path.exists():
            try:
                self.sock_path.unlink()
            except OSError:
                pass
        if self.pid_path.exists():
            try:
                self.pid_path.unlink()
            except OSError:
                pass

    # -- Public entry points ------------------------------------------------

    def run(self) -> None:
        """Start the daemon (foreground mode).

        Loads persisted jobs, writes the PID file, starts the scheduler
        thread, and enters the socket server loop.  On shutdown (via
        ``stop`` command or signal), cleans up all resources.
        """
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(os.getpid()))

        # Load persisted jobs
        with self._lock:
            self._jobs = _load_jobs(self.jobs_path)
        logger.info("Loaded %d persisted jobs", len(self._jobs))

        # Handle signals for graceful shutdown (only works in main thread)
        def _signal_handler(signum: int, frame: Any) -> None:
            logger.info("Received signal %d, shutting down", signum)
            self._stop_event.set()

        try:
            signal.signal(signal.SIGTERM, _signal_handler)
            signal.signal(signal.SIGINT, _signal_handler)
        except ValueError:
            pass  # Not in main thread — signals handled via stop command

        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        scheduler_thread.start()

        try:
            self._serve()
        finally:
            self._stop_event.set()
            scheduler_thread.join(timeout=5)
            self._cleanup()
            logger.info("Daemon stopped")


# ---------------------------------------------------------------------------
# Daemonize helper
# ---------------------------------------------------------------------------


def _daemonize() -> None:
    """Double-fork to fully detach from the controlling terminal.

    After this call the process is a proper daemon: detached from the
    terminal, in its own session, with stdin/stdout/stderr redirected
    to /dev/null (logging goes to the log file).
    """
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)
    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)


# ---------------------------------------------------------------------------
# Process management helpers
# ---------------------------------------------------------------------------


def _read_pid(pid_path: Path = PID_PATH) -> int | None:
    """Read the daemon PID from the PID file.

    Args:
        pid_path: Path to the PID file.

    Returns:
        The PID as an integer, or None if the file is missing or invalid.
    """
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return None


def _is_running(pid: int) -> bool:
    """Check whether a process with the given PID is alive.

    Args:
        pid: Process ID to check.

    Returns:
        True if the process exists and is reachable.
    """
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_daemon(
    sock_path: Path = SOCK_PATH,
    pid_path: Path = PID_PATH,
    jobs_path: Path = JOBS_PATH,
    foreground: bool = False,
) -> str:
    """Start the cron manager daemon.

    Args:
        sock_path: Path for the Unix domain socket.
        pid_path: Path for the PID file.
        jobs_path: Path for the jobs persistence file.
        foreground: If True, run in the foreground (don't daemonize).

    Returns:
        Status message string.
    """
    pid = _read_pid(pid_path)
    if pid and _is_running(pid):
        return f"Daemon already running (pid={pid})"

    # Clean up stale files
    for p in (sock_path, pid_path):
        if p.exists():
            p.unlink()

    if not foreground:
        _daemonize()

    # Set up file logging
    log_path = sock_path.parent / "cron_manager.log"
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    daemon = CronDaemon(sock_path=sock_path, pid_path=pid_path, jobs_path=jobs_path)
    daemon.run()
    return "Daemon stopped"


def stop_daemon(
    sock_path: Path = SOCK_PATH,
    pid_path: Path = PID_PATH,
) -> str:
    """Stop the cron manager daemon.

    Tries a graceful stop via the socket first, then falls back to
    SIGTERM.

    Args:
        sock_path: Path to the daemon socket.
        pid_path: Path to the PID file.

    Returns:
        Status message string.
    """
    # Try graceful stop via socket
    try:
        client = CronClient(sock_path=sock_path)
        client.stop_daemon()
        # Wait for process to exit
        pid = _read_pid(pid_path)
        if pid:
            for _ in range(50):
                if not _is_running(pid):
                    break
                time.sleep(0.1)
            else:
                os.kill(pid, signal.SIGTERM)
        return "Daemon stopped"
    except (ConnectionError, OSError):
        pass

    # Fall back to SIGTERM
    pid = _read_pid(pid_path)
    if not pid:
        return "Daemon not running (no PID file)"
    if not _is_running(pid):
        pid_path.unlink(missing_ok=True)
        return "Daemon not running (stale PID file cleaned)"
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not _is_running(pid):
            break
        time.sleep(0.1)
    return "Daemon stopped"


def daemon_status(
    pid_path: Path = PID_PATH,
    sock_path: Path = SOCK_PATH,
) -> str:
    """Check the daemon status.

    Args:
        pid_path: Path to the PID file.
        sock_path: Path to the daemon socket.

    Returns:
        Human-readable status string.
    """
    pid = _read_pid(pid_path)
    if not pid:
        return "Daemon not running (no PID file)"
    if not _is_running(pid):
        return f"Daemon not running (stale PID file for pid={pid})"
    # Try socket connection
    try:
        client = CronClient(sock_path=sock_path)
        resp = client.status()
        return f"Daemon running (pid={pid}, jobs={resp.get('job_count', '?')})"
    except (ConnectionError, OSError):
        return f"Daemon running (pid={pid}) but socket not responding"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class CronClient:
    """Client for communicating with the Cron Manager Daemon.

    Sends JSON commands over the Unix domain socket and returns
    parsed responses.

    Args:
        sock_path: Path to the daemon's Unix domain socket.
            Defaults to ``~/.kiss/cron_manager.sock``.
    """

    def __init__(self, sock_path: Path = SOCK_PATH) -> None:
        self.sock_path = sock_path

    def _send(self, data: dict[str, Any]) -> dict[str, Any]:
        """Send a command and return the response.

        Args:
            data: Command dictionary to send.

        Returns:
            Parsed JSON response dictionary.

        Raises:
            ConnectionError: If the daemon is not reachable.
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

    def add_job(self, schedule: str, command: str) -> str:
        """Add a cron job.

        Args:
            schedule: Cron expression (5 fields), e.g. ``"*/5 * * * *"``.
            command: Shell command to run on schedule.

        Returns:
            The job ID string.

        Raises:
            ConnectionError: If the daemon is not reachable.
            RuntimeError: If the daemon returns an error.
        """
        resp = self._send({"action": "add", "schedule": schedule, "command": command})
        if resp.get("status") != "ok":
            raise RuntimeError(resp.get("message", "Unknown error"))
        job_id: str = resp["job_id"]
        return job_id

    def remove_job(self, job_id: str) -> None:
        """Remove a cron job.

        Args:
            job_id: The job identifier to remove.

        Raises:
            ConnectionError: If the daemon is not reachable.
            RuntimeError: If the daemon returns an error.
        """
        resp = self._send({"action": "remove", "job_id": job_id})
        if resp.get("status") != "ok":
            raise RuntimeError(resp.get("message", "Unknown error"))

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all cron jobs.

        Returns:
            List of job dictionaries.

        Raises:
            ConnectionError: If the daemon is not reachable.
        """
        resp = self._send({"action": "list"})
        jobs: list[dict[str, Any]] = resp.get("jobs", [])
        return jobs

    def status(self) -> dict[str, Any]:
        """Get daemon status.

        Returns:
            Status dictionary with ``pid``, ``job_count``, etc.

        Raises:
            ConnectionError: If the daemon is not reachable.
        """
        return self._send({"action": "status"})

    def stop_daemon(self) -> None:
        """Send stop command to the daemon.

        Raises:
            ConnectionError: If the daemon is not reachable.
        """
        try:
            self._send({"action": "stop"})
        except ConnectionError:
            pass  # Daemon may close before responding


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for daemon management.

    Usage::

        python -m kiss.channels.cron_manager_daemon start [--foreground]
        python -m kiss.channels.cron_manager_daemon stop
        python -m kiss.channels.cron_manager_daemon status
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
