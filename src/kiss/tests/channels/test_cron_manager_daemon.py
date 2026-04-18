"""Tests for the Cron Manager Daemon.

Tests cover: cron parsing, job management, socket communication,
daemon start/stop/restart, persistence across restarts, concurrent
clients, error handling, and process kill recovery.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import pytest

from kiss.channels.cron_manager_daemon import (
    CronClient,
    CronDaemon,
    _is_running,
    _load_jobs,
    _read_pid,
    _save_jobs,
    cron_matches_time,
    parse_cron_expression,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_counter = 0


@pytest.fixture()
def test_paths(tmp_path: Path) -> dict[str, Path]:
    """Provide isolated paths for a test run.

    Uses /tmp for the socket to avoid AF_UNIX path length limits on macOS.
    """
    global _counter  # noqa: PLW0603
    _counter += 1
    tag = f"{os.getpid()}_{_counter}"
    sock = Path(f"/tmp/_kiss_test_{tag}.sock")
    return {
        "sock": sock,
        "pid": tmp_path / "cron_test.pid",
        "jobs": tmp_path / "cron_test_jobs.json",
    }


@pytest.fixture()
def daemon(test_paths: dict[str, Path]) -> CronDaemon:
    """Create a CronDaemon with test-specific paths."""
    return CronDaemon(
        sock_path=test_paths["sock"],
        pid_path=test_paths["pid"],
        jobs_path=test_paths["jobs"],
    )


@pytest.fixture()
def running_daemon(test_paths: dict[str, Path]) -> Generator[CronDaemon]:
    """Start a CronDaemon in a background thread and yield it.

    Cleans up on exit.
    """
    d = CronDaemon(
        sock_path=test_paths["sock"],
        pid_path=test_paths["pid"],
        jobs_path=test_paths["jobs"],
    )
    t = threading.Thread(target=d.run, daemon=True)
    t.start()
    # Wait for socket to be ready
    for _ in range(100):
        if test_paths["sock"].exists():
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("Daemon did not start in time")
    yield d
    d._stop_event.set()
    t.join(timeout=5)
    # Clean up socket if still present
    test_paths["sock"].unlink(missing_ok=True)


@pytest.fixture()
def client(test_paths: dict[str, Path], running_daemon: CronDaemon) -> CronClient:
    """Return a CronClient connected to the running test daemon."""
    return CronClient(sock_path=test_paths["sock"])


# ---------------------------------------------------------------------------
# Cron expression parsing
# ---------------------------------------------------------------------------


class TestCronParsing:
    def test_wildcard(self) -> None:
        parsed = parse_cron_expression("* * * * *")
        assert parsed["minute"] == set(range(0, 60))
        assert parsed["hour"] == set(range(0, 24))
        assert parsed["dom"] == set(range(1, 32))
        assert parsed["month"] == set(range(1, 13))
        assert parsed["dow"] == set(range(0, 7))

    def test_step(self) -> None:
        parsed = parse_cron_expression("*/15 */6 * * *")
        assert parsed["minute"] == {0, 15, 30, 45}
        assert parsed["hour"] == {0, 6, 12, 18}

    def test_range(self) -> None:
        parsed = parse_cron_expression("0 9-17 * * *")
        assert parsed["minute"] == {0}
        assert parsed["hour"] == set(range(9, 18))

    def test_range_with_step(self) -> None:
        parsed = parse_cron_expression("0 0-23/2 * * *")
        assert parsed["hour"] == {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}

    def test_comma_list(self) -> None:
        parsed = parse_cron_expression("0,30 * * * *")
        assert parsed["minute"] == {0, 30}

    def test_specific_values(self) -> None:
        parsed = parse_cron_expression("5 3 15 6 2")
        assert parsed["minute"] == {5}
        assert parsed["hour"] == {3}
        assert parsed["dom"] == {15}
        assert parsed["month"] == {6}
        assert parsed["dow"] == {2}

    def test_invalid_field_count(self) -> None:
        with pytest.raises(ValueError, match="5 fields"):
            parse_cron_expression("* *")

    def test_invalid_value_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            parse_cron_expression("60 * * * *")

    def test_invalid_step_zero(self) -> None:
        with pytest.raises(ValueError, match="Invalid step"):
            parse_cron_expression("*/0 * * * *")

    def test_invalid_range_step_zero(self) -> None:
        with pytest.raises(ValueError, match="Invalid step"):
            parse_cron_expression("0-10/0 * * * *")


class TestCronMatching:
    def test_matches_every_minute(self) -> None:
        parsed = parse_cron_expression("* * * * *")
        now = datetime(2025, 7, 15, 10, 30)
        assert cron_matches_time(parsed, now)

    def test_matches_specific_time(self) -> None:
        # 2025-07-15 is a Tuesday (weekday=1)
        parsed = parse_cron_expression("30 10 15 7 1")
        dt = datetime(2025, 7, 15, 10, 30)
        assert cron_matches_time(parsed, dt)

    def test_no_match_wrong_minute(self) -> None:
        parsed = parse_cron_expression("0 * * * *")
        dt = datetime(2025, 7, 15, 10, 30)
        assert not cron_matches_time(parsed, dt)

    def test_no_match_wrong_hour(self) -> None:
        parsed = parse_cron_expression("30 11 * * *")
        dt = datetime(2025, 7, 15, 10, 30)
        assert not cron_matches_time(parsed, dt)

    def test_no_match_wrong_day(self) -> None:
        parsed = parse_cron_expression("30 10 16 * *")
        dt = datetime(2025, 7, 15, 10, 30)
        assert not cron_matches_time(parsed, dt)

    def test_no_match_wrong_month(self) -> None:
        parsed = parse_cron_expression("30 10 15 8 *")
        dt = datetime(2025, 7, 15, 10, 30)
        assert not cron_matches_time(parsed, dt)

    def test_no_match_wrong_dow(self) -> None:
        # Tuesday=1, test with Wednesday=2
        parsed = parse_cron_expression("30 10 * * 2")
        dt = datetime(2025, 7, 15, 10, 30)  # Tuesday
        assert not cron_matches_time(parsed, dt)


# ---------------------------------------------------------------------------
# Job persistence
# ---------------------------------------------------------------------------


class TestJobPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.json"
        jobs = {"abc123": {"id": "abc123", "schedule": "* * * * *", "command": "echo hi"}}
        _save_jobs(path, jobs)
        loaded = _load_jobs(path)
        assert loaded == jobs

    def test_load_missing_file(self, tmp_path: Path) -> None:
        assert _load_jobs(tmp_path / "nonexistent.json") == {}

    def test_load_corrupt_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json")
        assert _load_jobs(path) == {}

    def test_load_non_dict_json(self, tmp_path: Path) -> None:
        path = tmp_path / "array.json"
        path.write_text("[1,2,3]")
        assert _load_jobs(path) == {}


# ---------------------------------------------------------------------------
# PID helpers
# ---------------------------------------------------------------------------


class TestPidHelpers:
    def test_read_pid_missing(self, tmp_path: Path) -> None:
        assert _read_pid(tmp_path / "nope.pid") is None

    def test_read_pid_valid(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pid"
        p.write_text("12345")
        assert _read_pid(p) == 12345

    def test_read_pid_invalid(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.pid"
        p.write_text("not a number")
        assert _read_pid(p) is None

    def test_is_running_current_process(self) -> None:
        assert _is_running(os.getpid())

    def test_is_running_dead_pid(self) -> None:
        # PID 99999999 is almost certainly not running
        assert not _is_running(99999999)


# ---------------------------------------------------------------------------
# Daemon command handling (unit tests)
# ---------------------------------------------------------------------------


class TestDaemonCommands:
    def test_add_and_list(self, daemon: CronDaemon) -> None:
        cmd = {"action": "add", "schedule": "* * * * *", "command": "echo x"}
        resp = daemon._handle_command(cmd)
        assert resp["status"] == "ok"
        job_id = resp["job_id"]

        resp = daemon._handle_command({"action": "list"})
        assert resp["status"] == "ok"
        assert len(resp["jobs"]) == 1
        assert resp["jobs"][0]["id"] == job_id

    def test_add_invalid_schedule(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command({"action": "add", "schedule": "bad", "command": "echo x"})
        assert resp["status"] == "error"

    def test_add_missing_fields(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command({"action": "add", "schedule": "", "command": ""})
        assert resp["status"] == "error"

    def test_remove(self, daemon: CronDaemon) -> None:
        cmd = {"action": "add", "schedule": "* * * * *", "command": "echo x"}
        resp = daemon._handle_command(cmd)
        job_id = resp["job_id"]
        resp = daemon._handle_command({"action": "remove", "job_id": job_id})
        assert resp["status"] == "ok"
        resp = daemon._handle_command({"action": "list"})
        assert len(resp["jobs"]) == 0

    def test_remove_nonexistent(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command({"action": "remove", "job_id": "nope"})
        assert resp["status"] == "error"

    def test_remove_missing_id(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command({"action": "remove", "job_id": ""})
        assert resp["status"] == "error"

    def test_status(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command({"action": "status"})
        assert resp["status"] == "ok"
        assert "pid" in resp

    def test_stop(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command({"action": "stop"})
        assert resp["status"] == "ok"
        assert daemon._stop_event.is_set()

    def test_unknown_action(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command({"action": "bogus"})
        assert resp["status"] == "error"

    def test_persistence(self, daemon: CronDaemon, test_paths: dict[str, Path]) -> None:
        cmd = {"action": "add", "schedule": "0 * * * *", "command": "echo persisted"}
        daemon._handle_command(cmd)
        # Reload from disk
        loaded = _load_jobs(test_paths["jobs"])
        assert len(loaded) == 1
        job = list(loaded.values())[0]
        assert job["command"] == "echo persisted"


# ---------------------------------------------------------------------------
# Client-daemon integration (socket communication)
# ---------------------------------------------------------------------------


class TestClientDaemonIntegration:
    def test_add_list_remove(self, client: CronClient) -> None:
        job_id = client.add_job("*/5 * * * *", "echo integration")
        assert isinstance(job_id, str)
        assert len(job_id) == 12

        jobs = client.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["id"] == job_id
        assert jobs[0]["schedule"] == "*/5 * * * *"

        client.remove_job(job_id)
        assert client.list_jobs() == []

    def test_status(self, client: CronClient) -> None:
        resp = client.status()
        assert resp["status"] == "ok"
        assert resp["pid"] == os.getpid()  # daemon runs in our process

    def test_add_invalid_schedule(self, client: CronClient) -> None:
        with pytest.raises(RuntimeError, match="5 fields"):
            client.add_job("bad", "echo x")

    def test_remove_nonexistent(self, client: CronClient) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            client.remove_job("doesnotexist")

    def test_multiple_jobs(self, client: CronClient) -> None:
        ids = [client.add_job(f"0 {h} * * *", f"echo job{h}") for h in range(5)]
        jobs = client.list_jobs()
        assert len(jobs) == 5
        for jid in ids:
            client.remove_job(jid)
        assert client.list_jobs() == []

    def test_concurrent_clients(self, client: CronClient, test_paths: dict[str, Path]) -> None:
        """Multiple clients adding jobs concurrently."""
        results: list[str] = []
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                c = CronClient(sock_path=test_paths["sock"])
                jid = c.add_job("* * * * *", f"echo concurrent-{i}")
                results.append(jid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"
        assert len(results) == 10
        assert len(set(results)) == 10  # all unique IDs

        jobs = client.list_jobs()
        assert len(jobs) == 10

    def test_client_unreachable(self, tmp_path: Path) -> None:
        """Client raises ConnectionError when daemon is not running."""
        c = CronClient(sock_path=tmp_path / "no_such.sock")
        with pytest.raises(ConnectionError):
            c.list_jobs()


# ---------------------------------------------------------------------------
# Daemon stop via client
# ---------------------------------------------------------------------------


class TestDaemonStopViaClient:
    def test_stop_daemon(self, test_paths: dict[str, Path]) -> None:
        d = CronDaemon(
            sock_path=test_paths["sock"],
            pid_path=test_paths["pid"],
            jobs_path=test_paths["jobs"],
        )
        t = threading.Thread(target=d.run, daemon=True)
        t.start()
        for _ in range(100):
            if test_paths["sock"].exists():
                break
            time.sleep(0.05)

        c = CronClient(sock_path=test_paths["sock"])
        c.stop_daemon()
        t.join(timeout=5)
        assert not t.is_alive()
        assert not test_paths["sock"].exists()
        # PID file should be cleaned up
        assert not test_paths["pid"].exists()
        # Extra cleanup just in case
        test_paths["sock"].unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Daemon restart with persistence
# ---------------------------------------------------------------------------


class TestDaemonRestart:
    def test_jobs_survive_restart(self, test_paths: dict[str, Path]) -> None:
        """Jobs added before restart are still present after restart."""
        # Start daemon #1
        d1 = CronDaemon(
            sock_path=test_paths["sock"],
            pid_path=test_paths["pid"],
            jobs_path=test_paths["jobs"],
        )
        t1 = threading.Thread(target=d1.run, daemon=True)
        t1.start()
        for _ in range(100):
            if test_paths["sock"].exists():
                break
            time.sleep(0.05)

        c1 = CronClient(sock_path=test_paths["sock"])
        job_id = c1.add_job("0 12 * * *", "echo survive")

        # Stop daemon #1
        c1.stop_daemon()
        t1.join(timeout=5)

        # Start daemon #2 with same paths
        d2 = CronDaemon(
            sock_path=test_paths["sock"],
            pid_path=test_paths["pid"],
            jobs_path=test_paths["jobs"],
        )
        t2 = threading.Thread(target=d2.run, daemon=True)
        t2.start()
        for _ in range(100):
            if test_paths["sock"].exists():
                break
            time.sleep(0.05)

        c2 = CronClient(sock_path=test_paths["sock"])
        jobs = c2.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["id"] == job_id
        assert jobs[0]["command"] == "echo survive"

        # Clean up
        c2.stop_daemon()
        t2.join(timeout=5)
        test_paths["sock"].unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Process kill and restart (subprocess-based)
# ---------------------------------------------------------------------------


class TestProcessKillRestart:
    def test_kill_and_restart(self, test_paths: dict[str, Path]) -> None:
        """Simulate killing the daemon process and restarting it."""
        src_root = str(Path(__file__).parents[3])
        sock = str(test_paths["sock"])
        pid = str(test_paths["pid"])
        jobs = str(test_paths["jobs"])
        daemon_script = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {src_root!r})
            from pathlib import Path
            from kiss.channels.cron_manager_daemon import CronDaemon
            d = CronDaemon(
                sock_path=Path({sock!r}),
                pid_path=Path({pid!r}),
                jobs_path=Path({jobs!r}),
            )
            d.run()
        """)

        # Start daemon in subprocess
        proc = subprocess.Popen(
            [sys.executable, "-c", daemon_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            # Wait for socket
            for _ in range(100):
                if test_paths["sock"].exists():
                    break
                time.sleep(0.05)
            else:
                out, err = proc.communicate(timeout=2)
                raise RuntimeError(f"Daemon subprocess did not start: {err.decode()}")

            # Add a job
            c = CronClient(sock_path=test_paths["sock"])
            job_id = c.add_job("30 2 * * *", "echo killed-test")
            assert len(c.list_jobs()) == 1

            # Kill the process (SIGKILL - hard kill)
            proc.kill()
            proc.wait(timeout=5)

            # Socket and PID file may still be there (stale)
            test_paths["sock"].unlink(missing_ok=True)

            # Restart daemon in new subprocess
            proc2 = subprocess.Popen(
                [sys.executable, "-c", daemon_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                for _ in range(100):
                    if test_paths["sock"].exists():
                        break
                    time.sleep(0.05)
                else:
                    out, err = proc2.communicate(timeout=2)
                    raise RuntimeError(f"Daemon subprocess did not restart: {err.decode()}")

                # Jobs should persist
                c2 = CronClient(sock_path=test_paths["sock"])
                jobs_list = c2.list_jobs()
                assert len(jobs_list) == 1
                assert jobs_list[0]["id"] == job_id
                assert jobs_list[0]["command"] == "echo killed-test"

                # Can still add new jobs
                c2.add_job("0 0 * * *", "echo new-after-restart")
                assert len(c2.list_jobs()) == 2

                c2.stop_daemon()
                proc2.wait(timeout=5)
            finally:
                if proc2.poll() is None:
                    proc2.kill()
                    proc2.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            test_paths["sock"].unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Handle client edge cases
# ---------------------------------------------------------------------------


class TestClientEdgeCases:
    def test_send_invalid_json(
        self, test_paths: dict[str, Path], running_daemon: CronDaemon
    ) -> None:
        """Send raw invalid JSON bytes to the daemon socket."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(5.0)
            sock.connect(str(test_paths["sock"]))
            sock.sendall(b"not json at all{{{")
            sock.shutdown(socket.SHUT_WR)
            data = sock.recv(4096)
            resp = json.loads(data)
            assert resp["status"] == "error"
            assert "Invalid JSON" in resp["message"]
        finally:
            sock.close()

    def test_send_empty_connection(
        self, test_paths: dict[str, Path], running_daemon: CronDaemon
    ) -> None:
        """Connect and immediately close — daemon should not crash."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(5.0)
            sock.connect(str(test_paths["sock"]))
        finally:
            sock.close()
        # Daemon should still be responsive
        time.sleep(0.1)
        c = CronClient(sock_path=test_paths["sock"])
        resp = c.status()
        assert resp["status"] == "ok"

    def test_stop_daemon_client_method(self, tmp_path: Path) -> None:
        """CronClient.stop_daemon() doesn't raise when daemon is gone."""
        c = CronClient(sock_path=tmp_path / "gone.sock")
        # Should not raise — stop_daemon catches ConnectionError
        c.stop_daemon()


# ---------------------------------------------------------------------------
# Scheduler job execution
# ---------------------------------------------------------------------------


class TestJobExecution:
    def test_run_job(self, daemon: CronDaemon, tmp_path: Path) -> None:
        """Directly test _run_job to verify subprocess execution."""
        marker = tmp_path / "ran.txt"
        daemon._jobs["test"] = {
            "id": "test",
            "schedule": "* * * * *",
            "command": f"echo done > {marker}",
            "last_run": None,
            "run_count": 0,
        }
        daemon._run_job("test", f"echo done > {marker}")
        assert marker.read_text().strip() == "done"
        assert daemon._jobs["test"]["run_count"] == 1
        assert daemon._jobs["test"]["last_run"] is not None

    def test_run_job_failure(self, daemon: CronDaemon) -> None:
        """_run_job handles command failure gracefully."""
        daemon._jobs["fail"] = {
            "id": "fail",
            "schedule": "* * * * *",
            "command": "false",
            "last_run": None,
            "run_count": 0,
        }
        # Should not raise
        daemon._run_job("fail", "false")
        assert daemon._jobs["fail"]["run_count"] == 1

    def test_run_job_nonexistent_command(self, daemon: CronDaemon) -> None:
        """_run_job handles missing commands."""
        daemon._jobs["bad"] = {
            "id": "bad",
            "schedule": "* * * * *",
            "command": "/nonexistent/command",
            "last_run": None,
            "run_count": 0,
        }
        daemon._run_job("bad", "/nonexistent/command")
        # run_count still increments (shell returns error but doesn't throw)
        assert daemon._jobs["bad"]["run_count"] == 1

    def test_run_job_removed_during_execution(self, daemon: CronDaemon) -> None:
        """_run_job handles the case where the job is removed while running."""
        # Don't add to _jobs — simulate removal
        daemon._run_job("ghost", "echo ghost")
        # Should not raise — just skips the metadata update
