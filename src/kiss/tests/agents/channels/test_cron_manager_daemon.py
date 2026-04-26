"""Tests for the Cron Manager Daemon.

The daemon now delegates scheduling/execution/persistence to the system
``crontab`` command. Tests substitute a tiny Python-based fake ``crontab``
binary (installed via a shebang script) so that real subprocess plumbing is
exercised without touching the developer's actual crontab.
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
from pathlib import Path

import pytest

from kiss.agents.third_party_agents.cron_manager_daemon import (
    CronClient,
    CronDaemon,
    CronJob,
    _is_running,
    _parse_kiss_jobs,
    _read_crontab,
    _read_pid,
    _remove_kiss_block,
    _validate_entry,
    _write_crontab,
    daemon_status,
    run_cron_job_lifecycle,
    start_daemon,
    stop_daemon,
)

_counter = 0


def _make_crontab_script(tmp_path: Path, name: str = "crontab") -> tuple[Path, Path]:
    """Create a fake ``crontab`` executable that reads/writes a per-test tab file.

    Returns ``(script_path, tab_path)``.
    """
    tab = tmp_path / "tab"
    tab.write_text("")
    script = tmp_path / name
    script.write_text(
        textwrap.dedent(f"""\
        #!{sys.executable}
        import sys, pathlib
        TAB = pathlib.Path({str(tab)!r})
        args = sys.argv[1:]
        if args == ["-l"]:
            content = TAB.read_text() if TAB.exists() else ""
            if content:
                sys.stdout.write(content)
                sys.exit(0)
            sys.stderr.write("no crontab for test\\n")
            sys.exit(1)
        elif args == ["-"]:
            TAB.write_text(sys.stdin.read())
            sys.exit(0)
        else:
            sys.stderr.write(f"unknown args: {{args}}\\n")
            sys.exit(2)
        """)
    )
    script.chmod(0o755)
    return script, tab


@pytest.fixture
def crontab_env(tmp_path: Path) -> dict[str, Path]:
    """Fresh fake crontab + isolated socket/pid paths per test."""
    global _counter  # noqa: PLW0603
    _counter += 1
    tag = f"{os.getpid()}_{_counter}"
    script, tab = _make_crontab_script(tmp_path)
    return {
        "crontab": script,
        "tab": tab,
        "sock": Path(f"/tmp/_kiss_cron_test_{tag}.sock"),
        "pid": tmp_path / "cron.pid",
    }


@pytest.fixture
def daemon(crontab_env: dict[str, Path]) -> CronDaemon:
    return CronDaemon(
        sock_path=crontab_env["sock"],
        pid_path=crontab_env["pid"],
        crontab_cmd=str(crontab_env["crontab"]),
    )


@pytest.fixture
def running_daemon(crontab_env: dict[str, Path]) -> Generator[CronDaemon]:
    d = CronDaemon(
        sock_path=crontab_env["sock"],
        pid_path=crontab_env["pid"],
        crontab_cmd=str(crontab_env["crontab"]),
    )
    t = threading.Thread(target=d.run, daemon=True)
    t.start()
    for _ in range(100):
        if crontab_env["sock"].exists():
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("Daemon did not start in time")
    yield d
    d._stop_event.set()
    t.join(timeout=5)
    crontab_env["sock"].unlink(missing_ok=True)


@pytest.fixture
def client(
    crontab_env: dict[str, Path], running_daemon: CronDaemon
) -> CronClient:
    return CronClient(sock_path=crontab_env["sock"])


class TestParseKissJobs:
    def test_empty(self) -> None:
        assert _parse_kiss_jobs("") == []

    def test_single(self) -> None:
        content = "# KISS-JOB abc123 2025-01-01T00:00:00\n* * * * * echo hi\n"
        jobs = _parse_kiss_jobs(content)
        assert jobs == [
            CronJob("abc123", "* * * * *", "echo hi", "2025-01-01T00:00:00")
        ]

    def test_multiple_in_order(self) -> None:
        content = (
            "# KISS-JOB a1 2025-01-01T00:00:00\n"
            "*/5 * * * * echo a\n"
            "# KISS-JOB b2 2025-01-02T00:00:00\n"
            "0 9 * * * echo b\n"
        )
        jobs = _parse_kiss_jobs(content)
        assert [j.id for j in jobs] == ["a1", "b2"]
        assert jobs[0].schedule == "*/5 * * * *"
        assert jobs[1].command == "echo b"

    def test_mixed_with_non_kiss(self) -> None:
        content = (
            "# user comment\n"
            "0 0 * * * user_command\n"
            "# KISS-JOB kiss1 ts\n"
            "* * * * * echo kiss\n"
            "SHELL=/bin/bash\n"
        )
        jobs = _parse_kiss_jobs(content)
        assert len(jobs) == 1
        assert jobs[0].id == "kiss1"
        assert jobs[0].command == "echo kiss"

    def test_orphan_marker_skipped(self) -> None:
        content = "# KISS-JOB orphan 2025-01-01T00:00:00\n"
        assert _parse_kiss_jobs(content) == []

    def test_malformed_marker_skipped(self) -> None:
        content = "# KISS-JOB\n* * * * * echo x\n"
        assert _parse_kiss_jobs(content) == []

    def test_malformed_schedule_skipped(self) -> None:
        content = "# KISS-JOB id1 ts\n* * * * \n"
        assert _parse_kiss_jobs(content) == []

    def test_command_with_spaces_preserved(self) -> None:
        content = (
            "# KISS-JOB id1 ts\n"
            "*/5 * * * * /bin/sh -c 'echo hello world'\n"
        )
        jobs = _parse_kiss_jobs(content)
        assert jobs[0].command == "/bin/sh -c 'echo hello world'"


class TestRemoveKissBlock:
    def test_no_match_returns_false(self) -> None:
        content = "# KISS-JOB aaa ts\n* * * * * echo\n"
        new, removed = _remove_kiss_block(content, "bbb")
        assert not removed
        assert new == content

    def test_removes_target(self) -> None:
        content = (
            "# KISS-JOB aaa ts\n* * * * * echo a\n"
            "# KISS-JOB bbb ts\n0 0 * * * echo b\n"
        )
        new, removed = _remove_kiss_block(content, "aaa")
        assert removed
        assert "aaa" not in new
        assert "bbb" in new
        assert new.endswith("\n")

    def test_preserves_non_kiss(self) -> None:
        content = (
            "SHELL=/bin/bash\n"
            "# KISS-JOB x ts\n"
            "* * * * * echo x\n"
            "0 0 * * * user\n"
        )
        new, removed = _remove_kiss_block(content, "x")
        assert removed
        assert "SHELL=/bin/bash" in new
        assert "0 0 * * * user" in new

    def test_removing_last_entry_drops_trailing_newline(self) -> None:
        content = "# KISS-JOB only ts\n* * * * * echo\n"
        new, removed = _remove_kiss_block(content, "only")
        assert removed
        assert new == ""

    def test_malformed_marker_not_treated_as_block(self) -> None:
        content = "# KISS-JOB\nregular line\n"
        new, removed = _remove_kiss_block(content, "anything")
        assert not removed
        assert new == content


class TestValidators:
    def test_entry_ok(self) -> None:
        _validate_entry("*/5 * * * * echo hello")

    def test_entry_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _validate_entry("")

    def test_entry_blank(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _validate_entry("   ")

    def test_entry_multiline(self) -> None:
        with pytest.raises(ValueError, match="single line"):
            _validate_entry("*/5 * * * * echo hi\nrm -rf /")

    def test_entry_carriage_return(self) -> None:
        with pytest.raises(ValueError, match="single line"):
            _validate_entry("*/5 * * * * echo hi\rrm")


class TestReadWriteCrontab:
    def test_read_empty_returns_blank(
        self, crontab_env: dict[str, Path]
    ) -> None:
        assert _read_crontab(str(crontab_env["crontab"])) == ""

    def test_write_then_read(self, crontab_env: dict[str, Path]) -> None:
        _write_crontab(
            "# KISS-JOB x ts\n* * * * * echo\n", str(crontab_env["crontab"])
        )
        assert _read_crontab(str(crontab_env["crontab"])) == (
            "# KISS-JOB x ts\n* * * * * echo\n"
        )

    def test_write_adds_trailing_newline(
        self, crontab_env: dict[str, Path]
    ) -> None:
        _write_crontab("abc", str(crontab_env["crontab"]))
        assert crontab_env["tab"].read_text() == "abc\n"

    def test_write_empty_no_newline_added(
        self, crontab_env: dict[str, Path]
    ) -> None:
        _write_crontab("", str(crontab_env["crontab"]))
        assert crontab_env["tab"].read_text() == ""

    def test_read_raises_on_unexpected_failure(self, tmp_path: Path) -> None:
        broken = tmp_path / "crontab"
        broken.write_text(
            f'#!{sys.executable}\nimport sys\n'
            'sys.stderr.write("database error\\n"); sys.exit(2)\n'
        )
        broken.chmod(0o755)
        with pytest.raises(RuntimeError, match="failed"):
            _read_crontab(str(broken))

    def test_write_raises_on_failure(self, tmp_path: Path) -> None:
        broken = tmp_path / "crontab"
        broken.write_text(
            f'#!{sys.executable}\nimport sys\n'
            'sys.stderr.write("bad schedule\\n"); sys.exit(1)\n'
        )
        broken.chmod(0o755)
        with pytest.raises(RuntimeError, match="failed"):
            _write_crontab("bad\n", str(broken))


class TestPidHelpers:
    def test_read_missing(self, tmp_path: Path) -> None:
        assert _read_pid(tmp_path / "nope.pid") is None

    def test_read_valid(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pid"
        p.write_text("12345")
        assert _read_pid(p) == 12345

    def test_read_invalid(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.pid"
        p.write_text("not a number")
        assert _read_pid(p) is None

    def test_is_running_self(self) -> None:
        assert _is_running(os.getpid())

    def test_is_running_dead(self) -> None:
        assert not _is_running(99999999)


class TestDaemonCommands:
    def test_add_and_list(
        self, daemon: CronDaemon, crontab_env: dict[str, Path]
    ) -> None:
        resp = daemon._handle_command(
            {"action": "add", "entry": "* * * * * echo x"}
        )
        assert resp["status"] == "ok"
        job_id = resp["job_id"]
        assert len(job_id) == 12

        tab = crontab_env["tab"].read_text()
        assert "# KISS-JOB " + job_id in tab
        assert "* * * * * echo x" in tab

        resp = daemon._handle_command({"action": "list"})
        assert resp["status"] == "ok"
        assert len(resp["jobs"]) == 1
        assert resp["jobs"][0]["id"] == job_id
        assert resp["jobs"][0]["command"] == "echo x"

    def test_add_empty_entry(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command({"action": "add", "entry": ""})
        assert resp["status"] == "error"
        assert "Missing" in resp["message"]

    def test_add_missing_entry(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command({"action": "add"})
        assert resp["status"] == "error"
        assert "Missing" in resp["message"]

    def test_add_multiline_entry_rejected(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command(
            {"action": "add", "entry": "* * * * * echo a\nrm -rf /"}
        )
        assert resp["status"] == "error"
        assert "single line" in resp["message"]

    def test_remove(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command(
            {"action": "add", "entry": "* * * * * echo x"}
        )
        job_id = resp["job_id"]
        resp = daemon._handle_command({"action": "remove", "job_id": job_id})
        assert resp["status"] == "ok"
        resp = daemon._handle_command({"action": "list"})
        assert resp["jobs"] == []

    def test_remove_nonexistent(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command({"action": "remove", "job_id": "nope"})
        assert resp["status"] == "error"
        assert "not found" in resp["message"]

    def test_remove_missing_id(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command({"action": "remove", "job_id": ""})
        assert resp["status"] == "error"

    def test_status(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command({"action": "status"})
        assert resp["status"] == "ok"
        assert resp["pid"] == os.getpid()
        assert resp["job_count"] == 0

    def test_status_counts_jobs(self, daemon: CronDaemon) -> None:
        daemon._handle_command(
            {"action": "add", "entry": "* * * * * echo a"}
        )
        daemon._handle_command(
            {"action": "add", "entry": "0 0 * * * echo b"}
        )
        resp = daemon._handle_command({"action": "status"})
        assert resp["job_count"] == 2

    def test_stop(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command({"action": "stop"})
        assert resp["status"] == "ok"
        assert daemon._stop_event.is_set()

    def test_unknown_action(self, daemon: CronDaemon) -> None:
        resp = daemon._handle_command({"action": "bogus"})
        assert resp["status"] == "error"

    def test_non_kiss_entries_preserved(
        self, daemon: CronDaemon, crontab_env: dict[str, Path]
    ) -> None:
        crontab_env["tab"].write_text("SHELL=/bin/bash\n0 9 * * * user_job\n")
        daemon._handle_command(
            {"action": "add", "entry": "* * * * * echo kiss"}
        )
        tab = crontab_env["tab"].read_text()
        assert "SHELL=/bin/bash" in tab
        assert "0 9 * * * user_job" in tab
        resp = daemon._handle_command({"action": "list"})
        assert len(resp["jobs"]) == 1

    def test_list_surface_crontab_error(self, tmp_path: Path) -> None:
        broken = tmp_path / "crontab"
        broken.write_text(
            f'#!{sys.executable}\nimport sys\n'
            'sys.stderr.write("disk error\\n"); sys.exit(2)\n'
        )
        broken.chmod(0o755)
        d = CronDaemon(
            sock_path=tmp_path / "s.sock",
            pid_path=tmp_path / "p.pid",
            crontab_cmd=str(broken),
        )
        resp = d._handle_command({"action": "list"})
        assert resp["status"] == "error"
        assert "failed" in resp["message"]

    def test_add_surfaces_write_error(self, tmp_path: Path) -> None:
        broken = tmp_path / "crontab"
        broken.write_text(
            f'#!{sys.executable}\n'
            'import sys\n'
            'args = sys.argv[1:]\n'
            'if args == ["-l"]:\n'
            '    sys.stderr.write("no crontab\\n"); sys.exit(1)\n'
            'sys.stderr.write("write denied\\n"); sys.exit(1)\n'
        )
        broken.chmod(0o755)
        d = CronDaemon(
            sock_path=tmp_path / "s.sock",
            pid_path=tmp_path / "p.pid",
            crontab_cmd=str(broken),
        )
        resp = d._handle_command(
            {"action": "add", "entry": "* * * * * echo x"}
        )
        assert resp["status"] == "error"
        assert "failed" in resp["message"]

    def test_add_preserves_no_trailing_newline(
        self, daemon: CronDaemon, crontab_env: dict[str, Path]
    ) -> None:
        """Existing crontab content without trailing newline is normalized."""
        crontab_env["tab"].write_text("EXISTING_LINE_NO_NEWLINE")
        resp = daemon._handle_command(
            {"action": "add", "entry": "* * * * * echo x"}
        )
        assert resp["status"] == "ok"
        tab = crontab_env["tab"].read_text()
        assert tab.startswith("EXISTING_LINE_NO_NEWLINE\n")
        assert "# KISS-JOB" in tab

    def test_remove_surfaces_write_error(self, tmp_path: Path) -> None:
        """``_remove_job`` surfaces a crontab-write RuntimeError as status=error."""
        tab = tmp_path / "tab"
        tab.write_text("# KISS-JOB abc ts\n* * * * * echo\n")
        script = tmp_path / "crontab"
        script.write_text(
            f'#!{sys.executable}\n'
            'import sys, pathlib\n'
            f'TAB = pathlib.Path({str(tab)!r})\n'
            'args = sys.argv[1:]\n'
            'if args == ["-l"]:\n'
            '    sys.stdout.write(TAB.read_text()); sys.exit(0)\n'
            'sys.stderr.write("write denied\\n"); sys.exit(1)\n'
        )
        script.chmod(0o755)
        d = CronDaemon(
            sock_path=tmp_path / "s.sock",
            pid_path=tmp_path / "p.pid",
            crontab_cmd=str(script),
        )
        resp = d._handle_command({"action": "remove", "job_id": "abc"})
        assert resp["status"] == "error"
        assert "failed" in resp["message"]

    def test_status_tolerates_crontab_error(self, tmp_path: Path) -> None:
        broken = tmp_path / "crontab"
        broken.write_text(
            f'#!{sys.executable}\nimport sys\n'
            'sys.stderr.write("boom\\n"); sys.exit(2)\n'
        )
        broken.chmod(0o755)
        d = CronDaemon(
            sock_path=tmp_path / "s.sock",
            pid_path=tmp_path / "p.pid",
            crontab_cmd=str(broken),
        )
        resp = d._handle_command({"action": "status"})
        assert resp["status"] == "ok"
        assert resp["job_count"] == -1
        assert "warning" in resp


class TestClientDaemonIntegration:
    def test_add_list_remove(self, client: CronClient) -> None:
        job_id = client.add_job("*/5 * * * * echo integration")
        assert isinstance(job_id, str)
        assert len(job_id) == 12

        jobs = client.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["id"] == job_id
        assert jobs[0]["schedule"] == "*/5 * * * *"
        assert jobs[0]["command"] == "echo integration"

        client.remove_job(job_id)
        assert client.list_jobs() == []

    def test_status(self, client: CronClient) -> None:
        resp = client.status()
        assert resp["status"] == "ok"
        assert resp["pid"] == os.getpid()
        assert resp["job_count"] == 0

    def test_add_multiline_raises(self, client: CronClient) -> None:
        with pytest.raises(RuntimeError, match="single line"):
            client.add_job("* * * * * echo a\nrm -rf /")

    def test_remove_unknown_raises(self, client: CronClient) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            client.remove_job("nosuch123456")

    def test_multiple_jobs_unique_ids(self, client: CronClient) -> None:
        ids = [client.add_job(f"0 {h} * * * echo j{h}") for h in range(5)]
        assert len(set(ids)) == 5
        jobs = client.list_jobs()
        assert len(jobs) == 5
        for jid in ids:
            client.remove_job(jid)
        assert client.list_jobs() == []

    def test_concurrent_clients(
        self, client: CronClient, crontab_env: dict[str, Path]
    ) -> None:
        results: list[str] = []
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                c = CronClient(sock_path=crontab_env["sock"])
                results.append(c.add_job(f"* * * * * echo c-{i}"))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"
        assert len(set(results)) == 10
        jobs = client.list_jobs()
        assert len(jobs) == 10

    def test_client_unreachable(self, tmp_path: Path) -> None:
        c = CronClient(sock_path=tmp_path / "no_such.sock")
        with pytest.raises(ConnectionError):
            c.list_jobs()


class TestDaemonStopViaClient:
    def test_stop_cleans_up(self, crontab_env: dict[str, Path]) -> None:
        d = CronDaemon(
            sock_path=crontab_env["sock"],
            pid_path=crontab_env["pid"],
            crontab_cmd=str(crontab_env["crontab"]),
        )
        t = threading.Thread(target=d.run, daemon=True)
        t.start()
        for _ in range(100):
            if crontab_env["sock"].exists():
                break
            time.sleep(0.05)

        c = CronClient(sock_path=crontab_env["sock"])
        c.stop_daemon()
        t.join(timeout=5)
        assert not t.is_alive()
        assert not crontab_env["sock"].exists()
        assert not crontab_env["pid"].exists()


class TestDaemonRestart:
    def test_jobs_persist_via_crontab_across_restart(
        self, crontab_env: dict[str, Path]
    ) -> None:
        def boot() -> tuple[CronDaemon, threading.Thread]:
            d = CronDaemon(
                sock_path=crontab_env["sock"],
                pid_path=crontab_env["pid"],
                crontab_cmd=str(crontab_env["crontab"]),
            )
            t = threading.Thread(target=d.run, daemon=True)
            t.start()
            for _ in range(100):
                if crontab_env["sock"].exists():
                    break
                time.sleep(0.05)
            return d, t

        _, t1 = boot()
        c1 = CronClient(sock_path=crontab_env["sock"])
        job_id = c1.add_job("0 12 * * * echo survive")
        c1.stop_daemon()
        t1.join(timeout=5)

        _, t2 = boot()
        c2 = CronClient(sock_path=crontab_env["sock"])
        jobs = c2.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["id"] == job_id
        assert jobs[0]["command"] == "echo survive"
        c2.stop_daemon()
        t2.join(timeout=5)


class TestProcessKillRestart:
    def test_kill_and_restart(self, crontab_env: dict[str, Path]) -> None:
        """Hard-killing the daemon must not lose jobs — they live in crontab."""
        src_root = str(Path(__file__).parents[3])
        sock = str(crontab_env["sock"])
        pid = str(crontab_env["pid"])
        cron = str(crontab_env["crontab"])
        daemon_script = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {src_root!r})
            from pathlib import Path
            from kiss.agents.third_party_agents.cron_manager_daemon import CronDaemon
            d = CronDaemon(
                sock_path=Path({sock!r}),
                pid_path=Path({pid!r}),
                crontab_cmd={cron!r},
            )
            d.run()
        """)

        def launch() -> subprocess.Popen[bytes]:
            proc = subprocess.Popen(
                [sys.executable, "-c", daemon_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(100):
                if crontab_env["sock"].exists():
                    return proc
                time.sleep(0.05)
            out, err = proc.communicate(timeout=2)
            raise RuntimeError(f"Daemon did not start: {err.decode()}")

        proc = launch()
        try:
            c = CronClient(sock_path=crontab_env["sock"])
            job_id = c.add_job("30 2 * * * echo killed")
            assert len(c.list_jobs()) == 1

            proc.kill()
            proc.wait(timeout=5)
            crontab_env["sock"].unlink(missing_ok=True)

            proc2 = launch()
            try:
                c2 = CronClient(sock_path=crontab_env["sock"])
                jobs = c2.list_jobs()
                assert len(jobs) == 1
                assert jobs[0]["id"] == job_id
                c2.add_job("0 0 * * * echo new")
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
            crontab_env["sock"].unlink(missing_ok=True)


class TestClientEdgeCases:
    def test_invalid_json(
        self, crontab_env: dict[str, Path], running_daemon: CronDaemon
    ) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(5.0)
            s.connect(str(crontab_env["sock"]))
            s.sendall(b"not json{{{")
            s.shutdown(socket.SHUT_WR)
            data = s.recv(4096)
            resp = json.loads(data)
            assert resp["status"] == "error"
            assert "Invalid JSON" in resp["message"]
        finally:
            s.close()

    def test_empty_connection(
        self, crontab_env: dict[str, Path], running_daemon: CronDaemon
    ) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(5.0)
            s.connect(str(crontab_env["sock"]))
        finally:
            s.close()
        time.sleep(0.1)
        c = CronClient(sock_path=crontab_env["sock"])
        assert c.status()["status"] == "ok"

    def test_message_too_large(
        self, crontab_env: dict[str, Path], running_daemon: CronDaemon
    ) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(10.0)
            s.connect(str(crontab_env["sock"]))
            s.sendall(b"x" * (1_048_576 + 10))
            s.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
            resp = json.loads(data)
            assert resp["status"] == "error"
            assert "too large" in resp["message"].lower()
        finally:
            s.close()

    def test_stop_daemon_when_gone(self, tmp_path: Path) -> None:
        CronClient(sock_path=tmp_path / "gone.sock").stop_daemon()


class TestLifecycleHelpers:
    def test_stop_daemon_no_pid_file(self, tmp_path: Path) -> None:
        msg = stop_daemon(
            sock_path=tmp_path / "s.sock", pid_path=tmp_path / "p.pid"
        )
        assert msg == "Daemon not running (no PID file)"

    def test_stop_daemon_stale_pid(self, tmp_path: Path) -> None:
        pid_path = tmp_path / "p.pid"
        pid_path.write_text("99999999")
        msg = stop_daemon(
            sock_path=tmp_path / "s.sock", pid_path=pid_path
        )
        assert "stale PID" in msg
        assert not pid_path.exists()

    def test_daemon_status_no_pid_file(self, tmp_path: Path) -> None:
        msg = daemon_status(
            pid_path=tmp_path / "p.pid", sock_path=tmp_path / "s.sock"
        )
        assert msg == "Daemon not running (no PID file)"

    def test_daemon_status_stale_pid(self, tmp_path: Path) -> None:
        pid_path = tmp_path / "p.pid"
        pid_path.write_text("99999999")
        msg = daemon_status(pid_path=pid_path, sock_path=tmp_path / "s.sock")
        assert "stale PID" in msg

    def test_daemon_status_running(
        self, crontab_env: dict[str, Path], running_daemon: CronDaemon
    ) -> None:
        msg = daemon_status(
            pid_path=crontab_env["pid"], sock_path=crontab_env["sock"]
        )
        assert "running" in msg
        assert f"pid={os.getpid()}" in msg
        assert "jobs=0" in msg

    def test_daemon_status_socket_unreachable(self, tmp_path: Path) -> None:
        pid_path = tmp_path / "p.pid"
        pid_path.write_text(str(os.getpid()))
        msg = daemon_status(
            pid_path=pid_path, sock_path=tmp_path / "no_such.sock"
        )
        assert "socket not responding" in msg

    def test_start_daemon_already_running(
        self, crontab_env: dict[str, Path], running_daemon: CronDaemon
    ) -> None:
        msg = start_daemon(
            sock_path=crontab_env["sock"],
            pid_path=crontab_env["pid"],
            foreground=True,
        )
        assert "already running" in msg


class TestCLI:
    def _run_cli(
        self, tmp_path: Path, *args: str
    ) -> subprocess.CompletedProcess[str]:
        """Invoke ``python -m`` CLI with a sandboxed $HOME so defaults are clean."""
        env = os.environ.copy()
        env["HOME"] = str(tmp_path)
        env["PYTHONPATH"] = str(Path(__file__).parents[3])
        return subprocess.run(
            [sys.executable, "-m", "kiss.agents.third_party_agents.cron_manager_daemon", *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_no_args_exits(self, tmp_path: Path) -> None:
        r = self._run_cli(tmp_path)
        assert r.returncode == 1
        assert "Usage:" in r.stdout

    def test_unknown_command(self, tmp_path: Path) -> None:
        r = self._run_cli(tmp_path, "wat")
        assert r.returncode == 1
        assert "Unknown command" in r.stdout

    def test_status_no_daemon(self, tmp_path: Path) -> None:
        r = self._run_cli(tmp_path, "status")
        assert r.returncode == 0
        assert "Daemon not running (no PID file)" in r.stdout

    def test_stop_no_daemon(self, tmp_path: Path) -> None:
        r = self._run_cli(tmp_path, "stop")
        assert r.returncode == 0
        assert "Daemon not running (no PID file)" in r.stdout

    def test_start_foreground_then_stop(self) -> None:
        """Launch the real daemon in the foreground, then stop it cleanly."""
        short_home = Path(f"/tmp/_kiss_cli_{os.getpid()}_{time.monotonic_ns()}")
        short_home.mkdir()
        env = os.environ.copy()
        env["HOME"] = str(short_home)
        env["PYTHONPATH"] = str(Path(__file__).parents[3])
        log_out = short_home / "daemon.log"
        with log_out.open("wb") as lo:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "kiss.agents.third_party_agents.cron_manager_daemon",
                    "start",
                    "--foreground",
                ],
                env=env,
                stdout=lo,
                stderr=lo,
                stdin=subprocess.DEVNULL,
            )
        try:
            sock = short_home / ".kiss" / "cron_manager.sock"
            for _ in range(200):
                if sock.exists():
                    break
                time.sleep(0.05)
            else:
                proc.kill()
                proc.wait(timeout=2)
                raise RuntimeError(
                    f"daemon start failed: {log_out.read_text()}"
                )

            stop = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kiss.agents.third_party_agents.cron_manager_daemon",
                    "stop",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert "Daemon stopped" in stop.stdout
            proc.wait(timeout=5)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            import shutil

            shutil.rmtree(short_home, ignore_errors=True)


class TestRunCronJobLifecycle:
    """Tests for the top-level ``run_cron_job_lifecycle`` function."""

    def test_parses_and_runs_full_lifecycle(
        self, crontab_env: dict[str, Path]
    ) -> None:
        """End-to-end: add → list → remove → stop against a real daemon."""
        d = CronDaemon(
            sock_path=crontab_env["sock"],
            pid_path=crontab_env["pid"],
            crontab_cmd=str(crontab_env["crontab"]),
        )
        t = threading.Thread(target=d.run, daemon=True)
        t.start()
        for _ in range(100):
            if crontab_env["sock"].exists():
                break
            time.sleep(0.05)

        result = run_cron_job_lifecycle(
            "*/5 * * * * echo hello", sock_path=crontab_env["sock"]
        )

        assert len(result["job_id"]) == 12
        # jobs list was captured AFTER add but BEFORE remove
        assert len(result["jobs"]) == 1
        assert result["jobs"][0]["id"] == result["job_id"]
        assert result["jobs"][0]["schedule"] == "*/5 * * * *"
        assert result["jobs"][0]["command"] == "echo hello"

        t.join(timeout=5)
        assert not t.is_alive()

    def test_complex_command(
        self, crontab_env: dict[str, Path]
    ) -> None:
        """Command part can contain spaces and special chars."""
        d = CronDaemon(
            sock_path=crontab_env["sock"],
            pid_path=crontab_env["pid"],
            crontab_cmd=str(crontab_env["crontab"]),
        )
        t = threading.Thread(target=d.run, daemon=True)
        t.start()
        for _ in range(100):
            if crontab_env["sock"].exists():
                break
            time.sleep(0.05)

        result = run_cron_job_lifecycle(
            "0 9 * * 1 /bin/sh -c 'echo monday morning'",
            sock_path=crontab_env["sock"],
        )

        assert len(result["job_id"]) == 12
        assert result["jobs"][0]["schedule"] == "0 9 * * 1"
        assert result["jobs"][0]["command"] == "/bin/sh -c 'echo monday morning'"
        t.join(timeout=5)

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            run_cron_job_lifecycle("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            run_cron_job_lifecycle("   ")

    def test_no_daemon_raises_connection_error(self, tmp_path: Path) -> None:
        """When no daemon is running, ConnectionError is raised."""
        with pytest.raises(ConnectionError):
            run_cron_job_lifecycle(
                "* * * * * echo hi",
                sock_path=tmp_path / "no_such.sock",
            )
