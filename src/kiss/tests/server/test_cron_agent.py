# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Daemon-integration tests for the cron automations (cron_agent).

Extracted from
``kiss.tests.agents.third_party_agents.test_cron_agent``: these
scenarios exercise the kiss-web daemon plumbing (the SEA
loader in ``kiss.agents.sorcar.sea_apply`` and the scheduler thread started by
``kiss.server.web_server.RemoteAccessServer``), so their dependency
closure is kiss.agents.sorcar + kiss.server only — no third-party
channel module is imported.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.cron_agent import (
    cron_job,
    load_jobs,
    start_scheduler_thread,
)
from kiss.core import tool_interrupt
from kiss.tests.agents.sorcar.test_cron_agent import (  # noqa: F401
    _create,
    _cron_thread,
    _isolated_kiss_home,
    _set_job_fields,
    _stop_scheduler,
    _stops_a_run,
    _wait_for_run,
)
from kiss.tests.conftest import find_free_port, posix_only
from kiss.tests.local_ws import open_local_connection
from kiss.tests.server.pause import Pause

# The kiss-web daemon in a process of its own (its SIGTERM path ends in a failsafe
# ``os._exit``), a client on its loop.  Job A runs when the SIGTERM arrives; the
# scheduler's stop is held where it stops that run, after it has looked for the runs
# to stop, and the client asks for a run of job B, whose answer the script prints.
_SIGTERMED_DAEMON = r"""
import asyncio, json, os, signal, sys
from pathlib import Path

from kiss.agents.sorcar import cron_agent
from kiss.core import tool_interrupt
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import open_local_connection
from kiss.tests.server.pause import Pause

home = Path(os.environ["KISS_HOME"])


def stops_a_run(local_vars):
    return local_vars["exc_type"] is cron_agent._HostStopping


async def main():
    server = RemoteAccessServer(
        host="127.0.0.1", port=int(sys.argv[1]), use_tunnel=False, work_dir=str(home),
        local_endpoint_file=str(home / "sorcar-local.json"),
    )
    server._install_signal_handlers()
    serving = asyncio.ensure_future(server._serve_async())
    while server._shutdown_future is None:
        await asyncio.sleep(0.05)
    reader, writer = await open_local_connection(server)

    async def run_now(job_id):
        ask = {"type": "cronAction", "action": "run_now", "jobId": job_id}
        writer.write(json.dumps(ask).encode() + b"\n")
        await writer.drain()
        while True:
            reply = json.loads(await asyncio.wait_for(reader.readline(), 60))
            if reply.get("type") == "cronJobs":
                return reply["error"]

    await run_now("A")
    while not (home / "A").exists():
        await asyncio.sleep(0.05)
    with Pause(tool_interrupt.inject_async_exception.__code__, stops_a_run) as stop:
        os.kill(os.getpid(), signal.SIGTERM)
        assert await asyncio.to_thread(stop.paused.wait, 60)
        print(await run_now("B"))
    await serving


asyncio.run(main())
"""


def test_agent_script_loaded_run_now_uses_daemon_endpoint_file(
    tmp_path: Path,
) -> None:
    # A run_agent(agent="cron", ...) session gets its cron_job tool from a
    # FRESH synthetic module (the daemon's SEA loader re-executes
    # this file), whose own _daemon_endpoint_file global is never set:
    # run_now must still target the endpoint file recorded in the canonical
    # module by the daemon's scheduler thread.
    import sys

    from kiss.agents.sorcar.sea_commands import load_sea

    custom_endpoint = tmp_path / "custom-daemon.json"
    stop_event = start_scheduler_thread(
        interval=999.0, endpoint_file=str(custom_endpoint),
    )
    try:
        sea = load_sea(Path(cron_agent.__file__))
        loaded_cron_job = sea.tools([])[0]
        # A distinct module copy — the very situation the canonical
        # lookup exists for.
        assert loaded_cron_job is not cron_job
        assert type(sea).__module__ != cron_agent.__name__
        assert sys.modules[type(sea).__module__]._daemon_endpoint_file is None
        job = _create(loaded_cron_job(
            "create", name="llm", prompt="say hi", schedule="every 1h",
            deliver="none",
        ))
        reply = yaml.safe_load(loaded_cron_job("run_now", job_id=job["id"]))
        assert reply["ran"]["last_status"] == "error"
        assert "custom-daemon.json" in load_jobs()[0]["last_summary"]
    finally:
        _stop_scheduler(stop_event)

def test_kiss_web_daemon_runs_scheduler_thread(tmp_path: Path) -> None:
    """The kiss-web daemon seeds the default jobs, starts the cron thread,
    the thread executes a due job, and shutdown stops the thread."""
    import asyncio

    from kiss.server.web_server import RemoteAccessServer

    job = _create(cron_job(
        "create", name="boot job", command="echo ran inside daemon",
        schedule="every 1h",
    ))
    _set_job_fields(job["id"], next_run_at=time.time() - 1)  # due on the first tick

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    async def _run() -> None:
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=port,
            use_tunnel=False,
            work_dir=str(tmp_path),
            local_endpoint_file=str(tmp_path / "sorcar-local.json"),
        )
        task = asyncio.ensure_future(server._serve_async())
        try:
            for _ in range(200):
                if (
                    load_jobs()[0]["last_status"] == "ok"
                    and server._shutdown_future is not None
                ):
                    break
                await asyncio.sleep(0.05)
            assert _cron_thread() is not None, "cron scheduler thread not started"
            assert load_jobs()[0]["last_summary"] == "ran inside daemon"
            # The daemon start installed the default jobs after the user's.
            assert [j["name"] for j in load_jobs()[1:]] == [
                spec["name"] for spec in cron_agent.DEFAULT_JOBS
            ]
            assert load_jobs()[1]["prompt"] == "/recommendations now"
            assert load_jobs()[1]["schedule"] == "0 */6 * * *"
        finally:
            for _ in range(200):
                if server._shutdown_future is not None:
                    break
                await asyncio.sleep(0.05)
            assert server._shutdown_future is not None
            if not server._shutdown_future.done():
                server._shutdown_future.set_result(None)
            await task
        for _ in range(200):
            if _cron_thread() is None:
                return
            await asyncio.sleep(0.05)
        raise AssertionError("cron scheduler thread still alive after shutdown")

    asyncio.run(_run())


@posix_only("the jobs are sh commands and the daemon is stopped by SIGTERM")
def test_run_now_is_refused_once_the_daemon_is_shutting_down(tmp_path: Path) -> None:
    """A Schedule-panel ``run_now`` handled while the daemon shuts down starts no run.

    The scheduler's stop ends the runs that are alive when it looks, once.  A
    run started after that, by a ``cronAction`` the daemon served before it
    closed its listeners, was left running: a command job's processes outlived
    the daemon (audit 2026-11-21).  Asked once the shutdown has begun, the run
    is refused, as a submit is.
    """
    job = {
        "deliver": "local", "enabled": True, "schedule": "every 1h",
        "next_run_at": time.time() + 3600,
    }
    cron_agent.save_jobs([
        # Ends by itself should a failed run leave it behind.
        {**job, "id": "A", "name": "running", "command": f"echo $$ > {tmp_path / 'A'}; sleep 60"},
        {**job, "id": "B", "name": "late", "command": f"touch {tmp_path / 'B'}"},
    ])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    daemon = subprocess.run(
        [sys.executable, "-c", _SIGTERMED_DAEMON, str(port)],
        capture_output=True, text=True, timeout=120,
    )
    assert daemon.returncode == 0, daemon.stderr
    assert daemon.stdout.strip() == "Server is shutting down; job not started.", daemon.stderr
    assert not (tmp_path / "B").exists()


@posix_only("the jobs are sh commands")
def test_a_stopping_scheduler_host_starts_no_run(tmp_path: Path) -> None:
    """Nothing is started once the scheduler's stop has looked for the runs to end.

    The stop ends the runs that are alive when it looks, once, and the daemon
    serves on until its loop is gone.  The serve loop ends here without the
    SIGTERM latch, as on Ctrl-C.  A Schedule-panel ``run_now`` handled after
    that look, a request the daemon's executor still held when the stop
    returned and a tick that outlasted the stop's wait each started a run that
    nobody stopped: a command job's processes outlived the daemon (audit
    2026-11-21).  The refused tick claims nothing, so the occurrence stays due,
    and the refusal ends with the daemon's loop: a later host in the same
    process starts the job.
    """
    import asyncio

    from kiss.server.web_server import RemoteAccessServer

    job = {
        "deliver": "local", "enabled": True, "schedule": "every 1h",
        "next_run_at": time.time() + 3600,
    }
    cron_agent.save_jobs([
        # Ends by itself should a failed run leave it behind.
        {**job, "id": "A", "name": "running", "command": f"touch {tmp_path / 'A'}; sleep 60"},
        {**job, "id": "B", "name": "late", "command": f"touch {tmp_path / 'B'}"},
    ])

    async def _serve_and_stop() -> list[object]:
        server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), use_tunnel=False, work_dir=str(tmp_path),
            local_endpoint_file=str(tmp_path / "sorcar-local.json"),
        )
        serving = asyncio.ensure_future(server._serve_async())
        try:
            while server._shutdown_future is None:
                await asyncio.sleep(0.05)
            reader, writer = await open_local_connection(server)

            async def run_now(job_id: str) -> str:
                ask = {"type": "cronAction", "action": "run_now", "jobId": job_id}
                writer.write(json.dumps(ask).encode() + b"\n")
                await writer.drain()
                while True:
                    reply = json.loads(await asyncio.wait_for(reader.readline(), 60))
                    if reply.get("type") == "cronJobs":
                        return str(reply["error"])

            assert await run_now("A") == ""
            while not (tmp_path / "A").exists():
                await asyncio.sleep(0.05)
            with Pause(tool_interrupt.inject_async_exception.__code__, _stops_a_run) as stop:
                server._shutdown_future.set_result(None)
                assert await asyncio.to_thread(stop.paused.wait, 60)
                asked_meanwhile = await run_now("B")
            await serving
            writer.close()
            _set_job_fields("B", next_run_at=time.time() - 1)  # due: a tick would start it
            return [
                asked_meanwhile,
                await asyncio.to_thread(cron_agent.apply_job_action, "run_now", "B"),
                await asyncio.to_thread(cron_agent.tick),
            ]
        finally:
            # ``_serve_async`` only unwinds the listeners; the registry
            # watcher and its subscription belong to ``stop_async``.
            await server.stop_async()

    refused = "job 'B' is not started: the scheduler's host is shutting down"
    assert asyncio.run(_serve_and_stop()) == [refused, refused, 0]
    assert not (tmp_path / "B").exists()
    late = next(job for job in load_jobs() if job["id"] == "B")
    assert late.get("last_run_at") is None, "the refused tick claimed the occurrence"
    assert cron_agent.apply_job_action("run_now", "B") == ""
    _wait_for_run("B")
    assert (tmp_path / "B").exists()
