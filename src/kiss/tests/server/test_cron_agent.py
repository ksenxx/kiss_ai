# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Daemon-integration tests for the cron automations (cron_agent).

Extracted from
``kiss.tests.agents.third_party_agents.test_cron_agent``: these two
scenarios exercise the kiss-web daemon plumbing (the tools-file loader
in ``kiss.server.tools_file`` and the scheduler thread started by
``kiss.server.web_server.RemoteAccessServer``), so their dependency
closure is kiss.agents.sorcar + kiss.server only — no third-party
channel module is imported.
"""

from __future__ import annotations

import socket
from pathlib import Path

import yaml

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.cron_agent import (
    cron_job,
    load_jobs,
    start_scheduler_thread,
)
from kiss.tests.agents.sorcar.test_cron_agent import (  # noqa: F401
    _create,
    _cron_thread,
    _isolated_kiss_home,
    _set_job_fields,
    _stop_scheduler,
)


def test_tools_file_loaded_run_now_uses_daemon_sock_path(
    tmp_path: Path,
) -> None:
    # A run_agent("cron", ...) session gets its cron_job tool from a
    # FRESH synthetic module (the daemon's tools-file loader re-executes
    # this file), whose own _daemon_sock_path global is never set:
    # run_now must still target the socket recorded in the canonical
    # module by the daemon's scheduler thread.
    from kiss.server.tools_file import ToolsFileError, execute_python_file

    custom_sock = tmp_path / "custom-daemon.sock"
    stop_event = start_scheduler_thread(interval=999.0, sock_path=str(custom_sock))
    try:
        namespace = execute_python_file(
            cron_agent.__file__, ToolsFileError, "tools file",
        )
        loaded_cron_job = namespace["get_tools"]()[0]
        # A distinct module copy — the very situation the canonical
        # lookup exists for.
        assert loaded_cron_job is not cron_job
        assert namespace["_daemon_sock_path"] is None
        job = _create(loaded_cron_job(
            "create", name="llm", prompt="say hi", schedule="every 1h",
            deliver="none",
        ))
        reply = yaml.safe_load(loaded_cron_job("run_now", job_id=job["id"]))
        assert reply["ran"]["last_status"] == "error"
        assert "custom-daemon.sock" in load_jobs()[0]["last_summary"]
    finally:
        _stop_scheduler(stop_event)

def test_kiss_web_daemon_runs_scheduler_thread(tmp_path: Path) -> None:
    """The kiss-web daemon starts the cron thread, the thread executes a
    due job, and shutdown stops the thread."""
    import asyncio

    from kiss.server.web_server import RemoteAccessServer

    job = _create(cron_job(
        "create", name="boot job", command="echo ran inside daemon",
        schedule="every 1h",
    ))
    _set_job_fields(job["id"], next_run_at=1.0)  # due on the first tick

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    async def _run() -> None:
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=port,
            use_tunnel=False,
            work_dir=str(tmp_path),
            uds_path=str(tmp_path / "sorcar.sock"),
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
