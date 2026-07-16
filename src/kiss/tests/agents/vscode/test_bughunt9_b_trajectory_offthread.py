# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: trajectory endpoints must not do disk I/O on the event loop.

``/trajectories``, ``/api/jobs`` and ``/api/jobs/<job>/trajectories``
walk the jobs root and parse trajectory YAML from disk.  Like the
neighbouring ``/voice-model.tar.gz`` and ``/media/`` branches, that work
must run via ``asyncio.to_thread`` so a large or slow jobs root does not
stall the event loop for every other client.  The test wraps the real
``list_jobs`` / ``load_job_trajectories`` functions (and the template
``Path``) with thin recorders that note which thread executes them, then
asserts none of them ran on the server's event-loop thread while the
responses stay byte-for-byte correct.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import tempfile
import threading
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import yaml

from kiss.server import web_server
from kiss.server.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import RemoteAccessServer


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _write_trajectory(jobs_root: Path, job: str, name: str) -> None:
    """Create a minimal trajectory YAML under ``<jobs_root>/<job>/trajectories``."""
    traj_dir = jobs_root / job / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "id": 1,
        "run_start_timestamp": 100,
        "run_end_timestamp": 200,
        "model": "test-model",
        "command": "do something",
        "step_count": 3,
        "max_steps": 10,
        "messages": [{"role": "user", "content": "hi"}],
    }
    (traj_dir / "trajectory_0.yaml").write_text(yaml.safe_dump(data))


class _ThreadRecordingPath(Path):
    """A real ``Path`` whose ``read_bytes`` records the calling thread."""

    recorded: list[int] = []

    def read_bytes(self) -> bytes:
        _ThreadRecordingPath.recorded.append(threading.get_ident())
        return super().read_bytes()


class TestTrajectoryEndpointsOffThread(IsolatedAsyncioTestCase):
    """Assert trajectory disk work runs off the server event-loop thread."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self._jobs_root = Path(tempfile.mkdtemp(prefix="kiss_traj_thr_")) / "jobs"
        self._jobs_root.mkdir(parents=True)
        _write_trajectory(self._jobs_root, "job_2024_01_01_00_00_00_1", "Agent A")
        self._orig_get_jobs_root = web_server.get_jobs_root
        web_server.get_jobs_root = lambda *a, **k: self._jobs_root

        # Wrap the real helpers with thin recorders (identical behavior).
        self._list_threads: list[int] = []
        self._load_threads: list[int] = []
        self._orig_list_jobs = web_server.list_jobs
        self._orig_load = web_server.load_job_trajectories
        orig_list, orig_load = self._orig_list_jobs, self._orig_load
        list_threads, load_threads = self._list_threads, self._load_threads

        def recording_list_jobs(artifact_dir: Path) -> list[dict]:
            list_threads.append(threading.get_ident())
            return orig_list(artifact_dir)

        def recording_load(artifact_dir: Path, job_name: str) -> list[dict]:
            load_threads.append(threading.get_ident())
            return orig_load(artifact_dir, job_name)

        web_server.list_jobs = recording_list_jobs
        web_server.load_job_trajectories = recording_load

        self._orig_template = web_server.TRAJECTORY_TEMPLATE
        _ThreadRecordingPath.recorded = []
        web_server.TRAJECTORY_TEMPLATE = _ThreadRecordingPath(
            str(self._orig_template)
        )

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()
        # IsolatedAsyncioTestCase runs the server on this very loop/thread.
        self._loop_thread = threading.get_ident()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        web_server.get_jobs_root = self._orig_get_jobs_root
        web_server.list_jobs = self._orig_list_jobs
        web_server.load_job_trajectories = self._orig_load
        web_server.TRAJECTORY_TEMPLATE = self._orig_template
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def _http_get(self, path: str) -> tuple[int, str]:
        import urllib.error
        import urllib.request

        url = f"https://127.0.0.1:{self.port}{path}"
        ctx = _no_verify_ssl()

        def _fetch() -> tuple[int, str]:
            try:
                resp = urllib.request.urlopen(url, timeout=5, context=ctx)
                return resp.status, resp.read().decode()
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode() if e.fp else ""

        return await asyncio.get_event_loop().run_in_executor(None, _fetch)

    async def test_api_jobs_runs_off_loop_thread(self) -> None:
        """``list_jobs`` for /api/jobs must not run on the event loop."""
        status, body = await self._http_get("/api/jobs")
        self.assertEqual(status, 200)
        jobs = json.loads(body)
        self.assertEqual(jobs[0]["name"], "job_2024_01_01_00_00_00_1")
        self.assertTrue(self._list_threads)
        for ident in self._list_threads:
            self.assertNotEqual(ident, self._loop_thread)

    async def test_job_trajectories_runs_off_loop_thread(self) -> None:
        """``load_job_trajectories`` must not run on the event loop."""
        status, body = await self._http_get(
            "/api/jobs/job_2024_01_01_00_00_00_1/trajectories"
        )
        self.assertEqual(status, 200)
        trajectories = json.loads(body)
        self.assertEqual(trajectories[0]["name"], "Agent A")
        self.assertTrue(self._load_threads)
        for ident in self._load_threads:
            self.assertNotEqual(ident, self._loop_thread)

    async def test_trajectories_page_read_off_loop_thread(self) -> None:
        """The visualizer template read must not run on the event loop."""
        status, body = await self._http_get("/trajectories/")
        self.assertEqual(status, 200)
        self.assertIn("<html", body.lower())
        self.assertTrue(_ThreadRecordingPath.recorded)
        for ident in _ThreadRecordingPath.recorded:
            self.assertNotEqual(ident, self._loop_thread)
