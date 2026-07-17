# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the trajectory-visualizer routes on the web server.

Verify that ``/trajectories/`` serves the visualizer HTML page and that the
``/api/jobs`` and ``/api/jobs/<job>/trajectories`` endpoints return the same
JSON the standalone :mod:`kiss.viz_trajectory.server` produces.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import yaml

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server import web_server
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


class TestTrajectoryRoutes(IsolatedAsyncioTestCase):
    """Test the ``/trajectories/`` page and its JSON API on the web server."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        # Redirect the server's jobs-root resolver to a private temp dir so
        # the test does not depend on (or pollute) the real artifact dir.
        self._jobs_root = Path(tempfile.mkdtemp(prefix="kiss_traj_test_")) / "jobs"
        self._jobs_root.mkdir(parents=True)
        _write_trajectory(self._jobs_root, "job_2024_01_01_00_00_00_1", "Agent A")
        self._orig_get_jobs_root = web_server.get_jobs_root
        web_server.get_jobs_root = lambda *a, **k: self._jobs_root

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        web_server.get_jobs_root = self._orig_get_jobs_root
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

    async def test_trajectories_page_served_with_slash(self) -> None:
        """GET /trajectories/ returns the visualizer HTML page."""
        status, body = await self._http_get("/trajectories/")
        self.assertEqual(status, 200)
        self.assertIn("/api/jobs", body)
        self.assertIn("<html", body.lower())

    async def test_trajectories_page_served_without_slash(self) -> None:
        """GET /trajectories also returns the visualizer HTML page."""
        status, body = await self._http_get("/trajectories")
        self.assertEqual(status, 200)
        self.assertIn("<html", body.lower())

    async def test_api_jobs_lists_job(self) -> None:
        """GET /api/jobs returns the seeded job with its trajectory count."""
        status, body = await self._http_get("/api/jobs")
        self.assertEqual(status, 200)
        jobs = json.loads(body)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["name"], "job_2024_01_01_00_00_00_1")
        self.assertEqual(jobs[0]["trajectory_count"], 1)

    async def test_api_job_trajectories(self) -> None:
        """GET /api/jobs/<job>/trajectories returns parsed trajectories."""
        status, body = await self._http_get(
            "/api/jobs/job_2024_01_01_00_00_00_1/trajectories"
        )
        self.assertEqual(status, 200)
        trajectories = json.loads(body)
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(trajectories[0]["name"], "Agent A")
        self.assertEqual(trajectories[0]["model"], "test-model")
        self.assertEqual(trajectories[0]["messages"], [{"role": "user", "content": "hi"}])

    async def test_api_job_not_found(self) -> None:
        """GET for a missing job returns 404."""
        status, _ = await self._http_get("/api/jobs/job_missing/trajectories")
        self.assertEqual(status, 404)

    async def test_api_job_invalid_name(self) -> None:
        """A job name containing path-traversal characters is rejected (400)."""
        status, _ = await self._http_get("/api/jobs/..%2Fetc/trajectories")
        self.assertEqual(status, 400)
