# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: the job name in ``/api/jobs/<job>/trajectories`` is decoded once.

``_process_request`` URL-decodes the whole request path before dispatch;
``_trajectory_job_response`` must NOT unquote the job segment a second
time, or a job directory whose name contains a literal percent-escape
(e.g. ``job%20a`` on disk) is double-decoded to ``job a`` and spuriously
404s.  The standalone visualizer decodes exactly once.
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


class TestJobNameSingleDecode(IsolatedAsyncioTestCase):
    """Job dirs with literal percent-escapes in their names must resolve."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self._jobs_root = Path(tempfile.mkdtemp(prefix="kiss_traj_pct_")) / "jobs"
        self._jobs_root.mkdir(parents=True)
        # A job directory whose on-disk name contains a literal ``%20``.
        _write_trajectory(self._jobs_root, "job%20a", "Agent Pct")
        _write_trajectory(self._jobs_root, "job_plain", "Agent Plain")
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

    async def test_percent_escaped_job_name_resolves(self) -> None:
        """A dir literally named ``job%20a`` is reachable via ``job%2520a``."""
        status, body = await self._http_get("/api/jobs/job%2520a/trajectories")
        self.assertEqual(status, 200)
        trajectories = json.loads(body)
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(trajectories[0]["name"], "Agent Pct")

    async def test_plain_job_name_still_resolves(self) -> None:
        """Behavior for normal (unescaped) job names is unchanged."""
        status, body = await self._http_get("/api/jobs/job_plain/trajectories")
        self.assertEqual(status, 200)
        trajectories = json.loads(body)
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(trajectories[0]["name"], "Agent Plain")

    async def test_traversal_job_name_still_rejected(self) -> None:
        """A job name decoding to path traversal is still rejected (400)."""
        status, _ = await self._http_get("/api/jobs/..%2Fetc/trajectories")
        self.assertEqual(status, 400)
