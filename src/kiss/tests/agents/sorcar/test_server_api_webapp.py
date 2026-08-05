# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The remote webapp's whole wire surface is served by the server API.

The remote webapp (the browser page ``kiss-web`` serves) has exactly
three kinds of daemon interactions, and every one of them must be a
call into the code API in :mod:`kiss.server.sorcar`:

1. the WSS password handshake — serviced by
   :meth:`kiss.server.sorcar.ServerApi.authenticate`;
2. catalog commands over the authenticated WebSocket — routed by
   :meth:`kiss.server.sorcar.ServerApi.dispatch`;
3. the trajectory-viewer HTTP data endpoints — serviced by
   :meth:`kiss.server.sorcar.ServerApi.trajectory_jobs` /
   :meth:`kiss.server.sorcar.ServerApi.job_trajectories`.

These tests drive the three surfaces end-to-end against a live
``RemoteAccessServer`` over real WSS / HTTPS connections (no mocks),
plus direct calls of the API's pure payload methods.
"""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import yaml
from websockets.asyncio.client import connect

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server import web_server
from kiss.server.sorcar import ServerApi, passwords_equal
from kiss.server.web_server import RemoteAccessServer

_PASSWORD = "webapp-api-test-password"


def _pick_free_port() -> int:
    """Return an OS-assigned free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    """Permissive SSL context for the dev self-signed cert."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _write_trajectory(jobs_root: Path, job: str, name: str) -> None:
    """Create a minimal trajectory YAML under ``<jobs_root>/<job>``."""
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


class TestPasswordsEqual(unittest.TestCase):
    """`passwords_equal` — the API's constant-time password compare."""

    def test_equal_strings(self) -> None:
        """Identical strings compare equal."""
        self.assertTrue(passwords_equal("hunter2", "hunter2"))

    def test_different_strings(self) -> None:
        """Different strings of the same length compare unequal."""
        self.assertFalse(passwords_equal("hunter2", "hunter3"))

    def test_different_lengths(self) -> None:
        """Strings of different lengths compare unequal."""
        self.assertFalse(passwords_equal("a", "abcdef"))

    def test_unicode(self) -> None:
        """Non-ASCII strings are compared by UTF-8 bytes."""
        self.assertTrue(passwords_equal("café", "café"))
        self.assertFalse(passwords_equal("café", "cafe"))

    def test_web_server_alias(self) -> None:
        """``RemoteAccessServer._passwords_equal`` is the API compare."""
        self.assertTrue(RemoteAccessServer._passwords_equal("x", "x"))
        self.assertFalse(RemoteAccessServer._passwords_equal("x", "y"))


class TestTrajectoryApiMethods(unittest.TestCase):
    """The trajectory HTTP data endpoints' payloads come from the API."""

    def setUp(self) -> None:
        """Seed a private jobs root and point the resolver at it."""
        self._jobs_root = (
            Path(tempfile.mkdtemp(prefix="kiss_api_traj_")) / "jobs"
        )
        self._jobs_root.mkdir(parents=True)
        _write_trajectory(
            self._jobs_root, "job_2024_01_01_00_00_00_1", "Agent A",
        )
        self._orig_get_jobs_root = web_server.get_jobs_root
        web_server.get_jobs_root = lambda *a, **k: self._jobs_root

    def tearDown(self) -> None:
        """Restore the real jobs-root resolver."""
        web_server.get_jobs_root = self._orig_get_jobs_root

    def test_trajectory_jobs_lists_job(self) -> None:
        """``trajectory_jobs`` returns a 200 JSON job list."""
        status, ctype, body = ServerApi.trajectory_jobs()
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "application/json")
        jobs = json.loads(body)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["name"], "job_2024_01_01_00_00_00_1")

    def test_job_trajectories_ok(self) -> None:
        """``job_trajectories`` returns the parsed trajectory list."""
        status, ctype, body = ServerApi.job_trajectories(
            "/api/jobs/job_2024_01_01_00_00_00_1/trajectories"
        )
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "application/json")
        trajectories = json.loads(body)
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(trajectories[0]["name"], "Agent A")

    def test_job_trajectories_rejects_traversal(self) -> None:
        """A job name containing a path separator or ``..`` is a 400."""
        for bad in ("..", "a/b", "a\\b"):
            status, _ctype, body = ServerApi.job_trajectories(
                f"/api/jobs/{bad}/trajectories"
            )
            self.assertEqual(status, 400, bad)
            self.assertIn(b"Invalid job name", body)

    def test_job_trajectories_unknown_job_is_404(self) -> None:
        """An unknown job name is answered with a 404 JSON error."""
        status, _ctype, body = ServerApi.job_trajectories(
            "/api/jobs/no_such_job/trajectories"
        )
        self.assertEqual(status, 404)
        self.assertIn(b"not found", body)


class TestRemoteWebappThroughApi(IsolatedAsyncioTestCase):
    """A remote browser's full session runs on the server's code API.

    Drives the exact frame sequence ``_WS_SHIM_JS`` produces — the
    ``auth`` handshake (:meth:`ServerApi.authenticate`), then catalog
    commands on the authenticated socket (:meth:`ServerApi.dispatch`)
    — against a live ``RemoteAccessServer`` over real WSS.
    """

    async def asyncSetUp(self) -> None:
        """Start a real ``RemoteAccessServer`` with a known password."""
        self._port = _pick_free_port()
        self._orig_config: str | None = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": _PASSWORD})
        self._server = RemoteAccessServer(
            host="127.0.0.1",
            port=self._port,
            work_dir=tempfile.mkdtemp(),
            use_tunnel=False,
        )
        await self._server.start_async()

    async def asyncTearDown(self) -> None:
        """Stop the server and restore the user's saved config."""
        await self._server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def _ws_connect(self) -> Any:
        """Open a fresh WSS connection to /ws on the test server."""
        return await connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=_no_verify_ssl(),
        )

    async def _recv_type(self, ws: Any, expected: str) -> dict[str, Any]:
        """Read frames until one of type *expected* arrives."""
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if isinstance(msg, dict) and msg.get("type") == expected:
                return msg

    async def test_handshake_then_commands_run_on_the_api(self) -> None:
        """Wrong→right password, then commands, exactly like the shim."""
        async with await self._ws_connect() as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await self._recv_type(ws, "auth_required")
            await ws.send(
                json.dumps({"type": "auth", "password": _PASSWORD})
            )
            await self._recv_type(ws, "auth_ok")
            await ws.send(json.dumps({"type": "activeTasksQuery"}))
            reply = await self._recv_type(ws, "activeTasksResponse")
            self.assertIn("count", reply)
            self.assertIn("tabs", reply)
            await ws.send(
                json.dumps({"type": "definitelyNotACommand"})
            )
            err = await self._recv_type(ws, "error")
            self.assertIn("Unknown command", err.get("text", ""))
            await ws.send(
                json.dumps({"type": "auth", "password": "again"})
            )
            await ws.send(json.dumps({"type": "activeTasksQuery"}))
            await self._recv_type(ws, "activeTasksResponse")

    async def test_wrong_second_attempt_closes_with_error(self) -> None:
        """Two wrong passwords end the handshake with ``error``."""
        async with await self._ws_connect() as ws:
            await ws.send(json.dumps({"type": "auth", "password": "nope"}))
            await self._recv_type(ws, "auth_required")
            await ws.send(json.dumps({"type": "auth", "password": "nope2"}))
            err = await self._recv_type(ws, "error")
            self.assertEqual(err.get("text"), "Authentication failed")


if __name__ == "__main__":
    unittest.main()
