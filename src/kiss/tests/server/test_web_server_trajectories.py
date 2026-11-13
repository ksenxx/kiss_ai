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
import base64
import json
import shutil
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from kiss.core.user_config import CONFIG_PATH, save_config
from kiss.server import web_server
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.conftest import find_free_port, no_verify_ssl, write_trajectory


class TestTrajectoryRoutes(IsolatedAsyncioTestCase):
    """Test the ``/trajectories/`` page and its JSON API on the web server."""

    async def asyncSetUp(self) -> None:
        self.port = find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        tmp = Path(tempfile.mkdtemp(prefix="kiss_traj_test_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        self._jobs_root = tmp / "jobs"
        self._jobs_root.mkdir(parents=True)
        (tmp / "work").mkdir()
        write_trajectory(self._jobs_root, "job_2024_01_01_00_00_00_1", "Agent A")
        self._orig_get_jobs_root = web_server.get_jobs_root
        web_server.get_jobs_root = lambda *a, **k: self._jobs_root

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=str(tmp / "work"),
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
        status, body, _headers = await self._fetch(path)
        return status, body

    async def _fetch(
        self,
        path: str,
        headers: dict[str, str] | list[tuple[str, str]] | None = None,
        port: int = 0,
    ) -> tuple[int, str, dict[str, str]]:
        """GET *path* with *headers*; return the status, body and reply headers.

        *headers* given as pairs may repeat a name: each pair is a header line.
        A ``Host`` among them replaces the one the client would send.
        """
        import http.client

        pairs = list(headers.items()) if isinstance(headers, dict) else headers or []
        own_host = any(name == "Host" for name, _ in pairs)

        def _get() -> tuple[int, str, dict[str, str]]:
            conn = http.client.HTTPSConnection(
                "127.0.0.1", port or self.port, timeout=5, context=no_verify_ssl(),
            )
            try:
                conn.putrequest("GET", path, skip_host=own_host)
                for name, value in pairs:
                    conn.putheader(name, value)
                conn.endheaders()
                resp = conn.getresponse()
                return resp.status, resp.read().decode(), dict(resp.headers)
            finally:
                conn.close()

        return await asyncio.get_event_loop().run_in_executor(None, _get)

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

    async def test_api_asks_a_relayed_visitor_for_the_remote_password(self) -> None:
        """The JSON data needs the password from everyone but this machine.

        A tunnel visitor reaches the daemon from loopback with a
        forwarded client IP; the WebSocket handshake would ask it for
        the password, so the trajectory endpoints must too (an HTTP
        Basic challenge the browser answers with a prompt).  A browser
        or program on this machine is served as before.
        """
        save_config({"remote_password": "traj-secret"})
        visitor = {"Cf-Connecting-Ip": "203.0.113.41"}
        job_path = "/api/jobs/job_2024_01_01_00_00_00_1/trajectories"
        for path in ("/api/jobs", job_path):
            status, body, headers = await self._fetch(path, visitor)
            self.assertEqual(status, 401, path)
            self.assertNotIn("job_2024", body)
            self.assertIn("Basic", headers.get("WWW-Authenticate", ""))
        # A proxy appends the visitor's address to the header it is sent:
        # an earlier hop is the visitor's own text and must not pass him
        # for this machine.  Neither do two credentials make one.
        right_pair = ("Authorization", "Basic " + _basic("traj-secret"))
        for claim in (
            [("X-Forwarded-For", "127.0.0.1, 203.0.113.41")],
            [("X-Forwarded-For", ", 203.0.113.41")],
            [("Cf-Connecting-Ip", "::1")],
            [("X-Forwarded-For", "127.0.0.1"), ("X-Forwarded-For", "203.0.113.41")],
            [*visitor.items(), right_pair, right_pair],
        ):
            status, body, _ = await self._fetch("/api/jobs", claim)
            self.assertEqual((status, "job_2024" in body), (401, False), claim)
        wrong = {**visitor, "Authorization": "Basic " + _basic("wrong")}
        status, body, _ = await self._fetch(job_path, wrong)
        self.assertEqual(status, 401)
        self.assertNotIn("Agent A", body)
        right = {**visitor, "Authorization": "Basic " + _basic("traj-secret")}
        for path, expected in (("/api/jobs", "job_2024"), (job_path, "Agent A")):
            status, body, _ = await self._fetch(path, right)
            self.assertEqual((status, expected in body), (200, True), path)
        status, body = await self._http_get(job_path)
        self.assertEqual((status, "Agent A" in body), (200, True), "direct loopback")
        # Remote access turned off: the old password opens nothing.
        save_config({"remote_password": ""})
        status, body, _ = await self._fetch(job_path, right)
        self.assertEqual((status, "Agent A" in body), (403, False))

    async def test_api_asks_a_page_of_another_site_for_the_remote_password(self) -> None:
        """What this machine's browser sends for another site's page is not a local request.

        After a DNS rebinding (the site's name re-pointed at 127.0.0.1)
        the page reaches the daemon over loopback as its own origin:
        only the ``Host`` it names tells its request from the web app's.
        A cross-site script's names the site as ``Origin``.  Both need
        the password, and get nothing while none is set.
        """
        job_path = "/api/jobs/job_2024_01_01_00_00_00_1/trajectories"
        foreign = (
            [("Host", f"evil.example:{self.port}")],
            [("Host", "evil.example")],
            [("Host", "[")],
            [("Host", f"127.0.0.1:{self.port}"), ("Host", "evil.example")],
            [("Origin", "https://evil.example")],
            [("Origin", "null")],
        )
        for password, refusal in (("", 403), ("traj-secret", 401)):
            save_config({"remote_password": password})
            for headers in foreign:
                for path in ("/api/jobs", job_path):
                    status, body, _ = await self._fetch(path, headers)
                    leaked = "job_2024" in body or "Agent A" in body
                    self.assertEqual((status, leaked), (refusal, False), (password, headers, path))
        right = ("Authorization", "Basic " + _basic("traj-secret"))
        status, body, _ = await self._fetch(job_path, [*foreign[0], right])
        self.assertEqual((status, "Agent A" in body), (200, True), "the password opens it")
        # The web app opened on this machine, under each of its names.
        for headers in (
            [("Host", f"localhost:{self.port}")],
            [("Host", f"[::1]:{self.port}"), ("Origin", f"http://[::1]:{self.port}")],
            [("Host", "LOCALHOST"), ("Origin", f"https://127.0.0.1:{self.port}")],
        ):
            status, body, _ = await self._fetch(job_path, headers)
            self.assertEqual((status, "Agent A" in body), (200, True), headers)

    async def test_private_daemon_serves_no_trajectory_data(self) -> None:
        """The token-only private daemon hands its jobs to no HTTP requester.

        It replaces an owner-only channel (:attr:`RemoteAccessServer.local_only`),
        and HTTP carries no local token, so the loopback admission of the
        public daemon's data routes would open the owner's trajectories
        to every user of the machine.
        """
        private = RemoteAccessServer(
            host="127.0.0.1", port=0, work_dir=self.server.work_dir,
            url_file=Path(tempfile.mkdtemp(prefix="kiss_private_")) / "unused-url.json",
            local_endpoint_file=Path(tempfile.mkdtemp(prefix="kiss_private_")) / "local.json",
        )
        await private.start_private_async()
        self.addAsyncCleanup(private.stop_async)
        job_path = "/api/jobs/job_2024_01_01_00_00_00_1/trajectories"
        for path in ("/api/jobs", job_path):
            status, body, _ = await self._fetch(path, port=private.port)
            self.assertEqual(status, 403, path)
            self.assertNotIn("job_2024", body)
            self.assertNotIn("Agent A", body)
        status, _body = await self._http_get("/api/jobs")
        self.assertEqual(status, 200, "the public daemon still serves this machine")

    async def test_api_password_guesses_share_the_handshake_lockout(self) -> None:
        """Wrong passwords lock the visitor's IP out; bare requests do not count."""
        save_config({"remote_password": "traj-secret"})
        visitor = {"X-Forwarded-For": "198.51.100.7"}
        right = {**visitor, "Authorization": "Basic " + _basic("traj-secret")}
        for _ in range(web_server._AUTH_FAIL_MAX):
            status, _body, _ = await self._fetch("/api/jobs", visitor)
            self.assertEqual(status, 401)
        status, _body, _ = await self._fetch("/api/jobs", right)
        self.assertEqual(status, 200, "credential-less page loads never lock anyone out")
        wrong = {**visitor, "Authorization": "Basic " + _basic("nope")}
        for _ in range(web_server._AUTH_FAIL_MAX):
            status, _body, _ = await self._fetch("/api/jobs", wrong)
            self.assertEqual(status, 401)
        status, body, headers = await self._fetch("/api/jobs", right)
        self.assertEqual((status, "job_2024" in body), (429, False))
        self.assertGreater(int(headers.get("Retry-After", "0")), 0)
        status, _body = await self._http_get("/api/jobs")
        self.assertEqual(status, 200, "the lockout is per visitor IP")


def _basic(password: str) -> str:
    """The ``Authorization: Basic`` credential a browser sends for *password*."""
    return base64.b64encode(f"trajectories:{password}".encode()).decode()
