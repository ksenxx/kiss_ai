# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: a cloudflared the pidfile does not know about.

The pidfile (``<home>/cloudflared.pid``) is the daemon's only link to
the tunnel it detached so the public URL survives restarts.  On
2026-10-03 that link broke repeatedly: the install switched the brand
(and with it the home directory, ``~/.kiss`` vs ``~/.s10s``, each with
its own pidfile) and two further restarts found no pidfile at all.
Each time the daemon spawned a fresh quick tunnel — a new
``*.trycloudflare.com`` hostname, so the URL the user had stopped
working — while the previous cloudflared lived on, unmonitored, still
forwarding its public URL to port 8787.

The fix scans the process table for cloudflared processes whose
``--url`` targets the daemon's port (:func:`_find_cloudflared_forwarding_to`):
:func:`_try_adopt_existing_cloudflared` adopts a healthy one when the
pidfile yields nothing, and :func:`_terminate_stray_cloudflared` /
:func:`_terminate_orphan_cloudflared` kill every other one.

Real fake-cloudflared processes (an interpreter named ``cloudflared``,
see ``install_fake_cloudflared``), real HTTP metrics servers and a real
pidfile; no mocks.  Every fake forwards to a random free port, never to
8787, so a developer's live tunnel is never touched.
"""

from __future__ import annotations

import http.server
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path

from kiss.server import web_server as ws
from kiss.tests.conftest import install_fake_cloudflared, posix_only

_FAKE_BODY = "import time\ntime.sleep(120)\n"


class _HealthyHandler(http.server.BaseHTTPRequestHandler):
    """A cloudflared metrics endpoint with one ready edge connection."""

    hostname = "stray-tunnel.trycloudflare.com"

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path.startswith("/ready"):
            body = json.dumps({"readyConnections": 1})
        elif self.path.startswith("/quicktunnel"):
            body = json.dumps({"hostname": self.hostname})
        else:
            body = ""
        data = body.encode()
        self.send_response(200 if body else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        pass


class _NoUrlHandler(_HealthyHandler):
    """Reachable metrics endpoint: zero ready connections, no hostname."""

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        body = json.dumps({"readyConnections": 0}) if self.path.startswith("/ready") else ""
        data = body.encode()
        self.send_response(200 if body else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_until(predicate: Callable[[], bool], timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


@posix_only("the scan parses ``ps``, which Windows does not have")
class TestStrayCloudflared(unittest.TestCase):
    """Adoption and cleanup of cloudflared processes absent from the pidfile."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._old_pidfile = ws._CLOUDFLARED_PIDFILE
        ws._CLOUDFLARED_PIDFILE = self.dir / "cloudflared.pid"
        self.fake = install_fake_cloudflared(self.dir, _FAKE_BODY)
        self.local_port = _free_port()
        self._procs: list[subprocess.Popen[bytes]] = []
        self._httpds: list[http.server.HTTPServer] = []

    def tearDown(self) -> None:
        ws._CLOUDFLARED_PIDFILE = self._old_pidfile
        for p in self._procs:
            if p.poll() is None:
                p.kill()
            p.wait()
        for h in self._httpds:
            h.shutdown()
        self._tmp.cleanup()

    def _metrics_server(self, handler: type[_HealthyHandler]) -> int:
        httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self._httpds.append(httpd)
        return int(httpd.server_address[1])

    def _spawn_fake(self, metrics_port: int, local_port: int) -> subprocess.Popen[bytes]:
        """Spawn the fake with the argv the product itself uses."""
        proc = subprocess.Popen(
            [
                str(self.fake), "tunnel",
                "--metrics", f"127.0.0.1:{metrics_port}",
                "--url", f"https://localhost:{local_port}",
                "--no-tls-verify",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._procs.append(proc)
        self.assertTrue(
            _wait_until(lambda: ws._looks_like_cloudflared(proc.pid)),
            "fake cloudflared did not come up",
        )
        return proc

    def test_scan_matches_only_our_port(self) -> None:
        """The scan returns ``(pid, metrics_port)`` for our port only."""
        ours = self._spawn_fake(41001, self.local_port)
        other = self._spawn_fake(41002, _free_port())
        found = ws._find_cloudflared_forwarding_to(self.local_port)
        self.assertEqual(found, [(ours.pid, 41001)])
        self.assertIsNone(other.poll())

    def test_scan_ignores_cloudflared_without_metrics(self) -> None:
        """A cloudflared without ``--metrics`` cannot be adopted and is skipped."""
        proc = subprocess.Popen(
            [str(self.fake), "tunnel", "--url", f"https://localhost:{self.local_port}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._procs.append(proc)
        self.assertTrue(_wait_until(lambda: ws._looks_like_cloudflared(proc.pid)))
        self.assertEqual(ws._find_cloudflared_forwarding_to(self.local_port), [])

    def _spawn_raw(self, *args: str) -> subprocess.Popen[bytes]:
        proc = subprocess.Popen(
            [str(self.fake), *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._procs.append(proc)
        self.assertTrue(_wait_until(lambda: ws._looks_like_cloudflared(proc.pid)))
        return proc

    def test_scan_skips_processes_that_cannot_be_ours(self) -> None:
        """Only the daemon's own argv shape matches; everything else is left alone.

        Each spawned process below is a cloudflared the daemon never
        spawns and must therefore neither adopt nor kill: unparsable
        ports; a named tunnel (``tunnel run --url ...``, legal syntax);
        metrics on ``[::1]`` (the probes connect to 127.0.0.1, so it
        could never be monitored); a ``--url`` to another machine's
        port; a portless ``--url``.
        """
        p = self.local_port
        ours = f"https://localhost:{p}"
        loop4 = f"http://127.0.0.1:{p}"
        not_ours = [
            ("tunnel", "--metrics", "127.0.0.1:notaport", "--url", ours),
            ("tunnel", "--metrics", "127.0.0.1:41011", "--url", "https://localhost:alsonot"),
            ("tunnel", "--metrics", "127.0.0.1:41012", "run", "--url", loop4, "prod"),
            ("tunnel", "--metrics", "[::1]:41013", "--url", ours),
            ("tunnel", "--metrics", "127.0.0.1:41014", "--url", f"http://192.0.2.50:{p}"),
            ("tunnel", "--metrics", "127.0.0.1:41015", "--url", "https://localhost"),
            ("proxy-dns", "--metrics", "127.0.0.1:41016", "--url", ours),
            ("tunnel", "--metrics", "127.0.0.1:41020", "--url", "https://[::1"),
            ("tunnel", "--metrics", "127.0.0.1:41021", "--url", f"https://[localhost]:{p}"),
            ("tunnel", "--metrics", "127.0.0.1:\u00b2", "--url", ours),
            ("tunnel", "--metrics", "127.0.0.1:" + "4" * 5000, "--url", ours),
        ]
        for argv in not_ours:
            self._spawn_raw(*argv)
        good = self._spawn_fake(41003, p)
        ipv4 = self._spawn_raw("tunnel", "--metrics", "127.0.0.1:41017", "--url", loop4)
        ipv6 = self._spawn_raw("tunnel", "--metrics", "127.0.0.1:41018", "--url", f"http://[::1]:{p}")

        self.assertEqual(
            sorted(ws._find_cloudflared_forwarding_to(self.local_port)),
            sorted([(good.pid, 41003), (ipv4.pid, 41017), (ipv6.pid, 41018)]),
        )
        ws._terminate_stray_cloudflared(self.local_port, good.pid)
        self.assertTrue(_wait_until(lambda: ipv4.poll() is not None, 5))
        self.assertTrue(_wait_until(lambda: ipv6.poll() is not None, 5))
        for proc in self._procs:
            if proc not in (ipv4, ipv6):
                self.assertIsNone(proc.poll(), f"killed a cloudflared not ours: {proc.args!r}")

    def test_healthy_stray_is_adopted_without_pidfile(self) -> None:
        """No pidfile at all: the healthy stray is adopted and its URL kept."""
        metrics_port = self._metrics_server(_HealthyHandler)
        proc = self._spawn_fake(metrics_port, self.local_port)

        result = ws._try_adopt_existing_cloudflared(self.local_port)

        self.assertEqual(
            result,
            (proc.pid, metrics_port, "https://stray-tunnel.trycloudflare.com"),
        )
        self.assertIsNone(proc.poll(), "adopted stray must keep running")

    def test_without_local_port_no_scan_happens(self) -> None:
        """The legacy call without a port still consults only the pidfile."""
        metrics_port = self._metrics_server(_HealthyHandler)
        proc = self._spawn_fake(metrics_port, self.local_port)
        self.assertIsNone(ws._try_adopt_existing_cloudflared())
        self.assertIsNone(proc.poll())

    def test_dead_pidfile_pid_falls_back_to_stray(self) -> None:
        """A pidfile naming a dead pid no longer forces a fresh tunnel."""
        dead = self._spawn_fake(41019, self.local_port)
        dead.kill()
        dead.wait()
        ws._cloudflared_pidfile().write_text(json.dumps({
            "pid": dead.pid, "metrics_port": 1, "url": "https://old.trycloudflare.com",
        }))
        metrics_port = self._metrics_server(_HealthyHandler)
        proc = self._spawn_fake(metrics_port, self.local_port)

        result = ws._try_adopt_existing_cloudflared(self.local_port)

        self.assertEqual(
            result,
            (proc.pid, metrics_port, "https://stray-tunnel.trycloudflare.com"),
        )

    def test_malformed_pidfile_falls_back_to_stray(self) -> None:
        """A pidfile without an integer ``metrics_port`` is skipped, not fatal."""
        ws._cloudflared_pidfile().write_text(json.dumps({"pid": os.getpid()}))
        metrics_port = self._metrics_server(_HealthyHandler)
        proc = self._spawn_fake(metrics_port, self.local_port)
        self.assertEqual(
            ws._try_adopt_existing_cloudflared(self.local_port),
            (proc.pid, metrics_port, "https://stray-tunnel.trycloudflare.com"),
        )

    def test_recorded_pid_is_not_retried_from_the_scan(self) -> None:
        """A declined pidfile pid is terminated once and not re-probed as a stray."""
        no_url_port = self._metrics_server(_NoUrlHandler)
        declined = self._spawn_fake(no_url_port, self.local_port)
        ws._cloudflared_pidfile().write_text(json.dumps({
            "pid": declined.pid, "metrics_port": no_url_port,
        }))
        healthy_port = self._metrics_server(_HealthyHandler)
        healthy = self._spawn_fake(healthy_port, self.local_port)

        result = ws._try_adopt_existing_cloudflared(self.local_port)

        self.assertEqual(
            result,
            (healthy.pid, healthy_port, "https://stray-tunnel.trycloudflare.com"),
        )
        self.assertTrue(
            _wait_until(lambda: declined.poll() is not None, 5),
            "declined pidfile cloudflared must be terminated",
        )

    def test_stray_without_url_is_terminated(self) -> None:
        """A stray whose URL cannot be discovered is killed, not adopted."""
        metrics_port = self._metrics_server(_NoUrlHandler)
        proc = self._spawn_fake(metrics_port, self.local_port)

        self.assertIsNone(ws._try_adopt_existing_cloudflared(self.local_port))
        self.assertTrue(
            _wait_until(lambda: proc.poll() is not None, 5),
            "undiscoverable stray must not be left forwarding to our port",
        )

    def test_terminate_strays_keeps_own_tunnel_and_pidfile(self) -> None:
        """Only the OTHER tunnels die; our pid and pidfile survive."""
        own = self._spawn_fake(41004, self.local_port)
        stray_a = self._spawn_fake(41005, self.local_port)
        stray_b = self._spawn_fake(41006, self.local_port)
        elsewhere = self._spawn_fake(41007, _free_port())
        ws._cloudflared_pidfile().write_text(json.dumps({
            "pid": own.pid, "metrics_port": 41004, "url": "https://own.trycloudflare.com",
        }))

        ws._terminate_stray_cloudflared(self.local_port, own.pid)

        self.assertTrue(_wait_until(lambda: stray_a.poll() is not None, 5))
        self.assertTrue(_wait_until(lambda: stray_b.poll() is not None, 5))
        self.assertIsNone(own.poll(), "our own tunnel was killed")
        self.assertIsNone(elsewhere.poll(), "a tunnel for another port was killed")
        self.assertTrue(
            ws._cloudflared_pidfile().exists(),
            "stray cleanup must not unlink the daemon's own pidfile",
        )

    def test_terminate_cloudflared_pid_skips_non_cloudflared(self) -> None:
        """A pid that is not cloudflared is never signalled."""
        bystander = subprocess.Popen(
            ["sleep", "60"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._procs.append(bystander)
        ws._terminate_cloudflared_pid(bystander.pid)
        time.sleep(0.3)
        self.assertIsNone(bystander.poll(), "non-cloudflared process was signalled")

    def test_empty_password_orphan_cleanup_kills_strays_too(self) -> None:
        """With no password, strays aimed at our port die along with the pidfile pid."""
        recorded = self._spawn_fake(41008, self.local_port)
        stray = self._spawn_fake(41009, self.local_port)
        ws._cloudflared_pidfile().write_text(json.dumps({
            "pid": recorded.pid, "metrics_port": 41008,
        }))

        ws._terminate_orphan_cloudflared(self.local_port)

        self.assertTrue(_wait_until(lambda: recorded.poll() is not None, 5))
        self.assertTrue(_wait_until(lambda: stray.poll() is not None, 5))
        self.assertFalse(ws._cloudflared_pidfile().exists())

    def test_empty_password_orphan_cleanup_without_pidfile(self) -> None:
        """No pidfile: the stray alone is found and killed."""
        stray = self._spawn_fake(41010, self.local_port)
        ws._terminate_orphan_cloudflared(self.local_port)
        self.assertTrue(_wait_until(lambda: stray.poll() is not None, 5))


if __name__ == "__main__":
    unittest.main()
