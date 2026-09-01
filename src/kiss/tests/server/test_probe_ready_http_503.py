# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E repro: a deregistered cloudflared tunnel is never restarted.

Root cause of "the trycloudflare.com link cannot be reached":

Real ``cloudflared`` serves its ``/ready`` metrics endpoint with HTTP
status **503** (body ``{"status":503,"readyConnections":0,...}``) while
the tunnel has zero ready edge connections — i.e. exactly when the
public ``*.trycloudflare.com`` hostname has been deregistered and stops
resolving.  ``urllib.request.urlopen`` raises ``HTTPError`` for 503,
and ``_probe_tunnel_ready``'s blanket ``except Exception: return None``
converted the canonical "tunnel is dead" signal into "no information".
The watchdog therefore never incremented ``_tunnel_unhealthy_ticks``
and never force-restarted the dead tunnel, so the published URL stayed
unreachable until the whole kiss-web daemon happened to restart.

The previous test-suite simulated ``/ready`` with HTTP **200** +
``readyConnections: 0`` — a reply real cloudflared never sends — which
is why the suite stayed green while production was broken.

Every test below talks to a REAL local HTTP server that reproduces the
exact wire format of a real cloudflared metrics endpoint (verified live
with ``curl -w %{http_code} http://127.0.0.1:<metrics>/ready`` against a
deregistered tunnel: status 503).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

import kiss.server.web_server as ws_mod
from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import (
    _TUNNEL_STARTUP_GRACE,
    _TUNNEL_UNHEALTHY_LIMIT_QUICK,
    RemoteAccessServer,
    _probe_tunnel_ready,
    _try_adopt_existing_cloudflared,
)


class _FakeCloudflaredMetrics:
    """A real HTTP server that mimics cloudflared's metrics endpoint.

    Mirrors real cloudflared behaviour exactly: ``GET /ready`` replies
    with status 200 when ``ready_connections > 0`` and **503**
    otherwise, always with a JSON body of the shape
    ``{"status": <code>, "readyConnections": N, "connectorId": "..."}``.
    ``GET /quicktunnel`` replies 200 with ``{"hostname": ...}``.

    Individual tests may override ``ready_status`` / ``ready_body`` to
    exercise degenerate replies (non-JSON body, unexpected codes), or
    pass ``ready_replies`` — a scripted ``[(status, body), ...]``
    sequence consumed one reply per ``/ready`` request (the last entry
    repeats forever).  ``ready_requests`` counts ``/ready`` hits.
    """

    def __init__(
        self,
        ready_connections: int = 0,
        hostname: str = "e2e-dead-tunnel.trycloudflare.com",
        ready_status: int | None = None,
        ready_body: str | None = None,
        ready_replies: list[tuple[int, str]] | None = None,
    ) -> None:
        self.ready_connections = ready_connections
        self.hostname = hostname
        self.ready_requests = 0
        status = ready_status if ready_status is not None else (
            200 if ready_connections > 0 else 503
        )
        body = ready_body if ready_body is not None else json.dumps({
            "status": status,
            "readyConnections": ready_connections,
            "connectorId": "d1fd5827-e9cb-41a6-b60e-68df63d15aa0",
        })
        replies = list(ready_replies) if ready_replies else [(status, body)]
        fake = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/ready":
                    fake.ready_requests += 1
                    st, bd = (
                        replies.pop(0) if len(replies) > 1 else replies[0]
                    )
                    payload = bd.encode("utf-8")
                    self.send_response(st)
                elif self.path == "/quicktunnel":
                    payload = json.dumps(
                        {"hostname": fake.hostname},
                    ).encode("utf-8")
                    self.send_response(200)
                else:
                    payload = b"not found"
                    self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port: int = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Shut the server down and join its thread."""
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


class TestProbeAgainstRealCloudflaredWireFormat(unittest.TestCase):
    """``_probe_tunnel_ready`` vs the exact replies real cloudflared sends."""

    def _probe(self, **kwargs: object) -> bool | None:
        srv = _FakeCloudflaredMetrics(**kwargs)  # type: ignore[arg-type]
        try:
            return _probe_tunnel_ready(srv.port)
        finally:
            srv.close()

    def test_503_zero_ready_is_confirmed_unhealthy(self) -> None:
        """THE BUG: real cloudflared replies 503 when deregistered.

        A 503 + ``readyConnections: 0`` reply is the canonical
        "tunnel deregistered, public URL is NXDOMAIN" signal and must
        be reported as confirmed-unhealthy (``False``), not as
        "no information" (``None``).
        """
        self.assertIs(self._probe(ready_connections=0), False)

    def test_200_positive_ready_is_healthy(self) -> None:
        """A 200 + ``readyConnections > 0`` reply is healthy."""
        self.assertIs(self._probe(ready_connections=4), True)

    def test_200_zero_ready_is_confirmed_unhealthy(self) -> None:
        """Defensive: 200 + ``readyConnections: 0`` is still unhealthy."""
        self.assertIs(
            self._probe(ready_connections=0, ready_status=200), False,
        )

    def test_503_with_positive_body_trusts_the_body(self) -> None:
        """If the JSON body parses, its ``readyConnections`` wins."""
        self.assertIs(
            self._probe(
                ready_status=503,
                ready_body=json.dumps({"readyConnections": 2}),
            ),
            True,
        )

    def test_503_with_unparsable_body_is_confirmed_unhealthy(self) -> None:
        """A 503 from ``/ready`` means "not ready" even without JSON."""
        self.assertIs(
            self._probe(ready_status=503, ready_body="Service Unavailable"),
            False,
        )

    def test_503_with_non_numeric_ready_is_confirmed_unhealthy(self) -> None:
        """A 503 whose body has a junk ``readyConnections`` is unhealthy."""
        self.assertIs(
            self._probe(
                ready_status=503,
                ready_body=json.dumps({"readyConnections": "junk"}),
            ),
            False,
        )

    def test_other_http_error_is_unknown(self) -> None:
        """A non-503 HTTP error (proxy hiccup, schema change) is unknown."""
        self.assertIs(
            self._probe(ready_status=500, ready_body="boom"), None,
        )

    def test_unreachable_endpoint_is_unknown(self) -> None:
        """Connection refused must stay "unknown" (no URL rotation)."""
        srv = _FakeCloudflaredMetrics()
        srv.close()  # free the port -> connection refused
        self.assertIs(_probe_tunnel_ready(srv.port), None)

    def test_200_with_non_numeric_ready_is_unknown(self) -> None:
        """A 200 reply with a junk ``readyConnections`` value is unknown."""
        self.assertIs(
            self._probe(
                ready_status=200,
                ready_body=json.dumps({"readyConnections": "junk"}),
            ),
            None,
        )

    def test_200_with_unparsable_body_is_unknown(self) -> None:
        """A 200 reply with a non-JSON body is unknown."""
        self.assertIs(
            self._probe(ready_status=200, ready_body="not json"), None,
        )


class TestWatchdogRestartsDeregisteredTunnel(IsolatedAsyncioTestCase):
    """E2E: the watchdog must replace a tunnel whose ``/ready`` is 503.

    Reproduces the exact production incident: the cloudflared process
    is alive (here: a real ``sleep`` subprocess) and its metrics
    endpoint (here: a real HTTP server speaking real cloudflared's
    wire format) reports 503 + ``readyConnections: 0`` on every probe.
    ``_check_and_restart_tunnel`` runs with the REAL probe function —
    no substitution — and must terminate the dead tunnel and request a
    replacement within ``_TUNNEL_UNHEALTHY_LIMIT_QUICK`` ticks.
    Before the fix the probe returned ``None`` every tick, the
    unhealthy counter stayed at 0 forever, and the public URL remained
    unreachable until the daemon was manually restarted.
    """

    async def asyncSetUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self._orig_pidfile = ws_mod._CLOUDFLARED_PIDFILE
        ws_mod._CLOUDFLARED_PIDFILE = (
            Path(self._tmp.name) / "cloudflared.pid"
        )
        # A live tunnel implies a configured password: the watchdog
        # now TERMINATES a tracked tunnel when remote_password is
        # empty, which would short-circuit the health-tick logic
        # under test here.
        self._orig_config = (
            CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        )
        save_config({"remote_password": "test-secret-tunnel"})
        self._metrics = _FakeCloudflaredMetrics(ready_connections=0)
        self._proc = subprocess.Popen(
            ["sleep", "60"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.server = RemoteAccessServer(use_tunnel=False)
        self.server._loop = asyncio.get_event_loop()
        self.server._tunnel_proc = self._proc
        self.server._tunnel_metrics_port = self._metrics.port
        self.server._tunnel_started_at = (
            time.monotonic() - _TUNNEL_STARTUP_GRACE - 1
        )
        self.restart_calls: list[float] = []

        async def fake_restart() -> None:
            self.restart_calls.append(time.monotonic())

        self.server._restart_tunnel_url = fake_restart  # type: ignore[method-assign]

    async def asyncTearDown(self) -> None:
        ws_mod._CLOUDFLARED_PIDFILE = self._orig_pidfile
        if self._orig_config is None:
            CONFIG_PATH.unlink(missing_ok=True)
        else:
            CONFIG_PATH.write_text(self._orig_config)
        self._metrics.close()
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        self.server._tunnel_proc = None
        self._tmp.cleanup()

    async def test_dead_tunnel_is_force_restarted(self) -> None:
        """503 ticks must reach the limit and force-restart the tunnel."""
        for _ in range(_TUNNEL_UNHEALTHY_LIMIT_QUICK):
            await self.server._check_and_restart_tunnel()
        self.assertEqual(
            len(self.restart_calls), 1,
            "A tunnel whose /ready endpoint reports 503 + "
            "readyConnections=0 (real cloudflared wire format for a "
            "deregistered tunnel) was never force-restarted — the "
            "public *.trycloudflare.com URL stays dead forever.",
        )
        self.assertIsNotNone(
            self._proc.poll(),
            "The deregistered cloudflared process must be terminated "
            "before a replacement is spawned.",
        )
        self.assertEqual(self.server._tunnel_force_restart_count, 1)


class TestAdoptionWithRealCloudflaredWireFormat(unittest.TestCase):
    """E2E: startup adoption vs a metrics endpoint speaking real 503s.

    Two behaviours are pinned:

    1. A tunnel still reporting 503 after the bounded re-probe ladder
       is adopted TENTATIVELY when its public URL is known: 503 +
       ``readyConnections: 0`` is also what a mid-reconnect tunnel
       reports (network switch, wake from sleep), and the watchdog
       already owns the recover-vs-replace decision with a far larger
       tick budget.  Killing it here rotated the public URL on every
       kiss-web restart that landed mid-reconnect.
    2. A *briefly* not-ready tunnel (503 while reconnecting after
       wake / WiFi recovery, then healthy) is NOT killed — it is
       adopted and keeps its public URL.  Terminating on the first
       503 would needlessly rotate a recoverable URL.
    """

    def _start(
        self, ready_replies: list[tuple[int, str]],
    ) -> None:
        self._tmp = TemporaryDirectory()
        self._orig_pidfile = ws_mod._CLOUDFLARED_PIDFILE
        ws_mod._CLOUDFLARED_PIDFILE = (
            Path(self._tmp.name) / "cloudflared.pid"
        )
        self._metrics = _FakeCloudflaredMetrics(
            hostname="e2e-adopted.trycloudflare.com",
            ready_replies=ready_replies,
        )
        # The decline path verifies the process is really named
        # "cloudflared" before signalling it (stale-pidfile / recycled
        # PID protection), so exec a real subprocess through a symlink
        # named ``cloudflared`` (copying a system binary breaks its
        # code signature on macOS and the kernel SIGKILLs it at exec).
        fake_bin = Path(self._tmp.name) / "cloudflared"
        os.symlink(sys.executable, fake_bin)
        self._proc = subprocess.Popen(
            [str(fake_bin), "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        pidfile = ws_mod._CLOUDFLARED_PIDFILE
        assert pidfile is not None
        pidfile.write_text(
            json.dumps({
                "pid": self._proc.pid,
                "metrics_port": self._metrics.port,
                "url": "https://e2e-adopted.trycloudflare.com",
            }) + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        ws_mod._CLOUDFLARED_PIDFILE = self._orig_pidfile
        self._metrics.close()
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        self._tmp.cleanup()

    def test_persistent_503_adopts_tentatively_after_reprobes(self) -> None:
        """Constant 503 with a known URL -> re-probe ladder, adopt.

        503 + ``readyConnections: 0`` cannot distinguish "deregistered
        for good" from "mid-reconnect after a network switch", and the
        watchdog already tolerates this state for minutes before
        rotating the URL.  The adoption path therefore adopts the
        process tentatively (preserving the public URL) and leaves the
        recover-vs-replace decision to the watchdog's tick budget.
        """
        dead_reply = (503, json.dumps({
            "status": 503, "readyConnections": 0,
        }))
        self._start(ready_replies=[dead_reply])
        result = _try_adopt_existing_cloudflared()
        self.assertEqual(
            self._metrics.ready_requests, 5,
            "A 503 tunnel must still be re-probed through the full "
            "bounded ladder (1 initial + 4 retries) — a recovery "
            "during the window upgrades the tentative adoption to a "
            "confirmed one.",
        )
        self.assertEqual(
            result,
            (
                self._proc.pid,
                self._metrics.port,
                "https://e2e-adopted.trycloudflare.com",
            ),
            "A persistently-503 cloudflared with a known URL must be "
            "adopted tentatively so the public URL is preserved.",
        )
        self.assertIsNone(
            self._proc.poll(),
            "The tentatively adopted cloudflared must stay alive.",
        )

    def test_transient_503_recovers_and_is_adopted(self) -> None:
        """503 -> 503 -> healthy: the tunnel is adopted, not killed."""
        dead_reply = (503, json.dumps({
            "status": 503, "readyConnections": 0,
        }))
        healthy_reply = (200, json.dumps({
            "status": 200, "readyConnections": 4,
        }))
        self._start(
            ready_replies=[dead_reply, dead_reply, healthy_reply],
        )
        result = _try_adopt_existing_cloudflared()
        self.assertIsNotNone(
            result,
            "A tunnel that reconnects during the re-probe window must "
            "be adopted — killing it would rotate a recoverable URL.",
        )
        assert result is not None
        pid, metrics_port, url = result
        self.assertEqual(pid, self._proc.pid)
        self.assertEqual(metrics_port, self._metrics.port)
        self.assertEqual(url, "https://e2e-adopted.trycloudflare.com")
        self.assertIsNone(
            self._proc.poll(),
            "The adopted cloudflared must stay alive.",
        )


if __name__ == "__main__":
    unittest.main()
