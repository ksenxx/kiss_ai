# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: declined cloudflared adoption must never kill a reused PID.

Commit 3fe9b33d added ``_terminate_declined_cloudflared`` so a
declined-but-alive cloudflared is not orphaned.  But the pidfile
records only a bare integer PID — if kiss-web (or the machine) died
without unlinking the pidfile and the OS later recycled that PID for
an UNRELATED process, the decline path would SIGTERM (and after ~2s
SIGKILL) an innocent process.  The pre-commit code never signalled the
recorded PID, so this was a regression.

The fix verifies the process behind the PID still looks like a
cloudflared binary (``ps -o comm=`` basename) before sending any
signal; a mismatch only unlinks the stale pidfile.

A later change made the adoption path symmetric with the watchdog: a
live cloudflared whose reachable metrics endpoint reports zero ready
connections (the mid-reconnect signature) is now adopted tentatively
when its public URL is known, instead of being terminated — killing it
rotated the public URL on every kiss-web restart (e.g. one triggered
by ``./install.sh``) that landed mid-reconnect.  Termination on the
decline path is reserved for processes whose URL cannot be discovered
at all or whose metrics endpoint is persistently unreachable (the
watchdog could never monitor such a tunnel).

These tests use a real subprocess, a real HTTP metrics server, and a
real pidfile.  No mocks.
"""

from __future__ import annotations

import http.server
import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from kiss.server import web_server as ws
from kiss.tests.conftest import posix_only

# A sibling daemon publishing its tunnel's record into the pidfile named
# by argv[1]; prints ``publishing`` once imported, just before the write.
_PUBLISH_FROM_SIBLING = """
import sys
from pathlib import Path
from kiss.server import web_server as ws
ws._CLOUDFLARED_PIDFILE = Path(sys.argv[1])
print("publishing", flush=True)
ws._save_cloudflared_pidfile(222, 2222, "https://successor.trycloudflare.com")
"""


class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    """Serves /ready reporting a confirmed-unhealthy tunnel."""

    ready_connections = 0
    hostname: str | None = None

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path.startswith("/ready"):
            body = json.dumps({"readyConnections": self.ready_connections})
        elif self.path.startswith("/quicktunnel") and self.hostname:
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


class TestAdoptDeclineStalePid(unittest.TestCase):
    """A reused (non-cloudflared) PID in the pidfile must not be killed."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._old_pidfile = ws._CLOUDFLARED_PIDFILE
        ws._CLOUDFLARED_PIDFILE = Path(self._tmp.name) / "cloudflared.pid"
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

    def _start_unhealthy_metrics_server(self) -> int:
        """Start a real metrics server reporting readyConnections=0."""
        httpd = http.server.HTTPServer(("127.0.0.1", 0), _MetricsHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self._httpds.append(httpd)
        return httpd.server_address[1]

    def _start_healthy_metrics_server(self, hostname: str) -> int:
        """Start a real metrics server of a ready tunnel advertising *hostname*."""
        handler = type("_Healthy", (_MetricsHandler,), {
            "ready_connections": 1, "hostname": hostname,
        })
        httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self._httpds.append(httpd)
        return httpd.server_address[1]

    def test_healthy_metrics_port_does_not_adopt_a_recycled_pid(self) -> None:
        """A21-4 (audit 2026-11-07): the identity check applies to a healthy candidate too.
        The stale record's pid is this test process, and its recorded metrics port
        is now a READY tunnel of someone else: adopting would advertise that
        tunnel's URL as ours and watch an unrelated process."""
        port = self._start_healthy_metrics_server("someone-elses.trycloudflare.com")
        self._write_pidfile(os.getpid(), port)
        local_port = 1 + max(h.server_address[1] for h in self._httpds)

        result = ws._try_adopt_existing_cloudflared(local_port)

        self.assertIsNone(result, "a recycled pid was adopted on a healthy metrics port")
        self.assertFalse(
            ws._cloudflared_pidfile().exists(),
            "stale (non-cloudflared) pidfile must still be unlinked",
        )

    def _write_pidfile(self, pid: int, metrics_port: int) -> None:
        ws._cloudflared_pidfile().write_text(json.dumps({
            "pid": pid,
            "metrics_port": metrics_port,
            "url": "https://saved.trycloudflare.com",
        }))

    def test_unrelated_process_with_reused_pid_survives(self) -> None:
        """Decline path must not signal a PID that is not cloudflared."""
        port = self._start_unhealthy_metrics_server()
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._procs.append(proc)
        self._write_pidfile(proc.pid, port)

        result = ws._try_adopt_existing_cloudflared()

        self.assertIsNone(result, "unhealthy tunnel must not be adopted")
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            self.assertIsNone(
                proc.poll(),
                "decline path killed an unrelated process whose PID was "
                "recycled from a stale cloudflared pidfile",
            )
            time.sleep(0.05)
        self.assertFalse(
            ws._cloudflared_pidfile().exists(),
            "stale (non-cloudflared) pidfile must still be unlinked",
        )

    def test_retiring_a_declined_pid_keeps_a_successors_record(self) -> None:
        """A21-3 (audit 2026-11-07): the unlink after the termination wait spares a
        record a successor daemon published meanwhile.  The recorded pid is this
        test process (not a cloudflared, so nothing is signalled) and the
        successor's publication lands at the entry of the unlink itself."""
        ws._save_cloudflared_pidfile(os.getpid(), 1111, None)
        unlink_code = ws._unlink_cloudflared_pidfile.__code__

        def _publish_successor(frame: object, event: str, arg: object) -> None:
            if event == "call" and getattr(frame, "f_code", None) is unlink_code:
                sys.settrace(None)
                ws._save_cloudflared_pidfile(222, 2222, "https://successor.trycloudflare.com")
            return None

        sys.settrace(_publish_successor)
        try:
            ws._terminate_declined_cloudflared(os.getpid(), grace_s=0.1)
        finally:
            sys.settrace(None)
        record = ws._load_cloudflared_pidfile()
        self.assertEqual(
            (record or {}).get("pid"), 222, f"the successor's record was deleted: {record}",
        )

    def _start_retirement_held_at_unlink(
        self, pid: int,
    ) -> tuple[threading.Thread, threading.Event]:
        """Retire *pid*'s record on a thread held right before its unlink.

        Returns the thread and the event that lets it go on.  While it
        is held, the record was read and compared and the unlink has
        not run: the window a successor's publication used to fall in.
        """
        code = ws._unlink_cloudflared_pidfile.__code__
        lines, first = inspect.getsourcelines(ws._unlink_cloudflared_pidfile)
        unlink_line = first + next(
            i for i, text in enumerate(lines) if "pidfile().unlink(" in text
        )
        held = threading.Event()
        go = threading.Event()

        def _trace(frame: object, event: str, arg: object) -> Any:
            if (
                getattr(frame, "f_code", None) is code and event == "line"
                and getattr(frame, "f_lineno", 0) == unlink_line and not held.is_set()
            ):
                held.set()
                go.wait(30)
            return _trace

        def _retire() -> None:
            sys.settrace(_trace)
            try:
                ws._unlink_cloudflared_pidfile(pid)
            finally:
                sys.settrace(None)

        thread = threading.Thread(target=_retire, daemon=True)
        self.addCleanup(go.set)
        thread.start()
        self.assertTrue(held.wait(10), "the retirement never reached its unlink")
        return thread, go

    def _assert_successor_record_kept(self) -> None:
        record = ws._load_cloudflared_pidfile()
        self.assertEqual(
            (record or {}).get("pid"), 222, f"the successor's record was deleted: {record}",
        )

    def test_a_publication_during_the_retirement_waits_for_it(self) -> None:
        """V7-F1 (audit 2026-11-08): the retirement's read, comparison and unlink
        are one locked step.  A successor publishing between the read and the
        unlink waits for the retirement instead of losing its record to it."""
        ws._save_cloudflared_pidfile(111, 1111, "https://retired.trycloudflare.com")
        retirement, go = self._start_retirement_held_at_unlink(111)
        publisher = threading.Thread(
            target=ws._save_cloudflared_pidfile,
            args=(222, 2222, "https://successor.trycloudflare.com"),
            daemon=True,
        )
        publisher.start()
        publisher.join(0.5)
        self.assertTrue(
            publisher.is_alive(), "the publication did not wait for the retirement",
        )
        go.set()
        retirement.join(10)
        publisher.join(10)
        self.assertFalse(retirement.is_alive() or publisher.is_alive())
        self._assert_successor_record_kept()

    def test_a_sibling_daemons_publication_waits_for_the_retirement(self) -> None:
        """The exclusion is a file lock: a sibling daemon's publication waits too."""
        ws._save_cloudflared_pidfile(111, 1111, "https://retired.trycloudflare.com")
        retirement, go = self._start_retirement_held_at_unlink(111)
        sibling = subprocess.Popen(
            [sys.executable, "-c", _PUBLISH_FROM_SIBLING, str(ws._cloudflared_pidfile())],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._procs.append(sibling)
        assert sibling.stdout is not None
        if sibling.stdout.readline().strip() != b"publishing":
            go.set()
            self.fail(
                "the sibling never got to publish: "
                f"{sibling.communicate(timeout=30)[1].decode(errors='replace')}",
            )
        try:
            sibling.wait(0.5)
        except subprocess.TimeoutExpired:
            pass
        self.assertIsNone(
            sibling.poll(), "the sibling's publication did not wait for the retirement",
        )
        go.set()
        retirement.join(10)
        _out, err = sibling.communicate(timeout=30)
        self.assertEqual(sibling.returncode, 0, err.decode(errors="replace"))
        self._assert_successor_record_kept()

    def test_similarly_named_process_survives(self) -> None:
        """An exact-name check: ``cloudflared-helper`` must NOT be killed."""
        port = self._start_unhealthy_metrics_server()
        link = Path(self._tmp.name) / "cloudflared-helper"
        os.symlink(sys.executable, link)
        proc = subprocess.Popen(
            [str(link), "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._procs.append(proc)
        self._write_pidfile(proc.pid, port)

        result = ws._try_adopt_existing_cloudflared()

        self.assertIsNone(result)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            self.assertIsNone(
                proc.poll(),
                "decline path killed a similarly-named but unrelated "
                "process (prefix match instead of exact match)",
            )
            time.sleep(0.05)

    @posix_only("a symlink named cloudflared cannot rename a Windows process image")
    def test_real_cloudflared_mid_reconnect_is_adopted_tentatively(
        self,
    ) -> None:
        """A live cloudflared with 0 ready connections and a known URL
        is ADOPTED, not killed.

        ``readyConnections == 0`` on a reachable metrics endpoint is
        the mid-reconnect signature (network switch, wake from sleep).
        The watchdog tolerates that state for minutes before rotating
        the URL; the adoption path must not be stricter, or every
        kiss-web restart landing mid-reconnect (e.g. one triggered by
        ``./install.sh``) kills a recoverable tunnel and resets the
        public URL.
        """
        port = self._start_unhealthy_metrics_server()
        link = Path(self._tmp.name) / "cloudflared"
        os.symlink(sys.executable, link)
        proc = subprocess.Popen(
            [str(link), "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._procs.append(proc)
        self._write_pidfile(proc.pid, port)

        result = ws._try_adopt_existing_cloudflared()

        self.assertEqual(
            result,
            (proc.pid, port, "https://saved.trycloudflare.com"),
            "mid-reconnect cloudflared with a known URL must be "
            "adopted tentatively so the public URL is preserved",
        )
        self.assertIsNone(
            proc.poll(),
            "mid-reconnect cloudflared was killed by the adoption path",
        )
        self.assertTrue(
            ws._cloudflared_pidfile().exists(),
            "pidfile of a tentatively adopted cloudflared must remain",
        )

    @posix_only("a symlink named cloudflared cannot rename a Windows process image")
    def test_real_cloudflared_without_known_url_is_terminated(self) -> None:
        """A 0-ready cloudflared whose URL is unknown IS terminated.

        Without a public URL the tunnel cannot be advertised, so the
        decline path must still reap the process (no orphan leak) and
        unlink its pidfile.
        """
        port = self._start_unhealthy_metrics_server()
        link = Path(self._tmp.name) / "cloudflared"
        os.symlink(sys.executable, link)
        proc = subprocess.Popen(
            [str(link), "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._procs.append(proc)
        ws._cloudflared_pidfile().write_text(json.dumps({
            "pid": proc.pid,
            "metrics_port": port,
        }))

        result = ws._try_adopt_existing_cloudflared()

        self.assertIsNone(result)
        deadline = time.monotonic() + 4.0
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNotNone(
            proc.poll(),
            "declined cloudflared was left running (orphan process leak)",
        )
        self.assertFalse(
            ws._cloudflared_pidfile().exists(),
            "pidfile of the declined cloudflared was not unlinked",
        )


if __name__ == "__main__":
    unittest.main()
