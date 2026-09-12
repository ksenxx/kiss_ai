# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``_read_url_from_stderr`` must keep draining a surviving subprocess.

Concurrency audit (W5) regression: after the URL wait timed out, the
reader thread used to stop at the next stderr line while the callers
kept the ``cloudflared`` process alive (quick tunnel via the metrics
fallback, named tunnel unconditionally).  With nobody reading the
pipe, the child blocks on its next stderr write once the 64 KiB pipe
buffer fills — for cloudflared a whole-process deadlock — and the
tunnel watchdog, unable to reach the wedged metrics endpoint, never
restarts it.

Real subprocesses only; the child writes well past the pipe capacity
after the timeout so a stopped drain makes it hang.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from kiss.server.web_server import (
    RemoteAccessServer,
    _parse_quick_tunnel_url,
    _read_url_from_stderr,
)

_LINES_AFTER_TIMEOUT = 4000
"""4000 x ~100 B lines = ~400 KiB, several times the pipe capacity."""


def _spawn_chatty_child() -> subprocess.Popen[str]:
    """Start a child that logs one line, waits, then floods stderr."""
    return subprocess.Popen(
        [
            sys.executable, "-u", "-c",
            "import sys, time\n"
            "sys.stderr.write('INF starting, no url here\\n')\n"
            "time.sleep(0.5)\n"
            f"for i in range({_LINES_AFTER_TIMEOUT}):\n"
            "    sys.stderr.write('INF ' + ('x' * 90) + f' {i}\\n')\n",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


def _reader_threads() -> list[threading.Thread]:
    """Return the live ``_stderr_reader_loop`` threads."""
    return [
        t for t in threading.enumerate()
        if t.is_alive() and t.name.startswith("cloudflared-stderr-drain")
    ]


class TestStderrDrainSurvivesTimeout(unittest.TestCase):
    """The drain must outlive the URL wait for as long as the child lives."""

    def test_child_is_not_blocked_after_url_timeout(self) -> None:
        proc = _spawn_chatty_child()
        try:
            url = _read_url_from_stderr(
                proc, _parse_quick_tunnel_url, timeout=0.2,
            )
            self.assertIsNone(url)
            # A drained child finishes its flood in well under a
            # second; an undrained one blocks forever on the full pipe.
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.fail(
                    "child blocked on a full stderr pipe: the drain "
                    "thread stopped after the URL timeout",
                )
            self.assertEqual(proc.returncode, 0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_reader_thread_exits_when_child_dies(self) -> None:
        before = {t.ident for t in _reader_threads()}
        proc = _spawn_chatty_child()
        try:
            url = _read_url_from_stderr(
                proc, _parse_quick_tunnel_url, timeout=0.2,
            )
            self.assertIsNone(url)
            proc.wait(timeout=10)
            deadline = time.monotonic() + 5
            leaked: list = []
            while time.monotonic() < deadline:
                leaked = [t for t in _reader_threads() if t.ident not in before]
                if not leaked:
                    break
                time.sleep(0.05)
            self.assertEqual(
                leaked, [], "drain thread must exit at stderr EOF",
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_url_returned_and_drain_continues(self) -> None:
        """Happy path: the URL is returned early and draining goes on."""
        proc = subprocess.Popen(
            [
                sys.executable, "-u", "-c",
                "import sys\n"
                "sys.stderr.write('INF https://w5.trycloudflare.com\\n')\n"
                f"for i in range({_LINES_AFTER_TIMEOUT}):\n"
                "    sys.stderr.write('INF ' + ('y' * 90) + f' {i}\\n')\n",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        try:
            url = _read_url_from_stderr(
                proc, _parse_quick_tunnel_url, timeout=5,
            )
            self.assertEqual(url, "https://w5.trycloudflare.com")
            proc.wait(timeout=10)
            self.assertEqual(proc.returncode, 0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


class TestDrainSurvivesInvalidUtf8(unittest.TestCase):
    """The drain must survive a non-UTF-8 byte in a cloudflared log line.

    F3 regression (reviewer R3 finding 3): the production ``Popen`` in
    ``_spawn_cloudflared`` decoded stderr strictly, so one invalid byte
    raised ``UnicodeDecodeError`` inside the sole drain thread and
    killed it; the live cloudflared then blocked on its next write once
    the pipe filled.  The fake cloudflared below writes ``b"bad\\xff\\n"``
    first and then floods stderr well past the pipe capacity; only a
    surviving drain lets it finish.
    """

    def test_child_finishes_flood_after_invalid_byte(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_dir = Path(tmp.name)
        fake = tmp_dir / "cloudflared"
        fake.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            "sys.stderr.buffer.write(b'bad\\xff\\n')\n"
            "sys.stderr.buffer.flush()\n"
            f"for i in range({_LINES_AFTER_TIMEOUT}):\n"
            "    sys.stderr.write('INF ' + ('z' * 90) + f' {i}\\n')\n"
            "    sys.stderr.flush()\n",
        )
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{tmp_dir}{os.pathsep}{old_path}"
        self.addCleanup(os.environ.__setitem__, "PATH", old_path)
        server = RemoteAccessServer(
            use_tunnel=False,
            url_file=tmp_dir / "remote-url.json",
            uds_path=tmp_dir / "kiss.sock",
        )
        # The undrained child blocks on the full pipe during the
        # fail-fast window, so it is still alive and gets published.
        server._spawn_cloudflared(["--url", "http://x"], launch_prefix=[])
        proc = server._tunnel_proc
        assert proc is not None
        try:
            self.assertIsNone(proc.poll(), "child must still be alive")
            url = _read_url_from_stderr(
                proc, _parse_quick_tunnel_url, timeout=0.2,
            )
            self.assertIsNone(url)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.fail(
                    "child blocked on a full stderr pipe: the drain "
                    "thread died on the invalid UTF-8 byte",
                )
            self.assertEqual(proc.returncode, 0)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            if proc.stderr is not None:
                proc.stderr.close()


if __name__ == "__main__":
    unittest.main()
