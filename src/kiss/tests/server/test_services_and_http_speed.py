# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End to end: a daemon's ``services.d`` helpers, and its gzip / immutable-media answers.

A real kiss-web daemon runs in its own process and ``KISS_HOME``
(:func:`kiss.tests.remote_daemon.start_remote_daemon`) and is reached over
plain HTTP on loopback, the way a browser on the same machine, or the s10s
hub's relay agent, reaches it.  It is stopped with ``SIGTERM``, as systemd or
launchd stop it.

Not reached here: :class:`kiss.server.http_speed.GzipCache` dropping its
oldest entries (64 MB of distinct compressed bodies) or compressing one body
for two requests at once, the restart delay going back to one second after a
service ran for a minute, and the races :class:`kiss.server.services.Services`
guards against (``start`` after ``stop``, a service spawned or exiting in the
instant the daemon stops).
"""

from __future__ import annotations

import gzip
import http.client
import json
import re
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.server.helpers import MEDIA_DIR
from kiss.server.http_speed import IMMUTABLE
from kiss.tests.conftest import posix_only
from kiss.tests.remote_daemon import RemoteDaemon, start_remote_daemon

RECORDER = """#!/bin/sh
cp "$KISS_HOME/sorcar-local.json" "$KISS_HOME/endpoint-seen.tmp"
mv "$KISS_HOME/endpoint-seen.tmp" "$KISS_HOME/endpoint-seen.json"
echo "started $$ in $(pwd)"
exec sleep 600
"""
FLAKY = """#!/bin/sh
echo run >> "$KISS_HOME/flaky-runs"
exit 3
"""
HIDDEN = """#!/bin/sh
touch "$KISS_HOME/hidden-ran"
"""
FORKER = """#!/bin/sh
sleep 600 &
echo $! >> "$KISS_HOME/forker-children"
"""  # breaks the rule that a service stays in the foreground


def wait_until(check: Callable[[], Any], seconds: float = 30.0) -> Any:
    """Poll *check* until it returns something truthy."""
    deadline = time.monotonic() + seconds
    while not (result := check()):
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.1)
    return result


def text_of(path: Path) -> str:
    """The file's text, or ``""`` while it does not exist."""
    return path.read_text() if path.is_file() else ""


def alive(pid: int) -> bool:
    """Whether process *pid* still runs (a zombie waiting to be reaped does not).

    ``ps`` rather than ``/proc``, which macOS does not have: its state
    column starts with ``Z`` for a zombie and is empty for a pid that
    is gone.
    """
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True,
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def executable(path: Path, text: str) -> None:
    """Write a script and make it executable."""
    path.write_text(text)
    path.chmod(0o755)


def stop(daemon: RemoteDaemon) -> None:
    """Stop the daemon as its service manager does, and wait for it to exit."""
    daemon.process.send_signal(signal.SIGTERM)
    try:
        daemon.process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        daemon.stop()
        raise


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """The daemon's ``KISS_HOME``."""
    return tmp_path / "home"


@posix_only("sh scripts, process groups and ps")
def test_the_daemon_runs_its_services_while_it_runs(home: Path) -> None:
    services = home / "services.d"
    services.mkdir(parents=True)
    executable(services / "recorder", RECORDER)
    executable(services / "flaky", FLAKY)
    executable(services / "forker", FORKER)
    executable(services / "broken", "#!/nonexistent/interpreter\n")  # cannot start
    executable(services / ".hidden", HIDDEN)  # dot files are not services
    (services / "notes.txt").write_text("not executable\n")
    (services / "subdir").mkdir()
    daemon = start_remote_daemon(home)
    try:
        # Started once the endpoint file it reads is written, in KISS_HOME.
        seen = wait_until(lambda: text_of(home / "endpoint-seen.json"))
        assert json.loads(seen)["pid"] == daemon.pid
        log = home / "logs" / "services" / "recorder.log"
        match = wait_until(lambda: re.search(r"started (\d+) in (.*)", text_of(log)))
        recorder, cwd = int(match.group(1)), match.group(2)
        assert Path(cwd).resolve() == home.resolve() and alive(recorder)
        # A service that exits is started again (after 1 s, then 2 s, ...).
        wait_until(lambda: len(text_of(home / "flaky-runs").split()) >= 2)
        assert not (home / "hidden-ran").exists()
        # What a service leaves running when it exits is stopped with it.
        child = int(wait_until(lambda: text_of(home / "forker-children").split())[0])
        wait_until(lambda: not alive(child), 20)
    finally:
        stop(daemon)
    # Stopping the daemon stops its services, and nothing restarts them.
    wait_until(lambda: not alive(recorder), 20)
    runs = (home / "flaky-runs").read_text()
    time.sleep(2.5)
    assert (home / "flaky-runs").read_text() == runs


def get(port: int, target: str, accept_encoding: str = "") -> tuple[int, dict[str, str], bytes]:
    """One GET from a browser on this machine (``Accept-Encoding: identity`` when none given)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    headers = {"Accept-Encoding": accept_encoding} if accept_encoding else {}
    conn.request("GET", target, headers=headers)
    response = conn.getresponse()
    body = response.read()
    conn.close()
    return response.status, {k.lower(): v for k, v in response.getheaders()}, body


@pytest.fixture
def daemon(home: Path) -> Iterator[RemoteDaemon]:
    """A daemon with no services."""
    running = start_remote_daemon(home)
    yield running
    stop(running)


def test_text_answers_are_compressed_and_hashed_media_kept(daemon: RemoteDaemon) -> None:
    status, headers, body = get(daemon.port, "/", "gzip, deflate, br")
    assert (status, headers["content-encoding"]) == (200, "gzip")
    assert headers["vary"] == "Accept-Encoding"
    assert int(headers["content-length"]) == len(body) and "no-store" in headers["cache-control"]
    page = gzip.decompress(body).decode()
    main_js = re.search(r'src="(/media/main\.js\?v=[0-9a-f]+)"', page)
    assert main_js is not None
    original = (MEDIA_DIR / "main.js").read_bytes()
    # A hashed media URL: compressed, and the browser may keep it for a year.
    status, headers, body = get(daemon.port, main_js.group(1), "gzip")
    assert (status, headers["content-encoding"]) == (200, "gzip")
    assert headers["cache-control"] == IMMUTABLE
    assert "pragma" not in headers and "expires" not in headers
    assert gzip.decompress(body) == original and len(body) < len(original) / 2
    # A browser that does not take gzip (or refuses it with q=0) gets the bytes as they are.
    for accept in ("", "gzip;q=0", "br"):
        status, headers, body = get(daemon.port, main_js.group(1), accept)
        assert (status, body, headers["cache-control"]) == (200, original, IMMUTABLE)
        assert "content-encoding" not in headers and headers["vary"] == "Accept-Encoding"
    assert get(daemon.port, main_js.group(1), "gzip;q=junk")[1]["content-encoding"] == "gzip"
    # The coding list may come as several header lines.
    conn = http.client.HTTPConnection("127.0.0.1", daemon.port, timeout=30)
    conn.putrequest("GET", main_js.group(1), skip_accept_encoding=True)
    conn.putheader("Accept-Encoding", "br")
    conn.putheader("Accept-Encoding", "gzip")
    conn.endheaders()
    response = conn.getresponse()
    assert (response.status, response.getheader("Content-Encoding")) == (200, "gzip")
    assert gzip.decompress(response.read()) == original
    conn.close()
    # Without its hash the same file stays uncached; a small file is not worth compressing.
    status, headers, body = get(daemon.port, "/media/main.js", "gzip")
    assert (headers["content-encoding"], "no-store" in headers["cache-control"]) == ("gzip", True)
    status, headers, body = get(daemon.port, "/media/brand.css", "gzip")
    assert (status, body) == (200, (MEDIA_DIR / "brand.css").read_bytes())
    assert "content-encoding" not in headers
    # An answer other than 200 is left alone.
    status, headers, _ = get(daemon.port, "/media/no-such-file.js", "gzip")
    assert status != 200 and "content-encoding" not in headers
