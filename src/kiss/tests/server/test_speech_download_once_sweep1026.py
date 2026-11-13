# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of :func:`kiss.server.speech.download_once`.

One helper now backs both ``speech._ensure_downloaded_model`` (Vosk
zips, unpacked) and ``web_server._ensure_voice_model`` (the browser
wake-word ``.tar.gz``, served as is).  The web server used to guard
its download with an in-process ``threading.Lock`` only, so two
daemons sharing ``$KISS_HOME`` both downloaded the 40 MB archive; the
lock file next to the archive now serialises them and the second one
finds the first one's result.  Everything runs against a real local
HTTP server and real child processes.
"""

from __future__ import annotations

import hashlib
import http.client
import http.server
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.core.file_lock import lock_exclusive, unlock
from kiss.core.utils import read_bytes_waiting_for_writer
from kiss.server import speech, web_server
from kiss.tests.conftest import posix_only


class _CountingHandler(http.server.BaseHTTPRequestHandler):
    """Serves ``server.payload`` slowly and counts the requests.

    With ``server.cut_after`` set, it promises the whole payload but
    closes the connection after that many bytes of it.
    """

    def do_GET(self) -> None:  # noqa: N802 — http.server API
        server = cast(Any, self.server)
        with server.count_lock:
            server.count += 1
        if server.status != 200:
            self.send_error(server.status)
            return
        payload = server.payload
        self.send_response(200)
        self.send_header("Content-Type", "application/gzip")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if server.cut_after is not None:
            self.wfile.write(payload[: server.cut_after])
            self.wfile.flush()
            self.close_connection = True
            return
        chunk = 64 * 1024
        for i in range(0, len(payload), chunk):
            self.wfile.write(payload[i : i + chunk])
            self.wfile.flush()
            time.sleep(0.05)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


@pytest.fixture
def served() -> Any:
    """A local server with ``payload``, ``status`` and a request ``count``."""
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _CountingHandler)
    server = cast(Any, httpd)
    server.payload = os.urandom(512 * 1024)
    server.status = 200
    server.cut_after = None
    server.count = 0
    server.count_lock = threading.Lock()
    server.url = f"http://127.0.0.1:{httpd.server_address[1]}/model.tar.gz"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        httpd.shutdown()


_CHILD = """\
import hashlib, sys
import kiss.server.web_server as ws
from kiss.core.utils import read_bytes_waiting_for_writer
ws.VOICE_MODEL_URL = sys.argv[1]
p = ws._ensure_voice_model()
print("NONE" if p is None else hashlib.sha256(read_bytes_waiting_for_writer(p)).hexdigest())
"""


def _leftovers(models_dir: Path) -> list[str]:
    return sorted(p.name for p in models_dir.iterdir())


def test_two_daemons_download_the_archive_once(
    served: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two processes sharing the cache: one download, both get the intact archive."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))  # the home the two daemons share
    cache = web_server._voice_model_cache_path()
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), served.url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd(),
        )
        for _ in range(2)
    ]
    results = [p.communicate(timeout=120) for p in procs]
    expected = hashlib.sha256(served.payload).hexdigest()
    assert [out.strip() for out, _err in results] == [expected, expected], results
    assert served.count == 1, "the second daemon must reuse the first one's download"
    # The status line the daemon log shows is printed by the one that downloaded.
    downloaders = [err for _out, err in results if f"downloading {served.url} ..." in err]
    assert len(downloaders) == 1, results
    assert _leftovers(cache.parent) == [f".{cache.name}.lock", cache.name]
    assert cache.read_bytes() == served.payload


def test_existing_archive_is_served_without_a_request(
    served: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    cache = web_server._voice_model_cache_path()
    cache.parent.mkdir()
    cache.write_bytes(b"cached")
    monkeypatch.setattr(web_server, "VOICE_MODEL_URL", served.url)
    assert web_server._ensure_voice_model() == cache
    assert served.count == 0
    assert _leftovers(cache.parent) == [cache.name], "no lock is taken for a cache hit"


def test_empty_archive_is_downloaded_again(served: Any, tmp_path: Path) -> None:
    """A zero-byte file (a crashed publish of an older build) does not count as cached."""
    cache = tmp_path / "models" / "model.tar.gz"
    cache.parent.mkdir()
    cache.touch()
    assert speech.download_once(served.url, cache) == cache
    assert served.count == 1
    assert read_bytes_waiting_for_writer(cache) == served.payload


def test_an_archive_cut_short_of_its_content_length_is_not_published(
    served: Any, tmp_path: Path
) -> None:
    """A peer that closes before the promised bytes arrived fails the download, caching nothing.

    ``read1`` reports that early EOF as an ordinary end of data; the
    three bytes used to be published and served as the archive from
    then on, without another request.
    """
    served.payload = os.urandom(300 * 1024)
    served.cut_after = 3  # the handler writes this many bytes and closes
    cache = tmp_path / "models" / "model.tar.gz"
    with pytest.raises(http.client.IncompleteRead):
        speech.download_once(served.url, cache)
    assert _leftovers(cache.parent) == [".model.tar.gz.lock"]
    served.cut_after = None
    assert speech.download_once(served.url, cache) == cache
    assert served.count == 2
    assert read_bytes_waiting_for_writer(cache) == served.payload


def test_failed_download_returns_none_and_leaves_no_temp_file(
    served: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served.status = 503
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    cache = web_server._voice_model_cache_path()
    deaf = socket.socket()  # bound, not listening: a connection to it is refused
    deaf.bind(("127.0.0.1", 0))
    try:
        for url in (
            served.url,
            f"http://127.0.0.1:{deaf.getsockname()[1]}/model.tar.gz",
            f"http://{'a' * 64}.invalid/model.tar.gz",  # a label too long to look up
        ):
            monkeypatch.setattr(web_server, "VOICE_MODEL_URL", url)
            assert web_server._ensure_voice_model() is None
    finally:
        deaf.close()
    assert served.count == 1
    assert _leftovers(cache.parent) == [f".{cache.name}.lock"]
    with open(cache.parent / f".{cache.name}.lock", "w") as lock_file:
        assert lock_exclusive(lock_file, blocking=False), "the lock was not released"
        unlock(lock_file)


def test_waiting_for_a_wedged_holder_is_bounded(
    served: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A holder that never finishes costs a waiter twice the download deadline, not forever."""
    monkeypatch.setenv("KISS_VOICE_DOWNLOAD_TOTAL_TIMEOUT", "0.5")
    cache = tmp_path / "models" / "model.tar.gz"
    cache.parent.mkdir()
    with open(cache.parent / ".model.tar.gz.lock", "w") as held:
        lock_exclusive(held)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            speech.download_once(served.url, cache)
        elapsed = time.monotonic() - started
        unlock(held)
    assert 1.0 <= elapsed < 5.0, f"waited {elapsed:.1f}s"
    assert served.count == 0
    assert not cache.exists()


_STALLED_RESOLVER = """\
import fcntl, os, socket, struct, subprocess, sys, threading, time
from pathlib import Path
from kiss.core.file_lock import lock_exclusive
from kiss.server import speech
dest = Path(sys.argv[1])
# This namespace's resolver asks a nameserver that never answers, for minutes.
conf = Path(os.path.realpath("/etc/resolv.conf"))
ours = dest.parent / "resolv.conf"
ours.write_text("nameserver 127.0.0.1\\noptions timeout:30 attempts:5\\n")
subprocess.run(["mount", "-t", "tmpfs", "tmpfs", "/run"], check=True)  # no resolver daemon to ask
if conf.is_relative_to("/run"):
    conf.parent.mkdir(parents=True)
    conf.write_text(ours.read_text())
else:
    subprocess.run(["mount", "--bind", str(ours), str(conf)], check=True)
fcntl.ioctl(socket.socket(), 0x8914, struct.pack("16sH", b"lo", 1))  # SIOCSIFFLAGS: lo up
silent = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
silent.bind(("127.0.0.1", 53))
probe = threading.Thread(target=socket.getaddrinfo, args=("stalled.invalid", 80), daemon=True)
probe.start()
probe.join(2)
if not probe.is_alive():
    sys.exit("the resolver does not stall here")
speech._LOOKUPS = threading.BoundedSemaphore(2)
os.environ["no_proxy"] = "*"  # the first lookup is of the URL's host, the second of the proxy
for host in ("direct", "proxied", "third"):
    started = time.monotonic()
    try:
        speech.download_once(f"http{'s' * (host == 'proxied')}://{host}.invalid/m.zip", dest)
    except OSError as error:
        with open(dest.parent / f".{dest.name}.lock", "w") as lock:
            free = lock_exclusive(lock, blocking=False)
        print(type(error).__name__, free, time.monotonic() - started < 15)
    os.environ.update(no_proxy="", https_proxy="http://proxy.invalid:3128")
"""


@posix_only("a resolver stalled in a network namespace of its own")
def test_a_stalled_address_lookup_ends_at_the_download_deadline(tmp_path: Path) -> None:
    """The lookup of the server's (or the proxy's) address is bounded like the transfer.

    ``getaddrinfo`` takes no timeout and a lookup has no socket for the
    deadline to shut down: it held the download, and the lock its
    caller holds, for as long as the resolver cared to wait.  Lookups
    left behind stalled are capped: with two of them, the third is
    refused at once.  The resolver stalls for real: in a throwaway
    namespace its one nameserver is a socket that never answers.
    """
    in_namespace = ["unshare", "-Urnm"]
    if (
        shutil.which("unshare") is None
        or subprocess.run([*in_namespace, "true"], capture_output=True).returncode
    ):
        pytest.skip("no unprivileged namespaces here")
    child = subprocess.run(
        [*in_namespace, sys.executable, "-c", _STALLED_RESOLVER, str(tmp_path / "model.zip")],
        env={
            **os.environ,
            "KISS_VOICE_DOWNLOAD_TOTAL_TIMEOUT": "1",
            "KISS_VOICE_DOWNLOAD_TIMEOUT": "30",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    if "does not stall" in child.stderr:
        pytest.skip(child.stderr.strip())
    assert child.stdout.splitlines() == [
        "TimeoutError True True",
        "TimeoutError True True",
        "URLError True True",
    ], child.stdout + child.stderr
