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
import http.server
import os
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


class _CountingHandler(http.server.BaseHTTPRequestHandler):
    """Serves ``server.payload`` slowly and counts the requests."""

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
from pathlib import Path
import kiss.server.web_server as ws
from kiss.core.utils import read_bytes_waiting_for_writer
ws.VOICE_MODEL_URL = sys.argv[1]
ws.VOICE_MODEL_CACHE = Path(sys.argv[2])
p = ws._ensure_voice_model()
print("NONE" if p is None else hashlib.sha256(read_bytes_waiting_for_writer(p)).hexdigest())
"""


def _leftovers(models_dir: Path) -> list[str]:
    return sorted(p.name for p in models_dir.iterdir())


def test_two_daemons_download_the_archive_once(served: Any, tmp_path: Path) -> None:
    """Two processes sharing the cache: one download, both get the intact archive."""
    cache = tmp_path / "models" / "model.tar.gz"
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), served.url, str(cache)],
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
    assert _leftovers(cache.parent) == [".model.tar.gz.lock", "model.tar.gz"]
    assert cache.read_bytes() == served.payload


def test_existing_archive_is_served_without_a_request(
    served: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "models" / "model.tar.gz"
    cache.parent.mkdir()
    cache.write_bytes(b"cached")
    monkeypatch.setattr(web_server, "VOICE_MODEL_URL", served.url)
    # Not through monkeypatch: reading the lazy attribute to save it would
    # freeze the default path as a permanent override on undo.
    web_server.VOICE_MODEL_CACHE = cache
    try:
        assert web_server._ensure_voice_model() == cache
    finally:
        del web_server.VOICE_MODEL_CACHE
    assert served.count == 0
    assert _leftovers(cache.parent) == ["model.tar.gz"], "no lock is taken for a cache hit"


def test_empty_archive_is_downloaded_again(served: Any, tmp_path: Path) -> None:
    """A zero-byte file (a crashed publish of an older build) does not count as cached."""
    cache = tmp_path / "models" / "model.tar.gz"
    cache.parent.mkdir()
    cache.touch()
    assert speech.download_once(served.url, cache) == cache
    assert served.count == 1
    assert read_bytes_waiting_for_writer(cache) == served.payload


def test_failed_download_returns_none_and_leaves_no_temp_file(
    served: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served.status = 503
    cache = tmp_path / "models" / "model.tar.gz"
    monkeypatch.setattr(web_server, "VOICE_MODEL_URL", served.url)
    web_server.VOICE_MODEL_CACHE = cache
    try:
        assert web_server._ensure_voice_model() is None
    finally:
        del web_server.VOICE_MODEL_CACHE
    assert served.count == 1
    assert _leftovers(cache.parent) == [".model.tar.gz.lock"]
    with open(cache.parent / ".model.tar.gz.lock", "w") as lock_file:
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
