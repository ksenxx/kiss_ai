# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The voice-model download must be timeout-bounded (daemon-hang fix).

Concurrency audit 2026 (WS-C1, HANG): ``_download_voice_model_to``
used ``urllib.request.urlretrieve``, which accepts no timeout, while
its caller ``_ensure_voice_model`` held the module-level
``_voice_model_lock`` on a default-executor thread.  A black-holed
connection (captive portal, firewall silently dropping packets) then
blocked that worker forever; every retrying ``/voice-model.tar.gz``
request parked another shared executor worker on the lock until the
whole daemon stopped dispatching commands.  ``voice_wake``'s
``_download_url_to_file`` exists precisely because of this bug class
(its docstring says so); the fix reuses it.

Tested for real: a local TCP server that accepts the connection and
never sends a byte.  ``web_server.VOICE_MODEL_URL`` and
``web_server.VOICE_MODEL_CACHE`` are documented, supported module
attribute overrides (see ``_voice_model_cache_path``'s docstring and
``test_wave2_webserver_bugs.py``); the per-read timeout is the
``KISS_VOICE_DOWNLOAD_TIMEOUT`` environment override read at call
time by ``voice_wake.download_timeout_seconds``.
"""

from __future__ import annotations

import http.server
import os
import socket
import tempfile
import threading
import unittest
from pathlib import Path

import kiss.server.web_server as ws


class TestVoiceModelDownloadTimeout(unittest.TestCase):
    """A stalled download errors out instead of wedging the lock holder."""

    def setUp(self) -> None:
        self._saved_url = ws.VOICE_MODEL_URL
        self._saved_cache = getattr(ws, "VOICE_MODEL_CACHE", None)
        self._saved_timeout = os.environ.get("KISS_VOICE_DOWNLOAD_TIMEOUT")
        self._saved_total = os.environ.get("KISS_VOICE_DOWNLOAD_TOTAL_TIMEOUT")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        ws.VOICE_MODEL_CACHE = Path(self._tmp.name) / "model.tar.gz"
        os.environ["KISS_VOICE_DOWNLOAD_TIMEOUT"] = "1"

    def tearDown(self) -> None:
        ws.VOICE_MODEL_URL = self._saved_url
        if self._saved_cache is None:
            if hasattr(ws, "VOICE_MODEL_CACHE"):
                del ws.VOICE_MODEL_CACHE
        else:
            ws.VOICE_MODEL_CACHE = self._saved_cache
        if self._saved_timeout is None:
            os.environ.pop("KISS_VOICE_DOWNLOAD_TIMEOUT", None)
        else:
            os.environ["KISS_VOICE_DOWNLOAD_TIMEOUT"] = self._saved_timeout
        if self._saved_total is None:
            os.environ.pop("KISS_VOICE_DOWNLOAD_TOTAL_TIMEOUT", None)
        else:
            os.environ["KISS_VOICE_DOWNLOAD_TOTAL_TIMEOUT"] = self._saved_total

    def test_black_holed_download_errors_instead_of_hanging(self) -> None:
        """A server that accepts and never replies cannot wedge the lock."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        accepted: list[socket.socket] = []

        def black_hole() -> None:
            try:
                conn, _ = listener.accept()
            except OSError:  # listener closed at cleanup
                return
            accepted.append(conn)  # keep the connection open, send nothing

        acceptor = threading.Thread(target=black_hole, daemon=True)
        acceptor.start()
        def close_accepted() -> None:
            for c in accepted:
                c.close()

        self.addCleanup(close_accepted)
        ws.VOICE_MODEL_URL = f"http://127.0.0.1:{port}/model.tar.gz"

        result: list[object] = []

        def fetch() -> None:
            result.append(ws._ensure_voice_model())

        worker = threading.Thread(target=fetch, daemon=True)
        worker.start()
        worker.join(timeout=20)
        self.assertFalse(
            worker.is_alive(),
            "BUG: the voice-model download hung on a black-holed "
            "connection while holding _voice_model_lock",
        )
        self.assertEqual(result, [None])
        self.assertFalse(ws.VOICE_MODEL_CACHE.exists())
        # The lock was released: a follow-up caller is not convoyed.
        self.assertTrue(ws._voice_model_lock.acquire(timeout=1))
        ws._voice_model_lock.release()

    def test_slow_trickle_download_hits_the_total_deadline(self) -> None:
        """Review finding 3: a peer sending one byte before every
        per-read socket timeout keeps each read alive forever — only a
        wall-clock TOTAL deadline can bound such a download.  The
        trickling server here would defeat any per-read timeout; the
        download must still error out promptly and release
        ``_voice_model_lock``."""
        os.environ["KISS_VOICE_DOWNLOAD_TIMEOUT"] = "0.5"
        os.environ["KISS_VOICE_DOWNLOAD_TOTAL_TIMEOUT"] = "1"
        stop_sending = threading.Event()
        self.addCleanup(stop_sending.set)

        class _TrickleHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 — stdlib handler API
                self.send_response(200)
                self.send_header("Content-Length", "1000000")
                self.end_headers()
                try:
                    while not stop_sending.is_set():
                        self.wfile.write(b"x")
                        self.wfile.flush()
                        stop_sending.wait(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                """Silence per-request stderr logging."""

        httpd = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), _TrickleHandler,
        )
        self.addCleanup(httpd.server_close)
        server_thread = threading.Thread(
            target=httpd.serve_forever, daemon=True,
        )
        server_thread.start()
        self.addCleanup(httpd.shutdown)
        ws.VOICE_MODEL_URL = (
            f"http://127.0.0.1:{httpd.server_address[1]}/model.tar.gz"
        )

        result: list[object] = []
        worker = threading.Thread(
            target=lambda: result.append(ws._ensure_voice_model()),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=15)
        self.assertFalse(
            worker.is_alive(),
            "BUG: a slow-trickle download outlived its total deadline "
            "while holding _voice_model_lock",
        )
        self.assertEqual(result, [None])
        self.assertFalse(ws.VOICE_MODEL_CACHE.exists())
        self.assertTrue(ws._voice_model_lock.acquire(timeout=1))
        ws._voice_model_lock.release()

    def test_total_deadline_env_rejects_nonfinite_values(self) -> None:
        """``inf`` / ``1e999`` must not disable the total deadline —
        the parser falls back to the finite default."""
        from kiss.server.voice_wake import (
            DEFAULT_DOWNLOAD_TOTAL_TIMEOUT_SECONDS,
            download_total_timeout_seconds,
        )

        for raw in ("inf", "1e999", "nan", "-1", "0", "junk", ""):
            os.environ["KISS_VOICE_DOWNLOAD_TOTAL_TIMEOUT"] = raw
            self.assertEqual(
                download_total_timeout_seconds(),
                DEFAULT_DOWNLOAD_TOTAL_TIMEOUT_SECONDS,
                f"non-usable override {raw!r} must fall back",
            )
        os.environ["KISS_VOICE_DOWNLOAD_TOTAL_TIMEOUT"] = "2.5"
        self.assertEqual(download_total_timeout_seconds(), 2.5)

    def test_working_download_still_succeeds_and_caches(self) -> None:
        """The timeout-bounded replacement still downloads correctly."""
        payload = b"fake-model-archive-bytes"

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                """Silence per-request stderr logging."""

        httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        self.addCleanup(httpd.server_close)
        server_thread = threading.Thread(
            target=httpd.serve_forever, daemon=True,
        )
        server_thread.start()
        self.addCleanup(httpd.shutdown)
        port = httpd.server_address[1]
        ws.VOICE_MODEL_URL = f"http://127.0.0.1:{port}/model.tar.gz"

        got = ws._ensure_voice_model()
        self.assertEqual(got, ws.VOICE_MODEL_CACHE)
        self.assertEqual(ws.VOICE_MODEL_CACHE.read_bytes(), payload)
        # Cached: a second call returns without re-downloading.
        self.assertEqual(ws._ensure_voice_model(), ws.VOICE_MODEL_CACHE)


if __name__ == "__main__":
    unittest.main()
