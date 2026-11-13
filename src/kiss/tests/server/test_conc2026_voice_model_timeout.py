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
whole daemon stopped dispatching commands.  ``kiss.server.speech``'s
``_download_url_to_file`` exists precisely because of this bug class
(its docstring says so); the fix reuses it.

Tested for real: a local TCP server that accepts the connection and
never sends a byte.  ``web_server.VOICE_MODEL_URL`` is pointed at it,
the archive lives in a ``KISS_HOME`` of the test's own, and the
per-read timeout is the ``KISS_VOICE_DOWNLOAD_TIMEOUT`` environment
override read at call time by ``speech.download_timeout_seconds``.
"""

from __future__ import annotations

import http.server
import os
import socket
import ssl
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import kiss.server.web_server as ws
from kiss.core.file_lock import lock_exclusive, unlock
from kiss.server import speech
from kiss.tests.local_ws import make_test_tls


def _trickle(listener: socket.socket, head: bytes, tail: bytes, stop: threading.Event) -> None:
    """Serve one connection of *listener*: *head* at once, then *tail* a byte every 0.05 s."""
    try:
        conn, _ = listener.accept()
    except OSError:  # listener closed at cleanup
        return
    with conn:
        try:
            conn.recv(65536)  # the request
            conn.sendall(head)
            for i in range(len(tail)):
                if stop.wait(0.05):
                    return
                conn.sendall(tail[i : i + 1])
        except OSError:  # the download gave up: what the tests want
            return


def _assert_download_lock_released(cache: Path) -> None:
    """The flock ``speech.download_once`` takes next to *cache* is free again."""
    with open(cache.parent / f".{cache.name}.lock", "w", encoding="utf-8") as lock_file:
        assert lock_exclusive(lock_file, blocking=False), "the download lock was not released"
        unlock(lock_file)


class TestVoiceModelDownloadTimeout(unittest.TestCase):
    """A stalled download errors out instead of wedging the lock holder."""

    def setUp(self) -> None:
        self._saved_url = ws.VOICE_MODEL_URL
        self._saved_timeout = os.environ.get("KISS_VOICE_DOWNLOAD_TIMEOUT")
        self._saved_total = os.environ.get("KISS_VOICE_DOWNLOAD_TOTAL_TIMEOUT")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.enterContext(pytest.MonkeyPatch.context()).setenv("KISS_HOME", self._tmp.name)
        self.cache = ws._voice_model_cache_path()
        self.cache.parent.mkdir()
        os.environ["KISS_VOICE_DOWNLOAD_TIMEOUT"] = "1"

    def tearDown(self) -> None:
        ws.VOICE_MODEL_URL = self._saved_url
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
        self.assertFalse(self.cache.exists())
        # The lock was released: a follow-up caller is not convoyed.
        _assert_download_lock_released(self.cache)

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
        self.assertFalse(self.cache.exists())
        _assert_download_lock_released(self.cache)

    def test_trickled_http_framing_hits_the_total_deadline(self) -> None:
        """The deadline covers the status line, the headers and a chunk's size line too.

        ``urlopen`` reads the headers, and ``read1`` a chunked body's
        size lines, in as many socket reads as the peer makes of them:
        a byte before every per-read timeout used to keep the download
        running past its total deadline, with the lock held.  So did an
        HTTPS proxy's reply to ``CONNECT``, which ``connect()`` reads
        before it returns the socket to register.
        """
        os.environ["KISS_VOICE_DOWNLOAD_TIMEOUT"] = "1"
        os.environ["KISS_VOICE_DOWNLOAD_TOTAL_TIMEOUT"] = "0.5"
        stop_sending = threading.Event()
        self.addCleanup(stop_sending.set)
        environ = self.enterContext(pytest.MonkeyPatch.context())
        slow = b"x" * 200  # a byte every 0.05 s: 10 s of framing, 20 deadlines' worth
        proxy_reply = b"HTTP/1.1 200 Connection established\r\n"
        for at_once, trickled in (
            (b"HTTP/1.1 200 OK\r\nContent-Length: 1000000\r\n", b"X-Slow: " + slow + b"\r\n\r\n"),
            (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n", b"5;ext=" + slow + b"\r\n"),
            (proxy_reply, b"X-Proxy: " + slow + b"\r\n\r\n"),
        ):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.addCleanup(listener.close)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            threading.Thread(
                target=_trickle, args=(listener, at_once, trickled, stop_sending), daemon=True,
            ).start()
            url = f"http://127.0.0.1:{port}/model.tar.gz"
            if at_once is proxy_reply:  # the listener is the proxy the environment names
                environ.setenv("https_proxy", f"http://127.0.0.1:{port}")
                environ.setenv("no_proxy", "")
                url = "https://model.invalid/model.tar.gz"
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                speech._download_url_to_file(url, self.cache)
            self.assertLess(time.monotonic() - started, 5.0, trickled[:8])

    def test_a_socket_registered_after_the_deadline_is_shut_down_too(self) -> None:
        """The deadline ends a connection made as it fires, not only those made before it.

        The deadline timer runs once: it shut down the sockets registered
        by then and a socket registered a moment later (a connect that
        completed as the deadline fired, a redirect's next connection)
        was left alone, so a server trickling its headers kept the
        download, and the lock its caller holds, blocked past the
        deadline for as long as it pleased.  Registering on an expired
        watch must end the transfer there and then.
        """
        os.environ["KISS_VOICE_DOWNLOAD_TIMEOUT"] = "1"
        stop_sending = threading.Event()
        self.addCleanup(stop_sending.set)
        head = b"HTTP/1.1 200 OK\r\nContent-Length: 1000000\r\n"
        tail = b"X-Slow: " + b"x" * 200 + b"\r\n\r\n"  # 10 s of headers, a byte per read
        listeners = []
        for _ in range(2):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.addCleanup(listener.close)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            threading.Thread(
                target=_trickle, args=(listener, head, tail, stop_sending), daemon=True,
            ).start()
            listeners.append(listener)
        # The watch of a download whose deadline fired before its connection registered.
        watched = speech._WatchedHandler(time.monotonic() + 60)
        watched.expire()
        started = time.monotonic()
        with self.assertRaises(OSError):  # a URLError or a RemoteDisconnected
            urllib.request.build_opener(watched).open(
                f"http://127.0.0.1:{listeners[0].getsockname()[1]}/model.tar.gz", timeout=1,
            )
        self.assertLess(time.monotonic() - started, 1.0, "the late socket was not shut down")
        # Through the download itself: a deadline so short that it fires
        # around the connect, whichever side of the registration it lands on.
        os.environ["KISS_VOICE_DOWNLOAD_TOTAL_TIMEOUT"] = "0.001"
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            speech._download_url_to_file(
                f"http://127.0.0.1:{listeners[1].getsockname()[1]}/model.tar.gz",
                self.cache,
            )
        self.assertLess(time.monotonic() - started, 5.0)

    def test_a_stalled_tls_handshake_or_connect_ends_at_the_total_deadline(self) -> None:
        """The deadline covers a TLS handshake and a connect, not only what follows them.

        A connection registered its socket with the deadline's watch
        once ``connect()`` had returned, which for HTTPS is after the
        handshake: a peer that accepted and then trickled its handshake
        kept the download (and the lock its caller holds) for a whole
        per-read timeout, whatever the deadline.  So did a connect nobody
        answers, once for every address of the host.
        """
        os.environ["KISS_VOICE_DOWNLOAD_TIMEOUT"] = "5"
        os.environ["KISS_VOICE_DOWNLOAD_TOTAL_TIMEOUT"] = "0.5"
        stop_sending = threading.Event()
        self.addCleanup(stop_sending.set)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        # The header of a 16 KiB TLS record, then its bytes one by one: 10 s of handshake.
        threading.Thread(
            target=_trickle,
            args=(listener, b"", b"\x16\x03\x03\x40\x00" + b"x" * 200, stop_sending),
            daemon=True,
        ).start()
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            speech._download_url_to_file(
                f"https://127.0.0.1:{listener.getsockname()[1]}/model.tar.gz", self.cache,
            )
        self.assertLess(time.monotonic() - started, 3.0, "the handshake outlived the deadline")
        # A connect nobody answers: the listener's queue of one is full, so the
        # kernel drops what else arrives (Linux; another system may answer it).
        full = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(full.close)
        full.bind(("127.0.0.1", 0))
        full.listen(0)
        self.addCleanup(socket.create_connection(full.getsockname()).close)
        try:
            socket.create_connection(full.getsockname(), 0.3).close()
        except TimeoutError:
            pass
        else:
            self.skipTest("a full listen queue does not stall a connect here")
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            speech._download_url_to_file(
                f"http://127.0.0.1:{full.getsockname()[1]}/model.tar.gz", self.cache,
            )
        self.assertLess(time.monotonic() - started, 3.0, "the connect outlived the deadline")

    def test_total_deadline_env_rejects_nonfinite_values(self) -> None:
        """``inf`` / ``1e999`` must not disable the total deadline —
        the parser falls back to the finite default."""
        from kiss.server.speech import (
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
        self.assertEqual(got, self.cache)
        self.assertEqual(self.cache.read_bytes(), payload)
        # Cached: a second call returns without re-downloading.
        self.assertEqual(ws._ensure_voice_model(), self.cache)
        # Over TLS the watched connection does the handshake itself (it registers
        # the socket first): with the trust and the name the stock one checks.
        certfile, keyfile, ca_file = make_test_tls(Path(self._tmp.name))
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(certfile, keyfile)
        https = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        self.addCleanup(https.server_close)
        https.socket = tls.wrap_socket(https.socket, server_side=True)
        threading.Thread(target=https.serve_forever, daemon=True).start()
        self.addCleanup(https.shutdown)
        url = f"https://127.0.0.1:{https.server_address[1]}/model.tar.gz"
        fetched = Path(self._tmp.name) / "fetched"
        with self.assertRaises(urllib.error.URLError) as untrusted:
            speech._download_url_to_file(url, fetched)
        self.assertIsInstance(untrusted.exception.reason, ssl.SSLCertVerificationError)
        self.enterContext(pytest.MonkeyPatch.context()).setenv("SSL_CERT_FILE", str(ca_file))
        speech._download_url_to_file(url, fetched)
        self.assertEqual(fetched.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
