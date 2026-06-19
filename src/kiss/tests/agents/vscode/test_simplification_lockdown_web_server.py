# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Characterization (lockdown) tests for ``web_server.py`` simplifications.

Pins the CURRENT externally-observable behavior of
:mod:`kiss.agents.vscode.web_server` so planned simplifications (see
``tmp/findings-5.md``, sections A2/A3/A6, B3/B5, F1-F3 and the listed
coverage gaps) cannot silently change it:

* HTTP endpoint matrix served by ``_process_request``: the substituted
  chat page (no leftover ``{{...}}`` placeholders, auth modal, main.js
  script tag), ``/media/`` static serving with MIME type, rejection of
  path-traversal attempts, 404 for unknown paths, and the raw
  ``HEAD`` 200/empty-body health-check reply from
  ``_HeadAwareServerConnection``.
* Silent dropping of VS Code-only webview commands (``pickFolder``,
  ``sizeReport``) on a live server connection, contrasted with the
  ``Unknown command`` error broadcast for genuinely unknown commands.
* ``_translate_webview_command``: pass-through of normal commands and
  the legacy ``resumeSession`` ``id`` -> ``chatId`` rename.
* ``_version_tuple`` / ``_compare_versions`` ordering semantics (guards
  the planned collapse into a single ``_is_newer`` helper).
* ``_get_local_ips`` returning only plain routable IPv4 strings.
* The ``remote-url.json`` write/read/remove lifecycle helpers.

All tests are hermetic: no cloudflared, no tunnels, no external network.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shutil
import socket
import ssl
import tempfile
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from kiss.agents.vscode.web_server import (
    MEDIA_DIR,
    RemoteAccessServer,
    _compare_versions,
    _generate_self_signed_cert,
    _get_local_ips,
    _read_url_from_file,
    _remove_url_file,
    _save_url_file,
    _translate_webview_command,
    _version_tuple,
)

# Matches an unsubstituted chat.html template placeholder such as
# ``{{MAIN_SRC}}``.  Used to prove _build_html substituted everything.
_PLACEHOLDER_RE = re.compile(r"\{\{[A-Z_]+\}\}")

_IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _find_free_port() -> int:
    """Return a free TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    """Return a TLS client context that accepts the self-signed cert."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _parse_http_response(raw: bytes) -> tuple[int, dict[str, str], bytes]:
    """Split a raw HTTP/1.1 response into (status, headers, body).

    Args:
        raw: The complete response bytes as read until EOF.

    Returns:
        Tuple of status code, lower-cased header dict, and body bytes.
    """
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split(" ")[1])
    headers: dict[str, str] = {}
    for line in lines[1:]:
        key, _, value = line.partition(":")
        headers[key.strip().lower()] = value.strip()
    return status, headers, body


class _ServerTestBase(IsolatedAsyncioTestCase):
    """Start a real RemoteAccessServer on a private port for each test."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss_lockdown_")
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = _find_free_port()
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=self.tmpdir,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _raw_request(self, payload: bytes) -> bytes:
        """Send raw HTTPS bytes to the server and read until EOF."""
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", self.port, ssl=_no_verify_ssl(),
            limit=16 * 1024 * 1024,
        )
        try:
            writer.write(payload)
            await writer.drain()
            return await asyncio.wait_for(reader.read(), timeout=10)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _http_get(self, path: str) -> tuple[int, dict[str, str], bytes]:
        """GET *path* over raw HTTPS (no client-side path normalization)."""
        payload = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{self.port}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        return _parse_http_response(await self._raw_request(payload))


class TestHttpEndpointMatrix(_ServerTestBase):
    """Lock down the HTTP responses produced by ``_process_request``."""

    async def test_chat_page_html_fully_substituted(self) -> None:
        """GET / serves chat.html with every {{...}} placeholder substituted."""
        status, headers, body = await self._http_get("/")
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "text/html; charset=utf-8")
        html = body.decode("utf-8")
        self.assertIsNone(
            _PLACEHOLDER_RE.search(html),
            "served chat page contains unsubstituted template placeholders",
        )
        self.assertIn('id="auth-modal"', html)
        self.assertIn('"/media/main.js"', html)
        self.assertIn('class="remote-chat"', html)

    async def test_media_file_served_with_mime_type(self) -> None:
        """GET /media/main.css returns the exact file bytes with a CSS MIME."""
        expected = (MEDIA_DIR / "main.css").read_bytes()
        status, headers, body = await self._http_get("/media/main.css")
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "text/css")
        self.assertEqual(headers["content-length"], str(len(expected)))
        self.assertEqual(body, expected)

    async def test_media_path_traversal_rejected(self) -> None:
        """GET /media/../web_server.py must not escape the media dir."""
        status, headers, body = await self._http_get("/media/../web_server.py")
        self.assertEqual(status, 404)
        self.assertEqual(body, b"Not Found")

    async def test_media_percent_encoded_traversal_rejected(self) -> None:
        """A percent-encoded traversal (%2e%2e) is also rejected."""
        status, _, body = await self._http_get("/media/%2e%2e/web_server.py")
        self.assertEqual(status, 404)
        self.assertEqual(body, b"Not Found")

    async def test_unknown_path_returns_404(self) -> None:
        """An unknown path returns 404 text/plain "Not Found"."""
        status, headers, body = await self._http_get("/definitely/not/here")
        self.assertEqual(status, 404)
        self.assertEqual(headers["content-type"], "text/plain")
        self.assertEqual(body, b"Not Found")

    async def test_head_request_returns_200_empty_body(self) -> None:
        """HEAD / gets the raw 200 empty-body health-check reply."""
        payload = (
            f"HEAD / HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        status, headers, body = _parse_http_response(
            await self._raw_request(payload),
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-length"], "0")
        self.assertEqual(body, b"")


class TestVscodeOnlyCommandsDropped(_ServerTestBase):
    """VS Code-only webview commands must be silently dropped."""

    async def test_vscode_only_commands_dropped_unknown_command_errors(
        self,
    ) -> None:
        """``pickFolder``/``sizeReport`` produce no error; an unknown
        command produces exactly the ``Unknown command`` error broadcast.

        Commands on one connection are dispatched strictly in order, so
        any (erroneous) broadcast caused by the VS Code-only commands
        would arrive before the unknown-command error sentinel.
        """
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path), limit=16 * 1024 * 1024,
        )
        try:
            for cmd in (
                {"type": "pickFolder"},
                {"type": "sizeReport", "width": 100, "height": 100},
                {"type": "unknownCmdLockdownXyz"},
            ):
                writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
            await writer.drain()
            seen: list[dict[str, object]] = []
            for _ in range(20):
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                self.assertTrue(line, "UDS closed before the error sentinel")
                msg = json.loads(line.decode("utf-8"))
                seen.append(msg)
                if msg.get("type") == "error":
                    break
            errors = [m for m in seen if m.get("type") == "error"]
            self.assertEqual(
                len(errors), 1, f"expected exactly one error event, got {seen}",
            )
            self.assertEqual(
                errors[0].get("text"), "Unknown command: unknownCmdLockdownXyz",
            )
            for msg in seen:
                text = str(msg.get("text", ""))
                self.assertNotIn("pickFolder", text)
                self.assertNotIn("sizeReport", text)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


class TestWebviewCommandTranslation(unittest.TestCase):
    """Lock down ``_translate_webview_command`` translations."""

    def test_normal_command_passes_through_unchanged(self) -> None:
        """A regular webview command is returned unchanged."""
        cmd = {"type": "userAnswer", "answer": "yes", "tabId": "t1"}
        self.assertEqual(
            _translate_webview_command(dict(cmd)),
            cmd,
        )

    def test_resume_session_id_renamed_to_chat_id(self) -> None:
        """Legacy ``resumeSession`` ``id`` is renamed to ``chatId``."""
        out = _translate_webview_command(
            {"type": "resumeSession", "id": "abc123", "tabId": "t2"},
        )
        self.assertEqual(
            out,
            {"type": "resumeSession", "chatId": "abc123", "tabId": "t2"},
        )
        self.assertNotIn("id", out)

    def test_resume_session_with_chat_id_untouched(self) -> None:
        """``resumeSession`` already carrying ``chatId`` is not modified."""
        cmd = {"type": "resumeSession", "id": "old", "chatId": "new"}
        self.assertEqual(_translate_webview_command(dict(cmd)), cmd)


class TestVersionComparison(unittest.TestCase):
    """Lock down ``_version_tuple`` / ``_compare_versions`` semantics."""

    def test_newer_version_compares_greater(self) -> None:
        """a > b returns a positive result."""
        self.assertGreater(_compare_versions("2026.6.14", "2026.6.13"), 0)
        self.assertGreater(_compare_versions("2027.1.0", "2026.12.31"), 0)

    def test_older_version_compares_less(self) -> None:
        """a < b returns a negative result."""
        self.assertLess(_compare_versions("2026.6.12", "2026.6.13"), 0)

    def test_equal_versions_compare_zero(self) -> None:
        """Identical versions compare equal."""
        self.assertEqual(_compare_versions("2026.6.13", "2026.6.13"), 0)

    def test_different_segment_counts_zero_padded(self) -> None:
        """Shorter tuples are right-padded with zeros before comparing."""
        self.assertEqual(_compare_versions("2026.6", "2026.6.0"), 0)
        self.assertGreater(_compare_versions("2026.6.1", "2026.6"), 0)
        self.assertLess(_compare_versions("2026.6", "2026.6.1"), 0)
        self.assertGreater(_compare_versions("2026.10", "2026.9.5"), 0)

    def test_unparseable_versions_compare_zero(self) -> None:
        """Unparseable input on either side compares equal (no update)."""
        self.assertEqual(_compare_versions("garbage", "2026.6.13"), 0)
        self.assertEqual(_compare_versions("2026.6.13", ""), 0)
        self.assertEqual(_compare_versions("", ""), 0)

    def test_version_tuple_parsing(self) -> None:
        """``_version_tuple`` parses CalVer strings and rejects garbage."""
        self.assertEqual(_version_tuple("2026.6.13"), (2026, 6, 13))
        self.assertEqual(_version_tuple(" 2026.6 "), (2026, 6))
        self.assertEqual(_version_tuple("2026..6"), (2026, 6))
        self.assertIsNone(_version_tuple("1.2.x"))
        self.assertIsNone(_version_tuple(""))
        self.assertIsNone(_version_tuple("."))


class TestGetLocalIps(unittest.TestCase):
    """Lock down the ``_get_local_ips`` output contract."""

    def test_returns_only_routable_ipv4_strings(self) -> None:
        """Every returned address is a plain dotted-quad IPv4 string,
        never an IPv6-mapped (``::ffff:``), loopback, or link-local one.
        """
        ips = _get_local_ips()
        self.assertIsInstance(ips, frozenset)
        for addr in ips:
            self.assertIsInstance(addr, str)
            self.assertRegex(addr, _IPV4_RE)
            self.assertFalse(addr.startswith("::ffff:"))
            self.assertFalse(addr.startswith("127."))
            self.assertFalse(addr.startswith("169.254."))


class TestUrlFileLifecycle(unittest.TestCase):
    """Lock down the remote-url.json write/read/remove helpers."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss_urlfile_"))
        self.url_file = self.tmpdir / "remote-url.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_then_read_roundtrip_local_only(self) -> None:
        """Without a tunnel URL, the local URL is stored and read back."""
        _save_url_file(self.url_file, "https://localhost:8787")
        data = json.loads(self.url_file.read_text())
        self.assertEqual(data, {"local": "https://localhost:8787"})
        self.assertEqual(
            _read_url_from_file(self.url_file), "https://localhost:8787",
        )

    def test_tunnel_url_preferred_over_local(self) -> None:
        """When a tunnel URL is stored, reading returns it over local."""
        _save_url_file(
            self.url_file,
            "https://localhost:8787",
            "https://example.trycloudflare.com",
        )
        data = json.loads(self.url_file.read_text())
        self.assertEqual(data["local"], "https://localhost:8787")
        self.assertEqual(data["tunnel"], "https://example.trycloudflare.com")
        self.assertEqual(
            _read_url_from_file(self.url_file),
            "https://example.trycloudflare.com",
        )

    def test_remove_clears_file_and_read_returns_none(self) -> None:
        """``_remove_url_file`` deletes the file; reads then return None."""
        _save_url_file(self.url_file, "https://localhost:8787")
        self.assertTrue(self.url_file.is_file())
        _remove_url_file(self.url_file)
        self.assertFalse(self.url_file.exists())
        self.assertIsNone(_read_url_from_file(self.url_file))
        # Removing an already-missing file is a silent no-op.
        _remove_url_file(self.url_file)


if __name__ == "__main__":
    unittest.main()
