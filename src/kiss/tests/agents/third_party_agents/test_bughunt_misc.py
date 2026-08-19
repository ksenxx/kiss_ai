# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for misc third-party-agent bugs.

Covers:
- govee.py: importing the module must not sys.exit when GOVEE_API_KEY is unset.
- imessage_agent.py: AppleScript source built by the real script-builder functions
  must escape backslashes and double quotes in interpolated values.
- phone_control_agent.py: poll_messages must persist its cursor to _last_msg_id so
  the ``oldest or self._last_msg_id`` fallback advances across calls (verified
  against a real local HTTP server that records request params).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from kiss.agents.third_party_agents.phone_control_agent import _config as _phone_config

_PHONE_CONFIG = _phone_config.path
_PHONE_CONFIG_BACKUP = _PHONE_CONFIG.with_suffix(".json.bughunt-bak")


class TestGoveeImportSafe(unittest.TestCase):
    """govee.py must be importable without GOVEE_API_KEY set."""

    def test_import_without_api_key_does_not_exit(self) -> None:
        """Importing the module with GOVEE_API_KEY unset must not kill the process."""
        env = {k: v for k, v in os.environ.items() if k != "GOVEE_API_KEY"}
        result = subprocess.run(
            [sys.executable, "-c", "import kiss.agents.third_party_agents.govee"],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"import exited with {result.returncode}: {result.stderr.strip()}",
        )

    def test_main_without_api_key_still_errors(self) -> None:
        """Running the CLI with a command but no key must still exit with the error."""
        env = {k: v for k, v in os.environ.items() if k != "GOVEE_API_KEY"}
        code = (
            "from kiss.agents.third_party_agents import govee; "
            "govee.main(['govee.py', 'list'])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GOVEE_API_KEY", result.stderr)


class TestIMessageAppleScriptEscaping(unittest.TestCase):
    """The real AppleScript builder functions must escape interpolated values."""

    def test_send_message_script_escapes_double_quotes(self) -> None:
        """A message like say "hi" must appear escaped in the generated script."""
        from kiss.agents.third_party_agents.imessage_agent import _build_send_message_script

        script = _build_send_message_script("+14155238886", 'say "hi"')
        self.assertIn('send "say \\"hi\\"" to targetBuddy', script)
        self.assertNotIn('send "say "hi"" to targetBuddy', script)

    def test_send_message_script_escapes_backslashes(self) -> None:
        """Backslashes in the text must be doubled before quote escaping."""
        from kiss.agents.third_party_agents.imessage_agent import _build_send_message_script

        script = _build_send_message_script("+14155238886", 'C:\\path "x"')
        self.assertIn('send "C:\\\\path \\"x\\"" to targetBuddy', script)

    def test_send_message_script_escapes_recipient(self) -> None:
        """The recipient is also interpolated and must be escaped."""
        from kiss.agents.third_party_agents.imessage_agent import _build_send_message_script

        script = _build_send_message_script('evil" & quit -- ', "hello")
        self.assertIn('buddy "evil\\" & quit -- " of targetService', script)

    def test_send_attachment_script_escapes_file_path(self) -> None:
        """File paths with quotes/backslashes must be escaped in the attachment script."""
        from kiss.agents.third_party_agents.imessage_agent import (
            _build_send_attachment_script,
        )

        script = _build_send_attachment_script("+14155238886", '/tmp/a "b"\\c.txt')
        self.assertIn('send POSIX file "/tmp/a \\"b\\"\\\\c.txt" to targetBuddy', script)

    def test_invalid_service_rejected(self) -> None:
        """A service value outside iMessage/SMS must be rejected, not interpolated."""
        from kiss.agents.third_party_agents.imessage_agent import _build_send_message_script

        with self.assertRaises(ValueError):
            _build_send_message_script("+14155238886", "hi", service='x" & quit')


class _PhoneApiHandler(BaseHTTPRequestHandler):
    """Real HTTP handler emulating the phone companion REST app."""

    sms_requests: list[dict[str, list[str]]] = []

    def do_GET(self) -> None:  # noqa: N802
        """Serve /api/device/info and /api/sms/messages with canned JSON."""
        parsed = urlparse(self.path)
        if parsed.path == "/api/device/info":
            body = json.dumps({"device_name": "testphone"})
        elif parsed.path == "/api/sms/messages":
            type(self).sms_requests.append(parse_qs(parsed.query, keep_blank_values=True))
            body = json.dumps(
                {
                    "messages": [
                        {"timestamp": 111, "from": "+15550001", "body": "hi", "id": "1"},
                        {"timestamp": 222, "from": "+15550001", "body": "there", "id": "2"},
                    ]
                }
            )
        else:
            body = json.dumps({})
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        """Silence request logging."""


class TestPhoneControlPollCursor(unittest.TestCase):
    """poll_messages must advance and reuse its stored cursor across calls."""

    def setUp(self) -> None:
        """Start a real local HTTP server and point the phone config at it."""
        _PhoneApiHandler.sms_requests = []
        self._server = HTTPServer(("127.0.0.1", 0), _PhoneApiHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._had_config = _PHONE_CONFIG.exists()
        if self._had_config:
            shutil.copy2(_PHONE_CONFIG, _PHONE_CONFIG_BACKUP)
        _PHONE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        port = self._server.server_address[1]
        _PHONE_CONFIG.write_text(
            json.dumps({"device_ip": "127.0.0.1", "device_port": str(port)})
        )

    def tearDown(self) -> None:
        """Stop the server and restore the original phone config."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        if self._had_config:
            shutil.move(_PHONE_CONFIG_BACKUP, _PHONE_CONFIG)
        else:
            _PHONE_CONFIG.unlink(missing_ok=True)

    def test_second_poll_sends_advanced_cursor(self) -> None:
        """After a poll returns messages, a later poll with oldest='' must reuse the cursor."""
        from kiss.agents.third_party_agents.phone_control_agent import (
            PhoneControlChannelBackend,
        )

        backend = PhoneControlChannelBackend()
        self.assertTrue(backend.connect())

        messages, new_oldest = backend.poll_messages("+15550001", "")
        self.assertEqual(len(messages), 2)
        self.assertEqual(new_oldest, "222")
        self.assertEqual(_PhoneApiHandler.sms_requests[0].get("since", [""])[0], "")

        messages2, _ = backend.poll_messages("+15550001", "")
        self.assertEqual(len(messages2), 2)
        self.assertEqual(
            _PhoneApiHandler.sms_requests[1].get("since", [""])[0],
            "222",
            "second poll with oldest='' must fall back to the stored _last_msg_id cursor",
        )


if __name__ == "__main__":
    unittest.main()
