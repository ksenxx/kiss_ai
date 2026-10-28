# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for misc third-party-agent bugs.

Covers:
- govee.py: importing the module must not sys.exit when GOVEE_API_KEY is unset.
- imessage_sea.py: AppleScript source built by the real script-builder functions
  must escape backslashes and double quotes in interpolated values.
- phone_sea.py: poll_messages must persist its cursor to _last_msg_id so
  the ``oldest or self._last_msg_id`` fallback advances across calls (verified
  against a real local HTTP server that records request params).
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import pytest

from kiss.agents.third_party_agents.imessage.imessage_sea import (
    _build_send_attachment_script,
    _build_send_message_script,
)
from kiss.agents.third_party_agents.phone.phone_sea import _config as _phone_config
from kiss.tests.agents.third_party_agents.recording_http import (
    PhoneApiHandler,
    RecordingServer,
    recording_server,
)


class TestGoveeImportSafe(unittest.TestCase):
    """govee.py must be importable without GOVEE_API_KEY set."""

    def test_import_does_not_exit_but_main_still_errors(self) -> None:
        """One interpreter: the import survives (sentinel printed), ``main`` then exits.

        A ``sys.exit`` at import time would never reach the sentinel, so
        the stdout check pins "importing does not exit" and the exit code
        plus stderr pin "running a command without the key still fails".
        """
        env = {k: v for k, v in os.environ.items() if k != "GOVEE_API_KEY"}
        code = (
            "from kiss.agents.third_party_agents import govee; print('imported', flush=True); "
            "govee.main(['govee.py', 'list'])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(result.stdout.strip(), "imported", result.stderr.strip())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GOVEE_API_KEY", result.stderr)


class TestIMessageAppleScriptEscaping(unittest.TestCase):
    """The real AppleScript builder functions must escape interpolated values."""

    def test_send_message_script_escapes_double_quotes(self) -> None:
        """A message like say "hi" must appear escaped in the generated script."""
        script = _build_send_message_script("+14155238886", 'say "hi"')
        self.assertIn('send "say \\"hi\\"" to targetBuddy', script)
        self.assertNotIn('send "say "hi"" to targetBuddy', script)

    def test_send_message_script_escapes_backslashes(self) -> None:
        """Backslashes in the text must be doubled before quote escaping."""
        script = _build_send_message_script("+14155238886", 'C:\\path "x"')
        self.assertIn('send "C:\\\\path \\"x\\"" to targetBuddy', script)

    def test_send_message_script_escapes_recipient(self) -> None:
        """The recipient is also interpolated and must be escaped."""
        script = _build_send_message_script('evil" & quit -- ', "hello")
        self.assertIn('buddy "evil\\" & quit -- " of targetService', script)

    def test_send_attachment_script_escapes_file_path(self) -> None:
        """File paths with quotes/backslashes must be escaped in the attachment script."""
        script = _build_send_attachment_script("+14155238886", '/tmp/a "b"\\c.txt')
        self.assertIn('send POSIX file "/tmp/a \\"b\\"\\\\c.txt" to targetBuddy', script)

    def test_invalid_service_rejected(self) -> None:
        """A service value outside iMessage/SMS must be rejected, not interpolated."""
        with self.assertRaises(ValueError):
            _build_send_message_script("+14155238886", "hi", service='x" & quit')


class _PhoneHandler(PhoneApiHandler):
    messages = [
        {"timestamp": 111, "from": "+15550001", "body": "hi", "id": "1"},
        {"timestamp": 222, "from": "+15550001", "body": "there", "id": "2"},
    ]


def _since_cursors(server: RecordingServer) -> list[str]:
    """The ``since`` query value of every ``/api/sms/messages`` poll, in order."""
    polls = [r for r in server.requests if r["path"] == "/api/sms/messages"]
    return [r["query"].get("since", [""])[0] for r in polls]


@pytest.mark.usefixtures("isolated_kiss_home")
class TestPhoneControlPollCursor(unittest.TestCase):
    """poll_messages must advance and reuse its stored cursor across calls."""

    def setUp(self) -> None:
        """Start a real local HTTP server and point the phone config at it."""
        self._server = self.enterContext(recording_server(_PhoneHandler))
        port = self._server.server_address[1]
        _phone_config.save({"device_ip": "127.0.0.1", "device_port": str(port)})

    def test_second_poll_sends_advanced_cursor(self) -> None:
        """After a poll returns messages, a later poll with oldest='' must reuse the cursor."""
        from kiss.agents.third_party_agents.phone.phone_sea import (
            PhoneControlChannelBackend,
        )

        backend = PhoneControlChannelBackend()
        self.assertTrue(backend.connect())

        messages, new_oldest = backend.poll_messages("+15550001", "")
        self.assertEqual(len(messages), 2)
        self.assertEqual(new_oldest, "222")
        self.assertEqual(_since_cursors(self._server), [""])

        messages2, _ = backend.poll_messages("+15550001", "")
        self.assertEqual(len(messages2), 2)
        self.assertEqual(
            _since_cursors(self._server),
            ["", "222"],
            "second poll with oldest='' must fall back to the stored _last_msg_id cursor",
        )


if __name__ == "__main__":
    unittest.main()
