# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for signal_agent's deduplicated send path (G-R3).

``send_signal_message`` used to duplicate ``send_message``'s CLI
invocation and error heuristic verbatim; it now wraps ``send_message``
and JSON-encodes the outcome.  Tested against a REAL executable
``signal-cli`` stand-in program placed on PATH — a real subprocess, not
a mock.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from kiss.agents.third_party_agents.signal_agent import SignalChannelBackend
from kiss.tests.conftest import install_cli_script

# A Python program (not a shell script) so the same stand-in runs on
# Windows, where ``install_cli_script`` adds the ``.cmd`` shim.
_FAKE_SIGNAL_CLI = """#!/usr/bin/env python3
import sys
args = sys.argv[1:]
if args[:1] == ["-u"]:
    args = args[2:]
if args[:1] == ["send"]:
    if args[-1] == "+FAIL":
        print("Failed to send message: ERROR unregistered recipient", file=sys.stderr)
        sys.exit(1)
    if args[-1] == "+WARN":
        print("WARNING: error while updating profile", file=sys.stderr)
sys.exit(0)
"""


class TestSendSignalMessage(unittest.TestCase):
    """send_signal_message delegates to send_message and reports JSON."""

    def setUp(self) -> None:
        """Install a real executable signal-cli stub on PATH."""
        self._tmpdir = tempfile.mkdtemp(prefix="rr-area-g-signal-")
        install_cli_script(Path(self._tmpdir) / "signal-cli", _FAKE_SIGNAL_CLI)
        self._old_path = os.environ["PATH"]
        os.environ["PATH"] = self._tmpdir + os.pathsep + self._old_path
        self._backend = SignalChannelBackend()
        self._backend._phone_number = "+1BOT"

    def tearDown(self) -> None:
        """Restore PATH and remove the stub."""
        os.environ["PATH"] = self._old_path
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_success_returns_ok(self) -> None:
        """A successful CLI send yields {"ok": true}."""
        result = json.loads(self._backend.send_signal_message("+1AAA", "hi"))
        self.assertEqual(result, {"ok": True})

    def test_cli_failure_returns_error_json(self) -> None:
        """A CLI failure surfaces send_message's RuntimeError as JSON."""
        result = json.loads(self._backend.send_signal_message("+FAIL", "hi"))
        self.assertFalse(result["ok"])
        self.assertIn("unregistered recipient", result["error"])
        self.assertIn("signal-cli send failed", result["error"])

    def test_stderr_error_heuristic_shared_with_send_message(self) -> None:
        """A zero-exit send whose stderr mentions 'error' also fails.

        This heuristic used to be duplicated; both entry points now share
        the single implementation in ``send_message``.
        """
        result = json.loads(self._backend.send_signal_message("+WARN", "hi"))
        self.assertFalse(result["ok"])
        with self.assertRaises(RuntimeError):
            self._backend.send_message("+WARN", "hi")

    def test_missing_cli_returns_error_json(self) -> None:
        """A missing signal-cli binary is reported as an error, not raised."""
        self._backend._signal_cli = "/nonexistent/signal-cli"
        result = json.loads(self._backend.send_signal_message("+1AAA", "hi"))
        self.assertFalse(result["ok"])
        self.assertTrue(result["error"])
