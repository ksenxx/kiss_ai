# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for round-2 bugs in signal_agent, sms_agent and nostr_agent.

Covers:
- signal_agent.py: ``poll_messages`` must filter by ``channel_id`` and respect
  ``limit``; ``send_message`` must raise ``RuntimeError`` on CLI failure. Tested
  end-to-end against a REAL executable ``signal-cli`` shell script placed on PATH
  (no mock libraries).
- sms_agent.py: ``from_number`` is a required config key (a config without it is
  invalid and ``connect()`` reports "No Twilio config found."); ``is_from_bot``
  keys on the bot's number. Runtime Twilio API behavior is skipif-guarded because
  the twilio package is optional.
- nostr_agent.py: ``is_from_bot`` must key on the message contract key ``user``
  (with a ``pubkey`` fallback); the pynostr publish flow is skipif-guarded.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from kiss.agents.third_party_agents.signal_agent import _config as _signal_config
from kiss.agents.third_party_agents.sms_agent import _config as _sms_config

_SIGNAL_CONFIG = _signal_config.path
_SIGNAL_BACKUP = _SIGNAL_CONFIG.with_suffix(".json.bughunt2-bak")
_SMS_CONFIG = _sms_config.path
_SMS_BACKUP = _SMS_CONFIG.with_suffix(".json.bughunt2-bak")

# A real shell program emulating signal-cli. For "receive" it prints JSON lines
# shaped like signal-cli output for two different senders (three messages total).
# For "send" to the magic recipient +FAIL it exits 1 with an error on stderr.
_FAKE_SIGNAL_CLI = """#!/bin/sh
if [ "$1" = "-u" ]; then shift 2; fi
cmd="$1"
shift
if [ "$cmd" = "receive" ]; then
cat <<'EOF'
{"envelope": {"source": "+1AAA", "timestamp": 111, "dataMessage": {"message": "hello A1"}}}
{"envelope": {"source": "+1BBB", "timestamp": 112, "dataMessage": {"message": "hello B"}}}
{"envelope": {"source": "+1AAA", "timestamp": 113, "dataMessage": {"message": "hello A2"}}}
EOF
exit 0
fi
if [ "$cmd" = "send" ]; then
  last=""
  for arg in "$@"; do last="$arg"; done
  if [ "$last" = "+FAIL" ]; then
    echo "Failed to send message: ERROR unregistered recipient" >&2
    exit 1
  fi
  exit 0
fi
exit 0
"""


def _backup_config(config: Path, backup: Path) -> None:
    """Move an existing config file aside so tests can install their own."""
    backup.unlink(missing_ok=True)
    if config.exists():
        shutil.move(str(config), str(backup))


def _restore_config(config: Path, backup: Path) -> None:
    """Restore the original config file (or remove the test one)."""
    config.unlink(missing_ok=True)
    if backup.exists():
        shutil.move(str(backup), str(config))


class TestSignalBackend(unittest.TestCase):
    """End-to-end tests driving SignalChannelBackend through a real fake signal-cli."""

    def setUp(self) -> None:
        """Install a real executable signal-cli script on PATH and a test config."""
        self._tmpdir = tempfile.mkdtemp(prefix="bughunt-signal-")
        cli = Path(self._tmpdir) / "signal-cli"
        cli.write_text(_FAKE_SIGNAL_CLI, encoding="utf-8")
        cli.chmod(0o755)
        self._old_path = os.environ["PATH"]
        os.environ["PATH"] = self._tmpdir + os.pathsep + self._old_path
        _backup_config(_SIGNAL_CONFIG, _SIGNAL_BACKUP)
        from kiss.agents.third_party_agents.signal_agent import (
            SignalChannelBackend,
            _config,
        )

        _config.save({"phone_number": "+1BOT"})
        self._backend = SignalChannelBackend()
        self.assertTrue(self._backend.connect())

    def tearDown(self) -> None:
        """Restore PATH and the original signal config."""
        os.environ["PATH"] = self._old_path
        _restore_config(_SIGNAL_CONFIG, _SIGNAL_BACKUP)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_poll_filters_by_channel_id(self) -> None:
        """Only messages whose sender equals channel_id are returned."""
        messages, _ = self._backend.poll_messages("+1AAA", "", limit=10)
        self.assertEqual(len(messages), 2)
        self.assertEqual({m["user"] for m in messages}, {"+1AAA"})
        self.assertEqual([m["text"] for m in messages], ["hello A1", "hello A2"])

    def test_poll_respects_limit(self) -> None:
        """poll_messages('+1AAA', '', limit=1) returns exactly one +1AAA message."""
        messages, _ = self._backend.poll_messages("+1AAA", "", limit=1)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["user"], "+1AAA")

    def test_poll_empty_channel_returns_all_senders(self) -> None:
        """An empty channel_id keeps messages from every sender."""
        messages, _ = self._backend.poll_messages("", "", limit=10)
        self.assertEqual(len(messages), 3)
        self.assertEqual({m["user"] for m in messages}, {"+1AAA", "+1BBB"})

    def test_send_message_failure_raises(self) -> None:
        """A CLI failure (nonzero exit + stderr) must raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            self._backend.send_message("+FAIL", "hi")

    def test_send_message_success_does_not_raise(self) -> None:
        """A successful send returns without raising."""
        self._backend.send_message("+1AAA", "hi")

    def test_receive_messages_with_large_timeout(self) -> None:
        """receive_messages(timeout=60) must not be killed by a 30s subprocess cap."""
        result = json.loads(self._backend.receive_messages(timeout=60))
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["messages"]), 3)


class TestSMSBackend(unittest.TestCase):
    """Tests for SMSChannelBackend config contract and bot detection."""

    def setUp(self) -> None:
        """Back up any existing SMS config."""
        _backup_config(_SMS_CONFIG, _SMS_BACKUP)

    def tearDown(self) -> None:
        """Restore the original SMS config."""
        _restore_config(_SMS_CONFIG, _SMS_BACKUP)

    def test_config_without_from_number_is_invalid(self) -> None:
        """A legacy config lacking from_number must be rejected by connect()."""
        from kiss.agents.third_party_agents.sms_agent import SMSChannelBackend, _config

        _config.save({"account_sid": "AC1", "auth_token": "tok"})
        backend = SMSChannelBackend()
        self.assertFalse(backend.connect())
        self.assertEqual(backend._connection_info, "No Twilio config found.")
        self.assertIsNone(_config.load())

    def test_is_from_bot_keys_on_from_number(self) -> None:
        """is_from_bot compares the message user against the bot's number."""
        from kiss.agents.third_party_agents.sms_agent import SMSChannelBackend

        backend = SMSChannelBackend()
        backend._from_number = "+1BOT"
        self.assertTrue(backend.is_from_bot({"user": "+1BOT"}))
        self.assertFalse(backend.is_from_bot({"user": "+1AAA"}))

    @unittest.skipIf(find_spec("twilio") is None, "twilio not installed")
    def test_poll_messages_with_bad_credentials_returns_empty(self) -> None:
        """poll_messages must swallow API failures and return ([], oldest)."""
        import importlib

        from kiss.agents.third_party_agents.sms_agent import SMSChannelBackend

        twilio_rest = importlib.import_module("twilio.rest")
        backend = SMSChannelBackend()
        backend._client = twilio_rest.Client("AC" + "0" * 32, "invalid-token")
        backend._from_number = "+1BOT"
        messages, cursor = backend.poll_messages("+1AAA", "123.0", limit=5)
        self.assertEqual(messages, [])
        self.assertEqual(cursor, "123.0")


class TestNostrBackend(unittest.TestCase):
    """Tests for NostrChannelBackend bot detection and publish flow."""

    def test_is_from_bot_checks_user_key(self) -> None:
        """is_from_bot must key on the contract key 'user'."""
        from kiss.agents.third_party_agents.nostr_agent import NostrChannelBackend

        backend = NostrChannelBackend()
        backend._public_key = "botpub123"
        self.assertTrue(backend.is_from_bot({"user": "botpub123"}))
        self.assertFalse(backend.is_from_bot({"user": "someoneelse"}))
        self.assertFalse(backend.is_from_bot({"user": "other", "pubkey": "botpub123"}))

    def test_is_from_bot_falls_back_to_pubkey(self) -> None:
        """Messages without 'user' fall back to the legacy 'pubkey' key."""
        from kiss.agents.third_party_agents.nostr_agent import NostrChannelBackend

        backend = NostrChannelBackend()
        backend._public_key = "botpub123"
        self.assertTrue(backend.is_from_bot({"pubkey": "botpub123"}))
        self.assertFalse(backend.is_from_bot({"pubkey": "someoneelse"}))

    def test_is_from_bot_without_key_is_false(self) -> None:
        """An unauthenticated backend never claims a message as its own."""
        from kiss.agents.third_party_agents.nostr_agent import NostrChannelBackend

        backend = NostrChannelBackend()
        self.assertFalse(backend.is_from_bot({"user": ""}))
        self.assertFalse(backend.is_from_bot({}))

    @unittest.skipIf(find_spec("pynostr") is None, "pynostr not installed")
    def test_publish_note_uses_pynostr_relay_manager_flow(self) -> None:
        """publish_note must use the pynostr RelayManager API (no open_connections)."""
        import importlib

        from kiss.agents.third_party_agents.nostr_agent import NostrChannelBackend

        pynostr_key = importlib.import_module("pynostr.key")
        backend = NostrChannelBackend()
        backend._private_key = pynostr_key.PrivateKey()
        backend._public_key = backend._private_key.public_key.hex()
        backend._relays = []
        result = json.loads(backend.publish_note("bughunt test note"))
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["event_id"])


if __name__ == "__main__":
    unittest.main()
