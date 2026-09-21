# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for round-2 bugs in signal_sea, sms_sea and nostr_sea.

Covers:
- signal_sea.py: ``poll_messages`` performs a DESTRUCTIVE ``signal-cli
  receive`` (the server acks every envelope), and the channel runner
  replies to the configured ``channel_id`` — so with a ``channel_id`` only
  that sender's envelopes may be surfaced (anything else would leak replies
  to the wrong contact), while consumed envelopes that are not surfaced are
  parked in the channel state bound via ``bind_channel_state`` (matching
  overflow past ``limit`` plus foreign senders) or, without bound state,
  foreign envelopes are dropped with a log and ``limit`` is ignored;
  ``send_message`` must raise ``RuntimeError`` on CLI failure. Tested
  end-to-end against a REAL executable ``signal-cli`` stand-in program
  placed on PATH (no mock libraries).
- sms_sea.py: ``from_number`` is a required config key (a config without it is
  invalid and ``connect()`` reports "No Twilio config found."); ``is_from_bot``
  keys on the bot's number. Runtime Twilio API behavior is skipif-guarded because
  the twilio package is optional.
- nostr_sea.py: ``is_from_bot`` must key on the message contract key ``user``
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

from kiss.agents.third_party_agents.signal_sea import _config as _signal_config
from kiss.agents.third_party_agents.sms_sea import _config as _sms_config
from kiss.tests.conftest import install_cli_script

_SIGNAL_CONFIG = _signal_config.path
_SIGNAL_BACKUP = _SIGNAL_CONFIG.with_suffix(".json.bughunt2-bak")
_SMS_CONFIG = _sms_config.path
_SMS_BACKUP = _SMS_CONFIG.with_suffix(".json.bughunt2-bak")

# A Python program (not a shell script) so the same stand-in runs on
# Windows, where ``install_cli_script`` adds the ``.cmd`` shim.
_FAKE_SIGNAL_CLI = """#!/usr/bin/env python3
import json
import sys
args = sys.argv[1:]
if args[:1] == ["-u"]:
    args = args[2:]
cmd = args[0] if args else ""
if cmd == "receive":
    for source, ts, text in (
        ("+1AAA", 111, "hello A1"),
        ("+1BBB", 112, "hello B"),
        ("+1AAA", 113, "hello A2"),
    ):
        print(json.dumps({"envelope": {
            "source": source, "timestamp": ts, "dataMessage": {"message": text},
        }}))
    sys.exit(0)
if cmd == "send" and args[-1] == "+FAIL":
    print("Failed to send message: ERROR unregistered recipient", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
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
        """Install a real executable signal-cli script on PATH and a test config.

        Each global mutation registers its restoration with ``addCleanup``
        immediately, so state is restored even when ``setUp`` itself fails
        part-way (unittest skips ``tearDown`` in that case, but cleanups run).
        """
        self._tmpdir = tempfile.mkdtemp(prefix="bughunt-signal-")
        self.addCleanup(shutil.rmtree, self._tmpdir, ignore_errors=True)
        install_cli_script(Path(self._tmpdir) / "signal-cli", _FAKE_SIGNAL_CLI)
        self._old_path = os.environ["PATH"]
        os.environ["PATH"] = self._tmpdir + os.pathsep + self._old_path
        self.addCleanup(os.environ.__setitem__, "PATH", self._old_path)
        _backup_config(_SIGNAL_CONFIG, _SIGNAL_BACKUP)
        self.addCleanup(_restore_config, _SIGNAL_CONFIG, _SIGNAL_BACKUP)
        from kiss.agents.third_party_agents.signal_sea import (
            SignalChannelBackend,
            _config,
        )

        _config.save({"phone_number": "+1BOT"})
        self._backend = SignalChannelBackend()
        self.assertTrue(self._backend.connect())

    def test_poll_with_channel_returns_only_that_sender(self) -> None:
        """A configured channel_id surfaces only that sender's envelopes.

        The channel runner replies to the CONFIGURED contact, so
        surfacing another sender's envelope would leak the reply to the
        wrong contact.
        """
        messages, _ = self._backend.poll_messages("+1AAA", "", limit=10)
        self.assertEqual([m["text"] for m in messages], ["hello A1", "hello A2"])
        self.assertEqual({m["user"] for m in messages}, {"+1AAA"})

    def test_poll_without_state_ignores_limit_for_matching(self) -> None:
        """Without bound state, matching envelopes are never truncated.

        There is nowhere to park the overflow of an already-consumed
        destructive read, so truncating would lose it permanently.
        """
        messages, _ = self._backend.poll_messages("+1AAA", "", limit=1)
        self.assertEqual(len(messages), 2)

    def test_poll_with_state_parks_foreign_and_overflow(self) -> None:
        """Bound state receives the foreign sender and the overflow.

        With ``limit=1`` only the first matching envelope is delivered;
        the second matching one and the foreign sender's are parked in
        ``pending_envelopes`` so the destructive read loses nothing.
        """
        state: dict = {"pending_envelopes": []}
        self._backend.bind_channel_state(state)
        messages, _ = self._backend.poll_messages("+1AAA", "", limit=1)
        self.assertEqual([m["text"] for m in messages], ["hello A1"])
        parked = state["pending_envelopes"]
        self.assertEqual(
            [(e["user"], e["text"]) for e in parked],
            [("+1AAA", "hello A2"), ("+1BBB", "hello B")],
        )

    def test_poll_with_state_delivers_queued_matching_first(self) -> None:
        """Envelopes parked on an earlier tick are delivered before new ones."""
        state: dict = {
            "pending_envelopes": [
                {"ts": "100", "user": "+1AAA", "text": "parked A0"},
                {"ts": "101", "user": "+1CCC", "text": "parked C"},
            ]
        }
        self._backend.bind_channel_state(state)
        messages, _ = self._backend.poll_messages("+1AAA", "", limit=10)
        self.assertEqual(
            [m["text"] for m in messages], ["parked A0", "hello A1", "hello A2"]
        )
        self.assertEqual(
            [(e["user"], e["text"]) for e in state["pending_envelopes"]],
            [("+1CCC", "parked C"), ("+1BBB", "hello B")],
        )

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
        from kiss.agents.third_party_agents.sms_sea import SMSChannelBackend, _config

        _config.save({"account_sid": "AC1", "auth_token": "tok"})
        backend = SMSChannelBackend()
        self.assertFalse(backend.connect())
        self.assertEqual(backend._connection_info, "No Twilio config found.")
        self.assertIsNone(_config.load())

    def test_is_from_bot_keys_on_from_number(self) -> None:
        """is_from_bot compares the message user against the bot's number."""
        from kiss.agents.third_party_agents.sms_sea import SMSChannelBackend

        backend = SMSChannelBackend()
        backend._from_number = "+1BOT"
        self.assertTrue(backend.is_from_bot({"user": "+1BOT"}))
        self.assertFalse(backend.is_from_bot({"user": "+1AAA"}))

    @unittest.skipIf(find_spec("twilio") is None, "twilio not installed")
    def test_poll_messages_with_bad_credentials_returns_empty(self) -> None:
        """poll_messages must swallow API failures and return ([], oldest).

        Hermetic: a local HTTP server emulates Twilio's exact
        authentication-error response (HTTP 401, error code 20003), and
        the SDK is routed to it through its supported ``http_client``
        parameter — a subclass of the real ``TwilioHttpClient`` that
        re-hosts each request URL onto the emulator and performs a REAL
        HTTP round trip.  No DNS, egress, or api.twilio.com dependency;
        the finite transport timeout stays as defense in depth.
        """
        import importlib
        import json as json_module
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.parse import urlsplit, urlunsplit

        from kiss.agents.third_party_agents.sms_sea import SMSChannelBackend

        twilio_rest = importlib.import_module("twilio.rest")
        twilio_http = importlib.import_module("twilio.http.http_client")

        hits: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                """Answer every request with Twilio's 20003 auth error."""
                hits.append(self.path)
                body = json_module.dumps(
                    {
                        "code": 20003,
                        "detail": "Your AccountSid or AuthToken was incorrect.",
                        "message": "Authentication Error - invalid username",
                        "more_info": "https://www.twilio.com/docs/errors/20003",
                        "status": 401,
                    }
                ).encode("utf-8")
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                """Silence request logging."""

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        local_netloc = f"127.0.0.1:{server.server_port}"
        threading.Thread(target=server.serve_forever, daemon=True).start()

        class LocalTwilioHttpClient(twilio_http.TwilioHttpClient):  # type: ignore[misc, name-defined]
            """Real Twilio transport re-hosted onto the local emulator."""

            def request(  # noqa: PLR0913
                self,
                method: str,
                url: str,
                params: dict | None = None,
                data: dict | None = None,
                headers: dict | None = None,
                auth: tuple | None = None,
                timeout: float | None = None,
                allow_redirects: bool = False,
            ):
                """Swap the URL's host for the emulator and really send."""
                parts = urlsplit(url)
                local = urlunsplit(
                    ("http", local_netloc, parts.path, parts.query, parts.fragment)
                )
                return super().request(
                    method, local, params=params, data=data, headers=headers,
                    auth=auth, timeout=timeout, allow_redirects=allow_redirects,
                )

        try:
            backend = SMSChannelBackend()
            backend._client = twilio_rest.Client(
                "AC" + "0" * 32,
                "invalid-token",
                http_client=LocalTwilioHttpClient(timeout=10),
            )
            backend._from_number = "+1BOT"
            messages, cursor = backend.poll_messages("+1AAA", "123.0", limit=5)
            self.assertEqual(messages, [])
            self.assertEqual(cursor, "123.0")
            # The SDK really hit the local Twilio-shaped endpoint.
            self.assertTrue(hits, "the emulator never received the request")
            self.assertIn("/2010-04-01/Accounts/AC", hits[0])
            self.assertIn("/Messages.json", hits[0])
        finally:
            server.shutdown()
            server.server_close()


class TestNostrBackend(unittest.TestCase):
    """Tests for NostrChannelBackend bot detection and publish flow."""

    def test_is_from_bot_checks_user_key(self) -> None:
        """is_from_bot must key on the contract key 'user'."""
        from kiss.agents.third_party_agents.nostr_sea import NostrChannelBackend

        backend = NostrChannelBackend()
        backend._public_key = "botpub123"
        self.assertTrue(backend.is_from_bot({"user": "botpub123"}))
        self.assertFalse(backend.is_from_bot({"user": "someoneelse"}))
        self.assertFalse(backend.is_from_bot({"user": "other", "pubkey": "botpub123"}))

    def test_is_from_bot_falls_back_to_pubkey(self) -> None:
        """Messages without 'user' fall back to the legacy 'pubkey' key."""
        from kiss.agents.third_party_agents.nostr_sea import NostrChannelBackend

        backend = NostrChannelBackend()
        backend._public_key = "botpub123"
        self.assertTrue(backend.is_from_bot({"pubkey": "botpub123"}))
        self.assertFalse(backend.is_from_bot({"pubkey": "someoneelse"}))

    def test_is_from_bot_without_key_is_false(self) -> None:
        """An unauthenticated backend never claims a message as its own."""
        from kiss.agents.third_party_agents.nostr_sea import NostrChannelBackend

        backend = NostrChannelBackend()
        self.assertFalse(backend.is_from_bot({"user": ""}))
        self.assertFalse(backend.is_from_bot({}))

    @unittest.skipIf(find_spec("pynostr") is None, "pynostr not installed")
    def test_publish_note_uses_pynostr_relay_manager_flow(self) -> None:
        """publish_note must use the pynostr RelayManager API (no open_connections)."""
        import importlib

        from kiss.agents.third_party_agents.nostr_sea import NostrChannelBackend

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
