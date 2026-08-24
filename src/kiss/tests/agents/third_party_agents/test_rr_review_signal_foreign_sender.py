# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end runner tests for Signal's foreign-sender handling.

Review finding: ``poll_messages`` used to return ALL envelopes of the
destructive ``signal-cli receive`` while the ChannelRunner (with no
``--allow-users``) handled every sender and sent the reply to the
CONFIGURED contact — acting on unintended senders and leaking replies
across contacts.

These tests drive the REAL ``ChannelRunner.run_once`` tick (connect →
poll → allow-list → handle → reply) against a REAL executable
``signal-cli`` shell script whose ``receive`` is genuinely destructive
(it truncates its spool file) and whose ``send`` records every
recipient.  Only ``_launch_task`` — the daemon/LLM boundary, per the
suite convention in ``test_hermes_runner.py`` — is overridden with a
recorder so no LLM runs.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from kiss.agents.third_party_agents._channel_agent_utils import (
    ChannelRunner,
    load_channel_state,
)
from kiss.agents.third_party_agents.signal_agent import (
    SignalChannelBackend,
    _config,
)

_SPOOL_CLI = """#!/bin/sh
if [ "$1" = "-u" ]; then shift 2; fi
cmd="$1"
shift
if [ "$cmd" = "receive" ]; then
  cat "$KISS_TEST_SPOOL" 2>/dev/null
  : > "$KISS_TEST_SPOOL"
  exit 0
fi
if [ "$cmd" = "send" ]; then
  last=""
  for arg in "$@"; do last="$arg"; done
  echo "$last" >> "$KISS_TEST_SENDS"
  exit 0
fi
exit 0
"""


def _envelope(sender: str, ts: int, text: str) -> str:
    """Build one signal-cli JSON envelope line."""
    return json.dumps(
        {
            "envelope": {
                "source": sender,
                "timestamp": ts,
                "dataMessage": {"message": text},
            }
        }
    )


class RecordingLaunchRunner(ChannelRunner):
    """ChannelRunner whose daemon launch is a recorder (no LLM)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.launches: list[tuple[str, str]] = []

    def _launch_task(
        self, channel_id: str, thread_ts: str, prompt: str, last_reply_ts: str
    ) -> str:
        """Record the launch instead of contacting the kiss-web daemon."""
        self.launches.append((channel_id, prompt))
        self._store_thread_state(thread_ts, "chat-1", last_reply_ts)
        return "success: true\nsummary: handled"


class TestSignalRunnerForeignSender(unittest.TestCase):
    """The real runner tick must neither act on nor reply about foreign mail."""

    def setUp(self) -> None:
        """Install a destructive spool-based signal-cli on PATH."""
        self._tmpdir = tempfile.mkdtemp(prefix="rr-review-signal-")
        tmp = Path(self._tmpdir)
        cli = tmp / "signal-cli"
        cli.write_text(_SPOOL_CLI, encoding="utf-8")
        cli.chmod(0o755)
        self._spool = tmp / "spool.jsonl"
        self._sends = tmp / "sends.log"
        self._spool.write_text("", encoding="utf-8")
        self._state_path = tmp / "channel_state.json"
        self._old_path = os.environ["PATH"]
        os.environ["PATH"] = self._tmpdir + os.pathsep + self._old_path
        os.environ["KISS_TEST_SPOOL"] = str(self._spool)
        os.environ["KISS_TEST_SENDS"] = str(self._sends)
        # The session conftest points KISS_HOME at a temp dir, so this
        # config write is sandboxed away from any real user config.
        _config.save({"phone_number": "+1BOT"})
        self._backend = SignalChannelBackend()
        self._backend._phone_number = "+1BOT"

    def tearDown(self) -> None:
        """Restore PATH, clear the test config, and drop the temp dir."""
        _config.clear()
        os.environ["PATH"] = self._old_path
        del os.environ["KISS_TEST_SPOOL"]
        del os.environ["KISS_TEST_SENDS"]
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_runner(self) -> RecordingLaunchRunner:
        """Build a runner monitoring contact +1AAA with persistent state."""
        return RecordingLaunchRunner(
            backend=self._backend,
            channel_name="+1AAA",
            agent_name="Signal Background Agent",
            tools_file="",
            model_name="test-model",
            max_budget=1.0,
            work_dir=self._tmpdir,
            allow_users=None,
            state_path=self._state_path,
        )

    def _sent_recipients(self) -> list[str]:
        """Recipients of every signal-cli send since setUp."""
        if not self._sends.exists():
            return []
        return self._sends.read_text(encoding="utf-8").split()

    def test_foreign_sender_triggers_no_launch_and_no_reply(self) -> None:
        """A foreign sender's message is parked: no task, no reply leak."""
        self._spool.write_text(
            _envelope("+1EVE", 111, "attacker text") + "\n", encoding="utf-8"
        )
        runner = self._make_runner()
        processed = runner.run_once()
        self.assertEqual(processed, 0)
        self.assertEqual(runner.launches, [])
        self.assertEqual(self._sent_recipients(), [])
        # The destructively consumed envelope is parked, not lost.
        self.assertEqual(self._spool.read_text(encoding="utf-8"), "")
        state = load_channel_state(self._state_path)
        self.assertEqual(
            [(e["user"], e["text"]) for e in state["pending_envelopes"]],
            [("+1EVE", "attacker text")],
        )
        # A later tick with an empty spool still does not act on it.
        processed = runner.run_once()
        self.assertEqual(processed, 0)
        self.assertEqual(runner.launches, [])
        self.assertEqual(self._sent_recipients(), [])
        state = load_channel_state(self._state_path)
        self.assertEqual(len(state["pending_envelopes"]), 1)

    def test_matching_sender_is_handled_and_replied(self) -> None:
        """The configured contact's message launches a task and gets a reply."""
        self._spool.write_text(
            _envelope("+1AAA", 222, "hello bot") + "\n", encoding="utf-8"
        )
        runner = self._make_runner()
        processed = runner.run_once()
        self.assertEqual(processed, 1)
        self.assertEqual(len(runner.launches), 1)
        self.assertEqual(runner.launches[0][0], "+1AAA")
        self.assertIn("hello bot", runner.launches[0][1])
        self.assertEqual(self._sent_recipients(), ["+1AAA"])

    def test_mixed_senders_only_configured_contact_is_served(self) -> None:
        """Foreign mail in the same tick is parked; only +1AAA is answered."""
        self._spool.write_text(
            _envelope("+1EVE", 301, "eve says hi")
            + "\n"
            + _envelope("+1AAA", 302, "real question")
            + "\n",
            encoding="utf-8",
        )
        runner = self._make_runner()
        processed = runner.run_once()
        self.assertEqual(processed, 1)
        self.assertEqual([c for c, _ in runner.launches], ["+1AAA"])
        self.assertNotIn("eve says hi", runner.launches[0][1])
        self.assertEqual(self._sent_recipients(), ["+1AAA"])
        state = load_channel_state(self._state_path)
        self.assertEqual(
            [(e["user"], e["text"]) for e in state["pending_envelopes"]],
            [("+1EVE", "eve says hi")],
        )

    def test_parked_matching_envelope_is_delivered_next_tick(self) -> None:
        """A matching envelope parked by an earlier tick is handled later."""
        state = load_channel_state(self._state_path)
        state["pending_envelopes"] = [
            {"ts": "400", "user": "+1AAA", "text": "parked question"}
        ]
        from kiss.agents.third_party_agents._channel_agent_utils import (
            save_channel_state,
        )

        save_channel_state(self._state_path, state)
        runner = self._make_runner()
        processed = runner.run_once()
        self.assertEqual(processed, 1)
        self.assertIn("parked question", runner.launches[0][1])
        self.assertEqual(self._sent_recipients(), ["+1AAA"])
        state = load_channel_state(self._state_path)
        self.assertEqual(state["pending_envelopes"], [])


if __name__ == "__main__":
    unittest.main()
