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
``signal-cli`` stand-in program whose ``receive`` is genuinely destructive
(it truncates its spool file) and whose ``send`` records every
recipient.  The runner is the real ``ChannelRunner``: its
``_launch_task`` travels the real wire (``run_agent_via_kiss_web`` →
``sorcar.run``) to a :class:`RecordingDaemon` that records every ``run``
command and answers with a scripted result, so no LLM runs.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from kiss.agents.third_party_agents import _kiss_web_launcher
from kiss.agents.third_party_agents._channel_agent_utils import (
    ChannelRunner,
    load_channel_state,
    save_channel_state,
)
from kiss.agents.third_party_agents.signal.signal_sea import (
    SignalChannelBackend,
    _config,
)
from kiss.tests.agents.third_party_agents.recording_daemon import RecordingDaemon
from kiss.tests.conftest import install_cli_script

# A Python program (not a shell script) so the same stand-in runs on
# Windows, where ``install_cli_script`` adds the ``.cmd`` shim.
_SPOOL_CLI = """#!/usr/bin/env python3
import os
import sys
args = sys.argv[1:]
if args[:1] == ["-u"]:
    args = args[2:]
cmd = args[0] if args else ""
if cmd == "receive":
    spool = os.environ["KISS_TEST_SPOOL"]
    with open(spool, encoding="utf-8") as fh:
        sys.stdout.write(fh.read())
    open(spool, "w").close()
elif cmd == "send":
    with open(os.environ["KISS_TEST_SENDS"], "a", encoding="utf-8") as fh:
        fh.write(args[-1] + "\\n")
sys.exit(0)
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


class TestSignalRunnerForeignSender(unittest.TestCase):
    """The real runner tick must neither act on nor reply about foreign mail."""

    def setUp(self) -> None:
        """Install a destructive spool-based signal-cli on PATH."""
        # Each mutation registers its undo immediately: ``addCleanup``
        # callbacks run even when a later line of ``setUp`` raises, whereas
        # ``tearDown`` would not, which could leave the destructive fake
        # ``signal-cli`` first on PATH for every later test in the process.
        self._tmpdir = tempfile.mkdtemp(prefix="rr-review-signal-")
        self.addCleanup(shutil.rmtree, self._tmpdir, ignore_errors=True)
        tmp = Path(self._tmpdir)
        install_cli_script(tmp / "signal-cli", _SPOOL_CLI)
        self._spool = tmp / "spool.jsonl"
        self._sends = tmp / "sends.log"
        self._spool.write_text("", encoding="utf-8")
        self._state_path = tmp / "channel_state.json"
        old_path = os.environ["PATH"]
        self.addCleanup(os.environ.__setitem__, "PATH", old_path)
        os.environ["PATH"] = self._tmpdir + os.pathsep + old_path
        self.addCleanup(os.environ.pop, "KISS_TEST_SPOOL", None)
        os.environ["KISS_TEST_SPOOL"] = str(self._spool)
        self.addCleanup(os.environ.pop, "KISS_TEST_SENDS", None)
        os.environ["KISS_TEST_SENDS"] = str(self._sends)
        # The session conftest points KISS_HOME at a temp dir, so this
        # config write is sandboxed away from any real user config.
        self.addCleanup(_config.clear)
        _config.save({"phone_number": "+1BOT"})
        self._backend = SignalChannelBackend()
        self._backend._phone_number = "+1BOT"
        # The daemon stand-in the runner's real ``_launch_task`` reaches:
        # ``run_agent_via_kiss_web`` resolves its endpoint file from the
        # module-level ``_ENDPOINT_FILE_OVERRIDE`` when none is passed.
        self._daemon = RecordingDaemon(text="handled", chat_id="chat-1")
        self.addCleanup(self._daemon.close)
        saved_override = _kiss_web_launcher._ENDPOINT_FILE_OVERRIDE
        self.addCleanup(setattr, _kiss_web_launcher, "_ENDPOINT_FILE_OVERRIDE", saved_override)
        _kiss_web_launcher._ENDPOINT_FILE_OVERRIDE = str(self._daemon.endpoint_file)

    def _launched_prompts(self) -> list[str]:
        """Prompts of the ``run`` commands the daemon received, in order."""
        return [str(command["prompt"]) for command in self._daemon.run_commands]

    def _make_runner(self) -> ChannelRunner:
        """Build a runner monitoring contact +1AAA with persistent state."""
        return ChannelRunner(
            backend=self._backend,
            channel_name="+1AAA",
            agent_name="Signal Background Agent",
            sea_path="",
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
        self.assertEqual(self._launched_prompts(), [])
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
        self.assertEqual(self._launched_prompts(), [])
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
        self.assertEqual(len(self._launched_prompts()), 1)
        self.assertIn("hello bot", self._launched_prompts()[0])
        self.assertEqual(self._sent_recipients(), ["+1AAA"])
        # The real launch stored the daemon's chat id for the thread.
        state = load_channel_state(self._state_path)
        self.assertEqual([t["chat_id"] for t in state["threads"].values()], ["chat-1"])

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
        self.assertEqual(len(self._launched_prompts()), 1)
        self.assertIn("real question", self._launched_prompts()[0])
        self.assertNotIn("eve says hi", self._launched_prompts()[0])
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
        save_channel_state(self._state_path, state)
        runner = self._make_runner()
        processed = runner.run_once()
        self.assertEqual(processed, 1)
        self.assertIn("parked question", self._launched_prompts()[0])
        self.assertEqual(self._sent_recipients(), ["+1AAA"])
        state = load_channel_state(self._state_path)
        self.assertEqual(state["pending_envelopes"], [])


if __name__ == "__main__":
    unittest.main()
