# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for :mod:`kiss.server.voice_wake_control`.

Exercises the daemon-side wake-word listener controller against REAL
child processes: a stand-in listener script speaks the exact stdout
protocol of ``kiss.server.voice_wake`` (``READY``, ``WAKE``,
``TRANSCRIBING``, ``NO_SPEECH``, ``SPEECH {json}``), so the
controller's spawn / event-forwarding / stop / crash-reporting
behaviour is verified without touching a microphone or downloading
models.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from typing import Any

from kiss.server.voice_wake_control import (
    VoiceWakeController,
    parse_protocol_line,
)

_PROTOCOL_SCRIPT = r"""
import sys, time
print("diagnostic noise ignored by the parser", file=sys.stderr, flush=True)
print("READY", flush=True)
print("WAKE", flush=True)
print("TRANSCRIBING", flush=True)
print('SPEECH {"text": "hello world", "speaker": 2, "language": "en"}',
      flush=True)
print("NO_SPEECH", flush=True)
print("not a protocol line", flush=True)
time.sleep(60)
"""

_CRASH_SCRIPT = r"""
import sys
print("READY", flush=True)
print("mic exploded", file=sys.stderr, flush=True)
sys.exit(3)
"""

_SLEEP_SCRIPT = r"""
import time
print("READY", flush=True)
time.sleep(60)
"""


def _args(script: str) -> list[str]:
    return [sys.executable, "-u", "-c", script]


class _EventCollector:
    """Collects controller events and lets tests await their arrival."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._arrived = asyncio.Event()

    async def send(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        self._arrived.set()

    async def wait_for(
        self, predicate: Any, timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Return the first collected event matching *predicate*."""
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            for event in self.events:
                if predicate(event):
                    return event
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise AssertionError(
                    f"no matching event in {self.events!r}"
                )
            self._arrived.clear()
            try:
                await asyncio.wait_for(self._arrived.wait(), remaining)
            except TimeoutError:
                pass


class TestParseProtocolLine(unittest.TestCase):
    """Branch coverage for :func:`parse_protocol_line`."""

    def test_simple_lines(self) -> None:
        self.assertEqual(
            parse_protocol_line("READY"),
            {"type": "voiceWakeEvent", "event": "ready"},
        )
        self.assertEqual(
            parse_protocol_line("WAKE"),
            {"type": "voiceWakeEvent", "event": "wake"},
        )
        self.assertEqual(
            parse_protocol_line("TRANSCRIBING"),
            {"type": "voiceWakeEvent", "event": "transcribing"},
        )
        self.assertEqual(
            parse_protocol_line("NO_SPEECH"),
            {"type": "voiceWakeEvent", "event": "no_speech"},
        )

    def test_speech_object_payload(self) -> None:
        event = parse_protocol_line(
            'SPEECH {"text": "hi", "speaker": 3, "language": "fr"}'
        )
        self.assertEqual(event, {
            "type": "voiceWakeEvent", "event": "speech",
            "text": "hi", "speaker": 3, "language": "fr",
        })

    def test_speech_string_payload(self) -> None:
        event = parse_protocol_line('SPEECH "just text"')
        self.assertEqual(event, {
            "type": "voiceWakeEvent", "event": "speech",
            "text": "just text", "speaker": None, "language": None,
        })

    def test_speech_junk_payloads_degrade_gracefully(self) -> None:
        for line in (
            "SPEECH {broken json",
            "SPEECH 42",
            'SPEECH {"text": 7}',
            'SPEECH {"text": "x", "speaker": 0, "language": ""}',
            'SPEECH {"text": "x", "speaker": true, "language": 5}',
        ):
            event = parse_protocol_line(line)
            assert event is not None
            self.assertEqual(event["event"], "speech")
            self.assertIsNone(event["speaker"])
            self.assertIsNone(event["language"])

    def test_unknown_line_is_ignored(self) -> None:
        self.assertIsNone(parse_protocol_line("mic watchdog: whatever"))
        self.assertIsNone(parse_protocol_line(""))


class TestVoiceWakeController(unittest.TestCase):
    """The controller against real protocol-speaking child processes."""

    def _run(self, coro: Any) -> Any:
        return asyncio.run(coro)

    def test_events_are_forwarded_and_stop_reaps_the_child(self) -> None:
        async def _scenario() -> None:
            controller = VoiceWakeController(_args(_PROTOCOL_SCRIPT))
            collector = _EventCollector()
            await controller.start("c1", 50, collector.send)
            self.assertTrue(controller.running("c1"))
            await collector.wait_for(
                lambda e: e.get("event") == "no_speech"
            )
            proc = controller._listeners["c1"].proc
            await controller.stop("c1")
            self.assertFalse(controller.running("c1"))
            self.assertIsNotNone(proc.returncode)
            events = [
                e.get("event")
                for e in collector.events
                if e["type"] == "voiceWakeEvent"
            ]
            self.assertEqual(
                events,
                ["ready", "wake", "transcribing", "speech", "no_speech"],
            )
            speech = next(
                e for e in collector.events if e.get("event") == "speech"
            )
            self.assertEqual(speech["text"], "hello world")
            self.assertEqual(speech["speaker"], 2)
            self.assertEqual(speech["language"], "en")
            ready_state = collector.events[0]
            self.assertEqual(
                ready_state,
                {"type": "voiceWakeState", "listening": True},
            )
            # A stopped listener reports NO final state: its owner
            # asked for the stop (or disconnected).
            self.assertNotIn(
                {"type": "voiceWakeState", "listening": False},
                collector.events,
            )

        self._run(_scenario())

    def test_duplicate_start_only_reconfirms_listening(self) -> None:
        async def _scenario() -> None:
            controller = VoiceWakeController(_args(_SLEEP_SCRIPT))
            collector = _EventCollector()
            await controller.start("c1", None, collector.send)
            await collector.wait_for(
                lambda e: e.get("type") == "voiceWakeState"
            )
            first_pid = controller._listeners["c1"].proc.pid
            await controller.start("c1", None, collector.send)
            self.assertEqual(controller._listeners["c1"].proc.pid, first_pid)
            states = [
                e for e in collector.events
                if e["type"] == "voiceWakeState"
            ]
            self.assertEqual(
                states[-1], {"type": "voiceWakeState", "listening": True},
            )
            await controller.stop("c1")

        self._run(_scenario())

    def test_self_exit_reports_error_with_stderr_detail(self) -> None:
        async def _scenario() -> None:
            controller = VoiceWakeController(_args(_CRASH_SCRIPT))
            collector = _EventCollector()
            await controller.start("c1", None, collector.send)
            final = await collector.wait_for(
                lambda e: e.get("type") == "voiceWakeState"
                and e.get("listening") is False
            )
            self.assertIn("code 3", final["error"])
            self.assertIn("mic exploded", final["error"])
            self.assertFalse(controller.running("c1"))

        self._run(_scenario())

    def test_spawn_failure_reports_error_state(self) -> None:
        async def _scenario() -> None:
            controller = VoiceWakeController(
                ["/nonexistent-binary-for-test"]
            )
            collector = _EventCollector()
            await controller.start("c1", None, collector.send)
            self.assertFalse(controller.running("c1"))
            self.assertEqual(len(collector.events), 1)
            event = collector.events[0]
            self.assertEqual(event["type"], "voiceWakeState")
            self.assertFalse(event["listening"])
            self.assertIn("failed to start", event["error"])

        self._run(_scenario())

    def test_stop_all_and_per_connection_isolation(self) -> None:
        async def _scenario() -> None:
            controller = VoiceWakeController(_args(_SLEEP_SCRIPT))
            a, b = _EventCollector(), _EventCollector()
            await controller.start("conn-a", None, a.send)
            await controller.start("conn-b", None, b.send)
            self.assertTrue(controller.running("conn-a"))
            self.assertTrue(controller.running("conn-b"))
            await controller.stop("conn-a")
            self.assertFalse(controller.running("conn-a"))
            self.assertTrue(controller.running("conn-b"))
            await controller.stop_all()
            self.assertFalse(controller.running("conn-b"))

        self._run(_scenario())

    def test_stop_joins_the_pump_tasks(self) -> None:
        """After stop() returns, no controller coroutine may be live.

        The disconnect cleanup and daemon shutdown rely on this: a
        pump still running after stop() could touch a torn-down
        endpoint.
        """

        async def _scenario() -> None:
            controller = VoiceWakeController(_args(_PROTOCOL_SCRIPT))
            collector = _EventCollector()
            await controller.start("c1", None, collector.send)
            listener = controller._listeners["c1"]
            await collector.wait_for(lambda e: e.get("event") == "wake")
            await controller.stop("c1")
            self.assertTrue(all(t.done() for t in listener.pumps))
            before = len(collector.events)
            await asyncio.sleep(0.05)
            self.assertEqual(len(collector.events), before)

        self._run(_scenario())

    def test_stop_without_listener_is_a_noop(self) -> None:
        async def _scenario() -> None:
            controller = VoiceWakeController(_args(_SLEEP_SCRIPT))
            await controller.stop("nobody")
            self.assertFalse(controller.running("nobody"))

        self._run(_scenario())

    def test_dead_client_send_does_not_kill_the_pump(self) -> None:
        async def _scenario() -> None:
            controller = VoiceWakeController(_args(_PROTOCOL_SCRIPT))
            collector = _EventCollector()
            calls = {"n": 0}

            async def flaky_send(event: dict[str, Any]) -> None:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise ConnectionError("client went away")
                await collector.send(event)

            await controller.start("c1", None, flaky_send)
            await collector.wait_for(
                lambda e: e.get("event") == "no_speech"
            )
            self.assertTrue(controller.running("c1"))
            await controller.stop("c1")

        self._run(_scenario())


if __name__ == "__main__":
    unittest.main()
