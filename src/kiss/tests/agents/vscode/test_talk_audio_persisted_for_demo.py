# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: talk tool_call events must persist the synthesized clip for demos.

The agentic loop broadcasts (and records/persists) the ``tool_call``
event for the ``talk`` tool BEFORE the tool executes, so that event can
never carry the ``audioB64`` clip synthesized DURING execution.  Demo
replay plays exactly ``extras.audioB64`` of the persisted ``tool_call``
event (the synthesis fallback is gone) — without an amendment every
replayed narration would be silent.

These tests exercise the amendment path end to end against a real
isolated SQLite database: ``JsonPrinter.attach_talk_audio`` must update
both the in-memory recording AND the persisted ``events`` row, exactly
once, targeting the newest audio-less ``talk`` call.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.persistence import (
    _add_task,
    _amend_last_talk_tool_call_audio,
    _flush_chat_events,
)
from kiss.agents.vscode.json_printer import JsonPrinter


class _PersistAgent:
    """Minimal agent stand-in exposing ``_last_task_id`` for persistence."""

    def __init__(self, task_id: str) -> None:
        self._last_task_id = task_id


class _Base(unittest.TestCase):
    """Isolated persistence DB + a recording printer per test."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-talk-audio-test-")
        self._saved_db = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None

        self.task_id, _chat = _add_task(
            "narrate something", chat_id="chat-talk",
        )
        self.printer = JsonPrinter()
        self.printer._thread_local.task_id = self.task_id
        with self.printer._lock:
            self.printer._persist_agents[self.task_id] = _PersistAgent(
                self.task_id,
            )
        self.printer.start_recording()

    def tearDown(self) -> None:
        _flush_chat_events()
        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _broadcast_talk_tool_call(self, text: str) -> None:
        """Broadcast the tool_call event as JsonPrinter formats it live."""
        self.printer.print(
            "talk",
            type="tool_call",
            tool_input={
                "language": "en-US",
                "text": text,
                "emotion": "cheerful",
            },
        )

    def _persisted_events(self) -> list[dict[str, Any]]:
        _flush_chat_events()
        with _persistence._rw_lock.read_lock():
            db = _persistence._get_db()
            rows = db.execute(
                "SELECT event_json FROM events WHERE task_id = ? "
                "ORDER BY seq",
                (self.task_id,),
            ).fetchall()
        return [json.loads(r["event_json"]) for r in rows]


class TestAttachTalkAudio(_Base):
    """attach_talk_audio amends the recording and the persisted row."""

    def test_recorded_tool_call_gains_audio(self) -> None:
        self._broadcast_talk_tool_call("Hello there!")
        self.assertTrue(self.printer.attach_talk_audio("QUJD", "audio/mpeg"))
        events = self.printer.stop_recording()
        talk_calls = [
            e for e in events
            if e.get("type") == "tool_call" and e.get("name") == "talk"
        ]
        self.assertEqual(len(talk_calls), 1)
        extras = talk_calls[0].get("extras") or {}
        self.assertEqual(extras.get("audioB64"), "QUJD")
        self.assertEqual(extras.get("audioMime"), "audio/mpeg")
        # The spoken text stays intact next to the clip.
        self.assertEqual(extras.get("text"), "Hello there!")

    def test_persisted_row_gains_audio(self) -> None:
        self._broadcast_talk_tool_call("Persisted narration.")
        self.assertTrue(self.printer.attach_talk_audio("REVG", "audio/mpeg"))
        talk_calls = [
            e for e in self._persisted_events()
            if e.get("type") == "tool_call" and e.get("name") == "talk"
        ]
        self.assertEqual(
            len(talk_calls), 1,
            "the amendment must UPDATE the existing row, never append a "
            "duplicate tool_call (a duplicate renders the panel twice on "
            "demo replay)",
        )
        extras = talk_calls[0].get("extras") or {}
        self.assertEqual(extras.get("audioB64"), "REVG")
        self.assertEqual(extras.get("audioMime"), "audio/mpeg")

    def test_second_talk_gets_its_own_clip(self) -> None:
        """Each utterance's clip lands on its own tool_call event."""
        self._broadcast_talk_tool_call("First utterance.")
        self.assertTrue(self.printer.attach_talk_audio("QQ==", "audio/mpeg"))
        self._broadcast_talk_tool_call("Second utterance.")
        self.assertTrue(self.printer.attach_talk_audio("Qg==", "audio/mpeg"))
        by_text = {
            (e.get("extras") or {}).get("text"): (e.get("extras") or {})
            for e in self._persisted_events()
            if e.get("type") == "tool_call" and e.get("name") == "talk"
        }
        self.assertEqual(by_text["First utterance."]["audioB64"], "QQ==")
        self.assertEqual(by_text["Second utterance."]["audioB64"], "Qg==")

    def test_no_talk_call_returns_false(self) -> None:
        self.printer.print(
            "Bash", type="tool_call", tool_input={"command": "ls"},
        )
        self.assertFalse(self.printer.attach_talk_audio("QUJD", "audio/mpeg"))

    def test_empty_audio_returns_false(self) -> None:
        self._broadcast_talk_tool_call("No clip came back.")
        self.assertFalse(self.printer.attach_talk_audio("", "audio/mpeg"))
        events = self.printer.stop_recording()
        extras = next(
            e.get("extras") or {}
            for e in events
            if e.get("type") == "tool_call" and e.get("name") == "talk"
        )
        self.assertNotIn("audioB64", extras)


class TestAmendPersistenceHelper(_Base):
    """Direct coverage of _amend_last_talk_tool_call_audio edge cases."""

    def test_amends_newest_audioless_call_only(self) -> None:
        self._broadcast_talk_tool_call("Old one.")
        self._broadcast_talk_tool_call("New one.")
        self.assertTrue(
            _amend_last_talk_tool_call_audio(
                self.task_id, "WFla", "audio/mpeg",
            ),
        )
        talk_calls = [
            e for e in self._persisted_events()
            if e.get("type") == "tool_call" and e.get("name") == "talk"
        ]
        with_audio = [
            e for e in talk_calls
            if (e.get("extras") or {}).get("audioB64")
        ]
        self.assertEqual(len(with_audio), 1)
        self.assertEqual(
            (with_audio[0].get("extras") or {}).get("text"), "New one.",
        )

    def test_unknown_task_returns_false(self) -> None:
        self.assertFalse(
            _amend_last_talk_tool_call_audio("999999", "QUJD", "audio/mpeg"),
        )

    def test_non_talk_tool_calls_untouched(self) -> None:
        # A Bash command whose *content* merely mentions "talk" must not
        # be amended (the LIKE prefilter alone would match it).
        self.printer.print(
            "Bash",
            type="tool_call",
            tool_input={"command": "echo tool_call talk"},
        )
        self.assertFalse(
            _amend_last_talk_tool_call_audio(
                self.task_id, "QUJD", "audio/mpeg",
            ),
        )


if __name__ == "__main__":
    unittest.main()
