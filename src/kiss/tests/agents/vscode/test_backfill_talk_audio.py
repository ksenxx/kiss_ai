# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the kiss-backfill-talk-audio admin command scans and repairs.

Recordings made before the clip-persistence fix carry ``talk``
``tool_call`` events without ``extras.audioB64`` and replay silently.
These tests exercise the backfill script end to end against a real
isolated SQLite database: the scan must find exactly the silent talk
calls (ignoring LIKE-prefilter false positives such as a Bash command
whose text mentions "talk"), and the repair must UPDATE the silent
rows in place (never append a duplicate) using an injected — real,
deterministic — synthesizer function so no paid TTS call happens.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.persistence import _add_task, _flush_chat_events
from kiss.scripts.backfill_talk_audio import (
    main,
    repair_silent_talk_calls,
    scan_talk_calls,
)
from kiss.server.json_printer import JsonPrinter


def _fixed_clip_synth(
    text: str, language: str = "", emotion: str = "",
) -> tuple[str, str] | None:
    """Deterministic synthesizer: returns a fixed valid base64 clip."""
    del text, language, emotion
    return "QUJD", "audio/mpeg"


def _failing_synth(
    text: str, language: str = "", emotion: str = "",
) -> tuple[str, str] | None:
    """Deterministic synthesizer that always fails (returns None)."""
    del text, language, emotion
    return None


class _PersistAgent:
    """Minimal agent stand-in exposing ``_last_task_id`` for persistence."""

    def __init__(self, task_id: str) -> None:
        self._last_task_id = task_id


class _Base(unittest.TestCase):
    """Isolated persistence DB seeded with talk and non-talk events."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-backfill-talk-test-")
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
            "narrate the demo", chat_id="chat-backfill",
        )
        self.printer = self._make_printer(self.task_id)

    def _make_printer(self, task_id: str) -> JsonPrinter:
        printer = JsonPrinter()
        printer._thread_local.task_id = task_id
        with printer._lock:
            printer._persist_agents[task_id] = _PersistAgent(task_id)
        printer.start_recording()
        return printer

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

    def _talk(
        self, printer: JsonPrinter, text: str, audio_b64: str = "",
    ) -> None:
        """Persist a talk tool_call; optionally attach a recorded clip."""
        printer.print(
            "talk",
            type="tool_call",
            tool_input={
                "language": "en-US",
                "text": text,
                "emotion": "cheerful",
            },
        )
        if audio_b64:
            self.assertTrue(
                printer.attach_talk_audio(audio_b64, "audio/mpeg"),
            )

    def _seed_standard_rows(self) -> None:
        """Seed one silent talk, one with-audio talk, one Bash decoy."""
        self._talk(self.printer, "Silent narration.")
        self._talk(self.printer, "Audible narration.", audio_b64="REVG")
        self.printer.print(
            "Bash",
            type="tool_call",
            tool_input={"command": "echo tool_call talk"},
        )
        _flush_chat_events()

    def _persisted_events(self, task_id: str) -> list[dict[str, Any]]:
        _flush_chat_events()
        with _persistence._rw_lock.read_lock():
            db = _persistence._get_db()
            rows = db.execute(
                "SELECT event_json FROM events WHERE task_id = ? "
                "ORDER BY seq",
                (task_id,),
            ).fetchall()
        return [json.loads(r["event_json"]) for r in rows]


class TestScan(_Base):
    """scan_talk_calls finds talk calls and classifies audio state."""

    def test_scan_classifies_silent_and_audible(self) -> None:
        self._seed_standard_rows()
        calls = scan_talk_calls()
        self.assertEqual(len(calls), 2, "Bash decoy must be excluded")
        by_text = {c.text: c for c in calls}
        self.assertFalse(by_text["Silent narration."].has_audio)
        self.assertTrue(by_text["Audible narration."].has_audio)
        silent = by_text["Silent narration."]
        self.assertEqual(silent.task_id, str(self.task_id))
        self.assertEqual(silent.language, "en-US")
        self.assertEqual(silent.emotion, "cheerful")
        self.assertEqual(silent.task, "narrate the demo")

    def test_scan_task_id_filter(self) -> None:
        self._seed_standard_rows()
        other_task, _chat = _add_task("other task", chat_id="chat-other")
        other_printer = self._make_printer(other_task)
        self._talk(other_printer, "Other narration.")
        _flush_chat_events()
        all_calls = scan_talk_calls()
        self.assertEqual(len(all_calls), 3)
        filtered = scan_talk_calls(task_id=str(other_task))
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].text, "Other narration.")

    def test_scan_empty_db(self) -> None:
        self.assertEqual(scan_talk_calls(), [])


class TestRepair(_Base):
    """repair_silent_talk_calls amends only silent rows, in place."""

    def test_repair_amends_only_silent_row(self) -> None:
        self._seed_standard_rows()
        before = self._persisted_events(self.task_id)
        calls = scan_talk_calls()
        repaired, failed, skipped = repair_silent_talk_calls(
            calls, synth=_fixed_clip_synth,
        )
        self.assertEqual((repaired, failed, skipped), (1, 0, 0))
        after = self._persisted_events(self.task_id)
        self.assertEqual(
            len(after), len(before),
            "repair must UPDATE in place, never append a duplicate row",
        )
        talk_calls = [
            e for e in after
            if e.get("type") == "tool_call" and e.get("name") == "talk"
        ]
        by_text = {
            (e.get("extras") or {}).get("text"): e.get("extras") or {}
            for e in talk_calls
        }
        self.assertEqual(by_text["Silent narration."]["audioB64"], "QUJD")
        self.assertEqual(
            by_text["Silent narration."]["audioMime"], "audio/mpeg",
        )
        # The already-audible row keeps its original clip untouched.
        self.assertEqual(by_text["Audible narration."]["audioB64"], "REVG")
        # A second scan reports everything audible now.
        self.assertTrue(all(c.has_audio for c in scan_talk_calls()))

    def test_repair_skips_empty_text(self) -> None:
        self._talk(self.printer, "")
        _flush_chat_events()
        calls = scan_talk_calls()
        self.assertEqual(len(calls), 1)
        with contextlib.redirect_stdout(io.StringIO()):
            repaired, failed, skipped = repair_silent_talk_calls(
                calls, synth=_fixed_clip_synth,
            )
        self.assertEqual((repaired, failed, skipped), (0, 0, 1))
        self.assertFalse(scan_talk_calls()[0].has_audio)

    def test_repair_counts_synthesis_failures(self) -> None:
        self._seed_standard_rows()
        with contextlib.redirect_stdout(io.StringIO()):
            repaired, failed, skipped = repair_silent_talk_calls(
                scan_talk_calls(), synth=_failing_synth,
            )
        self.assertEqual((repaired, failed, skipped), (0, 1, 0))
        by_text = {c.text: c for c in scan_talk_calls()}
        self.assertFalse(by_text["Silent narration."].has_audio)

    def test_repair_honors_limit(self) -> None:
        self._talk(self.printer, "First silent.")
        self._talk(self.printer, "Second silent.")
        self._talk(self.printer, "Third silent.")
        _flush_chat_events()
        with contextlib.redirect_stdout(io.StringIO()):
            repaired, failed, skipped = repair_silent_talk_calls(
                scan_talk_calls(), limit=2, synth=_fixed_clip_synth,
            )
        self.assertEqual((repaired, failed, skipped), (2, 0, 0))
        silent_left = [c for c in scan_talk_calls() if not c.has_audio]
        self.assertEqual(len(silent_left), 1)


class TestMainCli(_Base):
    """main() report modes against the isolated database."""

    def test_report_table_counts(self) -> None:
        self._seed_standard_rows()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main([])
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("Total persisted talk tool_call events: 2", text)
        self.assertIn("SILENT (no extras.audioB64):         1", text)
        self.assertIn("Silent narration.", text)
        self.assertNotIn("Audible narration.", text.split("Total")[0])

    def test_report_json(self) -> None:
        self._seed_standard_rows()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main(["--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["total_talk_calls"], 2)
        self.assertEqual(payload["with_audio"], 1)
        self.assertEqual(payload["silent"], 1)
        self.assertEqual(payload["distinct_tasks_with_silent"], 1)
        self.assertEqual(
            payload["silent_calls"][0]["text"], "Silent narration.",
        )

    def test_db_override_reports_other_database(self) -> None:
        """--db redirects the scan to a copied database file."""
        self._seed_standard_rows()
        # Copy the isolated DB elsewhere, then point --db at the copy.
        copy_dir = Path(self.tmpdir) / "copy"
        copy_dir.mkdir()
        db_copy = copy_dir / "sorcar.db"
        _flush_chat_events()
        with _persistence._rw_lock.read_lock():
            # The database is in WAL mode; checkpoint so the main file
            # alone (the copy target) contains every committed row.
            db = _persistence._get_db()
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        shutil.copy(_persistence._DB_PATH, db_copy)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main(["--db", str(db_copy), "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["total_talk_calls"], 2)
        self.assertEqual(payload["silent"], 1)

    def test_db_override_missing_file_exits(self) -> None:
        with self.assertRaises(SystemExit):
            main(["--db", str(Path(self.tmpdir) / "nope.db")])


if __name__ == "__main__":
    unittest.main()
