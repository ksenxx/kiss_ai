# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: tool output and replays are bounded in size.

A ``cat`` of a large file used to land in the events table as ONE
``system_output`` event of up to 100 MB (the Bash tool streams its raw
output), and every replay of the chat re-read, re-serialized and
re-sent it in one uncapped ``task_events`` frame.  The three caps
pinned here:

* :class:`JsonPrinter` cuts the output one tool call streams at
  ``MAX_SYSTEM_OUTPUT_CHARS`` (a marker line ends the stream; the next
  tool call starts a fresh budget), and the one-shot ``tool_output``
  message path is cut the same way;
* :func:`persistence._fetch_events_for_task_id` never loads a row
  above ``MAX_EVENT_ROW_CHARS`` into Python: SQLite returns the event
  type and the head of its text, and the row becomes a clipped event
  (or a ``system_output`` note when it carries no text);
* :func:`server._coalesced_replay_events` clips every delta event's
  text and keeps only the newest events that fit the frame budget,
  with a note standing in for the omitted prefix.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.server import server as server_mod
from kiss.server.json_printer import (
    MAX_SYSTEM_OUTPUT_CHARS,
    SYSTEM_OUTPUT_CAP_MARKER,
    JsonPrinter,
)


class _CapturingPrinter(JsonPrinter):
    """JsonPrinter that captures every broadcast event."""

    def __init__(self) -> None:
        super().__init__()
        self.captured: list[dict[str, Any]] = []
        self._captured_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Capture a copy of the (taskId-injected) event, then record it."""
        event = self._inject_task_id(event)
        with self._captured_lock:
            self.captured.append(dict(event))
        super().broadcast(event)


def _system_output_text(events: list[dict[str, Any]]) -> str:
    """Join the ``system_output`` texts of *events* in order."""
    return "".join(
        e.get("text", "") for e in events if e.get("type") == "system_output"
    )


class StreamCapTest(unittest.TestCase):
    """One tool call's streamed output stops at the cap."""

    def _printer(self) -> _CapturingPrinter:
        printer = _CapturingPrinter()
        printer._thread_local.task_id = "task-cap"
        printer.start_recording()
        return printer

    def test_bash_stream_is_cut_at_the_cap_per_tool_call(self) -> None:
        """A real Bash command printing 3x the cap yields ~1x plus a marker."""
        printer = self._printer()
        chunk = "x" * 1000
        line_count = (3 * MAX_SYSTEM_OUTPUT_CHARS) // 1001

        def _stream(text: str) -> None:
            printer.print(text, type="bash_stream")

        tools = UsefulTools(stream_callback=_stream)
        printer.print("Bash", type="tool_call", tool_input={"command": "yes"})
        tools.Bash(
            f"yes {chunk} | head -n {line_count}",
            description="flood the stream",
            timeout_seconds=60,
            max_output_chars=2000,
        )
        # A second call gets its own budget.
        printer.print("Bash", type="tool_call", tool_input={"command": "echo"})
        tools.Bash("echo second_call_visible", description="after the flood")
        printer.print("Read", type="tool_call", tool_input={})
        events = printer.stop_recording()

        call_indexes = [
            i for i, e in enumerate(events) if e.get("type") == "tool_call"
        ]
        self.assertGreaterEqual(len(call_indexes), 2, events[:3])
        first_call = _system_output_text(events[: call_indexes[1]])
        self.assertIn(SYSTEM_OUTPUT_CAP_MARKER, first_call)
        self.assertEqual(first_call.count(SYSTEM_OUTPUT_CAP_MARKER), 1)
        self.assertLessEqual(
            len(first_call), MAX_SYSTEM_OUTPUT_CHARS + len(SYSTEM_OUTPUT_CAP_MARKER),
        )
        self.assertTrue(first_call.endswith(SYSTEM_OUTPUT_CAP_MARKER))
        self.assertIn("second_call_visible", _system_output_text(events))

    def test_tool_result_resets_the_budget(self) -> None:
        """Fragments after the cap are dropped until a tool_result arrives."""
        printer = self._printer()
        printer.print("a" * (MAX_SYSTEM_OUTPUT_CHARS - 10), type="bash_stream")
        printer.print("b" * 100, type="bash_stream")
        printer.print("c" * 100, type="bash_stream")
        printer._flush_bash()
        text = _system_output_text(printer.captured)
        self.assertTrue(text.endswith(SYSTEM_OUTPUT_CAP_MARKER))
        body = text[: -len(SYSTEM_OUTPUT_CAP_MARKER)]
        self.assertEqual(len(body), MAX_SYSTEM_OUTPUT_CHARS)
        self.assertEqual(body.count("b"), 10)
        self.assertNotIn("c", body)
        # A capped stream is no substitute for the result: the bounded
        # result (the only copy of the tail) is shown, not blanked as
        # it is for a stream that was shown in full.
        printer.print(
            "tail FATAL_BUILD_ERROR", type="tool_result", tool_name="Bash",
            tool_input={"command": "x"},
        )
        results = [e for e in printer.captured if e.get("type") == "tool_result"]
        self.assertEqual(results[-1]["content"], "tail FATAL_BUILD_ERROR")
        # The tool result resets the budget for the next command.
        printer.print("d" * 50, type="bash_stream")
        printer._flush_bash()
        self.assertIn("d" * 50, _system_output_text(printer.captured))
        # A stream shown in full still blanks its (duplicate) result.
        printer.print(
            "d" * 50, type="tool_result", tool_name="Bash",
            tool_input={"command": "y"},
        )
        results = [e for e in printer.captured if e.get("type") == "tool_result"]
        self.assertEqual(results[-1]["content"], "")
        printer.stop_recording()

    def test_fragment_landing_exactly_on_the_cap_still_gets_a_marker(self) -> None:
        """The cut is marked even when the cap falls on a fragment boundary."""
        printer = self._printer()
        printer.print("e" * MAX_SYSTEM_OUTPUT_CHARS, type="bash_stream")
        printer.print("omitted", type="bash_stream")
        printer._flush_bash()
        text = _system_output_text(printer.captured)
        self.assertEqual(text, "e" * MAX_SYSTEM_OUTPUT_CHARS + SYSTEM_OUTPUT_CAP_MARKER)
        printer.stop_recording()

    def test_tool_output_message_is_clipped(self) -> None:
        """The one-shot ``tool_output`` message path is cut the same way."""
        printer = self._printer()

        class _Msg:
            subtype = "tool_output"
            data = {"content": "z" * (MAX_SYSTEM_OUTPUT_CHARS + 5)}

        printer._handle_message(_Msg())
        text = _system_output_text(printer.captured)
        self.assertEqual(
            text, "z" * MAX_SYSTEM_OUTPUT_CHARS + SYSTEM_OUTPUT_CAP_MARKER,
        )
        printer.stop_recording()


class OversizedRowTest(unittest.TestCase):
    """Legacy rows above the row cap are clipped by SQLite, not loaded."""

    def setUp(self) -> None:
        self._saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        self._dir = Path(tempfile.mkdtemp(prefix="kiss_replay_caps_"))
        th._KISS_DIR = self._dir
        th._DB_PATH = self._dir / "history.db"
        th._db_conn = None

    def tearDown(self) -> None:
        th._close_db()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved

    def _insert_raw_row(self, task_id: str, event_json: str) -> None:
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "INSERT INTO events (task_id, seq, timestamp, event_json) "
                "VALUES (?, (SELECT COALESCE(MAX(seq), 0) + 1 FROM events "
                "WHERE task_id = ?), 1.0, ?)",
                (task_id, task_id, event_json),
            )
            db.commit()

    def test_oversized_rows_come_back_clipped(self) -> None:
        task_id, _chat = th._add_task("big output task")
        big = "y" * (th.MAX_EVENT_ROW_CHARS + 100)
        self._insert_raw_row(task_id, json.dumps({"type": "prompt", "text": "p"}))
        self._insert_raw_row(
            task_id,
            json.dumps({"type": "system_output", "text": big, "kept": [1, 2]}),
        )
        self._insert_raw_row(
            task_id,
            json.dumps({"type": "tool_call", "content": big, "tool": "Write"}),
        )
        self._insert_raw_row(task_id, "{" + big)  # malformed, oversized
        self._insert_raw_row(task_id, json.dumps({"type": "text", "text": "end"}))
        row = th._load_chat_events_by_task_id(task_id)
        assert row is not None
        events = row["events"]
        assert isinstance(events, list)
        self.assertEqual(
            [e["type"] for e in events],
            ["prompt", "system_output", "system_output", "text"],
        )
        clipped = events[1]["text"]
        self.assertTrue(clipped.startswith("y" * th.MAX_EVENT_TEXT_CHARS))
        self.assertLess(len(clipped), th.MAX_EVENT_TEXT_CHARS + 200)
        self.assertIn("truncated", clipped)
        self.assertEqual(events[1]["kept"], [1, 2], "the small fields survive")
        self.assertIn("tool_call event", events[2]["text"])
        self.assertIn("was not replayed", events[2]["text"])
        self.assertTrue(all("_timestamp" in e for e in events))

    def test_oversized_text_row_with_large_other_fields_keeps_the_text(self) -> None:
        task_id, _chat = th._add_task("big text and big rest")
        big = "y" * (th.MAX_EVENT_ROW_CHARS + 100)
        self._insert_raw_row(task_id, json.dumps({
            "type": "text_delta", "text": big, "blob": "b" * (th.MAX_EVENT_REST_CHARS + 1),
        }))
        row = th._load_chat_events_by_task_id(task_id)
        assert row is not None
        events = row["events"]
        assert isinstance(events, list)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "text_delta")
        self.assertNotIn("blob", events[0])
        self.assertTrue(events[0]["text"].startswith("y" * th.MAX_EVENT_TEXT_CHARS))

    def test_metadata_only_load_reads_no_events(self) -> None:
        task_id, chat_id = th._add_task("meta only")
        self._insert_raw_row(task_id, json.dumps({"type": "prompt", "text": "p"}))
        row = th._load_latest_chat_events_by_chat_id(chat_id, with_events=False)
        assert row is not None
        self.assertEqual(row["task_id"], task_id)
        self.assertEqual(row["events"], [])
        self.assertEqual(
            th._load_subagent_rows_by_parent_task_id(task_id, with_events=False), [],
        )


class BoundReplayEventsTest(unittest.TestCase):
    """The coalesced replay is clipped per event and bounded per frame."""

    def test_small_transcript_is_untouched(self) -> None:
        evs = [{"type": "prompt", "text": "p"}, {"type": "text_delta", "text": "a"}]
        out = server_mod._coalesced_replay_events(list(evs))
        self.assertEqual(out, evs)

    def test_delta_text_is_clipped_head_and_tail_without_mutating_input(self) -> None:
        big = "h" * 150_000 + "m" * 100_000 + "t" * 150_000
        src = {"type": "system_output", "text": big}
        out = server_mod._coalesced_replay_events([src])
        self.assertEqual(len(out), 1)
        self.assertEqual(src["text"], big, "the live recording entry was edited")
        text = out[0]["text"]
        self.assertTrue(text.startswith("h" * 100_000))
        self.assertTrue(text.endswith("t" * 100_000))
        self.assertIn("200000 characters omitted from replay", text)
        # Non-delta events keep their text whole.
        tool = {"type": "tool_call", "text": big}
        self.assertEqual(server_mod._coalesced_replay_events([tool])[0]["text"], big)

    def test_coalesced_stream_is_clipped_too(self) -> None:
        """Many small legacy chunks merge into one event, then get cut."""
        chunks = [{"type": "system_output", "text": "c" * 50_000} for _ in range(10)]
        out = server_mod._coalesced_replay_events(chunks)
        self.assertEqual(len(out), 1)
        self.assertLess(len(out[0]["text"]), server_mod.MAX_REPLAY_EVENT_TEXT_CHARS + 100)

    def test_frame_budget_keeps_the_newest_events(self) -> None:
        per_event = server_mod.MAX_REPLAY_EVENT_TEXT_CHARS
        count = server_mod.MAX_REPLAY_FRAME_BYTES // per_event + 5
        evs = [
            {"type": "tool_call", "text": "e" * per_event, "n": i, "ts": 1000 + i}
            for i in range(count)
        ]
        out = server_mod._coalesced_replay_events(evs)
        self.assertEqual(out[0]["type"], "system_output")
        self.assertIn("earlier events are not shown", out[0]["text"])
        self.assertEqual(out[0]["ts"], out[1]["ts"])
        kept = [e["n"] for e in out[1:]]
        self.assertEqual(kept, list(range(count - len(kept), count)))
        self.assertEqual(kept[-1], count - 1, "the newest event must survive")
        total = sum(len(json.dumps(e)) for e in out[1:])
        self.assertLessEqual(total, server_mod.MAX_REPLAY_FRAME_BYTES)
        self.assertIn(f"[{count - len(kept)} earlier events", out[0]["text"])

    def test_frame_budget_carries_the_latest_usage_totals(self) -> None:
        """The newest usage_info in the cut prefix survives: the run's
        cost must not vanish because a long tool output followed it."""
        per_event = server_mod.MAX_REPLAY_EVENT_TEXT_CHARS
        count = server_mod.MAX_REPLAY_FRAME_BYTES // per_event + 5
        older = {"type": "usage_info", "text": "", "cost": "$0.1000", "ts": 1}
        latest = {"type": "usage_info", "text": "", "cost": "$0.4000", "ts": 2}
        evs = [
            {"type": "prompt", "text": "p", "ts": 0},
            older,
            latest,
            *(
                {"type": "tool_call", "text": "e" * per_event, "n": i, "ts": 1000 + i}
                for i in range(count)
            ),
        ]
        out = server_mod._coalesced_replay_events(evs)
        self.assertEqual(out[0]["type"], "system_output")
        self.assertEqual(out[1], latest)
        self.assertEqual([e for e in out if e.get("type") == "usage_info"], [latest])
        self.assertEqual(out[-1]["n"], count - 1)
        # A kept suffix that has its own totals carries nothing across.
        evs_with_totals = [*evs, {"type": "usage_info", "text": "", "cost": "$0.9000"}]
        out = server_mod._coalesced_replay_events(evs_with_totals)
        self.assertEqual(
            [e["cost"] for e in out if e.get("type") == "usage_info"], ["$0.9000"],
        )
        # No usage_info in the cut prefix: nothing to carry.
        out = server_mod._coalesced_replay_events(evs[3:])
        self.assertEqual([e for e in out if e.get("type") == "usage_info"], [])

    def test_note_without_timestamp(self) -> None:
        per_event = server_mod.MAX_REPLAY_EVENT_TEXT_CHARS
        count = server_mod.MAX_REPLAY_FRAME_BYTES // per_event + 2
        evs = [{"type": "tool_call", "text": "e" * per_event} for _ in range(count)]
        out = server_mod._coalesced_replay_events(evs)
        self.assertNotIn("ts", out[0])

    def test_budget_counts_serialized_bytes_not_characters(self) -> None:
        """Non-ASCII text is escaped on the wire: six bytes per character."""
        emoji = "\U0001f600" * 100_000  # 100 k chars, 1.2 MB serialized
        evs = [{"type": "tool_call", "text": emoji, "n": i} for i in range(20)]
        out = server_mod._coalesced_replay_events(evs)
        self.assertEqual(out[0]["type"], "system_output")
        total = sum(len(json.dumps(e)) for e in out[1:])
        self.assertLessEqual(total, server_mod.MAX_REPLAY_FRAME_BYTES)
        self.assertLess(len(out), 21)
        self.assertEqual(out[-1]["n"], 19)

    def test_newest_event_survives_even_when_over_budget_alone(self) -> None:
        huge = {"type": "result", "summary": "s" * (server_mod.MAX_REPLAY_FRAME_BYTES + 1)}
        out = server_mod._coalesced_replay_events([{"type": "prompt", "text": "p"}, huge])
        self.assertEqual([e["type"] for e in out], ["system_output", "result"])
        self.assertEqual(out[1]["summary"], huge["summary"])
        alone = server_mod._coalesced_replay_events([dict(huge)])
        self.assertEqual(len(alone), 1)


if __name__ == "__main__":
    unittest.main()
