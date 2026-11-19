# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A chat page that loads transcripts in pieces is sent each piece when its panel is opened.

``resumeSession`` with ``lazy: true`` answers a finished task with its
settings, a ``trajectory`` stub, its result and its last usage; a
running task with its outline (the ``summary`` calls and the panels no
summary folded).  ``getTaskEvents`` then serves the outline of a
finished task (its Trajectory panel was opened) and the panels one
``summary`` call folded (that summary was opened).  See
``kiss/server/lazy_replay.py``.

Real ``AgentServer`` with a real database; a real daemon and a real
link for the fetch over the wire; no mocks.  What the page builds from
the pieces is checked against a whole replay in
``tests/webapp/test_lazy_chat_loading_jsdom.py``.
"""

from __future__ import annotations

import asyncio
import base64
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest import TestCase

import kiss.agents.sorcar.machines as mach
import kiss.agents.sorcar.persistence as th
import kiss.core.user_config as vc
from kiss.server import lazy_replay, sidebar_panels
from kiss.server.server import (
    AgentServer,
    _first_piece_events,
    _lazy_piece_events,
    _replay_events,
)
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import _generate_self_signed_cert
from kiss.tests.persistence_helpers import close_db
from kiss.tests.server.test_web_server_tab_mirroring import TabMirroringBase

_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(range(64))).decode()
_OWN_TYPES = ("trajectory", "lazy_steps", "lazy_gap")


def _call(name: str, **fields: Any) -> dict[str, Any]:
    """A ``tool_call`` event of tool *name*."""
    return {"type": "tool_call", "name": name, **fields}


def _done(name: str, content: str = "ok", **fields: Any) -> dict[str, Any]:
    """The ``tool_result`` event of tool *name*."""
    return {"type": "tool_result", "content": content, "is_error": False, "tool_name": name,
            **fields}


def _thoughts(text: str) -> list[dict[str, Any]]:
    """A streamed thought."""
    return [
        {"type": "thinking_start"},
        {"type": "thinking_delta", "text": text[:3]},
        {"type": "thinking_delta", "text": text[3:]},
        {"type": "thinking_end"},
    ]


def rich_transcript() -> list[dict[str, Any]]:
    """A finished task with every kind of event its layout reads, around three ``summary`` calls."""
    picture = {"b64": _PNG_B64, "mime": "image/png", "path": "a.png"}
    return [
        {"type": "task_settings", "settings": {"model": "m", "work_dir": "/w"}},
        # Before the first step: a usage report, results with no call to land in.
        {"type": "usage_info", "text": "Steps: 0/100", "total_tokens": 1, "cost": "$0.00",
         "total_steps": 0},
        _done("Bash", "left over"),
        {**_done("Bash", "left over failure"), "is_error": True},
        {"type": "system_prompt", "text": "You are an agent.", "early": True},
        {"type": "system_prompt", "text": "You are the agent."},
        {"type": "prompt", "text": "Do the thing."},
        # Step 1: a thought and a command with output.
        *_thoughts("Let me look."),
        {"type": "llm_call", "model": "m", "step": 1, "cost": 0.01},
        {"type": "usage_info", "text": "Steps: 1/100", "total_tokens": 10, "cost": "$0.01",
         "total_steps": 1},
        {"type": "text_end"},
        _call("Bash", command="ls", description="list the files", ts=1794000001000),
        {"type": "system_output", "text": "a.txt\n"},
        {"type": "system_output", "text": "b.txt\n"},
        _done("Bash", "", ts=1794000002000),
        # Step 2: words and a thought in the panel that result opened, then a
        # picture, a typed message, a question, a fan-out and a banner.
        {"type": "text_delta", "text": "Two files."},
        {"type": "text_end"},
        *_thoughts("A picture next."),
        _call("Read", path="/w/a.png"),
        _done("Read", "a picture", images=[picture]),
        {"type": "prompt", "text": "also check b.txt", "steer": True, "promptIds": ["prompt-1"]},
        _call("ask_user_question", extras={"question": "Proceed?"}),
        _done("ask_user_question", "yes"),
        _call("run_parallel", description="fan out", extras={"tasks": '["one", "two"]'}),
        _done("run_parallel", "both done"),
        {"type": "warning", "text": "Careful."},
        _call("summary", description="- listed the files\n- asked"),
        _done("summary"),
        # Step 3: a failed command, loose output lines, commits, an answer, a sub-agent.
        *_thoughts("Next one."),
        _call("Bash", command="false", description="fail"),
        {**_done("Bash", "exit 1"), "is_error": True},
        {"type": "system_output", "text": "note 1\n"},
        {"type": "system_output", "text": "note 2\n"},
        {"type": "system_output", "text": ["not", "text"]},
        {"type": "autocommit_done", "manual": True, "success": True},
        {"type": "autocommit_done", "success": True, "text": "committed"},
        {"type": "ask_answer", "text": "an answer", "question": "a question"},
        _call("run_agent", description="delegate"),
        _done("run_agent", "delegated"),
        _call("summary", description="- failed, noted"),
        _done("summary"),
        # A summary with nothing to fold, then the end and what trails it.
        _call("summary", description="- nothing new"),
        _done("summary"),
        *_thoughts("Wrapping up."),
        _call("finish", extras={"success": "True"}),
        _done("finish", "finished"),
        {"type": "result", "text": "success: true", "summary": "<p>All done.</p>",
         "success": True, "step_count": 4, "total_tokens": 100, "cost": "$0.05"},
        {"type": "text_end"},
        {"type": "llm_call", "model": "m", "step": 5, "cost": 0.0},
        {"type": "usage_info", "text": "Steps: 4/100, Context: 1/2 tokens", "total_tokens": 100,
         "cost": "$0.05"},
        {"type": "task_done", "success": True},
        {"type": "followup_suggestion", "text": "Do more."},
    ]


def kinds(events: list[dict[str, Any]]) -> list[str]:
    """What each of *events* is: its type, a tool call's tool, a stand-in's event."""
    out = []
    for ev in events:
        kind = str(ev.get("type"))
        if kind == "tool_call":
            kind = str(ev.get("name"))
        elif kind == "tool_result":
            kind = f"{ev.get('tool_name')}="
        elif kind == "lazy_gap":
            kind = f"~{ev['of']}"
        out.append(kind)
    return out


#: The ``seq`` of the last event of :func:`rich_transcript`, stored.
LAST = len(rich_transcript()) - 1
FINISHED = [
    "task_settings", "Bash=", "~tool_call", "trajectory", "result", "usage_info",
    "followup_suggestion", "lazy_steps",
]
# A piece stands in for what it leaves out: the block of the result with
# no call (``~stop``), the result card, then the follow-up bar (``~stop``).
OUTLINE = [
    "~stop", "~tool_call", "Read", "Read=", "prompt", "ask_user_question",
    "ask_user_question=", "~tool_call", "summary", "summary=", "~tool_call", "ask_answer",
    "~tool_call", "summary", "summary=", "summary", "summary=", "thinking_start",
    "thinking_delta", "thinking_end", "finish", "finish=", "~result", "~stop",
]
SEGMENTS = [
    [
        "~stop", "~tool_result", "Bash=", "system_prompt", "system_prompt", "prompt",
        "thinking_start", "thinking_delta", "thinking_end", "llm_call", "text_end", "Bash",
        "system_output", "Bash=", "text_delta", "text_end", "thinking_start", "thinking_delta",
        "thinking_end", "~tool_call", "run_parallel", "run_parallel=", "warning", "~tool_call",
        "~result", "~stop",
    ],
    [
        "~stop", "~tool_call", "~tool_result", "thinking_start", "thinking_delta",
        "thinking_end", "Bash", "Bash=", "system_output", "system_output", "autocommit_done",
        "run_agent", "run_agent=", "~tool_call", "~result", "~stop",
    ],
]


def own(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """The events of *kind* among *events*."""
    return [ev for ev in events if ev.get("type") == kind]


class _DatabaseCase(TestCase):
    """A temporary history database and an ``AgentServer`` whose broadcasts are captured."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-lazy-chat-"))
        self._saved_persistence = (th._DB_PATH, th._KISS_DIR)
        kiss_dir = self.tmpdir / ".kiss"
        kiss_dir.mkdir()
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        self.server = AgentServer()
        self.server.work_dir = str(self.tmpdir)
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        original = self.server.printer.broadcast
        self.server.printer.broadcast = self._capturing(original)  # type: ignore[assignment]

    def _capturing(self, original: Any) -> Any:
        """A broadcast that records every event, then forwards it."""
        def capture(event: dict[str, Any]) -> None:
            with self._lock:
                self.events.append(dict(event))
            original(event)
        return capture

    def tearDown(self) -> None:
        close_db()
        th._DB_PATH, th._KISS_DIR = self._saved_persistence
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _persist(self, task: str, stream: list[Any], chat_id: str = "") -> tuple[str, str]:
        """Add a finished task with *stream* as its transcript; return (task_id, chat_id).

        A ``str`` in *stream* is stored as it is: a row that is not an event.
        """
        task_id, chat_id = th._add_task(task, chat_id=chat_id)
        for ev in stream:
            if isinstance(ev, str):
                th._flush_chat_events()
                db = th._get_db()
                seq = db.execute(
                    "SELECT coalesce(max(seq), -1) + 1 FROM events WHERE task_id = ?", (task_id,),
                ).fetchone()[0]
                db.execute(
                    "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, ?, ?, ?)",
                    (task_id, seq, ev, time.time()),
                )
                db.commit()
            else:
                th._queue_chat_event(dict(ev), task_id=task_id)
        th._flush_chat_events()
        th._save_task_result(result="done", task_id=task_id)
        return task_id, chat_id

    def _replayed(self, kind: str = "task_events") -> list[dict[str, Any]]:
        """The events of the last captured *kind* frame."""
        with self._lock:
            frames = own(self.events, kind)
        self.assertTrue(frames, f"no {kind} broadcast")
        return list(frames[-1]["events"])

    def _resume(self, chat_id: str, task_id: str = "", lazy: bool = True) -> list[dict[str, Any]]:
        """What ``resumeSession`` sends tab-1 for the chat (its latest task, or *task_id*)."""
        cmd: dict[str, Any] = {"type": "resumeSession", "chatId": chat_id, "tabId": "tab-1"}
        if task_id:
            cmd["taskId"] = task_id
        if lazy:
            cmd["lazy"] = True
        self.server._handle_command(cmd)
        return self._replayed()


class TestLazyReplay(_DatabaseCase):
    """What a lazy replay sends first, and what each fetch adds."""

    def test_finished_task_is_its_stub_and_result_until_panels_are_opened(self) -> None:
        task_id, chat_id = self._persist("rich", rich_transcript())
        first = self._resume(chat_id, task_id)
        self.assertEqual(kinds(first), FINISHED)
        self.assertEqual(
            own(first, "trajectory"),
            [{
                # The handle names the task and the last event laid out.
                "type": "trajectory", "count": 10, "lazy": {"taskId": task_id, "upto": LAST},
                # The messages typed into the task, for the page's lost-prompt check.
                "steers": [{
                    "type": "prompt", "steer": True, "text": "also check b.txt",
                    "promptIds": ["prompt-1"],
                }],
            }],
        )
        self.assertEqual(own(first, "lazy_steps"), [{"type": "lazy_steps", "steps": 4}])
        # The tool calls left out: the stand-in counts the two fan-outs among them.
        self.assertEqual(
            own(first, "lazy_gap"),
            [{"type": "lazy_gap", "of": "tool_call", "ts": 1794000001000, "fanouts": 2}],
        )
        # The Trajectory panel is opened: the summaries and what none of them folded.
        outline = self.server.task_events_piece(task_id, -1)
        self.assertEqual(kinds(outline), OUTLINE)
        self.assertEqual(
            [ev.get("lazy") for ev in outline if ev.get("name") == "summary"],
            [{"taskId": task_id, "seg": 0}, {"taskId": task_id, "seg": 1}, None],
        )
        self.assertEqual(
            [ev["images"] for ev in outline if "images" in ev],
            [[{"mime": "image/png", "path": "a.png", "lazy": {"taskId": task_id, "index": 0}}]],
        )
        # A summary is opened: the panels it folded.
        segments = [self.server.task_events_piece(task_id, seg) for seg in (0, 1)]
        self.assertEqual([kinds(piece) for piece in segments], SEGMENTS)
        self.assertEqual(
            own(segments[1], "lazy_gap")[:3],
            [
                {"type": "lazy_gap", "of": "stop", "fanouts": 1},
                {"type": "lazy_gap", "of": "tool_call", "ts": 1794000001000},
                {"type": "lazy_gap", "of": "tool_result", "ts": segments[1][3]["ts"]},
            ],
        )
        # Every event of the whole replay is in exactly one piece, but
        # for those that show nothing.
        whole = self._resume(chat_id, task_id, lazy=False)
        self.assertEqual([ev for ev in whole if ev["type"] in _OWN_TYPES], [])
        sent = [
            {k: v for k, v in ev.items() if k != "lazy"}
            for piece in (first, outline, *segments)
            for ev in piece
            if ev["type"] not in _OWN_TYPES
        ]
        whole = [
            ev if "images" not in ev else next(e for e in outline if "images" in e)
            for ev in whole
        ]
        self.assertEqual([ev for ev in sent if ev not in whole], [])
        self.assertEqual(
            kinds([ev for ev in whole if ev not in sent]),
            ["usage_info", "usage_info", "autocommit_done", "text_end", "llm_call", "task_done"],
        )
        self.assertEqual(len(sent) + 6, len(whole))

    def test_latest_task_of_a_chat_and_its_neighbour_are_sent_lazily(self) -> None:
        first_id, chat_id = self._persist("first", rich_transcript())
        time.sleep(0.01)  # the adjacent walk orders the chat's tasks by timestamp
        second_id, _ = self._persist(
            "second", [{"type": "result", "summary": "ok", "success": True}], chat_id,
        )
        latest = self._resume(chat_id)
        # Nothing but a result: no Trajectory panel to open.
        self.assertEqual(kinds(latest), ["task_settings", "result", "lazy_steps"])
        self.assertEqual(own(latest, "lazy_steps"), [{"type": "lazy_steps", "steps": 0}])
        self.server._handle_command({
            "type": "getAdjacentTask", "tabId": "tab-1", "taskId": second_id,
            "direction": "prev", "lazy": True,
        })
        neighbour = self._replayed("adjacent_task_events")
        self.assertEqual(kinds(neighbour), FINISHED)
        self.assertEqual(
            own(neighbour, "trajectory")[0]["lazy"], {"taskId": first_id, "upto": LAST},
        )
        # A page that does not load in pieces gets the neighbour whole.
        self.server._handle_command({
            "type": "getAdjacentTask", "tabId": "tab-1", "taskId": second_id, "direction": "prev",
        })
        self.assertIn("summary", kinds(self._replayed("adjacent_task_events")))

    def test_unknown_task_or_summary_has_no_panels(self) -> None:
        task_id, _ = self._persist("rich", rich_transcript())
        self.assertEqual(self.server.task_events_piece("no-such-task", -1), [])
        self.assertEqual(self.server.task_events_piece("no-such-task", 0), [])
        # The third summary folded nothing; there is no tenth.
        for seg in (2, 9):
            piece = self.server.task_events_piece(task_id, seg)
            self.assertEqual(kinds(piece), ["~stop", "~tool_call", "~result", "~stop"])

    def test_task_stopped_before_its_result_keeps_its_stub_last(self) -> None:
        stream = rich_transcript()[:20]  # up to the words after the first command
        task_id, chat_id = self._persist("stopped", stream)
        first = self._resume(chat_id, task_id)
        self.assertEqual(
            kinds(first),
            ["task_settings", "Bash=", "usage_info", "~tool_call", "trajectory", "lazy_steps"],
        )
        self.assertEqual(own(first, "trajectory"), [
            {"type": "trajectory", "count": 6, "lazy": {"taskId": task_id, "upto": 19}},
        ])
        self.assertEqual(own(first, "lazy_steps"), [{"type": "lazy_steps", "steps": 2}])

    def test_result_with_no_call_before_it_leaves_its_thinking_panel_to_the_trajectory(
        self,
    ) -> None:
        # The call's row was too long to decode: its result is a block
        # of its own, shown with the result cards, and the Thinking
        # panel the page opens after a result is the Trajectory's.
        stream = [_done("Bash", "out"), {"type": "warning", "text": "stopped"}]
        task_id, chat_id = self._persist("cut", stream)
        first = self._resume(chat_id, task_id)
        # Nothing after the result withdraws that panel, so a tool
        # call's stand-in does: the page shows the block and the stub.
        self.assertEqual(
            kinds(first), ["task_settings", "Bash=", "~tool_call", "trajectory", "lazy_steps"],
        )
        self.assertEqual(own(first, "lazy_gap"), [{"type": "lazy_gap", "of": "tool_call"}])
        self.assertEqual(own(first, "trajectory")[0]["count"], 2)
        # The Trajectory panel, opened, holds the Thinking panel and the banner.
        self.assertEqual(
            kinds(self.server.task_events_piece(task_id, -1, 1)),
            ["~stop", "~tool_result", "warning"],
        )
        # A result card after the block withdraws the panel itself.
        done = [stream[0], {"type": "result", "summary": "ok", "success": True}]
        task_id, chat_id = self._persist("cut and done", done)
        self.assertEqual(
            kinds(self._resume(chat_id, task_id)),
            ["task_settings", "Bash=", "result", "lazy_steps"],
        )
        # Words stand in the Thinking panel when such a result comes: it
        # ends them as its call would have, its block ends a summary's
        # reach, and the panel it opens takes the words that follow.
        task_id, chat_id = self._persist("cut in words", [*_thoughts("Hm."), *stream])
        self.assertEqual(own(self._resume(chat_id, task_id), "trajectory")[0]["count"], 3)
        self.assertEqual(
            kinds(self.server.task_events_piece(task_id, -1)),
            ["thinking_start", "thinking_delta", "thinking_end", "~tool_call", "~stop",
             "~tool_result", "warning"],
        )

    def test_words_after_two_results_in_a_row_are_one_thinking_panel(self) -> None:
        # Two calls made at once report in a row, and the words that
        # follow are stored in two rows, which every replay merges.
        stream = [
            _call("Read", path="/w/a"), _call("Read", path="/w/b"),
            _done("Read", "a"), _done("Read", "b"),
            {"type": "text_delta", "text": "They are "}, {"type": "text_delta", "text": "alike."},
        ]
        task_id, chat_id = self._persist("pair", stream)
        first = self._resume(chat_id, task_id)
        # The two commands and the one panel of the words.
        self.assertEqual(own(first, "trajectory")[0]["count"], 3)
        self.assertEqual(own(first, "lazy_steps"), [{"type": "lazy_steps", "steps": 1}])

    def test_trajectory_is_filled_from_the_transcript_its_stub_was_made_of(self) -> None:
        task_id, chat_id = self._persist("rich", rich_transcript())
        first = self._resume(chat_id, task_id)
        upto = own(first, "trajectory")[0]["lazy"]["upto"]
        # An event that comes late (an answer, a message): the page
        # shows it as it arrives, below the Trajectory panel.
        th._append_chat_event({"type": "warning", "text": "late"}, task_id=task_id)
        outline = self.server.task_events_piece(task_id, -1, upto)
        self.assertEqual(kinds(outline), OUTLINE)
        # The panels inside are read from that transcript too.
        self.assertEqual(
            [ev.get("lazy") for ev in outline if ev.get("name") == "summary"],
            [
                {"taskId": task_id, "upto": upto, "seg": 0},
                {"taskId": task_id, "upto": upto, "seg": 1},
                None,
            ],
        )
        self.assertEqual(
            [kinds(self.server.task_events_piece(task_id, seg, upto)) for seg in (0, 1)], SEGMENTS,
        )
        # The transcript as it is now has the late event: a chat opened
        # again counts it in.
        self.assertEqual(kinds(self.server.task_events_piece(task_id, -1)), [*OUTLINE, "warning"])
        again = own(self._resume(chat_id, task_id), "trajectory")[0]
        self.assertEqual((again["count"], again["lazy"]["upto"]), (11, upto + 1))


class TestTranscriptSources(_DatabaseCase):
    """Where a transcript is read: a running task's recording, the events table, odd rows."""

    def test_running_task_is_sent_its_outline_from_the_recording(self) -> None:
        printer = self.server.printer
        task_id, _chat_id = th._add_task("live")
        session = th._load_chat_events_by_task_id(task_id, with_events=False)
        assert session is not None
        printer.ensure_recording_for_task(task_id)
        try:
            for ev in rich_transcript():
                printer.broadcast({**ev, "taskId": task_id})
            recording = printer.peek_recording_for_task(task_id)
            live = _replay_events(session, recording, task_id, lazy=True, live=True)
            # The outline, the results and the status where they happened.
            self.assertEqual(
                kinds(live),
                [*FINISHED[:2], *OUTLINE[1:-2], *FINISHED[4:]],
            )
            self.assertEqual(own(live, "lazy_steps"), [{"type": "lazy_steps", "steps": 4}])
            # A summary's panels are read from the recording too.
            self.assertEqual(
                [kinds(self.server.task_events_piece(task_id, seg)) for seg in (0, 1)], SEGMENTS,
            )
            self.assertIsInstance(lazy_replay.transcript(task_id, recording), lazy_replay.Recorded)
        finally:
            printer.cleanup_task(task_id)

    def test_rows_that_are_no_plain_events_are_laid_out_as_they_decode(self) -> None:
        picture = {"b64": _PNG_B64, "mime": "image/png", "path": "p.png"}
        task_id, chat_id = self._persist("odd", [
            "not json",
            "[1, 2]",
            # Too long to load: clipped to the head of its text, or dropped.
            {"type": "system_output", "text": "x" * (th.MAX_EVENT_ROW_CHARS + 10)},
            "y" * (th.MAX_EVENT_ROW_CHARS + 10),
            {"type": 5},
            *_thoughts("One step."),
            # Step counts that are none: a flag, a text, no number, an endless one.
            {"type": "usage_info", "total_steps": True},
            {"type": "usage_info", "total_steps": "7"},
            {"type": "usage_info", "text": 7},
            {"type": "usage_info", "text": "no count"},
            '{"type": "usage_info", "total_steps": Infinity}',
            # Pictures are numbered over the transcript, a streamed event's too.
            {"type": "thinking_delta", "text": "!", "images": [picture]},
            _call("Read", path="/w/p.png"),
            _done("Read", images=["junk", {"b64": ""}, {"b64": _PNG_B64}, picture]),
            _call("summary", description="- looked"),
            _done("summary"),
        ])
        # A row time that is no number: the event is sent without one.
        db = th._get_db()
        db.execute(
            "UPDATE events SET timestamp = x'00' WHERE task_id = ? AND event_json LIKE ?",
            (task_id, '%"name": "Read"%'),
        )
        db.commit()
        plan = lazy_replay.transcript(task_id, []).plan
        self.assertEqual(plan.steps, 1)
        # The summary, the picture it left out and the Thinking panel its result opened.
        self.assertEqual(plan.panels, 3)
        first = self._resume(chat_id, task_id)
        self.assertEqual(
            kinds(first),
            ["task_settings", "usage_info", "~tool_call", "trajectory", "lazy_steps"],
        )
        outline = self.server.task_events_piece(task_id, -1)
        self.assertEqual(kinds(outline), ["Read", "Read=", "summary", "summary="])
        self.assertEqual(
            [("ts" in ev, "_timestamp" in ev) for ev in outline[:2]],
            [(False, False), (True, True)],
        )
        self.assertEqual(
            [im.get("lazy") for im in outline[1]["images"][2:]],
            [{"taskId": task_id, "index": 1}, {"taskId": task_id, "index": 2}],
        )
        folded = self.server.task_events_piece(task_id, 0)
        self.assertEqual(
            kinds(folded),
            ["system_output", "thinking_start", "thinking_delta", "thinking_end",
             "thinking_delta", "~tool_call"],
        )
        self.assertIn("[truncated: this system_output event was 4 MB]", folded[0]["text"])
        self.assertEqual(folded[4]["images"][0]["lazy"], {"taskId": task_id, "index": 0})

    def test_scan_is_kept_while_the_transcript_stands_and_only_for_the_last_tasks(self) -> None:
        task_id, _ = self._persist("rich", rich_transcript())
        stored = lazy_replay.transcript(task_id, [])
        self.assertIsInstance(stored, lazy_replay.Stored)
        self.assertIs(lazy_replay.transcript(task_id, []), stored)
        # A new event: the scan no longer holds.
        th._append_chat_event({"type": "warning", "text": "late"}, task_id=task_id)
        grown = lazy_replay.transcript(task_id, [])
        self.assertIsNot(grown, stored)
        self.assertEqual(len(grown.plan.lights), len(stored.plan.lights) + 1)
        # Four other tasks later it is scanned again.
        others = [self._persist(f"other {n}", rich_transcript()[:9])[0] for n in range(4)]
        for other in others:
            lazy_replay.transcript(other, [])
        self.assertIs(lazy_replay.transcript(others[0], []), lazy_replay.transcript(others[0], []))
        self.assertIsNot(lazy_replay.transcript(task_id, []), grown)

    def test_events_deleted_since_the_scan_are_left_out(self) -> None:
        task_id, _ = self._persist("rich", rich_transcript())
        stored = lazy_replay.transcript(task_id, [])
        piece = stored.plan.outline(live=False)
        th._delete_task(task_id)
        self.assertEqual(
            kinds(_lazy_piece_events(stored, piece, task_id)),
            [kind for kind in OUTLINE if kind.startswith("~")],
        )


class TestFirstPieceKept(_DatabaseCase):
    """What a finished task shows first is laid out once and kept in the database."""

    def _kept(self, task_id: str) -> list[tuple[Any, ...]]:
        """The ``first_pieces`` rows of *task_id*: (events, last_seq, layout, piece)."""
        rows = th._get_db().execute(
            "SELECT events, last_seq, layout, piece FROM first_pieces WHERE task_id = ?",
            (task_id,),
        )
        return [tuple(row) for row in rows]

    def test_piece_is_read_where_it_was_kept_until_the_task_gains_an_event(self) -> None:
        task_id, chat_id = self._persist("rich", rich_transcript())
        events = len(rich_transcript())
        self.assertEqual(self._kept(task_id), [])
        first = self._resume(chat_id, task_id)
        ((count, last_seq, layout, piece),) = self._kept(task_id)
        self.assertEqual((count, last_seq, layout), (events, events - 1, lazy_replay.LAYOUT))
        # It names rows and holds none of what the user typed.
        self.assertNotIn("also check b.txt", piece)
        # Opened again, the chat is sent the piece kept: the transcript is not laid out.
        db = th._get_db()
        db.execute(
            """UPDATE first_pieces SET piece = replace(piece, '"count": 10', '"count": 77')""",
        )
        again = self._resume(chat_id, task_id)
        self.assertEqual(own(again, "trajectory")[0]["count"], 77)
        self.assertEqual(
            [ev for ev in again if ev["type"] != "trajectory"],
            [ev for ev in first if ev["type"] != "trajectory"],
        )
        # A new event: laid out again, and kept in the old piece's place.
        th._append_chat_event({"type": "warning", "text": "late"}, task_id=task_id)
        grown = self._resume(chat_id, task_id)
        self.assertEqual(own(grown, "trajectory")[0]["count"], 11)
        self.assertEqual([row[:2] for row in self._kept(task_id)], [(events + 1, events)])

    def test_piece_that_cannot_be_used_or_kept_is_laid_out_again(self) -> None:
        task_id, chat_id = self._persist("rich", rich_transcript())
        first = self._resume(chat_id, task_id)
        db = th._get_db()
        # Unreadable, or of another layout: laid out again and kept anew.
        for spoiled in ("piece = 'not json'", "layout = layout + 1"):
            db.execute(f"UPDATE first_pieces SET {spoiled}")
            self.assertEqual(self._resume(chat_id, task_id), first)
            self.assertEqual([row[2] for row in self._kept(task_id)], [lazy_replay.LAYOUT])
            self.assertEqual(self._resume(chat_id, task_id), first)
        # A database that cannot be written: the piece is sent all the
        # same, and kept once it can be.
        db.execute("DELETE FROM first_pieces")
        db.execute("PRAGMA query_only = ON")
        try:
            self.assertEqual(self._resume(chat_id, task_id), first)
        finally:
            db.execute("PRAGMA query_only = OFF")
        self.assertEqual(self._kept(task_id), [])
        # Another process in the middle of a write: it is not waited
        # for (this connection waits 30 s for a writer otherwise).
        writer = sqlite3.connect(th._DB_PATH, isolation_level=None)
        writer.execute("BEGIN IMMEDIATE")
        try:
            started = time.monotonic()
            self.assertEqual(self._resume(chat_id, task_id), first)
            self.assertLess(time.monotonic() - started, 10)
        finally:
            writer.execute("ROLLBACK")
            writer.close()
        self.assertEqual(self._kept(task_id), [])
        self.assertEqual(db.execute("PRAGMA busy_timeout").fetchone()[0], 30000)
        self.assertEqual(self._resume(chat_id, task_id), first)
        self.assertEqual(len(self._kept(task_id)), 1)

    def test_piece_goes_with_its_task(self) -> None:
        task_id, chat_id = self._persist("rich", rich_transcript())
        self._resume(chat_id, task_id)
        kept = lazy_replay.first_piece(task_id)
        th._delete_task(task_id)
        self.assertEqual(self._kept(task_id), [])
        # A piece read just before: its rows are gone, its own events are left.
        self.assertEqual(
            _first_piece_events(task_id, kept),
            [
                {"type": "lazy_gap", "of": "tool_call", "ts": 1794000001000, "fanouts": 2},
                {"type": "trajectory", "count": 10, "lazy": {"taskId": task_id, "upto": LAST}},
                {"type": "lazy_steps", "steps": 4},
            ],
        )
        # Nothing is kept of a task that is gone.
        self.assertEqual(kinds(_first_piece_events(task_id, lazy_replay.first_piece(task_id))),
                         ["lazy_steps"])
        self.assertEqual(self._kept(task_id), [])


class TestFetchOverTheDaemonAndTheLink(TestCase):
    """``getTaskEvents`` over a real daemon, directly and forwarded to a machine."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-lazy-chat-daemon-"))
        self._saved_persistence = (th._DB_PATH, th._KISS_DIR)
        kiss_dir = self.tmpdir / ".kiss"
        kiss_dir.mkdir()
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        self._saved_cfg = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = self.tmpdir / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"
        certfile, keyfile = self.tmpdir / "cert.pem", self.tmpdir / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, certfile=str(certfile), keyfile=str(keyfile),
            url_file=self.tmpdir / "remote-url.json",
            local_endpoint_file=self.tmpdir / "sorcar-local.json",
            work_dir=str(self.tmpdir),
        )
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self.server.start_async(), self.loop).result(60)
        mach.save_machines([])
        self.machine = mach.add_machine("box", f"https://127.0.0.1:{self.port}", "")
        self.task_id, _ = th._add_task("rich")
        for ev in rich_transcript():
            th._queue_chat_event(dict(ev), task_id=self.task_id)
        th._flush_chat_events()
        th._save_task_result(result="done", task_id=self.task_id)

    def tearDown(self) -> None:
        try:
            mach.save_machines([])
            asyncio.run_coroutine_threadsafe(self.server.stop_async(), self.loop).result(60)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(timeout=30)
            self.loop.close()
            close_db()
            th._DB_PATH, th._KISS_DIR = self._saved_persistence
            vc.CONFIG_DIR, vc.CONFIG_PATH = self._saved_cfg
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _fetch(self, **fields: Any) -> dict[str, Any]:
        """The daemon's answer to a ``getTaskEvents`` with *fields*."""
        link = mach.link_for(self.machine)
        return link.request(
            {"type": "getTaskEvents", "taskId": self.task_id, **fields}, "taskEventsPiece",
            timeout=30,
        )

    def test_link_request_answers_the_piece_of_the_panel_opened(self) -> None:
        outline = self._fetch(token="lazy-1")
        self.assertEqual(
            {k: outline.get(k) for k in ("taskId", "seg", "token")},
            {"taskId": self.task_id, "seg": -1, "token": "lazy-1"},
        )
        self.assertEqual(kinds(outline["events"]), OUTLINE)
        folded = self._fetch(seg=1, token="lazy-2")
        self.assertEqual((folded["seg"], folded["token"]), (1, "lazy-2"))
        self.assertEqual(kinds(folded["events"]), SEGMENTS[1])
        # A handle's ``upto``: the transcript up to that event (here,
        # before the first summary's call, so nothing is folded yet);
        # anything else, all of it.
        early = kinds(self._fetch(upto=30)["events"])
        self.assertEqual((early[:4], "summary" in early), (SEGMENTS[0][:4], False))
        for upto in (True, "30"):
            self.assertEqual(kinds(self._fetch(upto=upto)["events"]), OUTLINE)
        # A handle that names no summary is the Trajectory panel's.
        for seg in (True, "1", None):
            self.assertEqual(kinds(self._fetch(seg=seg)["events"]), OUTLINE)
        # The history database unreadable (a directory in its place):
        # the panel still gets its answer, or it would show as loading
        # for good.
        close_db()
        th._DB_PATH = self.tmpdir / "db-in-the-way"
        th._DB_PATH.mkdir()
        unreadable = self._fetch(token="lazy-3")
        self.assertEqual(unreadable.get("token"), "lazy-3")
        self.assertNotIn("events", unreadable)
        self.assertIn("Failed to load the events", unreadable["error"])

    def test_forwarded_fetch_comes_back_stamped_with_the_machine(self) -> None:
        cmd = {
            "type": "getTaskEvents", "taskId": self.task_id, "seg": 0, "token": "lazy-9",
            "machine": "box", "connId": "c1",
        }
        reply = sidebar_panels.machine_task_events_reply("box", cmd)
        self.assertEqual(reply.get("machine"), "box")
        self.assertEqual(reply.get("token"), "lazy-9")
        self.assertEqual(kinds(reply["events"]), SEGMENTS[0])
        # An unknown machine answers with the handle and the error.
        refused = sidebar_panels.machine_task_events_reply("nowhere", cmd)
        self.assertEqual(
            {k: refused.get(k) for k in ("type", "taskId", "seg", "token", "machine")},
            {"type": "taskEventsPiece", "taskId": self.task_id, "seg": 0, "token": "lazy-9",
             "machine": "nowhere"},
        )
        self.assertIn("error", refused)
        self.assertNotIn("events", refused)


class TestPageOverTheSocket(TabMirroringBase):
    """A page's lazy replay, its fetch of a panel and its ``ready``, over a real WebSocket."""

    async def _replay_of(self, ws: Any) -> list[str]:
        """What the next ``task_events`` for tab-t on *ws* holds."""
        replay = await self._wait_for_event(
            ws, "task_events", pred=lambda ev: ev.get("tabId") == "tab-t",
        )
        assert replay is not None, "no task_events for the tab"
        return kinds(replay["events"])

    async def test_only_a_page_that_loads_in_pieces_is_replayed_to_lazily(self) -> None:
        task_id, chat_id = self._seed_chat("Rich chat", rich_transcript())
        opener = await self._connect_ok()
        await self._ready(opener)
        await self._send(opener, {"type": "openTab", "tabId": "tab-t", "title": "new chat"})
        await self._send(opener, {
            "type": "resumeSession", "chatId": chat_id, "taskId": task_id, "tabId": "tab-t",
            "lazy": True,
        })
        self.assertEqual(await self._replay_of(opener), FINISHED)
        await self._send(opener, {
            "type": "getTaskEvents", "taskId": task_id, "tabId": "tab-t", "seg": 0,
            "token": "lazy-1",
        })
        piece = await self._wait_for_event(opener, "taskEventsPiece")
        assert piece is not None, "no answer to getTaskEvents"
        self.assertEqual((piece["token"], kinds(piece["events"])), ("lazy-1", SEGMENTS[0]))
        # A page that comes back showing the tab: in pieces when its
        # ``ready`` says so, whole when it does not.
        reloaded = await self._connect_ok()
        await self._send(
            reloaded, {"type": "ready", "tabId": "tab-t", "restoredTabs": [], "lazy": True},
        )
        self.assertEqual(await self._replay_of(reloaded), FINISHED)
        older = await self._connect_ok()
        await self._ready(older, tab_id="tab-t")
        self.assertIn("summary", await self._replay_of(older))

    async def test_history_click_comes_in_pieces_only_when_every_page_loads_that_way(self) -> None:
        task_id, chat_id = self._seed_chat("Rich chat", rich_transcript())
        clicker, mirror = await self._connect_ok(), await self._connect_ok()
        for ws in (clicker, mirror):
            await self._send(ws, {"type": "ready", "tabId": "", "restoredTabs": [], "lazy": True})
        await self._send(clicker, {"type": "openTab", "tabId": "tab-t", "title": "new chat"})
        click = {
            "type": "resumeSession", "chatId": chat_id, "taskId": task_id, "tabId": "tab-t",
            "lazy": True,
        }
        # A history click is replayed to every page: in pieces, as each of them loads.
        await self._send(clicker, click)
        for ws in (clicker, mirror):
            self.assertEqual(await self._replay_of(ws), FINISHED)
        # A page that loads transcripts whole joins: the click's one
        # replay is whole, or that page would show the task without its panels.
        older = await self._connect_ok()
        await self._ready(older)
        await self._send(clicker, click)
        for ws in (clicker, mirror, older):
            self.assertIn("summary", await self._replay_of(ws))
        # What a page asks for itself alone still comes in pieces.
        await self._send(clicker, {**click, "scoped": True})
        self.assertEqual(await self._replay_of(clicker), FINISHED)
