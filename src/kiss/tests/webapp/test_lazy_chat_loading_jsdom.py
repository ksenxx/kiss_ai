# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A transcript loaded in pieces ends up on the chat page as the transcript loaded whole does.

A page that opens a chat is sent a part of each transcript and asks for
the rest panel by panel (:mod:`kiss.server.lazy_replay`).  Here the
real server cuts several transcripts up, of finished tasks and of
running ones, and ``webapp/test/lazy_replay_diff.js`` shows each to two
pages in jsdom: one is sent the transcript whole, the other its first
piece and then, as every panel is opened, the piece that panel asks
for.  Both must show the same transcript, the same "N events" on the
folded Trajectory panel and the same status row.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server.server import _replay_events
from kiss.tests.server.test_lazy_chat_loading import (
    _PNG_B64,
    _call,
    _DatabaseCase,
    _done,
    _thoughts,
    own,
    rich_transcript,
)

_WEBAPP_DIR = Path(__file__).resolve().parents[2] / "webapp"
_DIFF_JS = _WEBAPP_DIR / "test" / "lazy_replay_diff.js"
_JSDOM_PKG = _WEBAPP_DIR / "node_modules" / "jsdom" / "package.json"
_SETTINGS = {"type": "task_settings", "settings": {"model": "m", "work_dir": "/w"}}
#: The daemon's clock (ms) when each transcript began.
_T0 = 1794000000000

#: A transcript, and how many of its events a page that loads the task while it runs is shown.
Transcript = tuple[list[dict[str, Any]], int]


def _usage(steps: int) -> dict[str, Any]:
    """The status report after step *steps*."""
    return {"type": "usage_info", "text": f"Steps: {steps}/100", "total_tokens": 10 * steps,
            "cost": f"$0.0{steps}", "total_steps": steps}


def _end(steps: int) -> list[dict[str, Any]]:
    """How a task of *steps* steps ends: its ``finish`` call, its result card, its last status."""
    return [
        _call("finish", extras={"success": "True"}),
        _done("finish", "finished"),
        {"type": "result", "text": "success: true", "summary": "<p>All done.</p>", "success": True,
         "step_count": steps, "total_tokens": 100, "cost": "$0.05"},
        _usage(steps),
        {"type": "task_done", "success": True},
    ]


def _continued(steps: int) -> dict[str, Any]:
    """The result card of a session that stops to be continued."""
    return {"type": "result", "text": "to be continued", "summary": "<p>More to do.</p>",
            "success": False, "is_continue": True, "step_count": steps,
            "total_tokens": 40 * steps, "cost": "$0.02"}


def _picture(path: str) -> dict[str, Any]:
    """A picture a tool result shows."""
    return {"b64": _PNG_B64, "mime": "image/png", "path": path}


def _rich() -> Transcript:
    """The transcript of the server's own tests, loaded as its last thought ends."""
    stream = rich_transcript()
    return stream, stream.index(_call("finish", extras={"success": "True"}))


def _stopped() -> Transcript:
    """A task stopped inside a command; a page that loads it meanwhile is shown the same events."""
    stream = rich_transcript()
    stream = stream[: stream.index(_call("Bash", command="false", description="fail")) + 1]
    return stream, len(stream)


def _plain() -> Transcript:
    """A task that never called ``summary``, loaded while its command prints."""
    so_far: list[dict[str, Any]] = [
        _SETTINGS,
        {"type": "prompt", "text": "Count the files."},
        *_thoughts("Let me count."),
        _call("Bash", command="ls | wc -l", description="count"),
        {"type": "system_output", "text": "2\n"},
    ]
    return [*so_far, _done("Bash", ""), _usage(1), *_end(1)], len(so_far)


def _fan_outs() -> Transcript:
    """Fan-outs before, between and after two summaries, loaded with the last one still out.

    The helper's result names a picture it made: its panel shows the picture, which keeps
    it out of the summary that follows.
    """
    so_far: list[dict[str, Any]] = [
        _SETTINGS,
        {"type": "prompt", "text": "Split the work."},
        *_thoughts("Two at once."),
        _call("run_parallel", description="first wave", extras={"tasks": '["a", "b"]'}),
        _done("run_parallel", "both done"),
        _usage(1),
        _call("summary", description="- ran the first wave"),
        _done("summary"),
        *_thoughts("A helper next."),
        _call("run_agent", description="a helper"),
        _done("run_agent", "helped", images=[_picture("chart.png")]),
        _call("Bash", command="ls", description="look"),
        _done("Bash", "a.txt"),
        _usage(2),
        _call("summary", description="- asked a helper"),
        _done("summary"),
        *_thoughts("And the rest."),
        _call("run_parallel", description="last wave", extras={"tasks": '["c"]'}),
    ]
    return [*so_far, _done("run_parallel", "done"), _usage(3), *_end(3)], len(so_far)


def _result_cards() -> Transcript:
    """A task continued three times, one summary between its four result cards; loaded as that
    reports.

    The cards a piece leaves out stand after a command, between two thoughts, and between
    a thought and a command.
    """
    so_far: list[dict[str, Any]] = [
        _SETTINGS,
        {"type": "prompt", "text": "Do it in four goes."},
        *_thoughts("First go."),
        _call("Bash", command="ls", description="look"),
        _done("Bash", "a.txt"),
        _continued(1),
        *_thoughts("Second go."),
        _call("Bash", command="pwd", description="where"),
        _done("Bash", "/w"),
        _call("summary", description="- did the second go"),
        _done("summary"),
    ]
    rest = [
        *_thoughts("To be continued."),
        _continued(2),
        *_thoughts("Third go."),
        _continued(3),
        _call("Bash", command="date", description="when"),
        _done("Bash", "now"),
        *_end(4),
    ]
    return [*so_far, *rest], len(so_far)


def _pictures() -> Transcript:
    """Pictures inside and after a summary's reach, loaded as a typed message lands."""
    so_far: list[dict[str, Any]] = [
        _SETTINGS,
        {"type": "prompt", "text": "Plot it."},
        *_thoughts("One plot."),
        _call("Bash", command="plot a", description="plot"),
        _done("Bash", "wrote a.png", images=[_picture("a.png")]),
        *_thoughts("Looks right."),
        _call("Bash", command="ls", description="look"),
        _done("Bash", "a.png"),
        _call("summary", description="- plotted a"),
        _done("summary"),
        *_thoughts("Two more."),
        _call("Bash", command="plot b c", description="plot"),
        _done("Bash", "wrote b.png and c.png", images=[_picture("b.png"), _picture("c.png")]),
        {"type": "prompt", "text": "one more", "steer": True},
    ]
    rest = [_call("summary", description="- plotted b and c"), _done("summary"), *_end(3)]
    return [*so_far, *rest], len(so_far)


def _odd() -> Transcript:
    """What an interrupted task leaves: a result whose call was never recorded, a question
    never answered, a command's second result, a command that never reported, output after
    a picture, words cut short by a tool call; loaded as its summary reports back."""
    so_far: list[dict[str, Any]] = [
        _SETTINGS,
        {"type": "prompt", "text": "Look around."},
        _done("Bash", "left over"),
        *_thoughts("May I?"),
        _call("ask_user_question", extras={"question": "Proceed?"}),
        _call("Bash", command="ls", description="look"),
        {"type": "system_output", "text": "a.txt\n"},
        _done("Bash", ""),
        {**_done("Bash", "killed"), "is_error": True},
        _call("Bash", command="sleep 100", description="wait"),
        {"type": "system_output", "text": "waiting\n"},
        _call("Read", path="/w/a.png"),
        _done("Read", "a picture", images=[_picture("a.png")]),
        {"type": "system_output", "text": "a late line\n"},
        {"type": "text_delta", "text": "So far so"},
        _call("summary", description="- looked around"),
        _done("summary"),
    ]
    return [*so_far, *_end(2)], len(so_far)


def _model_call(steps: int) -> list[dict[str, Any]]:
    """What an agent records as the model's answer of step *steps* ends."""
    return [
        {"type": "llm_call", "model": "m", "step": steps, "duration_ms": 2063,
         "input_tokens": 4, "output_tokens": 84, "cost": 0.0154, "stop_reason": "tool_use"},
        _usage(steps),
        {"type": "text_end"},
    ]


def _recorded() -> Transcript:
    """A transcript in the order an agent records one: a command the harness ran as an edit
    and a command, a compaction, a session whose ``finish`` reports nothing, and a stop by
    the user; loaded as the second session thinks."""
    so_far: list[dict[str, Any]] = [
        _SETTINGS,
        {"type": "system_prompt", "text": "You are an agent."},
        {"type": "prompt", "text": "Fix the script."},
        *_thoughts("An edit, then a run."),
        *_model_call(1),
        _call("Bash", command="sed -i s/a/b/ x.py && python x.py", description="fix and run"),
        {"type": "text_end"},
        _call("Edit", path="/w/x.py", old_string="a", new_string="b"),
        _done("Edit", "Successfully replaced 1 occurrence(s)", path="/w/x.py"),
        {"type": "text_end"},
        _call("Bash", command="python x.py", description="fix and run"),
        {"type": "system_output", "text": "fine\n"},
        _done("Bash", ""),
        _done("Bash", "Applied as Edit(file_path='/w/x.py')"),
        {"type": "gate_stage", "stage": "charge", "seconds": 0.0, "outcome": "charged"},
        *_thoughts("Time to recap."),
        *_model_call(2),
        _call("summary", description="- fixed the script"),
        _done("summary"),
        {"type": "compaction", "step": 2, "context_tokens": 150352, "dropped_tokens": 43673},
        *_thoughts("Out of room."),
        *_model_call(3),
        _call("finish", extras={"success": "False", "is_continue": "True"}),
        {"type": "text_end"},
        _continued(3),
        {"type": "system_prompt", "text": "You are an agent."},
        {"type": "prompt", "text": "Go on."},
        *_thoughts("Where was I?"),
    ]
    rest = [
        *_model_call(4),
        _call("Bash", command="sleep 100", description="wait"),
        {"type": "system_output", "text": "waiting\n"},
        {"type": "task_stopped"},
    ]
    return [*so_far, *rest], len(so_far)


def _lost_calls() -> Transcript:
    """Results whose calls were never recorded: two before any call, the first ending the
    agent's words; one that finds words after a command; a picture that lands in a summary's
    panel and unfolds it.  Then two calls made at once, whose results come in a row before
    words in two pieces; loaded as those words end."""
    so_far: list[dict[str, Any]] = [
        _SETTINGS,
        {"type": "prompt", "text": "Carry on."},
        *_thoughts("Where was I?"),
        _done("Bash", "left over"),
        _done("Read", "left over too"),
        *_thoughts("Look first."),
        _call("Bash", command="ls", description="look"),
        _done("Bash", "a.png"),
        {"type": "text_delta", "text": "One file."},
        _done("Bash", "a late report"),
        *_thoughts("Recap."),
        _call("summary", description="- looked"),
        _done("summary"),
        _done("Read", "a picture", images=[_picture("a.png")]),
        *_thoughts("Both at once."),
        _call("Read", path="/w/a.txt"),
        _call("Read", path="/w/b.txt"),
        _done("Read", "a"),
        _done("Read", "b"),
        {"type": "text_delta", "text": "They are "},
        {"type": "text_delta", "text": "alike."},
    ]
    return [*so_far, *_end(4)], len(so_far)


def _unfolded_twice() -> Transcript:
    """Twice over: a file read, a summary of it, then a picture from a call never recorded,
    which lands in the summary's panel and unfolds it.  Both panels ask for their steps by
    themselves; loaded as the second picture arrives."""
    so_far: list[dict[str, Any]] = [_SETTINGS, {"type": "prompt", "text": "Compare the two."}]
    for name in ("a", "b"):
        so_far += [
            _call("Read", path=f"/w/{name}.txt"),
            _done("Read", name),
            _call("summary", description=f"- read {name}.txt"),
            _done("summary"),
            _done("Read", "a picture", images=[_picture(f"{name}.png")]),
        ]
    return [*so_far, *_end(4)], len(so_far)


_TRANSCRIPTS = {
    "every kind of event": _rich,
    "stopped inside a command": _stopped,
    "no summary": _plain,
    "fan-outs around summaries": _fan_outs,
    "four result cards": _result_cards,
    "pictures and a typed message": _pictures,
    "an interrupted task's leftovers": _odd,
    "as an agent records it": _recorded,
    "results of lost calls": _lost_calls,
    "two summaries unfolded by pictures": _unfolded_twice,
}


def _stamped(stream: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*stream* with each event stamped a second after the one before it."""
    return [{**ev, "ts": _T0 + 1000 * i} for i, ev in enumerate(stream)]


def _key(handle: dict[str, Any]) -> str:
    """What names the piece of *handle* among its task's: the handle without the task."""
    rest = {name: value for name, value in handle.items() if name != "taskId"}
    return json.dumps(rest, sort_keys=True, separators=(",", ":"))


class TestLazyReplayMatchesWholeReplay(_DatabaseCase):
    """Both replays of each transcript, shown by the chat page in jsdom."""

    def _pieces(self, first: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Every piece a page holding *first* can ask for, by the handle that asks for it."""
        pieces: dict[str, list[dict[str, Any]]] = {}
        unread = [first]
        while unread:
            for ev in unread.pop():
                handle = ev.get("lazy")
                if not isinstance(handle, dict) or _key(handle) in pieces:
                    continue
                piece = self.server.task_events_piece(
                    handle["taskId"], handle.get("seg", -1), handle.get("upto"),
                )
                pieces[_key(handle)] = piece
                unread.append(piece)
        return pieces

    def _frame(self) -> dict[str, Any]:
        """The ``task_events`` frame the server sent last, without its events."""
        with self._lock:
            frame = dict(own(self.events, "task_events")[-1])
        del frame["events"]
        return frame

    def _finished(self, name: str, stream: list[dict[str, Any]]) -> dict[str, Any]:
        """The task of transcript *stream*, over: what a page is sent of it, whole and in pieces."""
        task_id, chat_id = self._persist(name, stream)
        first = self._resume(chat_id, task_id)
        return {
            "name": f"{name}, finished", "running": False, "frame": self._frame(),
            "whole": self._resume(chat_id, task_id, lazy=False),
            "first": first, "pieces": self._pieces(first),
        }

    def _running(self, name: str, stream: list[dict[str, Any]]) -> dict[str, Any]:
        """The task of transcript *stream* so far, running: the same, read from its recording."""
        printer = self.server.printer
        task_id, chat_id = th._add_task(name)
        session = th._load_chat_events_by_task_id(task_id, with_events=False)
        assert session is not None
        # The frame a page that opens the task is sent, before the task records anything.
        self._resume(chat_id, task_id)
        frame = self._frame()
        printer.ensure_recording_for_task(task_id)
        try:
            for ev in stream:
                printer.broadcast({**ev, "taskId": task_id})
            recording = printer.peek_recording_for_task(task_id)
            first = _replay_events(session, recording, task_id, lazy=True, live=True)
            return {
                "name": f"{name}, running", "running": True, "frame": frame,
                "whole": _replay_events(session, recording, task_id),
                "first": first, "pieces": self._pieces(first),
            }
        finally:
            printer.cleanup_task(task_id)

    def _cases(self) -> list[dict[str, Any]]:
        """Each transcript twice: over, and running as far as a page loads it meanwhile."""
        cases = []
        for name, transcript in _TRANSCRIPTS.items():
            stream, so_far = transcript()
            stream = _stamped(stream)
            cases.append(self._finished(name, stream))
            cases.append(self._running(name, stream[:so_far]))
        return cases

    def test_page_shows_a_transcript_loaded_in_pieces_as_one_loaded_whole(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available on PATH")
        if not _JSDOM_PKG.is_file():
            self.skipTest(f"jsdom is not installed under {_WEBAPP_DIR / 'node_modules'}")
        cases = self._cases()
        fixture = self.tmpdir / "cases.json"
        fixture.write_text(json.dumps(cases))
        proc = subprocess.run(
            ["node", "--no-maglev", "--no-concurrent-sparkplug", str(_DIFF_JS), str(fixture)],
            cwd=str(_WEBAPP_DIR), capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(
            proc.returncode, 0,
            f"lazy_replay_diff.js failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )
        for case in cases:
            self.assertIn(f"ok - {case['name']}", proc.stdout)
