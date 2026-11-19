# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""What a lazily loaded chat is sent of a task's transcript, and when.

Opening a chat used to send every event of the task shown.  A long
task has a hundred thousand events and tens of megabytes of them, and
the page folds nearly all of it away at once: a finished task reads as
its text, one collapsed "Trajectory" panel and its result, and inside
the trajectory each ``summary`` tool call hides the panels of the
steps it recounts.  A lazily loaded transcript is sent in the pieces
the reader opens:

* a **finished task** is its settings, a ``trajectory`` stub saying
  how many panels the Trajectory panel holds, its result and its
  final usage (:meth:`Plan.finished`);
* the **outline** is what the Trajectory panel shows when opened: the
  summary calls and every panel no summary hides
  (:meth:`Plan.outline`).  A running task is sent its outline at once;
* a **segment** is the panels one summary hides, asked for when the
  reader opens that summary (:meth:`Plan.segment`).

Which piece an event belongs to follows from how the chat page lays a
transcript out (``webapp/media/main.js``: ``streamBegin`` /
``handleOutputEvent`` / ``streamEnd`` and the ``summary`` case of
``tool_call``).  :func:`plan_transcript` runs that layout on a
:class:`Light` record per event, read without the event's body
(``persistence._scan_task_events``), so that no body leaves the
database before its panel is opened.

A piece is a list of event positions with three kinds of event of its
own in between, which the page understands: ``trajectory`` (the stub),
``lazy_steps`` (the task's step count, which the page cannot count
from a piece) and ``lazy_gap`` (what an event left out here did to
the stream: see :meth:`Plan._piece`).

Laying a transcript out reads a record of every event of it, which
costs the database what reading the events does.  What a finished
task shows first is therefore kept in the database once laid out
(:func:`first_piece`): opening its chat again reads the task's row,
the piece kept and the few event rows the piece names.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
from collections import OrderedDict
from typing import Any, NamedTuple

from kiss.agents.sorcar import persistence
from kiss.core.locked import locked_call

#: Part of the panels no summary folded: sent with the outline.
OUTLINE = -1
#: Part of what a finished task shows beside its Trajectory panel: its
#: result cards and suggested follow-up.
RESULT = -2
#: Part of the events behind the status row and the Task Info panel:
#: the task's settings and its last usage report.
STATUS = -3
#: Part of the events that show nothing: never sent lazily.
DROPPED = -4
#: The version of the layout: a first piece kept by another one is laid
#: out again (:func:`first_piece`).  Raised with every change to what
#: :func:`plan_transcript` or :meth:`Plan.finished` make of a transcript.
LAYOUT = 1

_STREAM_TYPES = frozenset(persistence.STREAM_EVENT_TYPES)
#: The banners: one plain panel each.
_BANNER_TYPES = frozenset({"warning", "notice", "error"})
#: The tools whose call starts sub-agents (``tc-run-parallel`` panels in main.js).
FANOUT_TOOLS = frozenset({"run_parallel", "run_agent"})
# "Steps: 7/100": a usage_info without the numeric fields reports its
# step count in its text (``reportedStepCount`` in main.js).
_STEPS_TEXT_RE = re.compile(r"Steps:\s*([0-9]+)/[0-9]+")


class Light(NamedTuple):
    """What the chat page's layout reads from one event."""

    type: str
    #: A tool call's tool.
    name: str = ""
    #: A prompt the user typed into the running task.
    steer: bool = False
    #: A prompt panel shown before the authoritative text replaces it.
    early: bool = False
    #: A failed tool result.
    is_error: bool = False
    #: A tool call with a shell command: its output streams into the panel.
    command: bool = False
    #: An ``autocommit_done`` a toast reports: it adds no panel.
    quiet: bool = False
    #: A ``system_output`` a replay merges with its neighbours (its text is a string).
    merges: bool = False
    #: The step count a ``usage_info`` or ``result`` reports, 0 for none.
    steps: int = 0
    #: Embedded pictures: replay handles number them over the transcript.
    images: int = 0
    #: At least one embedded picture the page shows.
    media: bool = False
    #: The event's time in epoch milliseconds, when it has one.
    ts: object = None


def _js_truthy(value: object) -> bool:
    """Whether a JSON *value* is truthy in JavaScript (``[]`` and ``{}`` are)."""
    return value is not None and value is not False and value != 0 and value != ""


def _count(value: object) -> int:
    """A reported step count as an int, 0 for anything that is not a usable number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0
    return int(value)


def _reported_steps(kind: str, total_steps: object, step_count: object, text: object) -> int:
    """The step count a ``usage_info`` or ``result`` event reports, 0 for none."""
    if kind == "result":
        return _count(step_count)
    if kind != "usage_info":
        return 0
    if total_steps is not None:
        return _count(total_steps)
    # The head of the text, which is all the scan reads of it.
    match = _STEPS_TEXT_RE.search(text[:1000]) if isinstance(text, str) else None
    return int(match.group(1)) if match else 0


def _shown_pictures(images: object) -> tuple[int, bool]:
    """How many pictures *images* embeds, and whether the page shows one of them."""
    embedded = [
        im for im in (images if isinstance(images, list) else [])
        if isinstance(im, dict) and isinstance(im.get("b64"), str) and im["b64"]
    ]
    return len(embedded), any(isinstance(im.get("mime"), str) and im["mime"] for im in embedded)


#: The record every event of a stream type without pictures shares.
_STREAM_LIGHTS = {kind: Light(kind) for kind in _STREAM_TYPES}


def _light(
    kind: object, facts: dict[str, Any], text: object, images: int, media: bool, legacy: object,
) -> Light:
    """The :class:`Light` record of an event.

    Args:
        kind: Its ``type``.
        facts: Its ``persistence.EVENT_SCAN_FIELDS`` (the event itself will do).
        text: Its ``text``, or as much of it as its layout reads.
        images: How many pictures it embeds.
        media: Whether the page shows one of them.
        legacy: Its row's time in seconds, for an event without a ``ts``.
    """
    if not isinstance(kind, str):
        kind = ""
    if kind in _STREAM_LIGHTS:
        return Light(kind, images=images) if images else _STREAM_LIGHTS[kind]
    name = facts.get("name")
    ts = facts.get("ts")
    if (ts is None and isinstance(legacy, (int, float)) and math.isfinite(legacy)
            and 0 < legacy <= 8.64e12):
        ts = int(legacy * 1000)
    return Light(
        type=kind,
        name=name if isinstance(name, str) else "",
        steer=_js_truthy(facts.get("steer")),
        early=_js_truthy(facts.get("early")),
        is_error=_js_truthy(facts.get("is_error")),
        command=_js_truthy(facts.get("command")),
        quiet=_js_truthy(facts.get("manual")) and _js_truthy(facts.get("success")),
        merges=kind == "system_output" and isinstance(text, str),
        steps=_reported_steps(kind, facts.get("total_steps"), facts.get("step_count"), text),
        images=images,
        media=media,
        ts=ts,
    )


def light_from_event(ev: dict[str, Any]) -> Light:
    """The :class:`Light` record of a decoded event."""
    images, media = _shown_pictures(ev["images"]) if "images" in ev else (0, False)
    return _light(ev.get("type"), ev, ev.get("text"), images, media, ev.get("_timestamp"))


def light_from_row(row: sqlite3.Row) -> Light:
    """The :class:`Light` record of a ``persistence._scan_task_events`` row (not an odd one)."""
    facts: dict[str, Any] = {}
    if row["facts"]:
        facts = dict(zip(persistence.EVENT_SCAN_FIELDS, json.loads(row["facts"])))
    # The scan reads of a text what the layout does: whether a
    # ``system_output`` has one, and the head of a ``usage_info``'s.
    text = "" if row["merges"] else row["usage_text"]
    return _light(row["type"], facts, text, row["images"], bool(row["media"]), row["timestamp"])


#: Panel kinds, by what a ``summary`` call's reach does with the panel.
_PLAIN = 0  # folded into the summary
_KEPT = 1  # left out and moved below the summary (a picture, a user message, a question)
_STOP = 2  # ends the reach (a summary, a panel a summary moved, a result card)


class _Panel:
    """One top-level element of the laid-out transcript."""

    __slots__ = ("events", "kind", "part")

    def __init__(self, kind: int, part: int) -> None:
        self.kind = kind
        #: The piece its events are sent with.
        self.part = part
        #: Positions of the events rendered into it.
        self.events: list[int] = []


class _Layout:
    """The chat page's layout of a transcript, run on :class:`Light` records.

    Mirrors ``streamBegin`` / ``handleOutputEvent`` / ``streamEnd`` of
    ``webapp/media/main.js``: which top-level panel each event lands
    in, which panels a ``summary`` call folds, and the step count.
    """

    def __init__(self) -> None:
        #: Every panel made, the withdrawn ones included.
        self.panels: list[_Panel] = []
        #: The top-level elements, in page order.
        self.top: list[_Panel] = []
        #: The Thinking panel the stream is in, and whether it is still empty.
        self.llm: _Panel | None = None
        self.provisional = False
        self.pending = False
        self.steps = 0
        #: The last tool call's panel, its tool, and whether output streams into it.
        self.tool: _Panel | None = None
        self.tool_name = ""
        self.bash = False
        #: The last top-level ``system_output``: its position, panel and ``merges``.
        self.sys: tuple[int, _Panel, bool] | None = None
        #: The Prompt panels awaiting their authoritative text, by event type.
        self.early: dict[str, list[_Panel]] = {"prompt": [], "system_prompt": []}
        #: Each ``summary`` call, in order: its position and whether it folded a panel.
        self.summary_at: list[int] = []
        self.folded: list[bool] = []
        #: Position of each tool result that opened a Thinking panel, and the panel.
        self.births: dict[int, _Panel] = {}
        #: Positions of the tool results that found words in the Thinking panel.
        self.sealed: set[int] = set()
        #: Positions of the ``task_settings`` events and of the last ``usage_info``.
        self.settings: list[int] = []
        self.usage = -1

    def add(self, position: int, kind: int = _PLAIN, part: int = OUTLINE) -> _Panel:
        """Append a top-level panel made by the event at *position*."""
        panel = _Panel(kind, part)
        panel.events.append(position)
        self.panels.append(panel)
        self.top.append(panel)
        return panel

    def open_thoughts(self, provisional: bool) -> _Panel:
        """``streamOpenThoughts``: a Thinking panel takes the stream's words from here on."""
        panel = _Panel(_PLAIN, OUTLINE)
        self.panels.append(panel)
        self.top.append(panel)
        self.llm = panel
        self.provisional = provisional
        self.pending = False
        return panel

    def leave_thoughts(self) -> None:
        """Leave the Thinking panel; one nothing was said into is withdrawn, cost and all."""
        if self.llm is not None and self.provisional:
            self.llm.part = DROPPED
            # Still on the page (a summary's call leaves it before
            # folding), with at most a few panels below it.
            at = len(self.top) - 1
            while self.top[at] is not self.llm:
                at -= 1
            del self.top[at]
        self.llm = None
        self.pending = True

    def stream(self, position: int, kind: str) -> None:
        """A thinking or text event: into the Thinking panel, opening one when due."""
        if kind in ("thinking_start", "text_delta"):
            if self.llm is not None and self.provisional:
                self.steps += 1
                self.provisional = False
                self.pending = False
            elif self.pending or self.steps == 0:
                self.steps += 1
                self.open_thoughts(provisional=False)
        if self.llm is not None:
            self.llm.events.append(position)

    def tool_call(self, position: int, light: Light) -> None:
        """A tool call: its panel, and for ``summary`` the panels it folds."""
        self.tool_name = light.name
        self.leave_thoughts()
        self.bash = light.command
        if light.name == "summary":
            self.tool = self.add(position, _STOP)
            self.summary_at.append(position)
            self.summarize()
        else:
            self.tool = self.add(position, _KEPT if light.name == "ask_user_question" else _PLAIN)

    def summarize(self) -> None:
        """Fold the panels above the summary just added, back to where its reach ends."""
        seg = len(self.folded)
        reach = len(self.top) - 2
        kept: list[_Panel] = []
        folded = False
        while reach >= 0 and self.top[reach].kind != _STOP:
            panel = self.top[reach]
            if panel.kind == _KEPT:
                panel.kind = _STOP
                kept.append(panel)
            else:
                panel.part = seg
                folded = True
            reach -= 1
        del self.top[reach + 1 : -1]
        self.top.extend(reversed(kept))
        self.folded.append(folded)

    def tool_result(self, position: int, light: Light) -> None:
        """A tool result: into its tool's panel; a question's answer is a panel of its own."""
        thinking = self.tool_name != "finish"
        if thinking:
            if self.llm is not None and not self.provisional:
                # The words end here as at a tool call (``streamBegin``).
                self.sealed.add(position)
                self.llm = None
            self.pending = True
        had_bash, self.bash = self.bash, False
        if self.tool is None:
            # No panel to land in: a failure is a panel, anything else a bare block.
            self.add(position, *((_PLAIN, OUTLINE) if light.is_error else (_STOP, RESULT)))
        else:
            self.tool.events.append(position)
            if light.is_error:
                pass
            elif self.tool_name == "ask_user_question" and not had_bash:
                answer = _Panel(_KEPT, OUTLINE)
                self.panels.append(answer)
                self.top.append(answer)
            elif light.media and self.tool.kind == _PLAIN:
                self.tool.kind = _KEPT
        if thinking and self.llm is None:
            self.births[position] = self.open_thoughts(provisional=True)

    def system_output(self, position: int, light: Light) -> None:
        """Tool output: into the running command's panel, else a line of its own."""
        if self.bash and self.tool is not None:
            self.tool.events.append(position)
            return
        last = self.sys
        if last is not None and last[0] == position - 1 and last[2] and light.merges:
            # A replay merges consecutive outputs into one event, so into one line.
            last[1].events.append(position)
            self.sys = (position, last[1], True)
        else:
            self.sys = (position, self.add(position), light.merges)

    def prompt(self, position: int, light: Light) -> None:
        """A prompt: a user message when typed into the run, else a Prompt panel."""
        pending = self.early[light.type]
        if light.steer:
            self.add(position, _KEPT)
        elif not light.early and pending:
            pending.pop().events.append(position)
        else:
            panel = self.add(position)
            if light.early:
                pending.append(panel)

    def result(self, position: int, light: Light) -> None:
        """A result card: it ends a summary's reach and stays out of the trajectory."""
        self.add(position, _STOP, RESULT)
        self.leave_thoughts()
        if light.steps:
            self.steps = light.steps

    def feed(self, position: int, light: Light) -> None:
        """Lay out the event at *position*."""
        kind = light.type
        if kind in _STREAM_TYPES:
            self.stream(position, kind)
        elif kind == "tool_call":
            self.tool_call(position, light)
        elif kind == "tool_result":
            self.tool_result(position, light)
        elif kind == "system_output":
            self.system_output(position, light)
        elif kind == "result":
            self.result(position, light)
        elif kind in self.early:
            self.prompt(position, light)
        elif kind == "llm_call":
            # The call's price shows under the Thinking panel holding its words.
            if self.llm is not None:
                self.llm.events.append(position)
        elif kind == "usage_info":
            if self.steps > 0 and light.steps:
                self.steps = light.steps
            self.usage = position
        elif kind == "task_settings":
            self.settings.append(position)
        elif kind == "followup_suggestion":
            self.add(position, _STOP, RESULT)
        elif kind == "ask_answer":
            self.add(position, _KEPT)
        elif kind in _BANNER_TYPES or (kind == "autocommit_done" and not light.quiet):
            self.add(position)


#: A piece of a transcript: event positions, and the piece's own events between them.
Piece = list["int | dict[str, Any]"]


def _leave_out(run: list[dict[str, Any]], of: str, ts: object = None) -> None:
    """Add to *run* the stand-in of a left-out event of type *of* (at *ts*, when it has a time).

    Two of a kind in a row do to the page what the first does, so the
    second is not added.
    """
    if run and run[-1]["of"] == of:
        return
    gap: dict[str, Any] = {"type": "lazy_gap", "of": of}
    if ts is not None:
        gap["ts"] = ts
    run.append(gap)


class Plan(NamedTuple):
    """Which piece of a lazily loaded transcript each of its events is sent with."""

    lights: list[Light]
    #: Per event: OUTLINE, RESULT, STATUS, DROPPED, or the ordinal of the summary folding it.
    parts: list[int]
    #: Position of each tool result that opened a Thinking panel still shown -> the panel's part.
    births: dict[int, int]
    #: Positions of the tool results that sealed a Thinking panel words stood in.
    sealed: set[int]
    #: Position of each ``summary`` call that folded a panel -> its ordinal.
    summaries: dict[int, int]
    #: How many panels the Trajectory panel of the finished task holds.
    panels: int
    #: The task's step count, as the page counts it over the whole transcript.
    steps: int
    #: The Trajectory panel stands below the result cards made up to this position.
    stub_after: int

    def messages(self) -> list[int]:
        """The messages the user typed into the task while it ran."""
        return [
            at for at, light in enumerate(self.lights) if light.type == "prompt" and light.steer
        ]

    def finished(self, task_id: str, upto: int) -> Piece:
        """A finished task before its Trajectory panel is opened: stub, results, status.

        Args:
            task_id: The task the stub's handle names.
            upto: The ``seq`` of the last event laid out, which the
                handle names too: the panel is filled from the
                transcript up to it (:func:`transcript`), whatever the
                task is sent after this piece.
        """
        piece = self._piece(RESULT, (RESULT, STATUS))
        if self.panels:
            stub: dict[str, Any] = {
                "type": "trajectory",
                "count": self.panels,
                "lazy": {"taskId": task_id, "upto": upto},
            }
            below = (
                i
                for i, at in enumerate(piece)
                if isinstance(at, int) and at > self.stub_after and self.parts[at] == RESULT
            )
            piece.insert(next(below, len(piece)), stub)
        piece.append({"type": "lazy_steps", "steps": self.steps})
        return piece

    def outline(self, live: bool) -> Piece:
        """The panels no summary folded.

        Args:
            live: The task is running: its outline is its whole replay,
                results and status included.  Otherwise the outline
                fills the Trajectory panel of :meth:`finished`.
        """
        if not live:
            return self._piece(OUTLINE, (OUTLINE,))
        steps: dict[str, Any] = {"type": "lazy_steps", "steps": self.steps}
        return [*self._piece(OUTLINE, (OUTLINE, RESULT, STATUS)), steps]

    def segment(self, seg: int) -> Piece:
        """The panels the *seg*-th ``summary`` call folded."""
        return self._piece(seg, (seg,))

    def _piece(self, part: int, wanted: tuple[int, ...]) -> Piece:
        """The events of the *wanted* parts, with stand-ins for what happened in between.

        The page lays a piece out with the machine it runs on a whole
        transcript, so it must be told what the events left out did to
        that machine: a tool call sealed the Thinking panel the stream
        was in and closed the command output above it (``tool_call``;
        so did a tool result that found words in the panel), a result
        card left the panel and ended the reach of the next summary
        (``result``), a block that is no panel (a tool result with no
        call before it, a follow-up bar) only ended that reach
        (``stop``), and a tool result opened the Thinking panel that
        the next words of *part* land in (``tool_result``).  One
        ``lazy_gap`` stands in for each, where it happened; the first
        of a run also says how many of the run's tool calls were
        fan-outs (``fanouts``), because the page numbers a fan-out's
        panel by its place among the transcript's.

        A tool result sent may open a Thinking panel that is another
        piece's to show (a result with no call before it is a block
        of its own, sent with the result cards, while the panel is the
        outline's).  The page withdraws the panel at the next tool
        call or result card, sent or stood in for; when the piece ends
        first, a tool call's stand-in ends it.
        """
        piece: Piece = []
        # The stand-ins of the events left out since the last one sent.
        run: list[dict[str, Any]] = []
        # A Thinking panel of another piece is open on the page.
        stray = False
        for position, light in enumerate(self.lights):
            if light.type in ("tool_call", "result"):
                stray = False
            if self.parts[position] in wanted:
                piece += run
                run = []
                piece.append(position)
                born = self.births.get(position)
                stray = stray or (born is not None and born not in wanted)
                continue
            if light.type == "tool_call":
                _leave_out(run, "tool_call", light.ts)
                if light.name in FANOUT_TOOLS:
                    run[0]["fanouts"] = run[0].get("fanouts", 0) + 1
            elif position in self.sealed:
                _leave_out(run, "tool_call", light.ts)
            if self.parts[position] == RESULT:
                _leave_out(run, "result" if light.type == "result" else "stop")
            if self.births.get(position) == part:
                _leave_out(run, "tool_result", light.ts)
        if stray:
            _leave_out(run, "tool_call")
        return piece + run

    def pictures_before(self) -> list[int]:
        """How many pictures the transcript embeds before each event (their handles count them)."""
        counts = [0]
        for light in self.lights:
            counts.append(counts[-1] + light.images)
        return counts


def plan_transcript(lights: list[Light]) -> Plan:
    """Lay *lights* out as the chat page does and say which piece each event is sent with."""
    layout = _Layout()
    for position, light in enumerate(lights):
        layout.feed(position, light)
    parts = [DROPPED] * len(lights)
    for panel in layout.panels:
        for position in panel.events:
            parts[position] = panel.part
    for position in layout.settings:
        parts[position] = STATUS
    if layout.usage >= 0:
        parts[layout.usage] = STATUS
    held = [i for i, panel in enumerate(layout.top) if panel.part == OUTLINE]
    # The Trajectory panel stands where the first panel it holds stood:
    # below the result cards above that one.
    above = layout.top[: held[0]] if held else []
    return Plan(
        lights=lights,
        parts=parts,
        births={at: panel.part for at, panel in layout.births.items() if panel.part != DROPPED},
        sealed=layout.sealed,
        summaries={at: seg for seg, at in enumerate(layout.summary_at) if layout.folded[seg]},
        panels=len(held),
        steps=layout.steps,
        stub_after=max((at for panel in above for at in panel.events), default=-1),
    )


class Recorded:
    """A running task's transcript: the display events of the printer's recording."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.plan = plan_transcript([light_from_event(ev) for ev in events])

    def load(self, positions: list[int]) -> dict[int, dict[str, Any]]:
        """The events at *positions*."""
        return {position: self.events[position] for position in positions}


class Stored:
    """A task's transcript in the events table, laid out without reading the events' bodies."""

    def __init__(self, task_id: str, stamp: tuple[object, int, int]) -> None:
        self.task_id = task_id
        #: The state of the table the scan is of (``persistence._task_events_stamp``).
        self.stamp = stamp
        rows = persistence._scan_task_events(task_id, stamp[2])
        # A row the decoder clips or drops is read whole: what it
        # decodes to, if anything, is what the page lays out.
        odd = persistence._load_task_events_by_seq(
            task_id, [(row["seq"], row["seq"]) for row in rows if row["odd"]],
        )
        #: The ``seq`` of each event the decoder keeps.
        self.seqs: list[int] = []
        lights: list[Light] = []
        for row in rows:
            if not row["odd"]:
                light = light_from_row(row)
            elif row["seq"] in odd:
                light = light_from_event(odd[row["seq"]])
            else:
                continue
            self.seqs.append(row["seq"])
            lights.append(light)
        self.plan = plan_transcript(lights)

    def load(self, positions: list[int]) -> dict[int, dict[str, Any]]:
        """The events at *positions* (ascending), read in as few ranges as they form."""
        spans: list[tuple[int, int]] = []
        previous = -2
        for position in positions:
            seq = self.seqs[position]
            if position == previous + 1:
                spans[-1] = (spans[-1][0], seq)
            else:
                spans.append((seq, seq))
            previous = position
        by_seq = persistence._load_task_events_by_seq(self.task_id, spans)
        return {
            position: by_seq[self.seqs[position]]
            for position in positions
            if self.seqs[position] in by_seq
        }


#: The stored transcripts scanned last, by task: opening a folded panel
#: of one of them costs no scan of its own.
_scanned: OrderedDict[str, Stored] = OrderedDict()
_scanned_lock = threading.RLock()
_SCANNED_KEPT = 4


def _scanned_at(task_id: str, stamp: tuple[object, int, int]) -> Stored | None:
    """Under ``_scanned_lock``: the scan of *task_id* kept, if it is of *stamp*."""
    stored = _scanned.get(task_id)
    if stored is None or stored.stamp != stamp:
        return None
    _scanned.move_to_end(task_id)
    return stored


def _keep_scanned(stored: Stored) -> None:
    """Under ``_scanned_lock``: keep *stored* as the scan read last, dropping the oldest."""
    _scanned[stored.task_id] = stored
    _scanned.move_to_end(stored.task_id)
    while len(_scanned) > _SCANNED_KEPT:
        _scanned.popitem(last=False)


def transcript(
    task_id: str, recorded: list[dict[str, Any]], upto: int | None = None,
) -> Recorded | Stored:
    """Where *task_id*'s transcript is read, with its layout (``plan``).

    A running task's is the printer's recording of it, which the events
    table lags behind; any other is scanned in the events table, once
    for as long as the table holds the same events of it.  A handle
    with ``upto`` names the stored transcript as it was when its piece
    was laid out (:meth:`Plan.finished`): what the task was sent since
    (an answer or a message that came late) the page has shown by
    then, and would show twice.

    Args:
        task_id: The task.
        recorded: The display events of the printer's recording of the
            task; empty when the printer keeps none.
        upto: The ``seq`` of the last stored event to read; ``None``
            for the transcript as it is now.
    """
    if recorded and upto is None:
        return Recorded(recorded)
    return _stored(task_id, persistence._task_events_stamp(task_id, upto))


def _stored(task_id: str, stamp: tuple[object, int, int]) -> Stored:
    """The scan of *task_id*'s stored transcript at *stamp*: the one kept, else a new one."""
    stored = locked_call(_scanned_lock, _scanned_at, task_id, stamp)
    if stored is None:
        stored = Stored(task_id, stamp)
        locked_call(_scanned_lock, _keep_scanned, stored)
    return stored


def first_piece(task_id: str) -> dict[str, Any]:
    """What a stored task shows before its Trajectory panel is opened, as the database keeps it.

    ``piece`` is :meth:`Plan.finished` with each event named by its
    row: ``[seq, pictures the transcript embeds before it]``.
    ``steers`` are the ``seq`` of the messages typed into the task
    (:meth:`Plan.messages`): the page reads them while the Trajectory
    panel holds them back, to tell whether a prompt an outage left
    unanswered ran (``settleLostPrompt`` in main.js).  The piece is
    laid out once for as long as the task has the same events and
    kept in the database (``persistence._keep_first_piece``), so that
    opening the chat again scans nothing.

    Args:
        task_id: The task.
    """
    stamp = persistence._task_events_stamp(task_id)
    kept = persistence._first_piece_kept(task_id, stamp, LAYOUT)
    if kept is None:
        stored = _stored(task_id, stamp)
        plan = stored.plan
        pictures = plan.pictures_before()
        kept = {
            "piece": [
                item if isinstance(item, dict) else [stored.seqs[item], pictures[item]]
                for item in plan.finished(task_id, stamp[2])
            ],
            "steers": [stored.seqs[at] for at in plan.messages()],
        }
        persistence._keep_first_piece(task_id, stamp, LAYOUT, kept)
    return kept
